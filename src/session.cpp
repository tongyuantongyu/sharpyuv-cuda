// Session: validation, color and filter constants, launches.

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "internal.hpp"
#include "sharpyuv_cuda/sharpyuv.hpp"

namespace sharpyuv {
namespace {

// Library defaults.
// Decoder filter for an unspecified FilterInfo: bilinear, center (libwebp).
constexpr DecoderFilter kDefaultFilter{FilterKind::kBilinear, ChromaLocation::kCenter};
// Iterations derived from the filter: pure nearest converges in ~3 passes,
// anything with an interpolating member needs ~6; the count depends on the
// filter only, not on ColorInfo (experiments/convergence/RESULTS.md).
constexpr int32_t kNearestIterations = 3;
constexpr int32_t kDefaultIterations = 6;
// Upper bound of FilterInfo::SetIterations: the tiles are exact with a
// halo of the iteration count, and 6 converges every filter.
constexpr int32_t kMaxIterations = 6;
// Tile overlap (halo) in blocks, capped at the iteration count: 3 by default
// (<= 1 code at tile seams, experiments/convergence/halo_e1.py), the full
// iteration count with FilterInfo::SetExact.
constexpr int32_t kDefaultHaloCap = 3;
static_assert(kDefaultIterations <= kMaxIterations && kMaxIterations <= detail::kMaxHalo);

// HLG OOTF for the BT.2100 nominal display: gamma = 1.2 + 0.42 log10(Lw/1000).
constexpr double kHlgNominalPeak = 1000.0;  // cd/m^2

// Largest supported width / height (kImageTooLarge above).
constexpr int kMaxDimension = 1 << 16;

bool IsDefaultStream(cudaStream_t s) {
  return s == nullptr || s == cudaStreamLegacy || s == cudaStreamPerThread;
}

class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    if (cudaGetDevice(&previous_) == cudaSuccess && previous_ != device) {
      restore_ = cudaSetDevice(device) == cudaSuccess;
    }
  }
  ~DeviceGuard() {
    if (restore_) cudaSetDevice(previous_);
  }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int previous_ = 0;
  bool restore_ = false;
};

size_t SampleBytes(SampleType t) {
  switch (t) {
    case SampleType::kU8: return 1;
    case SampleType::kU16:
    case SampleType::kF16: return 2;
    case SampleType::kF32: return 4;
  }
  return 0;
}

bool Aligned(const void* p, size_t a) {
  return reinterpret_cast<uintptr_t>(p) % a == 0;
}

}  // namespace

namespace detail {

KernelConfig& InternalKernelConfig() {
  static KernelConfig config;
  return config;
}

}  // namespace detail

// ---------------------------------------------------------------------------

struct Session::Impl {
  cudaStream_t stream = nullptr;
  int device = 0;
  cudaError_t last_error = cudaSuccess;

  // Color
  bool has_color = false;
  detail::ColorParams color{};

  // Decoder filter (see SetFilterInfo; Create sets the library default).
  detail::Window window = detail::Window::kBox;
  detail::FilterParams filter{};
  int32_t iterations = 0;
  int32_t halo = 0;  // tile overlap: min(iterations, cap)

  Status Fail(cudaError_t e) {
    last_error = e;
    return e == cudaErrorMemoryAllocation ? Status::kOutOfMemory
                                          : Status::kCudaError;
  }
};

Session::Session() : impl_(std::make_unique<Impl>()) {}

Session::~Session() = default;

Status Session::Create(const SessionConfig& config,
                       std::unique_ptr<Session>* out) {
  if (out == nullptr) return Status::kInvalidArgument;
  int device = 0;
  cudaError_t e = IsDefaultStream(config.stream)
                      ? cudaGetDevice(&device)
                      : cudaStreamGetDevice(config.stream, &device);
  if (e != cudaSuccess) return Status::kInvalidArgument;
  std::unique_ptr<Session> s(new Session());
  s->impl_->stream = config.stream;
  s->impl_->device = device;
  const Status st = s->SetFilterInfo(FilterInfo{});  // the library default filter
  if (st != Status::kOk) return st;
  *out = std::move(s);
  return Status::kOk;
}

cudaStream_t Session::stream() const { return impl_->stream; }

cudaError_t Session::LastCudaError() const { return impl_->last_error; }

Status Session::SetColorInfo(const ColorInfo& info) {
  using detail::TfKind;
  if (!info.primaries.specified() || !info.transfer.specified() ||
      !info.matrix.specified() || info.range == Range::kUnspecified) {
    return Status::kInvalidArgument;
  }
  const int n = info.yuv_bit_depth;
  if (n != 8 && n != 10 && n != 12 && n != 16) return Status::kInvalidArgument;

  detail::ColorParams c{};
  switch (info.transfer.kind()) {
    case TransferFunction::Kind::kParametric: {
      const ParametricCurve& p = info.transfer.curve();
      c.tf = TfKind::kParametric;
      c.tfp.gamma = float(p.gamma);
      c.tfp.inv_gamma = float(1.0 / p.gamma);
      c.tfp.alpha = float(p.alpha);
      c.tfp.inv_alpha = float(1.0 / p.alpha);
      c.tfp.alpha_m1 = float(p.alpha - 1.0);
      c.tfp.beta = float(p.beta);
      c.tfp.slope = float(p.slope);
      c.tfp.inv_slope = p.slope > 0 ? float(1.0 / p.slope) : 0.f;
      c.tfp.v_beta = float(p.slope * p.beta);
      break;
    }
    // The cheaper PQ decode is accurate enough for <= 12-bit output.
    case TransferFunction::Kind::kPq: c.tf = n <= 12 ? TfKind::kPq12 : TfKind::kPq; break;
    case TransferFunction::Kind::kHlg: {
      c.tf = TfKind::kHlg;
      const double g = 1.2 + 0.42 * std::log10(kHlgNominalPeak / 1000.0);
      c.tfp.ootf_dec_exp = float(g - 1.0);
      c.tfp.ootf_enc_exp = float((1.0 - g) / g);
      c.tfp.ootf_inv_gamma = float(1.0 / g);
      break;
    }
    case TransferFunction::Kind::kLinear: c.tf = TfKind::kLinear; break;
    default: return Status::kInvalidArgument;
  }
  const double lr = info.primaries.kr(), lb = info.primaries.kb();
  c.kl[0] = float(lr);
  c.kl[1] = float(1.0 - lr - lb);
  c.kl[2] = float(lb);
  // Chroma-derived NCL (MC 12): the matrix uses the primaries' weights.
  const MatrixCoefficients& m = info.matrix;
  const double mr = m.from_primaries() ? lr : m.kr();
  const double mb = m.from_primaries() ? lb : m.kb();
  c.km[0] = float(mr);
  c.km[1] = float(1.0 - mr - mb);
  c.km[2] = float(mb);
  c.u_coef = float(0.5 / (1.0 - mb));
  c.v_coef = float(0.5 / (1.0 - mr));
  c.g_from_r = float(-mr / (1.0 - mr - mb));
  c.g_from_b = float(-mb / (1.0 - mr - mb));
  c.depth = n;
  const double unit = std::ldexp(1.0, n - 8);  // 2^(n-8)
  if (info.range == Range::kFull) {
    const double maxv = std::ldexp(1.0, n) - 1.0;
    c.y_scale = float(maxv);
    c.y_offset = 0.f;
    c.c_scale = float(maxv);
    c.c_offset = float(std::ldexp(1.0, n - 1));
  } else {
    c.y_scale = float(219.0 * unit);
    c.y_offset = float(16.0 * unit);
    c.c_scale = float(224.0 * unit);
    c.c_offset = float(128.0 * unit);
  }
  // Chroma code range [0, 2^n - 1] mapped back to offsets.
  const double cmax = std::ldexp(1.0, n) - 1.0;
  const double cb_lo = (0.0 - c.c_offset) / c.c_scale;
  const double cb_hi = (cmax - c.c_offset) / c.c_scale;
  c.db_min = float(cb_lo / c.u_coef);
  c.db_max = float(cb_hi / c.u_coef);
  c.dr_min = float(cb_lo / c.v_coef);
  c.dr_max = float(cb_hi / c.v_coef);
  impl_->color = c;
  impl_->has_color = true;
  return Status::kOk;
}

Status Session::SetFilterInfo(const FilterInfo& filter) {
  FilterInfo f = filter;
  if (!f.specified()) {
    const Status st = FilterInfo::FromFilter(kDefaultFilter, &f);
    if (st != Status::kOk) return st;
  }
  const bool nearest_only =
      f.Weight({FilterKind::kNearest, ChromaLocation::kCenter}) +
          f.Weight({FilterKind::kNearest, ChromaLocation::kLeft}) +
          f.Weight({FilterKind::kNearest, ChromaLocation::kTopLeft}) >=
      1.0 - 1e-12;
  if (filter.iterations() > kMaxIterations) return Status::kInvalidArgument;
  const int32_t iterations =
      filter.iterations() >= 0
          ? filter.iterations()
          : (nearest_only ? kNearestIterations : kDefaultIterations);
  detail::BuildFilterParams(f, &impl_->filter, &impl_->window);
  impl_->iterations = iterations;
  // Block-local filters (pure nearest) need no overlap: exact at halo 0.
  impl_->halo = impl_->window == detail::Window::kLocal ? 0
                : filter.exact()                      ? iterations
                                                      : std::min(iterations, kDefaultHaloCap);
  return Status::kOk;
}

Status Session::Convert(const RgbImage& src, const YuvImage& dst) {
  Impl& s = *impl_;
  if (!s.has_color) return Status::kNotReady;
  const int w = src.width, h = src.height;
  if (w < 1 || h < 1 || dst.width != w || dst.height != h) {
    return Status::kInvalidArgument;
  }
  if (w > kMaxDimension || h > kMaxDimension) return Status::kImageTooLarge;

  const int bw = (w + 1) / 2, bh = (h + 1) / 2;

  // Input
  detail::InputDesc in{};
  const size_t sb = SampleBytes(src.type);
  if (sb == 0) return Status::kInvalidArgument;
  switch (src.type) {
    case SampleType::kU8:
      in.type = detail::InType::kU8;
      in.scale = 1.f / 255.f;
      break;
    case SampleType::kU16: {
      const int bits = src.u16_bits == 0 ? 16 : src.u16_bits;
      if (bits < 1 || bits > 16) return Status::kInvalidArgument;
      in.type = detail::InType::kU16;
      in.scale = float(1.0 / (std::ldexp(1.0, bits) - 1.0));
      break;
    }
    case SampleType::kF16:
    case SampleType::kF32:
      if (!std::isfinite(src.float_scale) || !std::isfinite(src.float_offset)) {
        return Status::kInvalidArgument;
      }
      in.type = src.type == SampleType::kF16 ? detail::InType::kF16
                                             : detail::InType::kF32;
      in.scale = src.float_scale;
      in.offset = src.float_offset;
      break;
  }
  in.linear = src.encoding == InputEncoding::kLinear;
  switch (src.alpha) {
    case AlphaSource::kNone: break;
    case AlphaSource::kInterleaved: {
      const ChannelOrder o = src.order;
      if (src.planar || (o != ChannelOrder::kRGBA && o != ChannelOrder::kBGRA &&
                         o != ChannelOrder::kARGB && o != ChannelOrder::kABGR)) {
        return Status::kInvalidArgument;
      }
      break;
    }
    case AlphaSource::kPlane:
      // Planar colour only: alpha comes with 4-channel interleaved input or
      // as the fourth plane of planar input.
      if (!src.planar || src.alpha_data == nullptr || !Aligned(src.alpha_data, sb) ||
          src.alpha_pitch % sb != 0 || src.alpha_pitch < size_t(w) * sb) {
        return Status::kInvalidArgument;
      }
      in.ptr[3] = static_cast<const unsigned char*>(src.alpha_data);
      in.pitch[3] = src.alpha_pitch;
      break;
    default: return Status::kInvalidArgument;
  }
  const bool alpha = src.alpha != AlphaSource::kNone;
  in.alpha = alpha;
  in.planar = src.planar;
  if (src.planar) {
    for (int c = 0; c < 3; ++c) {
      if (src.data[c] == nullptr || !Aligned(src.data[c], sb) ||
          src.pitch[c] % sb != 0 || src.pitch[c] < size_t(w) * sb) {
        return Status::kInvalidArgument;
      }
      in.ptr[c] = static_cast<const unsigned char*>(src.data[c]);
      in.pitch[c] = src.pitch[c];
    }
  } else {
    int stride = 3, r = 0, g = 1, b = 2, a = 3;
    switch (src.order) {
      case ChannelOrder::kRGB: break;
      case ChannelOrder::kBGR: r = 2, b = 0; break;
      case ChannelOrder::kRGBA: stride = 4; break;
      case ChannelOrder::kBGRA: stride = 4, r = 2, b = 0; break;
      case ChannelOrder::kARGB: stride = 4, r = 1, g = 2, b = 3, a = 0; break;
      case ChannelOrder::kABGR: stride = 4, r = 3, g = 2, b = 1, a = 0; break;
      default: return Status::kInvalidArgument;
    }
    if (src.data[0] == nullptr || !Aligned(src.data[0], sb) ||
        src.pitch[0] % sb != 0 || src.pitch[0] < size_t(w) * stride * sb) {
      return Status::kInvalidArgument;
    }
    in.ptr[0] = static_cast<const unsigned char*>(src.data[0]);
    in.pitch[0] = src.pitch[0];
    in.stride = stride;
    in.off[0] = r;
    in.off[1] = g;
    in.off[2] = b;
    in.off[3] = a;
  }

  // Output
  detail::OutputDesc out{};
  const size_t ob = s.color.depth == 8 ? 1 : 2;
  auto plane_ok = [&](const void* p, size_t pitch, size_t row_samples) {
    return p != nullptr && Aligned(p, ob) && pitch % ob == 0 &&
           pitch >= row_samples * ob;
  };
  const size_t cw = size_t(bw);
  if (!plane_ok(dst.data[0], dst.pitch[0], size_t(w))) {
    return Status::kInvalidArgument;
  }
  out.y = static_cast<unsigned char*>(dst.data[0]);
  out.y_pitch = dst.pitch[0];
  out.biplanar = dst.biplanar;
  out.swap_uv = dst.swap_uv;
  if (dst.biplanar) {
    if (!plane_ok(dst.data[1], dst.pitch[1], 2 * cw)) return Status::kInvalidArgument;
    out.u = static_cast<unsigned char*>(dst.data[1]);
    out.u_pitch = dst.pitch[1];
  } else {
    if (!plane_ok(dst.data[1], dst.pitch[1], cw) ||
        !plane_ok(dst.data[2], dst.pitch[2], cw)) {
      return Status::kInvalidArgument;
    }
    out.u = static_cast<unsigned char*>(dst.data[1]);
    out.u_pitch = dst.pitch[1];
    out.v = static_cast<unsigned char*>(dst.data[2]);
    out.v_pitch = dst.pitch[2];
  }
  out.shift = (s.color.depth > 8 && dst.alignment == SampleAlignment::kMsb)
                  ? 16 - s.color.depth
                  : 0;
  if (dst.alpha_data != nullptr &&
      (!alpha || !plane_ok(dst.alpha_data, dst.alpha_pitch, size_t(w)))) {
    return Status::kInvalidArgument;  // alpha output needs alpha processing
  }
  out.a = static_cast<unsigned char*>(dst.alpha_data);
  out.a_pitch = dst.alpha_pitch;

  const detail::Geometry geo{w, h, bw, bh};
  detail::ColorParams cp = s.color;
  cp.window = s.window;
  cp.filter = s.filter;
  DeviceGuard guard(s.device);
  const detail::KernelConfig& kc = detail::InternalKernelConfig();
  const int32_t halo = kc.halo < 0 ? s.halo : kc.halo;
  if ((cp.tf == detail::TfKind::kPq || cp.tf == detail::TfKind::kPq12) && kc.pq_fast >= 0) {
    cp.tf = kc.pq_fast ? detail::TfKind::kPq12 : detail::TfKind::kPq;
  }
  const cudaError_t e =
      detail::LaunchConvert(cp, in, out, geo, s.iterations, kc.tile, halo, s.stream);
  if (e != cudaSuccess) return s.Fail(e);
  return Status::kOk;
}

}  // namespace sharpyuv
