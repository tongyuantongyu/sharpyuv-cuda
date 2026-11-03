// Session (C API): defaults, validation, color and filter constants, launches.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

#include "internal.hpp"

namespace sharpyuv::detail {
namespace {

// Library defaults.
// Signal: JPEG (BT.709 primaries, sRGB, BT.601 matrix, full range, 8-bit).
constexpr int32_t kDefaultPrimaries = 1;
constexpr int32_t kDefaultTransfer = 13;
constexpr int32_t kDefaultMatrix = 6;
constexpr syc_range kDefaultRange = SYC_RANGE_FULL;
constexpr int32_t kDefaultBitDepth = 8;
// Decoder filter: bilinear, center (libwebp).
constexpr syc_decoder_filter kDefaultFilter{SYC_FILTER_BILINEAR, SYC_CHROMA_LOCATION_CENTER};
// Iterations derived from the filter: pure nearest converges in ~3 passes,
// anything with an interpolating member needs ~6; the count depends on the
// filter only, not on the signal (private/experiments/convergence/RESULTS.md).
constexpr int32_t kNearestIterations = 3;
constexpr int32_t kDefaultIterations = 6;
// Upper bound of syc_session::iterations: the tiles are exact with a halo of
// the iteration count, and 6 converges every filter.
constexpr int32_t kMaxIterations = 6;
// Tile overlap (halo) in blocks, capped at the iteration count: 3 by default
// (<= 1 code at tile seams, private/experiments/convergence/halo_e1.py), the full
// iteration count in exact mode.
constexpr int32_t kDefaultHaloCap = 3;
static_assert(kDefaultIterations <= kMaxIterations && kMaxIterations <= kMaxHalo);

// HLG OOTF for the BT.2100 nominal display: gamma = 1.2 + 0.42 log10(Lw/1000).
constexpr double kHlgNominalPeak = 1000.0;  // cd/m^2

// Largest supported width / height (SYC_STATUS_IMAGE_TOO_LARGE above).
constexpr int32_t kMaxDimension = 1 << 16;

// The public struct is the base, so syc_session* and Session* convert both
// ways; the signature rejects pointers not from syc_session_create. The
// members are the state of syc_session_init (called once per session).
struct Session : syc_session {
  constexpr static uint32_t SIGN = 0xe8de022a;
  uint32_t signature = SIGN;

  bool ready = false;
  int device = 0;
  cudaError_t last_error = cudaSuccess;
  ColorParams color{};  // with the filter tables and window
  int32_t passes = 0;   // iterations, SYC_ITERATIONS_AUTO resolved
  int32_t halo = 0;  // tile overlap: min(passes, cap), or passes if exact

  static Session* FromPublic(syc_session* s) {
    auto* session = static_cast<Session*>(s);
    if (session->signature != SIGN) return nullptr;
    return session;
  }

  static const Session* FromPublic(const syc_session* s) {
    auto* session = static_cast<const Session*>(s);
    if (session->signature != SIGN) return nullptr;
    return session;
  }

  syc_status Fail(cudaError_t e) {
    last_error = e;
    return e == cudaErrorMemoryAllocation ? SYC_STATUS_OUT_OF_MEMORY : SYC_STATUS_CUDA_ERROR;
  }
};

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

size_t SampleBytes(syc_sample_type t) {
  switch (t) {
    case SYC_SAMPLE_U8: return 1;
    case SYC_SAMPLE_U16:
    case SYC_SAMPLE_F16: return 2;
    case SYC_SAMPLE_F32: return 4;
  }
  return 0;
}

bool Aligned(const void* p, size_t a) { return reinterpret_cast<uintptr_t>(p) % a == 0; }

// Color constants of the session's signal description.
syc_status BuildColorParams(const syc_session& s, ColorParams* out) {
  const int32_t n = s.yuv_bit_depth;
  if (n != 8 && n != 10 && n != 12 && n != 16) return SYC_STATUS_INVALID_ARGUMENT;
  if (s.range != SYC_RANGE_LIMITED && s.range != SYC_RANGE_FULL) {
    return SYC_STATUS_INVALID_ARGUMENT;
  }
  const double lr = s.primaries.kr, lb = s.primaries.kb;
  if (!ValidWeights(lr, lb)) return SYC_STATUS_INVALID_ARGUMENT;
  double mr = 0.0, mb = 0.0;
  switch (s.matrix.kind) {
    case SYC_MATRIX_COEFFICIENTS:
      mr = s.matrix.kr;
      mb = s.matrix.kb;
      if (!ValidWeights(mr, mb)) return SYC_STATUS_INVALID_ARGUMENT;
      break;
    case SYC_MATRIX_FROM_PRIMARIES:  // chroma-derived NCL (MC 12)
      mr = lr;
      mb = lb;
      break;
    default: return SYC_STATUS_INVALID_ARGUMENT;
  }

  ColorParams c{};
  switch (s.transfer.kind) {
    case SYC_TRANSFER_PARAMETRIC: {
      const syc_parametric_curve& p = s.transfer.curve;
      if (!ValidCurve(p)) return SYC_STATUS_INVALID_ARGUMENT;
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
    case SYC_TRANSFER_PQ: c.tf = n <= 12 ? TfKind::kPq12 : TfKind::kPq; break;
    case SYC_TRANSFER_HLG: {
      c.tf = TfKind::kHlg;
      const double g = 1.2 + 0.42 * std::log10(kHlgNominalPeak / 1000.0);
      c.tfp.ootf_dec_exp = float(g - 1.0);
      c.tfp.ootf_enc_exp = float((1.0 - g) / g);
      c.tfp.ootf_inv_gamma = float(1.0 / g);
      break;
    }
    case SYC_TRANSFER_LINEAR: c.tf = TfKind::kLinear; break;
    default: return SYC_STATUS_INVALID_ARGUMENT;
  }
  c.kl[0] = float(lr);
  c.kl[1] = float(1.0 - lr - lb);
  c.kl[2] = float(lb);
  c.km[0] = float(mr);
  c.km[1] = float(1.0 - mr - mb);
  c.km[2] = float(mb);
  c.u_coef = float(0.5 / (1.0 - mb));
  c.v_coef = float(0.5 / (1.0 - mr));
  c.g_from_r = float(-mr / (1.0 - mr - mb));
  c.g_from_b = float(-mb / (1.0 - mr - mb));
  c.depth = n;
  const double unit = std::ldexp(1.0, n - 8);  // 2^(n-8)
  if (s.range == SYC_RANGE_FULL) {
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
  *out = c;
  return SYC_STATUS_OK;
}

// Input layout of an interpretation: samples per pixel, R, G, B, A offsets
// (interleaved), planar, alpha. False for an unknown interpretation.
struct Layout {
  int32_t stride, r, g, b, a;
  bool planar, alpha;
};

bool GetLayout(syc_channel_interpretation i, Layout* out) {
  switch (i) {
    case SYC_INTERP_PLANAR_RGB: *out = {1, 0, 0, 0, 0, true, false}; return true;
    case SYC_INTERP_PLANAR_RGBA: *out = {1, 0, 0, 0, 0, true, true}; return true;
    case SYC_INTERP_RGB: *out = {3, 0, 1, 2, 0, false, false}; return true;
    case SYC_INTERP_BGR: *out = {3, 2, 1, 0, 0, false, false}; return true;
    case SYC_INTERP_RGBA: *out = {4, 0, 1, 2, 3, false, true}; return true;
    case SYC_INTERP_BGRA: *out = {4, 2, 1, 0, 3, false, true}; return true;
    case SYC_INTERP_ARGB: *out = {4, 1, 2, 3, 0, false, true}; return true;
    case SYC_INTERP_ABGR: *out = {4, 3, 2, 1, 0, false, true}; return true;
    case SYC_INTERP_RGBX: *out = {4, 0, 1, 2, 3, false, false}; return true;
    case SYC_INTERP_BGRX: *out = {4, 2, 1, 0, 3, false, false}; return true;
    case SYC_INTERP_XRGB: *out = {4, 1, 2, 3, 0, false, false}; return true;
    case SYC_INTERP_XBGR: *out = {4, 3, 2, 1, 0, false, false}; return true;
  }
  return false;
}

syc_status BuildInput(const syc_rgb_image& src, InputDesc* out) {
  InputDesc in{};
  const int32_t w = src.width;
  const size_t sb = SampleBytes(src.type);
  switch (src.type) {
    case SYC_SAMPLE_U8:
      in.type = InType::kU8;
      in.scale = 1.f / 255.f;
      break;
    case SYC_SAMPLE_U16:
      if (src.u16_bits < 1 || src.u16_bits > 16) return SYC_STATUS_INVALID_ARGUMENT;
      in.type = InType::kU16;
      in.scale = float(1.0 / (std::ldexp(1.0, src.u16_bits) - 1.0));
      break;
    case SYC_SAMPLE_F16:
    case SYC_SAMPLE_F32: {
      const double lo = src.float_min, hi = src.float_max;
      if (!std::isfinite(lo) || !std::isfinite(hi) || lo == hi) {
        return SYC_STATUS_INVALID_ARGUMENT;
      }
      in.type = src.type == SYC_SAMPLE_F16 ? InType::kF16 : InType::kF32;
      in.scale = float(1.0 / (hi - lo));
      in.offset = float(-lo / (hi - lo));
      break;
    }
    default: return SYC_STATUS_INVALID_ARGUMENT;
  }
  switch (src.encoding) {
    case SYC_INPUT_ENCODED: break;
    case SYC_INPUT_LINEAR: in.linear = 1; break;
    default: return SYC_STATUS_INVALID_ARGUMENT;
  }
  Layout l;
  if (!GetLayout(src.interpretation, &l)) return SYC_STATUS_INVALID_ARGUMENT;
  in.planar = l.planar;
  in.alpha = l.alpha;
  const int planes = l.planar ? (l.alpha ? 4 : 3) : 1;
  for (int c = 0; c < planes; ++c) {
    if (src.data[c] == nullptr || !Aligned(src.data[c], sb) || src.pitch[c] % sb != 0 ||
        src.pitch[c] < size_t(w) * l.stride * sb) {
      return SYC_STATUS_INVALID_ARGUMENT;
    }
    in.ptr[c] = static_cast<const unsigned char*>(src.data[c]);
    in.pitch[c] = src.pitch[c];
  }
  in.stride = l.stride;
  in.off[0] = l.r;
  in.off[1] = l.g;
  in.off[2] = l.b;
  in.off[3] = l.a;
  *out = in;
  return SYC_STATUS_OK;
}

syc_status BuildOutput(const syc_yuv_image& dst, int32_t depth, bool alpha, OutputDesc* out) {
  const size_t ob = depth == 8 ? 1 : 2;
  const size_t w = size_t(dst.width), cw = size_t((dst.width + 1) / 2);
  auto plane_ok = [&](const void* p, size_t pitch, size_t row_samples) {
    return p != nullptr && Aligned(p, ob) && pitch % ob == 0 && pitch >= row_samples * ob;
  };
  if (dst.alignment != SYC_ALIGN_LSB && dst.alignment != SYC_ALIGN_MSB) {
    return SYC_STATUS_INVALID_ARGUMENT;
  }
  OutputDesc o{};
  if (!plane_ok(dst.data[0], dst.pitch[0], w)) return SYC_STATUS_INVALID_ARGUMENT;
  o.y = static_cast<unsigned char*>(dst.data[0]);
  o.y_pitch = dst.pitch[0];
  o.biplanar = dst.biplanar;
  o.swap_uv = dst.swap_uv;
  if (dst.biplanar) {
    if (!plane_ok(dst.data[1], dst.pitch[1], 2 * cw)) return SYC_STATUS_INVALID_ARGUMENT;
  } else if (!plane_ok(dst.data[1], dst.pitch[1], cw) ||
             !plane_ok(dst.data[2], dst.pitch[2], cw)) {
    return SYC_STATUS_INVALID_ARGUMENT;
  }
  o.u = static_cast<unsigned char*>(dst.data[1]);
  o.u_pitch = dst.pitch[1];
  if (!dst.biplanar) {
    o.v = static_cast<unsigned char*>(dst.data[2]);
    o.v_pitch = dst.pitch[2];
  }
  o.shift = (depth > 8 && dst.alignment == SYC_ALIGN_MSB) ? 16 - depth : 0;
  // Alpha output: needs alpha input.
  if (dst.data[3] != nullptr && (!alpha || !plane_ok(dst.data[3], dst.pitch[3], w))) {
    return SYC_STATUS_INVALID_ARGUMENT;
  }
  o.a = static_cast<unsigned char*>(dst.data[3]);
  o.a_pitch = dst.pitch[3];
  *out = o;
  return SYC_STATUS_OK;
}

}  // namespace

KernelConfig& InternalKernelConfig() {
  static KernelConfig config;
  return config;
}

}  // namespace sharpyuv::detail

using namespace sharpyuv::detail;

extern "C" {

syc_session* syc_session_create(void) {
  auto* s = new (std::nothrow) Session();
  if (s == nullptr) return nullptr;
  s->stream = nullptr;
  if (syc_primaries_from_h273(kDefaultPrimaries, &s->primaries) != SYC_STATUS_OK ||
      syc_transfer_from_h273(kDefaultTransfer, &s->transfer) != SYC_STATUS_OK ||
      syc_matrix_from_h273(kDefaultMatrix, &s->matrix) != SYC_STATUS_OK) {
    delete s;
    return nullptr;
  }
  s->range = kDefaultRange;
  s->yuv_bit_depth = kDefaultBitDepth;
  s->filter = kDefaultFilter;
  s->weighted_filters = nullptr;
  s->num_weighted_filters = 0;
  s->iterations = SYC_ITERATIONS_AUTO;
  s->exact = false;
  return s;
}

syc_status syc_session_init(syc_session* session) {
  if (session == nullptr) return SYC_STATUS_INVALID_ARGUMENT;
  auto* s = Session::FromPublic(session);
  if (s == nullptr) return SYC_STATUS_INVALID_ARGUMENT;
  
  s->ready = false;

  int device = 0;
  const cudaError_t e = IsDefaultStream(s->stream) ? cudaGetDevice(&device)
                                                  : cudaStreamGetDevice(s->stream, &device);
  if (e != cudaSuccess) return SYC_STATUS_INVALID_ARGUMENT;

  auto& color = s->color;
  const syc_status st = BuildColorParams(*s, &color);
  if (st != SYC_STATUS_OK) return st;

  FilterMix mix{};
  const syc_weighted_filter single{1.0, s->filter};
  const bool list = s->num_weighted_filters > 0;
  if (list && s->weighted_filters == nullptr) return SYC_STATUS_INVALID_ARGUMENT;
  if (!BuildFilterMix(list ? s->weighted_filters : &single, list ? s->num_weighted_filters : 1,
                      &mix)) {
    return SYC_STATUS_INVALID_ARGUMENT;
  }
  if (s->iterations < SYC_ITERATIONS_AUTO || s->iterations > kMaxIterations) {
    return SYC_STATUS_INVALID_ARGUMENT;
  }
  const bool nearest_only = mix.w[0][0] + mix.w[0][1] + mix.w[0][2] >= 1.0 - 1e-12;
  const int32_t iterations = s->iterations >= 0 ? s->iterations
                             : nearest_only    ? kNearestIterations
                                               : kDefaultIterations;
  BuildFilterParams(mix, &color.filter, &color.window);

  s->passes = iterations;
  // Block-local filters (pure nearest) need no overlap: exact at halo 0.
  s->halo = color.window == Window::kLocal ? 0
           : s->exact                  ? iterations
                                      : std::min(iterations, kDefaultHaloCap);
  s->device = device;
  s->ready = true;
  return SYC_STATUS_OK;
}

void syc_session_destroy(syc_session* session) {
  if (session == nullptr) return;
  auto* s = Session::FromPublic(session);
  if (s == nullptr) return;
  if (s->ready) {
    DeviceGuard guard(s->device);
    cudaStreamSynchronize(s->stream);
  }
  delete s;
}

syc_status syc_session_convert(syc_session* session, const syc_rgb_image* src,
                               const syc_yuv_image* dst) {
  if (session == nullptr || src == nullptr || dst == nullptr) {
    return SYC_STATUS_INVALID_ARGUMENT;
  }
  auto* s = Session::FromPublic(session);
  if (s == nullptr) return SYC_STATUS_INVALID_ARGUMENT;
  if (!s->ready) return SYC_STATUS_NOT_READY;
  const int32_t w = src->width, h = src->height;
  // Output: the input's size, or padded to even.
  auto size_ok = [](int32_t in, int32_t out) { return out == in || (in % 2 == 1 && out == in + 1); };
  if (w < 1 || h < 1 || !size_ok(w, dst->width) || !size_ok(h, dst->height)) {
    return SYC_STATUS_INVALID_ARGUMENT;
  }
  if (w > kMaxDimension || h > kMaxDimension) return SYC_STATUS_IMAGE_TOO_LARGE;

  InputDesc in{};
  syc_status st = BuildInput(*src, &in);
  if (st != SYC_STATUS_OK) return st;
  OutputDesc out{};
  st = BuildOutput(*dst, s->color.depth, in.alpha != 0, &out);
  if (st != SYC_STATUS_OK) return st;

  const Geometry geo{w, h, (w + 1) / 2, (h + 1) / 2, dst->width, dst->height};
  ColorParams cp = s->color;
  DeviceGuard guard(s->device);
  const KernelConfig& kc = InternalKernelConfig();
  const int32_t halo = kc.halo < 0 ? s->halo : kc.halo;
  if ((cp.tf == TfKind::kPq || cp.tf == TfKind::kPq12) && kc.pq_fast >= 0) {
    cp.tf = kc.pq_fast ? TfKind::kPq12 : TfKind::kPq;
  }
  const cudaError_t e = LaunchConvert(cp, in, out, geo, s->passes, kc.tile, halo, s->stream);
  if (e != cudaSuccess) return s->Fail(e);
  return SYC_STATUS_OK;
}

int32_t syc_session_last_cuda_error(const syc_session* session) {
  if (session == nullptr) return 0;
  auto* s = Session::FromPublic(session);
  if (s == nullptr) return 0;
  return int32_t(s->last_error);
}

}  // extern "C"
