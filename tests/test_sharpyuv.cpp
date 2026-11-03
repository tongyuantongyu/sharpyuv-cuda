// sharpyuv-cuda tests: factories, session validation, exactness on uniform
// images, format equivalence, and sharp-YUV quality on synthetic edges
// (host-side decoder).

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "../src/internal.hpp"  // InternalKernelConfig: tile size / halo overrides
#include "sharpyuv_cuda/sharpyuv.hpp"

using namespace sharpyuv;

namespace {

int g_failures = 0;
std::string g_context;

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      ++g_failures;                                                          \
      std::printf("FAIL %s:%d [%s] %s\n", __FILE__, __LINE__,                \
                  g_context.c_str(), #cond);                                 \
    }                                                                        \
  } while (0)

#define CHECK_STATUS(expr, want)                                             \
  do {                                                                       \
    const syc_status s_ = (expr);                                            \
    if (s_ != (want)) {                                                      \
      ++g_failures;                                                          \
      std::printf("FAIL %s:%d [%s] %s -> %s (want %s)\n", __FILE__, __LINE__, \
                  g_context.c_str(), #expr, syc_status_string(s_),           \
                  syc_status_string(want));                                  \
    }                                                                        \
  } while (0)

#define CHECK_OK(expr) CHECK_STATUS(expr, SYC_STATUS_OK)

// expr throws sharpyuv::Error with the given status (SYC_STATUS_OK: does not
// throw).
#define CHECK_THROWS(expr, want)                                             \
  do {                                                                       \
    syc_status s_ = SYC_STATUS_OK;                                           \
    try {                                                                    \
      (void)(expr);                                                          \
    } catch (const Error& e_) {                                              \
      s_ = e_.status();                                                      \
    }                                                                        \
    if (s_ != (want)) {                                                      \
      ++g_failures;                                                          \
      std::printf("FAIL %s:%d [%s] %s threw %s (want %s)\n", __FILE__,       \
                  __LINE__, g_context.c_str(), #expr, syc_status_string(s_), \
                  syc_status_string(want));                                  \
    }                                                                        \
  } while (0)

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    const cudaError_t e_ = (expr);                                           \
    if (e_ != cudaSuccess) {                                                 \
      std::printf("CUDA error %s:%d %s: %s\n", __FILE__, __LINE__, #expr,    \
                  cudaGetErrorString(e_));                                   \
      std::exit(2);                                                          \
    }                                                                        \
  } while (0)

// Device buffer holding a copy of a host vector.
struct DeviceBuffer {
  void* ptr = nullptr;
  size_t size = 0;
  explicit DeviceBuffer(size_t bytes) : size(bytes) {
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    CUDA_CHECK(cudaMemset(ptr, 0xCD, bytes));
  }
  template <class T>
  explicit DeviceBuffer(const std::vector<T>& v) : DeviceBuffer(v.size() * sizeof(T)) {
    CUDA_CHECK(cudaMemcpy(ptr, v.data(), size, cudaMemcpyHostToDevice));
  }
  ~DeviceBuffer() { cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  template <class T>
  std::vector<T> Download() const {
    std::vector<T> v(size / sizeof(T));
    CUDA_CHECK(cudaMemcpy(v.data(), ptr, size, cudaMemcpyDeviceToHost));
    return v;
  }
};

// ---------------------------------------------------------------------------
// Images

RgbImage Interleaved(const void* data, size_t pitch, int w, int h, syc_sample_type type,
                     syc_channel_interpretation interp = SYC_INTERP_RGB) {
  RgbImage img;
  img.width = w;
  img.height = h;
  img.type = type;
  img.interpretation = interp;
  img.data[0] = data;
  img.pitch[0] = pitch;
  return img;
}

// Planar R, G, B, and A if a is not null; one pitch for all planes.
RgbImage Planar(const void* r, const void* g, const void* b, size_t pitch, int w, int h,
                syc_sample_type type, const void* a = nullptr) {
  RgbImage img;
  img.width = w;
  img.height = h;
  img.type = type;
  img.interpretation = a ? SYC_INTERP_PLANAR_RGBA : SYC_INTERP_PLANAR_RGB;
  const void* planes[4] = {r, g, b, a};
  for (int c = 0; c < 4; ++c) {
    img.data[c] = planes[c];
    img.pitch[c] = planes[c] ? pitch : 0;
  }
  return img;
}

// ---------------------------------------------------------------------------
// Host reference color math (double)

struct Rgb {
  double r, g, b;
};

// SHARPYUV_TEST_QUICK set: reduced matrices for sanitizer runs (docs/BUILD.md).
bool Quick() {
  static const bool quick = std::getenv("SHARPYUV_TEST_QUICK") != nullptr;
  return quick;
}

struct Setup {
  const char* name;
  int cp, tc, mc;
  syc_range range;
  int depth;
  syc_chroma_location loc;
};

// An initialized session for a setup: the given filter list, or bilinear at
// the setup's location; iterations SYC_ITERATIONS_AUTO: the filter's
// default. A failed Init is reported, and the session's conversions then
// fail with SYC_STATUS_NOT_READY.
Session MakeSession(cudaStream_t stream, const Setup& setup,
                    std::span<const syc_weighted_filter> filters = {},
                    int32_t iterations = SYC_ITERATIONS_AUTO, bool exact = false) {
  Session s;
  try {
    s.SetStream(stream)
        .SetColor(ColorPrimaries::FromH273(setup.cp), TransferFunction::FromH273(setup.tc),
                  MatrixCoefficients::FromH273(setup.mc), setup.range, setup.depth)
        .SetEffort(iterations, exact);
    if (filters.empty()) {
      s.SetFilters(syc_decoder_filter{SYC_FILTER_BILINEAR, setup.loc});
    } else {
      s.SetFilters(filters);
    }
    s.Init();
  } catch (const Error& e) {
    ++g_failures;
    std::printf("FAIL [%s] MakeSession: %s\n", g_context.c_str(), e.what());
  }
  return s;
}

using FilterList = std::vector<syc_weighted_filter>;

// Decoder filters covering every kernel window: the setup's bilinear
// location (kBox / kCols), top-left (kFull), linear left (kCols), nearest
// (kLocal), and blends (kBox, kFull).
std::vector<std::pair<std::string, FilterList>> TestFilters(syc_chroma_location loc) {
  constexpr syc_decoder_filter kCenter{SYC_FILTER_BILINEAR, SYC_CHROMA_LOCATION_CENTER};
  constexpr syc_decoder_filter kTopLeft{SYC_FILTER_BILINEAR, SYC_CHROMA_LOCATION_TOP_LEFT};
  constexpr syc_decoder_filter kLinearLeft{SYC_FILTER_LINEAR, SYC_CHROMA_LOCATION_LEFT};
  constexpr syc_decoder_filter kNearest{SYC_FILTER_NEAREST, SYC_CHROMA_LOCATION_CENTER};
  return {
      {"bilinear", {{1, {SYC_FILTER_BILINEAR, loc}}}},
      {"top-left", {{1, kTopLeft}}},
      {"linear left", {{1, kLinearLeft}}},
      {"nearest", {{1, kNearest}}},
      {"center+nearest", {{1, kCenter}, {1, kNearest}}},
      {"c+tl+lin+nn", {{1, kCenter}, {1, kTopLeft}, {1, kLinearLeft}, {1, kNearest}}},
  };
}

struct Quant {
  double y_scale, y_offset, c_scale, c_offset;
  int maxv;
};

Quant MakeQuant(syc_range range, int n) {
  const double unit = std::ldexp(1.0, n - 8);
  if (range == SYC_RANGE_FULL) {
    const double m = std::ldexp(1.0, n) - 1;
    return {m, 0, m, std::ldexp(1.0, n - 1), int(m)};
  }
  return {219 * unit, 16 * unit, 224 * unit, 128 * unit, (1 << n) - 1};
}

// Standard (non-sharp) Y'CbCr of an encoded RGB' triple, quantized.
void ReferenceYuv(Rgb p, const Setup& s, int* y, int* cb, int* cr) {
  const ColorPrimaries prim = ColorPrimaries::FromH273(s.cp);
  const MatrixCoefficients m = MatrixCoefficients::FromH273(s.mc);
  const bool derived = m.kind == SYC_MATRIX_FROM_PRIMARIES;  // MC 12
  const double kr = derived ? prim.kr : m.kr;
  const double kb = derived ? prim.kb : m.kb, kg = 1 - kr - kb;
  const double yy = kr * p.r + kg * p.g + kb * p.b;
  const double u = (p.b - yy) / (2 * (1 - kb));
  const double v = (p.r - yy) / (2 * (1 - kr));
  const Quant q = MakeQuant(s.range, s.depth);
  auto quant = [&](double x, double sc, double o) {
    return std::clamp(int(std::lround(x * sc + o)), 0, q.maxv);
  };
  *y = quant(yy, q.y_scale, q.y_offset);
  *cb = quant(u, q.c_scale, q.c_offset);
  *cr = quant(v, q.c_scale, q.c_offset);
}

// ---------------------------------------------------------------------------
// YUV output buffers + readback

struct YuvBuffers {
  int w, h, depth;
  bool biplanar;
  size_t bytes;  // per sample
  size_t y_pitch, c_pitch;
  DeviceBuffer y, u, v;
  YuvBuffers(int w_, int h_, int depth_, bool biplanar_)
      : w(w_), h(h_), depth(depth_), biplanar(biplanar_),
        bytes(depth_ == 8 ? 1 : 2),
        y_pitch((size_t(w_) + 13) * bytes),  // padded pitches on purpose
        c_pitch((size_t((w_ + 1) / 2) * (biplanar_ ? 2 : 1) + 7) * bytes),
        y(y_pitch * h_), u(c_pitch * ((h_ + 1) / 2)), v(c_pitch * ((h_ + 1) / 2)) {}

  YuvImage Image(syc_sample_alignment a = SYC_ALIGN_LSB, bool swap = false) {
    YuvImage img;
    img.width = w;
    img.height = h;
    img.biplanar = biplanar;
    img.swap_uv = swap;
    img.alignment = a;
    img.data[0] = y.ptr;
    img.pitch[0] = y_pitch;
    img.data[1] = u.ptr;
    img.pitch[1] = c_pitch;
    if (!biplanar) {
      img.data[2] = v.ptr;
      img.pitch[2] = c_pitch;
    }
    return img;
  }

  // Planes as ints: Y (w*h), Cb, Cr ((w+1)/2 * (h+1)/2). Undoes biplanar
  // interleaving and MSB alignment.
  void Read(std::vector<int>* Y, std::vector<int>* Cb, std::vector<int>* Cr,
            int shift = 0, bool swap = false) const {
    const int cw = (w + 1) / 2, ch = (h + 1) / 2;
    auto get = [&](const std::vector<unsigned char>& raw, size_t pitch, int x, int yy) {
      const unsigned char* row = raw.data() + pitch * yy;
      if (bytes == 1) return int(row[x]);
      uint16_t s;
      std::memcpy(&s, row + 2 * x, 2);
      return int(s) >> shift;
    };
    const auto ry = y.Download<unsigned char>();
    const auto ru = u.Download<unsigned char>();
    const auto rv = v.Download<unsigned char>();
    Y->assign(size_t(w) * h, 0);
    Cb->assign(size_t(cw) * ch, 0);
    Cr->assign(size_t(cw) * ch, 0);
    for (int j = 0; j < h; ++j)
      for (int i = 0; i < w; ++i) (*Y)[size_t(j) * w + i] = get(ry, y_pitch, i, j);
    for (int j = 0; j < ch; ++j) {
      for (int i = 0; i < cw; ++i) {
        int a, b;
        if (biplanar) {
          a = get(ru, c_pitch, 2 * i, j);
          b = get(ru, c_pitch, 2 * i + 1, j);
        } else {
          a = get(ru, c_pitch, i, j);
          b = get(rv, c_pitch, i, j);
        }
        if (swap) std::swap(a, b);
        (*Cb)[size_t(j) * cw + i] = a;
        (*Cr)[size_t(j) * cw + i] = b;
      }
    }
  }
};

// All planes of a conversion, concatenated.
std::vector<int> Convert(Session& s, const RgbImage& src, int depth) {
  YuvBuffers out(src.width, src.height, depth, false);
  CHECK_OK(s.Convert(src, out.Image()));
  std::vector<int> Y, Cb, Cr;
  out.Read(&Y, &Cb, &Cr);
  Y.insert(Y.end(), Cb.begin(), Cb.end());
  Y.insert(Y.end(), Cr.begin(), Cr.end());
  return Y;
}

const Setup kSetups[] = {
    {"sRGB/709 full 8 center", 1, 13, 1, SYC_RANGE_FULL, 8, SYC_CHROMA_LOCATION_CENTER},
    {"709/601 limited 8 left", 1, 1, 6, SYC_RANGE_LIMITED, 8, SYC_CHROMA_LOCATION_LEFT},
    {"PQ 2020 limited 10 left", 9, 16, 9, SYC_RANGE_LIMITED, 10, SYC_CHROMA_LOCATION_LEFT},
    {"HLG 2020 limited 10 left", 9, 18, 9, SYC_RANGE_LIMITED, 10, SYC_CHROMA_LOCATION_LEFT},
    {"P3 sRGB/601 full 12", 12, 13, 6, SYC_RANGE_FULL, 12, SYC_CHROMA_LOCATION_CENTER},
    {"linear 709 full 16", 1, 8, 1, SYC_RANGE_FULL, 16, SYC_CHROMA_LOCATION_CENTER},
    {"2.2 709 limited 16", 1, 4, 1, SYC_RANGE_LIMITED, 16, SYC_CHROMA_LOCATION_LEFT},
    {"P3 sRGB/chroma-derived full 8", 12, 13, 12, SYC_RANGE_FULL, 8, SYC_CHROMA_LOCATION_CENTER},
};

// ---------------------------------------------------------------------------
// Tests

void TestFactories() {
  g_context = "factories";
  CHECK(syc_version() == SYC_VERSION);
  for (int s = SYC_STATUS_OK; s <= SYC_STATUS_CUDA_ERROR; ++s) {
    CHECK(syc_status_string(syc_status(s)) != nullptr);
  }

  const ColorPrimaries p709 = ColorPrimaries::FromH273(1);
  CHECK(std::fabs(p709.kr - 0.2126) < 2e-4 && std::fabs(p709.kb - 0.0722) < 2e-4);
  const ColorPrimaries p2020 = ColorPrimaries::FromH273(9);
  CHECK(std::fabs(p2020.kr - 0.2627) < 2e-4 && std::fabs(p2020.kb - 0.0593) < 2e-4);
  const ColorPrimaries p3 = ColorPrimaries::FromH273(12);
  CHECK(std::fabs(p3.kr - 0.2290) < 2e-4 && std::fabs(p3.kb - 0.0793) < 2e-4);
  CHECK_THROWS(ColorPrimaries::FromH273(2), SYC_STATUS_UNSUPPORTED);
  CHECK_THROWS(ColorPrimaries::FromLumaCoefficients(0.7, 0.4), SYC_STATUS_INVALID_ARGUMENT);
  CHECK_THROWS(ColorPrimaries::FromChromaticities({0.64, 0.33}, {0.64, 0.33}, {0.15, 0.06},
                                                  {0.3127, 0.329}),
               SYC_STATUS_INVALID_ARGUMENT);  // degenerate
  CHECK_STATUS(syc_primaries_from_h273(1, nullptr), SYC_STATUS_INVALID_ARGUMENT);

  for (int code : {1, 4, 5, 6, 7, 8, 13, 14, 15, 16, 18}) {
    CHECK_THROWS(TransferFunction::FromH273(code), SYC_STATUS_OK);
  }
  CHECK(TransferFunction::FromH273(16).kind == SYC_TRANSFER_PQ);
  CHECK_THROWS(TransferFunction::FromH273(2), SYC_STATUS_UNSUPPORTED);
  CHECK_THROWS(TransferFunction::FromParametric({2.4, 1.055, 0.0031308, 10.0}),
               SYC_STATUS_INVALID_ARGUMENT);  // discontinuous
  CHECK_THROWS(TransferFunction::FromParametric({2.4, 1.0, 0.0, 0.0}), SYC_STATUS_OK);

  CHECK_THROWS(MatrixCoefficients::FromH273(0), SYC_STATUS_UNSUPPORTED);
  CHECK_THROWS(MatrixCoefficients::FromH273(10), SYC_STATUS_UNSUPPORTED);
  CHECK_THROWS(MatrixCoefficients::FromH273(13), SYC_STATUS_UNSUPPORTED);  // chroma-derived CL
  CHECK(MatrixCoefficients::FromH273(9).kind == SYC_MATRIX_COEFFICIENTS);
  CHECK(MatrixCoefficients::FromH273(12).kind == SYC_MATRIX_FROM_PRIMARIES);
  CHECK(MatrixCoefficients::FromCoefficients(0.2126, 0.0722).kind == SYC_MATRIX_COEFFICIENTS);
  CHECK_THROWS(MatrixCoefficients::FromCoefficients(0.0, 0.1), SYC_STATUS_INVALID_ARGUMENT);
}

// The C session: defaults, and syc_session_init validation (one fresh
// session per case: init is called once per session).
void TestSessionInit(cudaStream_t stream) {
  g_context = "session init";
  syc_session_destroy(nullptr);  // no-op
  CHECK_STATUS(syc_session_init(nullptr), SYC_STATUS_INVALID_ARGUMENT);

  syc_session* d = syc_session_create();
  CHECK(d != nullptr);
  CHECK(d->stream == nullptr && d->range == SYC_RANGE_FULL && d->yuv_bit_depth == 8);
  CHECK(std::fabs(d->primaries.kr - 0.2126) < 2e-4);
  CHECK(d->transfer.kind == SYC_TRANSFER_PARAMETRIC && d->transfer.curve.gamma == 2.4);
  CHECK(d->matrix.kind == SYC_MATRIX_COEFFICIENTS && d->matrix.kr == 0.299);
  CHECK(d->filter.kind == SYC_FILTER_BILINEAR &&
        d->filter.location == SYC_CHROMA_LOCATION_CENTER);
  CHECK(d->weighted_filters == nullptr && d->num_weighted_filters == 0);
  CHECK(d->iterations == SYC_ITERATIONS_AUTO && !d->exact);
  CHECK(syc_session_last_cuda_error(d) == 0);
  syc_session_destroy(d);

  auto init = [&](const std::function<void(syc_session*)>& change) {
    syc_session* s = syc_session_create();
    s->stream = stream;
    change(s);
    const syc_status st = syc_session_init(s);
    syc_session_destroy(s);
    return st;
  };
  const syc_weighted_filter zero[] = {{0, {SYC_FILTER_BILINEAR, SYC_CHROMA_LOCATION_CENTER}}};
  const syc_weighted_filter negative[] = {
      {-1, {SYC_FILTER_BILINEAR, SYC_CHROMA_LOCATION_CENTER}}};
  const syc_weighted_filter no_location[] = {
      {1, {SYC_FILTER_NEAREST, SYC_CHROMA_LOCATION_UNSPECIFIED}}};
  CHECK_OK(init([](syc_session*) {}));
  CHECK_OK(init([](syc_session* s) { s->iterations = 0; }));
  CHECK_OK(init([](syc_session* s) { s->iterations = 6; }));
  CHECK_STATUS(init([](syc_session* s) { s->iterations = 7; }), SYC_STATUS_INVALID_ARGUMENT);
  CHECK_STATUS(init([](syc_session* s) { s->iterations = -2; }), SYC_STATUS_INVALID_ARGUMENT);
  CHECK_STATUS(init([](syc_session* s) { s->yuv_bit_depth = 9; }), SYC_STATUS_INVALID_ARGUMENT);
  CHECK_STATUS(init([](syc_session* s) { s->range = SYC_RANGE_UNSPECIFIED; }),
               SYC_STATUS_INVALID_ARGUMENT);
  CHECK_STATUS(init([](syc_session* s) { s->primaries = {0, 0}; }),
               SYC_STATUS_INVALID_ARGUMENT);
  CHECK_STATUS(init([](syc_session* s) { s->transfer.kind = SYC_TRANSFER_UNSPECIFIED; }),
               SYC_STATUS_INVALID_ARGUMENT);
  CHECK_STATUS(init([](syc_session* s) { s->transfer.curve.gamma = 0.5; }),
               SYC_STATUS_INVALID_ARGUMENT);
  CHECK_STATUS(init([](syc_session* s) { s->matrix.kind = SYC_MATRIX_UNSPECIFIED; }),
               SYC_STATUS_INVALID_ARGUMENT);
  CHECK_STATUS(init([](syc_session* s) { s->filter.kind = SYC_FILTER_UNSPECIFIED; }),
               SYC_STATUS_INVALID_ARGUMENT);
  CHECK_STATUS(init([](syc_session* s) { s->num_weighted_filters = 1; }),  // null list
               SYC_STATUS_INVALID_ARGUMENT);
  for (const syc_weighted_filter* list : {zero, negative, no_location}) {
    CHECK_STATUS(init([&](syc_session* s) {
                   s->weighted_filters = list;
                   s->num_weighted_filters = 1;
                 }),
                 SYC_STATUS_INVALID_ARGUMENT);
  }

  // A failed init leaves the session unusable.
  syc_session* bad = syc_session_create();
  bad->iterations = 7;
  CHECK_STATUS(syc_session_init(bad), SYC_STATUS_INVALID_ARGUMENT);
  std::vector<uint8_t> px(4 * 4 * 3, 0);
  DeviceBuffer in(px);
  YuvBuffers out(4, 4, 8, false);
  const RgbImage src = Interleaved(in.ptr, 12, 4, 4, SYC_SAMPLE_U8);
  const YuvImage dst = out.Image();
  CHECK_STATUS(syc_session_convert(bad, &src, &dst), SYC_STATUS_NOT_READY);
  syc_session_destroy(bad);

  // C++: Init throws.
  Session t;
  t.SetEffort(7, false);
  CHECK_THROWS(t.Init(), SYC_STATUS_INVALID_ARGUMENT);
}

void TestConvertErrors(cudaStream_t stream) {
  g_context = "convert errors";
  YuvBuffers out(8, 8, 8, false);
  std::vector<uint8_t> px(8 * 8 * 4, 0);
  DeviceBuffer in(px);
  const RgbImage src = Interleaved(in.ptr, 24, 8, 8, SYC_SAMPLE_U8);
  {
    Session s;
    s.SetStream(stream);
    CHECK_STATUS(s.Convert(src, out.Image()), SYC_STATUS_NOT_READY);  // before Init
  }
  Session s = MakeSession(stream, kSetups[0]);
  CHECK_OK(s.Convert(src, out.Image()));
  CHECK_STATUS(syc_session_convert(nullptr, &src, nullptr), SYC_STATUS_INVALID_ARGUMENT);
  YuvBuffers out7(7, 8, 8, false);
  CHECK_STATUS(s.Convert(src, out7.Image()), SYC_STATUS_INVALID_ARGUMENT);  // size mismatch
  // Above the supported size (checked before any pointer is used).
  const int huge = (1 << 16) + 1;
  const RgbImage huge_src = Interleaved(in.ptr, size_t(huge) * 3, huge, 1, SYC_SAMPLE_U8);
  YuvImage huge_dst = out.Image();
  huge_dst.width = huge;
  huge_dst.height = 1;
  CHECK_STATUS(s.Convert(huge_src, huge_dst), SYC_STATUS_IMAGE_TOO_LARGE);

  RgbImage bad = src;
  bad.pitch[0] = 23;  // < a row
  CHECK_STATUS(s.Convert(bad, out.Image()), SYC_STATUS_INVALID_ARGUMENT);
  bad = src;
  bad.interpretation = syc_channel_interpretation(99);
  CHECK_STATUS(s.Convert(bad, out.Image()), SYC_STATUS_INVALID_ARGUMENT);
  bad = Interleaved(in.ptr, 48, 8, 8, SYC_SAMPLE_U16);
  bad.u16_bits = 0;
  CHECK_STATUS(s.Convert(bad, out.Image()), SYC_STATUS_INVALID_ARGUMENT);
  bad = Interleaved(in.ptr, 96, 8, 1, SYC_SAMPLE_F32);
  bad.float_max = bad.float_min;
  YuvBuffers out1(8, 1, 8, false);
  CHECK_STATUS(s.Convert(bad, out1.Image()), SYC_STATUS_INVALID_ARGUMENT);
  YuvImage bad_dst = out.Image();
  bad_dst.alignment = syc_sample_alignment(5);
  CHECK_STATUS(s.Convert(src, bad_dst), SYC_STATUS_INVALID_ARGUMENT);

  // Alpha: 4-channel interleaved, or planar with an A plane.
  YuvBuffers small(4, 4, 8, false);
  const unsigned char* p = static_cast<const unsigned char*>(in.ptr);
  const RgbImage rgba = Interleaved(p, 16, 4, 4, SYC_SAMPLE_U8, SYC_INTERP_RGBA);
  CHECK_OK(s.Convert(rgba, small.Image()));
  const RgbImage planar = Planar(p, p + 16, p + 32, 4, 4, 4, SYC_SAMPLE_U8, p + 48);
  CHECK_OK(s.Convert(planar, small.Image()));
  RgbImage bad_alpha = planar;
  bad_alpha.pitch[3] = 3;  // < width
  CHECK_STATUS(s.Convert(bad_alpha, small.Image()), SYC_STATUS_INVALID_ARGUMENT);
  bad_alpha = planar;
  bad_alpha.data[3] = nullptr;
  CHECK_STATUS(s.Convert(bad_alpha, small.Image()), SYC_STATUS_INVALID_ARGUMENT);
  // Alpha output: needs alpha input.
  YuvImage alpha_out = small.Image();
  alpha_out.data[3] = alpha_out.data[0];
  alpha_out.pitch[3] = alpha_out.pitch[0];
  CHECK_STATUS(s.Convert(Interleaved(p, 12, 4, 4, SYC_SAMPLE_U8), alpha_out),
               SYC_STATUS_INVALID_ARGUMENT);
  CHECK_STATUS(s.Convert(Interleaved(p, 16, 4, 4, SYC_SAMPLE_U8, SYC_INTERP_RGBX), alpha_out),
               SYC_STATUS_INVALID_ARGUMENT);
  CHECK_OK(s.Convert(rgba, alpha_out));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// A uniform image must give exactly the standard conversion (the sharp
// iteration has nothing to correct).
void TestUniform(cudaStream_t stream) {
  const int w = 37, h = 21;
  const Rgb colors[] = {{0, 0, 0},       {1, 1, 1},       {0.5, 0.5, 0.5},
                        {1, 0, 0},       {0, 1, 0},       {0, 0, 1},
                        {0.9, 0.2, 0.6}, {0.1, 0.8, 0.75}};
  for (const Setup& setup : kSetups) {
    g_context = std::string("uniform ") + setup.name;
    Session session = MakeSession(stream, setup);
    for (const Rgb& c : colors) {
      for (syc_sample_type type : {SYC_SAMPLE_U8, SYC_SAMPLE_U16, SYC_SAMPLE_F32}) {
        // Quantize the color to the input type; expected = from that value.
        Rgb q = c;
        std::vector<unsigned char> raw;
        size_t sb = 1;
        if (type == SYC_SAMPLE_U8) {
          q = {std::round(c.r * 255) / 255, std::round(c.g * 255) / 255,
               std::round(c.b * 255) / 255};
          for (int i = 0; i < w * h; ++i)
            for (double v : {q.r, q.g, q.b}) raw.push_back(uint8_t(std::lround(v * 255)));
        } else if (type == SYC_SAMPLE_U16) {
          sb = 2;
          q = {std::round(c.r * 65535) / 65535, std::round(c.g * 65535) / 65535,
               std::round(c.b * 65535) / 65535};
          for (int i = 0; i < w * h; ++i)
            for (double v : {q.r, q.g, q.b}) {
              const uint16_t s = uint16_t(std::lround(v * 65535));
              raw.insert(raw.end(), reinterpret_cast<const unsigned char*>(&s),
                         reinterpret_cast<const unsigned char*>(&s) + 2);
            }
        } else {
          sb = 4;
          for (int i = 0; i < w * h; ++i)
            for (double v : {c.r, c.g, c.b}) {
              const float f = float(v);
              raw.insert(raw.end(), reinterpret_cast<const unsigned char*>(&f),
                         reinterpret_cast<const unsigned char*>(&f) + 4);
            }
        }
        DeviceBuffer in(raw);
        YuvBuffers out(w, h, setup.depth, false);
        const RgbImage src = Interleaved(in.ptr, size_t(w) * 3 * sb, w, h, type);
        CHECK_OK(session.Convert(src, out.Image()));
        std::vector<int> Y, Cb, Cr;
        out.Read(&Y, &Cb, &Cr);
        int ey, ecb, ecr;
        ReferenceYuv(q, setup, &ey, &ecb, &ecr);
        int worst = 0;
        for (int v : Y) worst = std::max(worst, std::abs(v - ey));
        for (int v : Cb) worst = std::max(worst, std::abs(v - ecb));
        for (int v : Cr) worst = std::max(worst, std::abs(v - ecr));
        // 16-bit: fp32 internals + transfer approximations, allow a few LSB.
        const int tol = setup.depth == 16 ? 4 : 1;
        if (worst > tol) {
          std::printf("  color (%.2f %.2f %.2f) type %d: worst |diff| %d (exp Y %d Cb %d Cr %d, got Y %d Cb %d Cr %d)\n",
                      c.r, c.g, c.b, int(type), worst, ey, ecb, ecr, Y[0], Cb[0], Cr[0]);
        }
        CHECK(worst <= tol);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Sharp quality on synthetic edges, decoded on the host (bilinear, siting).

double SrgbToLinear(double v) {
  return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4);
}
double SrgbToGamma(double l) {
  l = std::clamp(l, 0.0, 1.0);
  return l <= 0.0031308 ? 12.92 * l : 1.055 * std::pow(l, 1 / 2.4) - 0.055;
}

std::vector<Rgb> EdgeImage(int w, int h) {
  std::mt19937 rng(1);
  std::normal_distribution<double> noise(0, 0.02);
  std::vector<Rgb> img(size_t(w) * h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      Rgb c{0.1, 0.2, 0.8};
      if ((x / 3 + y / 5) % 2 == 0) c = {0.9, 0.1, 0.1};
      if (x % 7 == 1) c = {0.1, 0.9, 0.2};
      if (y % 9 == 2) c = {1.0, 1.0, 1.0};
      if ((x - y) % 11 == 0) c = {0.95, 0.9, 0.05};
      c = {std::clamp(c.r + noise(rng), 0.0, 1.0), std::clamp(c.g + noise(rng), 0.0, 1.0),
           std::clamp(c.b + noise(rng), 0.0, 1.0)};
      img[size_t(y) * w + x] = c;
    }
  }
  return img;
}

// Alpha test mask: an anti-aliased disc in a transparent surround (empty
// blocks next to visible ones), scattered visible pixels in the surround, and
// a transparent hole in the disc. RGB under alpha 0 is black, as in cut-out
// assets.
std::vector<double> AlphaMask(int w, int h) {
  std::mt19937 rng(2);
  std::uniform_real_distribution<double> u(0, 1);
  std::vector<double> a(size_t(w) * h);
  const double cx = 0.45 * w, cy = 0.5 * h, r = 0.35 * std::min(w, h) + 1;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      double v = std::clamp(r - std::hypot(x - cx, y - cy), 0.0, 1.0);  // 1 px rim
      if (v == 0 && u(rng) < 0.03) v = u(rng);
      if (std::abs(x - cx) < 2 && std::abs(y - cy) < 2) v = 0;
      a[size_t(y) * w + x] = v;
    }
  }
  return a;
}

// Interleaved RGBA float image of EdgeImage under AlphaMask (black where
// alpha is 0).
std::vector<float> RgbaImage(int w, int h) {
  const std::vector<Rgb> img = EdgeImage(w, h);
  const std::vector<double> a = AlphaMask(w, h);
  std::vector<float> f;
  for (size_t i = 0; i < img.size(); ++i) {
    const double k = a[i] > 0 ? 1.0 : 0.0;
    f.insert(f.end(), {float(k * img[i].r), float(k * img[i].g), float(k * img[i].b),
                       float(a[i])});
  }
  return f;
}

// The tiles against the whole-image solve, without and with alpha. An image
// within one 32-block tile (<= 64x64 px) at halo 0 is a single tile with no
// seams: the whole-image solve itself. Larger images are compared across
// tile sizes, whose seams fall in different places. Exact mode, and the
// default whenever the halo cap (3) covers the iteration count, must be
// bit-identical; the default at 4..6 iterations is an approximation,
// bounded here on a worst-case synthetic image. (private/experiments/algo/
// check_cuda.py compares exact mode with the independent Python solve.)
void TestTiles(cudaStream_t stream) {
  using detail::KernelConfig;
  KernelConfig& kc = detail::InternalKernelConfig();
  const KernelConfig saved = kc;
  struct Approx {
    int worst = 0, runs = 0;
    double differing = 0;
  };
  std::map<std::string, Approx> approx;  // by filter
  // Quick mode (sanitizer runs): every filter and tile size, fewer setups,
  // sizes and iteration counts.
  const std::vector<Setup> setups =
      Quick() ? std::vector<Setup>{kSetups[0]}
              : std::vector<Setup>{kSetups[0], kSetups[1], kSetups[2], kSetups[3], kSetups[4]};
  const std::vector<std::pair<int, int>> sizes =
      Quick() ? std::vector<std::pair<int, int>>{{67, 33}, {64, 64}, {3, 5}}
              : std::vector<std::pair<int, int>>{{301, 157}, {67, 33}, {64, 64}, {1, 1}, {3, 5}};
  const std::vector<int> iteration_counts =
      Quick() ? std::vector<int>{3, 6} : std::vector<int>{0, 1, 3, 4, 6};
  for (const Setup& setup : setups) {
    for (auto [w, h] : sizes) {
      const std::vector<Rgb> img = EdgeImage(w, h);
      std::vector<float> f;
      for (const Rgb& c : img) f.insert(f.end(), {float(c.r), float(c.g), float(c.b)});
      DeviceBuffer in(f), in_rgba(RgbaImage(w, h));
      const RgbImage rgba =
          Interleaved(in_rgba.ptr, size_t(w) * 16, w, h, SYC_SAMPLE_F32, SYC_INTERP_RGBA);
      const RgbImage rgb = Interleaved(in.ptr, size_t(w) * 12, w, h, SYC_SAMPLE_F32);
      for (bool alpha : {false, true}) {
        const RgbImage& src = alpha ? rgba : rgb;
        for (const auto& [filter_name_, filters] : TestFilters(setup.loc)) {
          const std::string filter_name = filter_name_ + (alpha ? " +alpha" : "");
          for (int iters : iteration_counts) {
            g_context = std::string("tiles ") + setup.name + " " + filter_name + " " +
                        std::to_string(w) + "x" + std::to_string(h) + " iters " +
                        std::to_string(iters);
            Session exact = MakeSession(stream, setup, filters, iters, true);
            Session fast = MakeSession(stream, setup, filters, iters, false);
            auto run = [&](KernelConfig c, Session& s) {
              kc = c;
              std::vector<int> all = Convert(s, src, setup.depth);
              kc = saved;
              return all;
            };
            const bool one_tile = (w + 1) / 2 <= 32 && (h + 1) / 2 <= 32;
            const std::vector<int> ref =
                one_tile ? run(KernelConfig{32, 0}, fast) : run(KernelConfig{32}, exact);
            for (int tile : {0, 16, 32}) {
              CHECK(run({tile}, exact) == ref);  // exact: halo = iterations
              const std::vector<int> got = run({tile}, fast);
              if (iters <= 3 || filter_name_ == "nearest") {
                CHECK(got == ref);  // the halo covers every iteration (nearest: none needed)
              } else {
                Approx& a = approx[filter_name];
                size_t n = 0;
                for (size_t i = 0; i < ref.size(); ++i) {
                  const int d = std::abs(got[i] - ref[i]);
                  a.worst = std::max(a.worst, d);
                  n += d != 0;
                }
                a.differing += double(n) / double(ref.size());
                ++a.runs;
              }
            }
          }
        }
      }
    }
  }
  kc = saved;
  std::printf("  tiles, default halo cap (4..6 iterations), worst-case synthetic image:\n");
  for (const auto& [name, a] : approx) {
    std::printf("    %-24s worst %d codes, %.3f%% of codes differ (mean)\n", name.c_str(),
                a.worst, 100.0 * a.differing / a.runs);
    g_context = "tiles (approximate) " + name;
    CHECK(a.worst <= 16);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// Mean |encoded luminance error| of the decoded image, in 8-bit codes.
double DecodeLuminanceError(const std::vector<Rgb>& src, int w, int h,
                            const std::vector<int>& Y, const std::vector<int>& Cb,
                            const std::vector<int>& Cr, syc_chroma_location loc) {
  const double kr = 0.2126, kb = 0.0722, kg = 1 - kr - kb;  // BT.709 full
  const int cw = (w + 1) / 2, ch = (h + 1) / 2;
  auto chroma = [&](const std::vector<int>& p, int x, int y) {
    x = std::clamp(x, 0, cw - 1);
    y = std::clamp(y, 0, ch - 1);
    return (p[size_t(y) * cw + x] - 128) / 255.0;
  };
  auto up = [&](const std::vector<int>& p, int px, int py) {
    const int by = py / 2, byn = (py % 2 == 0) ? by - 1 : by + 1;
    auto v = [&](int bx) { return 0.75 * chroma(p, bx, by) + 0.25 * chroma(p, bx, byn); };
    const int bx = px / 2;
    if (loc == SYC_CHROMA_LOCATION_CENTER) {
      return 0.75 * v(bx) + 0.25 * v(px % 2 == 0 ? bx - 1 : bx + 1);
    }
    return px % 2 == 0 ? v(bx) : 0.5 * (v(bx) + v(bx + 1));
  };
  double sum = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const double yy = Y[size_t(y) * w + x] / 255.0;
      const double u = up(Cb, x, y), v = up(Cr, x, y);
      // Linear Y'CbCr -> R'G'B', then clamp each channel.
      const double ru = yy + 2 * (1 - kr) * v;
      const double bu = yy + 2 * (1 - kb) * u;
      const double gu = (yy - kr * ru - kb * bu) / kg;
      const double r = std::clamp(ru, 0.0, 1.0);
      const double g = std::clamp(gu, 0.0, 1.0);
      const double b = std::clamp(bu, 0.0, 1.0);
      const double lum = kr * SrgbToLinear(r) + kg * SrgbToLinear(g) + kb * SrgbToLinear(b);
      const Rgb& s = src[size_t(y) * w + x];
      const double lum_s =
          kr * SrgbToLinear(s.r) + kg * SrgbToLinear(s.g) + kb * SrgbToLinear(s.b);
      sum += std::fabs(SrgbToGamma(lum) - SrgbToGamma(lum_s)) * 255;
    }
  }
  return sum / (double(w) * h);
}

void TestSharpQuality(cudaStream_t stream) {
  const int w = 64, h = 64;
  const std::vector<Rgb> img = EdgeImage(w, h);
  std::vector<float> f;
  for (const Rgb& c : img) f.insert(f.end(), {float(c.r), float(c.g), float(c.b)});
  DeviceBuffer in(f);
  const RgbImage src = Interleaved(in.ptr, size_t(w) * 12, w, h, SYC_SAMPLE_F32);
  for (syc_chroma_location loc : {SYC_CHROMA_LOCATION_CENTER, SYC_CHROMA_LOCATION_LEFT}) {
    g_context = loc == SYC_CHROMA_LOCATION_CENTER ? "quality center" : "quality left";
    const Setup setup{"", 1, 13, 1, SYC_RANGE_FULL, 8, loc};
    auto error = [&](int iters) {
      Session s = MakeSession(stream, setup, {}, iters);
      YuvBuffers out(w, h, 8, false);
      CHECK_OK(s.Convert(src, out.Image()));
      std::vector<int> Y, Cb, Cr;
      out.Read(&Y, &Cb, &Cr);
      return DecodeLuminanceError(img, w, h, Y, Cb, Cr, loc);
    };
    const double err[3] = {error(0), error(1), error(4)};
    std::printf("  %s: mean luminance error (8-bit codes) iters 0/1/4: %.2f / %.2f / %.2f\n",
                g_context.c_str(), err[0], err[1], err[2]);
    if (std::getenv("SHARPYUV_TRACE")) {
      for (int it = 0; it <= 6; ++it) std::printf("    iter %2d: %.3f\n", it, error(it));
    }
    // Measured: center 19.3 / 3.3 / 1.6, left 19.7 / 3.5 / 2.0.
    CHECK(err[1] < 0.25 * err[0]);
    CHECK(err[2] < 0.7 * err[1]);
    CHECK(err[2] < 2.5);
  }
}

// Different input layouts / types and output layouts give the same result.
void TestFormats(cudaStream_t stream) {
  g_context = "formats";
  const int w = 33, h = 19;
  const std::vector<Rgb> img = EdgeImage(w, h);

  struct Variant {
    const char* name;
    std::function<RgbImage(std::vector<std::unique_ptr<DeviceBuffer>>&)> make;
  };
  std::vector<uint8_t> rgb8, bgrx8, xrgb8, planes8[3];
  std::vector<uint16_t> rgb10;
  std::vector<float> planes32[3], rgb255;
  for (const Rgb& c : img) {
    const uint8_t r = uint8_t(std::lround(c.r * 255)), g = uint8_t(std::lround(c.g * 255)),
                  b = uint8_t(std::lround(c.b * 255));
    rgb8.insert(rgb8.end(), {r, g, b});
    bgrx8.insert(bgrx8.end(), {b, g, r, 7});  // X is ignored
    xrgb8.insert(xrgb8.end(), {0, r, g, b});
    planes8[0].push_back(r), planes8[1].push_back(g), planes8[2].push_back(b);
    // Exactly the U8 path's mapping (v * (1 / 255.f)): r / 255.f can differ
    // by one ulp, which may flip a code at a rounding boundary.
    planes32[0].push_back(r * (1.f / 255.f)), planes32[1].push_back(g * (1.f / 255.f)),
        planes32[2].push_back(b * (1.f / 255.f));
  }
  std::vector<Variant> variants = {
      {"rgb8", [&](auto& keep) {
         keep.push_back(std::make_unique<DeviceBuffer>(rgb8));
         return Interleaved(keep.back()->ptr, size_t(w) * 3, w, h, SYC_SAMPLE_U8);
       }},
      {"bgrx8", [&](auto& keep) {
         keep.push_back(std::make_unique<DeviceBuffer>(bgrx8));
         return Interleaved(keep.back()->ptr, size_t(w) * 4, w, h, SYC_SAMPLE_U8, SYC_INTERP_BGRX);
       }},
      {"xrgb8", [&](auto& keep) {
         keep.push_back(std::make_unique<DeviceBuffer>(xrgb8));
         return Interleaved(keep.back()->ptr, size_t(w) * 4, w, h, SYC_SAMPLE_U8, SYC_INTERP_XRGB);
       }},
      {"planar8", [&](auto& keep) {
         for (auto& p : planes8) keep.push_back(std::make_unique<DeviceBuffer>(p));
         const size_t n = keep.size();
         return Planar(keep[n - 3]->ptr, keep[n - 2]->ptr, keep[n - 1]->ptr, size_t(w), w, h,
                       SYC_SAMPLE_U8);
       }},
      {"planar32", [&](auto& keep) {
         for (auto& p : planes32) keep.push_back(std::make_unique<DeviceBuffer>(p));
         const size_t n = keep.size();
         return Planar(keep[n - 3]->ptr, keep[n - 2]->ptr, keep[n - 1]->ptr, size_t(w) * 4, w,
                       h, SYC_SAMPLE_F32);
       }},
  };

  for (const Setup& setup : {kSetups[0], kSetups[2]}) {
    Session session = MakeSession(stream, setup);
    std::vector<int> refY, refCb, refCr;
    bool first = true;
    for (Variant& v : variants) {
      g_context = std::string("formats ") + setup.name + " " + v.name;
      std::vector<std::unique_ptr<DeviceBuffer>> keep;
      const RgbImage src = v.make(keep);
      YuvBuffers out(w, h, setup.depth, false);
      CHECK_OK(session.Convert(src, out.Image()));
      std::vector<int> Y, Cb, Cr;
      out.Read(&Y, &Cb, &Cr);
      if (first) {
        refY = Y, refCb = Cb, refCr = Cr, first = false;
      } else {
        CHECK(Y == refY && Cb == refCb && Cr == refCr);
      }
    }
    // Output layouts: NV12/P010 (MSB), swapped chroma.
    std::vector<std::unique_ptr<DeviceBuffer>> keep;
    const RgbImage src = variants[0].make(keep);
    for (bool swap : {false, true}) {
      g_context = std::string("formats ") + setup.name + (swap ? " biplanar swap" : " biplanar");
      YuvBuffers out(w, h, setup.depth, true);
      const auto align = setup.depth > 8 ? SYC_ALIGN_MSB : SYC_ALIGN_LSB;
      CHECK_OK(session.Convert(src, out.Image(align, swap)));
      std::vector<int> Y, Cb, Cr;
      out.Read(&Y, &Cb, &Cr, setup.depth > 8 ? 16 - setup.depth : 0, swap);
      CHECK(Y == refY && Cb == refCb && Cr == refCr);
    }
  }
}

// Odd and tiny sizes must work; the default session must be the documented
// JPEG signal with bilinear centre and 6 iterations; duplicate filters merge
// and weights are relative.
void TestDefaultsAndSizes(cudaStream_t stream) {
  Session defaults;
  defaults.SetStream(stream).Init();
  const Setup jpeg{"jpeg", 1, 13, 6, SYC_RANGE_FULL, 8, SYC_CHROMA_LOCATION_CENTER};
  Session explicit_jpeg = MakeSession(stream, jpeg, {}, 6);
  const syc_decoder_filter kCenter{SYC_FILTER_BILINEAR, SYC_CHROMA_LOCATION_CENTER};
  const syc_decoder_filter kNearest{SYC_FILTER_NEAREST, SYC_CHROMA_LOCATION_CENTER};
  const FilterList blend = {{1, kCenter}, {1, kNearest}};
  const FilterList blend_dup = {{1, kCenter}, {4, kNearest}, {3, kCenter}};
  Session a = MakeSession(stream, jpeg, blend);
  Session b = MakeSession(stream, jpeg, blend_dup);
  for (auto [w, h] : {std::pair{1, 1}, {2, 1}, {1, 2}, {3, 3}, {129, 67}, {64, 31}}) {
    g_context = "size " + std::to_string(w) + "x" + std::to_string(h);
    const std::vector<Rgb> img = EdgeImage(w, h);
    std::vector<uint8_t> raw;
    for (const Rgb& c : img)
      for (double v : {c.r, c.g, c.b}) raw.push_back(uint8_t(std::lround(v * 255)));
    DeviceBuffer in(raw);
    const RgbImage src = Interleaved(in.ptr, size_t(w) * 3, w, h, SYC_SAMPLE_U8);
    CHECK(Convert(defaults, src, 8) == Convert(explicit_jpeg, src, 8));
    CHECK(Convert(a, src, 8) == Convert(b, src, 8));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());
}

// Alpha: the output plane, alpha = 1 everywhere against no alpha, a fully
// transparent image, and the planar layout against interleaved.
void TestAlpha(cudaStream_t stream) {
  const int w = 37, h = 23;
  for (int depth : {8, 10}) {
    g_context = "alpha depth " + std::to_string(depth);
    Setup setup = kSetups[0];
    setup.depth = depth;
    Session session = MakeSession(stream, setup);
    const std::vector<float> f = RgbaImage(w, h);
    DeviceBuffer in(f);
    const RgbImage rgba = Interleaved(in.ptr, size_t(w) * 16, w, h, SYC_SAMPLE_F32, SYC_INTERP_RGBA);

    // Alpha output: the input alpha, full range at the output depth (MSB
    // aligned with the other planes).
    const size_t bytes = depth == 8 ? 1 : 2, a_pitch = size_t(w + 5) * bytes;
    DeviceBuffer a_out(a_pitch * h);
    YuvBuffers out(w, h, depth, false);
    YuvImage dst = out.Image(depth == 8 ? SYC_ALIGN_LSB : SYC_ALIGN_MSB);
    dst.data[3] = a_out.ptr;
    dst.pitch[3] = a_pitch;
    CHECK_OK(session.Convert(rgba, dst));
    const auto raw = a_out.Download<unsigned char>();
    const int maxv = (1 << depth) - 1, shift = depth == 8 ? 0 : 16 - depth;
    int bad = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int want = int(std::lround(double(f[(size_t(y) * w + x) * 4 + 3]) * maxv));
        int got = raw[y * a_pitch + x * bytes];
        if (bytes == 2) {
          uint16_t v;
          std::memcpy(&v, &raw[y * a_pitch + x * 2], 2);
          got = v >> shift;
        }
        bad += got != want;
      }
    }
    CHECK(bad == 0);

    // Planar colour + A plane gives the same result as interleaved RGBA.
    std::vector<float> planes(size_t(w) * h * 4);
    for (size_t i = 0; i < size_t(w) * h; ++i) {
      for (int c = 0; c < 4; ++c) planes[c * size_t(w) * h + i] = f[i * 4 + c];
    }
    DeviceBuffer in_planar(planes);
    const float* pp = static_cast<const float*>(in_planar.ptr);
    const size_t n = size_t(w) * h;
    const RgbImage planar =
        Planar(pp, pp + n, pp + 2 * n, size_t(w) * 4, w, h, SYC_SAMPLE_F32, pp + 3 * n);
    YuvBuffers out_planar(w, h, depth, false);
    CHECK_OK(session.Convert(planar, out_planar.Image()));
    std::vector<int> Y1, Cb1, Cr1, Y2, Cb2, Cr2;
    out.Read(&Y1, &Cb1, &Cr1, shift);
    out_planar.Read(&Y2, &Cb2, &Cr2);
    CHECK(Y1 == Y2 && Cb1 == Cb2 && Cr1 == Cr2);

    // Alpha = 1 everywhere: the weighted targets reduce to the unweighted
    // ones (up to float rounding: within 1 code).
    std::vector<float> opaque = f;
    for (size_t i = 3; i < opaque.size(); i += 4) opaque[i] = 1.f;
    DeviceBuffer in_opaque(opaque);
    RgbImage op = rgba;
    op.data[0] = in_opaque.ptr;
    YuvBuffers out_a(w, h, depth, false), out_n(w, h, depth, false);
    CHECK_OK(session.Convert(op, out_a.Image()));
    op.interpretation = SYC_INTERP_RGBX;  // the same samples, alpha ignored
    CHECK_OK(session.Convert(op, out_n.Image()));
    out_a.Read(&Y1, &Cb1, &Cr1);
    out_n.Read(&Y2, &Cb2, &Cr2);
    int worst = 0;
    for (size_t i = 0; i < Y1.size(); ++i) worst = std::max(worst, std::abs(Y1[i] - Y2[i]));
    for (size_t i = 0; i < Cb1.size(); ++i) {
      worst = std::max({worst, std::abs(Cb1[i] - Cb2[i]), std::abs(Cr1[i] - Cr2[i])});
    }
    CHECK(worst <= 1);

    // Fully transparent: every block empty, nothing to inpaint from (blocks
    // keep their unweighted colour, luma is filled from it). No division by
    // a zero weight sum: valid codes.
    std::vector<float> clear = f;
    for (size_t i = 3; i < clear.size(); i += 4) clear[i] = 0.f;
    DeviceBuffer in_clear(clear);
    RgbImage cl = rgba;
    cl.data[0] = in_clear.ptr;
    YuvBuffers out_c(w, h, depth, false);
    CHECK_OK(session.Convert(cl, out_c.Image()));
    out_c.Read(&Y1, &Cb1, &Cr1);
    bool in_range = true;
    for (int v : Y1) in_range &= v >= 0 && v <= maxv;
    CHECK(in_range);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace

int main() {
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  // Progress with elapsed time on stderr, so a long (sanitizer) run shows
  // where it is.
  const auto start = std::chrono::steady_clock::now();
  auto run = [&](const char* name, auto&& test) {
    try {
      test();
    } catch (const std::exception& e) {
      ++g_failures;
      std::printf("FAIL [%s] uncaught exception: %s\n", g_context.c_str(), e.what());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::fprintf(stderr, "[%8.2f s] %s done\n", t, name);
    std::fflush(stderr);
  };
  if (Quick()) std::fprintf(stderr, "quick mode (SHARPYUV_TEST_QUICK)\n");
  run("factories", [] { TestFactories(); });
  run("session init", [&] { TestSessionInit(stream); });
  run("convert errors", [&] { TestConvertErrors(stream); });
  run("uniform", [&] { TestUniform(stream); });
  run("formats", [&] { TestFormats(stream); });
  run("defaults and sizes", [&] { TestDefaultsAndSizes(stream); });
  run("sharp quality", [&] { TestSharpQuality(stream); });
  run("tiles", [&] { TestTiles(stream); });
  run("alpha", [&] { TestAlpha(stream); });
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
  if (g_failures == 0) {
    std::printf("All tests passed.\n");
    return 0;
  }
  std::printf("%d check(s) failed.\n", g_failures);
  return 1;
}
