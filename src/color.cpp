// Color descriptors: H.273 tables and validating factories.

#include <algorithm>
#include <array>
#include <cmath>

#include "sharpyuv_cuda/sharpyuv.hpp"

namespace sharpyuv {
namespace {

struct PrimariesEntry {
  int code;
  Chromaticity r, g, b, w;
};

constexpr Chromaticity kD65{0.3127, 0.3290};
constexpr Chromaticity kIlluminantC{0.310, 0.316};
constexpr Chromaticity kDci{0.314, 0.351};

// H.273 ColourPrimaries.
constexpr PrimariesEntry kPrimaries[] = {
    {1, {0.640, 0.330}, {0.300, 0.600}, {0.150, 0.060}, kD65},          // BT.709
    {4, {0.670, 0.330}, {0.210, 0.710}, {0.140, 0.080}, kIlluminantC},  // BT.470M
    {5, {0.640, 0.330}, {0.290, 0.600}, {0.150, 0.060}, kD65},          // BT.601 625
    {6, {0.630, 0.340}, {0.310, 0.595}, {0.155, 0.070}, kD65},          // BT.601 525
    {7, {0.630, 0.340}, {0.310, 0.595}, {0.155, 0.070}, kD65},          // SMPTE 240M
    {8, {0.681, 0.319}, {0.243, 0.692}, {0.145, 0.049}, kIlluminantC},  // film
    {9, {0.708, 0.292}, {0.170, 0.797}, {0.131, 0.046}, kD65},          // BT.2020
    {11, {0.680, 0.320}, {0.265, 0.690}, {0.150, 0.060}, kDci},         // DCI-P3
    {12, {0.680, 0.320}, {0.265, 0.690}, {0.150, 0.060}, kD65},         // Display P3
    {22, {0.630, 0.340}, {0.295, 0.605}, {0.155, 0.077}, kD65},         // EBU 3213
};

struct TransferEntry {
  int code;
  TransferFunction::Kind kind;
  ParametricCurve curve;  // kParametric only
};

constexpr ParametricCurve kBt709Curve{1 / 0.45, 1.09929682680944,
                                      0.018053968510807, 4.5};
constexpr ParametricCurve kSrgbCurve{2.4, 1.055, 0.0031308, 12.92};
constexpr ParametricCurve kNone{0, 0, 0, 0};

// H.273 TransferCharacteristics. The BT.709 family is the camera curve
// (inverse OETF), not BT.1886.
constexpr TransferEntry kTransfers[] = {
    {1, TransferFunction::Kind::kParametric, kBt709Curve},
    {4, TransferFunction::Kind::kParametric, {2.2, 1.0, 0.0, 0.0}},
    {5, TransferFunction::Kind::kParametric, {2.8, 1.0, 0.0, 0.0}},
    {6, TransferFunction::Kind::kParametric, kBt709Curve},
    {7, TransferFunction::Kind::kParametric, {1 / 0.45, 1.1115, 0.0228, 4.0}},
    {8, TransferFunction::Kind::kLinear, kNone},
    {13, TransferFunction::Kind::kParametric, kSrgbCurve},
    {14, TransferFunction::Kind::kParametric, kBt709Curve},
    {15, TransferFunction::Kind::kParametric, kBt709Curve},
    {16, TransferFunction::Kind::kPq, kNone},
    {18, TransferFunction::Kind::kHlg, kNone},
};

struct MatrixEntry {
  int code;
  double kr, kb;
};

// H.273 MatrixCoefficients (Y'CbCr, non-constant luminance only). Code 12
// (chroma-derived) has no fixed coefficients: see FromH273.
constexpr int kChromaDerivedNcl = 12;
constexpr MatrixEntry kMatrices[] = {
    {1, 0.2126, 0.0722},  // BT.709
    {4, 0.30, 0.11},      // FCC
    {5, 0.299, 0.114},    // BT.601 625
    {6, 0.299, 0.114},    // BT.601 525
    {7, 0.212, 0.087},    // SMPTE 240M
    {9, 0.2627, 0.0593},  // BT.2020 NCL
};

bool ValidWeights(double kr, double kb) {
  return std::isfinite(kr) && std::isfinite(kb) && kr > 0 && kb > 0 &&
         kr + kb < 1;
}

bool ValidChromaticity(Chromaticity c) {
  return std::isfinite(c.x) && std::isfinite(c.y) && c.x >= 0 && c.y > 0 &&
         c.x + c.y <= 1;
}

}  // namespace

const char* ToString(Status s) noexcept {
  switch (s) {
    case Status::kOk: return "ok";
    case Status::kInvalidArgument: return "invalid argument";
    case Status::kUnsupported: return "unsupported";
    case Status::kNotReady: return "not ready";
    case Status::kImageTooLarge: return "image too large";
    case Status::kOutOfMemory: return "out of memory";
    case Status::kCudaError: return "CUDA error";
  }
  return "unknown status";
}

// ---------------------------------------------------------------------------

Status ColorPrimaries::FromH273(int code, ColorPrimaries* out) {
  if (out == nullptr) return Status::kInvalidArgument;
  for (const PrimariesEntry& e : kPrimaries) {
    if (e.code == code) return FromChromaticities(e.r, e.g, e.b, e.w, out);
  }
  return Status::kUnsupported;
}

// Luminance weights = Y row of the RGB -> XYZ matrix normalized to the white
// point: solve [X_r X_g X_b; Y_r Y_g Y_b; Z_r Z_g Z_b] s = W_XYZ (Y_w = 1)
// with primaries at Y = 1; the weights are s.
Status ColorPrimaries::FromChromaticities(Chromaticity r, Chromaticity g,
                                          Chromaticity b, Chromaticity w,
                                          ColorPrimaries* out) {
  if (out == nullptr) return Status::kInvalidArgument;
  for (Chromaticity c : {r, g, b, w}) {
    if (!ValidChromaticity(c)) return Status::kInvalidArgument;
  }
  auto col = [](Chromaticity c) {
    return std::array<double, 3>{c.x / c.y, 1.0, (1 - c.x - c.y) / c.y};
  };
  const auto cr = col(r), cg = col(g), cb = col(b), cw = col(w);
  // Cramer's rule on M = [cr cg cb] (columns).
  auto det3 = [](const std::array<double, 3>& a, const std::array<double, 3>& b,
                 const std::array<double, 3>& c) {
    return a[0] * (b[1] * c[2] - b[2] * c[1]) -
           b[0] * (a[1] * c[2] - a[2] * c[1]) +
           c[0] * (a[1] * b[2] - a[2] * b[1]);
  };
  const double det = det3(cr, cg, cb);
  if (!std::isfinite(det) || std::fabs(det) < 1e-12) {
    return Status::kInvalidArgument;
  }
  const double sr = det3(cw, cg, cb) / det;
  const double sg = det3(cr, cw, cb) / det;
  const double sb = det3(cr, cg, cw) / det;
  if (!(sr > 0 && sg > 0 && sb > 0)) return Status::kInvalidArgument;
  return FromLumaCoefficients(sr, sb, out);
}

Status ColorPrimaries::FromLumaCoefficients(double kr, double kb,
                                            ColorPrimaries* out) {
  if (out == nullptr || !ValidWeights(kr, kb)) return Status::kInvalidArgument;
  out->specified_ = true;
  out->kr_ = kr;
  out->kb_ = kb;
  return Status::kOk;
}

Status TransferFunction::FromH273(int code, TransferFunction* out) {
  if (out == nullptr) return Status::kInvalidArgument;
  for (const TransferEntry& e : kTransfers) {
    if (e.code != code) continue;
    if (e.kind == Kind::kParametric) return FromParametric(e.curve, out);
    out->kind_ = e.kind;
    out->curve_ = {};
    return Status::kOk;
  }
  return Status::kUnsupported;
}

Status TransferFunction::FromParametric(const ParametricCurve& c,
                                        TransferFunction* out) {
  if (out == nullptr) return Status::kInvalidArgument;
  if (!(std::isfinite(c.gamma) && std::isfinite(c.alpha) &&
        std::isfinite(c.beta) && std::isfinite(c.slope)) ||
      c.gamma < 1.0 || c.gamma > 10.0 || c.alpha < 1.0 || c.beta < 0.0 ||
      c.beta >= 1.0 || c.slope < 0.0) {
    return Status::kInvalidArgument;
  }
  if (c.beta > 0.0) {
    // Continuity at beta, and the toe must be the lower segment's tangent
    // side (encoded threshold inside [0, 1]).
    const double upper = c.alpha * std::pow(c.beta, 1.0 / c.gamma) - (c.alpha - 1.0);
    const double lower = c.slope * c.beta;
    if (!(lower > 0.0 && lower < 1.0) ||
        std::fabs(upper - lower) > 1e-3 * std::max(lower, 1e-3)) {
      return Status::kInvalidArgument;
    }
  }
  out->kind_ = Kind::kParametric;
  out->curve_ = c;
  return Status::kOk;
}

Status MatrixCoefficients::FromH273(int code, MatrixCoefficients* out) {
  if (out == nullptr) return Status::kInvalidArgument;
  if (code == kChromaDerivedNcl) {
    *out = MatrixCoefficients{};
    out->specified_ = true;
    out->from_primaries_ = true;
    return Status::kOk;
  }
  for (const MatrixEntry& e : kMatrices) {
    if (e.code == code) return FromCoefficients(e.kr, e.kb, out);
  }
  return Status::kUnsupported;
}

Status MatrixCoefficients::FromCoefficients(double kr, double kb,
                                            MatrixCoefficients* out) {
  if (out == nullptr || !ValidWeights(kr, kb)) return Status::kInvalidArgument;
  out->specified_ = true;
  out->from_primaries_ = false;
  out->kr_ = kr;
  out->kb_ = kb;
  return Status::kOk;
}

// ---------------------------------------------------------------------------

RgbImage RgbImage::Interleaved(const void* data, size_t pitch, int width,
                               int height, SampleType type, ChannelOrder order) {
  RgbImage img;
  img.width = width;
  img.height = height;
  img.type = type;
  img.planar = false;
  img.data = {data, nullptr, nullptr};
  img.pitch = {pitch, 0, 0};
  img.order = order;
  return img;
}

RgbImage RgbImage::Planar(const void* r, const void* g, const void* b,
                          size_t pitch, int width, int height, SampleType type) {
  RgbImage img;
  img.width = width;
  img.height = height;
  img.type = type;
  img.planar = true;
  img.data = {r, g, b};
  img.pitch = {pitch, pitch, pitch};
  return img;
}

YuvImage YuvImage::Planar(void* y, size_t y_pitch, void* u, size_t u_pitch,
                          void* v, size_t v_pitch, int width, int height) {
  YuvImage img;
  img.width = width;
  img.height = height;
  img.biplanar = false;
  img.data = {y, u, v};
  img.pitch = {y_pitch, u_pitch, v_pitch};
  return img;
}

YuvImage YuvImage::Biplanar(void* y, size_t y_pitch, void* uv, size_t uv_pitch,
                            int width, int height, SampleAlignment alignment) {
  YuvImage img;
  img.width = width;
  img.height = height;
  img.biplanar = true;
  img.alignment = alignment;
  img.data = {y, uv, nullptr};
  img.pitch = {y_pitch, uv_pitch, 0};
  return img;
}

}  // namespace sharpyuv
