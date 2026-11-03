// Color descriptors (C API): H.273 tables and validating factories.

#include <algorithm>
#include <array>
#include <cmath>

#include "internal.hpp"

namespace {

struct PrimariesEntry {
  int32_t code;
  syc_chromaticity r, g, b, w;
};

constexpr syc_chromaticity kD65{0.3127, 0.3290};
constexpr syc_chromaticity kIlluminantC{0.310, 0.316};
constexpr syc_chromaticity kDci{0.314, 0.351};

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
  int32_t code;
  syc_transfer_kind kind;
  syc_parametric_curve curve;  // SYC_TRANSFER_PARAMETRIC only
};

constexpr syc_parametric_curve kBt709Curve{1 / 0.45, 1.09929682680944, 0.018053968510807, 4.5};
constexpr syc_parametric_curve kSrgbCurve{2.4, 1.055, 0.0031308, 12.92};
constexpr syc_parametric_curve kNone{0, 0, 0, 0};

// H.273 TransferCharacteristics. The BT.709 family is the camera curve
// (inverse OETF), not BT.1886.
constexpr TransferEntry kTransfers[] = {
    {1, SYC_TRANSFER_PARAMETRIC, kBt709Curve},
    {4, SYC_TRANSFER_PARAMETRIC, {2.2, 1.0, 0.0, 0.0}},
    {5, SYC_TRANSFER_PARAMETRIC, {2.8, 1.0, 0.0, 0.0}},
    {6, SYC_TRANSFER_PARAMETRIC, kBt709Curve},
    {7, SYC_TRANSFER_PARAMETRIC, {1 / 0.45, 1.1115, 0.0228, 4.0}},
    {8, SYC_TRANSFER_LINEAR, kNone},
    {13, SYC_TRANSFER_PARAMETRIC, kSrgbCurve},
    {14, SYC_TRANSFER_PARAMETRIC, kBt709Curve},
    {15, SYC_TRANSFER_PARAMETRIC, kBt709Curve},
    {16, SYC_TRANSFER_PQ, kNone},
    {18, SYC_TRANSFER_HLG, kNone},
};

struct MatrixEntry {
  int32_t code;
  double kr, kb;
};

// H.273 MatrixCoefficients (Y'CbCr, non-constant luminance only). Code 12
// (chroma-derived) has no fixed coefficients: see syc_matrix_from_h273.
constexpr int32_t kChromaDerivedNcl = 12;
constexpr MatrixEntry kMatrices[] = {
    {1, 0.2126, 0.0722},  // BT.709
    {4, 0.30, 0.11},      // FCC
    {5, 0.299, 0.114},    // BT.601 625
    {6, 0.299, 0.114},    // BT.601 525
    {7, 0.212, 0.087},    // SMPTE 240M
    {9, 0.2627, 0.0593},  // BT.2020 NCL
};

bool ValidChromaticity(syc_chromaticity c) {
  return std::isfinite(c.x) && std::isfinite(c.y) && c.x >= 0 && c.y > 0 && c.x + c.y <= 1;
}

}  // namespace

namespace sharpyuv::detail {

bool ValidWeights(double kr, double kb) {
  return std::isfinite(kr) && std::isfinite(kb) && kr > 0 && kb > 0 && kr + kb < 1;
}

bool ValidCurve(const syc_parametric_curve& c) {
  if (!(std::isfinite(c.gamma) && std::isfinite(c.alpha) && std::isfinite(c.beta) &&
        std::isfinite(c.slope)) ||
      c.gamma < 1.0 || c.gamma > 10.0 || c.alpha < 1.0 || c.beta < 0.0 || c.beta >= 1.0 ||
      c.slope < 0.0) {
    return false;
  }
  if (c.beta > 0.0) {
    // Continuity at beta, and the toe must be the lower segment's tangent
    // side (encoded threshold inside [0, 1]).
    const double upper = c.alpha * std::pow(c.beta, 1.0 / c.gamma) - (c.alpha - 1.0);
    const double lower = c.slope * c.beta;
    if (!(lower > 0.0 && lower < 1.0) ||
        std::fabs(upper - lower) > 1e-3 * std::max(lower, 1e-3)) {
      return false;
    }
  }
  return true;
}

}  // namespace sharpyuv::detail

using sharpyuv::detail::ValidCurve;
using sharpyuv::detail::ValidWeights;

extern "C" {

uint32_t syc_version(void) { return SYC_VERSION; }

const char* syc_status_string(syc_status status) {
  switch (status) {
    case SYC_STATUS_OK: return "ok";
    case SYC_STATUS_INVALID_ARGUMENT: return "invalid argument";
    case SYC_STATUS_UNSUPPORTED: return "unsupported";
    case SYC_STATUS_NOT_READY: return "not ready";
    case SYC_STATUS_IMAGE_TOO_LARGE: return "image too large";
    case SYC_STATUS_OUT_OF_MEMORY: return "out of memory";
    case SYC_STATUS_CUDA_ERROR: return "CUDA error";
  }
  return "unknown status";
}

syc_status syc_primaries_from_h273(int32_t code, syc_color_primaries* out) {
  if (out == nullptr) return SYC_STATUS_INVALID_ARGUMENT;
  for (const PrimariesEntry& e : kPrimaries) {
    if (e.code == code) return syc_primaries_from_chromaticities(e.r, e.g, e.b, e.w, out);
  }
  return SYC_STATUS_UNSUPPORTED;
}

// Luminance weights = Y row of the RGB -> XYZ matrix normalized to the white
// point: solve [X_r X_g X_b; Y_r Y_g Y_b; Z_r Z_g Z_b] s = W_XYZ (Y_w = 1)
// with primaries at Y = 1; the weights are s.
syc_status syc_primaries_from_chromaticities(syc_chromaticity r, syc_chromaticity g,
                                             syc_chromaticity b, syc_chromaticity w,
                                             syc_color_primaries* out) {
  if (out == nullptr) return SYC_STATUS_INVALID_ARGUMENT;
  for (syc_chromaticity c : {r, g, b, w}) {
    if (!ValidChromaticity(c)) return SYC_STATUS_INVALID_ARGUMENT;
  }
  auto col = [](syc_chromaticity c) {
    return std::array<double, 3>{c.x / c.y, 1.0, (1 - c.x - c.y) / c.y};
  };
  const auto cr = col(r), cg = col(g), cb = col(b), cw = col(w);
  // Cramer's rule on M = [cr cg cb] (columns).
  auto det3 = [](const std::array<double, 3>& a, const std::array<double, 3>& b,
                 const std::array<double, 3>& c) {
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1]) +
           c[0] * (a[1] * b[2] - a[2] * b[1]);
  };
  const double det = det3(cr, cg, cb);
  if (!std::isfinite(det) || std::fabs(det) < 1e-12) return SYC_STATUS_INVALID_ARGUMENT;
  const double sr = det3(cw, cg, cb) / det;
  const double sg = det3(cr, cw, cb) / det;
  const double sb = det3(cr, cg, cw) / det;
  if (!(sr > 0 && sg > 0 && sb > 0)) return SYC_STATUS_INVALID_ARGUMENT;
  return syc_primaries_from_luma_coefficients(sr, sb, out);
}

syc_status syc_primaries_from_luma_coefficients(double kr, double kb, syc_color_primaries* out) {
  if (out == nullptr || !ValidWeights(kr, kb)) return SYC_STATUS_INVALID_ARGUMENT;
  out->kr = kr;
  out->kb = kb;
  return SYC_STATUS_OK;
}

syc_status syc_transfer_from_h273(int32_t code, syc_transfer_function* out) {
  if (out == nullptr) return SYC_STATUS_INVALID_ARGUMENT;
  for (const TransferEntry& e : kTransfers) {
    if (e.code != code) continue;
    if (e.kind == SYC_TRANSFER_PARAMETRIC) return syc_transfer_from_parametric(&e.curve, out);
    out->kind = e.kind;
    out->curve = kNone;
    return SYC_STATUS_OK;
  }
  return SYC_STATUS_UNSUPPORTED;
}

syc_status syc_transfer_from_parametric(const syc_parametric_curve* curve,
                                        syc_transfer_function* out) {
  if (out == nullptr || curve == nullptr || !ValidCurve(*curve)) {
    return SYC_STATUS_INVALID_ARGUMENT;
  }
  out->kind = SYC_TRANSFER_PARAMETRIC;
  out->curve = *curve;
  return SYC_STATUS_OK;
}

syc_status syc_matrix_from_h273(int32_t code, syc_matrix_coefficients* out) {
  if (out == nullptr) return SYC_STATUS_INVALID_ARGUMENT;
  if (code == kChromaDerivedNcl) {
    *out = {SYC_MATRIX_FROM_PRIMARIES, 0.0, 0.0};
    return SYC_STATUS_OK;
  }
  for (const MatrixEntry& e : kMatrices) {
    if (e.code == code) return syc_matrix_from_coefficients(e.kr, e.kb, out);
  }
  return SYC_STATUS_UNSUPPORTED;
}

syc_status syc_matrix_from_coefficients(double kr, double kb, syc_matrix_coefficients* out) {
  if (out == nullptr || !ValidWeights(kr, kb)) return SYC_STATUS_INVALID_ARGUMENT;
  *out = {SYC_MATRIX_COEFFICIENTS, kr, kb};
  return SYC_STATUS_OK;
}

}  // extern "C"
