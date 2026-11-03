// Device-side transfer functions. Choices follow private/experiments/tf_bench/RESULTS.md:
// MUFU intrinsics for parametric curves and HLG, the numerically stable
// rewrite for PQ -> linear (a cheaper fit for <= 12-bit output), a log-log
// fit for PQ -> encoded. Piecewise curves
// evaluate both segments and select (a ?: around the expensive segment
// compiles to a real branch).
//
// Interface per TfKind (vector-valued because the HLG OOTF couples channels):
//   ToLinear(rgb')     encoded RGB -> linear light (display light for HLG)
//   ToEncoded(rgb)     linear light -> encoded RGB
//   EncodeLuminance(Y) linear-light luminance -> encoded value comparable to W
//   EncodeScene(c)     per-channel OETF, for linear-light input
#pragma once

#include "internal.hpp"

namespace sharpyuv::detail {

struct Rgb {
  float r, g, b;
};

__device__ __forceinline__ float Dot(const float k[3], Rgb c) {
  return fmaf(k[0], c.r, fmaf(k[1], c.g, k[2] * c.b));
}
__device__ __forceinline__ Rgb Scale(Rgb c, float s) {
  return {c.r * s, c.g * s, c.b * s};
}
template <class F>
__device__ __forceinline__ Rgb Map(Rgb c, F f) {
  return {f(c.r), f(c.g), f(c.b)};
}
__device__ __forceinline__ float Clamp01(float v) {
  return fminf(fmaxf(v, 0.f), 1.f);  // also maps NaN to 0
}

// ---------------------------------------------------------------------------
// Parametric power curve with linear toe.

__device__ __forceinline__ float ParamToLinear(float v, const TfParams& p) {
  const float y = __powf(fmaf(v, p.inv_alpha, p.alpha_m1 * p.inv_alpha), p.gamma);
  return v < p.v_beta ? v * p.inv_slope : y;
}
__device__ __forceinline__ float ParamToEncoded(float l, const TfParams& p) {
  const float y = fmaf(p.alpha, __powf(l, p.inv_gamma), -p.alpha_m1);
  return l < p.beta ? l * p.slope : y;
}

// ---------------------------------------------------------------------------
// PQ (SMPTE ST 2084). Constants: 1 - c1 == c2 - c3 == 21/128.

namespace pq {
inline constexpr float kM1 = 2610.f / 16384.f;
inline constexpr float kM2 = 2523.f / 4096.f * 128.f;
inline constexpr float kC3 = 2392.f / 4096.f * 32.f;
inline constexpr float kA = 21.f / 128.f;
inline constexpr float kBlack = 7.30955903e-07f;  // PQ(0) = c1^m2

// expm1(t) = t P(t) on [-0.19, 0], rel. err <= 7e-10
__device__ __forceinline__ float Expm1Small(float t) {
  float p = 0.0076997816f;
  p = fmaf(p, t, 0.041562375f);
  p = fmaf(p, t, 0.1666596f);
  p = fmaf(p, t, 0.49999982f);
  p = fmaf(p, t, 1.0f);
  return t * p;
}

// Stable form: q = x^(1/m2) - 1, L = ((A + q) / (A - c3 q))^(1/m1).
__device__ __forceinline__ float ToLinear(float x) {
  const float t =
      fmaxf(__log2f(fmaxf(x, 1e-30f)) * (0.69314718f / kM2), -0.19f);
  const float q = Expm1Small(t);
  const float r = __fdividef(fmaxf(kA + q, 0.f), fmaf(-kC3, q, kA));
  return __powf(r, 1.f / kM1);
}

// For <= 12-bit output: L = R32(u)^8, u = log2(x) / 10 + 1 on [2^-20, 1],
// error is at most 0.051 12-bit codes (enough for 8/10/12-bit output).
// One lg2 and one rcp, instead of the stable form's lg2, rcp, lg2, ex2 and polynomial.
__device__ __forceinline__ float ToLinear12(float x) {
  const float u = fmaf(__log2f(fmaxf(x, 9.536743e-07f)), 0.1f, 1.0f);
  float p = 0.01615351f;
  p = fmaf(p, u, -0.06608771f);
  p = fmaf(p, u, -0.015321574f);
  p = fmaf(p, u, 0.089093514f);
  float q = 0.679887f;
  q = fmaf(q, u, -1.656049f);
  q = fmaf(q, u, 1.0f);
  float z = __fdividef(p, q);
  z *= z;
  z *= z;
  return z * z;
}

// log2(x) = P8(u), u = log2(L) / 20 + 1 on [2^-40, 1]; 0.011 12-bit codes.
__device__ __forceinline__ float ToEncoded(float l) {
  const float u = fmaf(__log2f(fmaxf(l, 9.094947e-13f)), 0.05f, 1.0f);
  float p = -0.08041643f;
  p = fmaf(p, u, 0.51168746f);
  p = fmaf(p, u, -0.9593791f);
  p = fmaf(p, u, -0.15927273f);
  p = fmaf(p, u, 2.7286592f);
  p = fmaf(p, u, -2.2299592f);
  p = fmaf(p, u, -4.280261f);
  p = fmaf(p, u, 10.043622f);
  p = fmaf(p, u, -5.574677f);
  const float x = exp2f(p);
  return l > 0.f ? x : kBlack;
}
}  // namespace pq

// ---------------------------------------------------------------------------
// HLG (ARIB STD-B67 / BT.2100): OETF and inverse, scene light.

namespace hlg {
inline constexpr float kA = 0.17883277f;
inline constexpr float kB = 0.28466892f;   // 1 - 4a
inline constexpr float kC = 0.55991073f;   // 0.5 - a ln(4a)

__device__ __forceinline__ float InvOetf(float v) {
  const float y = (__expf((v - kC) * (1.f / kA)) + kB) * (1.f / 12.f);
  return v <= 0.5f ? v * v * (1.f / 3.f) : y;
}
// sqrt(s) = s rsqrt(s) with the flush-to-zero MUFU op to avoid divergent branch:
// s >= 3e-30 is never denormal, but rsqrtf's adds a denormal guard causing divergent.
__device__ __forceinline__ float RsqrtFtz(float s) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(s));
  return r;
}
__device__ __forceinline__ float Oetf(float e) {
  const float y = fmaf(kA, __logf(fmaxf(fmaf(12.f, e, -kB), 1e-30f)), kC);
  const float s = 3.f * fmaxf(e, 1e-30f);
  return e <= 1.f / 12.f ? s * RsqrtFtz(s) : y;
}
}  // namespace hlg

// ---------------------------------------------------------------------------

template <TfKind kTf>
struct Transfer;

template <>
struct Transfer<TfKind::kParametric> {
  static __device__ Rgb ToLinear(Rgb c, const ColorParams& cp) {
    return Map(c, [&](float v) { return ParamToLinear(v, cp.tfp); });
  }
  static __device__ Rgb ToEncoded(Rgb c, const ColorParams& cp) {
    return Map(c, [&](float v) { return ParamToEncoded(v, cp.tfp); });
  }
  static __device__ float EncodeLuminance(float y, const ColorParams& cp) {
    return ParamToEncoded(y, cp.tfp);
  }
  static __device__ float EncodeScene(float v, const ColorParams& cp) {
    return ParamToEncoded(v, cp.tfp);
  }
};

template <>
struct Transfer<TfKind::kPq> {
  static __device__ Rgb ToLinear(Rgb c, const ColorParams&) {
    return Map(c, [](float v) { return pq::ToLinear(v); });
  }
  static __device__ Rgb ToEncoded(Rgb c, const ColorParams&) {
    return Map(c, [](float v) { return pq::ToEncoded(v); });
  }
  static __device__ float EncodeLuminance(float y, const ColorParams&) {
    return pq::ToEncoded(y);
  }
  static __device__ float EncodeScene(float v, const ColorParams&) {
    return pq::ToEncoded(v);
  }
};

// PQ for <= 12-bit output: use cheaper decode, the rest as kPq.
template <>
struct Transfer<TfKind::kPq12> : Transfer<TfKind::kPq> {
  static __device__ Rgb ToLinear(Rgb c, const ColorParams&) {
    return Map(c, [](float v) { return pq::ToLinear12(v); });
  }
};

// HLG with the BT.2100 OOTF (normalized display light, alpha = 1, beta = 0;
// both cancel in the algorithm). Y uses the CP luminance weights.
template <>
struct Transfer<TfKind::kHlg> {
  static __device__ Rgb ToLinear(Rgb c, const ColorParams& cp) {
    const Rgb e = Map(c, [](float v) { return hlg::InvOetf(v); });
    const float ys = Dot(cp.kl, e);
    return Scale(e, __powf(ys, cp.tfp.ootf_dec_exp));  // 0^x == 0
  }
  static __device__ Rgb ToEncoded(Rgb f, const ColorParams& cp) {
    const float yd = fmaxf(Dot(cp.kl, f), 1e-30f);
    const Rgb e = Scale(f, __powf(yd, cp.tfp.ootf_enc_exp));
    return Map(e, [](float v) { return hlg::Oetf(v); });
  }
  // Achromatic pixel: Y_s = Y_d^(1/gamma).
  static __device__ float EncodeLuminance(float y, const ColorParams& cp) {
    return hlg::Oetf(__powf(y, cp.tfp.ootf_inv_gamma));
  }
  // Linear-light input is scene light.
  static __device__ float EncodeScene(float v, const ColorParams&) {
    return hlg::Oetf(v);
  }
};

template <>
struct Transfer<TfKind::kLinear> {
  static __device__ Rgb ToLinear(Rgb c, const ColorParams&) { return c; }
  static __device__ Rgb ToEncoded(Rgb c, const ColorParams&) { return c; }
  static __device__ float EncodeLuminance(float y, const ColorParams&) {
    return y;
  }
  static __device__ float EncodeScene(float v, const ColorParams&) { return v; }
};

}  // namespace sharpyuv::detail
