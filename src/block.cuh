// Per-block math of the conversion kernel (kernel.cu): loading, transfer,
// upsampling, the target filter, init / Jacobi step / store of one 2x2 block.
// (The archived multi-kernel path, archive/multi_kernel, ran the same steps.)
#pragma once

#include <cuda_fp16.h>

#include "internal.hpp"
#include "transfer.cuh"

namespace sharpyuv::detail {

__device__ __forceinline__ Rgb Add(Rgb a, Rgb b) {
  return {a.r + b.r, a.g + b.g, a.b + b.b};
}
__device__ __forceinline__ Rgb Madd(Rgb a, float s, Rgb acc) {
  return {fmaf(a.r, s, acc.r), fmaf(a.g, s, acc.g), fmaf(a.b, s, acc.b)};
}

// ---------------------------------------------------------------------------
// Input

__device__ __forceinline__ float LoadSample(const unsigned char* p, InType t,
                                            ptrdiff_t index) {
  switch (t) {
    case InType::kU8: return float(p[index]);
    case InType::kU16: return float(reinterpret_cast<const uint16_t*>(p)[index]);
    case InType::kF16: return __half2float(reinterpret_cast<const __half*>(p)[index]);
    default: return reinterpret_cast<const float*>(p)[index];
  }
}

// Channels 0..kN-1 (R, G, B, and A for kN == 4) of pixel (x, y), clamped
// into the image, mapped to [0, 1]. One address computation for all.
template <int kN>
__device__ __forceinline__ void LoadChannels(const InputDesc& in, const Geometry& geo, int x,
                                             int y, float v[kN]) {
  x = min(max(x, 0), geo.width - 1);
  y = min(max(y, 0), geo.height - 1);
  if (in.planar) {
#pragma unroll
    for (int c = 0; c < kN; ++c) {
      v[c] = LoadSample(in.ptr[c] + size_t(y) * in.pitch[c], in.type, x);
    }
  } else {
    const unsigned char* row = in.ptr[0] + size_t(y) * in.pitch[0];
    const ptrdiff_t base = ptrdiff_t(x) * in.stride;
    // Vector load possible but this is not bottleneck yet
#pragma unroll
    for (int c = 0; c < kN; ++c) v[c] = LoadSample(row, in.type, base + in.off[c]);
  }
#pragma unroll
  for (int c = 0; c < kN; ++c) v[c] = Clamp01(fmaf(v[c], in.scale, in.offset));
}

template <TfKind kTf>
__device__ __forceinline__ Rgb EncodeInput(const InputDesc& in, const ColorParams& cp,
                                           const float v[3]) {
  if (!in.linear) return {v[0], v[1], v[2]};
  return {Transfer<kTf>::EncodeScene(v[0], cp), Transfer<kTf>::EncodeScene(v[1], cp),
          Transfer<kTf>::EncodeScene(v[2], cp)};
}

// Source pixel (x, y), clamped into the image, as encoded RGB' in [0, 1].
template <TfKind kTf>
__device__ Rgb LoadPixel(const InputDesc& in, const Geometry& geo,
                         const ColorParams& cp, int x, int y) {
  float v[3];
  LoadChannels<3>(in, geo, x, y, v);
  return EncodeInput<kTf>(in, cp, v);
}

// Pixel (x, y) as LoadPixel, plus its alpha in [0, 1] (straight alpha; same
// sample mapping), loaded together.
template <TfKind kTf>
__device__ __forceinline__ Rgb LoadPixelAlpha(const InputDesc& in, const Geometry& geo,
                                              const ColorParams& cp, int x, int y, float* a) {
  float v[4];
  LoadChannels<4>(in, geo, x, y, v);
  *a = v[3];
  return EncodeInput<kTf>(in, cp, v);
}

// ---------------------------------------------------------------------------
// Chroma neighbourhood and reconstruction

// Chroma state of a block: R'-Y' and B'-Y'. G'-Y' is never stored: k_M . d
// == 0 gives it, also for upsampled values (the upsampler is linear), so the
// state and shared memory hold 2 floats (one 8-byte load).
struct alignas(8) Chroma {
  float r, b;
};

__device__ __forceinline__ Chroma Madd(Chroma a, float s, Chroma acc) {
  return {fmaf(a.r, s, acc.r), fmaf(a.b, s, acc.b)};
}

// G'-Y' from R'-Y' and B'-Y' (k_M . d == 0).
__device__ __forceinline__ float DeriveG(Chroma d, const ColorParams& cp) {
  return fmaf(cp.g_from_r, d.r, cp.g_from_b * d.b);
}

struct Chroma3x3 {
  Chroma d[3][3];  // [dy + 1][dx + 1]
};

// Linear block colours of a 3x3 neighbourhood (the alpha fill).
struct Rgb3x3 {
  Rgb d[3][3];
};

// W of the pixels around the block that the target window reads (see
// Window): left[sy] = pixel (sy, -1), top[sx] = pixel (-1, sx), corner =
// pixel (-1, -1).
struct NeighbourW {
  float left[2], top[2], corner;
};

// Upsampled chroma of a pixel at offset (oy, ox) in {-1, 0, 1} from the
// block's first pixel. Every 1-D mode reads taps (-1, 0) for phase 0 and
// (0, +1) for phase 1 (filter_coeffs.py), so a pixel reads a fixed 2x2 of the
// 3x3 blocks whatever the mixture. Offset -1 is phase 1 of the block before:
// its taps (0, +1) are the centre block's (-1, 0). Indices are compile-time
// after unrolling.
template <Window kWin, class T>
__device__ __forceinline__ T UpsampleTaps(const T (&n)[3][3], const FilterParams& f, int oy,
                                          int ox) {
  if constexpr (kWin == Window::kLocal) {
    return n[1][1];
  } else {
    const int by = oy < 0 ? -1 : 0, py = oy < 0 ? 1 : oy;
    const int bx = ox < 0 ? -1 : 0, px = ox < 0 ? 1 : ox;
    T acc{};
#pragma unroll
    for (int dy = 0; dy < 3; ++dy) {
#pragma unroll
      for (int dx = 0; dx < 3; ++dx) {
        if (dy == (py == 0 ? 2 : 0) || dx == (px == 0 ? 2 : 0)) continue;  // zero tap
        acc = Madd(n[by + dy][bx + dx], f.up[py][px][dy][dx], acc);
      }
    }
    return acc;
  }
}

// Upsampled chroma offsets (R'-Y', G'-Y', B'-Y') of the pixel at (oy, ox).
template <Window kWin>
__device__ __forceinline__ Rgb Upsample(const Chroma3x3& n, const ColorParams& cp, int oy,
                                        int ox) {
  const Chroma c = UpsampleTaps<kWin>(n.d, cp.filter, oy, ox);
  return {c.r, DeriveG(c, cp), c.b};
}

// Upsampled linear colour of the pixel at (oy, ox) (the alpha fill).
template <Window kWin>
__device__ __forceinline__ Rgb UpsampleColour(const Rgb3x3& n, const FilterParams& f, int oy,
                                              int ox) {
  return UpsampleTaps<kWin>(n.d, f, oy, ox);
}

__device__ __forceinline__ Rgb Recon(float w, Rgb up) {
  return {Clamp01(w + up.r), Clamp01(w + up.g), Clamp01(w + up.b)};
}

// Target / feedback filter weights, accumulated pixel by pixel (no array of
// reconstructed pixels: it would hold 27 floats next to the 27 of the
// neighbourhood). The 2x2 windows (kBox, kLocal) sum the block's own pixels
// and scale once (every centred or nearest member's target is that box);
// wider windows use the table. At the image's left / top edge pixel -1
// replicates pixel 0, so its weight moves there and it is not reconstructed.
template <Window kWin>
struct TargetWeights {
  static constexpr bool kTable = kHasCols<kWin> || kHasRows<kWin>;
  const FilterParams& f;
  bool left_edge, top_edge;

  // Weight of the pixel at (oy, ox) (compile-time after unrolling); 1 for
  // the 2x2 windows (Finish scales).
  __device__ __forceinline__ float Weight(int oy, int ox) const {
    if constexpr (kTable) {
      // Read from the (constant-bank) table at each use: holding an
      // edge-adjusted copy would cost 9 registers.
      float t = f.tgt[oy + 1][ox + 1];
      const bool from_left = kHasCols<kWin> && ox == 0 && left_edge;
      const bool from_top = kHasRows<kWin> && oy == 0 && top_edge;
      if (from_left) t += f.tgt[oy + 1][0];
      if (from_top) t += f.tgt[0][ox + 1];
      if (from_left && from_top) t += f.tgt[0][0];
      return t;
    } else {
      return 1.f;
    }
  }
  // Adds the pixel at (oy, ox).
  __device__ __forceinline__ void Add(Rgb* acc, Rgb lin, int oy, int ox) const {
    if constexpr (kTable) {
      *acc = Madd(lin, Weight(oy, ox), *acc);
    } else {
      *acc = detail::Add(*acc, lin);
    }
  }
  __device__ __forceinline__ Rgb Finish(Rgb acc) const {
    if constexpr (kTable) {
      return acc;
    } else {
      return Scale(acc, 0.25f);
    }
  }
};

// Linear-light chroma (the filtered average) -> encoded chroma offsets
// R'-Y', B'-Y'.
template <TfKind kTf>
__device__ __forceinline__ Chroma ChromaOffsets(Rgb avg, const ColorParams& cp) {
  const Rgb c = Transfer<kTf>::ToEncoded(avg, cp);
  const float y = Dot(cp.km, c);
  return {c.r - y, c.b - y};
}

// Keep d representable (anti-windup): clamp R'-Y' and B'-Y' to the code
// range (G'-Y' follows from k_M . d == 0). Without this, chroma keeps growing
// where the reconstruction clips, the output quantizer clamps it, and the
// decoder no longer sees what the model optimized.
__device__ __forceinline__ Chroma ProjectChroma(Chroma d, const ColorParams& cp) {
  return {fminf(fmaxf(d.r, cp.dr_min), cp.dr_max), fminf(fmaxf(d.b, cp.db_min), cp.db_max)};
}

// ---------------------------------------------------------------------------
// Per-block steps. W and targets are indexed [sy][sx] over the block's 2x2
// pixels (the even-padded image, so all four exist).

struct BlockInit {
  float wt[2][2];  // W target: encoded linear-light luminance
  float w[2][2];   // start value: k_M . src'
  Chroma dt;       // chroma target (and start value)
};

template <TfKind kTf, Window kWin>
__device__ __forceinline__ BlockInit InitBlock(const InputDesc& in, const Geometry& geo,
                                               const ColorParams& cp, int bx, int by) {
  BlockInit b;
  // LoadPixel clamps, so pixel -1 at the image edge already replicates
  // pixel 0: the table weights apply unchanged.
  const TargetWeights<kWin> tw{cp.filter, false, false};
  Rgb acc{0.f, 0.f, 0.f};
#pragma unroll
  for (int oy = -1; oy < 2; ++oy) {
#pragma unroll
    for (int ox = -1; ox < 2; ++ox) {
      if ((oy < 0 && !kHasRows<kWin>) || (ox < 0 && !kHasCols<kWin>)) continue;
      const Rgb src = LoadPixel<kTf>(in, geo, cp, 2 * bx + ox, 2 * by + oy);
      const Rgb lin = Transfer<kTf>::ToLinear(src, cp);
      tw.Add(&acc, lin, oy, ox);
      if (oy >= 0 && ox >= 0) {
        b.wt[oy][ox] = Transfer<kTf>::EncodeLuminance(Dot(cp.kl, lin), cp);
        b.w[oy][ox] = Dot(cp.km, src);
      }
    }
  }
  b.dt = ChromaOffsets<kTf>(tw.Finish(acc), cp);
  return b;
}

// Alpha state of a block, constant over the iterations (docs/API.md "Alpha"):
// the target / feedback weight of each window pixel, q = table weight x alpha
// (edge-adjusted: pixel -1 at the image's left / top edge is not read, its
// weight moves to pixel 0), and inv = 1 / sum q. The own pixels' q are
// rounded to half and packed (2 registers instead of 4: the 1024-thread
// tiles are capped at 64 registers and spilled). inv is taken over the
// rounded values, so the weights still sum to 1; init and iterations use the
// same values. q > 0 exactly when alpha > 0 (the smallest product,
// 1/65535 x 1/16, is far above the half range's bottom).
template <Window kWin>
struct BlockAlpha {
  __half2 own[2];  // row sy: (q(sy, 0), q(sy, 1))
  // Wide windows: q of pixel column / row -1, [oy + 1][ox + 1] (row / column
  // 0 used only).
  float ext[3][3];
  float inv;  // 0: no visible pixel (empty block: inpainted, no feedback)

  __device__ __forceinline__ float Q(int oy, int ox) const {
    if (oy >= 0 && ox >= 0) return ox == 0 ? __low2float(own[oy]) : __high2float(own[oy]);
    return ext[oy + 1][ox + 1];
  }
  __device__ __forceinline__ bool Transparent(int sy, int sx) const { return !(Q(sy, sx) > 0.f); }
};

// Whether the window pixel (oy, ox) is read for the block: pixel -1 at the
// image's left / top edge replicates pixel 0, so it is folded into it.
template <Window kWin>
__device__ __forceinline__ bool WindowPixel(int oy, int ox, bool left_edge, bool top_edge) {
  if (oy < 0 && (!kHasRows<kWin> || top_edge)) return false;
  if (ox < 0 && (!kHasCols<kWin> || left_edge)) return false;
  return true;
}

// Alpha init, first part: targets as InitBlock, except that the block colour
// is the alpha-weighted target filter of linear light (blk, linear; the
// unweighted one for an empty block). The chroma target and the luma of
// transparent pixels follow in FillAlpha, after the 1-ring inpaint. b null:
// the colour only (a neighbour's, for the inpaint). a_own: the own pixels'
// alpha (for the alpha output), if not null.
template <TfKind kTf, Window kWin>
__device__ __forceinline__ void InitBlockAlpha(const InputDesc& in, const Geometry& geo,
                                               const ColorParams& cp, int bx, int by,
                                               BlockAlpha<kWin>* al, Rgb* blk,
                                               BlockInit* b = nullptr,
                                               float (*a_own)[2] = nullptr) {
  const bool left_edge = bx == 0, top_edge = by == 0;
  const TargetWeights<kWin> tw{cp.filter, left_edge, top_edge};
  Rgb acc{0.f, 0.f, 0.f}, acc_w{0.f, 0.f, 0.f};
  float sum = 0.f, q_own[2][2];
#pragma unroll
  for (int oy = -1; oy < 2; ++oy) {
#pragma unroll
    for (int ox = -1; ox < 2; ++ox) {
      if (!WindowPixel<kWin>(oy, ox, false, false)) continue;  // not in the window
      if (!WindowPixel<kWin>(oy, ox, left_edge, top_edge)) {
        al->ext[oy + 1][ox + 1] = 0.f;  // folded into pixel 0
        continue;
      }
      float a;
      const Rgb src = LoadPixelAlpha<kTf>(in, geo, cp, 2 * bx + ox, 2 * by + oy, &a);
      const Rgb lin = Transfer<kTf>::ToLinear(src, cp);
      tw.Add(&acc, lin, oy, ox);
      float q = tw.Weight(oy, ox) * a;
      if (oy >= 0 && ox >= 0) {
        q = __half2float(__float2half_rn(q));
        q_own[oy][ox] = q;
        if (a_own != nullptr) a_own[oy][ox] = a;
        if (b != nullptr) {
          b->wt[oy][ox] = Transfer<kTf>::EncodeLuminance(Dot(cp.kl, lin), cp);
          b->w[oy][ox] = Dot(cp.km, src);
        }
      } else {
        al->ext[oy + 1][ox + 1] = q;
      }
      acc_w = Madd(lin, q, acc_w);
      sum += q;
    }
  }
#pragma unroll
  for (int sy = 0; sy < 2; ++sy) al->own[sy] = __floats2half2_rn(q_own[sy][0], q_own[sy][1]);
  al->inv = sum > 0.f ? 1.f / sum : 0.f;
  *blk = sum > 0.f ? Scale(acc_w, al->inv) : tw.Finish(acc);
}

// 1-ring inpaint of an empty block: the mean colour of the non-empty blocks
// among its 3x3 neighbours inside the image (none: keeps its own colour).
// neighbour(nbx, nby, &c) returns whether block (nbx, nby) is non-empty and
// its colour.
template <class F>
__device__ __forceinline__ Rgb Inpaint(Rgb own, bool empty, int bx, int by, const Geometry& geo,
                                       F&& neighbour) {
  if (!empty) return own;
  Rgb sum{0.f, 0.f, 0.f};
  int n = 0;
#pragma unroll
  for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
    for (int dx = -1; dx <= 1; ++dx) {
      const int nx = bx + dx, ny = by + dy;
      if ((dx == 0 && dy == 0) || nx < 0 || ny < 0 || nx >= geo.bw || ny >= geo.bh) continue;
      Rgb c;
      if (neighbour(nx, ny, &c)) {
        sum = Add(sum, c);
        ++n;
      }
    }
  }
  return n > 0 ? Scale(sum, 1.f / float(n)) : own;
}

// Alpha init, second part. nb: the (inpainted) block colours, linear, of the
// 3x3 neighbourhood. Transparent pixels get target and start value from the
// upsampled block colours (smooth luma under alpha 0); the chroma target is
// the block's colour.
template <TfKind kTf, Window kWin>
__device__ __forceinline__ void FillAlpha(const ColorParams& cp, const Rgb3x3& nb,
                                          const BlockAlpha<kWin>& al, BlockInit* b) {
#pragma unroll
  for (int sy = 0; sy < 2; ++sy) {
#pragma unroll
    for (int sx = 0; sx < 2; ++sx) {
      if (!al.Transparent(sy, sx)) continue;
      const Rgb up = UpsampleColour<kWin>(nb, cp.filter, sy, sx);
      b->wt[sy][sx] = b->w[sy][sx] = Transfer<kTf>::EncodeLuminance(Dot(cp.kl, up), cp);
    }
  }
  b->dt = ChromaOffsets<kTf>(nb.d[1][1], cp);
}

// Reconstructed linear light of the pixel at offset (oy, ox) with W = wv.
template <TfKind kTf, Window kWin>
__device__ __forceinline__ Rgb ReconLinear(const ColorParams& cp, const Chroma3x3& nb, float wv,
                                           int oy, int ox) {
  return Transfer<kTf>::ToLinear(Recon(wv, Upsample<kWin>(nb, cp, oy, ox)), cp);
}

// One Jacobi step for a block. nb: chroma of the 3x3 neighbourhood (state k;
// kLocal reads only the centre); w: own W; nw: the neighbours' W the window
// reads (state k; ignored at the image's left / top edge, where pixel -1
// replicates pixel 0). kAlpha: al weights the chroma feedback; an empty block
// keeps its chroma (projected only), its luma still converges.
template <TfKind kTf, Window kWin, bool kAlpha = false>
__device__ __forceinline__ void IterateBlock(const ColorParams& cp, const Chroma3x3& nb,
                                             bool left_edge, bool top_edge, const float w[2][2],
                                             const NeighbourW& nw, const float wt[2][2],
                                             Chroma dt, float wn[2][2], Chroma* dn,
                                             const BlockAlpha<kWin>* al = nullptr) {
  const TargetWeights<kWin> tw{cp.filter, left_edge, top_edge};
  bool feedback = true;
  if constexpr (kAlpha) feedback = al->inv > 0.f;
  // Target / feedback filter: the table (or box) weights, or the alpha
  // weights q (normalized in Finish).
  Rgb acc{0.f, 0.f, 0.f};
  auto add = [&](Rgb lin, int oy, int ox) {
    if constexpr (kAlpha) {
      acc = Madd(lin, al->Q(oy, ox), acc);
    } else {
      tw.Add(&acc, lin, oy, ox);
    }
  };
#pragma unroll
  for (int sy = 0; sy < 2; ++sy) {
#pragma unroll
    for (int sx = 0; sx < 2; ++sx) {
      const Rgb lin = ReconLinear<kTf, kWin>(cp, nb, w[sy][sx], sy, sx);
      add(lin, sy, sx);
      const float wr = Transfer<kTf>::EncodeLuminance(Dot(cp.kl, lin), cp);
      wn[sy][sx] = Clamp01(w[sy][sx] + (wt[sy][sx] - wr));
    }
  }
  if constexpr (kHasCols<kWin>) {
    if (!left_edge && feedback) {
#pragma unroll
      for (int sy = 0; sy < 2; ++sy) {
        add(ReconLinear<kTf, kWin>(cp, nb, nw.left[sy], sy, -1), sy, -1);
      }
    }
  }
  if constexpr (kHasRows<kWin>) {
    if (!top_edge && feedback) {
#pragma unroll
      for (int sx = 0; sx < 2; ++sx) {
        add(ReconLinear<kTf, kWin>(cp, nb, nw.top[sx], -1, sx), -1, sx);
      }
      if constexpr (kHasCols<kWin>) {
        if (!left_edge) add(ReconLinear<kTf, kWin>(cp, nb, nw.corner, -1, -1), -1, -1);
      }
    }
  }
  const Chroma d0 = nb.d[1][1];
  if (!feedback) {
    *dn = ProjectChroma(d0, cp);
    return;
  }
  Rgb avg;
  if constexpr (kAlpha) {
    avg = Scale(acc, al->inv);
  } else {
    avg = tw.Finish(acc);
  }
  const Chroma dr = ChromaOffsets<kTf>(avg, cp);
  *dn = ProjectChroma({d0.r + (dt.r - dr.r), d0.b + (dt.b - dr.b)}, cp);
}

__device__ __forceinline__ void StoreSample(unsigned char* row, int index,
                                            float v, const ColorParams& cp,
                                            int shift) {
  const int maxv = (1 << cp.depth) - 1;
  const int q = min(max(__float2int_rn(v), 0), maxv);
  if (cp.depth == 8) {
    row[index] = static_cast<unsigned char>(q);
  } else {
    reinterpret_cast<uint16_t*>(row)[index] = static_cast<uint16_t>(q << shift);
  }
}

// Alpha output of a block's pixels inside the image: full range at the
// output depth. a[sy][sx]: alpha in [0, 1].
__device__ __forceinline__ void StoreAlphaBlock(const ColorParams& cp, const OutputDesc& out,
                                                const Geometry& geo, int bx, int by,
                                                const float a[2][2]) {
  const float maxv = float((1 << cp.depth) - 1);
#pragma unroll
  for (int sy = 0; sy < 2; ++sy) {
    const int y = 2 * by + sy;
    if (y >= geo.height) break;
    unsigned char* row = out.a + size_t(y) * out.a_pitch;
#pragma unroll
    for (int sx = 0; sx < 2; ++sx) {
      const int x = 2 * bx + sx;
      if (x >= geo.width) break;
      StoreSample(row, x, a[sy][sx] * maxv, cp, out.shift);
    }
  }
}

// Quantize and store a block's Y (pixels inside the image) and Cb/Cr.
__device__ __forceinline__ void StoreBlock(const ColorParams& cp, const OutputDesc& out,
                                           const Geometry& geo, int bx, int by,
                                           const float w[2][2], Chroma d) {
#pragma unroll
  for (int sy = 0; sy < 2; ++sy) {
    const int y = 2 * by + sy;
    if (y >= geo.height) break;
    unsigned char* row = out.y + size_t(y) * out.y_pitch;
#pragma unroll
    for (int sx = 0; sx < 2; ++sx) {
      const int x = 2 * bx + sx;
      if (x >= geo.width) break;
      StoreSample(row, x, fmaf(Clamp01(w[sy][sx]), cp.y_scale, cp.y_offset), cp, out.shift);
    }
  }
  const float cb = fmaf(d.b * cp.u_coef, cp.c_scale, cp.c_offset);
  const float cr = fmaf(d.r * cp.v_coef, cp.c_scale, cp.c_offset);
  const float first = out.swap_uv ? cr : cb;
  const float second = out.swap_uv ? cb : cr;
  if (out.biplanar) {
    unsigned char* row = out.u + size_t(by) * out.u_pitch;
    StoreSample(row, 2 * bx, first, cp, out.shift);
    StoreSample(row, 2 * bx + 1, second, cp, out.shift);
  } else {
    StoreSample(out.u + size_t(by) * out.u_pitch, bx, first, cp, out.shift);
    StoreSample(out.v + size_t(by) * out.v_pitch, bx, second, cp, out.shift);
  }
}

}  // namespace sharpyuv::detail
