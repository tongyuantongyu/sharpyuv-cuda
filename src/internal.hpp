// Internal types shared by host code (session.cpp) and kernels (kernel.cu).
// Plain data only; passed to kernels by value.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "sharpyuv_cuda/sharpyuv.h"

namespace sharpyuv::detail {

// kPq12: PQ with the cheaper decode, for <= 12-bit output (transfer.cuh).
enum class TfKind : int32_t { kParametric, kPq, kPq12, kHlg, kLinear };

// Kernel specialization by the filter's windows (compile-time in kernels):
//   kLocal  pure nearest: the upsampler reads only the block itself, so
//           blocks are independent (no chroma exchange, tiles exact at
//           halo 0); 2x2 target
//   kBox    2x2 target (only centred and nearest members): no pixels
//           besides the block's own are reconstructed
//   kCols   + pixel column -1 (a horizontally co-sited member)
//   kRows   + pixel row -1 (a vertically co-sited member)
//   kFull   both, and pixel (-1, -1)
enum class Window : int32_t { kBox = 0, kCols = 1, kRows = 2, kFull = 3, kLocal = 4 };
template <Window kWin>
inline constexpr bool kHasCols = (int32_t(kWin) & 3 & 1) != 0;
template <Window kWin>
inline constexpr bool kHasRows = (int32_t(kWin) & 3 & 2) != 0;

// Decoder filter (a weighted mixture of pure filters) as generic tables,
// weights relative to the block at (by, bx):
//   up[py][px][dy][dx]  chroma of pixel (2by+py, 2bx+px) = sum over the 3x3
//                       neighbour blocks (by+dy-1, bx+dx-1). Phase 1 never
//                       reads dy == 0 / dx == 0 (the blocks before).
//   tgt[oy][ox]         chroma target / feedback filter over the pixels
//                       (2by+oy-1, 2bx+ox-1); row / column 0 (pixel -1) is
//                       zero unless the window includes it.
// Reference: private/experiments/convergence/filter_coeffs.py.
struct FilterParams {
  float up[2][2][3][3];
  float tgt[3][3];
};

// Transfer function constants (see transfer.cuh).
struct TfParams {
  // Parametric: V = alpha L^(1/gamma) - (alpha - 1) for L >= beta, slope L below.
  float gamma, inv_gamma, alpha, inv_alpha, alpha_m1, beta, slope, inv_slope,
      v_beta;  // v_beta = slope * beta: threshold on the encoded side
  // HLG OOTF with system gamma g: F_D = Y_s^(g - 1) E_s (normalized, alpha = 1,
  // beta = 0). Exponents precomputed: ootf_dec_exp = g - 1 (to display light),
  // ootf_enc_exp = (1 - g) / g (back to scene light), ootf_inv_gamma = 1 / g.
  float ootf_dec_exp, ootf_enc_exp, ootf_inv_gamma;
};

// Constants of one syc_session_init: signal and decoder filter.
struct ColorParams {
  TfKind tf;
  Window window;
  FilterParams filter;
  TfParams tfp;
  float kl[3];  // linear-light luminance weights (from CP)
  float km[3];  // Y'CbCr luma weights (from MC)
  float u_coef;  // Cb = (B' - Y') * u_coef, u_coef = 1 / (2 (1 - kb))
  float v_coef;  // Cr = (R' - Y') * v_coef
  // G' - Y' = g_from_r (R' - Y') + g_from_b (B' - Y'): -kr / kg, -kb / kg.
  float g_from_r, g_from_b;
  // Quantization: Yq = W * y_scale + y_offset, Cq = C * c_scale + c_offset.
  float y_scale, y_offset, c_scale, c_offset;
  // Representable range of the chroma offsets (from the code range), used to
  // project d after each update (anti-windup): d_r in [dr_min, dr_max],
  // d_b in [db_min, db_max].
  float dr_min, dr_max, db_min, db_max;
  int32_t depth;  // 8, 10, 12, 16
};

enum class InType : int32_t { kU8, kU16, kF16, kF32 };

struct InputDesc {
  // Interleaved: ptr[0] (+ channel offsets). Planar: R, G, B, and A (alpha).
  const unsigned char* ptr[4];
  size_t pitch[4];
  InType type;
  int32_t planar;
  int32_t stride;   // interleaved: samples per pixel (3 or 4)
  int32_t off[4];   // interleaved: sample offset of R, G, B, A within a pixel
  float scale;      // value -> [0, 1] (before clamping); alpha too
  float offset;
  int32_t linear;   // 1: input is linear light, encode with the OETF
  int32_t alpha;    // 1: alpha processing (planar: ptr[3]; interleaved: off[3])
};

struct OutputDesc {
  unsigned char* y;
  unsigned char* u;   // biplanar: interleaved chroma plane
  unsigned char* v;   // biplanar: unused
  size_t y_pitch, u_pitch, v_pitch;
  int32_t biplanar;
  int32_t swap_uv;
  int32_t shift;      // MSB alignment: 16 - depth, else 0
  unsigned char* a;   // alpha output plane (full range, Y's depth); null: none
  size_t a_pitch;
};

struct Geometry {
  int32_t width, height;  // image
  int32_t bw, bh;         // blocks: ceil(width / 2), ceil(height / 2)
};

// ---------------------------------------------------------------------------
// The conversion kernel: overlapped tiles with a halo (kernel.cu). The
// multi-kernel path it was once tested against is in archive/multi_kernel.

// The halo is the tile overlap in blocks; halo >= iterations is exact. (The
// alpha init, which also reads neighbours, is computed exactly per tile from
// the input over a 2-block ring.)
inline constexpr int32_t kMaxHalo = 6;

// Internal overrides for benchmarks and tests (not public API).
struct KernelConfig {
  int32_t tile = 0;    // tile edge in blocks: 16 or 32; 0: chosen per image
  int32_t halo = -1;   // 0 .. kMaxHalo; < 0: from the session (syc_session::exact)
  // PQ decode: < 0 by output depth (cheaper fit for <= 12-bit); 0: always the
  // stable form; 1: always the fit.
  int32_t pq_fast = -1;
};

// Process-wide; read by syc_session_convert. Not thread-safe: set it before
// converting (benchmarks, tests).
KernelConfig& InternalKernelConfig();

// Validation shared by the factories and syc_session_init (color.cpp).
bool ValidWeights(double kr, double kb);
bool ValidCurve(const syc_parametric_curve& c);

// Normalized weights of a decoder filter mixture, [kind - 1][location - 1]
// (filter.cpp).
struct FilterMix {
  double w[3][3];
};
// False if an entry is invalid or no weight is > 0. Duplicates are merged.
bool BuildFilterMix(const syc_weighted_filter* list, size_t count, FilterMix* out);
// Tables and window of a mixture.
void BuildFilterParams(const FilterMix& mix, FilterParams* out, Window* window);

// Kernel launcher. Returns the launch error, if any. iterations: fixed number
// of refinement passes (>= 0); tile 0 picks the tile size for the image; halo
// 0 .. kMaxHalo (and < tile / 2). Alpha processing: in.alpha.
cudaError_t LaunchConvert(const ColorParams& color, const InputDesc& in,
                        const OutputDesc& out, const Geometry& geo, int32_t iterations,
                        int32_t tile, int32_t halo, cudaStream_t stream);

}  // namespace sharpyuv::detail
