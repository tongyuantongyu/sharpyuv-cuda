# CUDA Kernel Architecture

This document describes the GPU execution model, memory hierarchy design, register and shared-memory optimizations, and device-side transfer function implementations in `src/kernel.cu`, `src/block.cuh`, and `src/transfer.cuh`. For the mathematical formulation of the algorithm, see [Algorithm & Color Science](ALGORITHM.md).

---

## 1. Why a Single Fused Overlapped-Tile Kernel

A straightforward GPU port of Jacobi refinement launches a sequence of global passes: `InitKernel`, $N \times \text{IterateKernel}$, and `FinalKernel`, storing the intermediate state ($W, W_t$ per pixel and $d, d_t$ per $2 \times 2$ block) in a global-memory workspace. That approach has three structural bottlenecks:
1. **DRAM bandwidth per iteration:** Every iteration reads and writes the full-frame state in global memory ($\sim 220\ \mu\text{s}$ per iteration at 4K on an RTX 5080).
2. **Large workspace footprint:** Holding `float32` targets and ping-pong state requires $56\text{ B}$ per $2 \times 2$ block ($14\text{ B/pixel}$: $\sim 116\text{ MB}$ at 4K, $\sim 465\text{ MB}$ at 8K).
3. **Launch overhead on small images:** Launching 8 separate kernels for a 6-iteration solve adds $\sim 90\ \mu\text{s}$ of host/driver overhead on small $256 \times 256$ images where the GPU work itself takes $< 15\ \mu\text{s}$.

### Overlapped Tiles in On-Chip Memory

Because each Jacobi iteration only exchanges information with immediate $3 \times 3$ neighboring blocks, the domain of dependence expands by **at most 1 chroma block per iteration**.

`ConvertKernel` (`src/kernel.cu`) exploits this locality by partitioning the frame into **independent overlapped tiles**:
- Each Cooperative Thread Array (CTA) of $T \times T$ threads (one thread per $2 \times 2$ luma block, $T \in \{16, 32\}$) loads a $T \times T$-block region that includes an **$H$-block halo** around an interior of $(T - 2H) \times (T - 2H)$ blocks.
- The CTA computes its targets in registers, runs all $N$ Jacobi iterations entirely in registers and shared memory, and quantizes/stores only the central $(T - 2H) \times (T - 2H)$ interior blocks to global memory.
- **Zero workspace, zero inter-CTA synchronization:** No tile ever reads another tile's output. CTAs do not need to be co-resident on the GPU at the same time; the hardware scheduler executes the grid in as many waves as needed for any image up to $65536 \times 65536$.
- Global memory traffic is reduced to **one source read** (with adjacent tiles hitting L2 cache on the overlapped halo) and **one destination write**.

---

## 2. Tile Geometry, Halo Width & Adaptive Tile Sizing

### Halo Width ($H$) and Exactness

For a tile of $T \times T$ blocks with halo $H$, boundary threads inside the CTA clamp neighbor lookups at the outer edge of the $T \times T$ tile, just as blocks at the outer edge of the image do:
- **Exact mode (`s->exact = true`):** Sets $H = N$ (up to `kMaxHalo = 6`). Because the halo is at least as wide as the number of iterations (and alpha initialization loads its 2-block outer ring from source RGB), the boundary approximation at the outer edge of the halo never reaches the $(T - 2H) \times (T - 2H)$ interior. The output is **bit-identical** to a whole-image global Jacobi solve.
- **Default mode (`s->exact = false`):** Caps the halo at $H = \min(N, 3)$.
  - For $N \le 3$ (including `SYC_FILTER_NEAREST`, where $H = 0$), default mode is already exact.
  - For $N = 6$, each additional halo block reduces tile-seam discrepancies by roughly $8\times$. At $H = 3$, $> 99.9\%$ of output codes across SDR and HDR test sets are identical to the exact whole-image solve, and the remaining $< 0.1\%$ of codes near tile seams differ by at most **1 code** (the same tolerance as `float32` vs. `float64` rounding).
  - For $T = 32$, capping $H$ at 3 instead of 6 increases the interior from $20 \times 20$ ($39\%$ useful work) to $26 \times 26$ ($66\%$ useful work), making 4K and 8K conversions $\sim 1.6\times$ faster.

| Halo $H$ (with $T = 32$) | Interior Blocks | Interior Pixels | Useful Work $(T - 2H)^2 / T^2$ | Max Seam Diff ($N = 6$, Real Crops) |
|---:|---|---|---:|---|
| `0` (`kLocal` / nearest) | $32 \times 32$ | $64 \times 64$ | $100\%$ | $0$ (exact for nearest) |
| `3` (default at $N = 6$) | $26 \times 26$ | $52 \times 52$ | $66\%$ | $\le 1$ code on $< 0.1\%$ of samples |
| `6` (`exact = true` at $N = 6$) | $20 \times 20$ | $40 \times 40$ | $39\%$ | $0$ (exact) |

### Adaptive Tile Size Selection (`ChooseTile`)

`ConvertKernel` is instantiated for two tile dimensions: $T = 16$ ($256$ threads/CTA) and $T = 32$ ($1024$ threads/CTA).
- **$T = 32$ ($1024$ threads):** Amortizes a $2H$-block halo over a much larger interior ($66\%$ useful work at $H = 3$, vs. $39\%$ for $T = 16$). Preferred for medium-to-large images when $H \ge 2$.
- **$T = 16$ ($256$ threads):** Allows multiple CTAs to reside simultaneously on each SM ($1536$ vs. $1024$ resident threads, $\sim 1.25\times$ higher per-thread throughput at $H = 0$), and avoids leaving SMs idle on smaller images (such as $640 \times 480$ or $256 \times 256$) where a $32 \times 32$-block grid would launch too few CTAs to fill the GPU.

Before launching, `ChooseTile` (`src/kernel.cu`) queries the device SM count (`cudaDevAttrMultiProcessorCount`) and evaluates a per-SM wave cost model:
$$\text{Cost}(T) = \left\lceil \frac{\text{TileCount}(W, H_{\text{img}}, T, H)}{\text{SMs}} \right\rceil \times T^2 \times w_T, \qquad (w_{16} = 4,\; w_{32} = 5)$$
picking $T = 16$ when $\text{Cost}(16) < \text{Cost}(32)$ and $T = 32$ otherwise. Across 70 benchmarked `(resolution, format, halo)` states, this heuristic matches the faster fixed tile size within $5\%$ in 69 of 70 cases.

---

## 3. Window Specializations & Shared Memory Layout

### Compile-Time `Window` Specializations

In `syc_session_init()`, `BuildFilterParams` (`src/filter.cpp`) inspects which neighbor blocks and pixel offsets have non-zero weights in the active decoder filter (or weighted mixture) and selects one of five compile-time `detail::Window` specializations (`src/internal.hpp`):

| `Window` | Active Filter Class | Reconstructed Pixels / Block | Shared Memory Exchanges per Iteration |
|---|---|---|---|
| `Window::kLocal` | Pure `SYC_FILTER_NEAREST` | $4$ (own $2 \times 2$) | **None** (100% register-local, no `__syncthreads()`, $H = 0$) |
| `Window::kBox` | Centered bilinear, nearest, or mixtures of the two | $4$ (own $2 \times 2$) | Chroma `d` ($8\text{ B}$/block) |
| `Window::kCols` | Horizontally co-sited (`LEFT`) or mixtures with `LEFT` | $6$ (own $2 \times 2$ + pixel column $-1$) | Chroma `d` + right-column $W$ |
| `Window::kRows` | Vertically co-sited only | $6$ (own $2 \times 2$ + pixel row $-1$) | Chroma `d` + bottom-row $W$ |
| `Window::kFull` | Co-sited both axes (`TOP_LEFT`) or general mixtures | $9$ (own $2 \times 2$ + row $-1$, col $-1$, corner $(-1, -1)$) | Chroma `d` + right-column & bottom-row $W$ |

Why `kCols`, `kRows`, and `kFull` exchange boundary $W$:
When a filter includes a co-sited component, its $3 \times 3$ target filter `tgt[oy][ox]` has non-zero weights on pixel offset $o_x = -1$ (the rightmost column of the neighbor block to the left) and/or $o_y = -1$ (the bottom row of the neighbor block above). Rather than storing reconstructed linear RGB in shared memory ($36\text{ B}$/block), each block publishes up to 3 floats of its own updated luma $W$ (`w[0][1]`, `w[1][1]`, `w[1][0]`, $12\text{ B}$/block) into shared memory alongside its 2-float chroma `Chroma{r, b}` ($8\text{ B}$/block), and reconstructs the boundary pixels on the fly.

### Shared Memory Union (`TileSmem`)

`TileSmem<kT, kWin, kAlpha>` (`src/kernel.cu`) overlays the alpha initialization buffers and the Jacobi iteration ping-pong buffers in a single `union`:

```cpp
template <int kT, Window kWin, bool kAlpha>
struct TileSmem {
  static constexpr bool kLocal = kWin == Window::kLocal;
  static constexpr bool kExchangeW = kHasCols<kWin> || kHasRows<kWin>;
  static constexpr bool kAlphaInit = kAlpha && !kLocal;
  static constexpr int kR = kT + 4;  // tile + 2-block outer ring for alpha inpaint
  union {
    struct {
      Chroma d[2][kLocal ? 1 : kT * kT];
      float w[2][kExchangeW ? kT * kT : 1][3];
    } it;
    struct {
      Rgb colour[kAlphaInit ? kR * kR : 1];
      unsigned char visible[kAlphaInit ? kR * kR : 1];
    } init;
  };
};
```

- For $T = 32$, `it.d` uses $2 \times 1024 \times 8\text{ B} = 16\text{ KB}$, and `it.w` (when `kExchangeW` is true) adds $2 \times 1024 \times 12\text{ B} = 24\text{ KB}$ ($40\text{ KB}$ total, well within standard per-block shared memory without requiring dynamic opt-in).
- When `kAlpha` is enabled, `init` uses $(32 + 4)^2 \times 13\text{ B} \approx 16.8\text{ KB}$ before the first iteration, and aliases `it` once `__syncthreads()` completes at the end of initialization.
- Because `it.d` and `it.w` are double-buffered (`cur = n & 1`, `nxt = cur ^ 1`), each Jacobi iteration requires only **one `__syncthreads()` barrier** at the end of the loop body instead of two.

---

## 4. Register Footprint & Per-Iteration Execution Flow

A $32 \times 32$-thread CTA uses $1024$ threads — the architectural maximum per CTA — which limits each thread to **64 registers** ($65536 / 1024$) before the compiler spills to local memory. Several deliberate design choices in `src/block.cuh` and `src/kernel.cu` keep register usage at $\sim 55\text{ registers/thread}$ with zero local-memory spills across all 100 kernel instantiations:

1. **2-Float Chroma Representation (`struct alignas(8) Chroma { float r, b; }`):**
   As shown in [Algorithm & Color Science §3.1](ALGORITHM.md#31-separating-linear-luminance-cp-from-gamma-luma-mc), $k_M \cdot d = 0$ allows $d_g = G' - Y'$ to be derived via a single FMA (`DeriveG`) whenever a pixel is reconstructed. Storing only $(d_r, d_b)$ saves 11 registers per thread (target $d_t$, current $d$, and the $3 \times 3$ neighborhood `Chroma3x3`), turns 27 scalar shared-memory loads into nine 8-byte vector loads (`LDS.64`), and removes one third of the upsampling FMAs.
2. **Hoisted Loop-Invariant Neighbor Addressing:**
   Each thread's clamped 1-D shared-memory offsets to its vertical and horizontal neighbors (`row_off[3]` and `col_off[3]`, where out-of-image or out-of-tile neighbors clamp to offset `0`) are computed once before the iteration loop. Inside the loop, neighbor $(d_y, d_x)$ is read at `self + row_off[dy] + col_off[dx]`.
3. **Compile-Time 4-Tap Upsampling & Streaming Target Accumulation:**
   - In `UpsampleTaps`, pixel phase $0$ never reads neighbor block $+1$ and phase $1$ never reads block $-1$. After `#pragma unroll`, zero taps are eliminated at compile time, reducing every pixel's chroma upsample from 9 taps to at most **4 taps** ($2 \times 2$ blocks).
   - In `IterateBlock`, reconstructed linear pixels are accumulated directly into a single running `Rgb acc` via `TargetWeights` (reading edge-adjusted weights from constant memory) rather than buffering an array of up to 9 reconstructed `Rgb` pixels (which would cost 27 registers).
4. **Packed `half2` Alpha Weights (`BlockAlpha`):**
   Storing a block's 4 own pixel weights $q(s_y, s_x) = w_{\text{tgt}}(s_y, s_x) \cdot \alpha(s_y, s_x)$ as `float` pushed 1024-thread alpha kernels over the 64-register limit. `BlockAlpha` rounds the 4 own weights to IEEE `half` and packs them into two `__half2` registers (`own[2]`), computing the normalization factor $\text{inv} = 1 / \sum q$ over the rounded values so the weights still sum to $1$ exactly and $q > 0 \iff \alpha > 0$.
5. **Work-Avoiding Alpha Initialization:**
   - **Single 4-channel load (`LoadChannels<4>`):** Loads R, G, B, and A with a single address calculation and DRAM sector request.
   - **Conditional in-place inpainting:** Each thread sets `empty = !(al.inv > 0.f)`. The 1-ring inpaint stencil runs **only if `__syncthreads_or(empty)` is non-zero**; fully opaque tiles skip the inpaint pass and barrier entirely.
   - **Conditional transparent luma fill:** `FillAlpha` (which upsamples the $3 \times 3$ linear block colors to fill $W_t$ and $W^{(0)}$ under $\alpha = 0$) runs only for blocks that contain at least one $\alpha = 0$ pixel.
   - **Exact-only outer ring:** The 2-block outer ring (`kR = kT + 4`) around the tile is loaded and inpainted only when `halo >= iterations` (exact mode); in default mode, inpainting clamps to the $T \times T$ tile boundary just like the Jacobi iterations.

---

## 5. Device Transfer Functions & Template Dispatch

For the mathematical derivations and accuracy tiers of the transfer functions, see [Algorithm & Color Science — Transfer Function Approximation Details](ALGORITHM.md#4-transfer-function-approximation-details).

### Device-Side Optimizations (`src/transfer.cuh`)

Profiling the fused tile kernel with Nsight Compute shows that it is instruction/issue-bound, with 64–76% of executed instructions spent inside transfer-function evaluations. Three device-level optimizations keep this path fast and divergence-free:

1. **Branchless Piecewise Selection:**
   Standard SDR and HLG curves have a linear/quadratic/sqrt toe joined to a power/exp/log segment. Writing `v < threshold ? toe : pow(...)` causes `nvcc` to emit a conditional branch around the MUFU instructions, costing $1.6\text{--}1.8\times$ when threads in a warp straddle the threshold. `ParamToLinear`, `ParamToEncoded`, `hlg::InvOetf`, and `hlg::Oetf` evaluate both expressions unconditionally and select the result with a predicated select (`SELP`), keeping all 32 warp lanes converged.
2. **Flush-to-Zero `rsqrt.approx.ftz.f32` in HLG OETF:**
   The HLG OETF evaluates $\sqrt{3E}$ for $E \le 1/12$. Calling CUDA's standard `rsqrtf(s)` inserts a software denormal-handling branch at every invocation (creating 52 divergence regions in the unrolled kernel). Because the argument is clamped to $s = 3 \max(E, 10^{-30})$ and is never denormal, `hlg::RsqrtFtz` invokes the hardware MUFU instruction directly via inline PTX:
   ```cpp
   asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(s));
   ```
   eliminating all warp divergence and reducing 4K HLG kernel time by ~10%.
3. **Dual PQ Decode Specialization (`kPq12` vs. `kPq`):**
   As shown in the error plot in [Algorithm & Color Science §4.2](ALGORITHM.md#42-smpte-st-2084-pq-numerical-conditioning--approximations), `syc_session_init()` automatically selects `TfKind::kPq12` (`pq::ToLinear12`, rational fit $R_{3,2}(\log_2 x)^8$, 2 MUFU ops) when `yuv_bit_depth <= 12`, and `TfKind::kPq` (`pq::ToLinear`, numerically stable `expm1` form, 4 MUFU ops) when `yuv_bit_depth == 16`. Both share the degree-8 log-log polynomial encode `pq::ToEncoded`.

### Compile-Time vs. Runtime Dispatch (`LaunchConvert`)

`LaunchConvert` (`src/kernel.cu`) dispatches across four compile-time template axes (`5 × 5 × 2 × 2 = 100` instantiations):
1. **`TfKind` (`5`):** `kParametric`, `kPq`, `kPq12`, `kHlg`, `kLinear`.
2. **`Window` (`5`):** `kBox`, `kCols`, `kRows`, `kFull`, `kLocal`.
3. **Tile edge `kT` (`2`):** `16` or `32` blocks.
4. **Alpha mode `kAlpha` (`2`):** `true` or `false`.

All I/O format parameters — input sample type (`U8`, `U16`, `F16`, `F32`), planar vs. interleaved layout, channel byte order (`RGB`, `BGR`, `RGBA`, `ARGB`, etc.), linear-input flag, output bit depth (`8`, `10`, `12`, `16`), planar vs. biplanar (`I420`/`YV12` vs. `NV12`/`P010`), `swap_uv`, and LSB/MSB bit shift — are handled via runtime descriptors (`InputDesc`, `OutputDesc`) in global load/store helpers that execute only **once per block** at the start and end of the kernel, avoiding combinatorial template bloat while staying within ~5% of fully templated I/O.

---

## 6. Performance Characteristics & Benchmarks

Measured on an NVIDIA GeForce RTX 5080 (`sm_120`, 84 SMs, Release build).
- **Cold ($\mu\text{s}$):** L2 cache flushed before every measurement so all input and output traffic comes from DRAM (representative of single-frame conversions in a larger pipeline).
- **Batch ($\mu\text{s}$):** Back-to-back launches with warm caches.
- Default iteration counts: `6` for `center`, `left`, `topleft`, `windows` (linear-left), and mixtures (`c+nn`, `c+tl+win+nn`); `3` for `nearest`.

### Default Mode (`exact = false`), Cold / Batch ($\mu\text{s}$)

| Format & Alpha | Filter | $7680 \times 4320$ (8K) | $3840 \times 2160$ (4K) | $1920 \times 1080$ | $640 \times 480$ | $256 \times 256$ |
|---|---|---:|---:|---:|---:|---:|
| **8-bit sRGB** (`U8` RGB $\to$ `I420`), Alpha Off | `center` | 1585 / 1624 | 408 / 409 | 116 / 109 | 27.7 / 24.8 | 12.3 / 11.8 |
| | `left` / `windows` | 1981 / 1974 | 506 / 486 | 138 / 131 | 30.9 / 28.8 | 14.5 / 12.6 |
| | `topleft` | 2608 / 2603 | 662 / 655 | 179 / 174 | 39.1 / 48.0 | 16.6 / 16.6 |
| | `nearest` | 477 / 464 | 123 / 119 | 33.0 / 32.9 | 8.5 / 8.9 | 6.3 / 6.6 |
| | `c+nn` (blend) | 1473 / 1468 | 377 / 369 | 103 / 99.6 | 23.4 / 22.6 | 10.5 / 11.5 |
| | `c+tl+win+nn` (blend) | 2606 / 2605 | 662 / 655 | 178 / 174 | 39.1 / 37.0 | 16.6 / 16.5 |
| **8-bit sRGB** (`U8` RGBA $\to$ `I420` + A), Alpha On | `center` | 1621 / 1618 | 416 / 402 | 114 / 108 | 25.7 / 24.7 | 12.5 / 11.1 |
| | `left` | 2284 / 2281 | 583 / 573 | 160 / 154 | 35.1 / 32.9 | 16.3 / 14.6 |
| | `topleft` | 3122 / 3121 | 795 / 782 | 215 / 207 | 47.5 / 45.1 | 23.0 / 22.6 |
| | `nearest` | 548 / 529 | 142 / 135 | 38.0 / 37.0 | 10.4 / 8.8 | 6.3 / 6.8 |
| **10-bit PQ** (`U16` Planar $\to$ `P010`), Alpha Off | `center` | 1914 / 1907 | 494 / 480 | 136 / 127 | 30.9 / 28.7 | 13.2 / 12.5 |
| | `left` | 2461 / 2463 | 632 / 619 | 173 / 164 | 37.1 / 34.9 | 16.6 / 16.4 |
| | `topleft` | 3384 / 3395 | 863 / 853 | 233 / 224 | 49.4 / 47.2 | 20.7 / 20.5 |
| | `nearest` | 645 / 625 | 171 / 158 | 45.6 / 43.0 | 11.1 / 10.6 | 6.4 / 7.4 |
| **10-bit HLG** (`U16` Planar $\to$ `P010`), Alpha Off | `center` | 1940 / 1936 | 499 / 488 | 137 / 128 | 30.9 / 28.7 | 13.8 / 12.5 |
| | `left` | 2428 / 2426 | 620 / 610 | 169 / 160 | 37.1 / 34.9 | 16.7 / 16.4 |
| | `topleft` | 3225 / 3224 | 820 / 809 | 220 / 213 | 47.4 / 45.2 | 20.7 / 20.5 |
| | `nearest` | 628 / 610 | 166 / 156 | 45.3 / 42.7 | 11.4 / 11.0 | 6.4 / 7.8 |

### Key Takeaways

- **Target window governs filter cost:** `center` (`kBox`) reconstructs 4 pixels per block-iteration; `left` and `windows` (`kCols`) reconstruct 6 (~1.25× `center`); `topleft` and full mixtures (`kFull`) reconstruct 9 (~1.6–1.7× `center`). Any weighted filter mixture runs at the speed of its widest member window.
- **Pure nearest (`kLocal`):** Because it runs 3 iterations with $H = 0$ and no shared-memory exchange, it completes in ~3.5× less time than `center` (e.g. $123\ \mu\text{s}$ at 4K U8).
- **Alpha optimization + output alpha plane overhead:** On `center`, enabling alpha processing and writing the de-interleaved alpha plane adds only $\sim 2\%$ on 8-bit RGBA and $8\text{--}11\%$ on 10-bit planar inputs.
- **Exact mode (`exact = true`, $H = 6$):** Costs $1.55\text{--}1.67\times$ default mode on `center` at $\ge 1080\text{p}$ (e.g., $672\ \mu\text{s}$ vs. $408\ \mu\text{s}$ at 4K U8 `center`), and has zero overhead on `nearest` ($H = 0$).
