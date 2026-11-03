// The conversion kernel (docs/KERNEL.md): overlapped tiles. A CTA of
// T x T threads (one 2x2 block each) covers a T x T-block region, runs every
// iteration in registers + shared memory (the per-block steps of
// block.cuh) and writes its central (T - 2 halo)^2 blocks. Tiles are
// independent: no workspace, no inter-CTA synchronization, any image size (the
// grid has one CTA per tile). With halo >= iterations the result equals the
// global solve; a shorter halo replicates the region edge (<= 1 code at halo
// 3, private/experiments/convergence/halo_e1.py).

#include <cstdint>
#include <exception>
#include <stdexcept>

#include "block.cuh"

namespace sharpyuv::detail {
namespace {

template <auto First, auto... Rest, class F>
constexpr decltype(auto) __forceinline__ dispatch(decltype(First) key, F&& f) {
  static_assert((std::same_as<decltype(First), decltype(Rest)> && ...));
  decltype(std::forward<F>(f).template operator()<First>()) out {};

  auto try_dispatch = [&]<auto V, class Key>(Key k) -> bool {
    if (k != V) {
      return false;
    }

    out = std::forward<F>(f).template operator()<V>();
    return true;
  };

  bool matched =
      (try_dispatch.template operator()<First>(key) || (... || (try_dispatch.template operator()<Rest>(key))));

  if (!matched) {
    std::abort();
  }

  return out;
}

// Shared memory of a tile. The alpha init (tile plus a ring, computed before
// the first iteration) aliases the iteration buffers.
template <int kT, Window kWin, bool kAlpha>
struct TileSmem {
  static constexpr bool kLocal = kWin == Window::kLocal;
  static constexpr bool kExchangeW = kHasCols<kWin> || kHasRows<kWin>;
  static constexpr bool kAlphaInit = kAlpha && !kLocal;
  static constexpr int kR = kT + 4;  // block colours: the tile + 2 rings
  union {
    struct {
      // Chroma of every block (kLocal: blocks are independent, nothing shared).
      Chroma d[2][kLocal ? 1 : kT * kT];
      // Target windows beyond 2x2: W of each block's right column and bottom
      // row, [0] = (0, 1), [1] = (1, 1), [2] = (1, 0), read as the
      // neighbours' pixel column / row -1.
      float w[2][kExchangeW ? kT * kT : 1][3];
    } it;
    struct {
      // Block colours (linear); empty blocks inpainted in place.
      Rgb colour[kAlphaInit ? kR * kR : 1];
      unsigned char visible[kAlphaInit ? kR * kR : 1];  // block has a visible pixel
    } init;
  };
};

template <TfKind kTf, Window kWin, int kT, bool kAlpha>
__global__ void __launch_bounds__(kT* kT)
    ConvertKernel(ColorParams cp, InputDesc in, OutputDesc out, Geometry geo, int halo, int iterations) {
  using Smem = TileSmem<kT, kWin, kAlpha>;
  constexpr bool kLocal = Smem::kLocal;
  constexpr bool kExchangeW = Smem::kExchangeW;
  constexpr int kR = Smem::kR;
  __shared__ Smem sm;
  auto& s_d = sm.it.d;
  auto& s_w = sm.it.w;

  const int interior = kT - 2 * halo;
  const int ox = int(blockIdx.x) * interior - halo;
  const int oy = int(blockIdx.y) * interior - halo;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int bx = ox + tx, by = oy + ty;
  const bool inside = bx >= 0 && bx < geo.bw && by >= 0 && by < geo.bh;
  const int self = ty * kT + tx;
  const bool interior_block = inside && tx >= halo && tx < halo + interior && ty >= halo && ty < halo + interior;
  // Shared-memory slots of the 8 neighbours: a neighbour outside the image
  // (like the global solve) or outside the loaded region (the halo
  // approximation) is clamped to this block's row / column, i.e. offset 0.
  // Loop-invariant: slot = self + row_off[dy] + col_off[dx].
  const int row_off[3] = {ty > 0 && by > 0 ? -kT : 0, 0, ty < kT - 1 && by < geo.bh - 1 ? kT : 0};
  const int col_off[3] = {tx > 0 && bx > 0 ? -1 : 0, 0, tx < kT - 1 && bx < geo.bw - 1 ? 1 : 0};
  auto publish = [&](int k, const float w[2][2], Chroma d) {
    if constexpr (!kLocal) s_d[k][self] = d;
    if constexpr (kExchangeW) {
      s_w[k][self][0] = w[0][1];
      s_w[k][self][1] = w[1][1];
      s_w[k][self][2] = w[1][0];
    }
  };

  BlockInit b{};
  BlockAlpha<kWin> al{};
  if constexpr (!kAlpha) {
    if (inside) b = InitBlock<kTf, kWin>(in, geo, cp, bx, by);
  } else {
    Rgb blk{};
    if (inside) {
      float a[2][2];
      InitBlockAlpha<kTf, kWin>(in, geo, cp, bx, by, &al, &blk, &b, a);
      if (interior_block && out.a != nullptr) StoreAlphaBlock(cp, out, geo, bx, by, a);
    }
    if constexpr (kLocal) {
      // Block-local: no inpaint (empty blocks are invisible), the fill
      // upsamples the block's own colour.
      Rgb3x3 nb{};
      nb.d[1][1] = blk;
      if (inside) FillAlpha<kTf, kWin>(cp, nb, al, &b);
    } else {
      // The inpaint and the fill read neighbours up to 2 blocks out. Their
      // inputs depend on the source only, so when the tile is to be exact
      // (halo >= iterations), the ring outside it is computed here from the
      // input, and the halo only has to cover the iterations. Otherwise the
      // tile is an approximation anyway: the ring is skipped, and the
      // inpaint and fill see the tile only (edge blocks replicated, like the
      // iterations' halo approximation).
      const bool ring = halo >= iterations;
      auto index = [&](int x, int y) { return (y - oy + 2) * kR + (x - ox + 2); };
      bool empty = false;  // this thread stored an empty block that may need inpainting
      if (inside) {
        sm.init.colour[index(bx, by)] = blk;
        sm.init.visible[index(bx, by)] = al.inv > 0.f;
        empty = !(al.inv > 0.f);
      }
      constexpr int kRing = kR * kR - kT * kT;  // 2 rows / columns around the tile
      for (int k = self; k < kRing; k += kT * kT) {
        // Ring position k: the 2 rows above, the 2 below, then the 2 columns
        // left and right of the tile's rows.
        int rx, ry;
        if (k < 4 * kR) {
          ry = k / kR;
          ry = ry < 2 ? ry : ry + kT;
          rx = k % kR;
        } else {
          const int j = k - 4 * kR;
          ry = 2 + j / 4;
          rx = j % 4 < 2 ? j % 4 : kT + j % 4;
        }
        const int x = ox - 2 + rx, y = oy - 2 + ry;
        const int i = ry * kR + rx;
        if (!ring || x < 0 || y < 0 || x >= geo.bw || y >= geo.bh) {
          sm.init.visible[i] = 0;  // never inpainted from; colour never read
          continue;
        }
        BlockAlpha<kWin> ral{};
        InitBlockAlpha<kTf, kWin>(in, geo, cp, x, y, &ral, &sm.init.colour[i]);
        sm.init.visible[i] = ral.inv > 0.f;
        // The first ring is inpainted too (the fill of the tile's edge blocks
        // reads it).
        empty |= !(ral.inv > 0.f) && rx >= 1 && rx <= kT + 2 && ry >= 1 && ry <= kT + 2;
      }
      // Inpaint in place, only if the tile has an empty block (the common
      // fully opaque tile skips it): empty blocks take the mean of their
      // visible neighbours. Only visible blocks are read and only empty ones
      // written, so one buffer suffices.
      if (__syncthreads_or(empty)) {
        const int lo = ring ? -1 : 0, n = kT - 2 * lo;  // tile (+ first ring)
        for (int k = self; k < n * n; k += kT * kT) {
          const int x = ox + lo + k % n, y = oy + lo + k / n;
          if (x < 0 || y < 0 || x >= geo.bw || y >= geo.bh) continue;
          const int i = index(x, y);
          if (sm.init.visible[i]) continue;
          sm.init.colour[i] = Inpaint(sm.init.colour[i], true, x, y, geo, [&](int nx, int ny, Rgb* c) {
            const int j = index(nx, ny);
            *c = sm.init.colour[j];
            return sm.init.visible[j] != 0;
          });
        }
        __syncthreads();
      }
      if (inside) {
        bool transparent = false;
#pragma unroll
        for (int sy = 0; sy < 2; ++sy) {
#pragma unroll
          for (int sx = 0; sx < 2; ++sx) transparent |= al.Transparent(sy, sx);
        }
        if (transparent) {
          // Neighbours clamped to the image; without the ring also to the
          // tile.
          const int lo = ring ? -1 : 0, hi = ring ? kT : kT - 1;
          Rgb3x3 nb{};
#pragma unroll
          for (int dy = 0; dy < 3; ++dy) {
            const int y = min(max(min(max(by + dy - 1, 0), geo.bh - 1) - oy, lo), hi);
#pragma unroll
            for (int dx = 0; dx < 3; ++dx) {
              const int x = min(max(min(max(bx + dx - 1, 0), geo.bw - 1) - ox, lo), hi);
              nb.d[dy][dx] = sm.init.colour[(y + 2) * kR + (x + 2)];
            }
          }
          FillAlpha<kTf, kWin>(cp, nb, al, &b);
        } else {
          b.dt = ChromaOffsets<kTf>(blk, cp);  // fully visible: its own colour
        }
      }
      __syncthreads();  // the init buffers alias the iteration buffers
    }
  }

  float w[2][2], wt[2][2];
#pragma unroll
  for (int sy = 0; sy < 2; ++sy) {
#pragma unroll
    for (int sx = 0; sx < 2; ++sx) {
      w[sy][sx] = b.w[sy][sx];
      wt[sy][sx] = b.wt[sy][sx];
    }
  }
  const Chroma dt = b.dt;
  Chroma d = dt;
  if (inside) publish(0, w, d);
  if constexpr (!kLocal) __syncthreads();
  for (int n = 0; n < iterations; ++n) {
    const int cur = n & 1, nxt = cur ^ 1;
    if (inside) {
      Chroma3x3 nb{};
      if constexpr (kLocal) {
        nb.d[1][1] = d;
      } else {
#pragma unroll
        for (int dy = 0; dy < 3; ++dy) {
#pragma unroll
          for (int dx = 0; dx < 3; ++dx) nb.d[dy][dx] = s_d[cur][self + row_off[dy] + col_off[dx]];
        }
      }
      NeighbourW nw{};
      if constexpr (kHasCols<kWin>) {
        const int l = self + col_off[0];
        nw.left[0] = s_w[cur][l][0];
        nw.left[1] = s_w[cur][l][1];
      }
      if constexpr (kHasRows<kWin>) {
        const int a = self + row_off[0];
        nw.top[0] = s_w[cur][a][2];
        nw.top[1] = s_w[cur][a][1];
        if constexpr (kHasCols<kWin>) nw.corner = s_w[cur][self + row_off[0] + col_off[0]][1];
      }
      float wn[2][2];
      IterateBlock<kTf, kWin, kAlpha>(cp, nb, bx == 0, by == 0, w, nw, wt, dt, wn, &d, &al);
#pragma unroll
      for (int sy = 0; sy < 2; ++sy) {
#pragma unroll
        for (int sx = 0; sx < 2; ++sx) w[sy][sx] = wn[sy][sx];
      }
      publish(nxt, w, d);
    }
    if constexpr (!kLocal) __syncthreads();  // reads of `cur` done, writes of `nxt` visible
  }
  if (interior_block) StoreBlock(cp, out, geo, bx, by, w, d);
}

template <TfKind kTf, Window kWin, int kT, bool kAlpha>
cudaError_t LaunchConvertT(const ColorParams& cp, const InputDesc& in, const OutputDesc& out, const Geometry& geo,
                           int32_t iterations, int32_t halo, cudaStream_t stream) {
  const int interior = kT - 2 * halo;
  const dim3 block(kT, kT);
  const dim3 grid((geo.bw + interior - 1) / interior, (geo.bh + interior - 1) / interior);
  ConvertKernel<kTf, kWin, kT, kAlpha><<<grid, block, 0, stream>>>(cp, in, out, geo, halo, iterations);
  return cudaGetLastError();
}

int TileCount(const Geometry& geo, int tile, int halo) {
  const int interior = tile - 2 * halo;
  return ((geo.bw + interior - 1) / interior) * ((geo.bh + interior - 1) / interior);
}

// Per-SM cost model (private/benchmarks/RESULTS.md "Tile size"): tiles per SM x
// threads per tile x per-thread cost. 32-block tiles waste less on the halo,
// but their 1024-thread CTAs leave the SM less occupied (measured ~1.25x per
// thread at halo 0); 16-block tiles also fill small images better.
cudaError_t ChooseTile(const Geometry& geo, int halo, int32_t* tile) {
  int device = 0, sms = 0;
  cudaError_t e = cudaGetDevice(&device);
  if (e == cudaSuccess) e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (e != cudaSuccess) return e;
  auto per_sm = [&](int t) { return int64_t(TileCount(geo, t, halo) + sms - 1) / sms; };
  *tile = per_sm(16) * 16 * 16 * 4 < per_sm(32) * 32 * 32 * 5 ? 16 : 32;
  return cudaSuccess;
}

}  // namespace

cudaError_t LaunchConvert(const ColorParams& cp, const InputDesc& in, const OutputDesc& out, const Geometry& geo,
                          int32_t iterations, int32_t tile, int32_t halo, cudaStream_t stream) {
  if ((tile != 0 && tile != 16 && tile != 32) || halo < 0 || halo > kMaxHalo || 2 * halo >= (tile == 0 ? 16 : tile)) {
    return cudaErrorInvalidValue;
  }
  if (tile == 0) {
    cudaError_t e = ChooseTile(geo, halo, &tile);
    if (e != cudaSuccess) return e;
  }

  using enum TfKind;
  using enum Window;

  return dispatch<kParametric, kPq, kPq12, kHlg, kLinear>(cp.tf, [&]<auto kTf>() {
    return dispatch<kBox, kCols, kRows, kFull, kLocal>(cp.window, [&]<auto kWin>() {
      return dispatch<16, 32>(tile, [&]<int kT>() {
        return dispatch<true, false>(in.alpha, [&]<bool kAlpha>() {
          return LaunchConvertT<kTf, kWin, kT, kAlpha>(cp, in, out, geo, iterations, halo, stream);
        });
      });
    });
  });
}

}  // namespace sharpyuv::detail
