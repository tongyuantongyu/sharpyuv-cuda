// Decoder filter: the weighted mixture of pure filters, and its upsampler /
// target tables for the kernel.

#include <cmath>

#include "internal.hpp"

namespace sharpyuv::detail {
namespace {

// Per axis a pure filter is one of three 1-D modes (filter_coeffs.py):
// nearest (sample replicated to both pixels), bilinear centred (3/4, 1/4) or
// bilinear co-sited (even pixel = sample, odd = 1/2, 1/2).
enum Mode { kN, kC, kS };
// Upsampler [mode][phase][tap: block -1, 0, +1].
constexpr double kUp1[3][2][3] = {{{0, 1, 0}, {0, 1, 0}},
                                  {{0.25, 0.75, 0}, {0, 0.75, 0.25}},
                                  {{0, 1, 0}, {0, 0.5, 0.5}}};
// Target / feedback filter [mode][pixel offset -1, 0, +1].
constexpr double kTgt1[3][3] = {{0, 0.5, 0.5}, {0, 0.5, 0.5}, {0.25, 0.5, 0.25}};

// (vertical, horizontal) modes. Nearest ignores the location; linear is
// vertical nearest x horizontal bilinear; location: center (C, C), left
// (C, S), top-left (S, S).
void Axes(syc_filter_kind kind, syc_chroma_location loc, Mode* v, Mode* h) {
  if (kind == SYC_FILTER_NEAREST) {
    *v = *h = kN;
    return;
  }
  const Mode lv = loc == SYC_CHROMA_LOCATION_TOP_LEFT ? kS : kC;
  *h = loc == SYC_CHROMA_LOCATION_CENTER ? kC : kS;
  *v = kind == SYC_FILTER_LINEAR ? kN : lv;
}

bool ValidFilter(syc_decoder_filter f) {
  return f.kind >= SYC_FILTER_NEAREST && f.kind <= SYC_FILTER_BILINEAR &&
         f.location >= SYC_CHROMA_LOCATION_CENTER && f.location <= SYC_CHROMA_LOCATION_TOP_LEFT;
}

}  // namespace

bool BuildFilterMix(const syc_weighted_filter* list, size_t count, FilterMix* out) {
  FilterMix m{};
  double total = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const syc_weighted_filter& e = list[i];
    if (!ValidFilter(e.filter) || !std::isfinite(e.weight) || e.weight < 0.0) return false;
    m.w[e.filter.kind - 1][e.filter.location - 1] += e.weight;
    total += e.weight;
  }
  if (!(total > 0.0) || !std::isfinite(total)) return false;
  for (auto& row : m.w) {
    for (double& w : row) w /= total;
  }
  *out = m;
  return true;
}

void BuildFilterParams(const FilterMix& mix, FilterParams* out, Window* window) {
  double up[2][2][3][3] = {}, tgt[3][3] = {};
  for (int k = 0; k < 3; ++k) {
    for (int l = 0; l < 3; ++l) {
      const double wgt = mix.w[k][l];
      if (!(wgt > 0.0)) continue;
      Mode v, h;
      Axes(syc_filter_kind(k + 1), syc_chroma_location(l + 1), &v, &h);
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          tgt[a][b] += wgt * kTgt1[v][a] * kTgt1[h][b];
          for (int py = 0; py < 2; ++py) {
            for (int px = 0; px < 2; ++px) up[py][px][a][b] += wgt * kUp1[v][py][a] * kUp1[h][px][b];
          }
        }
      }
    }
  }
  bool local = true, cols = false, rows = false;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      out->tgt[a][b] = float(tgt[a][b]);
      cols |= b == 0 && tgt[a][b] != 0.0;
      rows |= a == 0 && tgt[a][b] != 0.0;
      for (int py = 0; py < 2; ++py) {
        for (int px = 0; px < 2; ++px) {
          out->up[py][px][a][b] = float(up[py][px][a][b]);
          local &= (a == 1 && b == 1) || up[py][px][a][b] == 0.0;
        }
      }
    }
  }
  *window = local && !cols && !rows
                ? Window::kLocal
                : Window((cols ? int32_t(Window::kCols) : 0) | (rows ? int32_t(Window::kRows) : 0));
}

}  // namespace sharpyuv::detail
