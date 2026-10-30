// FilterInfo: validated, normalized mixture of pure decoder filters.

#include <cmath>

#include "internal.hpp"
#include "sharpyuv_cuda/sharpyuv.hpp"

namespace sharpyuv {
namespace {

// Slot of a pure filter: (kind, location) with both specified, else -1.
int Slot(DecoderFilter f) {
  const int k = static_cast<int>(f.kind), l = static_cast<int>(f.location);
  const int kinds = static_cast<int>(FilterKind::kBilinear);
  const int locations = static_cast<int>(ChromaLocation::kTopLeft);
  if (k < 1 || k > kinds || l < 1 || l > locations) return -1;
  return (k - 1) * locations + (l - 1);
}

}  // namespace

Status FilterInfo::FromFilter(DecoderFilter filter, FilterInfo* out) {
  const WeightedFilter one{1.0, filter};
  return FromWeighted(std::span<const WeightedFilter>(&one, 1), out);
}

Status FilterInfo::FromWeighted(std::span<const WeightedFilter> list,
                                FilterInfo* out) {
  if (out == nullptr) return Status::kInvalidArgument;
  FilterInfo f;
  double total = 0.0;
  for (const WeightedFilter& e : list) {
    const int slot = Slot(e.filter);
    if (slot < 0 || !std::isfinite(e.weight) || e.weight < 0.0) {
      return Status::kInvalidArgument;
    }
    f.weights_[slot] += e.weight;
    total += e.weight;
  }
  if (!(total > 0.0) || !std::isfinite(total)) return Status::kInvalidArgument;
  for (double& w : f.weights_) w /= total;
  *out = f;
  return Status::kOk;
}

bool FilterInfo::specified() const {
  for (double w : weights_) {
    if (w > 0.0) return true;
  }
  return false;
}

double FilterInfo::Weight(DecoderFilter filter) const {
  const int slot = Slot(filter);
  return slot < 0 ? 0.0 : weights_[slot];
}

namespace detail {
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
void Axes(FilterKind kind, ChromaLocation loc, Mode* v, Mode* h) {
  if (kind == FilterKind::kNearest) {
    *v = *h = kN;
    return;
  }
  const Mode lv = loc == ChromaLocation::kTopLeft ? kS : kC;
  *h = loc == ChromaLocation::kCenter ? kC : kS;
  *v = kind == FilterKind::kLinear ? kN : lv;
}

}  // namespace

void BuildFilterParams(const FilterInfo& f, FilterParams* out, Window* window) {
  double up[2][2][3][3] = {}, tgt[3][3] = {};
  for (FilterKind k : {FilterKind::kNearest, FilterKind::kLinear, FilterKind::kBilinear}) {
    for (ChromaLocation l :
         {ChromaLocation::kCenter, ChromaLocation::kLeft, ChromaLocation::kTopLeft}) {
      const double wgt = f.Weight({k, l});
      if (!(wgt > 0.0)) continue;
      Mode v, h;
      Axes(k, l, &v, &h);
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

}  // namespace detail

}  // namespace sharpyuv
