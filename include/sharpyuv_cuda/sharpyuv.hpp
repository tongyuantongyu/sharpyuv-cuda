// sharpyuv-cuda: C++ interface, header-only over the C API (sharpyuv.h).
//
// Usage:
//   sharpyuv::Session s;                                 defaults: JPEG signal
//   s.SetStream(stream)
//       .SetColor(sharpyuv::ColorPrimaries::FromH273(9),
//                 sharpyuv::TransferFunction::FromH273(16),
//                 sharpyuv::MatrixCoefficients::FromH273(9), SYC_RANGE_LIMITED, 10)
//       .SetFilters({SYC_FILTER_BILINEAR, SYC_CHROMA_LOCATION_LEFT})
//       .Init();
//   syc_status st = s.Convert(src, dst);                 repeatable, asynchronous
//
// Errors: the constructor, factories and Init() throw (std::bad_alloc,
// sharpyuv::Error); Convert() returns a syc_status.

#pragma once

#include <initializer_list>
#include <new>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sharpyuv.h"

inline std::ostream& operator<<(std::ostream& os, const syc_status status) {
  return os << std::string_view(syc_status_string(status));
}

namespace sharpyuv {

// Thrown on configuration errors.
// Don't swallow, your code is buggy if you see one thrown.
class Error : public std::runtime_error {
 public:
  explicit Error(syc_status status, const char* what)
      : std::runtime_error(std::string(what) + ": " + syc_status_string(status)),
        status_(status) {}
  [[nodiscard]] syc_status status() const { return status_; }

 private:
  syc_status status_;
};

namespace internal {
inline void Check(syc_status status, const char* what) {
  if (status != SYC_STATUS_OK) throw Error(status, what);
}
}  // namespace internal

// Colour primaries; see syc_color_primaries.
class ColorPrimaries : public syc_color_primaries {
 public:
  [[nodiscard]] static ColorPrimaries FromH273(int32_t code) {
    ColorPrimaries p;
    internal::Check(syc_primaries_from_h273(code, &p), "ColorPrimaries::FromH273");
    return p;
  }
  // Red, green, blue and white point.
  [[nodiscard]] static ColorPrimaries FromChromaticities(syc_chromaticity r, syc_chromaticity g,
                                                         syc_chromaticity b, syc_chromaticity w) {
    ColorPrimaries p;
    internal::Check(syc_primaries_from_chromaticities(r, g, b, w, &p),
                    "ColorPrimaries::FromChromaticities");
    return p;
  }
  [[nodiscard]] static ColorPrimaries FromLumaCoefficients(double kr, double kb) {
    ColorPrimaries p;
    internal::Check(syc_primaries_from_luma_coefficients(kr, kb, &p),
                    "ColorPrimaries::FromLumaCoefficients");
    return p;
  }

 private:
  ColorPrimaries() : syc_color_primaries{} {}
};

// Transfer function; see syc_transfer_function.
class TransferFunction : public syc_transfer_function {
 public:
  [[nodiscard]] static TransferFunction FromH273(int32_t code) {
    TransferFunction t;
    internal::Check(syc_transfer_from_h273(code, &t), "TransferFunction::FromH273");
    return t;
  }
  [[nodiscard]] static TransferFunction FromParametric(const syc_parametric_curve& curve) {
    TransferFunction t;
    internal::Check(syc_transfer_from_parametric(&curve, &t), "TransferFunction::FromParametric");
    return t;
  }

 private:
  TransferFunction() : syc_transfer_function{} {}
};

// Matrix coefficients; see syc_matrix_coefficients.
class MatrixCoefficients : public syc_matrix_coefficients {
 public:
  [[nodiscard]] static MatrixCoefficients FromH273(int32_t code) {
    MatrixCoefficients m;
    internal::Check(syc_matrix_from_h273(code, &m), "MatrixCoefficients::FromH273");
    return m;
  }
  [[nodiscard]] static MatrixCoefficients FromCoefficients(double kr, double kb) {
    MatrixCoefficients m;
    internal::Check(syc_matrix_from_coefficients(kr, kb, &m),
                    "MatrixCoefficients::FromCoefficients");
    return m;
  }

 private:
  MatrixCoefficients() : syc_matrix_coefficients{} {}
};

// RGB input; see syc_rgb_image.
struct RgbImage : syc_rgb_image {
  RgbImage() : syc_rgb_image{} {
    u16_bits = 16;
    float_min = 0.0f;
    float_max = 1.0f;
  }
};

// YUV output, see syc_yuv_image.
struct YuvImage : syc_yuv_image {
  YuvImage() : syc_yuv_image{} {}
};

// Conversion settings bound to one CUDA stream (see syc_session for the
// settings and their defaults). Call the setters, then Init() once; the
// setters must not be called after Init(). For other settings, create
// another session.
//
// Move-only. Destroying a session waits for its conversions in flight. A
// moved-from session must not be used except to assign to or destroy it.
class Session {
 public:
  // Throws std::bad_alloc if out of memory.
  Session() : session_(syc_session_create()) {
    if (session_ == nullptr) throw std::bad_alloc();
  }
  ~Session() { syc_session_destroy(session_); }

  Session(Session&& other) noexcept
      : session_(std::exchange(other.session_, nullptr)), filters_(std::move(other.filters_)) {}
  Session& operator=(Session&& other) noexcept {
    if (this != &other) {
      syc_session_destroy(session_);
      session_ = std::exchange(other.session_, nullptr);
      filters_ = std::move(other.filters_);
    }
    return *this;
  }
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  // Conversions run on this stream and its device; it must outlive the
  // session. Default: null, the default stream.
  Session& SetStream(cudaStream_t stream) {
    session_->stream = stream;
    return *this;
  }

  // Signal description. Default: JPEG (sRGB primaries and transfer function,
  // BT.601 matrix, full range, 8-bit).
  Session& SetColor(const ColorPrimaries& cp, const TransferFunction& tf,
                    const MatrixCoefficients& mc, syc_range range, int32_t yuv_bit_depth) {
    session_->primaries = cp;
    session_->transfer = tf;
    session_->matrix = mc;
    session_->range = range;
    session_->yuv_bit_depth = yuv_bit_depth;
    return *this;
  }

  // The decoder to optimize for. Default: bilinear, centre.
  Session& SetFilters(const syc_decoder_filter& filter) {
    filters_.clear();
    session_->weighted_filters = nullptr;
    session_->num_weighted_filters = 0;
    session_->filter = filter;
    return *this;
  }
  // Several decoders, weighted (see syc_session). The list is copied. An
  // empty list keeps the single filter of the last SetFilters(filter).
  Session& SetFilters(std::span<const syc_weighted_filter> filters) {
    filters_.assign(filters.begin(), filters.end());
    session_->weighted_filters = filters_.empty() ? nullptr : filters_.data();
    session_->num_weighted_filters = filters_.size();
    return *this;
  }
  Session& SetFilters(std::initializer_list<syc_weighted_filter> filters) {
    return SetFilters(std::span<const syc_weighted_filter>(filters.begin(), filters.size()));
  }

  // Refinement passes (0 to 6, or SYC_ITERATIONS_AUTO) and exact mode (see
  // syc_session). Default: SYC_ITERATIONS_AUTO, not exact.
  Session& SetEffort(int32_t iterations, bool exact) {
    session_->iterations = iterations;
    session_->exact = exact;
    return *this;
  }

  // Validate the settings and prepare for conversions; throws sharpyuv::Error
  // on failure. Call it once per session. After an error the session cannot
  // be used; destroy it.
  Session& Init() {
    internal::Check(syc_session_init(session_), "Session::Init");
    return *this;
  }

  // Enqueue one conversion on the session's stream. The buffers must stay
  // valid until it completes.
  [[nodiscard]] syc_status Convert(const RgbImage& src, const YuvImage& dst) {
    return syc_session_convert(session_, &src, &dst);
  }

  // The cudaError_t behind the last SYC_STATUS_CUDA_ERROR; 0 if none.
  [[nodiscard]] int32_t LastCudaError() const { return syc_session_last_cuda_error(session_); }

 private:
  syc_session* session_;
  std::vector<syc_weighted_filter> filters_;  // the list SetFilters points to
};

}  // namespace sharpyuv
