// sharpyuv-cuda: "sharp" RGB -> YUV 4:2:0 conversion on CUDA.
//
// Typical flow:
//   1. Session::Create(stream)      -- bind to a CUDA stream (and its device)
//   2. session->SetColorInfo()      -- CP / TC / MC / range / depth, host-side
//   3. session->SetFilterInfo()     -- optional: decoder model, iterations, host-side
//   4. session->Convert(src, dst)   -- asynchronous on the session's stream, repeatable
//
// Conventions
//   * No exceptions. Every fallible call returns Status; [[nodiscard]].
//   * All image memory (input, output) must be device-accessible
//     (device, managed, or mapped pinned host memory).
//   * Pitches are in bytes. Dimensions are in pixels.
//   * All GPU work of a session is ordered on its stream. Status reports
//     argument validation and launch errors only; execution errors surface
//     through the stream (or the next call).

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>

typedef struct CUstream_st *cudaStream_t;

namespace sharpyuv {

// ---------------------------------------------------------------------------
// Status

enum class Status : int32_t {
  kOk = 0,
  kInvalidArgument,     // bad pointer, size, pitch, alignment, or enum value
  kUnsupported,         // valid but not supported (e.g. an H.273 code)
  kNotReady,            // call order violated (no color info)
  kImageTooLarge,       // more than 65536 pixels wide or high
  kOutOfMemory,
  kCudaError,           // see Session::LastCudaError()
};

const char* ToString(Status s) noexcept;

// ---------------------------------------------------------------------------
// Color description

// Chromaticity coordinate (CIE 1931 xy).
struct Chromaticity {
  double x;
  double y;
};

// Colour primaries (CP). Used only to derive the linear-light luminance
// weights (Y row of the RGB -> XYZ matrix) that the algorithm preserves.
// Default-constructed: unspecified.
class ColorPrimaries {
 public:
  // H.273 ColourPrimaries code. Unsupported codes: kUnsupported. The list of
  // supported codes is in docs/API.md (table in the implementation).
  [[nodiscard]] static Status FromH273(int code, ColorPrimaries* out);
  // Red, green, blue primaries and white point.
  [[nodiscard]] static Status FromChromaticities(Chromaticity r, Chromaticity g,
                                                 Chromaticity b, Chromaticity w,
                                                 ColorPrimaries* out);
  // Luminance weights directly (kg = 1 - kr - kb).
  [[nodiscard]] static Status FromLumaCoefficients(double kr, double kb,
                                                   ColorPrimaries* out);

  [[nodiscard]] bool specified() const { return specified_; }
  [[nodiscard]] double kr() const { return kr_; }
  [[nodiscard]] double kb() const { return kb_; }

 private:
  bool specified_ = false;
  double kr_{}, kb_{};
};

// Parametric transfer curve with a linear toe, in OETF form
// (linear L in [0, 1] -> encoded V in [0, 1]):
//   V = alpha * L^(1/gamma) - (alpha - 1)   for L >= beta
//   V = slope * L                           for L <  beta
// A pure power law has alpha = 1, beta = 0. The curve must be continuous at
// beta (validated with a tolerance). Standard curves (sRGB, BT.709, ...) are
// available through TransferFunction::FromH273.
struct ParametricCurve {
  double gamma;
  double alpha;
  double beta;
  double slope;
};

// Transfer characteristics (TC) of the RGB input / YUV output signal.
// Default-constructed: unspecified.
class TransferFunction {
 public:
  enum class Kind : uint8_t { kUnspecified, kParametric, kPq, kHlg, kLinear };

  // H.273 TransferCharacteristics code. The BT.709 family (1/6/14/15) is the
  // inverse of the BT.709 OETF (camera curve), not BT.1886. PQ (16) and HLG
  // (18) are only available here. Unsupported codes: kUnsupported. See
  // docs/API.md.
  [[nodiscard]] static Status FromH273(int code, TransferFunction* out);
  [[nodiscard]] static Status FromParametric(const ParametricCurve& curve,
                                             TransferFunction* out);

  [[nodiscard]] bool specified() const { return kind_ != Kind::kUnspecified; }
  [[nodiscard]] Kind kind() const { return kind_; }
  [[nodiscard]] const ParametricCurve& curve() const { return curve_; }  // kParametric only

 private:
  Kind kind_ = Kind::kUnspecified;
  ParametricCurve curve_{};
};

// Matrix coefficients (MC) of the RGB -> YCbCr conversion.
// Default-constructed: unspecified.
class MatrixCoefficients {
 public:
  // H.273 MatrixCoefficients code. 12 (chroma-derived non-constant luminance)
  // takes kr, kb from ColorInfo::primaries (see from_primaries()). Non-YCbCr /
  // constant-luminance codes (identity, BT.2020 CL, chroma-derived CL, ICtCp,
  // ...) are kUnsupported. See docs/API.md.
  [[nodiscard]] static Status FromH273(int code, MatrixCoefficients* out);
  [[nodiscard]] static Status FromCoefficients(double kr, double kb,
                                               MatrixCoefficients* out);

  [[nodiscard]] bool specified() const { return specified_; }
  // True for H.273 code 12: the coefficients are the primaries' luminance
  // weights, resolved by SetColorInfo; kr() and kb() are 0.
  [[nodiscard]] bool from_primaries() const { return from_primaries_; }
  [[nodiscard]] double kr() const { return kr_; }
  [[nodiscard]] double kb() const { return kb_; }

 private:
  bool specified_ = false;
  bool from_primaries_ = false;
  double kr_{}, kb_{};
};

enum class Range : uint8_t {
  kUnspecified,
  kLimited,  // H.273 video_full_range_flag = 0
  kFull,     // video_full_range_flag = 1
};

// Every field must be set; SetColorInfo() rejects unspecified ones. The
// chroma sample position is not part of ColorInfo: it belongs to the decoder
// filter the conversion is optimized for (FilterInfo).
struct ColorInfo {
  ColorPrimaries primaries;
  TransferFunction transfer;
  MatrixCoefficients matrix;
  Range range = Range::kUnspecified;
  int yuv_bit_depth = 0;  // 8, 10, 12 or 16
};

// ---------------------------------------------------------------------------
// Decoder filter
//
// Sharp YUV optimizes the 4:2:0 signal for a model of the decoder's chroma
// upsampler; the model also sets the chroma target filter (the linear-light
// color a chroma sample represents). Target the decoder you care about: the
// gain is largest when the model matches it. A weighted list of filters is
// also supported (one model for several decoders) at the cost of a smaller
// gain under each of them. See docs/API.md.

// Position of each chroma sample relative to its 2x2 luma block.
enum class ChromaLocation : uint8_t {
  kUnspecified,
  kCenter,   // H.273 ChromaSampleLocType 1 (JPEG, WebP; AVIF/HEIF in practice)
  kLeft,     // H.273 ChromaSampleLocType 0 (MPEG-2+ video; AV1 "vertical")
  kTopLeft,  // H.273 ChromaSampleLocType 2 (AV1 "colocated")
};

// Chroma upsampler of the target decoder.
enum class FilterKind : uint8_t {
  kUnspecified,
  kNearest,   // each sample replicated to its 2x2 block (location has no effect)
  kLinear,    // bilinear horizontally at the location, replicated vertically
  kBilinear,  // bilinear on both axes at the location
};

// A pure decoder filter. Both fields must be specified (also for kNearest).
struct DecoderFilter {
  FilterKind kind = FilterKind::kUnspecified;
  ChromaLocation location = ChromaLocation::kUnspecified;
};

struct WeightedFilter {
  double weight = 0.0;  // finite, >= 0; normalized over the list
  DecoderFilter filter;
};

// Decoder model plus the refinement iteration count. Default-constructed:
// unspecified, which selects the library default filter (defined in the
// implementation). Coefficients are derived by Session::SetFilterInfo().
class FilterInfo {
 public:
  [[nodiscard]] static Status FromFilter(DecoderFilter filter, FilterInfo* out);
  // Duplicates are merged. At least one weight must be > 0.
  [[nodiscard]] static Status FromWeighted(std::span<const WeightedFilter> list,
                                           FilterInfo* out);

  [[nodiscard]] bool specified() const;

  // Refinement passes, at most 6 (SetFilterInfo rejects more). Negative (the
  // default): derived from the filter (table in the implementation). 0: plain
  // linear-light downsample, no refinement. Also applies to an unspecified
  // FilterInfo.
  void SetIterations(int32_t iterations) { iterations_ = iterations; }
  [[nodiscard]] int32_t iterations() const { return iterations_; }

  // The kernel solves overlapping image tiles independently. By default the
  // overlap is limited: near tile seams a few codes may differ from a
  // whole-image solve, typically by 1. Exact makes the overlap cover every
  // iteration, so the result equals the whole-image solve, at a higher cost.
  void SetExact(bool exact) { exact_ = exact; }
  [[nodiscard]] bool exact() const { return exact_; }

  // Normalized weight of a pure filter in this mixture (0 if absent or
  // unspecified).
  [[nodiscard]] double Weight(DecoderFilter filter) const;

 private:
  // One slot per (kind, location) pair.
  static constexpr int kSlots = 9;
  std::array<double, kSlots> weights_{};
  int32_t iterations_ = -1;
  bool exact_ = false;
};

// ---------------------------------------------------------------------------
// Images

enum class SampleType : uint8_t { kU8, kU16, kF16, kF32 };

// Channel order of interleaved input. The A channel is used only when
// RgbImage::alpha selects it.
enum class ChannelOrder : uint8_t { kRGB, kBGR, kRGBA, kBGRA, kARGB, kABGR };

// Alpha processing must be requested explicitly. Alpha is straight
// (non-premultiplied); premultiplied data must be un-premultiplied by the
// caller. Alpha is only read, never written. With alpha, the chroma of each
// block is optimized for its visible pixels (alpha-weighted), and fully
// transparent pixels are filled smoothly. See docs/API.md.
enum class AlphaSource : uint8_t {
  kNone,         // no alpha processing (an RGBA order alone does not enable it)
  kInterleaved,  // the A channel of an interleaved RGBA/BGRA/ARGB/ABGR input
  kPlane,        // a separate plane (RgbImage::alpha_data / alpha_pitch) of
                 // planar input
};

// Meaning of the input sample values.
enum class InputEncoding : uint8_t {
  kEncoded,  // transfer-encoded (the usual case), same TC as ColorInfo
  kLinear,   // linear light (e.g. some NN outputs); encoded internally
};

struct RgbImage {
  int width = 0;
  int height = 0;
  SampleType type = SampleType::kU8;
  // U16 only: number of significant bits (LSB-aligned), e.g. 10 or 12;
  // 0 means all 16.
  int u16_bits = 0;
  // Float only: values are clamped to [0, 1] after this affine map.
  float float_scale = 1.0f;
  float float_offset = 0.0f;
  InputEncoding encoding = InputEncoding::kEncoded;

  bool planar = false;
  // Interleaved: data[0], pitch[0], order. Planar: data[0..2] = R, G, B.
  std::array<const void*, 3> data{};
  std::array<size_t, 3> pitch{};
  ChannelOrder order = ChannelOrder::kRGB;

  // Alpha (see AlphaSource). A plane has the same sample type, u16_bits and
  // float mapping as the color samples; planar input only (interleaved input
  // carries alpha as its fourth channel).
  AlphaSource alpha = AlphaSource::kNone;
  const void* alpha_data = nullptr;
  size_t alpha_pitch = 0;

  static RgbImage Interleaved(const void* data, size_t pitch, int width,
                              int height, SampleType type,
                              ChannelOrder order = ChannelOrder::kRGB);
  static RgbImage Planar(const void* r, const void* g, const void* b,
                         size_t pitch, int width, int height, SampleType type);
};

// Storage of > 8-bit YUV samples in 16-bit words.
enum class SampleAlignment : uint8_t {
  kLsb,  // value in low bits (yuv420p10le, ...)
  kMsb,  // value in high bits, low bits zero (P010 / P012 / P016)
};

// 4:2:0 output; chroma planes are ceil(width/2) x ceil(height/2).
// Sample type is uint8_t for yuv_bit_depth == 8, uint16_t otherwise.
struct YuvImage {
  int width = 0;
  int height = 0;
  bool biplanar = false;  // false: Y, U, V planes (I420); true: Y, UV (NV12)
  bool swap_uv = false;   // planar: YV12 plane order; biplanar: NV21 / VU
  SampleAlignment alignment = SampleAlignment::kLsb;
  // Planar: data = {Y, U, V}. Biplanar: data = {Y, UV, unused}.
  std::array<void*, 3> data{};
  std::array<size_t, 3> pitch{};

  // Optional alpha output: the input alpha (RgbImage::alpha) de-interleaved
  // into its own width x height plane, ready for an AVIF/HEIF alpha image.
  // Same bit depth, word size and alignment as Y; always full range (the usual
  // convention for AVIF/HEIF alpha, also with limited-range color). Requires
  // alpha processing (RgbImage::alpha != kNone); null: not written.
  void* alpha_data = nullptr;
  size_t alpha_pitch = 0;

  static YuvImage Planar(void* y, size_t y_pitch, void* u, size_t u_pitch,
                         void* v, size_t v_pitch, int width, int height);
  static YuvImage Biplanar(void* y, size_t y_pitch, void* uv, size_t uv_pitch,
                           int width, int height,
                           SampleAlignment alignment = SampleAlignment::kMsb);
};

// ---------------------------------------------------------------------------
// Session

struct SessionConfig {
  // All conversions of the session are ordered on this stream; the session
  // runs on the stream's device. The stream must outlive the session.
  // Legacy/per-thread default streams are allowed.
  cudaStream_t stream = nullptr;
};

// A session holds the color and filter settings and is bound to one stream.
// It needs no device memory of its own. For concurrency, use one session per
// stream. A session is not thread-safe; calls must not overlap on the host.
class Session {
 public:
  [[nodiscard]] static Status Create(const SessionConfig& config,
                                     std::unique_ptr<Session>* out);
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] cudaStream_t stream() const;

  // Validate and precompute color constants (host-side only, no CUDA calls).
  // Takes effect for subsequent Convert() calls.
  [[nodiscard]] Status SetColorInfo(const ColorInfo& info);

  // Optional. Validate the decoder filter and derive its coefficients and
  // iteration count (host-side only). Without it, or with an unspecified
  // FilterInfo, the library default filter applies. Takes effect for
  // subsequent Convert() calls.
  [[nodiscard]] Status SetFilterInfo(const FilterInfo& filter);

  // Enqueue one conversion on the session stream. Requires SetColorInfo().
  // src and dst must have the same width/height. All per-image state is in
  // src/dst, so a batched overload can be added later without changing this
  // one.
  [[nodiscard]] Status Convert(const RgbImage& src, const YuvImage& dst);

  [[nodiscard]] cudaError_t LastCudaError() const;

 private:
  Session();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sharpyuv
