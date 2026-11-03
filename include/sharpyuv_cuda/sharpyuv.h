/* sharpyuv-cuda: "sharp" RGB -> YUV 4:2:0 conversion on CUDA.
 *
 * Sharp YUV picks the luma and chroma samples so that the image the decoder
 * reconstructs (after chroma upsampling) stays close to the source RGB. It
 * avoids the dark fringes and colour bleeding of a plain 4:2:0 conversion at
 * sharp colour edges.
 *
 * Usage:
 *   syc_session* s = syc_session_create();      defaults filled in
 *   syc_primaries_from_h273(1, &s->primaries);  describe the signal
 *   syc_transfer_from_h273(13, &s->transfer);
 *   syc_matrix_from_h273(1, &s->matrix);
 *   s->range = SYC_RANGE_FULL;
 *   s->yuv_bit_depth = 8;
 *   s->stream = stream;                         other fields: optional
 *   syc_session_init(s);
 *   syc_session_convert(s, &src, &dst);         repeatable, asynchronous
 *   syc_session_destroy(s);
 *
 * Conventions:
 *   * Functions returning syc_status write their outputs only on success.
 *   * Image structs: zero-initialize, then set the fields you need.
 *   * Image memory must be accessible from the device: device, managed, or
 *     mapped pinned host memory. Pitches are in bytes; sizes in pixels.
 *   * Conversions run asynchronously on the session's stream. A returned
 *     status covers argument checks and the kernel launch; errors during
 *     execution are reported by the stream (e.g. cudaStreamSynchronize).
 */

#ifndef SHARPYUV_CUDA_SHARPYUV_H_
#define SHARPYUV_CUDA_SHARPYUV_H_

#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

/* The CUDA runtime's stream type; no CUDA header is needed to use this one. */
typedef struct CUstream_st* cudaStream_t;

/* Linking against the shared library: define SYC_SHARED. */
#if defined(SYC_SHARED)
#if defined(_WIN32)
#if defined(SYC_BUILDING)
#define SYC_API __declspec(dllexport)
#else
#define SYC_API __declspec(dllimport)
#endif
#else
#define SYC_API __attribute__((visibility("default")))
#endif
#else
#define SYC_API
#endif

#if (defined(__cplusplus) && __cplusplus >= 201703L) || \
    (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) ||   \
    (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)
#define SYC_NODISCARD [[nodiscard]]
#elif defined(__GNUC__)
#define SYC_NODISCARD __attribute__((warn_unused_result))
#else
#define SYC_NODISCARD
#endif

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Version and status

/* Version of this header. syc_version() returns the version of the linked
 * library, in the same (major << 16) | (minor << 8) | patch form. */
#define SYC_VERSION_MAJOR 0
#define SYC_VERSION_MINOR 1
#define SYC_VERSION_PATCH 0
#define SYC_VERSION \
  ((SYC_VERSION_MAJOR << 16) | (SYC_VERSION_MINOR << 8) | SYC_VERSION_PATCH)

SYC_API uint32_t syc_version(void);

typedef enum syc_status {
  SYC_STATUS_OK = 0,
  SYC_STATUS_INVALID_ARGUMENT, /* null or misaligned pointer, bad size, pitch
                                  or enum value, unspecified descriptor */
  SYC_STATUS_UNSUPPORTED,      /* valid but not supported, e.g. an H.273 code */
  SYC_STATUS_NOT_READY,        /* conversion before a successful
                                  syc_session_init */
  SYC_STATUS_IMAGE_TOO_LARGE,  /* wider or higher than 65536 pixels */
  SYC_STATUS_OUT_OF_MEMORY,
  SYC_STATUS_CUDA_ERROR        /* see syc_session_last_cuda_error() */
} syc_status;

/* A short description of the status. Never null. */
SYC_API const char* syc_status_string(syc_status status);

// -----------------------------------------------------------------------------
// Color description

/* CIE 1931 xy chromaticity. */
typedef struct syc_chromaticity {
  double x;
  double y;
} syc_chromaticity;

/* Colour primaries, as the luminance weights of R and B
 * (kg = 1 - kr - kb). The conversion preserves the luminance these weights
 * define. */
typedef struct syc_color_primaries {
  double kr;
  double kb;
} syc_color_primaries;

/* H.273 ColourPrimaries.
 *
 * Supported codes:
 *   - 1 (BT.709 / sRGB)
 *   - 4 (BT.470 M)
 *   - 5 (BT.601 625)
 *   - 6, 7 (BT.601 525, SMPTE 240M)
 *   - 8 (generic film)
 *   - 9 (BT.2020)
 *   - 11 (DCI-P3)
 *   - 12 (Display P3)
 *   - 22 (EBU 3213)
 *
 * Returns SYC_STATUS_UNSUPPORTED for other codes. */
SYC_NODISCARD SYC_API syc_status syc_primaries_from_h273(int32_t code,
                                                         syc_color_primaries* out);
/* Custom primaries: red, green, blue and white point. */
SYC_NODISCARD SYC_API syc_status syc_primaries_from_chromaticities(
    syc_chromaticity r, syc_chromaticity g, syc_chromaticity b, syc_chromaticity w,
    syc_color_primaries* out);
/* Luminance weights directly; kr, kb > 0 and kr + kb < 1. */
SYC_NODISCARD SYC_API syc_status syc_primaries_from_luma_coefficients(
    double kr, double kb, syc_color_primaries* out);

/* Custom SDR transfer curve with a linear toe, as an OETF
 * (linear L in [0, 1] -> encoded V in [0, 1]):
 *   V = alpha * L^(1/gamma) - (alpha - 1)   for L >= beta
 *   V = slope * L                           for L <  beta
 * A pure power law has alpha = 1, beta = 0. The two segments must meet at
 * beta. Requires 1 <= gamma <= 10, alpha >= 1, 0 <= beta < 1, slope >= 0. */
typedef struct syc_parametric_curve {
  double gamma;
  double alpha;
  double beta;
  double slope;
} syc_parametric_curve;

typedef enum syc_transfer_kind {
  SYC_TRANSFER_UNSPECIFIED = 0,
  SYC_TRANSFER_PARAMETRIC,
  SYC_TRANSFER_PQ,
  SYC_TRANSFER_HLG,
  SYC_TRANSFER_LINEAR
} syc_transfer_kind;

/* Transfer characteristics of the RGB input and the YUV output (the same). */
typedef struct syc_transfer_function {
  syc_transfer_kind kind;
  syc_parametric_curve curve; /* SYC_TRANSFER_PARAMETRIC only */
} syc_transfer_function;

/* H.273 TransferCharacteristics.
 *
 * Supported codes:
 *   - 1, 6, 14, 15 (BT.709 camera curve, inverse BT.709 OETF)
 *   - 4 (power 2.2)
 *   - 5 (power 2.8)
 *   - 7 (SMPTE 240M)
 *   - 8 (linear)
 *   - 13 (sRGB)
 *   - 16 (PQ)
 *   - 18 (HLG, with the BT.2100 OOTF for a 1000 cd/m^2 display)
 *
 * Note: Use syc_transfer_from_parametric for a pure 2.4 power (BT.1886 with
 * zero black level).
 *
 * Returns SYC_STATUS_UNSUPPORTED for other codes.
 *
 */
SYC_NODISCARD SYC_API syc_status syc_transfer_from_h273(int32_t code,
                                                        syc_transfer_function* out);
SYC_NODISCARD SYC_API syc_status syc_transfer_from_parametric(
    const syc_parametric_curve* curve, syc_transfer_function* out);

typedef enum syc_matrix_kind {
  SYC_MATRIX_UNSPECIFIED = 0,
  SYC_MATRIX_COEFFICIENTS,  /* the kr, kb fields */
  SYC_MATRIX_FROM_PRIMARIES /* the luminance weights of the session's
                               primaries (H.273 code 12); kr, kb are unused */
} syc_matrix_kind;

/* Matrix coefficients of the R'G'B' -> Y'CbCr conversion (non-constant
 * luminance). */
typedef struct syc_matrix_coefficients {
  syc_matrix_kind kind;
  double kr;
  double kb;
} syc_matrix_coefficients;

/* H.273 MatrixCoefficients.
 *
 * Supported codes:
 *   - 1 (BT.709)
 *   - 4 (FCC)
 *   - 5 and 6 (BT.601)
 *   - 7 (SMPTE 240M)
 *   - 9 (BT.2020 non-constant luminance)
 *   - 12 (chroma-derived non-constant luminance)
 *
 * Returns SYC_STATUS_UNSUPPORTED for other codes.
 */
SYC_NODISCARD SYC_API syc_status syc_matrix_from_h273(int32_t code,
                                                      syc_matrix_coefficients* out);
/* Custom coefficients; kr, kb > 0 and kr + kb < 1. */
SYC_NODISCARD SYC_API syc_status syc_matrix_from_coefficients(
    double kr, double kb, syc_matrix_coefficients* out);

typedef enum syc_range {
  SYC_RANGE_UNSPECIFIED = 0,
  SYC_RANGE_LIMITED, /* "video" / "TV" range: H.273 video_full_range_flag 0 */
  SYC_RANGE_FULL     /* video_full_range_flag 1 */
} syc_range;

// -----------------------------------------------------------------------------
// Decoder filter

/*
 * The conversion is optimized for the chroma upsampling of the decoder that
 * will display the image, and works best when this model matches it. A
 * weighted list of filters targets several decoders at once, with a smaller
 * gain under each. Default: bilinear, centred chroma.
 *
 * Common decoders:
 *   nearest                Firefox, Safari, Android; HEIC on iOS
 *   linear, left           Windows codecs (8-bit)
 *   bilinear, centre       Chrome; WebP
 *   bilinear, top-left     iOS AVIF (8-bit)
 *
 * The library writes no metadata: signal the chroma sample position in the
 * container or bitstream yourself. */

/* Position of each chroma sample relative to its 2x2 luma block. */
typedef enum syc_chroma_location {
  SYC_CHROMA_LOCATION_UNSPECIFIED = 0,
  SYC_CHROMA_LOCATION_CENTER,  /* H.273 ChromaSampleLocType 1 (JPEG, WebP;
                                  AVIF / HEIF in practice) */
  SYC_CHROMA_LOCATION_LEFT,    /* ChromaSampleLocType 0 (MPEG-2 and later
                                  video; AV1 "vertical") */
  SYC_CHROMA_LOCATION_TOP_LEFT /* ChromaSampleLocType 2 (AV1 "colocated") */
} syc_chroma_location;

/* Chroma upsampling of the decoder. */
typedef enum syc_filter_kind {
  SYC_FILTER_UNSPECIFIED = 0,
  SYC_FILTER_NEAREST, /* each sample repeated over its 2x2 block (the
                         location has no effect) */
  SYC_FILTER_LINEAR,  /* bilinear horizontally at the location, repeated
                         vertically */
  SYC_FILTER_BILINEAR /* bilinear in both directions at the location */
} syc_filter_kind;

/* Both fields are required, also for SYC_FILTER_NEAREST. */
typedef struct syc_decoder_filter {
  syc_filter_kind kind;
  syc_chroma_location location;
} syc_decoder_filter;

typedef struct syc_weighted_filter {
  double weight; /* finite, >= 0; relative to the other weights of the list */
  syc_decoder_filter filter;
} syc_weighted_filter;

// -----------------------------------------------------------------------------
// Images

typedef enum syc_sample_type {
  SYC_SAMPLE_U8 = 0, /* uint8_t; 0..255 */
  SYC_SAMPLE_U16,    /* uint16_t; 0..2^u16_bits - 1 */
  SYC_SAMPLE_F16,    /* IEEE half; see float_min, float_max */
  SYC_SAMPLE_F32     /* float; see float_min, float_max */
} syc_sample_type;

/* Channel interpretation.
 * If alpha is present, the output is further optimized based on it.
 * Alpha is always straight (not premultiplied).
 */
typedef enum syc_channel_interpretation {
  // Planar formats.
  // R, G, B, data[3] unused
  SYC_INTERP_PLANAR_RGB = 0,
  // R, G, B, A
  SYC_INTERP_PLANAR_RGBA,

  // Interleaved formats, only data[0] used. Name is in byte order.
  // 3 channel opaque formats.
  SYC_INTERP_RGB,
  SYC_INTERP_BGR,

  // 4 channel formats with alpha.
  SYC_INTERP_RGBA,
  SYC_INTERP_BGRA,
  SYC_INTERP_ARGB,
  SYC_INTERP_ABGR,

  // 4 channel formats, ignore alpha.
  SYC_INTERP_RGBX,
  SYC_INTERP_BGRX,
  SYC_INTERP_XRGB,
  SYC_INTERP_XBGR
} syc_channel_interpretation;

typedef enum syc_input_encoding {
  SYC_INPUT_ENCODED = 0, /* encoded with the session's transfer function
                            (the usual case) */
  SYC_INPUT_LINEAR       /* linear light, 1.0 = the transfer function's peak
                            (PQ: 10000 cd/m^2); scene light for HLG */
} syc_input_encoding;

/* RGB Input. Pointers and pitches must be multiples of the sample size, and
 * a pitch must cover a row. Odd sizes are supported. */
typedef struct syc_rgb_image {
  int32_t width;
  int32_t height;
  const void* data[4];
  size_t pitch[4];
  syc_channel_interpretation interpretation;

  syc_sample_type type;
  // For U16 input: number of significant (low) bits, e.g. 10, 12, 16
  int32_t u16_bits;
  // For F16, F32 input: the values that map to black and to white;
  // values outside are clamped. E.g. 0, 255 or -1, 1.
  float float_min;
  float float_max;
  syc_input_encoding encoding;

  uint8_t unused[32];
} syc_rgb_image;

/* Placement of 10- to 16-bit samples in their 16-bit words. */
typedef enum syc_sample_alignment {
  SYC_ALIGN_LSB = 0, /* low bits (yuv420p10le, ...) */
  SYC_ALIGN_MSB      /* high bits, low bits zero (P010, P012, P016) */
} syc_sample_alignment;

/* YUV 4:2:0 output. Chroma planes are ceil(width / 2) x ceil(height / 2).
 * Samples are uint8_t for yuv_bit_depth 8, uint16_t otherwise. Pointers
 * and pitches must be multiples of the sample size, and a pitch must cover
 * a row. */
typedef struct syc_yuv_image {
  // Equal to the input's; or, for an odd input width / height, 1 larger
  // (padded to even): the last row / column of Y (and A) is replicated.
  int32_t width;
  int32_t height;
  void* data[4];
  size_t pitch[4];

  // Planar:   Y, U, V, A.
  // Biplanar: Y, UV, unused, A.
  bool biplanar;

  // false: U before V
  // true:  V before U
  bool swap_uv;
  uint8_t unused[2];
  syc_sample_alignment alignment;

  // Note: YUV bits, range are set in the syc_session

  uint8_t unused2[48];
} syc_yuv_image;

// -----------------------------------------------------------------------------
// Session

/* Refinement passes chosen per filter (see iterations). */
#define SYC_ITERATIONS_AUTO (-1)

/* Conversion settings, bound to one CUDA stream. Create it with
 * syc_session_create, set the fields, then call syc_session_init once. The
 * fields must not be changed after syc_session_init; for other settings,
 * create another session.
 *
 * Future versions may add fields: only use sessions from syc_session_create,
 * never allocate or copy the struct yourself.
 *
 * A session allocates no device memory. It is not thread-safe: use one
 * session per thread, and one per stream for concurrent conversions. */
typedef struct syc_session {
  // Conversions run on this stream and its device. The stream must outlive
  // the session. Default: null, the default stream.
  cudaStream_t stream;

  // Color description. Defaults to JPEG configuration.
  syc_color_primaries primaries;
  syc_transfer_function transfer;
  syc_matrix_coefficients matrix;
  syc_range range;
  int32_t yuv_bit_depth; /* 8, 10, 12 or 16 */

  // The decoder to optimize for (see "Decoder filter").
  // Use weighted_filters if num_weighted_filters > 0, otherwise use filter.
  // The list is copied by syc_session_init.
  // Default: filter = bilinear, centre; no list.
  syc_decoder_filter filter;
  const syc_weighted_filter* weighted_filters;
  size_t num_weighted_filters;

  // Refinement passes, 0 to 6. More passes get closer to the source image and
  // take longer; 0 is a plain downsample in linear light.
  // Default: SYC_ITERATIONS_AUTO, use 3 for nearest and 6 otherwise.
  int32_t iterations;

  // Produce exact output using a slower implementation.
  bool exact;
} syc_session;

// A new session with the defaults configuration. Null if out of memory.
// The returned session must be destroyed with syc_session_destroy.
SYC_API syc_session* syc_session_create(void);

// Validate the configuration and initialize the internal state for conversions.
// Call it once per session. The session cannot be used if any error is
// returned; destroy it.
SYC_NODISCARD SYC_API syc_status syc_session_init(syc_session* session);

// Waits for any inflight conversions, and destroy the session.
SYC_API void syc_session_destroy(syc_session* session);

// Enqueue one conversion on the session's stream. The buffers must stay
// valid until it completes.
SYC_NODISCARD SYC_API syc_status syc_session_convert(syc_session* session,
                                                     const syc_rgb_image* src,
                                                     const syc_yuv_image* dst);

// Return The cudaError_t behind the last SYC_STATUS_CUDA_ERROR; 0 if none.
SYC_API int32_t syc_session_last_cuda_error(const syc_session* session);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SHARPYUV_CUDA_SHARPYUV_H_ */
