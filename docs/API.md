# API Guide & Reference

`sharpyuv-cuda` provides two public headers in `include/sharpyuv_cuda/`:

- **[`<sharpyuv_cuda/sharpyuv.h>`](../include/sharpyuv_cuda/sharpyuv.h)** — The core C99 API and stable ABI exported by static and shared builds. Does not require including any CUDA Toolkit headers.
- **[`<sharpyuv_cuda/sharpyuv.hpp>`](../include/sharpyuv_cuda/sharpyuv.hpp)** — A header-only C++20 wrapper over the C API.

---

## 1. Overview & Execution Model

### Session Lifecycle

All conversions are performed through a **session** (`syc_session` in C, `sharpyuv::Session` in C++), which precomputes colorimetric matrices, transfer function parameters, and decoder filter tables once on the host:

```
syc_session_create() ──► configure fields ──► syc_session_init() ──► syc_session_convert() (repeatable) ──► syc_session_destroy()
```

1. **Create:** Call `syc_session_create()` (or construct `sharpyuv::Session`). The returned session is pre-populated with default settings (8-bit full-range JPEG colorimetry, centered bilinear chroma upsampler, automatic iteration count, default CUDA stream).
2. **Configure:** Modify the struct fields on `syc_session*` (or call `SetStream`, `SetColor`, `SetFilters`, `SetEffort` on `sharpyuv::Session`).
3. **Initialize:** Call `syc_session_init(s)` (or `session.Init()`) **once**. This validates all settings, resolves the target device from the stream, and builds the internal kernel tables.
   - Settings **must not be modified** after `syc_session_init()`. To convert with different colorimetry, bit depth, or filter settings, create another session.
   - Only pass `syc_session*` pointers returned by `syc_session_create()`; never stack-allocate or memcpy `syc_session` directly.
4. **Convert:** Call `syc_session_convert(s, &src, &dst)` (or `session.Convert(src, dst)`) for as many frames as needed, of any dimensions ($1 \times 1$ to $65536 \times 65536$).
5. **Destroy:** Call `syc_session_destroy(s)` (or let `sharpyuv::Session` go out of scope). Destruction automatically synchronizes the session's CUDA stream to wait for any in-flight conversions before freeing host memory.

### Stream, Memory & Thread-Safety Semantics

- **Bound Stream & Device:** Each session is bound to one `cudaStream_t` (`nullptr` uses the default stream) and executes on the CUDA device associated with that stream.
- **Fully Asynchronous:** `syc_session_convert()` validates image descriptors on the host and launches a single CUDA kernel on the session's stream. It performs no host-device synchronization, no host-device memory copies, and **no device memory allocations** (all intermediate state is kept in registers and shared memory; see [CUDA Kernel Architecture](KERNEL.md)).
- **Device-Accessible Buffers:** All plane pointers in `syc_rgb_image` and `syc_yuv_image` must be accessible from the session's GPU (device memory via `cudaMalloc`/`cudaMallocPitch`, managed memory, or mapped pinned host memory). All buffers must remain valid until the conversion completes on the stream.
- **Alignment & Pitches:** Plane pointers and row pitches (`pitch[i]`, in bytes) must be multiples of the sample size (`1` byte for `U8` / 8-bit YUV, `2` bytes for `U16`/`F16` / 10–16-bit YUV, `4` bytes for `F32`). Each pitch must be at least as wide as a row of samples.
- **Thread Safety:** A session is **not** thread-safe. Use one session per host thread, and use separate sessions bound to separate streams for concurrent GPU execution.

---

## 2. Quick Start Examples

### C++20 (`<sharpyuv_cuda/sharpyuv.hpp>`)

In the C++ wrapper, factories (`ColorPrimaries::FromH273`, etc.) and `Session::Init()` throw `sharpyuv::Error` (a subclass of `std::runtime_error`) on invalid configuration, while `Session::Convert()` returns a `[[nodiscard]] syc_status`.

```cpp
#include <sharpyuv_cuda/sharpyuv.hpp>

void ConvertHdrFrame(cudaStream_t stream,
                     const void* r_dev, const void* g_dev, const void* b_dev, size_t rgb_pitch,
                     void* y_dev, size_t y_pitch, void* uv_dev, size_t uv_pitch,
                     int32_t width, int32_t height) {
  using namespace sharpyuv;

  // BT.2100 PQ (CP = 9, TC = 16, MC = 9), 10-bit limited range, left-sited bilinear decoder.
  Session session;
  session.SetStream(stream)
      .SetColor(ColorPrimaries::FromH273(9),
                TransferFunction::FromH273(16),
                MatrixCoefficients::FromH273(9),
                SYC_RANGE_LIMITED, 10)
      .SetFilters({SYC_FILTER_BILINEAR, SYC_CHROMA_LOCATION_LEFT})
      .Init();

  // Planar 16-bit RGB input with 10 significant LSB bits.
  RgbImage src;
  src.width = width;
  src.height = height;
  src.data[0] = r_dev;
  src.data[1] = g_dev;
  src.data[2] = b_dev;
  src.pitch[0] = src.pitch[1] = src.pitch[2] = rgb_pitch;
  src.interpretation = SYC_INTERP_PLANAR_RGB;
  src.type = SYC_SAMPLE_U16;
  src.u16_bits = 10;

  // P010 output: 10-bit biplanar Y + UV, MSB-aligned in 16-bit words.
  YuvImage dst;
  dst.width = width;
  dst.height = height;
  dst.data[0] = y_dev;
  dst.data[1] = uv_dev;
  dst.pitch[0] = y_pitch;
  dst.pitch[1] = uv_pitch;
  dst.biplanar = true;
  dst.alignment = SYC_ALIGN_MSB;

  syc_status st = session.Convert(src, dst);
  if (st != SYC_STATUS_OK) {
    // Inspect st or session.LastCudaError()
  }
}
```

### C (`<sharpyuv_cuda/sharpyuv.h>`)

```c
#include <sharpyuv_cuda/sharpyuv.h>

syc_status convert_rgba_with_alpha_plane(cudaStream_t stream,
                                         const void* rgba_dev, size_t rgba_pitch,
                                         void* y_dev, size_t y_pitch,
                                         void* u_dev, size_t u_pitch,
                                         void* v_dev, size_t v_pitch,
                                         void* a_dev, size_t a_pitch,
                                         int32_t width, int32_t height) {
  syc_session* s = syc_session_create();
  if (!s) return SYC_STATUS_OUT_OF_MEMORY;

  // Configure BT.709 primaries, sRGB transfer, BT.709 matrix, full range, 8-bit.
  syc_status st = syc_primaries_from_h273(1, &s->primaries);
  if (st == SYC_STATUS_OK) st = syc_transfer_from_h273(13, &s->transfer);
  if (st == SYC_STATUS_OK) st = syc_matrix_from_h273(1, &s->matrix);
  s->range = SYC_RANGE_FULL;
  s->yuv_bit_depth = 8;
  s->stream = stream;

  if (st == SYC_STATUS_OK) st = syc_session_init(s);
  if (st != SYC_STATUS_OK) {
    syc_session_destroy(s);
    return st;
  }

  // Zero-initialize image descriptors before setting fields.
  syc_rgb_image src = {0};
  src.width = width;
  src.height = height;
  src.data[0] = rgba_dev;
  src.pitch[0] = rgba_pitch;
  src.interpretation = SYC_INTERP_RGBA;  // Straight alpha optimization enabled
  src.type = SYC_SAMPLE_U8;

  syc_yuv_image dst = {0};
  dst.width = width;
  dst.height = height;
  dst.data[0] = y_dev;
  dst.data[1] = u_dev;
  dst.data[2] = v_dev;
  dst.data[3] = a_dev;  // Optional: de-interleaves full-range alpha plane
  dst.pitch[0] = y_pitch;
  dst.pitch[1] = u_pitch;
  dst.pitch[2] = v_pitch;
  dst.pitch[3] = a_pitch;

  st = syc_session_convert(s, &src, &dst);
  syc_session_destroy(s);
  return st;
}
```

---

## 3. Color Description

Unlike traditional converters that only take a $Y'CbCr$ matrix, `sharpyuv-cuda` requires a complete colorimetric description of the signal: **primaries**, **transfer characteristics**, **matrix coefficients**, **range**, and **YUV bit depth**. For the mathematical role of each component, see [Algorithm & Color Science — Color Science Basics & H.273](ALGORITHM.md#11-color-science-basics--itu-t-h273) and [Separating Luminance from Luma](ALGORITHM.md#31-separating-linear-luminance-cp-from-gamma-luma-mc).

When a session is created with `syc_session_create()`, its color fields default to **8-bit full-range JPEG**:
- Primaries: H.273 code `1` (BT.709 / sRGB)
- Transfer: H.273 code `13` (sRGB)
- Matrix: H.273 code `6` (BT.601)
- Range: `SYC_RANGE_FULL`
- Bit depth: `8`

### Color Primaries (`syc_color_primaries` / `sharpyuv::ColorPrimaries`)

Color primaries define the linear-light RGB-to-luminance weights $k_L = (k_r, 1 - k_r - k_b, k_b)$ (the $Y$ row of the RGB $\to$ CIE XYZ matrix) that the algorithm preserves at every pixel.

| C Function | C++ Static Method | Description |
|---|---|---|
| `syc_primaries_from_h273(code, &out)` | `ColorPrimaries::FromH273(code)` | Standard ITU-T H.273 `ColourPrimaries` code. |
| `syc_primaries_from_chromaticities(r, g, b, w, &out)` | `ColorPrimaries::FromChromaticities(r, g, b, w)` | Custom CIE 1931 $(x, y)$ chromaticity coordinates for R, G, B and white point. |
| `syc_primaries_from_luma_coefficients(kr, kb, &out)` | `ColorPrimaries::FromLumaCoefficients(kr, kb)` | Direct linear luminance weights ($k_r, k_b > 0$ and $k_r + k_b < 1$). |

**Supported H.273 `ColourPrimaries` codes:**

| Code | Standard / Gamut | White Point | Derived $(k_r, k_b)$ |
|---:|---|---|---|
| `1` | Rec. ITU-R BT.709 / sRGB | D65 | $(0.2126, 0.0722)$ |
| `4` | Rec. ITU-R BT.470 System M | Illuminant C | $(0.2989, 0.1145)$ |
| `5` | Rec. ITU-R BT.601 (625-line / PAL) | D65 | $(0.2220, 0.0713)$ |
| `6`, `7` | Rec. ITU-R BT.601 (525-line / NTSC), SMPTE 240M | D65 | $(0.2124, 0.0866)$ |
| `8` | Generic film | Illuminant C | $(0.2536, 0.0638)$ |
| `9` | Rec. ITU-R BT.2020 / BT.2100 | D65 | $(0.2627, 0.0593)$ |
| `11` | SMPTE ST 431-2 (DCI-P3) | DCI | $(0.2095, 0.0689)$ |
| `12` | SMPTE EG 432-1 (Display P3) | D65 | $(0.2290, 0.0793)$ |
| `22` | EBU Tech. 3213-E | D65 | $(0.2320, 0.0819)$ |

Other codes return `SYC_STATUS_UNSUPPORTED`.

### Transfer Characteristics (`syc_transfer_function` / `sharpyuv::TransferFunction`)

Defines the electro-optical / opto-electronic mapping between transfer-encoded signals and linear light.

| C Function | C++ Static Method | Description |
|---|---|---|
| `syc_transfer_from_h273(code, &out)` | `TransferFunction::FromH273(code)` | Standard ITU-T H.273 `TransferCharacteristics` code. |
| `syc_transfer_from_parametric(&curve, &out)` | `TransferFunction::FromParametric(curve)` | Custom continuous SDR power law with a linear toe (`syc_parametric_curve`). |

**Supported H.273 `TransferCharacteristics` codes:**

| Code | Curve | Notes |
|---:|---|---|
| `1`, `6`, `14`, `15` | BT.709 / BT.601 / BT.2020 camera curve | Inverse of the BT.709 OETF ($\gamma = 1/0.45$, $\alpha \approx 1.0993$, $\beta \approx 0.0181$, $\text{slope} = 4.5$). |
| `4` | Assumed display gamma 2.2 | Pure power law ($\gamma = 2.2$). |
| `5` | Assumed display gamma 2.8 | Pure power law ($\gamma = 2.8$). |
| `7` | SMPTE 240M | Parametric curve ($\gamma = 1/0.45$, $\alpha = 1.1115$, $\beta = 0.0228$, $\text{slope} = 4.0$). |
| `8` | Linear light | Identity transfer ($V = L$). |
| `13` | IEC 61966-2-1 sRGB | Piecewise sRGB curve ($\gamma = 2.4$, $\alpha = 1.055$, $\beta = 0.0031308$, $\text{slope} = 12.92$). |
| `16` | SMPTE ST 2084 (PQ) | Perceptual Quantizer ($L = 1.0 \leftrightarrow 10\,000\text{ cd/m}^2$). |
| `18` | ARIB STD-B67 (HLG) | Hybrid Log-Gamma, modeled in display light with the BT.2100 OOTF for a $1000\text{ cd/m}^2$ nominal peak display ($\gamma_{\text{sys}} = 1.2$). |

**Custom Parametric Curves (`syc_parametric_curve`):**

Defines an SDR OETF mapping linear light $L \in [0, 1]$ to encoded value $V \in [0, 1]$:
$$V = \begin{cases} \alpha L^{1/\gamma} - (\alpha - 1) & \text{if } L \ge \beta \\ \text{slope} \cdot L & \text{if } L < \beta \end{cases}$$

- Valid constraints: $1 \le \gamma \le 10$, $\alpha \ge 1$, $0 \le \beta < 1$, $\text{slope} \ge 0$.
- If $\beta > 0$, the two segments must meet continuously at $\beta$, with $0 < \text{slope} \cdot \beta < 1$.
- For a pure power law (such as BT.1886 with zero black lift, $\gamma = 2.4$), set `{.gamma = 2.4, .alpha = 1.0, .beta = 0.0, .slope = 0.0}`.

### Matrix Coefficients (`syc_matrix_coefficients` / `sharpyuv::MatrixCoefficients`)

Defines the non-constant-luminance $R'G'B' \to Y'CbCr$ conversion matrix and gamma-space luma decomposition weights $k_M = (k_r, 1 - k_r - k_b, k_b)$.

| C Function | C++ Static Method | Description |
|---|---|---|
| `syc_matrix_from_h273(code, &out)` | `MatrixCoefficients::FromH273(code)` | Standard ITU-T H.273 `MatrixCoefficients` code. |
| `syc_matrix_from_coefficients(kr, kb, &out)` | `MatrixCoefficients::FromCoefficients(kr, kb)` | Custom $(k_r, k_b)$ weights ($k_r, k_b > 0$ and $k_r + k_b < 1$). |

**Supported H.273 `MatrixCoefficients` codes:**

| Code | Standard | $(k_r, k_b)$ |
|---:|---|---|
| `1` | Rec. ITU-R BT.709 | $(0.2126, 0.0722)$ |
| `4` | FCC Title 47 | $(0.30, 0.11)$ |
| `5`, `6` | Rec. ITU-R BT.601 (625 / 525) | $(0.299, 0.114)$ |
| `7` | SMPTE 240M | $(0.212, 0.087)$ |
| `9` | Rec. ITU-R BT.2020 non-constant luminance | $(0.2627, 0.0593)$ |
| `12` | Chroma-derived non-constant luminance | Uses the $(k_r, k_b)$ of the session's `primaries` (`SYC_MATRIX_FROM_PRIMARIES`). |

### Range & Output Bit Depth

- **`range` (`syc_range`):**
  - `SYC_RANGE_FULL`: $Y' \in [0, 2^n - 1]$, $Cb, Cr \in [0, 2^n - 1]$ centered at $2^{n-1}$.
  - `SYC_RANGE_LIMITED`: H.273 narrow/studio range: $Y' = \text{round}((219 \cdot W + 16) \cdot 2^{n-8})$, $Cb, Cr = \text{round}((224 \cdot C + 128) \cdot 2^{n-8})$.
- **`yuv_bit_depth` (`int32_t`):** Supported bit depths are `8`, `10`, `12`, and `16`.

---

## 4. Decoder Filter & Chroma Siting

Sharp YUV optimizes the 4:2:0 output for a model of the **decoder's chroma upsampler**. The quality gain is highest when the configured model matches the decoder that will display the image. For how the upsampler and target filters are constructed mathematically, see [Algorithm & Color Science — Decoder Filter Models, Chroma Siting & Blends](ALGORITHM.md#33-decoder-filter-models-chroma-siting--blends).

### Pure Decoder Filters (`syc_decoder_filter`)

A single decoder filter is specified by a `(kind, location)` pair:

- **`syc_filter_kind`:**
  - `SYC_FILTER_NEAREST`: Each chroma sample is replicated across its $2 \times 2$ luma block (location has no effect, but must still be set to a valid enum value).
  - `SYC_FILTER_LINEAR`: Bilinear interpolation horizontally at `location`, replicated vertically.
  - `SYC_FILTER_BILINEAR`: Bilinear interpolation in both directions at `location`.
- **`syc_chroma_location`:**
  - `SYC_CHROMA_LOCATION_CENTER`: H.273 `ChromaSampleLocType` 1 — centered both horizontally and vertically $(0.5, 0.5)$.
  - `SYC_CHROMA_LOCATION_LEFT`: H.273 `ChromaSampleLocType` 0 (AV1 `"vertical"`) — co-sited horizontally, centered vertically $(0.0, 0.5)$.
  - `SYC_CHROMA_LOCATION_TOP_LEFT`: H.273 `ChromaSampleLocType` 2 (AV1 `"colocated"`) — co-sited both horizontally and vertically $(0.0, 0.0)$.

Both `kind` and `location` must be specified (non-zero), even when `kind == SYC_FILTER_NEAREST`.

> **Note on real-world decoders (status as of testing, subject to change across OS/browser versions):**
> In empirical probing of image viewers and browsers (which generally ignore bitstream chroma-location flags and apply fixed upsamplers):
> - **Bilinear, Center (`SYC_FILTER_BILINEAR`, `SYC_CHROMA_LOCATION_CENTER`):** Chromium-based browsers (Chrome, Edge), `libwebp` fancy upsampler, iOS 10-bit AVIF.
> - **Bilinear, Top-Left (`SYC_FILTER_BILINEAR`, `SYC_CHROMA_LOCATION_TOP_LEFT`):** iOS Photos 8-bit AVIF.
> - **Linear, Left (`SYC_FILTER_LINEAR`, `SYC_CHROMA_LOCATION_LEFT`):** Windows 8-bit WIC codecs (Windows Photos, Photoshop).
> - **Nearest (`SYC_FILTER_NEAREST`):** Firefox, Safari, Android photo viewer, iOS HEIC.
>
> `sharpyuv-cuda` operates on raw pixel planes and does not write container or bitstream metadata; signal the chroma sample location in your encoder/container as needed.

### Weighted Filter Mixtures (`syc_weighted_filter`)

When an image will be viewed across different decoders whose upsamplers disagree, you can pass a list of `syc_weighted_filter` entries (`s->weighted_filters` and `s->num_weighted_filters` in C, or `session.SetFilters(span_or_init_list)` in C++):

```cpp
session.SetFilters({
    {0.5, {SYC_FILTER_BILINEAR, SYC_CHROMA_LOCATION_CENTER}},
    {0.5, {SYC_FILTER_NEAREST,  SYC_CHROMA_LOCATION_CENTER}},
});
```

- Each `weight` must be finite and $\ge 0$, with at least one weight $> 0$.
- Duplicate `(kind, location)` entries are summed and weights are normalized to sum to $1$.
- Because both the upsampler and target filter are linear, a mixture executes as a single combined filter table at the cost of a single reconstruction per iteration (with a $3 \times 3$ target window if any co-sited member has positive weight).
- The result will be **worse** than optimizing for a specific filter, but still better than naive conversion.

---

## 5. Input & Output Image Descriptors

In C, always zero-initialize `syc_rgb_image` and `syc_yuv_image` (`= {0}`) before populating fields so reserved `unused` padding is zeroed. In C++, `sharpyuv::RgbImage` and `sharpyuv::YuvImage` zero-initialize automatically and set `u16_bits = 16`, `float_min = 0.0f`, `float_max = 1.0f`.

### RGB Input (`syc_rgb_image` / `sharpyuv::RgbImage`)

| Field | Type | Description |
|---|---|---|
| `width`, `height` | `int32_t` | Image dimensions in pixels ($1 \dots 65536$). Odd dimensions are supported. |
| `data[4]` | `const void*` | Plane pointers. Planar formats use `data[0..2]` (`R, G, B`) and `data[3]` (`A` for `PLANAR_RGBA`); interleaved formats use `data[0]` only. |
| `pitch[4]` | `size_t` | Row stride in bytes for each active plane. Must be a multiple of sample size and $\ge \text{width} \times \text{channels\_in\_plane} \times \text{sample\_bytes}$. |
| `interpretation` | `syc_channel_interpretation` | Channel layout and alpha behavior (see below). |
| `type` | `syc_sample_type` | `SYC_SAMPLE_U8`, `SYC_SAMPLE_U16`, `SYC_SAMPLE_F16`, or `SYC_SAMPLE_F32`. |
| `u16_bits` | `int32_t` | Required when `type == SYC_SAMPLE_U16`: number of significant LSB-aligned bits ($1 \dots 16$, e.g. `10`, `12`, `16`). |
| `float_min`, `float_max` | `float` | Required when `type` is `F16` or `F32`: finite values mapping to black ($0.0$) and white ($1.0$), with `float_min != float_max`. Values outside are clamped to $[0, 1]$. |
| `encoding` | `syc_input_encoding` | `SYC_INPUT_ENCODED` (default: already transfer-encoded with the session's transfer function) or `SYC_INPUT_LINEAR` (linear light where $1.0$ = transfer peak; scene light for HLG). |

**Channel Interpretations (`syc_channel_interpretation`):**

The SharpYUV algorithm doesn't work well if the alpha is pre-multiplied. **If alpha is provided, it's assumed to be straight.**

| Enum Value | Layout | Planes Used | Alpha Behavior |
|---|---|---|---|
| `SYC_INTERP_PLANAR_RGB` | Planar R, G, B | `data[0..2]` | Opaque |
| `SYC_INTERP_PLANAR_RGBA` | Planar R, G, B, A | `data[0..3]` | Alpha optimization enabled |
| `SYC_INTERP_RGB`, `SYC_INTERP_BGR` | 3-channel interleaved (byte order) | `data[0]` | Opaque |
| `SYC_INTERP_RGBA`, `SYC_INTERP_BGRA`, `SYC_INTERP_ARGB`, `SYC_INTERP_ABGR` | 4-channel interleaved (byte order) | `data[0]` | Alpha optimization enabled |
| `SYC_INTERP_RGBX`, `SYC_INTERP_BGRX`, `SYC_INTERP_XRGB`, `SYC_INTERP_XBGR` | 4-channel interleaved (byte order) | `data[0]` | 4th channel ignored (opaque) |

### YUV 4:2:0 Output (`syc_yuv_image` / `sharpyuv::YuvImage`)

Output samples are `uint8_t` when `session->yuv_bit_depth == 8`, and `uint16_t` when `yuv_bit_depth` is `10`, `12`, or `16`. Chroma plane dimensions are always $W_c = \lceil \text{width} / 2 \rceil$ by $H_c = \lceil \text{height} / 2 \rceil$.

| Field | Type | Description |
|---|---|---|
| `width`, `height` | `int32_t` | Output luma dimensions (equal to input dimensions, or padded to even — see [Dimensions & Odd-Size Padding](#dimensions--odd-size-padding)). |
| `data[4]` | `void*` | Plane pointers: `data[0]` = Y; `data[1]` = U (planar) or interleaved UV/VU (biplanar); `data[2]` = V (planar) or unused (biplanar); `data[3]` = optional output A plane (or `nullptr`). |
| `pitch[4]` | `size_t` | Row stride in bytes for each non-null plane. For biplanar `data[1]`, each row has $2 \times W_c$ samples. |
| `biplanar` | `bool` | `false` for 3-plane planar (`Y`, `U`, `V`); `true` for 2-plane semi-planar (`Y`, interleaved `UV`). |
| `swap_uv` | `bool` | `false` for `U` ($Cb$) before `V` ($Cr$); `true` for `V` ($Cr$) before `U` ($Cb$). |
| `alignment` | `syc_sample_alignment` | `SYC_ALIGN_LSB` (low bits of `uint16_t`) or `SYC_ALIGN_MSB` (high bits of `uint16_t`, shifted left by $16 - \text{depth}$; ignored for 8-bit). |

**Common YUV 4:2:0 Format Recipes:**

| Format | `yuv_bit_depth` | `biplanar` | `swap_uv` | `alignment` | Active Planes |
|---|---:|---|---|---|---|
| `I420` (`YUV420P`) | `8` | `false` | `false` | `SYC_ALIGN_LSB` | `data[0..2]` (`Y, U, V`) |
| `YV12` | `8` | `false` | `true` | `SYC_ALIGN_LSB` | `data[0..2]` (`Y, V, U`) |
| `NV12` | `8` | `true` | `false` | `SYC_ALIGN_LSB` | `data[0..1]` (`Y, UV`) |
| `NV21` | `8` | `true` | `true` | `SYC_ALIGN_LSB` | `data[0..1]` (`Y, VU`) |
| `yuv420p10le` / `12le` / `16le` | `10` / `12` / `16` | `false` | `false` | `SYC_ALIGN_LSB` | `data[0..2]` (`Y, U, V`) |
| `P010` / `P012` / `P016` | `10` / `12` / `16` | `true` | `false` | `SYC_ALIGN_MSB` | `data[0..1]` (`Y, UV`) |

### Dimensions & Odd-Size Padding

Both even and odd source dimensions ($1 \dots 65536$) are supported:
- For any source dimension $D$ (`width` or `height`), `dst.width` (and independently `dst.height`) may be either:
  1. **Exact size ($D$):** `dst.width == src.width`. Only $D$ luma (and alpha) samples are written per row/column.
  2. **Even-padded ($D + 1$ when $D$ is odd):** `dst.width == src.width + 1` (for odd `src.width`). The last column/row of $Y$ (and $A$, if present) is replicated to fill the even $2 \times 2$ block, which is useful when feeding hardware encoders that require even luma dimensions.
- In both cases, chroma planes have dimensions $\lceil \text{src.width} / 2 \rceil \times \lceil \text{src.height} / 2 \rceil$.

---

## 6. Alpha-aware optimization & Alpha Output Plane

When `src.interpretation` includes an active alpha channel (`SYC_INTERP_PLANAR_RGBA`, `SYC_INTERP_RGBA`, `SYC_INTERP_BGRA`, `SYC_INTERP_ARGB`, `SYC_INTERP_ABGR`), `sharpyuv-cuda` optimizes luma and chroma around transparency boundaries. For algorithm details, see [Algorithm & Color Science — Alpha-aware optimization & 1-Ring Inpainting](ALGORITHM.md#35-alpha-aware-optimization--1-ring-inpainting).

- **Straight Alpha Only:** Alpha is always interpreted as **straight (unassociated / non-premultiplied)**. Premultiplied 4:2:0 cannot avoid chroma bleed onto transparent pixels; if your source data is premultiplied, un-premultiply it before calling `Convert`.
- **Input Alpha Scaling:** In `SYC_INTERP_PLANAR_RGBA` or interleaved RGBA, the alpha channel uses the same sample type and range mapping (`u16_bits` or `float_min`/`float_max`) as RGB, clamped to $[0, 1]$, and is never passed through the color transfer curve.
- **Optional Alpha Output Plane (`dst.data[3]`, `dst.pitch[3]`):**
  - If `dst.data[3]` is non-null, the kernel writes a de-interleaved `dst.width × dst.height` alpha plane alongside YUV in the same pass.
  - The output alpha plane uses the same bit depth, word size (`uint8_t` or `uint16_t`), and `SYC_ALIGN_LSB`/`SYC_ALIGN_MSB` alignment as $Y$, but is **always quantized to full range** ($[0, 2^n - 1]$), even when color is limited range.
  - Setting `dst.data[3] != nullptr` when the input interpretation has no active alpha channel returns `SYC_STATUS_INVALID_ARGUMENT`.

---

## 7. Effort, Exact Mode & Error Handling

### Refinement Iterations & Exact Mode

Configured via `s->iterations` and `s->exact` in C, or `session.SetEffort(iterations, exact)` in C++. For how tiles and halos work internally, see [CUDA Kernel Architecture — Tile Geometry, Halo Width & Adaptive Tile Sizing](KERNEL.md#2-tile-geometry-halo-width--adaptive-tile-sizing).

- **`iterations` (`int32_t`):**
  - `SYC_ITERATIONS_AUTO` (`-1`, default): Selects `3` iterations when the filter is pure `SYC_FILTER_NEAREST`, and `6` iterations for all other filters and mixtures.
  - `0 .. 6`: Fixed number of Jacobi refinement passes. `0` performs a single-pass linear-light downsample without iterative refinement. Values outside `-1 .. 6` are rejected by `syc_session_init()` with `SYC_STATUS_INVALID_ARGUMENT`.
- **`exact` (`bool`, default `false`):**
  - `false` (default): Caps the inter-tile halo overlap at 3 blocks (`min(iterations, 3)`). Up to 3 iterations (and for `SYC_FILTER_NEAREST` at any iteration count), this is identical to a whole-image solve; at 4–6 iterations, $< 0.1\%$ of output codes near tile boundaries differ by at most 1 code.
  - `true`: Sets the tile halo equal to `iterations` (up to 6 blocks) and computes the outer 2-block alpha inpaint ring, producing output identical to an untiled whole-image solve at roughly $1.6\times$ the runtime cost.

### Status Codes & Error Handling

| `syc_status` | Meaning |
|---|---|
| `SYC_STATUS_OK` (`0`) | Success. |
| `SYC_STATUS_INVALID_ARGUMENT` | Null or misaligned pointer, invalid dimension or pitch, out-of-range enum/weight/curve parameter, or invalid session pointer. |
| `SYC_STATUS_UNSUPPORTED` | Valid H.273 code that is not supported by the library. |
| `SYC_STATUS_NOT_READY` | `syc_session_convert()` called before a successful `syc_session_init()`. |
| `SYC_STATUS_IMAGE_TOO_LARGE` | Image width or height exceeds $65536$ pixels. |
| `SYC_STATUS_OUT_OF_MEMORY` | Host or CUDA memory allocation failed. |
| `SYC_STATUS_CUDA_ERROR` | A CUDA runtime API call or kernel launch failed; call `syc_session_last_cuda_error(s)` / `session.LastCudaError()` to retrieve the `cudaError_t` code. |

- `syc_status_string(status)` returns a non-null human-readable string for any `syc_status` (also supported via `operator<<` on `std::ostream` in C++).
- `SYC_VERSION` (`(MAJOR << 16) | (MINOR << 8) | PATCH`) gives the compile-time header version, and `syc_version()` returns the linked library's runtime version.
