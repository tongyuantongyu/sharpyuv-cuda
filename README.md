# sharpyuv-cuda

**`sharpyuv-cuda`** is a high-performance CUDA library for "sharp" RGB to YUV 4:2:0 chroma subsampling, based on the core idea of `libwebp`'s `sharpyuv`. It redesigns the algorithm and execution model for modern GPUs, wide-gamut / HDR workflows, and added support for configurable decoder chroma filters and alpha-channel compositing.

Standard 4:2:0 subsampling averages chroma in non-linear (gamma) space and computes luma independently of how the decoder will reconstruct chroma. Across saturated edges and fine colored details, this causes dark fringes, color bleeding, and color drift. Sharp YUV iteratively refines the 4:2:0 output to better match the input appearance. See [Algorithm & Color Science](docs/ALGORITHM.md) for more detail.

---

## Features

- **Fused Single-Kernel Implementation**
  - Runs initialization, iterative refinement, and output quantization inside a single CUDA kernel, supporting arbitrary image dimensions (including odd sizes). See [CUDA Kernel Architecture](docs/KERNEL.md).

- **Accurate SDR & HDR Colorimetry**
  - Supports precise color description and various transfer functions, including **sRGB**, **BT.709**, custom **parametric curves**, **linear light**, **SMPTE ST 2084 (PQ)**, and **ARIB STD-B67 (HLG)** (with the BT.2100 display OOTF). See [Algorithm & Color Science](docs/ALGORITHM.md).

- **Configurable Decoder Filter & Chroma Siting Model**
  - Optimizes for the target decoder's chroma upsampling filter (`NEAREST`, `LINEAR`, `BILINEAR`) and chroma sample location (`CENTER`, `LEFT`, `TOP_LEFT`), or a **weighted mixture** of filters to target multiple decoders simultaneously. See [API Guide & Reference](docs/API.md#4-decoder-filter--chroma-siting).

- **Flexible Image Formats**
  - Supports planar and interleaved RGB/RGBA inputs (`U8`, `U16`, `F16`, `F32`) and planar or biplanar 4:2:0 YUV outputs across 8, 10, 12, and 16-bit depths (full or limited range, LSB or MSB aligned), with optional alpha-aware optimization. See [API Guide & Reference](docs/API.md#5-input--output-image-descriptors).

---

## Performance

Measured on an NVIDIA GeForce RTX 5080 (cold L2 cache, default configuration):

| Format & Signal | Decoder Filter | 8K ($7680 \times 4320$) | 4K ($3840 \times 2160$) | 1080p ($1920 \times 1080$) | $256 \times 256$ |
|---|---|---:|---:|---:|---:|
| **8-bit sRGB** (`U8` RGB $\to$ `I420`) | Bilinear, Center | 1.59 ms | 0.41 ms | 0.12 ms | 12.3 µs |
| **8-bit sRGB** (`U8` RGB $\to$ `I420`) | Bilinear, Left | 1.98 ms | 0.51 ms | 0.14 ms | 14.5 µs |
| **8-bit sRGB** (`U8` RGB $\to$ `I420`) | Nearest | 0.48 ms | 0.12 ms | 0.03 ms | 6.3 µs |
| **8-bit sRGB + Alpha** (`U8` RGBA $\to$ `I420` + A) | Bilinear, Center | 1.62 ms | 0.42 ms | 0.11 ms | 12.5 µs |
| **10-bit HDR PQ** (`U16` Planar $\to$ `P010`) | Bilinear, Center | 1.91 ms | 0.49 ms | 0.14 ms | 13.2 µs |
| **10-bit HDR HLG** (`U16` Planar $\to$ `P010`) | Bilinear, Center | 1.94 ms | 0.50 ms | 0.14 ms | 13.8 µs |

---

## Quick Start

Both C (`<sharpyuv_cuda/sharpyuv.h>`) and C++20 (`<sharpyuv_cuda/sharpyuv.hpp>`) headers are provided. See [API Guide & Reference](docs/API.md) for more examples.

```c
#include <sharpyuv_cuda/sharpyuv.h>

syc_status convert_srgb_to_i420(cudaStream_t stream,
                                const void* rgb_dev, size_t rgb_pitch,
                                void* y_dev, size_t y_pitch,
                                void* u_dev, size_t u_pitch,
                                void* v_dev, size_t v_pitch,
                                int32_t width, int32_t height) {
  // Default session configuration is 8-bit full-range JPEG (BT.709/sRGB/BT.601)
  // with centered bilinear chroma upsampling.
  syc_session* s = syc_session_create();
  if (!s) return SYC_STATUS_OUT_OF_MEMORY;

  s->stream = stream;
  syc_status st = syc_session_init(s);
  if (st != SYC_STATUS_OK) {
    syc_session_destroy(s);
    return st;
  }

  syc_rgb_image src = {0};
  src.width = width;
  src.height = height;
  src.data[0] = rgb_dev;
  src.pitch[0] = rgb_pitch;
  src.interpretation = SYC_INTERP_RGB;
  src.type = SYC_SAMPLE_U8;

  syc_yuv_image dst = {0};
  dst.width = width;
  dst.height = height;
  dst.data[0] = y_dev;
  dst.data[1] = u_dev;
  dst.data[2] = v_dev;
  dst.pitch[0] = y_pitch;
  dst.pitch[1] = u_pitch;
  dst.pitch[2] = v_pitch;

  st = syc_session_convert(s, &src, &dst);
  syc_session_destroy(s);  // Waits for in-flight work on s->stream before freeing
  return st;
}
```

---

## Building & Integration

### Requirements

- **CMake** $\ge 3.28$
- **CUDA Toolkit** (with C++20 device compiler support)
- **C++20 host compiler** (GCC, Clang, or MSVC)

### Build & Install

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSHARPYUV_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
cmake --install build --prefix /path/to/install
```

Pass `-DBUILD_SHARED_LIBS=ON` to build a shared library (`.dll` / `.so`) instead of a static library.

### CMake Integration

Via `find_package`:

```cmake
find_package(sharpyuv_cuda 0.1 REQUIRED)
target_link_libraries(my_target PRIVATE sharpyuv_cuda::sharpyuv_cuda)
```

Or via `add_subdirectory`:

```cmake
add_subdirectory(sharpyuv-cuda)
target_link_libraries(my_target PRIVATE sharpyuv_cuda::sharpyuv_cuda)
```

Consumer targets do not need to enable the `CUDA` language in CMake.

---

## Documentation

- **[API Guide & Reference (`docs/API.md`)](docs/API.md)** — API contract and usage guide in detail.
- **[Build & Integration Guide (`docs/BUILD.md`)](docs/BUILD.md)** — Requirements and procedures to build from source.
- **[Algorithm & Color Science (`docs/ALGORITHM.md`)](docs/ALGORITHM.md)** — Mathematical problem definition and algorithm design details.
- **[CUDA Kernel Architecture (`docs/KERNEL.md`)](docs/KERNEL.md)** — Kernel design details.
