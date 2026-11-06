# Build & Integration Guide

---

## 1. Requirements & Supported Platforms

- **CMake:** $\ge 3.28$ (uses `FILE_SET HEADERS` and modern CUDA target properties).
- **CUDA Toolkit:** CUDA 12+ / 13+ with C++20 device support and the shared CUDA runtime (`CUDA::cudart`).
- **Host C++ Compiler:** C++20 support (GCC, Clang, or MSVC x64).
  - **Windows / MSVC note:** `CMakeLists.txt` automatically passes `/Zc:preprocessor` (required by CCCL) and `/utf-8` to both `cl` and `nvcc`. Always configure and build from an **x64** Visual Studio developer environment (`vcvars64.bat` / x64 Native Tools Command Prompt); an x86 host toolchain (`Hostx86\x86\cl.exe`) is not supported by the CUDA headers.
- **Target GPU Architectures (`CMAKE_CUDA_ARCHITECTURES`):**
  - If `CMAKE_CUDA_ARCHITECTURES` is not explicitly set (or left at CMake's default detection), `CMakeLists.txt` defaults to:
    ```
    75-real;80-real;86-real;90a-real;100f-real;120f
    ```
    This list covers all GPU architectures supported by **CUDA 13.0** (from Turing `sm_75` through Ampere `sm_80`/`sm_86`, Hopper `sm_90a`, and Blackwell `sm_100f`/`sm_120f`, including PTX via `-f` family targets for forward compatibility within those families).
  - To build only for the GPU installed in the current machine (significantly faster compilation and smaller binary), pass `-DCMAKE_CUDA_ARCHITECTURES=native`.

---

## 2. CMake Configuration Options

| Option | Default | Description |
|---|---|---|
| `BUILD_SHARED_LIBS` | `OFF` | Build `sharpyuv_cuda` as a shared library (`.dll` / `.so`) instead of a static library (`.lib` / `.a`). |
| `SHARPYUV_INSTALL` | `${PROJECT_IS_TOP_LEVEL}` | Generate install rules (`cmake --install`) and CMake package configuration files. |
| `SHARPYUV_BUILD_TESTS` | `OFF` | Build the `sharpyuv_tests` unit test executable and register it with CTest. |
| `CMAKE_CUDA_ARCHITECTURES` | `75-real;80-real;86-real;90a-real;100f-real;120f` | Semicolon-separated list of target CUDA architectures (or `native`). |

---

## 3. Building the Library

### Linux / macOS (or Windows inside x64 Developer Shell)

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSHARPYUV_BUILD_TESTS=ON
cmake --build build
```

### Windows (PowerShell with Visual Studio x64 Environment)

When running from a standard PowerShell session where `cl.exe` and `ninja.exe` are not on `PATH`, initialize `vcvars64.bat` before invoking CMake:

```powershell
$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul 2>&1 && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSHARPYUV_BUILD_TESTS=ON && cmake --build build"
```

*(Adjust the Visual Studio path for your installed edition/version, e.g., `2022\Community` or `18\Community`.)*

### Static vs. Shared Library Builds

- **Static (`-DBUILD_SHARED_LIBS=OFF`, default):** Produces `sharpyuv_cuda.lib` (Windows) or `libsharpyuv_cuda.a` (Linux).
- **Shared (`-DBUILD_SHARED_LIBS=ON`):** Produces `sharpyuv_cuda.dll` + import library `sharpyuv_cuda.lib` (Windows) or `libsharpyuv_cuda.so` (Linux).
  - The shared library compiles with hidden symbol visibility and exports **only the C API functions** (`SYC_API` in `<sharpyuv_cuda/sharpyuv.h>`). The C++20 header (`<sharpyuv_cuda/sharpyuv.hpp>`) is header-only on top of the C ABI and works seamlessly with both static and shared builds.
  - When linking via CMake (`sharpyuv_cuda::sharpyuv_cuda`), `SYC_SHARED` is defined automatically on the target's `INTERFACE`. If you consume the shared library without CMake, define `SYC_SHARED` before including `<sharpyuv_cuda/sharpyuv.h>` so `dllimport` attributes are applied on Windows.

---

## 4. Installing & Consuming in Your Project

### Installing

```bash
cmake --install build --prefix /path/to/install
```

This installs:
- `include/sharpyuv_cuda/sharpyuv.h` and `include/sharpyuv_cuda/sharpyuv.hpp`
- `lib/` — static library or import library / `.so`
- `bin/` — `.dll` (on Windows when `BUILD_SHARED_LIBS=ON`)
- `lib/cmake/sharpyuv_cuda/` — `sharpyuv_cudaConfig.cmake`, `sharpyuv_cudaConfigVersion.cmake`, and exported targets

### Consuming via `find_package`

Point `CMAKE_PREFIX_PATH` to the installation prefix and link `sharpyuv_cuda::sharpyuv_cuda`:

```cmake
find_package(sharpyuv_cuda 0.1 REQUIRED)
target_link_libraries(my_app PRIVATE sharpyuv_cuda::sharpyuv_cuda)
```

- Package version compatibility uses `SameMinorVersion` while major version is `0.x`.
- `sharpyuv_cudaConfig.cmake` calls `find_dependency(CUDAToolkit)` automatically. Consumer projects only need the CUDA Toolkit installed; they **do not** need to enable the `CUDA` language in their own `project(...)` call.

### Consuming via `add_subdirectory` or `FetchContent`

When included as a subproject, `SHARPYUV_INSTALL` defaults to `OFF` and the `sharpyuv_cuda::sharpyuv_cuda` alias target is available directly:

```cmake
add_subdirectory(external/sharpyuv-cuda)
target_link_libraries(my_app PRIVATE sharpyuv_cuda::sharpyuv_cuda)
```

---

## 5. Running Tests & CUDA Sanitizers

### Unit Tests

When configured with `-DSHARPYUV_BUILD_TESTS=ON`, run the test suite via CTest or execute `sharpyuv_tests` directly:

```bash
ctest --test-dir build --output-on-failure
# or directly:
./build/sharpyuv_tests
```

`sharpyuv_tests` validates:
1. All H.273 color primaries, transfer characteristics, parametric curves, and matrix coefficient factories, plus rejection of invalid inputs.
2. Session lifecycle, argument validation, alignment/pitch checks, and error reporting.
3. Exactness on uniform colors across all supported color spaces and bit depths.
4. Format equivalence across interleaved (`RGB`, `BGR`, `RGBA`, `BGRA`, `ARGB`, `ABGR`, `*RGBX*`) and planar inputs (`U8`, `U16` at 10/12/16-bit, `F16`, `F32`, linear and encoded) and planar/biplanar outputs (`I420`, `YV12`, `NV12`, `NV21`, `P010`, `P016`, LSB/MSB).
5. Odd image dimensions ($1 \times 1$, odd width, odd height) with both exact-dimension and even-padded outputs.
6. Agreement across tile sizes ($16 \times 16$ vs. $32 \times 32$), halo widths, decoder filter presets/mixtures, and alpha inpainting against a double-precision host reference.

### Running with NVIDIA `compute-sanitizer`

To verify memory safety, barrier synchronization, and shared-memory race freedom with `compute-sanitizer`, you can set `SHARPYUV_TEST_QUICK=1` to run a reduced test matrix that still exercises every kernel specialization (`TfKind`, `Window`, tile size, and alpha path) in a fraction of the time:

```bash
SHARPYUV_TEST_QUICK=1 compute-sanitizer --tool memcheck ./build/sharpyuv_tests
SHARPYUV_TEST_QUICK=1 compute-sanitizer --tool synccheck ./build/sharpyuv_tests
SHARPYUV_TEST_QUICK=1 compute-sanitizer --tool racecheck ./build/sharpyuv_tests
```

Unset `SHARPYUV_TEST_QUICK` to run the full test matrix under the sanitizers.
