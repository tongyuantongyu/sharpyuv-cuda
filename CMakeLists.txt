cmake_minimum_required(VERSION 3.28)
project(sharpyuv_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "75")
    set(CMAKE_CUDA_ARCHITECTURES native)
endif ()

find_package(CUDAToolkit REQUIRED)

# CCCL with MSVC needs the conforming preprocessor; /utf-8 avoids C4819.
set(SHARPYUV_MSVC_FLAGS
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8 /Zc:preprocessor /W4>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/utf-8,/Zc:preprocessor>)

# Library
add_library(sharpyuv_cuda STATIC
        src/color.cpp
        src/filter.cpp
        src/session.cpp
        src/kernel.cu)
target_include_directories(sharpyuv_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_sources(sharpyuv_cuda PUBLIC
        FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES include/sharpyuv_cuda/sharpyuv.hpp)
target_link_libraries(sharpyuv_cuda PUBLIC CUDA::cudart)
target_compile_options(sharpyuv_cuda PRIVATE ${SHARPYUV_MSVC_FLAGS}
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# Tests
option(SHARPYUV_BUILD_TESTS "Build the tests" OFF)
if (SHARPYUV_BUILD_TESTS)
    enable_testing()
    add_executable(sharpyuv_tests tests/test_sharpyuv.cpp)
    target_link_libraries(sharpyuv_tests PRIVATE sharpyuv_cuda)
    target_compile_options(sharpyuv_tests PRIVATE ${SHARPYUV_MSVC_FLAGS})
    add_test(NAME sharpyuv_tests COMMAND sharpyuv_tests)
endif ()

option(SHARPYUV_BUILD_BENCHMARKS "Build the NVBench benchmarks" OFF)
if (SHARPYUV_BUILD_BENCHMARKS)
    include_directories(benchmarks)
endif()
