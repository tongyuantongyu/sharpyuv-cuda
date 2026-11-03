cmake_minimum_required(VERSION 3.28)

# The version comes from the public header (SYC_VERSION_*).
file(STRINGS include/sharpyuv_cuda/sharpyuv.h _syc_version_lines
     REGEX "^#define SYC_VERSION_(MAJOR|MINOR|PATCH) ")
foreach (_part MAJOR MINOR PATCH)
    string(REGEX REPLACE ".*SYC_VERSION_${_part} ([0-9]+).*" "\\1" _syc_${_part}
           "${_syc_version_lines}")
endforeach ()
project(sharpyuv_cuda VERSION ${_syc_MAJOR}.${_syc_MINOR}.${_syc_PATCH} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "75")
    set(CMAKE_CUDA_ARCHITECTURES 75-real;80-real;86-real;90a-real;100f-real;120f)
endif ()

find_package(CUDAToolkit REQUIRED)
include(GNUInstallDirs)

# CCCL with MSVC needs the conforming preprocessor; /utf-8 avoids C4819.
set(SHARPYUV_MSVC_FLAGS
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/utf-8 /Zc:preprocessor /W4>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/utf-8,/Zc:preprocessor>)

# Library: static, or shared with BUILD_SHARED_LIBS. The sources are compiled
# once (sharpyuv_cuda_objects) for the library and for sharpyuv_cuda_internal,
# a static library with every symbol for the tests and private tools (a
# shared library exports only the C API).
option(BUILD_SHARED_LIBS "Build sharpyuv_cuda as a shared library" OFF)
add_library(sharpyuv_cuda_objects OBJECT
        src/color.cpp
        src/filter.cpp
        src/session.cpp
        src/kernel.cu)
target_include_directories(sharpyuv_cuda_objects PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(sharpyuv_cuda_objects PUBLIC CUDA::cudart)
target_compile_options(sharpyuv_cuda_objects PRIVATE ${SHARPYUV_MSVC_FLAGS}
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set_target_properties(sharpyuv_cuda_objects PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
if (BUILD_SHARED_LIBS)
    target_compile_definitions(sharpyuv_cuda_objects PRIVATE SYC_SHARED SYC_BUILDING)
    set_target_properties(sharpyuv_cuda_objects PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_VISIBILITY_PRESET hidden
            CUDA_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN ON)
endif ()

add_library(sharpyuv_cuda $<TARGET_OBJECTS:sharpyuv_cuda_objects>)
add_library(sharpyuv_cuda::sharpyuv_cuda ALIAS sharpyuv_cuda)
target_sources(sharpyuv_cuda PUBLIC
        FILE_SET HEADERS BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
        FILES include/sharpyuv_cuda/sharpyuv.h include/sharpyuv_cuda/sharpyuv.hpp)
target_link_libraries(sharpyuv_cuda PRIVATE CUDA::cudart)
set_target_properties(sharpyuv_cuda PROPERTIES
        LINKER_LANGUAGE CXX
        CUDA_RUNTIME_LIBRARY Shared
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})
if (BUILD_SHARED_LIBS)
    target_compile_definitions(sharpyuv_cuda INTERFACE SYC_SHARED)
endif ()

if (BUILD_SHARED_LIBS)
    # Linking the object library directly archives its objects (once).
    add_library(sharpyuv_cuda_internal STATIC)
    target_link_libraries(sharpyuv_cuda_internal PUBLIC sharpyuv_cuda_objects)
    set_target_properties(sharpyuv_cuda_internal PROPERTIES LINKER_LANGUAGE CXX)
else ()
    add_library(sharpyuv_cuda_internal INTERFACE)
    target_link_libraries(sharpyuv_cuda_internal INTERFACE sharpyuv_cuda CUDA::cudart)
endif ()

# Install: library, headers, and a CMake package
#   find_package(sharpyuv_cuda) -> target sharpyuv_cuda::sharpyuv_cuda
option(SHARPYUV_INSTALL "Generate the install rules" ${PROJECT_IS_TOP_LEVEL})
if (SHARPYUV_INSTALL)
    include(CMakePackageConfigHelpers)
    set(SHARPYUV_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/sharpyuv_cuda)
    install(TARGETS sharpyuv_cuda
            EXPORT sharpyuv_cudaTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(EXPORT sharpyuv_cudaTargets
            NAMESPACE sharpyuv_cuda::
            DESTINATION ${SHARPYUV_CMAKE_DIR})
    configure_package_config_file(cmake/sharpyuv_cudaConfig.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/sharpyuv_cudaConfig.cmake
            INSTALL_DESTINATION ${SHARPYUV_CMAKE_DIR})
    # 0.x: minor versions may break the API.
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/sharpyuv_cudaConfigVersion.cmake
            COMPATIBILITY SameMinorVersion)
    install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/sharpyuv_cudaConfig.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/sharpyuv_cudaConfigVersion.cmake
            DESTINATION ${SHARPYUV_CMAKE_DIR})
endif ()

# Tests
option(SHARPYUV_BUILD_TESTS "Build the tests" OFF)
if (SHARPYUV_BUILD_TESTS)
    enable_testing()
    add_executable(sharpyuv_tests tests/test_sharpyuv.cpp)
    target_link_libraries(sharpyuv_tests PRIVATE sharpyuv_cuda_internal)
    target_compile_options(sharpyuv_tests PRIVATE ${SHARPYUV_MSVC_FLAGS})
    add_test(NAME sharpyuv_tests COMMAND sharpyuv_tests)
endif ()

# Experiment tools (debug CLI, benchmarks, experiments), if present.
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/private/CMakeLists.txt)
    add_subdirectory(private)
endif ()
