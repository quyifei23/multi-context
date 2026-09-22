cmake_minimum_required(VERSION 3.18)
project(context_ping_pong LANGUAGES CXX)

# Compile a device-only cubin separately; the host links only libcuda, not cudart.
find_package(CUDAToolkit 11.4 REQUIRED)
find_package(Threads REQUIRED)
set(BENCH_CUDA_ARCH "80" CACHE STRING "CUDA SM architecture (A100: 80)")
if(NOT BENCH_CUDA_ARCH MATCHES "^[0-9]+$")
  message(FATAL_ERROR "BENCH_CUDA_ARCH must be a numeric SM architecture")
endif()
set(BENCH_CUBIN "${CMAKE_CURRENT_BINARY_DIR}/kernels.cubin")
add_custom_command(
  OUTPUT "${BENCH_CUBIN}"
  COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --cubin --std=c++17 -O3 -lineinfo
          "-arch=sm_${BENCH_CUDA_ARCH}" -Xptxas=-v
          "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.cu" -o "${BENCH_CUBIN}"
  DEPENDS src/kernels.cu src/probe.hpp
  VERBATIM)
add_custom_target(bench_kernels ALL DEPENDS "${BENCH_CUBIN}")
add_executable(context_ping_pong src/main.cpp)
add_dependencies(context_ping_pong bench_kernels)
target_compile_features(context_ping_pong PRIVATE cxx_std_17)
target_compile_options(context_ping_pong PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(context_ping_pong PRIVATE
  BENCH_CUBIN_PATH="${BENCH_CUBIN}"
  BENCH_TOOLKIT_VERSION="${CUDAToolkit_VERSION}"
  BENCH_CUDA_ARCH=${BENCH_CUDA_ARCH})
target_link_libraries(context_ping_pong PRIVATE CUDA::cuda_driver Threads::Threads)
