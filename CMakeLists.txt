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

# Optional, unchanged Interception owner-scoped RM bridge. The original modes
# still build without this dependency. Build only rm_control in that project.
set(BENCH_INTERCEPTION_SOURCE "" CACHE PATH "Interception/active-preempt source directory")
set(BENCH_RM_LIBRARY "" CACHE FILEPATH "Matching 595.58.03 librm_control.so")
if(BENCH_INTERCEPTION_SOURCE OR BENCH_RM_LIBRARY)
  if(NOT EXISTS "${BENCH_INTERCEPTION_SOURCE}/src/ap_bridge.h" OR NOT EXISTS "${BENCH_RM_LIBRARY}")
    message(FATAL_ERROR "Set both BENCH_INTERCEPTION_SOURCE and BENCH_RM_LIBRARY")
  endif()
  find_path(BENCH_NVTX_INCLUDE nvtx3/nvToolsExt.h HINTS ${CUDAToolkit_INCLUDE_DIRS} REQUIRED)
  target_include_directories(context_ping_pong PRIVATE "${BENCH_INTERCEPTION_SOURCE}/src" "${BENCH_NVTX_INCLUDE}")
  target_compile_definitions(context_ping_pong PRIVATE BENCH_HAS_RM=1
    BENCH_RM_LIBRARY_PATH="${BENCH_RM_LIBRARY}"
    BENCH_INTERCEPTION_SOURCE_PATH="${BENCH_INTERCEPTION_SOURCE}")
  target_link_libraries(context_ping_pong PRIVATE "${BENCH_RM_LIBRARY}" ${CMAKE_DL_LIBS})
  set(BENCH_SENTINEL_CUBIN "${CMAKE_CURRENT_BINARY_DIR}/epoch_sentinel.cubin")
  add_custom_command(OUTPUT "${BENCH_SENTINEL_CUBIN}"
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --cubin --std=c++17 -O3 -lineinfo
            "-arch=sm_${BENCH_CUDA_ARCH}" "${CMAKE_CURRENT_SOURCE_DIR}/src/epoch_sentinel.cu"
            -o "${BENCH_SENTINEL_CUBIN}"
    DEPENDS src/epoch_sentinel.cu src/epoch_sentinel.hpp VERBATIM)
  add_custom_target(epoch_sentinel ALL DEPENDS "${BENCH_SENTINEL_CUBIN}")
  add_dependencies(context_ping_pong epoch_sentinel)
  target_compile_definitions(context_ping_pong PRIVATE BENCH_SENTINEL_CUBIN_PATH="${BENCH_SENTINEL_CUBIN}")
endif()
# Opt-in libc observation, plus bounded reads of validated existing USERD.
# This target never constructs an RM request or requests an extra mapping.
set(BENCH_NVIDIA_SOURCE "" CACHE PATH "Unmodified NVIDIA 595.58.03 source for USERD mapping audit")
if(BENCH_NVIDIA_SOURCE)
  if(NOT BENCH_INTERCEPTION_SOURCE OR NOT BENCH_RM_LIBRARY)
    message(FATAL_ERROR "USERD audit also requires the existing RM capture bridge")
  endif()
  execute_process(COMMAND git -C "${BENCH_NVIDIA_SOURCE}" rev-parse HEAD
    OUTPUT_VARIABLE USERD_SOURCE_HEAD OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE USERD_GIT_RESULT)
  execute_process(COMMAND git -C "${BENCH_NVIDIA_SOURCE}" diff --quiet HEAD
    RESULT_VARIABLE USERD_SOURCE_DIFF)
  if(NOT USERD_GIT_RESULT EQUAL 0 OR NOT USERD_SOURCE_DIFF EQUAL 0 OR
      NOT USERD_SOURCE_HEAD STREQUAL "db0c4e65c8e34c678d745ddb1317f53f90d1072b")
    message(FATAL_ERROR "USERD audit requires the unmodified pinned 595.58.03 checkout")
  endif()
  add_library(userd_observer SHARED src/userd_observer.cpp)
  target_compile_features(userd_observer PRIVATE cxx_std_17)
  target_compile_options(userd_observer PRIVATE -Wall -Wextra -Werror)
  target_include_directories(userd_observer PRIVATE
    "${BENCH_NVIDIA_SOURCE}/src/common/sdk/nvidia/inc"
    "${BENCH_NVIDIA_SOURCE}/src/common/inc"
    "${BENCH_NVIDIA_SOURCE}/src/nvidia/arch/nvalloc/unix/include")
  target_link_libraries(userd_observer PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
endif()
if(TARGET userd_observer)
  target_link_libraries(context_ping_pong PRIVATE userd_observer)
  target_compile_definitions(context_ping_pong PRIVATE BENCH_HAS_USERD_OBSERVER=1)
endif()
