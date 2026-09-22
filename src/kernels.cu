#include "probe.hpp"
#include <cuda/atomic>

__device__ __forceinline__ std::uint64_t global_ns() {
    std::uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
    return value;
}

extern "C" __global__ void compute_kernel(float* output, std::uint64_t iterations,
                                          StartMarker* marker) {
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (marker && tid == 0) {
        marker->gpu_ns = global_ns();
        cuda::atomic_ref<unsigned, cuda::thread_scope_system>(marker->started)
            .store(1, cuda::memory_order_release);
    }
    // Fixed work, eight independent register chains; no clock-based early exit.
    float a = 0.1f + (tid & 255) * 0.0001f, b = 0.2f, c = 0.3f, d = 0.4f;
    float e = 0.5f, f = 0.6f, g = 0.7f, h = 0.8f;
    for (std::uint64_t i = 0; i < iterations; ++i) {
#pragma unroll
        for (unsigned j = 0; j < 32; ++j) {
            a = __fmaf_rn(a, 0.999999f, 0.000001f);
            b = __fmaf_rn(b, 0.999998f, 0.000002f);
            c = __fmaf_rn(c, 0.999997f, 0.000003f);
            d = __fmaf_rn(d, 0.999996f, 0.000004f);
            e = __fmaf_rn(e, 0.999995f, 0.000005f);
            f = __fmaf_rn(f, 0.999994f, 0.000006f);
            g = __fmaf_rn(g, 0.999993f, 0.000007f);
            h = __fmaf_rn(h, 0.999992f, 0.000008f);
        }
    }
    output[tid] = a + b + c + d + e + f + g + h;
}

extern "C" __global__ void tiny_kernel(TinyResult* result, unsigned token) {
    // Exactly one block, one thread. These timestamps add some instrumentation
    // cost; they are observations inside the kernel, not scheduler timestamps.
    result->start_gpu_ns = global_ns();
    result->value = token ^ kTokenMask;
    result->end_gpu_ns = global_ns();
}
