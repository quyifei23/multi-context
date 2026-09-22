#include "epoch_sentinel.hpp"
#include <cuda/atomic>

extern "C" __global__ void epoch_sentinel(EpochResult* result, unsigned token) {
    std::uint64_t timestamp;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp) :: "memory");
    result->value = token ^ kEpochMask;
    result->gpu_ns = timestamp;
    cuda::atomic_ref<unsigned, cuda::thread_scope_system>(result->token)
        .store(token, cuda::memory_order_release);
}
