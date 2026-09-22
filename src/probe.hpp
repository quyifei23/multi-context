#pragma once
#include <cstdint>

// Only atomic loads/stores (no host/GPU RMW) touch the aligned mapped flag.
struct alignas(64) StartMarker {
    alignas(4) unsigned int started;
    std::uint64_t gpu_ns;
};

struct TinyResult {
    std::uint64_t start_gpu_ns;
    std::uint64_t end_gpu_ns;
    std::uint32_t value;
};

constexpr unsigned kFmasPerIteration = 8 * 32;
constexpr unsigned kTokenMask = 0x5a5a5a5aU;
