#pragma once
#include <cstdint>

// Separate, never-recycled old/new slots. The release token publishes all
// fields without depending on an Event that queue rewind might remove.
struct alignas(64) EpochResult {
    unsigned token = 0;
    unsigned value = 0;
    std::uint64_t gpu_ns = 0;
};
constexpr unsigned kEpochMask = 0x19e4ba73U;
