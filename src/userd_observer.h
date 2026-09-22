#pragma once
#include <stdint.h>
// Observation only. No RM request, GPU mapping or write API.
extern "C" int userd_observer_begin(const char* exclusive_log_path);
extern "C" uint64_t userd_observer_mark(const char* label);
extern "C" int userd_observer_ok();
// Only the owned probe calls this, after the existing mapping analyzer validates
// a fresh live snapshot. Refuses any intervening captured syscall/lifecycle event.
extern "C" int userd_observer_arm_reads(uint64_t validated_sequence, const uint64_t* userd_addresses,
                                      unsigned count);
extern "C" int userd_observer_sample(const char* label, uint64_t api_begin_ns,
                                   uint64_t api_end_ns, int api_result);
extern "C" int userd_observer_finish_reads();
