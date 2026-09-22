#pragma once
#include <stdint.h>
// Observation only. No RM request, GPU mapping, memory read or control API.
extern "C" int userd_observer_begin(const char* exclusive_log_path);
extern "C" uint64_t userd_observer_mark(const char* label);
extern "C" int userd_observer_ok();
