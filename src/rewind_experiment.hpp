#pragma once
#include "epoch_sentinel.hpp"

struct EpochSlot {
    EpochResult *old_result = nullptr, *new_result = nullptr;
    CUdeviceptr old_buffer = 0, new_buffer = 0;
    CUevent old_end = nullptr, new_end = nullptr;
    unsigned old_token = 0, new_token = 0;
};

unsigned epoch_token(EpochResult* result) {
    return cuda::atomic_ref<unsigned, cuda::thread_scope_system>(result->token).load(cuda::memory_order_acquire);
}

struct EpochState {
    CUmodule module = nullptr;
    CUfunction sentinel = nullptr;
    std::vector<EpochSlot> slots;
    explicit EpochState(const Options& o, GpuState& g) {
        CU(cuModuleLoad(&module, BENCH_SENTINEL_CUBIN_PATH));
        CU(cuModuleGetFunction(&sentinel, module, "epoch_sentinel"));
        slots.resize(o.trials * (o.rm_case == "rewind-paired" ? 2 : 1));
        void* host = nullptr;
        CU(cuMemHostAlloc(&host, (2 * slots.size() + 1) * sizeof(EpochResult), CU_MEMHOSTALLOC_DEVICEMAP));
        auto* memory = static_cast<EpochResult*>(host);
        for (std::size_t i = 0; i <= 2 * slots.size(); ++i) new (memory + i) EpochResult{};
        for (std::size_t i = 0; i < slots.size(); ++i) {
            auto& s = slots[i]; s.old_result = memory + 2 * i; s.new_result = memory + 2 * i + 1;
            s.old_token = 0x10000001u + static_cast<unsigned>(i);
            s.new_token = 0x20000001u + static_cast<unsigned>(i);
            CU(cuMemHostGetDevicePointer(&s.old_buffer, s.old_result, 0));
            CU(cuMemHostGetDevicePointer(&s.new_buffer, s.new_result, 0));
            CU(cuEventCreate(&s.old_end, CU_EVENT_DEFAULT));
            CU(cuEventCreate(&s.new_end, CU_EVENT_DEFAULT));
        }
        // Warm the exact sentinel before binding, using a separate slot. Never
        // reuse/reset a measured slot: a delayed old write must stay visible.
        CUdeviceptr warm = 0; unsigned token = 0x77777777u;
        CU(cuMemHostGetDevicePointer(&warm, memory + 2 * slots.size(), 0));
        void* args[] = {&warm, &token};
        g.note_external_launch();
        CU(cuLaunchKernel(sentinel, 1, 1, 1, 1, 1, 1, 0, g.stream(), args, nullptr));
        CU(cuEventRecord(slots[0].old_end, g.stream()));
        const Ns deadline = now_ns() + static_cast<Ns>(o.query_timeout_ms * 1e6);
        CUresult rc;
        while ((rc = cuEventQuery(slots[0].old_end)) == CUDA_ERROR_NOT_READY) {
            if (now_ns() >= deadline) throw std::runtime_error("Sentinel warmup timeout");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CU(rc);
        if (epoch_token(memory + 2 * slots.size()) != token || memory[2 * slots.size()].value != (token ^ kEpochMask))
            throw std::runtime_error("Sentinel warmup output mismatch");
    }
};

struct EpochCudaError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct EpochLog {
    std::ofstream calls, results;
    explicit EpochLog(const Options& o) {
        for (auto item : {std::make_pair(".epoch_calls.csv", &calls), std::make_pair(".epochs.csv", &results)}) {
            const std::string path = o.output + item.first;
            if (std::filesystem::exists(path)) throw std::runtime_error("Epoch output exists: " + path);
            item.second->exceptions(std::ios::failbit | std::ios::badbit); item.second->open(path);
        }
        calls << "case_index,phase,operation,edge,host_ns,cuda_result,old_token,new_token\n";
        results << "case_index,trial,order,rewind,classification,reason,iterations,reference_a_ms,hold_ms,query_timeout_ms,"
                   "old_expected,new_expected,old_observed,new_observed,old_value_ok,new_value_ok,"
                   "a_event_result,old_event_result,new_event_result,hold_observations,hold_all_pending,"
                   "context_same,buffer_same,context_before,context_after,compute_buffer,old_buffer,new_buffer,old_stream,new_stream,"
                   "t_old_enqueue_begin_ns,t_old_enqueue_return_ns,t_invalidate_ns,t_hold_end_ns,"
                   "t_new_launch_begin_ns,t_new_launch_return_ns,t_observation_end_ns,a_event_ms";
        for (const char* op : {"disable", "enable"})
            for (const char* field : {"begin_ns", "end_ns", "api_result", "attempted", "ioctl_begin_ns", "ioctl_end_ns", "syscall_result", "errno", "rm_status", "rejection", "operation_seq"})
                results << ',' << op << '_' << field;
        results << '\n'; calls.flush(); results.flush();
    }
    void mark(int index, const char* phase, const char* operation, const char* edge, int rc, const EpochSlot& s) {
        calls << index << ',' << phase << ',' << operation << ',' << edge << ',' << now_ns() << ',' << rc
              << ',' << epoch_token(s.old_result) << ',' << epoch_token(s.new_result) << '\n'; calls.flush();
    }
    template<class F> CUresult api(int index, const char* phase, const char* name, const EpochSlot& s, F fn) {
        mark(index, phase, name, "begin", -1, s);
        CUresult rc = fn();
        mark(index, phase, name, "end", rc, s);
        if (rc != CUDA_SUCCESS && rc != CUDA_ERROR_NOT_READY) throw EpochCudaError(std::string(name) + ": " + cuda_error(rc));
        return rc;
    }
};

// Does not recover by Context destruction. After a stop (or successful end),
// flush evidence then exit the process without CUDA cleanup/synchronization.
// The external timeout also bounds a single unexpectedly blocked Driver call.
[[noreturn]] void run_rewind(Worker& a, Worker& b, EpochState& epochs, RmBridge& bridge, const Options& o, Output& out) {
    EpochLog log(o);
    bool stopped = false;
    const bool paired = o.rm_case == "rewind-paired";
    int index = 0;
    out.meta("rewind_query_timeout_ms", o.query_timeout_ms);
    out.meta("epoch_schema_version", 1);
    out.meta("thread_a_launch_count_includes_sentinels", 1);
    out.meta("rewind_sentinel_module", BENCH_SENTINEL_CUBIN_PATH);
    out.meta("rewind_exit_policy", "flush_then_process_exit_no_context_recovery");
    try {
        for (int trial = 0; trial < o.trials && !stopped; ++trial) {
            const Calibration cal = a.submit([&](GpuState& g) { return calibrate(g, o.durations[0], o.warmup); }).get();
            for (int order = 0; order < (paired ? 2 : 1) && !stopped; ++order, ++index) {
                const bool rewind = paired ? (trial + order) % 2 : o.rm_case == "rewind-true";
                stopped = a.submit([&](GpuState& g) {
                    auto& slot = epochs.slots[index];
                    Sample s; s.experiment = "rewind"; s.condition = rewind ? "rewind-true" : "rewind-false";
                    s.trial = trial; s.order = order; s.target_ms = o.durations[0]; s.iterations = cal.iterations;
                    RmCall disable, enable;
                    CUstream fresh = nullptr; CUcontext current = nullptr;
                    const auto context = g.context_address(); const auto buffer = g.output_address();
                    Ns enq_begin = 0, enq_end = 0, hold_end = 0, new_begin = 0, new_end = 0, observed_end = 0;
                    CUresult a_rc = CUDA_ERROR_NOT_READY, old_rc = CUDA_ERROR_NOT_READY, new_rc = CUDA_ERROR_NOT_READY;
                    int hold_count = 0, hold_pending = 1, context_same = -1, buffer_same = -1;
                    std::string classification = "ambiguous", reason;
                    auto api = [&](const char* phase, const char* name, auto fn) { return log.api(index, phase, name, slot, fn); };
                    auto query_old = [&](const char* phase) {
                        a_rc = api(phase, "cuEventQuery_A", [&] { return g.query_done(); });
                        old_rc = api(phase, "cuEventQuery_old_sentinel", [&] { return cuEventQuery(slot.old_end); });
                    };
                    auto old_ok = [&] { return epoch_token(slot.old_result) == slot.old_token && slot.old_result->value == (slot.old_token ^ kEpochMask); };
                    auto new_ok = [&] { return epoch_token(slot.new_result) == slot.new_token && slot.new_result->value == (slot.new_token ^ kEpochMask); };
                    auto observe_old = [&](const char* phase) {
                        const Ns deadline = now_ns() + static_cast<Ns>(o.query_timeout_ms * 1e6);
                        do {
                            query_old(phase);
                            if (a_rc == CUDA_SUCCESS && old_rc == CUDA_SUCCESS && old_ok()) return;
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        } while (now_ns() < deadline);
                    };
                    try {
                        s.reference_ms = g.run_long(cal.iterations, true);
                        s.invalidate_delay_ms = o.invalidate_ms >= 0 ? o.invalidate_ms : s.reference_ms * o.invalidate_fraction;
                        if (o.hold_ms <= s.reference_ms) throw std::runtime_error("Hold is not longer than the full natural reference");
                        RmTrace trace(o.trace_marks, "rw.case/" + std::to_string(index));
                        g.launch_long(cal.iterations, true, s);
                        void* old_args[] = {&slot.old_buffer, &slot.old_token};
                        enq_begin = now_ns();
                        g.note_external_launch();
                        api("enqueue", "cuLaunchKernel_old_sentinel", [&] { return cuLaunchKernel(epochs.sentinel, 1, 1, 1, 1, 1, 1, 0, g.stream(), old_args, nullptr); });
                        api("enqueue", "cuEventRecord_old_sentinel", [&] { return cuEventRecord(slot.old_end, g.stream()); });
                        enq_end = now_ns();
                        g.wait_started(s, s.target_ms);
                        const Ns invalidate_at = s.a_start_observed + static_cast<Ns>(s.invalidate_delay_ms * 1e6);
                        while (now_ns() < invalidate_at) relax_cpu();
                        s.invalidate = now_ns();
                        if (o.trace_marks) nvtxMarkA(("rw.invalidate/" + std::to_string(index)).c_str());
                        query_old("before_disable");
                        if (a_rc != CUDA_ERROR_NOT_READY || old_rc != CUDA_ERROR_NOT_READY || epoch_token(slot.old_result) || epoch_token(slot.new_result))
                            throw std::runtime_error("Old/new epoch precondition is not distinguishable");
                        bridge.control(rewind ? "fifo_rewind" : "fifo_disable", false, disable, o.trace_marks, index);
                        const Ns hold_deadline = disable.end + static_cast<Ns>(o.hold_ms * 1e6);
                        do {
                            query_old("hold"); ++hold_count;
                            hold_pending &= a_rc == CUDA_ERROR_NOT_READY && old_rc == CUDA_ERROR_NOT_READY && epoch_token(slot.old_result) == 0;
                            std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        } while (now_ns() < hold_deadline);
                        hold_end = now_ns();
                        query_old("hold_end");
                        hold_pending &= a_rc == CUDA_ERROR_NOT_READY && old_rc == CUDA_ERROR_NOT_READY && epoch_token(slot.old_result) == 0;
                        bridge.control("fifo_enable", true, enable, o.trace_marks, index);
                        observe_old("after_enable_before_new");
                        // Create after enable. No stream synchronization or
                        // dependency on the old Event is inserted here.
                        api("new_epoch", "cuStreamCreate", [&] { return cuStreamCreate(&fresh, CU_STREAM_NON_BLOCKING); });
                        void* new_args[] = {&slot.new_buffer, &slot.new_token};
                        new_begin = now_ns();
                        g.note_external_launch();
                        api("new_epoch", "cuLaunchKernel_new_sentinel", [&] { return cuLaunchKernel(epochs.sentinel, 1, 1, 1, 1, 1, 1, 0, fresh, new_args, nullptr); });
                        new_end = now_ns();
                        api("new_epoch", "cuEventRecord_new_sentinel", [&] { return cuEventRecord(slot.new_end, fresh); });
                        const Ns new_deadline = now_ns() + static_cast<Ns>(o.query_timeout_ms * 1e6);
                        do {
                            new_rc = api("new_epoch", "cuEventQuery_new_sentinel", [&] { return cuEventQuery(slot.new_end); });
                            if (new_rc == CUDA_SUCCESS) break;
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        } while (now_ns() < new_deadline);
                        if (new_rc != CUDA_SUCCESS) throw std::runtime_error("New submission did not finish within deadline");
                        if (!new_ok()) throw EpochCudaError("New Event completed with incorrect sentinel output");
                        // A new submission could republish stale driver PUT:
                        // inspect the old slot again after new-epoch completion.
                        observe_old("after_new_complete");
                        api("identity", "cuCtxGetCurrent", [&] { return cuCtxGetCurrent(&current); });
                        context_same = reinterpret_cast<std::uintptr_t>(current) == context;
                        CUdeviceptr old_address = 0, new_address = 0, base = 0; std::size_t bytes = 0;
                        api("identity", "cuMemHostGetDevicePointer_old", [&] { return cuMemHostGetDevicePointer(&old_address, slot.old_result, 0); });
                        api("identity", "cuMemHostGetDevicePointer_new", [&] { return cuMemHostGetDevicePointer(&new_address, slot.new_result, 0); });
                        api("identity", "cuMemGetAddressRange", [&] { return cuMemGetAddressRange(&base, &bytes, buffer); });
                        buffer_same = old_address == slot.old_buffer && new_address == slot.new_buffer && base == buffer && bytes > 0;
                        if (a_rc == CUDA_SUCCESS) {
                            float ms = 0;
                            api("result", "cuEventElapsedTime_A", [&] { return cuEventElapsedTime(&ms, g.begin_event(), g.end_event()); });
                            s.a_event_ms = ms;
                        }
                        if (!hold_pending || !context_same || !buffer_same) reason = "hold_or_identity_invariant_failed";
                        else if (old_ok() && a_rc == CUDA_SUCCESS && old_rc == CUDA_SUCCESS) classification = "old_queue_preserved";
                        else if (!epoch_token(slot.old_result) && a_rc == CUDA_SUCCESS && old_rc == CUDA_ERROR_NOT_READY)
                            classification = "queued_sentinel_removed_context_reusable";
                        else reason = "old_completion_or_epoch_state_unresolved";
                        if (classification == "queued_sentinel_removed_context_reusable")
                            reason = "sentinel_absent_in_bounded_windows_not_a_permanent_discard_guarantee";
                    } catch (const EpochCudaError& e) { classification = "context_or_submission_poisoned"; reason = e.what(); }
                      catch (const std::exception& e) { reason = e.what(); }
                    observed_end = now_ns();
                    log.results << std::setprecision(17) << index << ',' << trial << ',' << order << ',' << rewind << ',' << classification << ',' << csv_quote(reason)
                        << ',' << cal.iterations << ',' << s.reference_ms << ',' << o.hold_ms << ',' << o.query_timeout_ms
                        << ',' << slot.old_token << ',' << slot.new_token << ',' << epoch_token(slot.old_result) << ',' << epoch_token(slot.new_result)
                        << ',' << old_ok() << ',' << new_ok() << ',' << a_rc << ',' << old_rc << ',' << new_rc << ',' << hold_count << ',' << hold_pending
                        << ',' << context_same << ',' << buffer_same << ',' << context << ',' << reinterpret_cast<std::uintptr_t>(current)
                        << ',' << buffer << ',' << slot.old_buffer << ',' << slot.new_buffer << ',' << reinterpret_cast<std::uintptr_t>(g.stream()) << ',' << reinterpret_cast<std::uintptr_t>(fresh)
                        << ',' << enq_begin << ',' << enq_end << ',' << s.invalidate << ',' << hold_end << ',' << new_begin << ',' << new_end << ',' << observed_end << ',' << s.a_event_ms;
                    for (const auto* c : {&disable, &enable})
                        log.results << ',' << c->begin << ',' << c->end << ',' << c->api_result << ',' << c->result.attempted << ',' << c->result.begin_ns << ',' << c->result.end_ns
                            << ',' << c->result.syscall_result << ',' << c->result.syscall_errno << ',' << c->result.rm_status << ',' << c->result.rejection << ',' << c->result.operation_seq;
                    log.results << '\n'; log.results.flush();
                    s.valid = classification != "ambiguous" && classification != "context_or_submission_poisoned";
                    s.status = classification; out.row(s);
                    bridge.save("rewind_case_" + std::to_string(index));
                    std::cerr << s.condition << ", trial " << trial << ": " << classification << "; " << reason << '\n';
                    return !s.valid;
                }).get();
            }
        }
        out.meta("thread_a_launch_count", a.submit([](GpuState& g) { return g.launch_count(); }).get());
        out.meta("thread_b_launch_count", b.submit([](GpuState& g) { return g.launch_count(); }).get());
        bridge.save("rewind_end");
        out.meta("run_status", stopped ? "stopped" : "completed");
    } catch (const std::exception& e) {
        stopped = true; out.meta("run_status", "stopped"); out.meta("error", e.what());
        try { bridge.save("rewind_failure"); } catch (...) {}
        std::cerr << "Rewind probe stopped: " << e.what() << '\n';
    }
    out.meta("explicit_context_destroy_calls_in_probe", 0);
    std::cerr.flush(); std::cout.flush();
    std::_Exit(stopped ? 3 : 0);
}
