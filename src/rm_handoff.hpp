#pragma once

// Optional experiment, using the existing benchmark's device code and owner
// threads. All RM transport, binding, generations and permissions stay in the
// unchanged Interception bridge; this file never constructs an ioctl.
#include <ap_bridge.h>
#include <nvtx3/nvToolsExt.h>

struct RmTrace {
    bool enabled;
    explicit RmTrace(bool on, const std::string& label) : enabled(on) {
        if (enabled) nvtxRangePushA(label.c_str());
    }
    ~RmTrace() { if (enabled) nvtxRangePop(); }
};

struct RmCall {
    Ns begin = 0, end = 0;
    int api_result = -999;
    ap_bridge_result_t result{};
    RmCall() { result.rm_status = 0xffffffffu; result.syscall_result = -1; }
};

class RmBridge {
    ap_bridge_session_t* session_ = nullptr;
    std::string uuid_;
    std::filesystem::path directory_;
public:
    static void check(int rc, const char* operation) {
        if (rc != AP_BRIDGE_OK)
            throw std::runtime_error(std::string(operation) + ": " + ap_bridge_last_error());
    }
    RmBridge(const Options& o, Output& out) {
        const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
        uuid_ = visible ? visible : "";
        if (uuid_.size() != 40 || uuid_.substr(0, 4) != "GPU-" || o.device != 0)
            throw std::runtime_error("RM mode needs CUDA_VISIBLE_DEVICES=<one full GPU UUID> and --device 0");
        directory_ = o.output + ".rm";
        if (!std::filesystem::create_directory(directory_))
            throw std::runtime_error("RM evidence directory already exists: " + directory_.string());
        const unsigned budget = static_cast<unsigned>(2 * o.trials * o.durations.size() + 2);
        check(ap_bridge_create(uuid_.c_str(), uuid_.c_str(), AP_BRIDGE_OWNER_BACKGROUND, 1, budget, &session_), "bridge create");
        check(ap_bridge_open_journal(session_, directory_.c_str(), "A"), "open RM journal");
        out.meta("rm_schema_version", 1);
        out.meta("rm_bridge_library", BENCH_RM_LIBRARY_PATH);
        out.meta("rm_bridge_source", BENCH_INTERCEPTION_SOURCE_PATH);
        out.meta("rm_control_budget_per_kind", budget);
        out.meta("rm_evidence_directory", directory_.string());
        out.meta("rm_hold_ms", o.hold_ms);
        out.meta("rm_trace_marks", o.trace_marks);
    }
    ~RmBridge() { ap_bridge_destroy(session_); }
    const std::string& uuid() const { return uuid_; }
    void begin_a_scope() { check(ap_bridge_begin_allocation_scope(session_, uuid_.c_str()), "begin A scope"); }
    void end_a_scope() { check(ap_bridge_end_allocation_scope(session_), "end A scope"); }
    void bind(Output& out) {
        check(ap_bridge_note_cuda_ready(session_, 8, 0), "record completed A100 warmup");
        ap_bridge_result_t info{};
        check(ap_bridge_bind_get_info(session_, &info), "bind A/GET_INFO");
        out.meta("a_hardware_tsg_id", info.hardware_tsg_id);
        save("bound");
    }
    void save(const std::string& phase) {
        check(ap_bridge_save_journal(session_), "save RM journal");
        auto write = [&](const char* stem, auto size_fn, auto json_fn) {
            std::vector<char> text(size_fn(session_));
            if (text.empty()) throw std::runtime_error("Empty bridge evidence");
            check(json_fn(session_, text.data(), text.size()), "serialize bridge evidence");
            std::ofstream file(directory_ / (phase + "." + stem + ".json"));
            file.exceptions(std::ios::failbit | std::ios::badbit);
            file << text.data() << '\n';
        };
        write("diagnostics", ap_bridge_diagnostics_json_size, ap_bridge_diagnostics_json);
        // A failed bind still has diagnostics, but no valid identity.
        if (ap_bridge_identity_json_size(session_))
            write("identity", ap_bridge_identity_json_size, ap_bridge_identity_json);
    }
    void control(const std::string& kind, bool enable, RmCall& call, bool trace, int index) {
        RmTrace range(trace, "mc." + kind + "/" + std::to_string(index));
        call.begin = now_ns();
        if (kind == "preempt") call.api_result = ap_bridge_preempt(session_, 1000000, &call.result);
        else if (kind == "schedule_disable" || kind == "schedule_enable")
            call.api_result = ap_bridge_set_enabled(session_, enable, &call.result);
        else call.api_result = ap_bridge_set_channels_enabled(session_, enable, &call.result);
        call.end = now_ns();
        if (call.api_result != AP_BRIDGE_OK) {
            const std::string error = ap_bridge_last_error();
            try { save("control_error"); } catch (...) {}
            throw std::runtime_error(kind + ": " + error);
        }
        if (!call.result.attempted || call.result.syscall_result != 0 || call.result.rm_status != 0)
            throw std::runtime_error("Bridge returned success without accepted RM control");
    }
};

struct RmRecord {
    int index = 0, pending_before_rearm = -1, output_matches = -1, reuse_ok = -1;
    int same_context = -1, same_buffer = -1;
    Ns control_ready = 0, hold_query_begin = 0, hold_query_end = 0;
    RmCall preempt, disable, enable;
    Sample reuse;
};

class RmOutput {
    std::ofstream file_;
public:
    explicit RmOutput(const Options& o) {
        const std::string path = o.output + ".controls.csv";
        if (std::filesystem::exists(path)) throw std::runtime_error("Controls CSV already exists");
        file_.exceptions(std::ios::failbit | std::ios::badbit);
        file_.open(path);
        file_ << "case_index,target_ms,condition,trial,order,valid,status,hold_ms,t_invalidate_ns,t_control_ready_ns";
        for (const char* p : {"preempt", "disable", "enable"})
            for (const char* k : {"call_begin_ns", "call_end_ns", "api_result", "attempted", "ioctl_begin_ns", "ioctl_end_ns",
                                  "syscall_result", "syscall_errno", "rm_status", "rejection", "operation_seq"})
                file_ << ',' << p << '_' << k;
        file_ << ",t_A_hold_query_begin_ns,t_A_hold_query_end_ns,a_pending_before_rearm,"
                 "a_context_same,a_buffer_same,a_output_matches_reference,a_reuse_ok,a_reuse_launch_seq,"
                 "t_A_reuse_launch_ns,t_A_reuse_complete_ns,a_reuse_event_ms,"
                 "invalidate_to_preempt_return_us,invalidate_to_disable_return_us,"
                 "invalidate_to_b_launch_us,invalidate_to_b_complete_us,a_event_ms,reference_a_ms\n";
        file_.flush();
    }
    void row(const Options& o, const Sample& s, const RmRecord& r) {
        auto delta = [](Ns end, Ns begin) { return end && begin ? (end - begin) / 1000.0 : kNaN; };
        file_ << std::setprecision(17) << r.index << ',' << s.target_ms << ',' << s.condition << ',' << s.trial << ','
              << s.order << ',' << s.valid << ',' << s.status << ',' << o.hold_ms << ',' << s.invalidate << ',' << r.control_ready;
        for (const RmCall* c : {&r.preempt, &r.disable, &r.enable})
            file_ << ',' << c->begin << ',' << c->end << ',' << c->api_result << ',' << c->result.attempted
                  << ',' << c->result.begin_ns << ',' << c->result.end_ns << ',' << c->result.syscall_result
                  << ',' << c->result.syscall_errno << ',' << c->result.rm_status << ',' << c->result.rejection << ',' << c->result.operation_seq;
        file_ << ',' << r.hold_query_begin << ',' << r.hold_query_end << ',' << r.pending_before_rearm
              << ',' << r.same_context << ',' << r.same_buffer << ',' << r.output_matches << ',' << r.reuse_ok
              << ',' << r.reuse.b_launch_seq << ',' << r.reuse.b_launch << ',' << r.reuse.b_complete << ',' << r.reuse.b_event_ms
              << ',' << delta(r.preempt.end, s.invalidate) << ',' << delta(r.disable.end, s.invalidate)
              << ',' << delta(s.b_launch, s.invalidate) << ',' << delta(s.b_complete, s.invalidate)
              << ',' << s.a_event_ms << ',' << s.reference_ms << '\n';
        file_.flush();
    }
};

inline int rm_pending(GpuState& g) {
    const CUresult result = g.query_done();
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_NOT_READY) CU(result);
    return result == CUDA_ERROR_NOT_READY;
}

void rm_trial(Worker& a, Worker& b, RmBridge& bridge, const Options& o, Sample& s, RmRecord& r,
              const std::vector<float>& reference) {
    Signals signals;
    const bool hold = s.condition == "preempt-hold" || s.condition == "schedule-hold" || s.condition == "fifo-hold";
    const std::string kind = s.condition == "schedule-hold" ? "schedule_" : "fifo_";
    auto b_future = b.submit([&](GpuState& g) {
        try {
            s.b_ready = now_ns();
            signals.b_armed.store(true, std::memory_order_release);
            if (!await_flag(signals.b_go, signals)) return;
            s.b_trigger_seen = now_ns();
            RmTrace trace(o.trace_marks, "mc.B/" + std::to_string(r.index));
            g.run_tiny(s, &signals.b_done);
        } catch (...) { signals.cancelled.store(true, std::memory_order_release); throw; }
    });
    auto a_future = a.submit([&](GpuState& g) {
        bool disabled = false;
        try {
            if (!await_flag(signals.b_armed, signals)) return;
            const auto ctx = g.context_address();
            const auto buffer = g.output_address();
            RmTrace trace(o.trace_marks, "mc.A/" + std::to_string(r.index));
            g.launch_long(s.iterations, true, s);
            g.wait_started(s, s.target_ms);
            signals.a_started.store(true, std::memory_order_release);
            if (!await_flag(signals.invalidate, signals)) return;
            s.invalidate_seen = now_ns();
            s.a_query_begin = now_ns();
            s.a_pending = rm_pending(g);
            s.a_query_end = now_ns();
            if (!s.a_pending) throw std::runtime_error("A finished before RM control; stop this run");
            if (s.condition != "live-handoff") {
                // FIFO_DISABLE_CHANNELS(onlyScheduling=false) already contains
                // synchronous preemption. fifo-hold tests that one primitive,
                // avoiding the resume window between two separate controls.
                if (s.condition != "fifo-hold")
                    bridge.control("preempt", false, r.preempt, o.trace_marks, r.index);
                if (hold) {
                    bridge.control(kind + "disable", false, r.disable, o.trace_marks, r.index);
                    disabled = true;
                }
                r.control_ready = now_ns();
                signals.b_go.store(true, std::memory_order_release);
            }
            if (!await_flag(signals.b_done, signals)) throw std::runtime_error("B failed during the RM experiment");
            s.a_post_b_query_begin = now_ns();
            s.a_pending_after_b = rm_pending(g);
            s.a_post_b_query_end = now_ns();
            s.a_alive_after_b = g.exists();
            // No more CUDA calls on A during the observation interval.
            const Ns deadline = r.control_ready + static_cast<Ns>(o.hold_ms * 1e6);
            while (now_ns() < deadline) relax_cpu();
            r.hold_query_begin = now_ns();
            r.pending_before_rearm = rm_pending(g);
            r.hold_query_end = now_ns();
            if (disabled) {
                bridge.control(kind + "enable", true, r.enable, o.trace_marks, r.index);
                disabled = false;
            }
            // Re-enable resumes the OLD queue; this is deliberately not discard.
            g.finish_long(s.iterations, s);
            r.output_matches = g.copy_output() == reference;
            if (!r.output_matches) throw std::runtime_error("A output differs from full fixed-work reference");
            {
                RmTrace reuse_trace(o.trace_marks, "mc.A_reuse/" + std::to_string(r.index));
                g.run_tiny(r.reuse);
            }
            r.same_context = ctx == g.context_address();
            r.same_buffer = buffer == g.output_address();
            r.reuse_ok = r.same_context && r.same_buffer;
            if (!r.reuse_ok) throw std::runtime_error("A resources changed across control/reuse");
        } catch (...) {
            signals.cancelled.store(true, std::memory_order_release);
            // Only undo a known-successful disable, through the same gate.
            if (disabled && r.enable.api_result == -999) {
                try { bridge.control(kind + "enable", true, r.enable, o.trace_marks, r.index); }
                catch (...) {}
            }
            throw;
        }
    });
    std::exception_ptr error;
    try {
        if (await_flag(signals.a_started, signals)) {
            const Ns deadline = s.a_start_observed + static_cast<Ns>(s.invalidate_delay_ms * 1e6);
            while (now_ns() < deadline && !signals.cancelled.load(std::memory_order_acquire)) relax_cpu();
            s.cpu_control = sched_getcpu();
            s.invalidate = now_ns();
            if (o.trace_marks) nvtxMarkA(("mc.invalidate/" + std::to_string(r.index)).c_str());
            if (s.condition == "live-handoff") r.control_ready = s.invalidate;
            signals.invalidate.store(true, std::memory_order_release);
            if (s.condition == "live-handoff") signals.b_go.store(true, std::memory_order_release);
        }
    } catch (...) { signals.cancelled.store(true, std::memory_order_release); error = std::current_exception(); }
    try { a_future.get(); } catch (...) { error = std::current_exception(); }
    try { b_future.get(); } catch (...) { if (!error) error = std::current_exception(); }
    if (error) std::rethrow_exception(error);
    if (!(s.a_start_observed <= s.invalidate && s.invalidate <= s.b_launch && s.b_launch <= s.b_complete &&
          s.b_complete <= s.a_post_b_query_begin && s.a_post_b_query_end <= r.hold_query_begin &&
          r.hold_query_end <= s.a_complete_observed && s.a_complete_observed <= r.reuse.b_launch))
        throw std::runtime_error("RM host timestamp order violated");
    if (hold && !((!r.preempt.end || r.preempt.end <= r.disable.begin) && r.disable.end <= s.b_launch && r.hold_query_end <= r.enable.begin))
        throw std::runtime_error("RM disable/re-enable ordering violated");
}

void rm_experiments(CUdevice device, const Options& o, Output& out, RmBridge& bridge) {
    RmOutput rm_out(o);
    Worker a, b;
    out.meta("thread_a_tid", a.submit([&](GpuState&) { pin_cpu(o.cpu_a); return syscall(SYS_gettid); }).get());
    out.meta("thread_b_tid", b.submit([&](GpuState&) { pin_cpu(o.cpu_b); return syscall(SYS_gettid); }).get());
    pin_cpu(o.cpu_control);
    try {
        a.submit([&](GpuState& g) {
            // Only A allocations receive the target-scope tag. B is captured
            // normally but is never eligible for this bridge's A identity.
            bridge.begin_a_scope();
            try {
                RmTrace trace(o.trace_marks, "mc.map.A");
                g.create(device, o, true);
                for (int i = 0; i < o.warmup; ++i) g.run_long(1024, true);
                bridge.end_a_scope();
            } catch (...) { bridge.end_a_scope(); throw; }
            out.meta("a_context_address", g.context_address());
            out.meta("a_output_address", g.output_address());
        }).get();
        b.submit([&](GpuState& g) {
            RmTrace trace(o.trace_marks, "mc.map.B");
            g.create(device, o, false);
            for (int i = 0; i < o.warmup; ++i) { Sample warm; g.run_tiny(warm); }
            out.meta("b_context_address", g.context_address());
        }).get();
        // B adds compute candidates to the registry. Bind only after BOTH
        // contexts are fully initialized, so the GET_INFO credential contains
        // the final candidate revision; keep all stale-binding checks intact.
        a.submit([&](GpuState&) { bridge.bind(out); }).get();
        // Prepare the reversible control on an idle A before timed requests;
        // FIFO preparation also performs the bridge's independent RM GPU UUID
        // query. Never hide that first query inside takeover latency.
        if (o.rm_probe || o.rm_case == "all" || o.rm_case == "preempt-hold" || o.rm_case == "schedule-hold" || o.rm_case == "fifo-hold") {
            a.submit([&](GpuState& g) {
                const std::string kind = o.rm_case == "schedule-hold" ? "schedule_" : "fifo_";
                RmCall disable, enable;
                bridge.control(kind + "disable", false, disable, o.trace_marks, -1);
                bridge.control(kind + "enable", true, enable, o.trace_marks, -1);
                Sample reuse; g.run_tiny(reuse);
            }).get();
        }
        bridge.save("prepared");
        if (!o.rm_probe) {
            std::vector<std::string> cases = o.rm_case == "all" ?
                std::vector<std::string>{"live-handoff", "preempt-resume", "preempt-hold"} :
                std::vector<std::string>{o.rm_case};
            int index = 0;
            for (double target : o.durations) {
                for (int trial = 0; trial < o.trials; ++trial) {
                    const Calibration cal = a.submit([&](GpuState& g) { return calibrate(g, target, o.warmup); }).get();
                    for (std::size_t order = 0; order < cases.size(); ++order) {
                        Sample s;
                        s.experiment = "preempt-hold"; s.condition = cases[(trial + order) % cases.size()];
                        s.target_ms = target; s.iterations = cal.iterations; s.trial = trial; s.order = static_cast<int>(order);
                        RmRecord r; r.index = index++;
                        const auto reference = a.submit([&](GpuState& g) {
                            RmTrace trace(o.trace_marks, "mc.reference/" + std::to_string(r.index));
                            s.reference_ms = g.run_long(cal.iterations, true);
                            return g.copy_output();
                        }).get();
                        s.invalidate_delay_ms = o.invalidate_ms >= 0 ? o.invalidate_ms : s.reference_ms * o.invalidate_fraction;
                        ClockAnchor pre = b.submit([](GpuState& g) { return clock_anchor(g); }).get();
                        try {
                            rm_trial(a, b, bridge, o, s, r, reference);
                            ClockAnchor post = b.submit([](GpuState& g) { return clock_anchor(g); }).get();
                            estimate_start(s, pre, post);
                        } catch (...) {
                            s.valid = 0; s.status = "error"; out.row(s); rm_out.row(o, s, r); throw;
                        }
                        out.row(s); rm_out.row(o, s, r);
                        bridge.save("case_" + std::to_string(r.index));
                        std::cerr << s.condition << ", trial " << trial << ": B complete "
                                  << (s.b_complete - s.invalidate) / 1000.0 << " us; A pending at hold end="
                                  << r.pending_before_rearm << "; A Event=" << s.a_event_ms << " ms; reuse=" << r.reuse_ok << '\n';
                    }
                }
            }
        }
        out.meta("thread_a_launch_count", a.submit([](GpuState& g) { return g.launch_count(); }).get());
        out.meta("thread_b_launch_count", b.submit([](GpuState& g) { return g.launch_count(); }).get());
        bridge.save("before_cleanup");
        a.submit([](GpuState& g) { g.close(); }).get();
        b.submit([](GpuState& g) { g.close(); }).get();
        bridge.save("after_cleanup");
    } catch (...) {
        try { bridge.save("failure"); } catch (...) {}
        throw;
    }
}
