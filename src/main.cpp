#include "probe.hpp"
#include <cuda.h>
#include <cuda/atomic>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <thread>
#include <type_traits>
#include <utility>
#include <unistd.h>
#include <vector>

using Ns = std::int64_t;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

Ns now_ns() {
    timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0)
        throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC_RAW) failed");
    return static_cast<Ns>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

void relax_cpu() {
#if defined(__x86_64__) || defined(__i386__)
    asm volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

void pin_cpu(int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
        throw std::runtime_error("Cannot bind host thread to CPU " + std::to_string(cpu));
}

std::string cuda_error(CUresult result) {
    const char* name = nullptr;
    const char* description = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    return std::string(name ? name : "unknown CUDA error") + " (" +
           std::to_string(static_cast<int>(result)) + "): " +
           (description ? description : "no description");
}

void check(CUresult result, const char* call) {
    if (result != CUDA_SUCCESS)
        throw std::runtime_error(std::string(call) + ": " + cuda_error(result));
}
#define CU(call) check((call), #call)

struct Options {
    std::string mode = "all", action = "destroy", output = "results/run.csv";
    std::string module = BENCH_CUBIN_PATH;
    std::vector<double> durations{10, 100, 1000};
    int device = 0, trials = 10, warmup = 3, threads = 256, blocks = 0;
    int cpu_a = -1, cpu_b = -1, cpu_control = -1;
    double invalidate_fraction = 0.1, invalidate_ms = -1;
    bool estimate_start = false;
};

void usage() {
    std::cout << R"(CUDA Driver API context ping-pong (Linux, default cubin: sm_80/A100)
Usage: context_ping_pong [options]
  --mode all|standby|takeover|live-handoff   Default: all (legacy two experiments)
  --long-ms 10,100,1000            Target durations; fixed work calibrated with Events
  --trials N                      Measured trials per case/target (default: 10)
  --warmup N                      Warmup launches (default: 3, minimum: 1)
  --device N                      CUDA visible device ordinal (default: 0)
  --blocks N                      A grid size (default: 4 * visible SM count)
  --threads N                     A block size (default: 256)
  --invalidate-fraction F         Delay after A start is observed (default: 0.1)
  --invalidate-ms MS              Absolute delay; overrides fraction
  --action destroy|wait-then-destroy|both   Both alternates a pair at fixed work
  --estimate-start                Calibrate B GPU clock against CPU brackets
  --cpu-a N --cpu-b N --cpu-control N  Optional affinity; use distinct physical cores
  --module PATH                   Device-only cubin (default: build-time path)
  --output PATH                   CSV, with metadata and nvidia-smi sidecars
  --help                         No CUDA initialization
Existing output files are never overwritten. All host timestamps are RAW monotonic ns.
)";
}

double number(const std::string& value) {
    std::size_t used = 0;
    double x = std::stod(value, &used);
    if (used != value.size() || !std::isfinite(x))
        throw std::runtime_error("Expected a finite number: " + value);
    return x;
}

int integer(const std::string& value) {
    double x = number(value);
    if (x < 0 || x > std::numeric_limits<int>::max() || std::floor(x) != x)
        throw std::runtime_error("Expected a nonnegative integer: " + value);
    return static_cast<int>(x);
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") { usage(); std::exit(0); }
        if (key == "--estimate-start") { o.estimate_start = true; continue; }
        if (i + 1 == argc) throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--mode") o.mode = value;
        else if (key == "--action") o.action = value;
        else if (key == "--output") o.output = value;
        else if (key == "--module") o.module = value;
        else if (key == "--device") o.device = integer(value);
        else if (key == "--trials") o.trials = integer(value);
        else if (key == "--warmup") o.warmup = integer(value);
        else if (key == "--threads") o.threads = integer(value);
        else if (key == "--blocks") o.blocks = integer(value);
        else if (key == "--cpu-a") o.cpu_a = integer(value);
        else if (key == "--cpu-b") o.cpu_b = integer(value);
        else if (key == "--cpu-control") o.cpu_control = integer(value);
        else if (key == "--invalidate-fraction") o.invalidate_fraction = number(value);
        else if (key == "--invalidate-ms") {
            o.invalidate_ms = number(value);
            if (o.invalidate_ms < 0) throw std::runtime_error("--invalidate-ms must be >= 0");
        } else if (key == "--long-ms") {
            o.durations.clear();
            if (value.empty() || value.back() == ',') throw std::runtime_error("Empty duration");
            std::istringstream input(value);
            std::string part;
            while (std::getline(input, part, ',')) o.durations.push_back(number(part));
        } else throw std::runtime_error("Unknown option: " + key);
    }
    if (o.mode != "all" && o.mode != "standby" && o.mode != "takeover" && o.mode != "live-handoff")
        throw std::runtime_error("--mode must be all, standby, takeover, or live-handoff");
    if (o.mode == "live-handoff") o.estimate_start = true;
    if (o.action != "destroy" && o.action != "wait-then-destroy" && o.action != "both")
        throw std::runtime_error("--action must be destroy, wait-then-destroy, or both");
    if (o.trials < 1 || o.warmup < 1 || o.threads < 32 || o.threads > 1024 || o.threads % 32)
        throw std::runtime_error("trials/warmup >= 1; threads must be a multiple of 32 in [32,1024]");
    if (!(o.invalidate_fraction >= 0 && o.invalidate_fraction < 1))
        throw std::runtime_error("--invalidate-fraction must be in [0,1)");
    for (double ms : o.durations) {
        if (!(ms > 0 && ms <= 60000)) throw std::runtime_error("--long-ms must be in (0,60000]");
        if (o.invalidate_ms >= ms) throw std::runtime_error("invalidate-ms must be below every target duration");
    }
    const int cpus[] = {o.cpu_a, o.cpu_b, o.cpu_control};
    for (int i = 0; i < 3; ++i) {
        if (cpus[i] >= CPU_SETSIZE) throw std::runtime_error("CPU index exceeds CPU_SETSIZE");
        for (int j = i + 1; j < 3; ++j)
            if (cpus[i] >= 0 && cpus[i] == cpus[j])
                throw std::runtime_error("Use distinct CPUs for the three host threads");
    }
    return o;
}

std::string csv_quote(const std::string& input) {
    std::string result = "\"";
    for (char c : input) { if (c == '"') result += '"'; result += c; }
    return result + '"';
}

struct Sample {
    std::string experiment, condition, status = "ok";
    int trial = 0, order = 0, valid = 1, a_pending = -1, destroy_result = -1;
    int cpu_a = -1, cpu_b = -1, cpu_control = -1;
    double target_ms = 0, reference_ms = kNaN, a_event_ms = kNaN, a_gflops = kNaN;
    double b_event_ms = kNaN, invalidate_delay_ms = kNaN;
    std::uint64_t iterations = 0, a_start_gpu_ns = 0, b_start_gpu_ns = 0, b_end_gpu_ns = 0;
    Ns a_launch = 0, a_launch_return = 0, a_start_observed = 0;
    Ns invalidate = 0, invalidate_seen = 0, a_query_begin = 0, a_query_end = 0;
    Ns destroy_begin = 0, destroy_return = 0;
    Ns b_ready = 0, b_trigger_seen = 0, b_launch = 0, b_launch_return = 0, b_complete = 0;
    double b_start_est = kNaN, b_start_low = kNaN, b_start_high = kNaN;
    double clock_bracket_ns = kNaN, clock_offset_change_ns = kNaN;
    int start_est_valid = 0;
    int a_pending_after_b = -1, a_alive_after_b = -1;
    Ns a_post_b_query_begin = 0, a_post_b_query_end = 0, a_complete_observed = 0;
    std::uint64_t a_launch_seq = 0, b_launch_seq = 0;
};

class Output {
    std::ofstream samples_, metadata_;
public:
    std::filesystem::path smi_path;
    explicit Output(const Options& o) {
        std::filesystem::path path(o.output), meta(path), smi(path);
        meta.replace_extension(".metadata.csv");
        smi.replace_extension(".nvidia-smi.txt");
        if (path == meta || path == smi) throw std::runtime_error("Output path collides with a sidecar");
        for (const auto& p : {path, meta, smi})
            if (std::filesystem::exists(p)) throw std::runtime_error("Output exists; choose a new --output: " + p.string());
        if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
        samples_.exceptions(std::ios::failbit | std::ios::badbit);
        metadata_.exceptions(std::ios::failbit | std::ios::badbit);
        samples_.open(path);
        metadata_.open(meta);
        smi_path = smi;
        metadata_ << "key,value\n";
        samples_ << "experiment,condition,trial,order,status,valid,target_ms,iterations,reference_a_ms,"
                    "a_event_ms,a_effective_gflops,b_event_ms,invalidate_delay_ms,a_pending_before_destroy,"
                    "destroy_result,cpu_a,cpu_b,cpu_control,t_A_launch_ns,t_A_launch_return_ns,"
                    "t_A_start_observed_ns,A_start_gpu_ns,t_invalidate_ns,t_invalidate_seen_ns,"
                    "t_A_query_begin_ns,t_A_query_end_ns,t_destroy_begin_ns,t_destroy_return_ns,"
                    "t_B_ready_ns,t_B_trigger_seen_ns,t_B_launch_ns,t_B_launch_return_ns,t_B_complete_ns,"
                    "invalidate_dispatch_us,destroy_latency_us,invalidate_to_destroy_return_us,"
                    "handoff_us,b_launch_api_us,b_request_us,b_takeover_us,B_start_gpu_ns,B_end_gpu_ns,"
                    "t_B_start_est_ns,t_B_start_lower_ns,t_B_start_upper_ns,clock_bracket_ns,"
                    "clock_offset_change_ns,start_est_valid,invalidate_to_b_launch_us,"
                    "invalidate_to_b_gpu_start_est_us,invalidate_to_b_gpu_start_lower_us,"
                    "invalidate_to_b_gpu_start_upper_us,a_pending_after_invalidate,"
                    "a_event_pending_after_b_complete,a_context_alive_after_b_complete,"
                    "t_A_post_B_query_begin_ns,t_A_post_B_query_end_ns,t_A_complete_observed_ns,"
                    "a_launch_seq,b_launch_seq\n";
        samples_.flush();
    }
    template<class T> void meta(const std::string& key, const T& value) {
        std::ostringstream s; s << std::setprecision(17) << value;
        metadata_ << csv_quote(key) << ',' << csv_quote(s.str()) << '\n';
        metadata_.flush();
    }
    void row(const Sample& s) {
        auto delta_us = [](Ns end, Ns begin) { return end && begin ? (end - begin) / 1000.0 : kNaN; };
        samples_ << std::setprecision(17)
            << s.experiment << ',' << s.condition << ',' << s.trial << ',' << s.order << ','
            << s.status << ',' << s.valid << ',' << s.target_ms << ',' << s.iterations << ','
            << s.reference_ms << ',' << s.a_event_ms << ',' << s.a_gflops << ',' << s.b_event_ms << ','
            << s.invalidate_delay_ms << ',' << s.a_pending << ',' << s.destroy_result << ','
            << s.cpu_a << ',' << s.cpu_b << ',' << s.cpu_control << ','
            << s.a_launch << ',' << s.a_launch_return << ',' << s.a_start_observed << ',' << s.a_start_gpu_ns << ','
            << s.invalidate << ',' << s.invalidate_seen << ',' << s.a_query_begin << ',' << s.a_query_end << ','
            << s.destroy_begin << ',' << s.destroy_return << ',' << s.b_ready << ',' << s.b_trigger_seen << ','
            << s.b_launch << ',' << s.b_launch_return << ',' << s.b_complete << ','
            << delta_us(s.invalidate_seen, s.invalidate) << ',' << delta_us(s.destroy_return, s.destroy_begin) << ','
            << delta_us(s.destroy_return, s.invalidate) << ',' << delta_us(s.b_launch, s.destroy_return) << ','
            << delta_us(s.b_launch_return, s.b_launch) << ',' << delta_us(s.b_complete, s.b_launch) << ','
            << delta_us(s.b_complete, s.invalidate) << ',' << s.b_start_gpu_ns << ',' << s.b_end_gpu_ns << ','
            << s.b_start_est << ',' << s.b_start_low << ',' << s.b_start_high << ',' << s.clock_bracket_ns << ','
            << s.clock_offset_change_ns << ',' << s.start_est_valid << ','
            << delta_us(s.b_launch, s.invalidate) << ','
            << (s.start_est_valid && s.invalidate ? (s.b_start_est - s.invalidate) / 1000.0 : kNaN) << ','
            << (s.start_est_valid && s.invalidate ? (s.b_start_low - s.invalidate) / 1000.0 : kNaN) << ','
            << (s.start_est_valid && s.invalidate ? (s.b_start_high - s.invalidate) / 1000.0 : kNaN) << ','
            << s.a_pending << ',' << s.a_pending_after_b << ',' << s.a_alive_after_b << ','
            << s.a_post_b_query_begin << ',' << s.a_post_b_query_end << ',' << s.a_complete_observed << ','
            << s.a_launch_seq << ',' << s.b_launch_seq << '\n';
        samples_.flush();
    }
};

class GpuState {
    CUcontext ctx_ = nullptr;
    CUmodule module_ = nullptr;
    CUstream stream_ = nullptr;
    CUevent begin_ = nullptr, end_ = nullptr;
    CUfunction compute_ = nullptr, tiny_ = nullptr;
    CUdeviceptr output_ = 0, tiny_output_ = 0, marker_device_ = 0;
    StartMarker* marker_ = nullptr;
    int blocks_ = 0, threads_ = 0;
    unsigned token_ = 0;
    // Counts every launch on the owner thread, including warmup/calibration,
    // across context recreation. Used to correlate an optional Nsight trace.
    std::uint64_t launch_sequence_ = 0;
public:
    ~GpuState() {
        if (ctx_) {
            CUresult result = destroy();
            if (result != CUDA_SUCCESS) std::cerr << "Context cleanup: " << cuda_error(result) << '\n';
        }
    }
    bool exists() const { return ctx_ != nullptr; }
    std::uint64_t launch_count() const { return launch_sequence_; }
    void create(CUdevice device, const Options& o, bool is_a) {
        if (ctx_) return;
#if CUDA_VERSION >= 13000
        CU(cuCtxCreate(&ctx_, nullptr, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, device));
#else
        CU(cuCtxCreate(&ctx_, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, device));
#endif
        blocks_ = o.blocks; threads_ = o.threads;
        CU(cuModuleLoad(&module_, o.module.c_str()));
        CU(cuModuleGetFunction(&tiny_, module_, "tiny_kernel"));
        CU(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING));
        CU(cuEventCreate(&begin_, CU_EVENT_DEFAULT));
        CU(cuEventCreate(&end_, CU_EVENT_DEFAULT));
        CU(cuMemAlloc(&tiny_output_, sizeof(TinyResult)));
        if (is_a) {
            CU(cuModuleGetFunction(&compute_, module_, "compute_kernel"));
            CU(cuMemAlloc(&output_, static_cast<std::size_t>(blocks_) * threads_ * sizeof(float)));
            void* host = nullptr;
            CU(cuMemHostAlloc(&host, sizeof(StartMarker), CU_MEMHOSTALLOC_DEVICEMAP));
            marker_ = new (host) StartMarker{};
            CU(cuMemHostGetDevicePointer(&marker_device_, host, 0));
        }
    }
    CUresult destroy(Ns* begin = nullptr, Ns* end = nullptr) {
        // Destroy the live context directly: no synchronize/free/unload first.
        // All handles, including A's Events and mapped memory, become unusable.
        CUcontext old = std::exchange(ctx_, nullptr);
        if (!old) return CUDA_SUCCESS;
        module_ = nullptr; stream_ = nullptr; begin_ = nullptr; end_ = nullptr;
        compute_ = nullptr; tiny_ = nullptr; output_ = tiny_output_ = marker_device_ = 0; marker_ = nullptr;
        if (begin) *begin = now_ns();
        CUresult result = cuCtxDestroy(old);
        if (end) *end = now_ns();
        return result;
    }
    void close() { CU(destroy()); }
    void launch_long(std::uint64_t iterations, bool marker, Sample& s) {
        CUdeviceptr probe = marker ? marker_device_ : 0;
        if (marker)
            cuda::atomic_ref<unsigned, cuda::thread_scope_system>(marker_->started)
                .store(0, cuda::memory_order_relaxed);
        void* args[] = {&output_, &iterations, &probe};
        CU(cuEventRecord(begin_, stream_));
        s.cpu_a = sched_getcpu();
        s.a_launch_seq = ++launch_sequence_;
        s.a_launch = now_ns();
        CUresult result = cuLaunchKernel(compute_, blocks_, 1, 1, threads_, 1, 1, 0, stream_, args, nullptr);
        s.a_launch_return = now_ns();
        CU(result);
        CU(cuEventRecord(end_, stream_));
    }
    double run_long(std::uint64_t iterations, bool marker = false, Sample* sample = nullptr) {
        Sample scratch;
        Sample& s = sample ? *sample : scratch;
        launch_long(iterations, marker, s);
        return finish_long(iterations, s);
    }
    double finish_long(std::uint64_t iterations, Sample& s) {
        CU(cuEventSynchronize(end_));
        s.a_complete_observed = now_ns();
        float ms = 0;
        CU(cuEventElapsedTime(&ms, begin_, end_));
        if (!(ms > 0 && std::isfinite(ms))) throw std::runtime_error("Invalid A Event duration");
        float value = 0;
        CU(cuMemcpyDtoH(&value, output_, sizeof(value)));
        if (!std::isfinite(value) || value <= 0) throw std::runtime_error("Invalid A compute output");
        s.a_event_ms = ms;
        s.a_gflops = static_cast<double>(iterations) * blocks_ * threads_ * (2 * kFmasPerIteration) / (ms * 1e6);
        return ms;
    }
    void wait_started(Sample& s, double target_ms) {
        const Ns deadline = now_ns() + static_cast<Ns>(std::max(5000.0, target_ms * 10) * 1e6);
        while (!cuda::atomic_ref<unsigned, cuda::thread_scope_system>(marker_->started)
                    .load(cuda::memory_order_acquire)) {
            if (now_ns() > deadline) throw std::runtime_error("A in-kernel start marker timeout");
            relax_cpu();
        }
        s.a_start_observed = now_ns();
        s.a_start_gpu_ns = marker_->gpu_ns;
    }
    CUresult query_done() { return cuEventQuery(end_); }
    void wait_done() { CU(cuEventSynchronize(end_)); }
    void run_tiny(Sample& s, std::atomic<bool>* completed = nullptr) {
        unsigned token = ++token_;
        void* args[] = {&tiny_output_, &token};
        s.cpu_b = sched_getcpu();
        CU(cuEventRecord(begin_, stream_));
        s.b_launch_seq = ++launch_sequence_;
        s.b_launch = now_ns();
        CUresult result = cuLaunchKernel(tiny_, 1, 1, 1, 1, 1, 1, 0, stream_, args, nullptr);
        s.b_launch_return = now_ns();
        CU(result);
        CU(cuEventRecord(end_, stream_));
        result = cuEventSynchronize(end_);
        s.b_complete = now_ns();
        CU(result);
        // Let A query its own Event immediately, before B's elapsed-time query
        // or DtoH copy can add observation delay. No CUDA handle crosses threads.
        if (completed) completed->store(true, std::memory_order_release);
        float ms = 0;
        CU(cuEventElapsedTime(&ms, begin_, end_));
        s.b_event_ms = ms;
        TinyResult host{};
        CU(cuMemcpyDtoH(&host, tiny_output_, sizeof(host)));
        if (host.value != (token ^ kTokenMask) || host.end_gpu_ns < host.start_gpu_ns)
            throw std::runtime_error("Invalid B output/timestamps");
        s.b_start_gpu_ns = host.start_gpu_ns;
        s.b_end_gpu_ns = host.end_gpu_ns;
    }
};

// The two owner threads live for the entire run. Each context is created, used,
// and destroyed on its owner. The controller never makes any context current.
class Worker {
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<std::function<void(GpuState&)>> queue_;
    bool stop_ = false;
    std::thread thread_;
public:
    Worker() : thread_([this] {
        GpuState state;
        for (;;) {
            std::function<void(GpuState&)> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
                if (queue_.empty() && stop_) return;
                task = std::move(queue_.front()); queue_.pop_front();
            }
            task(state);
        }
    }) {}
    ~Worker() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
        cv_.notify_one();
        thread_.join();
    }
    template<class F> auto submit(F fn) {
        using R = std::invoke_result_t<F, GpuState&>;
        auto task = std::make_shared<std::packaged_task<R(GpuState&)>>(std::move(fn));
        auto future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.emplace_back([task](GpuState& s) { (*task)(s); });
        }
        cv_.notify_one();
        return future;
    }
};

struct Calibration { std::uint64_t iterations; double ms; };
Calibration calibrate(GpuState& a, double target_ms, int warmup) {
    std::uint64_t iterations = 128;
    for (int i = 0; i < warmup; ++i) a.run_long(iterations);
    double ms = a.run_long(iterations);
    while (ms < 2 && iterations < (1ULL << 32)) {
        iterations *= 2;
        ms = a.run_long(iterations);
    }
    for (int i = 0; i < 3; ++i) {
        const double next = std::round(iterations * target_ms / ms);
        if (!std::isfinite(next) || next > (1ULL << 40)) throw std::runtime_error("Calibration iteration overflow");
        iterations = static_cast<std::uint64_t>(std::max(1.0, next));
        ms = a.run_long(iterations);
        if (std::abs(ms / target_ms - 1) < 0.05) break;
    }
    return {iterations, ms};
}

struct ClockAnchor {
    Ns before = 0, after = 0;
    std::uint64_t gpu_ns = 0;
    long double low() const { return static_cast<long double>(before) - gpu_ns; }
    long double high() const { return static_cast<long double>(after) - gpu_ns; }
};
ClockAnchor clock_anchor(GpuState& b) {
    ClockAnchor best;
    for (int i = 0; i < 7; ++i) {
        Sample s; b.run_tiny(s);
        if (!best.after || s.b_complete - s.b_launch < best.after - best.before)
            best = {s.b_launch, s.b_complete, s.b_start_gpu_ns};
    }
    return best;
}

void estimate_start(Sample& s, const ClockAnchor& pre, const ClockAnchor& post) {
    // Conditional envelope: unit-rate CPU/GPU clocks, offset contained in the
    // pre/post brackets. No subtraction of raw CPU and GPU timestamps as latency.
    const long double low = std::min(pre.low(), post.low());
    const long double high = std::max(pre.high(), post.high());
    s.clock_bracket_ns = static_cast<double>(std::max(pre.after - pre.before, post.after - post.before));
    s.clock_offset_change_ns = static_cast<double>((post.low() + post.high() - pre.low() - pre.high()) / 2);
    const long double raw_low = static_cast<long double>(s.b_start_gpu_ns) + low;
    const long double raw_high = static_cast<long double>(s.b_start_gpu_ns) + high;
    const long double bounded_low = std::max(raw_low, static_cast<long double>(s.b_launch));
    const long double bounded_high = std::min(raw_high, static_cast<long double>(s.b_complete));
    // Nonoverlapping offset brackets or nonmonotonic device samples invalidate
    // the constant-offset model; retain raw observations, leave estimate NaN.
    if (pre.gpu_ns <= s.b_start_gpu_ns && s.b_start_gpu_ns <= post.gpu_ns &&
        std::max(pre.low(), post.low()) <= std::min(pre.high(), post.high()) && bounded_low <= bounded_high) {
        s.b_start_low = static_cast<double>(bounded_low);
        s.b_start_high = static_cast<double>(bounded_high);
        s.b_start_est = static_cast<double>((bounded_low + bounded_high) / 2);
        s.start_est_valid = 1;
    }
}

struct Signals {
    std::atomic<bool> b_armed{false}, a_started{false}, invalidate{false}, b_go{false}, b_done{false}, cancelled{false};
};

bool await_flag(const std::atomic<bool>& flag, const Signals& signals) {
    while (!flag.load(std::memory_order_acquire)) {
        if (signals.cancelled.load(std::memory_order_acquire)) return false;
        relax_cpu();
    }
    return !signals.cancelled.load(std::memory_order_acquire);
}

void takeover(Worker& a, Worker& b, const Options& o, Sample& s) {
    Signals signals;
    const bool idle = o.action == "idle";
    const bool keep_a = idle || o.action == "live-handoff";
    auto b_future = b.submit([&](GpuState& state) {
        try {
            s.b_ready = now_ns();
            signals.b_armed.store(true, std::memory_order_release);
            if (!await_flag(signals.b_go, signals)) return;
            s.b_trigger_seen = now_ns();
            state.run_tiny(s, &signals.b_done);
        } catch (...) { signals.cancelled.store(true, std::memory_order_release); throw; }
    });
    auto a_future = a.submit([&](GpuState& state) {
        try {
            if (!await_flag(signals.b_armed, signals)) return;
            s.cpu_a = sched_getcpu();
            if (!idle) {
                state.launch_long(s.iterations, true, s);
                state.wait_started(s, s.target_ms);
            }
            signals.a_started.store(true, std::memory_order_release);
            if (!await_flag(signals.invalidate, signals)) return;
            s.invalidate_seen = now_ns();
            s.a_query_begin = now_ns();
            CUresult query = state.query_done();
            s.a_query_end = now_ns();
            if (query != CUDA_SUCCESS && query != CUDA_ERROR_NOT_READY) CU(query);
            s.a_pending = query == CUDA_ERROR_NOT_READY;
            if (keep_a) {
                if (!await_flag(signals.b_done, signals)) return;
                s.a_post_b_query_begin = now_ns();
                CUresult post = state.query_done();
                s.a_post_b_query_end = now_ns();
                if (post != CUDA_SUCCESS && post != CUDA_ERROR_NOT_READY) CU(post);
                s.a_pending_after_b = post == CUDA_ERROR_NOT_READY;
                s.a_alive_after_b = state.exists();
                // A finishes naturally; its context remains current and alive.
                if (!idle) state.finish_long(s.iterations, s);
            } else {
                if (o.action == "wait-then-destroy") state.wait_done();
                CUresult result = state.destroy(&s.destroy_begin, &s.destroy_return);
                s.destroy_result = static_cast<int>(result);
                if (result != CUDA_SUCCESS) CU(result);
                s.a_alive_after_b = 0;
                signals.b_go.store(true, std::memory_order_release);
            }
        } catch (...) { signals.cancelled.store(true, std::memory_order_release); throw; }
    });
    std::exception_ptr error;
    try {
        if (await_flag(signals.a_started, signals)) {
            const Ns deadline = idle ? 0 : s.a_start_observed + static_cast<Ns>(s.invalidate_delay_ms * 1e6);
            while (now_ns() < deadline && !signals.cancelled.load(std::memory_order_acquire)) relax_cpu();
            s.cpu_control = sched_getcpu();
            s.invalidate = now_ns();
            signals.invalidate.store(true, std::memory_order_release);
            // Live/idle launch is not gated by A's Event query or A's thread.
            if (keep_a) signals.b_go.store(true, std::memory_order_release);
        }
    } catch (...) { signals.cancelled.store(true, std::memory_order_release); error = std::current_exception(); }
    // Always drain both futures before releasing stack data, including on error.
    try { a_future.get(); } catch (...) { error = std::current_exception(); }
    try { b_future.get(); } catch (...) { if (!error) error = std::current_exception(); }
    if (error) std::rethrow_exception(error);
    if (!idle && !s.a_pending) { s.valid = 0; s.status = "a_already_complete"; }
    if (idle && s.a_pending) throw std::runtime_error("A was not idle in the idle baseline");
    if (!(s.invalidate <= s.b_launch && s.b_launch <= s.b_complete) ||
        (!idle && !(s.b_ready <= s.a_launch && s.a_start_observed <= s.invalidate)) ||
        (!keep_a && !(s.invalidate <= s.destroy_begin && s.destroy_begin <= s.destroy_return && s.destroy_return <= s.b_launch)) ||
        (keep_a && !(s.destroy_begin == 0 && s.destroy_return == 0 && s.a_alive_after_b == 1 &&
                     s.b_complete <= s.a_post_b_query_begin && s.a_post_b_query_begin <= s.a_post_b_query_end)))
        throw std::runtime_error("Host timestamp ordering violated");
}

int attribute(CUdevice device, CUdevice_attribute attr) {
    int value = 0; CU(cuDeviceGetAttribute(&value, attr, device)); return value;
}

void metadata_before_cuda(Output& out, const Options& o, int argc, char** argv) {
    out.meta("schema_version", 2);
    out.meta("run_status", "initializing");
    out.meta("toolkit_version", BENCH_TOOLKIT_VERSION);
    out.meta("cuda_header_version", CUDA_VERSION);
    out.meta("cubin_sm", BENCH_CUDA_ARCH);
    out.meta("module_path", std::filesystem::absolute(o.module).string());
    out.meta("host_timer", "CLOCK_MONOTONIC_RAW");
    timespec resolution{};
    if (clock_getres(CLOCK_MONOTONIC_RAW, &resolution) == 0)
        out.meta("host_timer_resolution_ns", resolution.tv_sec * 1000000000LL + resolution.tv_nsec);
    out.meta("run_start_host_ns", now_ns());
    out.meta("process_pid", getpid());
    out.meta("thread_control_tid", syscall(SYS_gettid));
    const std::time_t wall = std::time(nullptr);
    std::tm utc{}; gmtime_r(&wall, &utc);
    std::ostringstream date; date << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    out.meta("run_start_utc", date.str());
    utsname os{};
    if (uname(&os) == 0) out.meta("host_os", std::string(os.sysname) + " " + os.release + " " + os.machine);
    std::ifstream proc("/proc/driver/nvidia/version");
    std::ostringstream driver; driver << proc.rdbuf();
    out.meta("nvidia_kernel_driver", driver.str().empty() ? "unavailable" : driver.str());
    for (int i = 0; i < argc; ++i) out.meta("argv_" + std::to_string(i), argv[i]);
    for (const char* key : {"CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING", "CUDA_MODULE_LOADING",
                            "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"}) {
        const char* value = std::getenv(key); out.meta(key, value ? value : "<unset>");
    }
    out.meta("mps_active", "not_determined_by_this_program; verify independently");
    // Fixed read-only command, outside measurements. Ordinals here are physical
    // nvidia-smi indices; correlate with the Driver PCI bus ID / UUID below.
    std::ofstream smi(out.smi_path);
    const char* command = "nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,mig.mode.current,compute_mode --format=csv 2>&1";
    if (FILE* pipe = popen(command, "r")) {
        char buffer[1024];
        while (fgets(buffer, sizeof(buffer), pipe)) smi << buffer;
        out.meta("nvidia_smi_exit_status", pclose(pipe));
    } else { smi << "nvidia-smi unavailable\n"; out.meta("nvidia_smi_exit_status", "unavailable"); }
}

void experiments(Options& o, Output& out) {
    int driver = 0, count = 0;
    // This version query does not require initialization; preserve it even when
    // this execution environment cannot access a GPU.
    CU(cuDriverGetVersion(&driver));
    out.meta("cuda_driver_api_version", driver);
    out.meta("cuda_driver_api_version_display", std::to_string(driver / 1000) + "." + std::to_string((driver % 1000) / 10));
    CU(cuInit(0));
    CU(cuDeviceGetCount(&count));
    out.meta("visible_device_count", count);
    if (o.device >= count) throw std::runtime_error("Requested device is not visible");
    CUdevice device; CU(cuDeviceGet(&device, o.device));
    char name[256]{}, pci[64]{}; CUuuid uuid{}; std::size_t memory = 0;
    CU(cuDeviceGetName(name, sizeof(name), device));
    CU(cuDeviceGetPCIBusId(pci, sizeof(pci), device));
    CU(cuDeviceGetUuid_v2(&uuid, device)); CU(cuDeviceTotalMem(&memory, device));
    std::ostringstream uuid_text;
    for (unsigned i = 0; i < sizeof(uuid.bytes); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) uuid_text << '-';
        uuid_text << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(uuid.bytes[i]));
    }
    out.meta("gpu_name", name); out.meta("gpu_uuid", uuid_text.str());
    out.meta("gpu_pci_bus_id", pci); out.meta("gpu_total_memory_bytes", memory);
    out.meta("selected_visible_ordinal", o.device);
    const int major = attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    const int minor = attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    const int sms = attribute(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    const int mode = attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE);
    out.meta("gpu_compute_capability", std::to_string(major) + "." + std::to_string(minor));
    out.meta("gpu_sm_count", sms); out.meta("gpu_compute_mode", mode);
    out.meta("gpu_compute_preemption_supported", attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED));
    out.meta("gpu_kernel_exec_timeout", attribute(device, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT));
    out.meta("context_flags", "CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST");
    out.meta("stream_flags", "CU_STREAM_NON_BLOCKING");
    if (!o.blocks) o.blocks = 4 * sms;
    out.meta("a_blocks", o.blocks); out.meta("a_threads", o.threads);
    out.meta("fma_per_thread_per_iteration", kFmasPerIteration);
    out.meta("cpu_a_requested", o.cpu_a); out.meta("cpu_b_requested", o.cpu_b); out.meta("cpu_control_requested", o.cpu_control);
    if (major * 10 + minor != BENCH_CUDA_ARCH)
        throw std::runtime_error("GPU SM differs from cubin target; rebuild with -DBENCH_CUDA_ARCH=<SM>");
    if (mode != CU_COMPUTEMODE_DEFAULT) throw std::runtime_error("Two contexts require default compute mode for this experiment");
    if (!attribute(device, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY))
        throw std::runtime_error("Mapped host memory is required for the A start marker");
    if (o.threads > attribute(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) ||
        o.blocks > attribute(device, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X))
        throw std::runtime_error("Requested kernel launch dimensions exceed device limits");
    if (const char* blocking = std::getenv("CUDA_LAUNCH_BLOCKING"))
        if (std::string(blocking) != "0") throw std::runtime_error("Unset CUDA_LAUNCH_BLOCKING for asynchronous A launch");
    std::cerr << "GPU: " << name << ", Driver API " << driver << ", Toolkit " << BENCH_TOOLKIT_VERSION << '\n';
    Worker a, b;
    out.meta("thread_a_tid", a.submit([&](GpuState&) { pin_cpu(o.cpu_a); return syscall(SYS_gettid); }).get());
    out.meta("thread_b_tid", b.submit([&](GpuState&) { pin_cpu(o.cpu_b); return syscall(SYS_gettid); }).get());
    // Pin controller after creating workers so they do not inherit its affinity.
    pin_cpu(o.cpu_control);
    for (double target : o.durations) {
        b.submit([](GpuState& g) { g.close(); }).get();
        Calibration cal = a.submit([&](GpuState& g) {
            g.create(device, o, true); return calibrate(g, target, o.warmup);
        }).get();
        auto sample = [&](const std::string& experiment, const std::string& condition, int trial) {
            Sample s; s.experiment = experiment; s.condition = condition; s.trial = trial;
            s.target_ms = target; s.iterations = cal.iterations; s.reference_ms = cal.ms;
            return s;
        };
        Sample calibration = sample("calibration", "a_only", 0);
        calibration.a_event_ms = cal.ms;
        out.row(calibration);
        std::cerr << "Target " << target << " ms: " << cal.iterations << " iterations, reference " << cal.ms << " ms\n";
        auto warm_b = [&](GpuState& g) {
            if (!g.exists()) {
                g.create(device, o, false);
                for (int j = 0; j < o.warmup; ++j) { Sample scratch; g.run_tiny(scratch); }
            }
        };
        if (o.mode == "live-handoff") {
            b.submit(warm_b).get();
            const char* conditions[] = {"live-handoff", "serial-destroy", "idle"};
            for (int trial = 0; trial < o.trials; ++trial) {
                // Observed execution-time drift can halve a short kernel's
                // duration. Recalibrate once per triplet, then hold work fixed
                // across its three conditions. The device code is unchanged.
                cal = a.submit([&](GpuState& g) {
                    g.create(device, o, true); return calibrate(g, target, o.warmup);
                }).get();
                Sample triplet_cal = sample("calibration", "a_with_idle_b", trial);
                triplet_cal.a_event_ms = cal.ms;
                out.row(triplet_cal);
                for (int order = 0; order < 3; ++order) {
                    const std::string condition = conditions[(trial + order) % 3];
                    // Refresh the natural-duration reference immediately before
                    // each condition, at identical fixed work and with B idle.
                    const double reference = a.submit([&](GpuState& g) {
                        g.create(device, o, true);
                        for (int j = 0; j < o.warmup; ++j) g.run_long(std::min<std::uint64_t>(cal.iterations, 1024), true);
                        return g.run_long(cal.iterations, true);
                    }).get();
                    ClockAnchor pre = b.submit([](GpuState& g) { return clock_anchor(g); }).get();
                    Sample s = sample("live-handoff", condition, trial);
                    s.order = order;
                    s.reference_ms = reference;
                    s.invalidate_delay_ms = condition == "idle" ? 0 :
                        o.invalidate_ms >= 0 ? o.invalidate_ms : reference * o.invalidate_fraction;
                    Options action = o;
                    action.action = condition == "serial-destroy" ? "destroy" : condition;
                    try { takeover(a, b, action, s); }
                    catch (...) { s.valid = 0; s.status = "error"; out.row(s); throw; }
                    // Both workers have finished their measured work before
                    // any post-calibration kernels are submitted.
                    ClockAnchor post = b.submit([](GpuState& g) { return clock_anchor(g); }).get();
                    estimate_start(s, pre, post);
                    out.row(s);
                    std::cerr << condition << ", target " << target << " ms, trial " << trial
                              << ": B complete " << (s.b_complete - s.invalidate) / 1000.0
                              << " us, A Event pending after B=" << s.a_pending_after_b << ", " << s.status << '\n';
                }
            }
            continue;
        }
        if (o.mode != "takeover") {
            for (int trial = 0; trial < o.trials; ++trial) {
                // Alternate AB/BA paired trials to reduce systematic order bias.
                for (int order = 0; order < 2; ++order) {
                    const bool idle_b = ((trial + order) % 2) == 1;
                    b.submit([&](GpuState& g) { if (idle_b) warm_b(g); else g.close(); }).get();
                    Sample s = sample("standby", idle_b ? "a_with_idle_b" : "a_only", trial);
                    s.order = order;
                    a.submit([&](GpuState& g) {
                        for (int j = 0; j < o.warmup; ++j) g.run_long(std::min<std::uint64_t>(cal.iterations, 1024));
                        g.run_long(cal.iterations, false, &s);
                    }).get();
                    out.row(s);
                }
            }
        }
        if (o.mode != "standby") {
            b.submit(warm_b).get();
            for (int trial = 0; trial < o.trials; ++trial) {
                for (int order = 0; order < (o.action == "both" ? 2 : 1); ++order) {
                    Options action = o;
                    if (o.action == "both")
                        action.action = (trial + order) % 2 ? "wait-then-destroy" : "destroy";
                    a.submit([&](GpuState& g) {
                        g.create(device, o, true);
                        for (int j = 0; j < o.warmup; ++j) g.run_long(std::min<std::uint64_t>(cal.iterations, 1024), true);
                    }).get();
                    Sample idle = sample("takeover", "b_with_idle_a", trial);
                    idle.order = order;
                    ClockAnchor pre = b.submit([&](GpuState& g) {
                        g.run_tiny(idle);
                        return o.estimate_start ? clock_anchor(g) : ClockAnchor{};
                    }).get();
                    out.row(idle);
                    Sample s = sample("takeover", action.action, trial);
                    s.order = order;
                    s.invalidate_delay_ms = o.invalidate_ms >= 0 ? o.invalidate_ms : target * o.invalidate_fraction;
                    try { takeover(a, b, action, s); }
                    catch (...) { s.valid = 0; s.status = "error"; out.row(s); throw; }
                    if (o.estimate_start) {
                        ClockAnchor post = b.submit([](GpuState& g) { return clock_anchor(g); }).get();
                        estimate_start(s, pre, post);
                    }
                    out.row(s);
                    std::cerr << "Takeover " << action.action << ", " << target << " ms, trial " << trial << ": "
                              << (s.b_complete - s.invalidate) / 1000.0 << " us, " << s.status << '\n';
                }
            }
        }
    }
    out.meta("thread_a_launch_count", a.submit([](GpuState& g) { return g.launch_count(); }).get());
    out.meta("thread_b_launch_count", b.submit([](GpuState& g) { return g.launch_count(); }).get());
    a.submit([](GpuState& g) { g.close(); }).get();
    b.submit([](GpuState& g) { g.close(); }).get();
}

int main(int argc, char** argv) {
    try {
        Options o = parse(argc, argv);
        Output out(o);
        try {
            metadata_before_cuda(out, o, argc, argv);
            experiments(o, out);
            out.meta("run_status", "completed");
        } catch (const std::exception& e) {
            out.meta("run_status", "failed"); out.meta("error", e.what()); throw;
        }
        std::cerr << "CSV: " << std::filesystem::absolute(o.output) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
