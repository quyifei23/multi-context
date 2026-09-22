#pragma once
#include "userd_observer.h"
#include <rm_control.h>

// Reuse the original owner threads, kernels and bridge lifecycle discovery.
// Snapshots are idle. Optional progress reads surround one original tiny launch.
// Neither mode performs rewind or establishes a sentinel/entry mapping.
[[noreturn]] void run_userd_mapping(const Options& o, Output& out) {
    const auto directory = std::filesystem::path(o.output + ".userd");
    const bool progress = o.mode == "userd-progress";
    try {
        if (o.device != 0) throw std::runtime_error("userd-map requires visible device 0");
        const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
        const std::string requested = visible ? visible : "";
        if (!ap_bridge_uuid_is_valid(requested.c_str()) || requested.size() != 40)
            throw std::runtime_error("userd-map needs CUDA_VISIBLE_DEVICES=<one full GPU UUID>");
        if (!std::filesystem::create_directory(directory))
            throw std::runtime_error("USERD evidence directory already exists");
        std::filesystem::permissions(directory, std::filesystem::perms::owner_all);
        if (userd_observer_begin((directory / "observations.jsonl").c_str()) != 0)
            throw std::runtime_error("Passive observer initialization/LD_PRELOAD chain failed");
        ap_bridge_session_t* session = nullptr;
        auto bridge_check = [](int result) {
            if (result != AP_BRIDGE_OK) throw std::runtime_error(ap_bridge_last_error());
        };
        bridge_check(ap_bridge_create(requested.c_str(), requested.c_str(), AP_BRIDGE_OWNER_BACKGROUND, 0, 0, &session));
        if (!ap::profile_matches(ap::loaded_driver_version()))
            throw std::runtime_error("Runtime RM profile does not match the pinned source");
        CU(cuInit(0));
        CUdevice device; int count = 0, driver = 0; CUuuid uuid{}; char name[256]{};
        CU(cuDeviceGetCount(&count)); CU(cuDeviceGet(&device, 0));
        CU(cuDeviceGetUuid_v2(&uuid, device)); CU(cuDeviceGetName(name, sizeof(name), device));
        CU(cuDriverGetVersion(&driver));
        std::ostringstream hex;
        for (unsigned char c : uuid.bytes) hex << std::hex << std::setfill('0') << std::setw(2) << unsigned(c);
        std::string compact = requested.substr(4);
        compact.erase(std::remove(compact.begin(), compact.end(), '-'), compact.end());
        if (count != 1 || compact != hex.str() || attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) != 8 ||
            attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) != 0 ||
            attribute(device, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE) != CU_COMPUTEMODE_DEFAULT)
            throw std::runtime_error("Selected A100 physical UUID/compute-mode mismatch");
        out.meta("gpu_name", name); out.meta("gpu_uuid", requested);
        out.meta("cuda_driver_api_version", driver);
        out.meta("userd_probe", progress ? "bounded_existing_userd_reads" : "passive_mapping_only");
        out.meta("project_rm_controls", 0);
        if (!progress) out.meta("userd_memory_reads", 0);
        out.meta("explicit_context_destroy_calls", 0);
        Options probe = o;
        if (!probe.blocks) probe.blocks = 4 * attribute(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        // Retain owner threads on failure too: no destroy/reset recovery path.
        auto* a = new Worker; auto* b = new Worker;
        auto warm = [&](GpuState& g, bool is_a, int cpu) {
            pin_cpu(cpu); g.set_query_timeout(o.query_timeout_ms); g.create(device, probe, is_a);
            Sample sample; g.run_tiny(sample);
            return g.context_address();
        };
        bridge_check(ap_bridge_begin_allocation_scope(session, requested.c_str()));
        const auto context_a = a->submit([&](GpuState& g) { return warm(g, true, o.cpu_a); }).get();
        bridge_check(ap_bridge_end_allocation_scope(session));
        const auto context_b = b->submit([&](GpuState& g) { return warm(g, false, o.cpu_b); }).get();
        pin_cpu(o.cpu_control);
        bridge_check(ap_bridge_note_cuda_ready(session, 8, 0));
        auto write = [&](const std::string& file, const std::string& contents) {
            std::ofstream stream(directory / file);
            stream.exceptions(std::ios::badbit | std::ios::failbit); stream << contents << '\n';
        };
        uint64_t snapshot_seq = 0;
        auto snapshot = [&](const std::string& label) {
            // Existing owner/generation/profile/explicit-scope checks; no GET_INFO.
            const auto identity = ap::inspect_owned_tsg_for_gpu(compact);
            if (identity.binding().owner_pid != static_cast<uint32_t>(getpid()))
                throw std::runtime_error("Owner process changed");
            write(label + ".identity.json", identity.json());
            write(label + ".memory.json", ap::capture_ipc_memory_json());
            std::vector<char> diagnostics(ap_bridge_diagnostics_json_size(session));
            bridge_check(ap_bridge_diagnostics_json(session, diagnostics.data(), diagnostics.size()));
            write(label + ".diagnostics.json", diagnostics.data());
            std::ifstream maps("/proc/self/maps"); std::ostringstream owned_maps; std::string line;
            while (std::getline(maps, line)) if (line.find("/dev/nvidia") != std::string::npos) owned_maps << line << '\n';
            write(label + ".maps", owned_maps.str());
            snapshot_seq = userd_observer_mark(label.c_str());
            if (!snapshot_seq || !userd_observer_ok())
                throw std::runtime_error("Incomplete passive observation");
            return identity.json();
        };
        const auto first = snapshot("after_warmup");
        std::vector<uint64_t> addresses;
        if (progress) {
            // The parent launcher reuses the existing full mapping analyzer on
            // this live snapshot, then publishes a private plan atomically.
            // No saved address from an earlier process is accepted.
            write("ready.tmp", std::to_string(getpid()) + " " + std::to_string(snapshot_seq));
            std::filesystem::rename(directory / "ready.tmp", directory / "ready");
            const Ns deadline = now_ns() + 5000000000LL;
            while (!std::filesystem::exists(directory / "read.plan")) {
                if (now_ns() >= deadline) throw std::runtime_error("Live mapping validation timed out; no reads armed");
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            const auto current = ap::inspect_owned_tsg_for_gpu(compact);
            if (current.json() != first) throw std::runtime_error("Identity changed before live read validation");
            const auto& group = current.binding();
            std::ifstream plan(directory / "read.plan");
            std::string magic; uint64_t owner = 0, sequence = 0, registry = 0, count = 0;
            plan >> magic >> owner >> sequence >> registry >> count;
            if (!plan || magic != "USERD_READ_PLAN_V1" || owner != static_cast<uint64_t>(getpid()) ||
                sequence != snapshot_seq || registry != group.registry_instance || count != group.members.size() || count > 32)
                throw std::runtime_error("Live read plan owner/sequence/registry/member mismatch");
            for (unsigned i = 0; i < count; ++i) {
                uint64_t ordinal = 0, handle = 0, generation = 0, address = 0, entries = 0;
                plan >> ordinal >> handle >> generation >> address >> entries;
                const auto& token = group.members[i].channel.token;
                if (!plan || ordinal != i || handle != token.handle || generation != token.generation ||
                    !address || address % 512 || !entries || entries > (1u << 24))
                    throw std::runtime_error("Live read plan channel mismatch");
                addresses.push_back(address);
            }
            std::string extra;
            if (plan >> extra) throw std::runtime_error("Trailing live read plan fields");
        }
        // Same original tiny kernel, same context/resources; B remains standby.
        const auto after_a = a->submit([&](GpuState& g) {
            Sample sample;
            if (!progress) g.run_tiny(sample);
            else {
                if (g.context_address() != context_a ||
                    !userd_observer_arm_reads(snapshot_seq, addresses.data(), addresses.size()))
                    throw std::runtime_error("Live read audit became stale; no reads armed");
                auto read = [&](const char* label, Ns begin = 0, Ns end = 0, int result = -1) {
                    if (!userd_observer_sample(label, begin, end, result))
                        throw std::runtime_error("Bounded USERD read stopped: stale audit, deadline or capture failure");
                };
                auto window = [&](const char* label) {
                    for (int i = 0; i < 8; ++i) {
                        read(label);
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                    }
                };
                window("idle_before"); read("before_begin_event");
                g.run_tiny(sample, nullptr, [&](const char* label, Ns begin, Ns end, CUresult result) {
                    read(label, begin, end, result);
                    // Observe publication/GET with no intervening CUDA API.
                    // This deliberate host gap affects Event elapsed time.
                    if (std::string(label) == "after_launch" && result == CUDA_SUCCESS) window("launch_no_api");
                });
                window("idle_after");
                const int rounds = userd_observer_finish_reads();
                if (!rounds) throw std::runtime_error("Incomplete USERD read window");
                out.meta("userd_read_rounds", rounds);
                out.meta("userd_memory_reads", rounds * addresses.size() * 4);
                out.meta("tiny_result_verified", 1);
                std::ostringstream evidence;
                evidence << "{\"launch_begin_ns\":" << sample.b_launch << ",\"launch_end_ns\":" << sample.b_launch_return
                         << ",\"complete_observed_ns\":" << sample.b_complete << ",\"event_ms\":" << sample.b_event_ms
                         << ",\"start_gpu_ns\":" << sample.b_start_gpu_ns << ",\"end_gpu_ns\":" << sample.b_end_gpu_ns
                         << ",\"result_verified\":true,\"launch_sequence\":" << sample.b_launch_seq << '}';
                write("tiny.json", evidence.str());
            }
            return g.context_address();
        }).get();
        const auto after_b = b->submit([](GpuState& g) { return g.context_address(); }).get();
        const auto second = snapshot("after_tiny");
        if (first != second || context_a != after_a || context_b != after_b || context_a == context_b)
            throw std::runtime_error("Context or RM identity changed across snapshots");
        out.meta("context_A_identity", context_a); out.meta("context_B_identity", context_b);
        out.meta("identity_unchanged", 1); out.meta("userd_capture_complete", 1);
        out.meta("run_status", progress ? "captured_progress_not_yet_analyzed" : "captured_mapping_not_yet_analyzed");
        std::cerr << "USERD capture complete; analyze " << directory << '\n';
        std::cout.flush(); std::cerr.flush();
        // All submitted work completed. Contexts remain alive through the final
        // snapshot; the OS releases them at process exit, not as recovery.
        std::_Exit(0);
    } catch (const std::exception& e) {
        userd_observer_mark("failed");
        out.meta("run_status", "failed"); out.meta("error", e.what());
        std::cerr << "USERD mapping probe stopped: " << e.what() << '\n'; std::cerr.flush();
        std::_Exit(2);
    }
}
