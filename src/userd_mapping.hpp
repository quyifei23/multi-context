#pragma once
#include "userd_observer.h"
#include <rm_control.h>

// Reuse the original owner threads, kernels and bridge lifecycle discovery.
// All snapshots are taken while the two warmed contexts are alive and idle.
// This is a mapping test, not a rewind or sentinel/entry experiment.
[[noreturn]] void run_userd_mapping(const Options& o, Output& out) {
    const auto directory = std::filesystem::path(o.output + ".userd");
    try {
        if (o.device != 0) throw std::runtime_error("userd-map requires visible device 0");
        const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
        const std::string requested = visible ? visible : "";
        if (!ap_bridge_uuid_is_valid(requested.c_str()) || requested.size() != 40)
            throw std::runtime_error("userd-map needs CUDA_VISIBLE_DEVICES=<one full GPU UUID>");
        if (!std::filesystem::create_directory(directory))
            throw std::runtime_error("USERD evidence directory already exists");
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
        out.meta("cuda_driver_api_version", driver); out.meta("userd_probe", "passive_mapping_only");
        out.meta("project_rm_controls", 0); out.meta("userd_memory_reads", 0);
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
            if (!userd_observer_mark(label.c_str()) || !userd_observer_ok())
                throw std::runtime_error("Incomplete passive observation");
            return identity.json();
        };
        const auto first = snapshot("after_warmup");
        // Same original tiny kernel, same context/resources; B remains standby.
        const auto after_a = a->submit([&](GpuState& g) {
            Sample sample; g.run_tiny(sample); return g.context_address();
        }).get();
        const auto after_b = b->submit([](GpuState& g) { return g.context_address(); }).get();
        const auto second = snapshot("after_tiny");
        if (first != second || context_a != after_a || context_b != after_b || context_a == context_b)
            throw std::runtime_error("Context or RM identity changed across snapshots");
        out.meta("context_A_identity", context_a); out.meta("context_B_identity", context_b);
        out.meta("identity_unchanged", 1); out.meta("userd_capture_complete", 1);
        out.meta("run_status", "captured_mapping_not_yet_analyzed");
        std::cerr << "Passive mapping capture complete; analyze " << directory << '\n';
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
