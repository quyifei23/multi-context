#pragma once
#include <array>

// Original tiny kernel, four disjoint result slots, no new device code. Graph
// handles/arguments stay fixed after capture; all validation APIs are outside
// the before/after attribution window.
class GraphEntryProbe {
    GpuState& gpu_;
    CUdeviceptr output_ = 0;
    std::array<CUgraph, 2> graphs_{};
    std::array<CUgraphExec, 2> execs_{};
    std::array<uint64_t, 4> previous_start_{};
    static unsigned token(unsigned slot) { return 0x71300001U + slot; }
    CUresult direct(unsigned slot) {
        CUdeviceptr destination = output_ + slot * sizeof(TinyResult);
        unsigned value = token(slot); void* args[] = {&destination, &value};
        return cuLaunchKernel(gpu_.tiny_function(), 1, 1, 1, 1, 1, 1, 0, gpu_.stream(), args, nullptr);
    }
    void capture(unsigned which, unsigned first, unsigned count) {
        CU(cuStreamBeginCapture(gpu_.stream(), CU_STREAM_CAPTURE_MODE_GLOBAL));
        for (unsigned i = first; i < first + count; ++i) CU(direct(i));
        CU(cuStreamEndCapture(gpu_.stream(), &graphs_[which]));
        size_t nodes = 0, edges = 0;
        CU(cuGraphGetNodes(graphs_[which], nullptr, &nodes));
        CU(cuGraphGetEdges(graphs_[which], nullptr, nullptr, &edges));
        if (nodes != count || edges != count - 1) throw std::runtime_error("Unexpected captured graph topology");
        std::vector<CUgraphNode> handles(nodes), ordered(nodes, nullptr), from(edges), to(edges);
        CU(cuGraphGetNodes(graphs_[which], handles.data(), &nodes));
        for (auto h : handles) {
            CUgraphNodeType type; CUDA_KERNEL_NODE_PARAMS p{};
            CU(cuGraphNodeGetType(h, &type));
            if (type != CU_GRAPH_NODE_TYPE_KERNEL) throw std::runtime_error("Graph contains a non-kernel node");
            CU(cuGraphKernelNodeGetParams(h, &p));
            if (p.func != gpu_.tiny_function() || p.gridDimX != 1 || p.gridDimY != 1 || p.gridDimZ != 1 ||
                p.blockDimX != 1 || p.blockDimY != 1 || p.blockDimZ != 1 || p.sharedMemBytes || !p.kernelParams)
                throw std::runtime_error("Graph kernel parameters changed");
            const auto destination = *static_cast<CUdeviceptr*>(p.kernelParams[0]);
            if (destination < output_ || (destination - output_) % sizeof(TinyResult)) throw std::runtime_error("Graph result slot mismatch");
            const auto slot = (destination - output_) / sizeof(TinyResult);
            if (slot < first || slot >= first + count || ordered[slot - first] ||
                *static_cast<unsigned*>(p.kernelParams[1]) != token(slot)) throw std::runtime_error("Graph token/slot is not unique");
            ordered[slot - first] = h;
        }
        if (edges) {
            CU(cuGraphGetEdges(graphs_[which], from.data(), to.data(), &edges));
            for (unsigned i = 0; i + 1 < count; ++i) {
                unsigned matches = 0;
                for (unsigned e = 0; e < edges; ++e) matches += from[e] == ordered[i] && to[e] == ordered[i + 1];
                if (matches != 1) throw std::runtime_error("Captured graph is not the expected K1 -> K2 -> K3 chain");
            }
        }
        CU(cuGraphInstantiate(&execs_[which], graphs_[which], 0));
        CU(cuGraphUpload(execs_[which], gpu_.stream()));
        CU(cuGraphLaunch(execs_[which], gpu_.stream()));
        verify(first, count);
    }
public:
    struct Call { Ns begin, end; CUresult result; };
    explicit GraphEntryProbe(GpuState& gpu) : gpu_(gpu) {
        CU(cuMemAlloc(&output_, 4 * sizeof(TinyResult)));
        CU(cuMemsetD8(output_, 0, 4 * sizeof(TinyResult)));
        capture(0, 0, 1); capture(1, 1, 3);
    }
    uintptr_t exec(unsigned condition) const {
        return condition < 2 ? 0 : reinterpret_cast<uintptr_t>(execs_[condition == 2 ? 0 : 1]);
    }
    std::string identity() const {
        std::ostringstream s;
        s << "{\"context\":" << gpu_.context_address() << ",\"stream\":" << reinterpret_cast<uintptr_t>(gpu_.stream())
          << ",\"result_buffer\":" << output_ << ",\"graph_one_exec\":" << reinterpret_cast<uintptr_t>(execs_[0])
          << ",\"graph_three_exec\":" << reinterpret_cast<uintptr_t>(execs_[1])
          << ",\"graph_one_nodes\":1,\"graph_three_nodes\":3,\"graph_three_edges\":[[0,1],[1,2]],\"capture_instantiate_upload_warmup_complete\":true}";
        return s.str();
    }
    std::vector<Call> launch(unsigned condition) {
        std::vector<Call> calls; calls.reserve(3);
        const unsigned count = condition == 1 ? 3 : 1;
        for (unsigned i = 0; i < count; ++i) {
            const Ns begin = now_ns();
            const CUresult result = condition < 2 ? direct(condition == 0 ? 0 : 1 + i) : cuGraphLaunch(execs_[condition == 2 ? 0 : 1], gpu_.stream());
            const Ns end = now_ns(); calls.push_back({begin, end, result});
            if (result != CUDA_SUCCESS) break;
        }
        return calls;
    }
    std::string verify(unsigned first, unsigned count) {
        const Ns query_begin = now_ns(), deadline = query_begin + 3000000000LL;
        unsigned queries = 0;
        for (;;) {
            const auto result = cuStreamQuery(gpu_.stream()); ++queries;
            if (result == CUDA_SUCCESS) break;
            if (result != CUDA_ERROR_NOT_READY) CU(result);
            if (now_ns() >= deadline) throw std::runtime_error("Graph completion query deadline");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const Ns query_end = now_ns(); std::array<TinyResult, 4> values{};
        const Ns copy_begin = now_ns(); CU(cuMemcpyDtoH(values.data(), output_, sizeof(values))); const Ns copy_end = now_ns();
        std::ostringstream s;
        s << "{\"verified\":true,\"query_begin_ns\":" << query_begin << ",\"query_end_ns\":" << query_end
          << ",\"query_count\":" << queries << ",\"copy_begin_ns\":" << copy_begin << ",\"copy_end_ns\":" << copy_end << ",\"nodes\":[";
        for (unsigned i = first; i < first + count; ++i) {
            const auto& v = values[i];
            if (v.value != (token(i) ^ kTokenMask) || v.end_gpu_ns < v.start_gpu_ns || v.start_gpu_ns <= previous_start_[i])
                throw std::runtime_error("Graph node result incorrect or not fresh since prior invocation");
            if (i > first) s << ',';
            s << "{\"slot\":" << i << ",\"token\":" << token(i) << ",\"value\":" << v.value
              << ",\"previous_start_gpu_ns\":" << previous_start_[i] << ",\"start_gpu_ns\":" << v.start_gpu_ns
              << ",\"end_gpu_ns\":" << v.end_gpu_ns << '}';
            previous_start_[i] = v.start_gpu_ns;
        }
        s << "]}"; return s.str();
    }
    std::string verify_case(unsigned condition) { return verify(condition == 0 || condition == 2 ? 0 : 1, condition == 0 || condition == 2 ? 1 : 3); }
};

template<class Snapshot, class Write>
void graph_entry_experiment(const Options& o, Worker* a, const std::string& uuid,
                            const std::filesystem::path& directory, uint64_t& snapshot_seq,
                            Snapshot& snapshot, Write& write) {
    // Called only after the unchanged USERD discovery probe has completed.
    auto* graph = a->submit([](GpuState& g) { return new GraphEntryProbe(g); }).get();
    write("graph_state.json", graph->identity());
    const std::array<std::string, 5> cases{"direct_one", "direct_three", "graph_one", "graph_three", "graph_three_repeat"};
    std::string initial;
    for (unsigned index = 0; index < cases.size(); ++index) {
        const auto& name = cases[index]; const auto label = "graph_before_" + name;
        const auto before = snapshot(label);
        if (initial.empty()) initial = before;
        if (before != initial) throw std::runtime_error("Graph condition owner/generation changed");
        write(name + ".ready.tmp", std::to_string(getpid()) + " " + std::to_string(snapshot_seq));
        std::filesystem::rename(directory / (name + ".ready.tmp"), directory / (name + ".ready"));
        const Ns deadline = now_ns() + 5000000000LL;
        while (!std::filesystem::exists(directory / (name + ".plan"))) {
            if (now_ns() >= deadline) throw std::runtime_error("No valid existing compute ring set; stopped without reads");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        const auto current = ap::inspect_owned_tsg_for_gpu(uuid);
        if (current.json() != before) throw std::runtime_error("Graph plan identity became stale");
        const auto& group = current.binding(); std::ifstream plan(directory / (name + ".plan"));
        std::string magic; uint64_t owner = 0, sequence = 0, registry = 0, count = 0, active = 0;
        plan >> magic >> owner >> sequence >> registry >> count >> active;
        if (!plan || magic != "GRAPH_RING_PLAN_V1" || owner != static_cast<uint64_t>(getpid()) || sequence != snapshot_seq ||
            registry != group.registry_instance || count != group.members.size() || !count || count > 32 || active >= count)
            throw std::runtime_error("Graph read plan owner/generation/member mismatch");
        std::vector<uint64_t> userds(count), rings(count); std::vector<unsigned> entries(count);
        for (unsigned i = 0; i < count; ++i) {
            uint64_t ordinal = 0, handle = 0, generation = 0;
            plan >> ordinal >> handle >> generation >> userds[i] >> rings[i] >> entries[i];
            const auto& token = group.members[i].channel.token;
            if (!plan || ordinal != i || handle != token.handle || generation != token.generation)
                throw std::runtime_error("Graph channel token mismatch");
        }
        std::string extra; if (plan >> extra) throw std::runtime_error("Trailing graph plan fields");
        a->submit([&](GpuState& g) {
            if (!userd_observer_arm_ring_set(sequence, userds.data(), rings.data(), entries.data(), count))
                throw std::runtime_error("Graph read audit became stale");
            std::vector<uint32_t> puts(count);
            auto read = [&](const char* phase) {
                if (!userd_observer_ring_set_sample(phase, index, puts.data())) throw std::runtime_error("Bounded graph ring read stopped");
            };
            read("idle"); unsigned preparation = 0;
            if (o.entry_wrap && index == 3) {
                if (puts[active] >= entries[active]) throw std::runtime_error("Wrap PUT outside ring");
                preparation = entries[active] - 1 - puts[active];
                const Ns fill_deadline = now_ns() + 3000000000LL;
                for (unsigned j = 0; j < preparation; ++j) {
                    Sample s; CU(g.launch_tiny_only(s));
                    if (!userd_observer_ok() || now_ns() >= fill_deadline) throw std::runtime_error("Graph wrap preparation stopped");
                }
            }
            read("before_launch");
            if (o.entry_wrap && index == 3 && puts[active] != entries[active] - 1) throw std::runtime_error("Wrap preparation did not reach expected slot");
            const auto calls = graph->launch(index);
            read("after_launch");
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            read("settled");
            const auto rounds = userd_observer_finish_reads();
            if (!rounds) throw std::runtime_error("Incomplete graph ring read window");
            std::ostringstream evidence;
            evidence << "{\"condition\":\"" << name << "\",\"case_index\":" << index << ",\"graph_exec\":" << graph->exec(index)
                     << ",\"preparation_launches\":" << preparation << ",\"wrap_requested\":" << (o.entry_wrap ? "true" : "false")
                     << ",\"rounds\":" << rounds << ",\"calls\":[";
            for (unsigned i = 0; i < calls.size(); ++i) {
                if (i) evidence << ',';
                evidence << "{\"api\":\"" << (index < 2 ? "cuLaunchKernel" : "cuGraphLaunch") << "\",\"begin_ns\":" << calls[i].begin
                         << ",\"end_ns\":" << calls[i].end << ",\"result\":" << calls[i].result << '}';
            }
            evidence << "]}"; write(name + ".launch.json", evidence.str());
            for (const auto& c : calls) CU(c.result);
            write(name + ".result.json", graph->verify_case(index));
        }).get();
        if (snapshot("graph_after_" + name) != before) throw std::runtime_error("Graph owner/generation changed after launch");
    }
    write("graph_state_final.json", graph->identity());
}
