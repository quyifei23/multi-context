# 存活 Context 接管：A100 实验（2026-09-22）

**B 不必等待 A 的长 kernel 自然结束。** 在本次 A100/driver 配置中，保留 A Context、
在约 10% 处直接通知 B 后，30 个 live-handoff 样本均观测到 B 完成后 A 的 end Event 仍 pending。
独立 Nsight 运行的 6 个样本进一步确认：`A GPU start < B GPU start < B GPU end < A GPU end`，
A/B 位于同一 GPU 的不同 Context。

约 1 s 的 A 工作中，invalidate → B host 观测完成的**最小值 1.088 ms，中位数 1.123 ms**（10 次）。
相同组内工作量的串行销毁对照中位数为 **975.182 ms**，idle 对照为 **15.162 μs**。
这支持“存活 Context 间可以让 B 提前执行”，没有证明具体的硬件抢占粒度、同时执行或延迟保证。

## 问题与实验定义

| 项目 | 定义 |
|---|---|
| Hypothesis | A 已启动且尚未结束时，B 可以在另一个存活 Context 获得执行机会 |
| Baseline 1：`serial-destroy` | invalidate → A owner 查询 A end Event → `cuCtxDestroy(A)` 返回成功 → B launch |
| Baseline 2：`idle` | A Context 常驻，A 不提交测量 kernel；同一 host 通知路径触发 B |
| Variable：`live-handoff` | invalidate 直接通知 B；A 不销毁、不取消，继续固定工作量直到自然结束 |
| Metrics | invalidate → B host launch / GPU start / host complete；B 完成后 A 是否 pending；GPU trace 的 A/B 起止顺序 |
| Expected observation | 若 B start/end 均早于 A end，否定“必须等待 A kernel 自然结束”；若只有 host launch 提前，不足以支持假设 |

`invalidate` 只有 application host 通知语义，不是 CUDA 的取消或调度 API。
两个长期存活的 owner thread 各自拥有独立 Driver `CUcontext`；第三个线程只协调 host 信号。
没有 `cuCtxSetCurrent()`、Runtime host API、MPS、优先级 stream、VMM、PyTorch、NCCL 或取消 flag。

## 实现与时序

1. A/B 分别在 owner thread 内创建 Context；B 预热并保持待命。同一目标时长下 B Context 一直存活。
2. 每组三条件前用 Events 重新校准一次 iterations，**组内固定相同 iterations**，按 trial 轮换三条件顺序。
   每个条件前再测一次相同工作量的 A 自然时长，保存 `reference_a_ms`。
3. B 进入 host 待命后，A 提交原有长 kernel。原有启动标记被 host 观测后，等待参考时长的 10%，记录 `t_invalidate_ns`。
   `idle` 条件没有测量 A kernel、没有这段等待。
4. `live-handoff` / `idle` 由 controller 在 t0 直接 release-store 通知 B；B 不等待 A owner 的 Event query。
   `serial-destroy` 仅在 A owner 成功销毁 Context 后通知 B。
5. B 记录 begin Event、launch tiny、记录 end Event。end Event synchronize 返回后立即记录 `t_B_complete_ns`，
   随即通知 A；B 的 Event elapsed 查询、结果 DtoH 拷贝/校验均在完成时间戳之后。
6. 对存活 A，A owner 收到 B 完成信号后立即查询自己的 end Event，记录查询前后 CPU 时间和 pending 状态。
   `live-handoff` 随后等待 A 自然完成、记录 Event 时间并检查输出；Context 保持存活。
7. A/B 本次工作均结束后才进行后置 B 时钟校准和 CSV 输出。下一次需要的 Context 重建、预热不进入接管延迟。

`src/kernels.cu`、`src/probe.hpp` 和生成的 cubin 均与原始销毁实验一致。
A 仍是固定迭代的 compute-bound FMA kernel，432 blocks × 256 threads，每 thread 每 iteration 256 FMA。
没有改变循环、检查取消状态或提前退出。新增 `b_done` 等信号仅存在于 host 协调逻辑。

初次尝试固定整个 target 的 iterations 时，短 kernel 曾从约 10 ms 漂移到约 5.4 ms。
因此最终实现按三条件组重新校准，不改变 GPU 配置或 kernel 逻辑。仍须使用实际参考/Event 时间解释样本，
不能把 `target_ms` 当成严格的 GPU 墙钟时长。当前未锁频，也没有足够证据将漂移归因于具体机制。

## 构建与复现

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBENCH_CUDA_ARCH=80
cmake --build build -j2
./build/context_ping_pong --mode live-handoff --long-ms 10,100,1000 --trials 10 --invalidate-fraction 0.1 --cpu-a 0 --cpu-b 1 --cpu-control 2 --output results/live_handoff_calibrated.csv
python3 scripts/summarize.py results/live_handoff_calibrated.csv > results/live_handoff_calibrated_summary.csv
```

以上是实际采集命令；再次运行需使用新的 output 路径。其他主机请按可用物理 core/NUMA 拓扑选择 CPU 编号，
或去掉三个 affinity 参数。`--mode all` 仍运行原有两组实验；新增三条件实验独立使用 `--mode live-handoff`。
`--action` 不改变新模式的三条件集合。可用 `--invalidate-ms` 替代参考时长比例。

实测环境：NVIDIA A100-PCIE-40GB，108 SM，compute capability 8.0，visible device 0，
PCI `0000:18:00.0`，UUID `99e4e85f-1945-866c-9e00-130b51df7908`。
Driver package/kernel module **595.58.03**，Driver API **13020（13.2）**，Toolkit/nvcc **12.8.61**，cubin **sm_80**。
MIG Disabled、compute mode Default；实验前未发现 MPS server，未并行启动其他实验；未锁 GPU clocks。
A/B/controller 分别固定 CPU 0/1/2，独立 physical core、NUMA node 0。测量在沙盒外进行，以访问真实设备节点。
完整参数与版本见对应 `.metadata.csv` / `.nvidia-smi.txt`。

## 普通运行结果：每个条件各 10 次

数据：[原始 CSV](results/live_handoff_calibrated.csv)、[汇总 CSV](results/live_handoff_calibrated_summary.csv)、
[环境 metadata](results/live_handoff_calibrated.metadata.csv)。总计 90 个测量样本，均 `valid=1`；
另有 33 个 calibration 行，不进入延迟统计。各条件是分开的请求，三条件组内固定工作量。

| 目标 (ms) | live 条件 A 自然参考中位数 (ms) | live 条件 A Event 中位数 (ms) | 实际启动观测 → invalidate 中位数 (ms) |
|---:|---:|---:|---:|
| 10 | 9.986 | 10.383 | 0.999 |
| 100 | 99.791 | 100.191 | 9.979 |
| 1000 | 997.830 | 998.224 | 99.783 |

10 ms live 组参考范围 8.743–9.987 ms，100 ms 为 99.790–99.827 ms，1000 ms 为 997.829–998.173 ms。
实际 invalidate 间隔为参考时长的 10.0000%–10.0028%；这是 host 启动观测到通知的间隔，
不是对所有 SM 执行进度的精确测量。A Event 区间可包含上下文调度间隔，不能把全部差值解释为纯计算耗时。

| 目标 (ms) | live：t0 → B host launch 中位数 (μs) | live：t0 → B GPU start 估计中位数 (ms) | live：t0 → B complete 最小 / 中位数 / p95 (ms) | serial-destroy：complete 中位数 (ms) | idle：complete 中位数 (μs) |
|---:|---:|---:|---:|---:|---:|
| 10 | 3.417 | 1.286 | 1.288 / 1.295 / 1.465 | 91.472 | 20.755 |
| 100 | 2.305 | 0.696 | 0.697 / 0.704 / 0.706 | 162.451 | 19.003 |
| 1000 | 1.966 | 无效，见下节 | 1.088 / 1.123 / 1.125 | 975.182 | 15.162 |

p95 使用 nearest rank；只有 10 个样本时等于最大值，不代表稳定的尾延迟分布。
最小值只表示本实现、本配置的观测最快完成时间。各目标的延迟不单调，不能据此宣称固定 context-switch quantum。
`t_B_complete` 是 host 观测 end Event 的时间，包含 CPU 等待/调度，且不包含随后结果 DtoH。
销毁对照的耗时还包含 Context 清理，不能把与 live 的全部差值称为 GPU 调度开销。

30/30 个 live 样本的 `a_context_alive_after_b_complete=1`、`a_event_pending_after_b_complete=1`。
30 个 idle 样本均为 Context alive、A Event complete；30 个 serial 样本均成功销毁 A，销毁后不访问 A Event。
A Event pending 是 host 可观测证据；直接判断 kernel GPU 起止顺序采用下面的独立 trace。

## GPU 开始时间与独立时间线证据

普通运行自动启用 `%globaltimer` 与 CPU `CLOCK_MONOTONIC_RAW` 的前后 bracket 估计。
10/100 ms live 组各 10 个样本通过恒定 offset 一致性检查；1 s live 组 **0/10 通过**，
`start_est_valid=0`，GPU start 估计保留 `nan`。不把两个时钟的原始值直接相减，也不从 Event elapsed 倒推真实启动。
即使通过检查，估计区间仍以局部同速、offset 稳定为假设，不能视为硬件精度保证。

为回答 GPU 执行顺序，另用 **Nsight Systems 2024.6.2.225** 采集最终版本，每条件 2 次：

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none --export=sqlite --output=results/live_handoff_calibrated_trace ./build/context_ping_pong --mode live-handoff --long-ms 10,100,1000 --trials 2 --cpu-a 0 --cpu-b 1 --cpu-control 2 --output results/live_handoff_calibrated_profiled.csv
python3 scripts/analyze_nsys.py results/live_handoff_calibrated_trace.sqlite results/live_handoff_calibrated_profiled.csv > results/live_handoff_calibrated_trace_comparison.csv
```

分析脚本只用 Python 标准库。它检查进程 PID、两个 owner TID、各 owner 全部 launch 数量和递增序号，
再按 CUDA API correlation ID 关联对应 kernel；还检查 A/B 同 device、不同 Context，以及 kernel 名称。
本次 A/B 分别 152/279 次 launch，包括预热和校准，全部匹配；不能把普通运行 CSV 与另一进程的 trace 混用。
Nsight 2024.6 将 `cuLaunchKernel` 记录放在名为 `CUPTI_ACTIVITY_KIND_RUNTIME` 的表内；
脚本依据实际 API 名称区分层次，这不表示 benchmark 调用了 CUDA Runtime。

结果：[profiled 原始 CSV](results/live_handoff_calibrated_profiled.csv)、
[GPU trace 对照 CSV](results/live_handoff_calibrated_trace_comparison.csv)。
仓库收录实验 CSV 和关联后的 GPU 时间线。原始 `results/live_handoff_calibrated_trace.nsys-rep` / `.sqlite`
保留在本地，以上命令可重新生成；它们还包含实验之外的进程列表和环境信息，不纳入上传内容。

| 目标 (ms) | 两个 live 样本：B start/end 均早于 A end | B 结束后距 A GPU end (ms) | t0 → B GPU start，API bracket 估计 (ms) |
|---:|:---:|---:|---:|
| 10 | 2/2 | 8.194 / 7.987 | 1.431 / 1.380 |
| 100 | 2/2 | 77.015 / 89.392 | 0.743 / 0.708 |
| 1000 | 2/2 | 896.023 / 896.241 | 1.243 / 1.226 |

例如 1000 ms、trial 0，把 A GPU start 归零，profiler 时间线为：

| GPU 事件 | 相对 A GPU start (ms) |
|---|---:|
| A start，Context 7 | 0 |
| B start，Context 8 | 100.899643 |
| B end，Context 8 | 100.901531 |
| A end，Context 7 | 996.924212 |

这些 A/B 起止时间在**同一个 profiler 时间域**，顺序判断不需要 CPU–GPU 时钟换算。
6 个 serial 样本均为 A GPU end 早于 B GPU start。

`t0 → B GPU start` 需要另行将 CPU t0 对齐到 profiler 时间域。脚本用测量 B 的同一次
`cuLaunchKernel` host 调用前后时间包住 profiler 的 API entry/exit，构造 offset 区间：
`lower = GPU_start - API_start + host_launch - t0`，
`upper = GPU_start - API_end + host_launch_return - t0`。
这依赖局部时钟同速；CSV 给出区间和有效性，不保证纳秒级准确度。
**profiled 延迟与普通运行统计分开报告**，不将 profiler 样本混入普通运行分位数。

参考：[NVIDIA Events](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__EVENT.html)、
[`%globaltimer`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi)、
[Nsight Systems 分析文档](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)。

## 证明了什么、没证明什么、下一步

证明了：该 A100 环境中，host 直接提交 B 足以使它在存活 A 的长 kernel 结束前执行并完成；
不需要以销毁 A 作为 B 获得执行机会的前提。A 仍自然完成，当前机制没有取消 A 的应用语义。

没有证明：A/B 在同一时刻使用 SM、具体的抢占粒度/时间片、其他 driver/GPU/负载的行为，
或毫秒级延迟的严格上界。kernel 生命周期重叠本身不能区分同时执行与暂停/恢复，不能把观测命名为某个硬件机制。

下一步最小实验：固定约 1 s 的 A 和相同迭代数，只扫描 invalidate 相位，测量 B start/complete 延迟分布，
检验提前执行的等待时间是否随提交时机变化；当前未实现这个扩展。
