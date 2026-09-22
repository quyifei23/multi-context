# CUDA Driver API 双 Context 接管 microbenchmark

研究问题：**A 已开始执行长 kernel 后，host 请求销毁 A，预热的 B 最快多久完成一个短请求？**

`invalidate` 是本程序的 host 控制消息，不是 CUDA API。默认路径是
`invalidate → A owner 调用 cuCtxDestroy → 返回成功 → B owner 提交 tiny kernel → 完成 Event 被 host 观测`。
这测量 application host 控制路径与 CUDA Driver Context 的行为，不能直接归因于 GPU scheduler 或硬件抢占。

## 实验定义

| 实验 | Hypothesis | Baseline | Variable | Metric | Expected observation（待验证） |
|---|---|---|---|---|---|
| 待命成本 | 创建并预热的 idle B 对 A 的执行时间影响很小 | 进程内仅 A Context 存在，B host thread 睡眠且没有 Context | 是否常驻 B Context | A Event 时间、固定工作量的 effective GFLOP/s、成对时间比 | 两组相近；若有稳定差异，再定位原因 |
| 接管延迟 | 销毁 busy A 可以在自然完成前让 B 完成请求 | `--action wait-then-destroy`：先等 A 完成，再销毁并通知 B；另记 A idle 时 B 的请求时间 | 是否在销毁前显式等待 A 完成 | destroy latency、invalidate 到 B 完成、host handoff | 若直接销毁明显早于等待对照，说明此配置的接管路径有效；若随剩余工作量增长，则不支持快速接管假设 |

Driver 的 [`cuCtxDestroy` 文档](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__CTX.html)
描述资源销毁及并发 API 使用限制，没有提供“立即取消 kernel”的时限保证。本程序不把 destroy 当作已证实的 cancellation primitive。

## 实现边界

- `src/main.cpp` 只使用 CUDA **Driver API**。`nvcc` 仅生成 device-only cubin；host 不链接 `libcudart`。
- 两个长期存活的 owner thread 分别创建、使用、销毁各自的独立 `CUcontext`。第三个线程是 main/controller，不绑定 Context，负责发出 invalidate。
- 不调用 `cuCtxSetCurrent`、不使用 primary context、Graph、优先级 stream、PyTorch、NCCL 或 VMM。
- A 在待命实验中持续存在。严格 baseline 必须销毁 B；进入 test 后 B 预热完成才开始计时。AB/BA 成对交替顺序。
- 接管实验中 B 在同一目标时长的所有 trial 间持续存在；A 每次销毁后在原 owner thread 上重建、预热。Context 重建时间不计入接管延迟。
- A 为固定迭代数的 FP32 FMA kernel：每 thread 每 iteration 有 256 FMA，8 条独立寄存器依赖链，最后写出结果。默认 `4 × SM count` blocks、256 threads/block。不是按 `clock64` 或墙钟到期退出。
- 在只有 A 时通过 CUDA Events 自动校准目标时长，再固定迭代数用于该目标的所有条件。校准不是时长保证；实际 Event 时间保留在 CSV。GFLOP/s 是该合成固定工作的有效吞吐，不代表应用吞吐或峰值利用率。

## 编译

Linux、CMake ≥ 3.18、C++17 compiler、CUDA Toolkit ≥ 11.4；A100 默认 `sm_80`。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBENCH_CUDA_ARCH=80
cmake --build build -j
./build/context_ping_pong --help
```

无需可用 GPU 即可编译；实际运行需要正常工作的 NVIDIA driver。默认 module 路径写入 executable；移动产物后用
`--module /absolute/path/kernels.cubin` 指定位置。

## 运行

先确认目标为完整的单张 A100，记录并核对 MIG/MPS 状态。程序选择一个 CUDA-visible device；
`nvidia-smi` 的物理编号与 `CUDA_VISIBLE_DEVICES` 重编号不同，请用输出的 PCI bus ID/UUID 对应。
MIG slice 上的结果不等于完整 A100。独占 compute mode 不适用于本实验；程序只检查模式，不修改 GPU 配置。
MPS 是否真正启用不能由环境变量是否存在推断，需要在实验环境独立确认。测量期间避免其他 GPU workload。

先跑最小 smoke：

```bash
./build/context_ping_pong --mode all --long-ms 10 --trials 2 --output results/smoke.csv
```

正式采集两组实验：

```bash
./build/context_ping_pong --mode standby --long-ms 10,100,1000 --trials 20 --output results/standby.csv
./build/context_ping_pong --mode takeover --action both --long-ms 10,100,1000 --trials 20 --invalidate-fraction 0.1 --output results/takeover.csv
python3 scripts/summarize.py results/standby.csv results/takeover.csv
```

`--action both` 在同一轮校准、相同固定迭代数下，交替运行 `destroy` 与 `wait-then-destroy` 成对对照；
`--trials 20` 表示每个条件各 20 次。只测一条路径时使用 `--action destroy`（默认）或 `--action wait-then-destroy`。
分开运行的自动校准可能因 GPU clocks 不同产生不同迭代数，因此优先用 `both` 比较两条路径。

`--invalidate-fraction 0.1` 从 **host 观测 A kernel 启动** 起，等待目标时长的 10%；例如 100 ms 目标等待约 10 ms。
也可用 `--invalidate-ms 1` 固定延迟。实际触发间隔由时间戳计算，不把请求延迟当成实际值。
相同命令重复采集时使用新的 output 路径，已有 CSV/sidecar 不会被覆盖。

CPU 调度会进入接管延迟。可通过 `lscpu -e=CPU,CORE,SOCKET,NODE` 选择三个允许使用的独立物理 core，
追加 `--cpu-a <id> --cpu-b <id> --cpu-control <id>`；尽量位于 GPU 所在 NUMA node。
未指定则不主动固定 CPU，CSV 记录关键位置的实际 CPU ID。
实验一中 idle B host thread 在 condition variable 上睡眠；实验二中 A/B/controller 在通知边界 busy-poll。
`CU_CTX_SCHED_SPIN` 控制 CPU 等待方式，不是 GPU 调度优先级。

禁止 `CUDA_LAUNCH_BLOCKING=1`，否则 A 的异步前提失效。使用有限 kernel；若需对驱动长时间不返回设置进程期限，
可从外部使用 `timeout 180s ./build/context_ping_pong ...`。进程内的 marker timeout 不能中断卡住的 Driver 调用。

## 时间线与有效样本

1. A/B 的 module、内存、stream、Events 都已创建并完成预热，B 进入 host 待命。
2. A 记录 begin Event，launch 长 kernel，记录 end Event；A kernel 的 block 0/thread 0 发布启动标记。
3. A owner 观测启动标记；controller 等待配置延迟，在 `t_invalidate` 读取 CPU timer 后用 release-store 发信号。
4. A owner 观测通知，查询 end Event，记录查询开始/结束。若已经完成，标记 `a_already_complete`，不纳入接管统计。
5. 默认立即调用 `cuCtxDestroy`；调用前不做 synchronize、free 或 module unload。可选等待对照才先 synchronize。
6. 记录销毁返回时间。仅返回成功时通知 B；B 记录 Event、launch tiny kernel、记录 end Event并等待完成。
7. 在 end Event 的 synchronize 返回后立即读取 `t_B_complete`；Event elapsed 查询、B 输出拷回/校验、CSV 写盘均在此后。

启动标记使用 A 所属的 mapped pinned host memory、自然对齐 32-bit flag，以及 `cuda::atomic_ref` system-scope
release-store/acquire-load。仅使用 load/store，不使用 host/GPU RMW；依据
[CUDA C++ memory model 的 mapped-memory atomicity 条件](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html)。
这证明 A 的某个 thread 已进入 kernel；结合未完成 end Event，筛选“已启动且完成尚未被观测”的样本。
**它不证明 t0 时所有 SM 都正在执行 A，也不能消除 query 到 destroy 之间自然结束的窗口。** 因此先用较早的 invalidate，并检查参考时长。
启动标记只用于接管实验，带来一次 system-scope 通知开销；待命两组均不发布标记。
销毁后不再访问 A 的内存、stream、module 或 Events，因此 busy A 的 Event elapsed 留为 `nan`。

## CSV 与环境记录

`--output results/run.csv` 产生：

- `run.csv`：每个 trial 的原始记录；每行刷新。`calibration` 是校准参考，`standby` 是实验一，`takeover` 是实验二。
- `run.metadata.csv`：GPU 名称/UUID/PCI bus ID/显存/SM 数、compute capability、CUDA Driver API version、完整 Toolkit version、可读时的内核驱动版本、CPU timer、参数与相关环境变量。
- `run.nvidia-smi.txt`：计时前只读采集物理 GPU 型号、UUID、驱动包版本、MIG/compute mode；失败也保留错误。`cuDriverGetVersion` 表示 Driver 支持的 CUDA API 版本，不是驱动包版本。

metadata 是追加 key/value 记录，`run_status` 的**最后一个值**代表最终状态。进程被外部中断时可能只剩 `initializing`；
这不是完整运行。API 错误返回非零；失败 trial 标记 `valid=0,status=error`，不制造成功结果。
没有对应测量的 host timestamp 为 `0`、duration 为 `nan`、状态值为 `-1`。

| 关键字段 | 定义 |
|---|---|
| `t_*_ns` | `CLOCK_MONOTONIC_RAW` 纳秒；同一台主机、同一 clock domain。ns 是表示单位，不等于实际测量精度 |
| `t_invalidate_ns` | controller 发布 invalidate 之前 |
| `t_invalidate_seen_ns` | A owner 观测到 invalidate 之后 |
| `t_destroy_begin_ns` / `t_destroy_return_ns` | 紧贴 `cuCtxDestroy` 调用前/后 |
| `t_B_launch_ns` / `t_B_launch_return_ns` | 紧贴 B 的 `cuLaunchKernel` 调用前/后；launch return 不等于 GPU 开始执行 |
| `t_B_complete_ns` | host 观测 B end Event 完成；是 device completion 的上界，包含 host polling/调度延迟 |
| `destroy_latency_us` | `(t_destroy_return - t_destroy_begin) / 1000` |
| `invalidate_to_destroy_return_us` | `(t_destroy_return - t_invalidate) / 1000`，还包含通知、query，以及等待对照中的 wait |
| `b_takeover_us` | **`(t_B_complete - t_invalidate) / 1000`**，核心指标 |
| `handoff_us` | `(t_B_launch - t_destroy_return) / 1000`，包括通知、host 调度和 B begin Event 提交 |
| `b_request_us` | B launch 调用前到 host 观测完成；`b_with_idle_a` 提供预热短请求的参考成本 |
| `a_event_ms` / `b_event_ms` | 各自 Context 内同一 stream 的 Event 区间；不跨 Context 比较 Events |
| `a_pending_before_destroy` | invalidate 被观测后，A end Event query 是否返回 NOT_READY；等待对照中也在 wait 前查询 |

CUDA Events 的计时分辨率约为 0.5 μs，异步记录间隔可能包含其他调度工作，见
[Event 文档](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__EVENT.html)。
特别是 tiny kernel 的 `b_event_ms` 不应被当成纯 kernel 指令执行时间。
`t_B_complete` 不包含结果 DtoH 拷贝；如果请求语义要求 CPU 收到结果，需要另设那个完成点。

## 可选：估算 B 何时开始

```bash
./build/context_ping_pong --mode takeover --long-ms 100 --trials 10 --estimate-start --output results/start.csv
```

tiny kernel 在靠近入口/出口处读取 `%globaltimer`，原始值输出为 `B_start_gpu_ns` / `B_end_gpu_ns`。
默认就保留这两个值；它们不是 CPU 时钟，**不能直接减 `t_invalidate_ns`**。

`--estimate-start` 在 A 测量 launch 之前以及 B 完成之后，各采样 7 次 idle B tiny kernel，选最窄的 host launch/completion bracket。
GPU 采样发生在 bracket 内，从而构造 CPU–GPU clock offset 区间；在近似同速、局部 offset 稳定的假设下，
输出 `t_B_start_est_ns` 及 `lower/upper` 区间、`clock_bracket_ns` 和前后 offset 变化。
前后区间不重叠、GPU 时间不单调或估计与 host launch/completion 矛盾时，`start_est_valid=0`，估计留 `nan`。
这个检查不能证明计时期间没有 clock drift；区间是**带时钟假设的估计误差范围**，不是硬件保证。

[`%globaltimer` 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi)
声明其行为与 target 有关，主要供 NVIDIA 工具使用。实际采样点在 kernel 内，晚于调度器开始执行的瞬间。
本程序不使用跨 Context Event elapsed 或按 `t_B_complete - b_event_ms` 倒推启动。

## 如何解释结果

只在 `valid=1` 的 `destroy` 行中报告 `b_takeover_us` 的最小值，并同时报告样本数、median、p95、目标/参考时长和环境。
最小值是这个实现与环境下**观测到的最快 host 接管完成**，不是硬件最低可能延迟。启动估计有效性与 trial 有效性是独立字段。
汇总脚本会保留无效样本计数；没有有效样本时输出 `NA`。

- 结果能证明：这套明确顺序的 destroy→B 路径在所测环境中的可观测延迟；idle B 对固定工作的影响。
- 结果不能证明：A 的 kernel 被硬件立即终止、指令级抢占机制、任何驱动/GPU 上的延迟保证，或整个循环的可持续请求吞吐。A 的重建时间不在核心指标内。
- 下一步最小实验：比较同样 early invalidate 下 `destroy` 与 `wait-then-destroy` 的 10/100/1000 ms 曲线。只有看到稳定差异后，再考虑 B 与 destroy 并行提交等新机制。

实测结果见 [RESULTS.md](RESULTS.md)，验证范围见 [VALIDATION.md](VALIDATION.md)。
沙盒内 GPU 不可见时应先检查隔离/设备访问权限，不能据此判断主机没有可用 GPU。
