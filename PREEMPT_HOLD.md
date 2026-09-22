# Stock RM preempt-and-hold：A100 实测（2026-09-22）

**已有可用的 stock primitive：`NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS`。**
在当前 A100 / 595.58.03 上，对 A 的已绑定 compute TSG 的全部 8 个 channel，设置
`bDisable=true, bOnlyDisableScheduling=false, bRewindGpPut=false, pRunlistPreemptEvent=NULL`，
可以让约 1 s 的 A kernel 暂停并保持到显式 enable。B 在其间完成请求，A Context 和资源仍有效。

单控制 `fifo-hold` 的普通运行共 5 次：invalidate → B complete 最小 **0.784 ms**、中位数 **0.841 ms**。
独立 GPU trace 的两次 invalidate → B GPU start 估计分别 **0.809 / 0.865 ms**；
各约 **1.2 s** 的 hold 窗口内均没有观测到 A RESTORE 或 A context 驻留。
五轮普通运行、两轮 trace 均在 enable 后完成旧工作、通过全输出对照，并在原 Context 上完成新 tiny kernel。

**enable 会恢复旧队列，不是丢弃旧 epoch，也不是 reset。** 本阶段的 rearm 明确定义为：
enable → 等旧工作完成并验证 → 原 Context/resource 上 launch 新请求。未测试 rewind、queue discard 或跳过旧工作。

## 四个研究问题

| 问题 | 本次证据支持的回答 |
|---|---|
| RM 是否已有 preempt + hold + re-enable？ | 有。上述 FIFO control 本身包含同步 preemption 和持久禁调度；`bDisable=false` 重新 enable。无需先调用独立 Group PREEMPT |
| 控制对象是什么？ | RM 请求发到本进程的 **subdevice**，参数指定 **channel 列表**；本次覆盖 A 的一个 compute TSG 的全部 8 个已捕获 channel。不是 CUDA Context handle，也不是关闭整个 runlist |
| CUDA Context 与资源能否保留？ | 本工作量可以：未销毁 A/B Context；A 完整计算输出与同工作量 reference 相同，Context/compute buffer 地址未变，原 module/stream/Events/tiny buffer 上的新 launch 成功 |
| 缺少什么？ | 对本阶段的 compute preempt-and-hold，不需要新增硬件或 RM primitive。尚未验证的是“丢弃旧 epoch 后立即重用”的队列/提交状态一致性；不能据本实验宣称安全 discard/reset 已解决 |

范围是本次捕获的 compute TSG，而不是对任意 CUDA Context 所有隐藏资源、copy TSG 或其他 engine 的完整暂停保证。
本次对象图中 A/B 各有自己的 compute TSG；不把这个实例拓扑当作所有 CUDA 程序的固定规则。

## 接口考古与关键区别

直接复用 `Interception/active-preempt` 的 `ap_bridge`，没有修改其工作树、RM transport、权限、generation 或 binding gates。
控制头文件固定到 NVIDIA 595.58.03 commit `db0c4e65c8e34c678d745ddb1317f53f90d1072b`：

- [Group PREEMPT / GPFIFO_SCHEDULE](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/common/sdk/nvidia/inc/ctrl/ctrla06c.h)：对象为 group；PREEMPT 本身没有持久 hold 参数。
- [FIFO_DISABLE_CHANNELS](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080fifo.h)：channel 列表；`bOnlyDisableScheduling=false` 同时移出运行中的 channel 并禁止后续调度，直到反向 enable。
- `bOnlyDisableScheduling=true` 只禁止后续调度，不移出当前正在运行的 channel；本次没有使用该值。
- `bRewindGpPut=false` 保留排队工作；本次没有调用 bridge 的 discard/rewind 入口。

匹配头文件已从本地官方 checkout 读取，构建时核对 commit 与 header diff。公开接口契约与实际 GPU 行为分别验证，
不从 `NV_OK` 单独推断暂停成功，也不假设公开 KMD 代码等于完整 GSP firmware 实现。

## 实验定义与顺序

Hypothesis：stock RM 可以让 A 的 compute channels 在抢占后持续 non-runnable，同时保持 CUDA 执行环境有效。
Baseline 是保留 A 的 `live-handoff`；变量是 invalidate 后的 RM controls。
核心指标是 invalidate → B GPU start/complete，以及控制返回后至 enable 前旧 A 是否仍有执行/驻留证据。
Expected observation：有效 hold 的窗口应超过 A 的剩余自然工作时长，A 仍 pending、trace 不见 A 恢复；enable 后旧工作才继续。

| CSV condition | t0 后、B launch 前的操作 | 用途 |
|---|---|---|
| `live-handoff` | controller 立即通知 B；无 RM control | 原有 live baseline |
| `preempt-resume` | 同步 Group PREEMPT 返回后通知 B | 验证 PREEMPT 本身是否持续暂停 |
| `preempt-hold` | Group PREEMPT → FIFO disable 返回后通知 B | 用户目标顺序的串行组合；**两次调用之间可恢复** |
| `schedule-hold` | Group PREEMPT → group GPFIFO_SCHEDULE(false) → B | TSG schedule 候选的单次诊断对照 |
| `fifo-hold` | 直接 FIFO disable 返回后通知 B | 测试已有的单个 preempt-and-hold primitive |

`--rm-case all` 运行前三个条件，按 trial 轮换顺序；额外两个机制探针单独选择。
active 路径的 B 通知在同步控制返回之后，不与控制调用并行。延迟包含 bridge 身份核验、RM ioctl 和 host 通知。

两 owner thread 全程分别绑定 A/B Context；controller 不绑定 Context，不调用 `cuCtxSetCurrent()`。
只有 A 的创建/预热纳入显式 allocation scope；B 仍正常捕获，但不作为 A 的控制候选。
**在 A/B 都完成初始化后**才绑定 A 并 GET_INFO，保留全部成员、候选版本、原 retained FD 和 owner PID 检查。
FIFO control 还先完成独立 RM subdevice UUID 查询，与目标 GPU UUID 比对。

每个试验组重新校准 A 的固定 iterations，组内三条件工作量一致；每个条件前保存相同工作量的完整输出 reference。
A 原有 kernel 启动标记被 host 观测后约 10% reference 时长发出 invalidate。
B 预热待命，收到通知提交原 tiny kernel，end Event synchronize 后立即记录 host complete。
A owner 查询一次自己的 end Event，然后等待 1.2 s 观察窗口到期；窗口中间不再向 A 发 CUDA API。
到期先查询 pending，再显式 enable（仅 hold 条件），等待旧 A 自然完成、逐元素检查完整 reference，并用原资源 launch 新 tiny。

A/B Context 不在 trial 间重建。`src/kernels.cu`、`src/probe.hpp` 和 cubin 与原 live-handoff 实验相同；
没有取消 flag、缩短/拆分 A kernel、VMM、PyTorch、NCCL、MPS，也没有 firmware/KMD/全局调度配置修改。

## 逐步观测

| 步骤 | 证明了什么 | 没证明什么 | 随后最小实验 |
|---|---|---|---|
| 匹配 595 接口与代码 | stock channel disable/enable 契约存在，已有受限 bridge 可复用 | 实际 CUDA 效果 | idle 可逆探针 |
| 初次 idle 探针 | 在创建 B 后，提前保存的 A 凭据因候选版本变化被拒绝；active ioctl=0 | 不是 stock control 失败 | 两 Context 初始化后再绑定，不放松校验 |
| 修正顺序的 idle 探针 | GET_INFO、独立 RM UUID 查询和 disable/enable 均成功；A 同 Context 新 launch 成功 | 运行中 kernel 的持久 hold | 三条件 1 s smoke |
| 三条件 smoke / 5 轮主实验 | FIFO hold 后 1.2 s A 仍 pending；enable 后完整输出和复用通过 | 单靠 pending 无法量化 GPU 驻留 | 独立 context-switch trace |
| 三条件 trace | FIFO hold 窗口无 A 驻留；仅 PREEMPT 后 A 恢复；两 control 之间存在恢复窗口 | 不能把串行两 control 当成原子 preempt-and-hold | 单 FIFO control |
| TSG schedule 对照 | ioctl 成功，但 A 在 enable 前完成，trace 有 A 恢复 | 未定位失败原因，不能推广所有 group controls | 采用已验证的 channel primitive |
| 单 FIFO control | 无需独立 PREEMPT，连续 hold/re-enable 和同 Context 复用成立 | 不含旧队列 discard/reset | 本阶段完成；若扩展，再检验不恢复旧 epoch 的安全 rearm |

初始失败保存在 `idle_probe.csv` 对应 metadata/RM journal；其后使用新输出路径，没有覆盖或删除失败结果。

## 普通运行：CPU 时间与复用

GPU：NVIDIA A100-PCIE-40GB，108 SM，cc 8.0。公开报告省略设备唯一标识。
Driver package/KMD/libcuda **595.58.03**，Driver API **13020 / 13.2**，Toolkit **12.8.61**，sm_80。
MIG Disabled、compute mode Default；未发现 MPS server；选定 GPU 开始前无其他计算进程。
A/B/controller 固定 CPU 0/1/2；未锁 GPU clocks。别的 GPU 上的既有进程未被操作。

公开逐样本数据：[ordinary_relative.csv](results/preempt_hold_public/ordinary_relative.csv)。
`run=three_conditions` 为每条件 n=5，同一对 A/B Context 完成全部 15 次；
`run=fifo_only` 为另一次普通运行，n=5。仅保留相对延迟、工作量及验证标志，原始采集记录留在本地。

| 条件 | t0 → Group PREEMPT return 中位数 (ms) | t0 → disable return 中位数 (ms) | t0 → B host launch 中位数 (ms) | t0 → B complete 最小 / 中位数 / 最大 (ms) | 1.2 s 窗口末 A pending |
|---|---:|---:|---:|---:|---:|
| live-handoff | — | — | 0.002939 | 0.905 / 1.092 / 1.122 | 0/5 |
| preempt-resume | 0.623 | — | 0.632 | 2.922 / 2.935 / 2.986 | 0/5 |
| preempt-hold（两 control） | 0.598 | 1.184 | 1.187 | 1.209 / 1.292 / 1.357 | 5/5 |
| fifo-hold（单 control） | 未调用 | 0.732 | 0.735 | 0.784 / 0.841 / 0.961 | 5/5 |

单 FIFO 组是另外进程、另外校准，不能与三条件数据当成严格成对的性能收益；这里首先比较机制与观测成本。
主实验自然 reference 为约 998–1000 ms；live/resume A Event 中位数约 999 ms，
FIFO hold 的 A Event 中位数约 **2199 ms**，包含持久暂停窗口。
20/20 个普通样本全部通过完整 A 输出对照、Context/compute buffer 不变与原 Context 新请求验证。
这些是本环境的有限样本，不是最低延迟或上界保证。

## GPU timeline 与 B start

普通运行的前后 `%globaltimer` 恒定 offset 检查在这些长窗口中未通过，GPU-start 估计保留 `nan`。
**B GPU start 用独立 Nsight 运行报告**，不把 CPU launch 或 Event elapsed 当作 GPU start。

| 条件（独立 trace） | 样本数 | t0 → B GPU start 估计 (ms) | disable-return → enable 前的 A 观测 |
|---|---:|---:|---|
| live-handoff | 2 | 0.913 / 1.231 | 无 disable；A 继续并自然完成 |
| preempt-resume | 2 | 2.977 / 3.021 | 无 disable；PREEMPT 后仍有约 900 / 897 ms 的旧 kernel 驻留包络 |
| preempt-hold | 2 | 1.232 / 1.305 | 各约 1200 ms 窗口，A RESTORE=0、驻留=0；enable 前 A 未完成 |
| schedule-hold | 1 | 3.023 | A 恢复一次，旧 kernel 在窗口内有约 899 ms 的驻留包络并完成 |
| fifo-hold | 2 | 0.809 / 0.865 | 各约 1200 ms 窗口，A RESTORE=0、驻留=0；enable 前 A 未完成 |

**串行 `PREEMPT → FIFO disable` 的间隙仍有 A 驻留，分别约 456 / 450 μs。**
所以持续停止的边界是 FIFO disable 成功返回，而不是前一个 Group PREEMPT 返回。
直接 `fifo-hold` 不存在这两个独立 controls 之间的可恢复窗口。
所有 traced B tiny kernel 的执行区间内，均未观测到 A context 驻留；这也不等于 B 对整个 GPU 的长期独占保证。

数据：[profiled_relative.csv](results/preempt_hold_public/profiled_relative.csv) 包含三条件、TSG schedule 和单 FIFO 的逐样本相对时间及驻留统计。
本地保留同进程 GPU switch/NVTX 关联证据与原始 `.nsys-rep` / `.sqlite`；这些文件及 RM 对象日志未上传。

分析使用 Nsight Systems **2024.6.2.225** 的 `RESTORE_START` / `SAVE_END`。
先用 owner launch 序号、PID/TID、总 launch 数、API correlation ID 找到被测 kernel；
再用独占的 A reference 和 B warmup 区间建立 **CUDA context → switch context 的时间关联推论**。
不把 switch ID、CUDA context ID、RM TSG ID 的数值相等当作映射。关联不唯一或 trace/CSV 不匹配时拒绝分析。
驻留包络不是 SM active cycles；单纯的 Context 驻留也可能包含 idle，CSV 另给出裁剪到旧 kernel 生命周期内的包络。

CPU RAW 与 profiler 时钟不直接相减。t0 → B start 用同一次 B launch 的 host/API entry-exit bracket 对齐，
保留带局部同速假设的 lower/upper/valid；控制间 hold 窗口使用同一 profiler 时间域中的 NVTX 返回后/调用前边界。
工具提示 Driver API 13.2 超出该 Nsight build 支持范围，使用 12.8 trace backend；实际 kernel、NVTX、switch 事件已取得且相互对应，
但 drop count 仍 unknown，因此“零驻留”是当前 trace 中未观测到，不能升级为跨硬件的严格保证。

## 编译与复现

保留原构建：不指定 RM 依赖时仍只构建原 Driver API benchmark。
新增实验为可选依赖，直接链接用户提供的 `Interception/active-preempt` 中已有桥接。
该工作树包含此前未提交的 bridge additions，**Interception 的 HEAD 单独不足以表示本次源码**；
实际使用的 20 个构建输入和库指纹记录在本地 `results/preempt_hold/rm_build_inputs.json`。
本次没有提交或改写那个仓库；在另一机器复现需要同一组已记录的 bridge 源码和匹配 NVIDIA headers。

```bash
INTERCEPTION_SRC=/path/to/Interception/active-preempt
NVIDIA_595_SRC=/path/to/open-gpu-kernel-modules-595.58.03
cmake -S "$INTERCEPTION_SRC" -B build-rm -DAP_RM_PROFILE=595.58.03 -DNVIDIA_SOURCE="$NVIDIA_595_SRC" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-rm --target rm_control rm_contract_test group_binding_test group_preempt_test group_gpu_identity_test -j4
ctest --test-dir build-rm --output-on-failure -R '^(rm_contract_test|group_binding_test|group_preempt_test|group_gpu_identity_test|bridge_ctypes_contract_test)$'
cmake -S . -B build-hold -DCMAKE_BUILD_TYPE=Release -DBENCH_INTERCEPTION_SOURCE="$INTERCEPTION_SRC" -DBENCH_RM_LIBRARY="$PWD/build-rm/librm_control.so"
cmake --build build-hold -j2
```

先在一个完整 A100 上做 idle 可逆探针。将 `GPU_UUID` 设为所选设备的完整 UUID（可用 `nvidia-smi --query-gpu=uuid --format=csv,noheader` 查看）；CPU 编号按本机拓扑调整，已有输出路径不能复用：

```bash
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES="$GPU_UUID" LD_PRELOAD="$PWD/build-rm/librm_control.so" timeout --kill-after=5s 30s ./build-hold/context_ping_pong --mode preempt-hold --rm-probe --trials 1 --cpu-a 0 --cpu-b 1 --cpu-control 2 --output results/probe-new.csv
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES="$GPU_UUID" LD_PRELOAD="$PWD/build-rm/librm_control.so" timeout --kill-after=5s 150s ./build-hold/context_ping_pong --mode preempt-hold --long-ms 1000 --hold-ms 1200 --trials 5 --cpu-a 0 --cpu-b 1 --cpu-control 2 --output results/hold-new.csv
python3 scripts/summarize.py results/hold-new.csv
```

将第二条命令加入 `--rm-case fifo-hold` 即单 FIFO；`--rm-case schedule-hold` 即 TSG schedule 对照。
推荐首先复现机制，不据本文直接调驱动、优先级或时钟。

独立 GPU trace（原始采集命令相同，仅 output 名称不同）：

```bash
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES="$GPU_UUID" timeout --kill-after=5s 90s nsys profile --trace=cuda,nvtx --gpuctxsw=true --sample=none --cpuctxsw=none --export=sqlite --output=results/hold-trace-new --env-var=LD_PRELOAD="$PWD/build-rm/librm_control.so" ./build-hold/context_ping_pong --mode preempt-hold --long-ms 1000 --hold-ms 1200 --trials 2 --trace-marks --cpu-a 0 --cpu-b 1 --cpu-control 2 --output results/hold-profiled-new.csv
python3 scripts/analyze_hold_trace.py results/hold-trace-new.sqlite results/hold-profiled-new.csv --evidence results/hold-evidence-new.json > results/hold-comparison-new.csv
```

## 输出与验证边界

- 主 CSV 保持原 60 列，`experiment=preempt-hold`；`condition` 区分五条路径。
- `<output>.controls.csv` 保存三类控制的 host 调用边界、真实 ioctl 边界、bridge/syscall/RM 状态及 operation sequence；未调用的控制以 `attempted=0` 判断，不能把零初始化字段当作 NV_OK。新增正常采集为未调用 RM status 保留 `0xffffffff`；最初 smoke 保留原样。
- `<output>.rm/` 保留原桥接的预分配 journal、对象身份和诊断快照。run status、控制是否被接受、hold 是否成立、输出正确与 Context 可复用分别记录。
- 公开 Git 只保留相对指标 CSV 和去标识验证摘要；原始 CSV/metadata、controls、对象身份、诊断、journal 和 profiler 数据全部留在本地。公开结果不包含设备唯一标识、PID/TID、绝对 host 时间、主机路径或 RM handle。
- 独立 FIFO `disable` 就是组合的 preemption/hold 返回点；其 `preempt_*` 字段为空，避免伪造一个未调用的 Group PREEMPT。
- CSV 的 complete 是 host 观测 Event 完成，不包含随后 DtoH；回收与复用校验均在关键延迟之后。

构建成功，复用桥接的 5 项相关离线测试通过；原无 RM 构建及 10 ms live/serial-destroy/idle 回归通过。
真实运行检查覆盖 CPU 顺序、控制返回、目标 client/subdevice/group 与完整 8-channel payload、无 rewind、完整输出及资源复用。
最终 [公开核验摘要](results/preempt_hold_public/validation_summary.json) 覆盖 32 个 RM 样本及其中 9 个独立 trace 样本；原始核验记录在本地。
分析器也通过了错配 controls 时间戳的拒绝检查。
失败/未知状态不自动换对象或重试控制，不将错误后 cleanup 当作成功恢复。

**已证明**：这个 stock channel primitive 足以支撑本阶段的 preempt-and-hold；A Context 可保留并在旧工作排空后复用。
**没证明**：旧 epoch 可被无损丢弃、A 任意资源/engine 都被覆盖、跨驱动的延迟保证或精确指令级抢占机制。
**下一步最小实验**：若要推进完整 epoch 切换，只研究 hold 状态下如何让旧提交失效并保持 CUDA 提交状态一致；本轮不扩展该方向。
