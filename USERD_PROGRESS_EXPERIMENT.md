# 最小实验：一次 tiny submission 前后的 USERD GET/PUT

2026-09-22。**一次 smoke 完成：合法既有映射上的 USERD GET/PUT 可读，且随本次提交发生变化。**

A compute TSG 的 8 个 channel 中，仅 ordinal 1 的字段变化：idle `(4,4)` → begin Event 后 `(4,5)`
→ tiny launch 返回后 `(5,6)` → 无 CUDA API 窗口中 `(6,6)` → end Event 后 `(7,7)`。
另外 7 个 channel 保持不变。ordinal 是本进程已验证成员列表的展示编号，不是跨进程 channel ID。

**新增 project RM control、额外 GPU mapping、FIFO disable/enable、rewind 均为 0。**
这补上“字段能否读取、是否变化”的证据；没有补上 application submission → GPFIFO entry 的完整映射。

## 1. Research question

在重新验证的 owned CUDA channel / USERD 映射上，能否有界、只读地观察一次原 tiny kernel 提交前后的字段变化？
这一步不问 rewind 是否生效，也不把 `GET == PUT` 当作 kernel 完成判据。

## 2. System abstraction

本轮闭合的链条是：

`当前 A TSG 全体 channel → owner/generation/physical UUID → 已有 live USERD CPU mapping`

`→ 固定字段的 host read bracket → CUDA API 前后的观测序列`

仍未闭合：`application epoch → 指定 channel 上的确切 GPFIFO entry → 硬件消费/预取时刻 → 可 rewind 区间`。
CUDA Event、launch、结果回读分别标记，避免把多个 Driver API 引起的 PUT 变化误称为一个 kernel 的 entry 数。

## 3. Existing mechanism

沿用 [mapping 实验](USERD_MAPPING_EXPERIMENT.md)的固定 NVIDIA 595.58.03 source、原桥接与既有 memory mapping。
公开 C56F USERD 布局给出 `GPGet=0x88`、`GPPut=0x8c`，struct 大小 512 bytes；编译时 static_assert 核对。
[固定头文件](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/common/sdk/nvidia/inc/class/clc56f.h#L48)。

每个 channel 每轮按 `GET₀ → PUT₀ → GET₁ → PUT₁` 做四次 aligned volatile 32-bit load，
用 `CLOCK_MONOTONIC_RAW` 包围这些读取。x86 LFENCE 只约束 CPU 读顺序，**不刷新 GPU 状态、不保证原子快照或 GPU 新鲜度**。
不解引用 GP FIFO GPU VA，不读取 RAMFC，不修改 USERD，不映射新的设备内存。

## 4. Observability / live gates

- 新进程内按原 allocation scope 创建 A，并预热 A/B；在它们 idle、Context 存活时取得 `after_warmup` 快照。
- 启动器在 parent process 中复用完整 mapping 分析器，验证全部成员的 owner、generation、物理 UUID、Memory ownership、
  map/mmap 生命周期、当前 `/proc/<child>/maps`、原 bridge envelope 一致性和新增 controls=0。
  parent 只分析 metadata，不读 child memory，不打开 GPU。
- parent 原子发布本次私有 read plan；child 再核对 PID、registry、当前完整 group identity、成员 handle/generation 和 snapshot sequence。
  之前进程的地址或过期 snapshot 不可复用。证据目录权限 0700；原始 observer 日志权限 0600。
- 读取时与原 observer 使用同一个锁；读窗口里若出现任何新的被捕获 syscall/lifecycle record，就将 audit 标为失效，停止读取。
  不放松原 bridge GET_INFO/FIFO UUID gate，也不把被动 UUID 证据当作 active control credential。
- 最多 256 轮、3 秒；Event query deadline 3 秒，外层进程 deadline 45 秒。无无限 synchronize。
  最后再核对原 Context、group generations 和 USERD mapping。

限制仍在：拦截范围是原有 libc hooks；direct/hidden syscall 与未拦截路径不在覆盖保证内。
这不是通用的 driver snapshot API，更不是对任意 CUDA 程序的永久读 capability。

## 5. Experiment

| 项目 | 定义 |
|---|---|
| Hypothesis | 既有 USERD 映射上可以观察本次 tiny 提交前后的 GET/PUT 字段变化 |
| Baseline | A/B 已预热，两个 Context idle；8 轮无 CUDA API 读取 |
| Variable | A 原 stream 上再执行一次原 tiny，加入 host 只读观测点；B 保持 standby |
| Metric | 各 channel 的四次字段读数/host bracket、API 边界/返回值、Event 与结果正确性、identity 稳定性 |
| Expected observation | idle 读数稳定；某个 channel 在提交后变化；无 API 窗口可观察 GET 是否随后推进；不能预设变化数量 |

环境：NVIDIA A100-PCIE-40GB，cc 8.0；KMD/libcuda **595.58.03**，CUDA Driver API **13020 / 13.2**，
Toolkit **12.8.61**，Linux x86_64，MIG Disabled，compute mode Default。开始前和结束后目标 GPU 均无其他 compute 进程。
两个长期 owner thread 分别绑定 CPU 0/1，controller CPU 2；没有执行长 kernel、old/new sentinel 或 Nsight trace。

工作序列：A tiny warmup ×1、B tiny warmup ×1 → live binding validation → idle 读取 8 轮 →
begin Event → tiny launch → **不调用 CUDA API，读取 8 轮** → end Event → 有界 Event query →
elapsed-time query → 原结果 DtoH/校验 → idle 读取 8 轮 → final binding validation。
三个只读窗口的轮间 sleep 请求为 100 μs；记录真实时间，不假定 sleep 精确。

原 `kernels.cu`、`probe.hpp` 未改，cubin 与前阶段 SHA-256 相同。
只在 host 的 `run_tiny` 增加可选观测回调。Context 到最后一个快照仍存活；成功完成后进程退出释放资源，
显式 `cuCtxDestroy` / reset 调用为 0，没有错误恢复操作。

### 结果与时间边界

本轮只运行 **1 个 smoke**，不是多次稳定性或性能统计。它已回答本轮“字段是否可读且变化”的问题，不扩大 active 实验。
完整 [samples.csv](results/userd_progress_public/smoke_01/samples.csv) 和 [summary.json](results/userd_progress_public/smoke_01/summary.json)。

下表仅列 ordinal 1，时间为相对 tiny `cuLaunchKernel` entry 的 CPU μs，区间包围四次字段读取。
每组重复读取均相等，但不把这种相等升级为跨字段原子性保证。

| 观测点 | 字段 read bracket (μs) | GET | PUT |
|---|---:|---:|---:|
| idle 最后一次 | −349.911 .. −344.983 | 4 | 4 |
| begin Event 返回后 | −90.084 .. −85.133 | 4 | 5 |
| tiny launch 调用前 | −42.744 .. −37.800 | 4 | 5 |
| tiny launch 返回后 | 24.565 .. 37.837 | 5 | 6 |
| 无 CUDA API，第一轮 | 79.983 .. 84.887 | 5 | 6 |
| 无 CUDA API，第二轮 | 278.676 .. 283.657 | 6 | 6 |
| end Event 返回后 | 2885.873 .. 2890.807 | 7 | 7 |
| Event query 成功后 | 2943.520 .. 2948.447 | 7 | 7 |
| DtoH 后 | 3073.596 .. 3078.527 | 7 | 7 |

tiny launch API bracket 为 `0 .. 18.409 μs`；end Event API 在 `2871.690 .. 2879.145 μs`。
第一个 Event query 已返回 CUDA_SUCCESS（调用结束 `2937.387 μs`），因此本次 **没有观察到 Event NOT_READY**。
不能用它证明 `(5,6)` 时 tiny 尚未完成。

| 核验 | 结果 |
|---|---|
| 有效 channels / read rounds / volatile u32 loads | 8 / 32 / 1,024 |
| 完整观测窗口 | 6.424481 ms |
| GET/PUT 变化的 channel | 1 个；另外 7 个不变 |
| 256 组四读值中重复值发生变化 | 0；所有值位于所报 1,024-entry 范围内 |
| 原 bridge / observer 一致的 ioctl envelopes | 前后各 1,307；读窗口没有新 syscall record |
| 新增 RM control / 额外 mapping / rewind | 0 / 0 / 0 |
| Context/group/USERD mapping 跨窗口一致 | 是 |
| 被测 CUDA API / 原 tiny 输出校验 | 返回成功 / 通过 |

Event elapsed 为 **2.70438 ms**，包含主动插入的 host 读窗口，不能称为 tiny kernel execution time。
原 tiny 的两个 GPU `%globaltimer` 读数相等，本次无法由它们分辨单 kernel 时长。
这里测可观测性，不报告 takeover latency 或硬件更新时延；GET 变化只能定位在 host 采样之间。

## 6. Proven / not proven

| 步骤 | 证明了什么 | 没证明什么 | 下一步最小验证 |
|---|---|---|---|
| Live binding + 字段读取 | 当前 owned compute TSG 的既有 USERD 可安全完成本次有界读取 | 所有运行环境/隐藏 syscall 的覆盖 | 保持当前 gates 与有限结论 |
| 分开 Event / launch 边界 | 本样本 Event 与 tiny 周围均有 PUT 推进；不能将总 delta=3 全算给 tiny | 单次 CUDA logical submission 永远等于一个 entry | 建立具体 entry 的可验证对应 |
| 无 CUDA API 窗口 | 读到 GET 从 5 到 6，而 PUT 保持 6 | GET 的精确消费/预取语义、kernel 开始或完成时刻、RAMFC 同步性 | 不用这些读数直接推断 rewind 效果 |
| Event / DtoH / final snapshot | tiny 结果正确，Context 和映射保留至结束 | old sentinel 是否未消费、H1/H2/H3 | mapping 成立前不恢复 rewind |

代码验证：observer 可选构建和原无 RM 依赖构建通过；原 16 个 mapping CPU 测试及新增 11 个 timing/读数反例测试通过。
对实际原始证据的临时副本注入错误 UUID、owner、control 计数、缺失读取、窗口内 syscall、虚假结果完成，均被拒绝；原始证据未改。

## 7. Next minimum experiment

**只推进 logical submission → GPFIFO entry 的关联。**
先核对本次活动 channel 的 ring 是否有已存在、可按同等 ownership gates 验证的 CPU 映射；有则可围绕一次隔离提交
对照 entry 内容/sequence 与 PUT delta。没有合法 reader 就停在这一层，不新增 mapping 或 driver interface。
本轮 `5 → 6` 的单次 PUT delta 是关联线索，不是已经完成的 sentinel mapping。

**现有 stack 已提供合法的 USERD 字段观测机制；仍不足以让 runtime 判断某个 application epoch 是否处于可 rewind 的未消费区间。**

## 复现与原始证据

构建依赖与固定 source/bridge 要求同 [USERD_MAPPING_EXPERIMENT.md](USERD_MAPPING_EXPERIMENT.md)。
`Interception` 工作树已有未提交 bridge 源码；本轮未修改它，不能仅用其 Git HEAD 代替所用库/源码状态。

```bash
cmake --build build-userd -j 4
python3 -m unittest discover -s tests -p 'test_userd_*.py' -v
python3 scripts/run_userd_mapping.py --mode userd-progress --gpu 0 \
  --build build-userd --output results/userd_progress/new_run.csv
python3 scripts/analyze_userd_progress.py results/userd_progress/new_run.csv.userd \
  --public-dir results/userd_progress_public/new_run
```

真实设备节点需在可访问 GPU 的宿主环境执行；启动器校验选定 GPU 空闲及版本，只操作该 GPU，不覆盖旧文件。
直接执行 binary 而不经启动器不会获得 live read plan，5 秒后停止，不读取任意地址。

原始记录仅本地保存：`results/userd_progress/smoke_01.csv.userd/` 中的 `observations.jsonl`、`read.plan`、
`tiny.json`、两组 identity/memory/diagnostics/maps，以及邻接 launcher/exit/stdout/stderr/metadata sidecars。
公开结果仅有相对时间、ring 字段值、ordinal 和核验状态，不含 UUID、PID/TID、handles、CPU/GPU 地址或主机路径。
