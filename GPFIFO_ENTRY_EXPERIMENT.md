# 最小实验：isolated CUDA launch → exact GPFIFO entry

2026-09-22。**两次独立进程 smoke 成功：本次隔离 tiny launch 的 entry-level binding 已闭合。**
普通进程依次对应 `ring[7]`、`ring[8]`；第二进程在真实 ring 边界依次对应 `ring[1023]`、`ring[0]`。
每次完整 1,024-entry ring 仅一个 slot 改变，恰为 launch 前的 PUT，PUT 模 1,024 增加 1。
这不是“任意 application epoch 已有稳定 queue-entry API”的结论。

后续：[GRAPH_ENTRY_EXPERIMENT.md](GRAPH_ENTRY_EXPERIMENT.md)把一次已预热 GraphExec launch 作为 request，
验证了 compute-only Graph 的完整 entry set；下文保留 isolated direct launch 阶段的边界。

## 1. Research question

在已有合法 USERD 读数的基础上，一次明确的 CUDA Driver logical submission 能否唯一对应到当前 channel 的新增 GPFIFO entry？
本轮只验证关联，不恢复 rewind，也不验证尚未消费或硬件未预取。

| 项目 | 定义 |
|---|---|
| Hypothesis | 当前 libcuda 路径的一次隔离 tiny launch，可由唯一 channel 的 PUT delta 和唯一 ring slot 变化绑定 |
| Baseline | 同一 channel、同一 generation，无 CUDA API 时 ring 与 PUT 稳定 |
| Variable | 两个 ring 快照之间仅执行一次原 `cuLaunchKernel(tiny_kernel)` |
| Metrics | 全部 8 个 channel 的 GET/PUT、完整 ring 前后内容、变化 slot 集、API/read 时间边界及 ownership |
| Expected observation | 唯一 channel 的 PUT 模 N 加 1，完整 ring 仅旧 PUT 所指 slot 改变；下一 launch 对应下一 slot，包括 wrap |
| Failure rule | 无合法现有映射、身份变化、多个候选、PUT delta 不为 1、读数不稳定或需新增 mapping/control，均停止并拒绝唯一关联结论 |

## 2. System abstraction

Application runtime 的 launch/epoch 不是 driver 的 ring index。本次已验证的链条是：

`isolated cuLaunchKernel → 唯一 PUT 变化的 owned compute channel → PUT delta → 唯一改变的 ring[index]`

这里的 channel 来自 **libcuda 创建的 compute TSG**；不是 UVM 自有 channel 的 `cpu_put/gpu_get` 软件游标。
Channel ordinal 只是本次完整成员列表的编号，不能当作跨进程硬件 channel ID。
一个多操作 epoch 可能包含多个 launch、Event、copy 或隐式提交；本实验没有替它们建立完整的 entry 集合。

## 3. Existing mechanism / legal mapping contract

仍采用未修改的 NVIDIA 595.58.03 官方源码，commit `db0c4e65c8e34c678d745ddb1317f53f90d1072b`。
原 Interception bridge 及 owner/generation/physical UUID gates 未修改，project active-control budget 为 0。

| Object → ownership | 已有 interface / state | 如何绑定当前 channel |
|---|---|---|
| C56F compute channel → 当前 client/TSG | allocation 的 `gpFifoOffset/gpFifoEntries/hContextShare` | 保留正常 allocation；hObjectBuffer 已不使用，GPU VA 本身不可直接解引用 [S1] |
| Context-share / VASpace → 同 client、当前 parent generation | 已有 9067 context-share 与 90f1 VASpace allocation | 本布局 channel 的 hVASpace=0，由显式 context-share 绑定 VASpace [S2] |
| 已注册 GPU VASpace / channel → 本进程 UVM fd 与 RM client | 正常 `UVM_REGISTER_GPU_VASPACE`、`UVM_REGISTER_CHANNEL` | 与原 RM fd、channel handle、物理 UUID、allocation 生命周期关联 [S3] |
| Ring GPU VA → 该 VASpace 中的 Memory object | 已发生的 `UVM_MAP_EXTERNAL_ALLOCATION`，含 client/memory/base/offset/UUID | 完整 ring 位于唯一 GPU mapping；匹配已验证 USERD Memory 的当前 generation [S3][S4] |
| 同一 Memory → libcuda 已有 CPU mapping | 已有 RM map → mmap 返回 → 当前 readable/shared VMA | 由 Memory offset 推导 CPU 地址，完整 ring 均在此 VMA 内；不依赖 CPU/GPU 地址数值相等 |

在本次布局中，ring 与 USERD 共享已映射的 2 MiB Memory object；活动 ring 的 Memory offset 为 12,288，
ring 长 8,192 bytes（1,024 × 8）。该布局是实测前提，不是 CUDA 通用保证。[8-byte 格式定义 S5]
记录器只转发、被动保留 libcuda 原本发出的 UVM 请求，不构造新请求、不导出新 ioctl、不建立新 GPU/CPU mapping。
未知 ABI、映射失效/替换、fd reuse、未知 UVM mutator、另一 Memory 布局均拒绝继续，不扫描其他内存。

因此当前分类为 **`already_mapped_state_available`**，且仅覆盖已验证实例。
这条绑定利用现成 driver 机制与本地被动审计，不是 CUDA 对 application runtime 提供的公开、稳定 entry 查询接口。

## 4. Observability

实现位于 [userd_observer.cpp](src/userd_observer.cpp)、[userd_mapping.hpp](src/userd_mapping.hpp)、
[gpfifo_binding.py](scripts/gpfifo_binding.py) 和 [analyze_gpfifo_entry.py](scripts/analyze_gpfifo_entry.py)。
沿用原 read-plan 协议：parent 校验完整 live binding 后发布私有计划，child 再核对当前 PID、registry、
完整 group/member generation 与 observer sequence。之后才读取 ring；离线分析再核对窗口前后身份和映射。

每个 ring 快照：读取全部 channel 的 GET/PUT → 完整 ring 连读两遍 → 再读全部 GET/PUT。
每个 entry 仅按不透明的 aligned volatile 64-bit 内容记录，不解码 pushbuffer 或指令语义。
两遍内容必须相同；快照两侧 PUT 必须稳定。CPU 时间均为 `CLOCK_MONOTONIC_RAW`，每次 API 与快照都有 begin/end。
这是顺序重复读取，不是原子硬件快照；LFENCE 不提供 GPU 新鲜度或未预取保证。

读窗口上限为 256 rounds / 3 s；捕获到额外 syscall record 即使计划失效。
GPU completion query 上限 3 s，launcher 外层期限 45 s。成功后 Context 保留至末尾快照，由进程退出释放；
显式 `cuCtxDestroy`、reset、preempt、disable/enable、rewind 调用均为 0。
记录器与原 bridge 的覆盖限于被拦截的 libc 路径，不保证捕获 hidden/direct syscalls；本实验不外推这种覆盖。

## 5. Experiment

环境：NVIDIA A100-PCIE-40GB，cc 8.0；KMD/libcuda **595.58.03**，CUDA Driver API **13020 / 13.2**，
Toolkit **12.8.61**，Linux x86_64，MIG Disabled，compute mode Default。选定 GPU 开始时无 compute 进程。
A/B/controller 固定 CPU 0/1/2；A/B 各有长期 owner thread 与独立 Context，B 预热后 standby。
原 tiny kernel、device source 与 cubin 不变；没有长 kernel、cancellation polling 或 profiler。

每个成功进程先完成原 USERD progress 探针，用全体 8 个 channel 的 PUT 变化发现活动 channel。
该 prelude 包含 Event/校验，与测量窗口分开。验证 ring mapping 后执行：

1. Idle ring 快照。
2. Before 快照 → **一次 tiny launch** → after 快照。
3. 无任何 CUDA API，读取 GET/PUT 8 轮（轮间 sleep 请求 100 μs）→ settled ring 快照。
4. 再执行一次同样的隔离 launch/读窗口；下一 slot 应为前一 slot 模 N 加 1。
5. 结束只读窗口后，才记录 end Event、有界查询、回读并校验**最后一次** tiny 的输出；随后检查 Context/映射 identity。

Wrap 进程在步骤 1 前通过 **1,016 次普通 tiny launch** 将 PUT 从 7 推进到 1,023；准备阶段不写 PUT、不发 control。
准备次数单独记录，不把这些 launch 纳入一一映射样本。全过程有界，测量的两次 launch 间无 Event/copy/query。

### 尝试记录与一次校验模型修正

| Attempt | 结果 | Ring 读取 |
|---|---|---:|
| mapping_audit | 仅被动 metadata + 原 USERD progress，确认已有 GPU mapping 关联候选 | 0 |
| smoke_01 | 新增 ring gate 在放行前拒绝 `UVM_MM_INITIALIZE` 的 warning；launcher exit 125 | 0 |
| smoke_02 | 普通两次隔离 launch 关联成功，exit 0 | 7 个双读快照 |
| smoke_03_wrap | 独立进程、真实 wrap 的两次隔离 launch 关联成功，exit 0 | 8 个双读快照 |

首次拒绝来自新校验器误要求 MM 初始化只能返回 NV_OK。实际返回 `0x10006 = NV_WARN_NOTHING_TO_DO`：
固定源码明确说明 va_space_mm 未启用时无需 secondary MM fd，该 fd 可以释放。[S6]
校验器据此仅接受 NV_OK 或这一明确 warning；primary UVM fd 始终必须存活，NV_OK 时 secondary fd 也必须存活，
其他状态仍拒绝。原 owner/generation/UUID gates 未放松。
这是本地模型修正，不是忽略 CUDA/RM 故障；失败记录保留，未计入两次成功 smoke，也未用 reset/destroy 恢复。

### 成功样本

| Process / launch | PUT 前 → 后 | 唯一变化 slot | PUT delta mod 1024 | 改变的 slot 数 | 无 API 窗口 GET |
|---|---:|---:|---:|---:|---:|
| smoke_02 / 1 | 7 → 8 | 7 | 1 | 1 | 全部为 8 |
| smoke_02 / 2 | 8 → 9 | 8 | 1 | 1 | 全部为 9 |
| smoke_03_wrap / 1 | 1023 → 0 | 1023 | 1 | 1 | 全部为 0 |
| smoke_03_wrap / 2 | 0 → 1 | 0 | 1 | 1 | 全部为 1 |

每次完整 ring 其余 1,023 个 slot 不变，另 7 个 channel 的 PUT 不变；无 API 窗口 ring/PUT 保持不变。
两个 ring 读取窗口分别为 **26.270469 ms / 36.175602 ms**；共有 14,336 / 16,384 次 u64 ring load。
Post ring 快照约需 3 ms，结束时 GET 已追上。因此这两次实验**没有捕获无 API 窗口内的 GET 跳变**，
仅观测到末值；不能把观测延迟解释成 GPU 完成时间。

每个进程原 bridge 与 observer 的 RM ioctl envelopes 均前后 1,307 条一致；正常 UVM 请求被动记录 178 条。
新增 project RM controls / 额外 mappings 均为 0。前后 Context/group/Memory/mapping identity 一致。
四次被测 launch 返回 CUDA_SUCCESS，两个进程最后 tiny 的结果校验通过；第一 tiny 未单独做输出回读。

独立性由两个实际独立启动的 owner PID 确认。两进程的 registry counter 及 CPU arena 地址数值恰好相同，
不能据此当作同一对象，也不能宣称地址随机化；每次均从本进程全新 capture 重建 binding，没有复用旧 plan。
smoke_02 使用 V1 ring plan；随后 V2 只增加 wrap 准备所需 initial PUT，smoke_03_wrap 使用 V2。
两版的 isolated launch / 双读 / 唯一 slot 判据相同，当前分析器保留两版验证。

Observer 可选构建与原无 RM 依赖构建通过；20 个新增 CPU 测试及原 27 个测试通过。
对真实 capture 临时副本注入错误 plan 地址、Memory、owner、多个变化 slot、错误 PUT delta、窗口内 syscall、
错误 API 边界，7/7 被拒绝。原始证据未改。

## 6. Proven / not proven

| 步骤 | 证明了什么 | 没证明什么 | 下一步最小证据 |
|---|---|---|---|
| GPU VA → Memory → CPU VMA | 当前 libcuda compute channel 的 ring 已有合法可读映射 | 其他 CUDA/driver/分配布局可通用使用 | 每个新实例重新验证绑定 |
| 全 ring 差分 + PUT + 唯一 API 窗口 | 本次 4 个隔离 tiny launch 各唯一对应一个新增 slot | 每个 CUDA API 永远对应 1 entry；entry payload 的完整语义 | 如研究 epoch，列明其全部逻辑提交 |
| 两进程及 1023 → 0 | 连续 launch 对应下一 slot，包含真实 wrap | 多次 wrap 的绝对 sequence、长期并发提交归属 | 不把裸 index 当作永久 epoch ID |
| GET/PUT 观察 | 本次 ring 关联可与同 channel progress 配对 | GET=kernel 完成、未预取、USERD=RAMFC 或 rewind 安全 | 本轮不作这些推断 |

## 7. Next minimum experiment

**若 application epoch 就定义为本次隔离单 launch，其 exact-entry 关联已闭合。**
对完整 application epoch，缺失的最小证据是：明确列出该 epoch 的全部 Driver submissions，
验证它们对应的 entry 集合完整且排他，尤其区分 Event/copy/隐式提交。当前四个 launch 不能替代这一覆盖证明。
这是后续最小 mapping 问题，本轮不继续实现 runtime/cancellation/rewind。

## 复现与证据

构建方式见 [USERD_MAPPING_EXPERIMENT.md](USERD_MAPPING_EXPERIMENT.md#复现)。使用同一固定 NVIDIA checkout、
已匹配的 Interception source/library；外部 bridge 工作树已有未提交内容，Git HEAD 不能单独代表构建输入。

```bash
cmake --build build-userd -j 4
python3 -m unittest discover -s tests -p 'test_userd_*.py' -v
python3 -m unittest discover -s tests -p 'test_gpfifo_entry.py' -v

# 两次命令分别创建独立进程；需要宿主 GPU 访问，不覆盖旧证据。
python3 scripts/run_userd_mapping.py --mode gpfifo-entry --gpu 0 \
  --build build-userd --output results/gpfifo_entry/new_plain.csv
python3 scripts/analyze_gpfifo_entry.py results/gpfifo_entry/new_plain.csv.userd \
  --public-dir results/gpfifo_entry_public/new_plain
python3 scripts/run_userd_mapping.py --mode gpfifo-entry --entry-wrap --gpu 0 \
  --build build-userd --output results/gpfifo_entry/new_wrap.csv
python3 scripts/analyze_gpfifo_entry.py results/gpfifo_entry/new_wrap.csv.userd \
  --public-dir results/gpfifo_entry_public/new_wrap
```

原始记录仅本地保留在 `results/gpfifo_entry/{mapping_audit,smoke_01,smoke_02,smoke_03_wrap}.csv.userd/`，
含 `observations.jsonl`（完整原始 entry 双读）、identity/memory/diagnostics/maps、read/ring plan、binding、API 边界；
邻接 sidecars 保留命令、运行时版本、退出状态和实际 binary/library/script 指纹。未用当前源码指纹追溯替换旧运行记录。

去标识输出：[attempts.json](results/gpfifo_entry_public/attempts.json)、
[普通 summary](results/gpfifo_entry_public/smoke_02/summary.json) / [entries.csv](results/gpfifo_entry_public/smoke_02/entries.csv)、
[wrap summary](results/gpfifo_entry_public/smoke_03_wrap/summary.json) / [entries.csv](results/gpfifo_entry_public/smoke_03_wrap/entries.csv)。
公开 CSV 只含相对时间、ordinal/index/GET/PUT、计数与 opaque bytes 的 SHA-256，不含原始 entry、UUID、PID/TID、
RM handles、CPU/GPU 地址或主机路径。原始 entry 可能含 GPU 地址，因此不上传。

## 固定源码依据

- [S1：channel allocation](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/common/sdk/nvidia/inc/alloc/alloc_channel.h#L299)。
- [S2：channel 的 context-share VASpace](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/nvidia/src/kernel/gpu/fifo/kernel_channel.c#L1029)、[context-share ownership](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/nvidia/src/kernel/gpu/fifo/kernel_ctxshare.c#L108)。
- [S3：UVM registration 与 external mapping 参数](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/kernel-open/nvidia-uvm/uvm_ioctl.h#L304)。
- [S4：GPU VASpace / Memory mapping 实现](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/kernel-open/nvidia-uvm/uvm_map_external.c#L971)。
- [S5：GPFIFO entry 大小及格式](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/common/sdk/nvidia/inc/class/clc56f.h#L265)。
- [S6：MM fd 可选契约](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/kernel-open/nvidia-uvm/uvm.c#L78)。
