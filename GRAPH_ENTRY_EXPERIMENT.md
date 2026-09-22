# 最小实验：GraphExec launch → exact GPFIFO entry set

2026-09-22。**Yes，范围限于本配置下已 instantiate、upload、warmup 的单 stream、compute-only GraphExec。**
两次独立进程中，`graph-1`、`graph-3`、同一 GraphExec 的 `graph-3-repeat` 都各对应 **1 个 entry**；
`direct-3` 则对应 **3 个 entries**。每次所有变化 slot 恰好覆盖 PUT 推进区间，区间内无缺项、区间外无变化。
同一个三 node GraphExec 的 repeat 对应紧随其后的 entry，实际 `1023 → 0` wrap 也成立。

## 1. Research question

把一次 application request 定义为一次固定 GraphExec launch：能否完整、排他地识别它在 driver queue 中对应的 entry set？
本实验不预设 entry 数，不把 node 与 entry 逐一对应，也不研究硬件 GET 的消费语义。

| 项目 | 定义 |
|---|---|
| Hypothesis | 预热 GraphExec launch 的 PUT 推进区间，与完整 ring 的变化 slot 集唯一一致；同一 GraphExec 可重复绑定 |
| Baseline | 无 CUDA API 的 idle/settled 窗口；一次 direct tiny；三个按顺序独立提交的 direct tiny |
| Variable | 同一原 tiny kernel 经 direct launch 或预先捕获的 GraphExec launch 提交 |
| Metric | 所有 owned compute channels 的 PUT delta、完整 ring 差分、区间内缺项/区间外变化、GraphExec identity、逐 node 新鲜正确输出 |
| Expected observation | 变化集合恰好等于 `[old_put,new_put)` modulo ring size；后续 Graph launch 对应后续集合，含实际 wrap |
| Interesting outcomes | 任意 entry 数、多 compute channels、不同 launch 的 entry 数变化或延后 publication 都保留为事实，不猜测内部原因 |

## 2. System abstraction

| 层次 | 本轮对象 / 证据 |
|---|---|
| Application semantics | 一次 request/epoch = 一次测量窗口内的 GraphExec launch |
| Application runtime | 单 stream capture 的一个 kernel，或线性 DAG `K1 → K2 → K3`；固定 GraphExec，不 update |
| System runtime | `cuGraphLaunch` 的实际提交行为；公开 CUDA API 不给出 GPFIFO entry 数 |
| Driver | 同 owner/generation/UUID 下的 TSG、全部 8 个 compute channels、已有 USERD/ring 映射及 PUT |
| Hardware | GET 仅随快照原样记录，不用来判断 node 完成或推断预取 |

本轮验证的是 **Graph epoch → owned compute-channel entry set**，没有建立 payload/node 解释器，
也没有构造面向任意 CUDA 操作的通用 epoch runtime。

## 3. Existing mechanism / implementation

沿用 [isolated entry 实验](GPFIFO_ENTRY_EXPERIMENT.md)的 NVIDIA 595.58.03 固定源码
`db0c4e65c8e34c678d745ddb1317f53f90d1072b`，现有 Interception bridge 未修改。
USERD/ring 的 owner、generation、物理 UUID、Memory ownership、UVM GPU mapping、RM map/mmap、live VMA 和 fd 生命周期检查全部保留。
观察器只读取 libcuda 已建立的映射，不新建 mapping、不制造 RM control；未使用 reset/context destroy 恢复。

新增 [graph_entry.hpp](src/graph_entry.hpp) 和 [analyze_graph_entry.py](scripts/analyze_graph_entry.py)，
扩展原 observer，使每个快照读取**全部 8 个已验证 compute ring**。不预先假定 Graph 与 direct 使用同一个 channel。
如果任何所需 ring 没有现成合法 CPU reader，就停止；本次 8/8 均通过。

Graph 使用 Driver API：`cuStreamBeginCapture` → 原 `cuLaunchKernel` ×1 或 ×3 → `cuStreamEndCapture`
→ `cuGraphInstantiate(flags=0)` → `cuGraphUpload` → 一次 warmup launch 与结果验证。
通过 `cuGraphGetNodes/GetEdges`、node type 与 kernel params 检查实际 DAG 是仅含原 tiny 的 1 node / 3 node 线性链。
这些操作全部在第一个 Graph 实验测量窗口之前完成。[CUDA 12.8 Graph API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__GRAPH.html)、
[stream capture API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__STREAM.html)。

原 device source、`tiny_kernel` 和 cubin 不变。Graph-1 使用结果槽 0；Graph-3 的三个 node 使用槽 1/2/3，
每槽有不同固定 token。Direct baseline 使用相同 kernel、stream、参数与相应输出槽。
每轮输出验证同时检查正确 token、GPU timestamp 顺序，以及每个 node 的 start timestamp 严格晚于上一次已验证 invocation。
因此固定 token 的预热旧值不能被误判为本次执行成功。Timestamp 只用于新鲜度，不作延迟或 GET 语义判断。

## 4. Observability / protocol

两条长期 owner thread 各维持 A/B Context；B 预热后 standby。原 USERD progress prelude 先独立完成，
之后准备 Graph。每个条件重新发布 owner/generation/sequence 绑定的私有 all-ring read plan，窗口后再验证同一 binding。

条件依次为 `direct-1`、`direct-3`、`graph-1`、`graph-3`、`graph-3-repeat`。
每个条件有独立的：

1. Idle 快照：全部 GET/PUT、全部 ring 双读、再次全部 GET/PUT。
2. Before 快照，结构同上。
3. 一次 `cuGraphLaunch`；direct-1 为一次 `cuLaunchKernel`，direct-3 为三个独立 `cuLaunchKernel`。
4. After 完整快照。
5. 无 CUDA API，等待约 2 ms 后取 settled 完整快照，检查有无延后 ring/PUT 变化。
6. 结束读窗口；此后才有界 `cuStreamQuery`、`cuMemcpyDtoH` 和逐 node 验证，最后检查 identity/mapping。

**Before/after 之间没有 Event record/query、copy、graph update、instantiate 或其他 CUDA API。**
Result query/copy 位于两个条件之间；其前后的 ring/PUT 变化单独列入 `validation_gap_channels`，不算作 Graph entry。
本次所有这些 gap 的 PUT delta 与 ring 变化均为 0，所以 Graph-3 repeat 的 entry 确实紧接第一次。

第二进程在 graph-3 的 before 快照之前，通过 **1,007 次普通 tiny launch** 将 PUT 从 16 推进到 1,023。
准备阶段独立记录，不属于被测 Graph epoch；不写 USERD、不使用 driver control。

每个快照包含 8 × 1,024 entries 的完整双读。两遍必须相等，快照前后 PUT 必须稳定；
GET 可变化，不能将整组顺序读取看成原子快照。读窗口预算 256 rounds / 3 s，实际每条件 4 rounds；
completion query 上限 3 s，进程外层期限 45 s。CPU 边界均用 `CLOCK_MONOTONIC_RAW`。
被拦截的生命周期/syscall 事件出现在 read window 内会使 audit 失效；不放松原权限或身份 gates。

## 5. Experiment / results

环境：NVIDIA A100-PCIE-40GB，cc 8.0；KMD/libcuda **595.58.03**，CUDA Driver API **13020 / 13.2**，
Toolkit **12.8.61**，Linux x86_64；MIG Disabled，compute mode Default；A/B/controller 固定 CPU 0/1/2。
两次运行前所选 GPU 空闲，结束后也无 compute 进程；没有 profiler、额外 active RM control 或其他 GPU 的操作。

两个独立进程全部完成，exit 0；共 10 个条件记录，其中 6 次被测 Graph launch。
每次只有当前成员列表 **ordinal 1** 的 PUT/ring 改变，其余 7 个 compute ring 全部不变；ordinal 不当作跨进程 channel ID。

| 条件 | CUDA API 次数 / kernel 数 | 普通进程：PUT 前→后；entry set | wrap 进程：PUT 前→后；entry set | 完整、排他 / 输出正确且新鲜 |
|---|---|---|---|---|
| direct-1 | 1 / 1 | 11→12；{11} | 11→12；{11} | 两次均成立 |
| direct-3 | 3 / 3 | 12→15；{12,13,14} | 12→15；{12,13,14} | 两次均成立 |
| graph-1 | 1 / 1 | 15→16；{15} | 15→16；{15} | 两次均成立 |
| graph-3 | 1 / 3 | 16→17；{16} | 1023→0；{1023} | 两次均成立 |
| graph-3-repeat | 1 / 3 | 17→18；{17} | 0→1；{0} | 两次均成立 |

对每个 channel 用 `delta=(new_put-old_put) mod 1024` 构造有序 interval，然后比较完整 ring 的变化 slot 集。
**所有 10 个条件的区间内未变化 slot=0，区间外变化 slot=0；settled 窗口没有延后 publication。**
同一三 node GraphExec 的两次被测 launch identity 不变、entry 数均为 1；两个独立进程也一致。
这不解释 entry 如何实现三个 node，也不声称 node 与 entry 逐一对应。

每进程 20 个 all-ring 双读快照，共 **327,680 次 u64 ring load**；每条件窗口约 **81–96 ms**，
单个全 ring 快照约 20 ms。这是关联实验，不报告 Graph launch 性能或硬件更新时延。
所有前后快照原 bridge/observer 的 RM ioctl envelope 数均为 1,307 且逐项一致，
正常 UVM 请求各被动捕获 180 条；新增观察器 RM controls / 额外 GPU mappings 为 0。

两个 GPU smoke 均一次通过。Observer 可选构建与原无 RM 依赖构建通过；新增 14 个 entry-set CPU 反例测试、
原 47 个 mapping/progress/entry 测试通过。真实 capture 临时副本的 9 项检查也符合预期：
错误 GraphExec、旧/缺失/错误 node 输出、额外 CUDA API 和错误 plan 被拒绝；
区间内缺 entry、区间外额外 entry、inactive channel 变化被分类为 unresolved，保留事实而不是强行判成功。

## 6. Proven / not proven

| 步骤 | 证明了什么 | 没证明什么 | 下一步最小实验 |
|---|---|---|---|
| Capture/instantiate/upload/warmup + DAG/参数校验 | 本次 Graph 是既定 1/3 node compute-only DAG，测量只 launch 固定 GraphExec | 任意 DAG 的提交形态 | 保持结论限于此布局 |
| 全部 owned compute rings + PUT 区间 | 本次 Graph launch 的 changed-entry set 完整、排他；三个 node 均产生新鲜正确输出 | 公开稳定 API、任意内部/隐藏提交路径的覆盖 | 若扩展，单独测试下一类 node |
| 同 GraphExec repeat + 独立进程 wrap | 本次 entry 数稳定为 1，repeat 紧随前一 entry，wrap 成立 | 所有 GraphExec/驱动/规模下稳定，或长期多圈 sequence 标识 | 不从有限样本推广 |
| 结果校验与 GET 分离 | 完成来自 CUDA stream query 与逐 node 回读 | GET consumption、prefetch、任何取消控制语义 | 本轮不研究这些问题 |

**一次 GraphExec launch 可以被完整、排他地绑定到 entry set：本轮答案为 yes，适用于所测 warmed compute-only GraphExec 与已有合法 compute-ring 观测范围。**
记录器仍只覆盖被拦截的 libc 路径与本进程验证过的 compute TSG；不是对所有隐藏 syscall、其他 engine 或任意系统配置的保证。

## 7. Next minimum experiment

按既定边界，下一阶段才加入一种 memcpy/event node，先验证对应 channel/engine 的合法观测范围，
再测试 `Graph epoch → multi-channel / multi-engine entry set`。本轮没有执行这一步，也没有扩展任何控制机制。

## 复现与原始/去标识证据

依赖和可选构建沿用 [USERD_MAPPING_EXPERIMENT.md](USERD_MAPPING_EXPERIMENT.md#复现)。
实际 bridge source/library 使用匹配的既有版本，未修改外部仓库；不能仅用外部 Git HEAD 替代其已有工作树状态。

```bash
cmake --build build-userd -j 4
python3 -m unittest discover -s tests -p 'test_*entry.py' -v
python3 -m unittest discover -s tests -p 'test_userd_*.py' -v
python3 scripts/run_userd_mapping.py --mode graph-entry --gpu 0 \
  --build build-userd --output results/graph_entry/new_plain.csv
python3 scripts/analyze_graph_entry.py results/graph_entry/new_plain.csv.userd \
  --public-dir results/graph_entry_public/new_plain
python3 scripts/run_userd_mapping.py --mode graph-entry --entry-wrap --gpu 0 \
  --build build-userd --output results/graph_entry/new_wrap.csv
python3 scripts/analyze_graph_entry.py results/graph_entry/new_wrap.csv.userd \
  --public-dir results/graph_entry_public/new_wrap
```

真实 GPU 访问在宿主环境执行，launcher 校验 GPU/profile/空闲状态，拒绝覆盖旧结果。
原始记录仅本地保存于 `results/graph_entry/{smoke_01,smoke_02_wrap}.csv.userd/`：完整 ring 双读与 GET/PUT，
每条件前后 identity/memory/diagnostics/maps、plan/binding、launch/result JSON、初始/最终 GraphExec identity；
邻接 launcher/exit/metadata 文件保留版本、运行命令、实际 binary/library/script 指纹和 CPU 时间边界。

公开数据：[普通 entries.csv](results/graph_entry_public/smoke_01/entries.csv)、[summary](results/graph_entry_public/smoke_01/summary.json)；
[wrap entries.csv](results/graph_entry_public/smoke_02_wrap/entries.csv)、[summary](results/graph_entry_public/smoke_02_wrap/summary.json)。
[环境与核验摘要](results/graph_entry_public/validation_summary.json)记录两次独立进程及反例检查。
CSV 中 `entry_set` 使用 `channel_ordinal:slot`；`kernel_nodes` 对 graph 为 node 数，对 direct 为独立 kernel launch 数。
summary 保留每个 channel 的期望区间、变化集合、缺项与越界集合。
公开结果只含相对时间、索引、计数与核验状态；UUID、PID/TID、RM/Graph handles、CPU/GPU 地址、原始 entry 内容均留在本地。
