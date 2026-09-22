# FIFO GP_PUT rewind：CUDA old-epoch sentinel 实验（2026-09-22）

**结果：两次 smoke、5 对 matched trials，共 12/12 个样本均为 `old_queue_preserved`。**
在 A100 / 595.58.03 上，`bRewindGpPut=true` 被 RM 接受，但 old sentinel 仍在 enable 后执行，
且在提交 new sentinel **之前**已经被 host 观测到。原 Context 的新 stream 能完成正确的 new sentinel。

这证明本次 CUDA 提交路径没有获得 old-epoch sentinel 失效的效果；
**不能据此断言 RM 没有回退 GP_PUT，也不能断言所有尚未消费的 GPFIFO entry 都不可丢弃。**
本实验没有直接观测 GP_GET/GP_PUT 或 sentinel 对应的 GPFIFO entry。

## 1. 研究问题

stock `NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS` 的 `bRewindGpPut=true`，
能否保留 CUDA Context，同时让已 enqueue、尚未执行的 old-epoch sentinel 不再执行？

| 项目 | 定义 |
|---|---|
| Hypothesis | FIFO rewind 能移除排队 sentinel，并保留新 stream 的提交/完成能力 |
| Baseline | 已验证的 FIFO hold/enable，`bRewindGpPut=false` |
| Variable | disable 时仅将 `bRewindGpPut` 从 false 改为 true |
| Metrics | old/new token、A/old/new Event 状态、CUDA 错误、Context/buffer identity、RM 返回与 host 边界 |
| Expected observation | 若得到所需效果，old token 应缺席，同时同 Context 的新 stream 完成正确 new token；不能只看 `NV_OK` |

## 2. System abstraction

- **Application epoch**：old/new 是唯一 token 标记的逻辑请求；各有独立结果槽，不靠复用地址或清零掩盖迟到写入。
- **CUDA Driver API**：A 的原 stream 排队长 kernel、A end Event、old sentinel、old sentinel Event。
  enable 后创建另一条 `CU_STREAM_NON_BLOCKING` stream，提交 new sentinel 和独立 Event；不插入对 old Event 的依赖。
- **RM**：控制对象仍为本进程 A compute TSG 的完整 8-channel 列表；ioctl 发到已验证 subdevice。
- **Hardware/firmware**：GPFIFO 指针和预取/消费位置不是 CUDA stream 的逐 kernel 逻辑队列。

**CUDA kernel 尚未执行，不等于承载它的 GPFIFO entry 尚未消费。**
本实验能确认 old sentinel 的 CUDA enqueue 已返回、A 已进入长 kernel、sentinel 尚未产生 token；
不能确认 FIFO front end 此时尚未取得对应命令。

## 3. Mechanism contract

采用与前一实验相同的官方 595.58.03 头文件 commit：
[`ctrl2080fifo.h`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080fifo.h)。
已读取并核对本地匹配 checkout，未修改头文件、driver 或 firmware。

- `bDisable=true, bOnlyDisableScheduling=false, pRunlistPreemptEvent=NULL`：同步抢占所列 channels，并禁止后续调度，直到 enable。
- `bRewindGpPut=true`：把 channel RAMFC 中的 GP_PUT 设为 GP_GET。
- 契约不承诺撤销已消费/预取的命令、终止 in-flight kernel、修复被丢弃的 CUDA Event，或让 CUDA 软件提交指针自动一致。
- enable 使用 `bDisable=false, bRewindGpPut=false`。

直接复用未修改的 `Interception/active-preempt` bridge：
false 调 `ap_bridge_set_channels_enabled(..., 0, ...)`；true 调现有 `ap_bridge_discard_queued(...)`。
这只是实验入口的函数名，不作为结果结论。完整身份、独立 RM UUID 查询、generation、状态和预算检查均保留。
匹配 header/库的 FIFO rewind mock contract 与 C ABI 测试通过。

实际 journal 核验：所有 measured disable 的 536-byte payload 在同一 matched 进程中，
**除 offset 9 的 rewind 位外完全相同**；channel/client 列表完整，only-scheduling 为 false、event 指针为 NULL。
没有额外 Group PREEMPT。

## 4. Experiment

环境：NVIDIA A100-PCIE-40GB、108 SM、cc 8.0；KMD/libcuda **595.58.03**，
Driver API **13.2 / 13020**，Toolkit **12.8.61**。MIG Disabled，compute mode Default。
所选 GPU 开始时空闲；没有 MPS、全局调度修改、GPU reset 或对其他 GPU 进程的操作。

复用两条长期 owner thread 及各自 Context；B 已预热，实验期间保持 idle。
A owner 在开始标记被观测后约自然时间的 10% 发出 invalidate 并执行 control，本轮不测 B 接管延迟。
原 `compute_kernel`、`tiny_kernel`、`probe.hpp` 和原 cubin 均未改变。
新增独立 `epoch_sentinel.cubin`，只写校验值、GPU timestamp，最后以 system-scope release store 发布 token；无 cancellation polling。

每个条件执行：

1. 用原 kernel 校准约 1 s；每对 conditions 使用相同 iterations，另测同工作量 reference。
2. 原 stream launch 长 kernel，紧接着 enqueue old sentinel 和它的 end Event，不 synchronize。
3. 确认 A 已开始；约 10% 处确认 A/old Event pending、old/new token 均为空，再 FIFO disable。
4. hold 1200 ms，大于当次整个自然 reference；约每 100 ms 查询 A/old Event 与 token，窗口末再查一次。
5. FIFO enable；以 3 s deadline 查询 old 状态，若 A/old Event 和 token 均完成则提前结束该观察。
6. 在原 Context 中**新建 stream**，提交 new sentinel；以 3 s deadline 查询其 Event，并校验唯一 token/值。
7. 再检查 old 状态：若仍未完成，再观察最多 3 s，防止把 new submission 后重现的 old 工作误判为移除。
8. 查询当前 Context、原 compute allocation 的地址范围、old/new mapped buffer 地址，记录 identity 与结果。

所有测量槽预先分配、互不重叠，整个进程不回收/重置；matched 的 10 个 new stream 也都不同。
A/B 在 5 对间不重建。case 内没有 Event/Stream/Context synchronize；初始化和校准也改用有 deadline 的 Event query。
单个 Driver/RM API 若异常阻塞，由外层进程 timeout 截止；调用 begin/end 日志可指出未返回的位置。

### 顺序与实际结果

| 阶段 | 证明了什么 | 没证明什么 | 随后最小实验 |
|---|---|---|---|
| 契约、ABI、payload gate | 现有入口能只切换 rewind 位，保持原身份/状态检查 | GPU 行为 | false smoke |
| false smoke ×1 | hold 时 pending；enable 后 old 正确执行；新 stream 可用 | rewind 效果 | true smoke |
| true smoke ×1 | rewind ioctl 成功；old 仍执行；无 CUDA 错误且 new 正确 | GPFIFO 消费位置 | 5 对 matched trials |
| 5 对 matched | 两条件均稳定得到 `old_queue_preserved` | 其他提交布局/驱动的 rewind 效果 | 本轮结束；下一步仅定位 FIFO 消费边界 |

逐样本去标识数据：[relative_results.csv](results/rewind_public/relative_results.csv)；
核验摘要：[validation_summary.json](results/rewind_public/validation_summary.json)。

| 指标 | rewind=false，matched n=5 | rewind=true，matched n=5 |
|---|---:|---:|
| `old_queue_preserved` | 5/5 | 5/5 |
| hold 查询中 A/old 均 pending、old token 为空 | 5/5 | 5/5 |
| enable 后 old token 在 new launch 前出现 | 5/5 | 5/5 |
| new Event 完成且 token/校验值正确 | 5/5 | 5/5 |
| Context / buffers 相同 | 5/5 | 5/5 |
| CUDA async error（排除 NOT_READY） | 0 | 0 |
| invalidate → disable return，中位数 | 0.888 ms | 0.935 ms |
| disable ioctl，中位数 | 0.488 ms | 0.575 ms |
| 观测 hold 时长，中位数 | 1201.142 ms | 1201.088 ms |
| enable → old token 被 host 观测，中位数 | 898.051 ms | 897.598 ms |
| A Event elapsed，中位数 | 2199.324 ms | 2199.550 ms |

matched reference 约 996.8–1000.3 ms。enable 后约 0.9 s 才观测到 old token，与 baseline 的旧队列继续执行相符。
token 时间是 host 观测时间，含约 10 ms 查询粒度；A Event elapsed 包含暂停时间，不是 SM active time。
本轮不对微小 latency 差异作性能结论。

**12/12 的 old token 都在 new stream 提交前出现**，因此本次 old 结果不需要借助 new submission 才重新出现。
但查询等 Driver 行为是否影响软件 PUT/提交状态，本轮没有直接测量。
CPU/CUDA 证据足以区分 old/new 执行结果，所以未采集 Nsight；普通 kernel trace 也不能单独给出 GP_GET 消费边界。

### 分类与停止规则

只输出以下四类；尚未观察到的分支不宣称已经验证：

- `old_queue_preserved`：hold 前后条件成立；old/new token 和对应 Events 均正确完成，Context/buffer identity 未变。
- `queued_sentinel_removed_context_reusable`：old A Event 已完成，但 old sentinel token 在 enable 后、new 完成后的两个有界窗口仍为空，old sentinel Event pending；new 正确完成且 identity/hold 条件成立。这是有界观测分类，不是永久丢弃证明。
- `context_or_submission_poisoned`：测量中的 CUDA API 返回非 NOT_READY 错误，或 new Event 已完成但输出错误。记录首个出错 API；不通过重建 Context 证明恢复。
- `ambiguous`：其他情况，包括 RM 拒绝/错误、前置条件或 identity 不成立、超时、无法解释的 Event/token 组合。

错误或歧义立即停止剩余 cases。保存结果后 `_Exit`，不做 CUDA synchronize、Context destroy/recreate 或猜测式 RM 恢复；
正常结束也采用相同退出路径。进程退出后的驱动资源回收不作为 Context 可复用的证据。
本轮没有触发停止条件，所有 12 次 disable/enable 都成功，所有实验进程均退出。

## 5. Proven / not proven

**Proven**：在该提交布局上，仅设置 rewind 位并未让 old sentinel 失效。
现有 hold/enable 和新 stream 提交能力保留，CUDA Context、原 compute buffer 和 token buffers 在验证时有效。
所以目前没有证据把这个 stock rewind control 直接包装成 CUDA runtime 的 epoch-invalidation abstraction。

**Not proven**：RM rewind 是 no-op；尚未消费的 GPFIFO entry 无法移除；firmware 的具体行为；
已执行 kernel 被取消；任何 CUDA submission/Events 都能安全 discard/rearm；或者必须新增 driver primitive。
本次的 old sentinel 未执行前置条件，比“对应 GPFIFO 尚未消费”弱，不能混用。

## 6. Next minimal experiment

只定位一个边界：**在 disable 前后只读观测 GP_GET/GP_PUT，并把 old sentinel 的实际 GPFIFO entry 与消费位置关联**。
先确认现有受限、只读接口是否能提供该证据；若不能，就把它记录为观测缺口。
这能区分“sentinel 命令已经被消费/预取”与“PUT 回退后又被软件提交状态重新发布”。
本轮未增加该实验，也未引入 latest-request-wins、额外取消机制或新 driver 接口。

## 编译、运行与结果位置

复用 [PREEMPT_HOLD.md](PREEMPT_HOLD.md) 中匹配的 bridge 构建。
bridge 工作树包含此前未提交 additions，HEAD 单独不能描述它；本地 `results/rewind/build_inputs.json` 记录实际输入。
必须有现有 `ap_bridge_discard_queued` 符号；ABI/版本不匹配即停止，不自行放松 gate。

```bash
cmake -S . -B build-rewind -DCMAKE_BUILD_TYPE=Release -DBENCH_INTERCEPTION_SOURCE=/path/to/Interception/active-preempt -DBENCH_RM_LIBRARY="$PWD/build-rm/librm_control.so"
cmake --build build-rewind -j2
```

将 `GPU_UUID` 设为所选完整 A100 UUID；CPU 编号按本机调整。先逐条运行并检查分类，不能用无条件循环越过 smoke gate：

```bash
env -i PATH=/usr/local/cuda/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES="$GPU_UUID" LD_PRELOAD="$PWD/build-rm/librm_control.so" timeout --kill-after=5s 45s ./build-rewind/context_ping_pong --mode preempt-hold --rm-case rewind-false --long-ms 1000 --hold-ms 1200 --query-timeout-ms 3000 --trials 1 --cpu-a 0 --cpu-b 1 --cpu-control 2 --output results/rewind-new/smoke_false.csv
# 上一条有效且安全后：相同命令改为 --rm-case rewind-true，output 改为 smoke_true.csv。
# 两个 smoke 都可解释且安全后：改为 --rm-case rewind-paired --trials 5，timeout 150s，output 改为 matched.csv。
python3 scripts/analyze_rewind.py results/rewind-new/smoke_false.csv results/rewind-new/smoke_true.csv results/rewind-new/matched.csv > results/rewind-new/relative.csv
```

原始结果留在本地 `results/rewind/`：主 CSV、`.epochs.csv`、`.epoch_calls.csv`、metadata、RM binary/JSONL、身份与诊断。
`.epochs.csv` 包含控制时间边界、CUDA 状态、token、Context/stream/buffer identity；`.epoch_calls.csv` 每次 API 都先写 begin，再写返回。
主 CSV 的 B 请求字段在本轮为空；沿用 `.controls.csv` 是空 header，rewind 数据以 `.epochs.csv` 为准。
两个初始 smoke 的 legacy `thread_a_launch_count` 未含额外 sentinel；matched 前已补齐计数，分类和分析不依赖该字段。

公开数据仅包含逻辑 case/trial、相对时间和校验状态，不上传 UUID、PID/TID、主机路径、原始绝对时间或 RM handles/journals。
构建、原无 RM 构建、2 项现有 bridge contract 测试通过；分析器核对同进程 journal、完整 payload、token、配对工作量与状态证据。
