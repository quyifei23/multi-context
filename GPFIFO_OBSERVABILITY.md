# GPFIFO observability：595.58.03 source/interface archaeology

日期：2026-09-22。**Phase A 完成；Phase B 未执行。**

后续：[最小 USERD 映射实验](USERD_MAPPING_EXPERIMENT.md)已将本次 A compute TSG 的 8 个 channel
关联到既有 CPU 映射，并被动核对了物理 UUID。下文保留接口考古时的证据状态；
新结果只解决 USERD binding，尚未读取 GET/PUT 或建立 sentinel→entry 映射。

**当前可验证、可供本 benchmark 使用的接口不足以判断某个 CUDA epoch 是否仍在可 rewind 的未消费区间。**
公开源码存在 USERD GET/PUT 布局、按 owner memory object 建立 CPU 映射的路径，以及返回 context buffer 的 RM 查询候选；
但尚未把它们连成“本次 A channel → 有效 progress 读数 → old sentinel entry”的证据链。
因此本轮结论是 **observability/mapping gap，适用性仍有 `unknown`**，不是“已证明 stock RM 没有相应接口”，
更不是“已证明缺少 cancellation primitive”或“必须 patch driver”。

只检查固定源码、现有 bridge 和已保存日志；新增 GPU workload、RM control、driver/bridge/kernel 修改均为 **0**。

## 1. Research question

执行 `FIFO_DISABLE_CHANNELS(bRewindGpPut=true)` 时，承载 old sentinel 的提交是否仍在目标
CUDA compute channel 的 `GP_GET..GP_PUT` 未消费区间？

这里需要辨认 **FIFO front end 的消费边界**，而不是 kernel 是否完成、CUDA Event 是否 ready 或 token 是否出现。
长 kernel 正在执行、后继 kernel 尚未执行，都不能单独证明后继命令尚未被 front end 取得。

先校正样本数：[已有结果](results/rewind_public/relative_results.csv)共 12 个样本，
是 **rewind=false 6/6 + rewind=true 6/6**，并非 true 12/12；两条件均为 `old_queue_preserved`。
该事实及 [REWIND_EXPERIMENT.md](REWIND_EXPERIMENT.md) 的限制保持不变，本轮没有新增样本。

## 2. System abstraction

所需证据链为：

`application epoch/token → CUDA logical submission → publication batch → owned RM channel/generation → GPFIFO entry interval → progress snapshot`

| 层次 | 当前知道什么 | 尚不能替代什么 |
|---|---|---|
| Application | old/new token 独立，结果可区分 | token 不是 FIFO sequence number |
| CUDA Driver | stream 顺序、launch 返回、Event 完成 | launch 返回不公布具体 channel 或 entry index |
| Bridge/RM | A compute TSG 的完整 8 个成员及 owner、generation、UUID 校验 | group identity 不选择承载 sentinel 的那个 channel |
| FIFO submission | 存在 GPFIFO ring、pushbuffer、USERD publication | 一个 kernel 不由公开契约保证恰好对应一个 ring entry |
| Hardware/RM context state | rewind 契约涉及 RAMFC GP_PUT/GP_GET | 软件游标、USERD 字段和 RAMFC 保存状态不能不加区分地互相代替 |

需要的是可验证的区间及其生命周期；`PUT > GET` 只能作为无 wrap 时的简写。
实际判断必须包含 ring entry count、环绕顺序以及同一 channel generation，不能直接比较两个裸整数。

## 3. Existing mechanism

### 固定证据范围

官方 NVIDIA `open-gpu-kernel-modules` **595.58.03**，commit
`db0c4e65c8e34c678d745ddb1317f53f90d1072b`。
本地 checkout 的 HEAD/tag 相符，`git status --short` 为空；以下官方链接均固定到该 commit。
网页抓取未成功，结论来自本地匹配 checkout 的实际文件内容，不依赖搜索摘要。
公开 KMD 源码不是 libcuda submission 实现，也不包含所有 GSP handler 的实现体。

既有实验环境为 A100-PCIE-40GB、KMD/libcuda 595.58.03、CUDA Driver API 13.2、Toolkit 12.8.61；
本轮没有重新查询设备或运行实验。设备唯一标识、host 地址和 RM handles 不纳入公开报告。

### USERD、RAMFC 与 GPFIFO

1. **USERD 的字段定义存在。** Ampere `Nvc56fControl` 定义 `GPGet` 在 `0x88`、`GPPut` 在 `0x8c`；
   `GPGet` 标为 read-only。GA100 `dev_ram.h` 的 `NV_RAMUSERD_GP_GET/GP_PUT` 与这两个偏移一致。
   这证明布局存在，不能独立证明本次 channel 的映射地址、GET 更新时点或瞬时一致性。[S1][S2]
2. **客户端分配 USERD 的合法 ownership 链存在。** `NV_CHANNEL_ALLOC_PARAMS` 包含
   `hUserdMemory[] / userdOffset[]`、`gpFifoOffset / gpFifoEntries`、VASpace 和 engine；
   `hObjectBuffer` 已标为不再使用，不能拿它推导当前 ring 映射。
   `kchannelCreateUserdMemDescBc_GV100` 根据这些 handle/offset 设置 client-allocated USERD；
   内部从同一 `hClient` 查找 `Memory`，建立子 memdesc 并处理生命周期。[S3][S4]
3. **stock CPU mapping 路径存在，但映射不是 GET 查询。** `NVOS33` 指定 client、device、memory、offset、length，
   Linux wrapper 另有 mmap fd；escape 实现建立相应 mmap context。合法观察应使用应用自身已经成功建立的映射。
   `kchannelMap_TU102` 明确拒绝 client-allocated USERD：这种情况应追踪它的 memory object 映射，
   不能直接把 channel handle 当 CPU 地址，也不能据此新映射任意 RAMFC/物理内存。[S5][S6]
4. **GPFIFO entry 有公开格式。** Ampere entry 为 8 bytes，包含 pushbuffer 地址、长度、level/sync 等；
   格式中没有 application token/epoch 标签。公开格式有助于辨认命令范围，不自动建立 CUDA kernel→entry 关系。[S1]
5. **rewind 的目标明确是 RAMFC。** `FIFO_DISABLE_CHANNELS` 的契约是 RAMFC `GP_PUT ← GP_GET`。
   USERD GPPut 是提交侧字段；Volta+ 提交路径先更新 GP_PUT，再写 channel-specific doorbell token。
   公开契约没有保证 rewind 同时把 USERD GPPut 写成相同值，也没有规定这次操作的原子快照接口。[S7][S8]

因此，即使以后读到了 USERD，**USERD PUT 不变不能证明 RAMFC rewind 无效**；
USERD PUT 后续前进也不能单独证明旧 entry 被重新发布。
H2 所要求的“rewind 后 PUT==GET”必须明确是哪一层状态，并证实其与 RAMFC rewind 的关系。

| 状态域 | 定义 / ownership | 本轮能确定的边界 |
|---|---|---|
| USERD GPGet/GPPut | channel 的 USERD memory；可由客户端提供，RM 建立 channel→子 memdesc 关系；GET 对提交者只读，PUT 用于 publication | `0x88/0x8c` 是字段偏移，不是目标进程中的地址；更新/读回时点未验证 |
| RAMFC GP_GET/GP_PUT | RM 管理的 channel FIFO context state；`kfifoChannelGetFifoContextMemDesc(...FIFO_CTX_RAMFC...)` 获取其 memdesc；rewind 由 RM control 修改 | 物理 placement 可查询不等于 contents 可读；没有验证 A100 的 live/saved snapshot 解码 |
| GPFIFO base/count | channel allocation 的 `gpFifoOffset/gpFifoEntries`；ring 与 pushbuffer 属于提交路径，GPU 地址不是 CPU mapping | 参数定义已知，当前 channel 的实际值没有被 bridge 保留 |

RAMFC ownership 来自 GET_CHANNEL_MEM_INFO 的实际 memdesc 查询实现，而非把 UVM-owned control page 推广到 CUDA。[S14]

## 4. Observability：候选逐项分类

分类描述接口及其适用范围，不把“头文件有定义”当成“本机已实测可用”。

| 分类 | 本轮含义与结果 |
|---|---|
| `stock_readonly_interface_available` | 找到现有查询契约；已确认返回内容。下列可确认项只提供 placement/identity 等元数据，未构成目标 progress 接口 |
| `already_mapped_state_available` | 必须能绑定当前对象及有效 CPU 映射；**本次 CUDA channel 没有候选达到此状态** |
| `uvm_only_not_applicable` | UVM 自有 channel 的软件状态/内部指针不能移植为 libcuda compute channel 的读数 |
| `requires_new_driver_interface` | 把仅内核可用的 retained metadata 新导出给 user runtime 属于此类；本轮不实现，不据此断言所有其他路径都需要新接口 |
| `unknown` | stock 候选存在，但当前对象、返回语义、适用前提或映射未验证；不足以放行 Phase B |

### A. UVM 自有 channel：`uvm_only_not_applicable`

**Object → ownership → interface → returned state → target binding**

- `uvm_channel_t` → UVM channel manager 自建队列 → debug procfs `uvm_channel_print_info`
  → `cpu_put / gpu_get`、tracking semaphore、ring count/location → **不能绑定为当前 libcuda compute channel**。[S9]
- UVM 新分配的 channel → RM/UVM 内核客户端 → `nvUvmInterfaceChannelAllocate(..., UvmGpuChannelInfo*)`
  → `gpGet / gpPut` 指针、CPU ring 指针、entry count、GPU VA、hardware IDs
  → **是新建 UVM channel 的返回值，不是按任意已捕获 CUDA handle 查询的 API**。
  实现明确设置 `UVM_OWNED`，从自己的 `controlPage` 返回这些指针。[S10][S11]

关键修正：procfs 的 `gpu_get` **不是直接读取硬件 GP_GET**。
`uvm_channel_update_progress_with_max` 读取 tracking semaphore 完成值，逐 entry 检查完成情况并推进软件回收游标；
打印代码输出的就是该游标。`cpu_put` 同样属于 UVM 提交软件状态。
因此即使换成 UVM 工作负载，procfs 的 get/put 也不能不经语义核验就回答 HOST 是否已消费某 entry。[S9]

### B. UVM retain 当前用户 channel：元数据可关联，progress 不导出

**Object → ownership → interface → returned state → target binding**

用户 CUDA channel → 用户拥有、UVM/RM 持有引用 →
`nvUvmInterfaceRetainChannel(vaSpace,hClient,hChannel,...)` →
`UvmGpuChannelInstanceInfo` 的 instance base、runlistId/chId/tsgId、context resources、fault/doorbell 信息 →
**在内核验证后可以关联用户 channel，但返回结构没有 GP_GET/GP_PUT、GPFIFO base/count**。[S10][S12]

不要与 `UvmGpuChannelInfo` 混淆：二者是不同结构和入口。
user-facing `UVM_REGISTER_CHANNEL` 以 client/channel/VA range 为输入，仅返回 `rmStatus`，不是 progress query。
UVM 还明确警告注册后不能重新信任数值 handles：用户可能 free/reallocate；这支持继续保留 generation gates。[S13]

分类：将该 kernel-only retained state 当作现有 user-runtime progress API **不成立**；
若通过新增 UVM/debug ioctl 导出，属于 `requires_new_driver_interface`，本轮到此排除。
这不否定下面已有 USERD 映射候选。

### C. 当前 CUDA channel 的既有 USERD/ring 映射：`unknown`

**Object → ownership → interface → returned state → target binding**

C56F channel 及其 USERD `Memory` → libcuda/RM client；RM 持有 memdesc →
已发生的 channel allocation + memory allocation/map/unmap 元数据 →
原则上可关联 USERD handle/offset、有效 CPU mapping，另有 ring GPU base/count →
**source 层面的关联规则存在，本次 capture 没有保留足够数据，当前不能绑定**。[S3–S6]

已检查现有 `Interception/active-preempt`：

- `src/rm_observation.cpp:34–64` 对 memory allocation 只保留对象关系，明确不保留/解引用 pointer 与 limit；
  channel allocation 的 opaque payload 不解码，只有 TSG allocation 读取 engine。
- `src/rm_observation.h:5–14` 没有 USERD/ring/mapping 字段。
  `src/object_registry.h` 的 GroupBinding 保留全部成员，刻意没有 selected channel；Node 不保存 memory mapping。
- `src/ap_bridge.h` 的 diagnostics 只序列化已捕获状态，不增加 RM 查询；现有 API 没有 progress reader。
  journal 的 mmap 是日志文件映射，不是 NVIDIA USERD mapping。
- 已保存的 matched 末尾 diagnostics 含 **1,309** 条观察记录、**40** 条 C56F allocation envelope，
  都没有保留 USERD/ring 参数；其中 **70** 条 `NV_ESC_RM_MAP_MEMORY`、**14** 条 unmap 仅为 `metadata_only`。
  这是整个进程的 aggregate，不能把这些 allocation/map 数量当作 A 的 channel 数。
  A 的 identity 快照确实含 8 个成员，但没有 stream/sentinel→member 关系。

旧日志不能恢复已丢弃的 payload，更不能恢复已退出进程的有效映射。
因此不标为 `already_mapped_state_available`，也不标为 `requires_new_driver_interface`：
**被动保留应用自身已有 allocation/mapping 可能足够，当前没有证明。**

### D. 公开 RM placement/identity 查询

| Object / ownership | API/interface | Returned state | 能否绑定当前 CUDA channel | 分类 |
|---|---|---|---|---|
| 所属 GPU/subdevice，RM | `FIFO_GET_USERD_LOCATION` (`0x2080110d`) | sysmem/vidmem aperture、cache attribute；无地址/指针值 | 不接受 hChannel，不能定位它的 USERD | `stock_readonly_interface_available`，仅 placement [S7] |
| 同 client/device 下的 channel，RM | `FIFO_GET_CHANNEL_MEM_INFO` (`0x2080110c`) | instance/RAMFC/method-buffer 的物理 base、size、aperture；没有 contents 或 ring index | 实现按 hChannel 在该 client/device 查找，能关联对象 | stock 查询定义存在，但 flags `0x10004` 含 `PRIVILEGED`；当前 bridge 权限边界内的使用为 `unknown`，且无所需 progress [S14] |
| channel，RM | `GPFIFO_GET_WORK_SUBMIT_TOKEN` (`0xc36f0108`) | opaque 32-bit doorbell token | channel object 可作为请求对象；token 不等于 ring sequence/index | `stock_readonly_interface_available`，仅提交 identity [S15] |
| channel，RM/GSP | `GET_CHANNEL_HW_STATE` (`0xb06f010f`) | NEXT、CTX_RELOAD、FAULTED 等 bit | 请求对象是 channel；没有 GET/PUT | `unknown`：未验证本机适用性；返回结构本身已排除它作为 progress reader [S16] |
| debug DiagApi + channel，RM/GSP | `FIFO_GET_CHANNEL_STATE` (`0x208f0403`) | bound/enabled/scheduled/cpuMap/runlistSet 等 boolean | 结构可指定 client/channel；不是指针查询 | `unknown`：未新建 diag object、未调用；即使成功仍无 FIFO index [S17] |

`GET_CHANNEL_MEM_INFO` 中的 **物理地址不是可解引用 CPU 地址，也不是映射授权**。
本轮没有调用该 control 或基于返回地址建立映射。`FIFO_UPDATE_CHANNEL_INFO` 则是修改 channel 的输入接口，
不因包含 `gpFifoEntries/userdOffset` 就把它当 readonly 查询。[S7]

### E. RAMFC/context buffer 的 stock GSP 查询候选：`unknown`

**Object → ownership → interface → returned state → target binding**

KernelChannel → RM/GSP → `GET_ENGINE_CTX_SIZE / GET_ENGINE_CTX_DATA / SAVE_ENGINE_CTX_DATA`
(`0xb06f010b / 010c / 0111`) → engine context 的 size 或原始 bytes；SAVE 有 4,096-byte buffer，
头文件明确提及 RAMFC/instance buffer 与 vGPU migration → **可用 channel handle 作请求对象，
但当前 A100 bare-metal CUDA channel 的适用性、engine selection、live/RAMFC snapshot 语义未确定**。[S16]

生成 dispatch 表包含 GET/DATA/SAVE，`0x10048` 含 NON_PRIVILEGED 与 ROUTE_TO_PHYSICAL，
并有 vGPU GSP flag；**不能把它简单说成“只有 root 能调用”或“源码证明仅 vGPU 可调用”**。
相应实现体未在这份公开 CPU-KMD 源码中找到，不能仅凭返回 buffer 的名字保证它提供当前硬件 GET/PUT、
支持 running channel，或保证所需快照语义。[S18]

这是保留的 stock 候选，不是已证伪的接口。本轮没有试探调用、没有 decode 任意 RAMFC bytes，
也没有把它加入 bridge 的许可 control 集合；它尚不足以通过 Phase B gate。

## 5. Experiment：本轮执行的是接口审计

| 项目 | 定义 / 实际结果 |
|---|---|
| Hypothesis | 现有、合法、受 owner 约束的 readonly 状态足以关联本次 CUDA channel 及 sentinel entry |
| Baseline | 已有 rewind 实验的 token/Event/identity/RM 日志 |
| Variable | 只增加源码和已有记录的可观测性审计，不改变应用或 control |
| Metric | 能否闭合 channel binding、progress 语义、sentinel mapping 三项证据 |
| Expected observation | 找到适用的现成 reader，并能给出可验证 sentinel entry interval，才进入 GPU 采样 |
| Observed | 公开布局/查询候选存在；当前 reader 与 mapping 两项均未建立；Phase B gate 不满足 |

核对包括固定 tag/commit/干净工作树，control header/dispatch flags/实现，USERD memory ownership，
UVM allocate/retain/procfs 的结构与调用链，bridge decoder/schema，以及已有日志字段和样本计数。
没有执行新的 CUDA API、Nsight、GET/PUT 采样、disable/enable 或其他 RM control。

### Sentinel → entry mapping gate

目前 **不能可靠建立**。CUDA enqueue 返回、old token、Event ready 都不给出 ring index。
8-byte entry 指向 pushbuffer 命令范围；CUDA 可能如何分批/共享命令范围，不能从公开 entry layout 推断。
对单个 API 前后读到 PUT delta，也必须确认期间没有其他提交、publication 确实归属于该 API、
channel 已选对且 ring 未发生无法辨认的环绕，才可归因；并非看到 delta 就完成 mapping。

所以第二层 gap 是 **“即使未来能读 GET/PUT，当前仍没有 CUDA logical submission→GPFIFO entry 的验证依据”**。
不是已经实测“能读 GET/PUT，但 mapping 失败”：前一层本轮也没有执行。

### Phase B 的九个点：全部未采集

| 点 | 所需状态 | 本轮 |
|---|---|---|
| 1 | enqueue old sentinel 前 | 未采集 |
| 2 | enqueue 返回后 | 未采集 |
| 3 | A start marker 后 | 未采集 |
| 4 | invalidate 前 | 未采集 |
| 5 | FIFO disable 前 | 未采集 |
| 6 | FIFO disable 返回后 | 未采集 |
| 7 | enable 前 | 未采集 |
| 8 | enable 返回后 | 未采集 |
| 9 | old token 出现时 | 未采集 |

即使以后补齐九点，host 调用边界也只是 RM 操作瞬间的时间包络；
disable 前到真正 preempt/rewind 之间 GET 仍可能前进。没有界定该竞态，就不能声称测到“恰在 rewind 的瞬间”。

| 假设 | 本轮判定 | 缺少的决定性证据 |
|---|---|---|
| H1：sentinel 已被消费/预取 | 未证明、未排除 | sentinel entry interval + 有明确更新语义的 GET 跨越证据 |
| H2：rewind 移除后软件重新发布 | 未证明、未排除 | 同一 channel 的 RAMFC rewind 证据，以及相同旧命令再次 publication 的关联；USERD PUT 单独变化不够 |
| H3：未消费但行为不属上述两类 | 未证明、未排除 | 首先证明 sentinel 仍未消费，再记录同一状态域的指针行为 |

## 6. Proven / not proven

| 步骤 | 证明了什么 | 没证明什么 | 下一步最小验证 |
|---|---|---|---|
| 区分 UVM 两类对象 | 自建 channel 的 gpGet/gpPut 与 retained 用户 channel 是不同接口；procfs get 是软件完成游标 | 当前 CUDA channel 的硬件消费进度 | 追踪用户 USERD ownership/mapping |
| 追踪公开 USERD/RM 定义 | USERD 布局和 owner memory mapping 路径确实存在；rewind 的目标是 RAMFC | 当前 reader、USERD/RAMFC 同步语义；migration 查询在本机的适用性 | 验证现有映射能否被无歧义关联 |
| 检查 bridge/旧日志 | 现有 capture 丢失 USERD/ring/mmap payload，TSG identity 不选择 sentinel channel | “已有映射一定不存在”或“必需新 driver ioctl” | 被动 mapping audit，见下一节 |
| Phase B gate | 当前证据不能回答 sentinel 是否位于可 rewind 区间，应停止 active 实验 | H1/H2/H3 中任何一个成立；rewind 是 no-op；cancellation primitive 缺失 | 先解决一个 channel 的 readonly binding，不扩大机制 |

## 7. Next minimum experiment

**只验证“本进程一个已捕获 C56F channel 能否关联到 libcuda 已有的有效 USERD CPU 映射”。**
最小候选是被动记录正常 allocation 和已有 map/unmap 的已知 ABI 字段，按 owner/generation/UUID 对齐，
证明 memory handle、userdOffset、映射范围和生命周期一致；不新发 RM control、不创建额外映射、不扫描猜测地址。
这只是下一步候选，本轮没有修改 bridge 或启动该实验。

若关联不成立，仍停在第一层 gap；若成立，先验证 GET/PUT 读数语义，再做 sentinel entry mapping，
mapping 成立之前不恢复 rewind 的九点实验。两项都未验证之前，不能把 schema 级候选升级为
`already_mapped_state_available`，也不能把未知项升级为“stock stack 不支持”。

**回答研究问题：当前 stack 向本 runtime 暴露并经验证的 abstraction 尚不足以作出这个判断。
已定位的缺口是 application submission 与 FIFO progress 的关联及状态语义；
还不能判定它最终只需被动用户态观测，还是需要新的受限 driver 可观测接口。**

## 固定源码索引

以下行号对应本轮实际读取的 595.58.03 checkout，链接供独立复核。

- [S1] [Ampere channel USERD 与 entry layout：clc56f.h，48–64、265–284](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/common/sdk/nvidia/inc/class/clc56f.h#L48)
- [S2] [GA100 USERD fields：dev_ram.h，25–39](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/common/inc/swref/published/ampere/ga100/dev_ram.h#L25)
- [S3] [Channel allocation 参数：alloc_channel.h，295–352](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/common/sdk/nvidia/inc/alloc/alloc_channel.h#L295)
- [S4] [Client USERD ownership：kernel_channel_gv100.c，68–133、180–254](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/nvidia/src/kernel/gpu/fifo/arch/volta/kernel_channel_gv100.c#L68)
- [S5] [CPU map 参数：nvos.h，1847–1858](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/common/sdk/nvidia/inc/nvos.h#L1847)；[Linux wrapper/escape，578–620](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/nvidia/arch/nvalloc/unix/src/escape.c#L578)
- [S6] [Channel CPU map 的限制：kernel_channel_tu102.c，174–240](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/nvidia/src/kernel/gpu/fifo/arch/turing/kernel_channel_tu102.c#L174)
- [S7] [FIFO controls：ctrl2080fifo.h，217–388、711–745](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080fifo.h#L217)
- [S8] [Volta+ publication 与 doorbell：nv_gpu_ops.c，5600–5605](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/nvidia/src/kernel/rmapi/nv_gpu_ops.c#L5600)
- [S9] [UVM 软件完成游标：uvm_channel.c，185–240](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/kernel-open/nvidia-uvm/uvm_channel.c#L185)；[procfs 输出，4135–4156](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/kernel-open/nvidia-uvm/uvm_channel.c#L4135)
- [S10] [UVM ChannelAllocate / RetainChannel：nv_uvm_interface.h，542–575、1298–1323](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/kernel-open/common/inc/nv_uvm_interface.h#L542)
- [S11] [UVM-owned channel 分配与指针返回：nv_gpu_ops.c，5954–6144](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/nvidia/src/kernel/rmapi/nv_gpu_ops.c#L5954)
- [S12] [InstanceInfo / ChannelInfo 不同结构：nv_uvm_types.h，178–294](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/kernel-open/common/inc/nv_uvm_types.h#L178)；[Retain 实现，nv_gpu_ops.c，10236–10441](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/nvidia/src/kernel/rmapi/nv_gpu_ops.c#L10236)
- [S13] [UVM_REGISTER_CHANNEL ABI：uvm_ioctl.h，329–340](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/kernel-open/nvidia-uvm/uvm_ioctl.h#L329)；[handle lifetime 警告：uvm_user_channel.h，69–80](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/kernel-open/nvidia-uvm/uvm_user_channel.h#L69)
- [S14] [GET_CHANNEL_MEM_INFO 实现，kernel_fifo_ctrl.c，430–505](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/nvidia/src/kernel/gpu/fifo/kernel_fifo_ctrl.c#L430)；[dispatch flags，g_subdevice_nvoc.c，4974–5002](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/nvidia/generated/g_subdevice_nvoc.c#L4974)
- [S15] [Opaque work-submit token：ctrlc36f.h，60–85](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/common/sdk/nvidia/inc/ctrl/ctrlc36f.h#L60)
- [S16] [Engine context / HW state 查询契约：ctrlb06f.h，60–343](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/common/sdk/nvidia/inc/ctrl/ctrlb06f.h#L60)
- [S17] [Diag FIFO state 的返回字段：ctrl208ffifo.h，79–125](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/common/sdk/nvidia/inc/ctrl/ctrl208f/ctrl208ffifo.h#L79)
- [S18] [Channel query dispatch，g_kernel_channel_nvoc.c，411–518](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/nvidia/generated/g_kernel_channel_nvoc.c#L411)；[权限/route flag 定义，control.h，170–237、272–289](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/nvidia/inc/kernel/rmapi/control.h#L170)
