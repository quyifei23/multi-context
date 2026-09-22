# 最小实验：CUDA channel → 已有 USERD CPU mapping

2026-09-22。**结果：`already_mapped_state_available`，范围仅为 USERD binding。**

后续的 [USERD_PROGRESS_EXPERIMENT.md](USERD_PROGRESS_EXPERIMENT.md)已完成一次有界字段读取；下文保留 mapping 阶段的证据边界。

两次独立进程 smoke 均把 A compute TSG 的 **8/8 个 channel** 无歧义关联到 libcuda
已经建立的可读 CPU 映射；每次在 A/B 预热后、A 再完成一次 tiny kernel 后核对，身份和映射不变。
观测器没有额外发出 RM control 或 mapping 请求；USERD 内容访问与 rewind trial 均为 0。

这解决了上一轮的第一层 binding gap：**当前配置下无需新 driver 接口，即可找到受 owner 约束的 USERD 映射。**
尚未证明 USERD GET 的更新语义、它与 RAMFC 的关系，或某个 CUDA sentinel 对应哪个 GPFIFO entry。

## 1. Research question

本进程捕获的一个 C56F channel，能否通过明确的 memory ownership 关系，
对应到 libcuda 已经成功建立、当前仍有效且可读的 USERD CPU 映射？

这一步先解决“读哪里”的必要条件，尚不回答 old epoch 是否仍可 rewind。

## 2. System abstraction

本轮验证的关联为：

`A allocation scope → current compute TSG/member generation → hUserdMemory[0] + userdOffset[0]`

`→ same-client live Memory object → successful NVOS33 map setup → same-fd mmap return → live readable VMA`

application epoch、CUDA stream、某次 kernel launch 均未被赋予 ring index。
没有按数值顺序选一个“看起来像 A”的 channel：检查完整的已捕获 group，逐项输出全部 8 个成员。
公开 CSV 中的 ordinal 只是展示编号。

## 3. Existing mechanism / ownership contract

沿用 [GPFIFO_OBSERVABILITY.md](GPFIFO_OBSERVABILITY.md) 核对过的 595.58.03，
固定官方 commit `db0c4e65c8e34c678d745ddb1317f53f90d1072b`。

- Channel allocation 公开参数提供 `hUserdMemory[] / userdOffset[] / gpFifoOffset / gpFifoEntries`。
  本次为已知 C56F plain allocation，legacy `paramsSize=0`；按该固定 class ABI 解释。
  serialized/未知 channel ABI 不解码，并将 capture 标记失败。
- `NV_ESC_RM_MAP_MEMORY` 的已知 Linux wrapper 包含 `NVOS33` 和 mmap fd。
  `rm_create_mmap_context` 检查 client 的进程 ownership；Linux mmap 要求该 fd 已有 RM 验证的 context，且 offset 为 0。
  本轮只记录 libcuda 已经成功调用的这条路径。[固定实现](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/src/nvidia/arch/nvalloc/unix/src/osapi.c#L2485)、[mmap 验证](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/db0c4e65c8e34c678d745ddb1317f53f90d1072b/kernel-open/nvidia/nv-mmap.c#L547)。
- UUID 不只依赖应用 scope 字符串。libcuda 的正常初始化已经对同一 current subdevice
  发出 `GPU_GET_GID_INFO`；被动保留其成功返回的 binary SHA-1 UUID，与 selected CUDA UUID/scope 比较。
  **不为此再发查询。**

`Interception/active-preempt` 的 source、library、owner/generation/profile gates 均未修改。
使用其已有的 observation-only `inspect_owned_tsg_for_gpu`；bridge active budget 为 0。
GET_INFO credential 与 FIFO UUID-query gate 保持未建立，未把被动证据伪装成可执行 RM control 的授权。

## 4. Observability implementation

最小改动只在本仓库：

- [userd_observer.cpp](src/userd_observer.cpp)：额外 preload 记录器。
  ioctl 原样交给原 bridge 再到 libc；mmap/mmap64、munmap、close、mremap 原样转发应用请求。
  不构造额外 RM request，不创建额外 GPU mapping，不修改任何 protection 或 GPU memory。
- [userd_mapping.hpp](src/userd_mapping.hpp)：新增 `--mode userd-map`，复用已有 Worker、GpuState、原 tiny kernel。
  两个长期 host owner 各创建一个 Context，只在 A 创建/预热时打开原 allocation scope。
- [analyze_userd_mapping.py](scripts/analyze_userd_mapping.py)：离线关联与去标识输出。
  分析器不访问 GPU，也不解引用记录中的地址。

每个最终 binding 必须同时满足：

1. 固定 source/runtime profile、owner PID、registry instance、current group/member/parent generation 一致；原 bridge 无 incomplete 状态。
2. 同一 client 下的 USERD Memory 当前仍存活，parent device/generation 正确；没有 channel 分配后复用该 memory handle。
3. 只有 subdevice slot 0 非零；不从多个 slots 中猜选；USERD 的完整 512 bytes 位于 map 范围内。
4. RM map 成功，紧接的对应 fd mmap 成功；fd 的 device/inode/rdev/path、thread、length 匹配，offset=0，CPU protection 可读。
   fd 在二者之间 close/reuse、mmap context 被替换等情况均拒绝。
5. 后续没有使该映射失效的 RM unmap、CPU unmap、MAP_FIXED 覆盖或 mremap。
   当前 `/proc/self/maps` 的共享可读 VMA、inode/path/offset 与所推导 CPU 范围一致；多个可行 alias 不猜选。
6. 同一 subdevice generation 的原有成功 UUID 响应与目标物理 UUID 一致。
7. 记录器与原 bridge 观察到的原始 ioctl envelope 顺序逐条一致；新增 project RM control 计数为 0。

`NVOS33.pLinearAddress` **没有被直接当作可解引用地址**；CPU 范围来自实际 mmap 返回值及映射内 offset。
USERD 内容本轮未读取。没有把 RAMFC 物理 base、GPU VA 或旧进程地址拿来建立映射。

## 5. Experiment

| 项目 | 定义 |
|---|---|
| Hypothesis | 正常 CUDA 提交所需的既有 mapping 元数据，足以构造 owner-bound USERD CPU binding |
| Baseline | 上轮 capture 仅保留 RM 对象拓扑，缺失 channel allocation 和 map/mmap 关联字段 |
| Variable | 增加被动记录及离线 join；kernel 和原控制 gates 不变 |
| Metric | 完整 binding 数、跨快照 identity/mapping 一致性、UUID/生命周期验证、新增 control 数 |
| Expected observation | 至少一个目标 channel 的完整链条唯一且当前有效；缺任一必要证据则停止于 mapping gap |

环境：NVIDIA A100-PCIE-40GB，cc 8.0；KMD/libcuda **595.58.03**，Driver API **13020 / 13.2**，
Toolkit **12.8.61**，Linux x86_64。选定 GPU 开始时无 compute 进程，MIG Disabled、compute mode Default。
A/B/controller 使用 CPU 0/1/2。只在所选 GPU 创建工作；没有操作其他 GPU 的进程。

每次固定工作序列：A tiny warmup ×1，B tiny warmup ×1，snapshot 1；
A 在原 stream/Context 再执行同一个 tiny ×1，snapshot 2；B 保持 standby。
原 tiny 的输出/timestamps 校验和 CUDA Event 完成检查通过。
Event query 每次最多 3 s，外层进程截止 45 s；没有无限 synchronize。
Context 在最后一个快照时仍存活；完成后 `_Exit`，由进程退出释放资源，显式 cuCtxDestroy/reset 调用均为 0。

本次没有提交 1 s compute 或 old/new sentinel：这一步只验证映射，不改变既有 rewind 实验的 application semantics。
`kernels.cu`、`probe.hpp` 未改；新构建的原 cubin 与 `build-rewind` 中 cubin 的 SHA-256 相同。

先完成 `smoke_01`。补齐记录器对 mremap 失效事件的观测、通过生命周期反例测试后，
用最终记录器完成 `smoke_02`。两次都没有发生 mremap，不扩大到 active control 或 matched rewind trials。

| 观测 | smoke_01 | smoke_02 |
|---|---:|---:|
| 进程 exit code | 0 | 0 |
| 验证的 A group channels | 8/8 | 8/8 |
| 检查点 | 2 | 2 |
| Context / group generations / USERD mapping 跨检查点一致 | 是 | 是 |
| 原 bridge 与新记录器一致的 ioctl envelopes | 1,307 | 1,307 |
| 被动捕获的同一 subdevice UUID 成功响应 | 2 | 2 |
| 新增 project RM control | 0 | 0 |
| USERD GET/PUT 内容读取 | 0 | 0 |
| CUDA API/结果验证错误 | 0 | 0 |

两快照复用同一批 UUID 响应，不能算作更多独立 identity 样本。
8 个 channel 也共享 allocation，不把它们当作 8 个独立实验重复。

两个进程中，A 的 8 个 USERD 均位于各自已有的 **2,097,152-byte memory-object CPU mapping** 内，
相对 offsets 为 `8192, 20480, 32768, 45056, 57344, 69632, 81920, 94208`；每个 allocation 报告 ring count **1,024**。
这些是这两次运行的观测值，不是 libcuda 的通用布局保证。
ring base 已被动保存于本地，但尚未将其映射到 CPU 或解码 entry 内容。

### 证据与验证

- 原始记录留在本地 `results/userd_map/smoke_01.csv.userd/`、`smoke_02.csv.userd/`：
  `observations.jsonl`、两个 `.identity.json/.memory.json/.diagnostics.json/.maps`。
  邻接的 metadata、launcher/exit/stdout/stderr 文件保留版本、命令、deadline 和 binary hash。
- 去标识结果：
  [smoke_01 summary](results/userd_map_public/smoke_01/summary.json)、[bindings](results/userd_map_public/smoke_01/bindings.csv)；
  [smoke_02 summary](results/userd_map_public/smoke_02/summary.json)、[bindings](results/userd_map_public/smoke_02/bindings.csv)。
  公开输出使用固定字段白名单，去掉 PID、UUID、fd、handles、CPU/GPU 地址和主机路径。
- 可选 observer 构建和原无 RM 依赖构建均通过；Python 编译检查通过。
  16 个 CPU 单元测试覆盖 free/reuse、错误 owner/parent generation、fd close/reuse、失效/重叠映射、边界和未知编码等反例。
  另对实际 capture 的临时副本注入错误 UUID、错误 PID、缺失记录和非零 project control，四项均被拒绝；原始记录未改。

### 复现

准备匹配的未修改 NVIDIA checkout，以及前阶段已构建的 Interception `librm_control.so`。
`BENCH_INTERCEPTION_SOURCE` 与该 library 必须来自同一 bridge 源码状态；外部工作树本来含未提交文件，
不能仅用其 Git HEAD 代替实际构建输入。参考 [PREEMPT_HOLD.md](PREEMPT_HOLD.md) 的依赖说明。

```bash
# 三个变量分别指向固定 NVIDIA checkout、现有 bridge source、匹配的 library。
cmake -S . -B build-userd -DCMAKE_BUILD_TYPE=Release \
  -DBENCH_NVIDIA_SOURCE="$NVIDIA_595_SOURCE" \
  -DBENCH_INTERCEPTION_SOURCE="$INTERCEPTION_SOURCE" \
  -DBENCH_RM_LIBRARY="$RM_LIBRARY"
cmake --build build-userd -j 4
python3 -m unittest discover -s tests -p 'test_userd_mapping.py' -v

# 需要真实设备节点；在可访问 GPU 的宿主环境运行。已有输出不会被覆盖。
python3 scripts/run_userd_mapping.py --gpu 0 --build build-userd \
  --output results/userd_map/run_01.csv
python3 scripts/analyze_userd_mapping.py results/userd_map/run_01.csv.userd \
  --public-dir results/userd_map_public/run_01
```

launcher 核对驱动、MIG/compute mode 和选定 GPU 空闲状态，设置单一完整 UUID 及
`LD_PRELOAD=.../libuserd_observer.so`；记录器要求下一跳确实为原 `librm_control.so`。
仅 Linux x86_64 / 595.58.03 ABI 被本实现支持。原有 benchmark 默认构建无需新增依赖。

## 6. Proven / not proven

| 步骤 | 证明了什么 | 没证明什么 | 下一步最小实验 |
|---|---|---|---|
| 固定 ABI 与原样转发 | 现成 mapping 参数可被动记录；原 bridge envelope 顺序完整匹配 | 捕获所有潜在 hidden/direct syscalls | 只对已闭合证据链作结论 |
| 两次 smoke + 两检查点 | 当前 CUDA A group 的 USERD binding 可通过 stock mapping、owner/generation/物理 UUID 证据建立 | USERD GET/PUT 内容、新鲜度、硬件消费语义 | 在同类 live binding 上有界只读采样 |
| CPU 反例测试 | 已知对象/映射失效、错误 owner/UUID 和记录缺失会阻止正结论 | 一切未建模调用路径的正确性 | 延续明确的覆盖边界 |
| 范围结论 | 第一层 mapping gap 对这两个实例已补齐，无需新 driver interface | sentinel→entry mapping；H1/H2/H3；RAMFC rewind 是否生效 | 先验证 USERD 读数，再决定后续 mapping 实验 |

记录器与原 bridge 都基于被拦截的 libc 调用；direct/hidden syscalls 不在覆盖保证内。
当前验证是在应用完成 tiny、两 Context idle 的边界，不能外推为运行中 channel 的原子快照能力。

## 7. Next minimum experiment

**在一次新的运行中重新建立 live binding，在 Context 存活且 identity 未变时，仅有界读取 GET/PUT，
比较 idle → 一次原 tiny submission → completion 的字段变化。**
不能复用本轮已退出进程的 CPU 地址；本轮的映射有效性结论对应各自的存活快照，不是持久 capability。
先验证读数是否可用及何时推进，不执行 FIFO disable/enable，不读取任意 RAMFC，不宣称定位了 sentinel entry。
当字段语义验证完成，再单独建立 logical submission→entry 的可验证关系；之前不进入 rewind 的九点实验。

现有 stack 已提供这一步所需的合法 memory mapping mechanism。
**当前仍缺的是经验证的 FIFO progress 语义与 application submission 映射，
尚不能让 runtime 判断某个 application epoch 是否仍可 rewind。**
