# 本地验证记录（2026-09-22）

Stock RM preempt-and-hold 的独立验证见 [PREEMPT_HOLD.md](PREEMPT_HOLD.md)。
该实验不修改 A kernel 或 `Interception` 的既有桥接代码；原结果和失败记录保留。

这份记录区分编译/host 检查与真实 GPU 实验。沙盒内设备不可见；经用户指出隔离原因后，已在沙盒外完成 A100 实测。
结果见 [RESULTS.md](RESULTS.md)，当前数据不支持“销毁 busy Context 带来提前接管”的假设。
新增 [LIVE_HANDOFF.md](LIVE_HANDOFF.md) 单独验证保留 A 的路径，已观察到 B 在 A kernel 结束前执行；
两者的 host 顺序不同。

| 项目 | 本地观测 |
|---|---|
| CMake / host compiler | 3.22.1 / GNU C++ 11.4.0 |
| CUDA Toolkit / nvcc | 12.8.61 / V12.8.61 |
| CUDA header version | 12080 |
| `cuDriverGetVersion` | 返回 CUDA_SUCCESS，13020（支持 CUDA Driver API 13.2） |
| `/proc/driver/nvidia/version` | NVIDIA UNIX x86_64 Kernel Module 595.58.03 |
| GPU 型号 | 沙盒外：NVIDIA A100-PCIE-40GB，108 SM，compute capability 8.0 |
| GPU 选择 | 可见 GPU 0，PCI `0000:18:00.0`；只在这一张 GPU 上运行 |
| GPU 模式 | MIG Disabled，compute mode Default；开始前无运行中的 GPU 进程，未发现 MPS server |
| CPU affinity | A/B/controller 分别为 CPU 0/1/2，独立 physical core，GPU 所在 NUMA node 0 |
| 沙盒内访问 | 缺少设备节点；`cuInit(0)` 返回 CUDA_ERROR_NO_DEVICE (100)，`nvidia-smi` 无法通信 |
| 沙盒外访问 | `nvidia-smi` 与 Driver API 正常；这纠正了把沙盒不可见误当主机不可用的判断 |

已执行：

1. `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` 和 `cmake --build build -j2` 成功，无 host compiler warnings。
2. `ptxas`：`compute_kernel` 使用 19 registers，`tiny_kernel` 使用 10 registers；两者 0-byte stack、无 spill load/store。
3. `cuobjdump --dump-sass build/kernels.cubin`：计算循环内保留 256 条 FFMA，初始化另有 1 条 FFMA；无 LDL/STL。启动通知生成 system fence 与 system-scope store。此检查验证静态产物，不证明 GPU 利用率或通知延迟。
4. `ldd build/context_ping_pong`：直接 CUDA 依赖为 `libcuda.so.1`，无 `libcudart`。
5. `--help` 成功；12 组非法配置检查成功，包括 trials/warmup 为 0、非有限/空/零时长、非法 thread 数、无效 invalidate 延迟、重复 CPU 绑定、非法 action/option 和缺失参数。
6. 无 GPU 的 smoke 尝试非零退出，CSV 只有 header，没有伪造样本；metadata 标记 failed 并保存实际 CUDA 错误。重复 output 路径被拒绝，原文件保持不变。
7. 汇总脚本语法检查与合成 CSV 检查成功：无效 trial 不进入最小值；全无效组输出 NA；成对 overhead 和 nearest-rank p95 计算正确。合成数据不是 CUDA 实验结果。

真实 GPU 验证：

| 数据集 | 实测内容 | 结果 |
|---|---|---|
| `results/a100_smoke.csv` | 10 ms，2 次，待命两组及直接销毁，启用 clock 估计 | 完成；2 个有效接管样本 |
| `results/a100_main.csv` | 10/100/1000 ms，每组 10 次，待命两组及直接销毁 | 完成；60 个待命样本、30 个有效接管样本 |
| `results/a100_paired.csv` | 相同迭代数的 destroy/wait-then-destroy，三个时长各 10 对，AB/BA 交替 | 完成；60 个有效接管样本 |

三次运行的最终 metadata status 均为 completed。对全部 92 个接管样本检查了 CSV 列数、host 时间顺序、CPU 绑定、
启动后的 pending Event 状态、成功 destroy、派生延迟公式；对 30 对控制实验核对相同迭代数和交替 order，均通过。
所有 B 输出校验及源码中的 host 顺序检查在运行中通过。

clock 估计通过一致性检查的样本共 62 个；1 s 组的 30 个接管样本全部未通过恒定 offset 模型检查，估计保留为 nan，
没有伪造有效启动时间。失败 clock 估计不影响 CPU 接管完成时间测量。

未验证：Driver 内部是否/如何终止或抢占 kernel，其他 GPU/driver 环境的行为，以及去除测量 instrumentation 后的极限延迟。

## 新增 live-handoff 的验证

新增模式完成 Release 构建。`--help` 显示新模式；非法 mode、fraction、thread 配置仍被拒绝。
A 的 device source、probe header 及原有 cubin 内容比较相同，没有更改 A kernel 执行逻辑。
最终源码 review 范围为 host 协调、CSV/metadata、汇总脚本及可选 Nsight 分析脚本；没有新增构建依赖。

| 最终数据集 | 实际运行内容 | 验证结果 |
|---|---|---|
| `results/live_handoff_calibrated.csv` | 10/100/1000 ms，每条件 10 次，每组三条件校准并固定 iterations | 90 个测量行有效；30 个 live 样本的 A Context 存活、B 完成后 A Event pending |
| `results/live_handoff_calibrated_profiled.csv` | 同一最终实现，每条件 2 次；独立 Nsight CUDA trace | 18 个测量行有效；6 个 live 样本均为 A GPU start < B GPU start < B GPU end < A GPU end |
| `results/live_handoff_legacy_regression.csv` | `--mode all --action both --long-ms 10 --trials 1` | 原有两种 standby、两种 destroy、两个 idle 参考均完成；1 个校准行 |

对普通运行和 profiled 运行的全部 108 个测量行检查了：60 列 CSV、最终 metadata completed、
CPU affinity、host 时间顺序、延迟差值公式、A 在 invalidate 后 pending、
live/idle 无 destroy、serial 成功 destroy 后才提交 B、B 完成后 A query 的时间边界。
对 36 组三条件检查了相同 iterations、完整条件集合、轮换 order；invalidate 的配置延迟为每次 reference 的 10%。

独立 Nsight trace 的 A/B launch 总数为 152/279，和 metadata 完全匹配。
逐个使用 thread launch 序号与 API correlation ID 匹配被测 kernel，核对同一 GPU、不同 Context；
6 个 live 样本的 B 全部早于 A 结束，6 个 serial 样本的 B 全部晚于 A 结束。
错误地将普通运行 CSV 输入该 trace 的分析脚本时，程序非零退出并报告 process/thread 不匹配。

普通运行 10/100 ms live 组共 20 个 GPU-start 时钟估计通过一致性检查；1 s live 组的 10 个估计均无效、保留 nan。
独立 trace 的 18 个 B API bracket 时钟对齐区间均通过脚本一致性检查；它们仍受局部时钟同速假设限制，
不用于替换普通运行延迟。GPU A/B 顺序证据本身不需要这个对齐。

Python 脚本语法检查通过；汇总脚本处理旧 48 列和新 60 列 CSV 均成功。
旧模式回归的 7 行全部有效，destroy/wait-then-destroy 的成功状态与时间顺序检查通过。
最初未逐组校准的数据 `live_handoff.csv` 和 `live_handoff_profiled.csv` 仅保留为本地探索记录，
最终报告的普通运行统计只使用带 `_calibrated` 的数据集。
