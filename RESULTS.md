# A100 实测结果（2026-09-22）

**本实验的串行 `destroy → B launch` 路径没有显示提前接管优势。** 约 1 s 的固定工作，
在 host 观测 A 启动后 100 ms 发出 invalidate，直接销毁后 B 最快在 **967.895 ms** 完成；
显式等待 A 完成再销毁的对照为 **967.799 ms**。这些是各 10 个样本中的观测最小值，不是硬件下界或 CUDA 保证。

## 环境与复现

- GPU：NVIDIA A100-PCIE-40GB，108 SM，PCI `0000:18:00.0`，visible ordinal 0。
- UUID：`99e4e85f-1945-866c-9e00-130b51df7908`。
- NVIDIA driver package / kernel module：595.58.03；Driver API version：13020（13.2）；编译 Toolkit：12.8.61；cubin：sm_80。
- MIG disabled，compute mode default；开始前 GPU 无工作进程，未发现 `nvidia-cuda-mps-server`。
- A/B/controller 固定到 CPU 0/1/2，三个独立 core，均在 GPU 对应 NUMA node 0；没有锁定 GPU clocks。
- 测量必须在有设备访问权限的环境执行。这里沙盒内的 NO_DEVICE 是隔离造成的，沙盒外正常。

执行过的命令：

```bash
./build/context_ping_pong --mode all --long-ms 10 --trials 2 --estimate-start --cpu-a 0 --cpu-b 1 --cpu-control 2 --output results/a100_smoke.csv
./build/context_ping_pong --mode all --long-ms 10,100,1000 --trials 10 --estimate-start --cpu-a 0 --cpu-b 1 --cpu-control 2 --output results/a100_main.csv
./build/context_ping_pong --mode takeover --action both --long-ms 10,100,1000 --trials 10 --estimate-start --cpu-a 0 --cpu-b 1 --cpu-control 2 --output results/a100_paired.csv
python3 scripts/summarize.py results/a100_main.csv results/a100_paired.csv > results/a100_summary.csv
```

再次运行请改用新的 output 路径。原始 CSV 附有对应 `.metadata.csv` 与 `.nvidia-smi.txt`。

## 核心结果：相同工作量的接管对照

以下均来自 [a100_paired.csv](results/a100_paired.csv)。A 为 432 blocks × 256 threads；
每个 target 内两个条件使用同一个 iterations，每个条件 10 个有效样本。
对每对样本交替执行顺序；两条路径间仍可能有 host/GPU 状态变化，不能把很小的差值当成收益。

| 目标 / 校准参考 (ms) | 每 thread iterations | invalidate 延迟 (ms) | 直接销毁 takeover 最小 / 中位数 (ms) | 等待后销毁 takeover 最小 / 中位数 (ms) |
|---|---:|---:|---:|---:|
| 10 / 9.973 | 1,857 | 1 | 80.302 / 81.882 | 80.017 / 83.043 |
| 100 / 99.800 | 33,921 | 10 | 158.453 / 159.094 | 158.479 / 158.794 |
| 1000 / 997.828 | 342,818 | 100 | 967.895 / 969.836 | 967.799 / 970.478 |

`takeover = t_B_complete - t_invalidate`。invalidate 延迟从 host 观测启动标记开始；
校准参考不是每个被销毁 kernel 的实际完成时间，后者不能在 Context 销毁后读取 Event 来补测。

1 s 组的中位数分解：

| 项目 | 直接销毁 | 等待完成再销毁 |
|---|---:|---:|
| invalidate 到进入 destroy | 0.0016 ms | 897.849 ms |
| `cuCtxDestroy` 调用时间 | 969.716 ms | 72.504 ms |
| destroy 返回到 B launch | 3.948 μs | 4.677 μs |
| B launch 到 host 观测完成 | 110.666 μs | 110.308 μs |

各项中位数来自独立分布，不能要求精确相加等于总延迟中位数。
这与“直接 destroy 的可见耗时包含剩余工作等待及 Context 清理”相符；对照中的约 72.5 ms 清理成本仍然很大。
实验没有观察 Driver 内部实现，不能由此证明具体的同步点、抢占粒度或硬件终止语义。

## 待命成本

来自 [a100_main.csv](results/a100_main.csv)，每个目标 10 对 AB/BA 样本。

| 目标 (ms) | A-only Event 中位数 (ms) | A + idle B Event 中位数 (ms) | 成对时间开销中位数 |
|---|---:|---:|---:|
| 10 | 9.972736 | 9.972736 | 0.0000% |
| 100 | 66.765823 | 66.765823 | −0.0015% |
| 1000 | 997.827576 | 997.827576 | 0.0000% |

在这些样本里没有出现稳定的 idle B 开销信号；**不能据此声称成本严格为零**。
100 ms 目标在校准时为 97.612 ms，但后续相同工作量约为 66.766 ms，说明校准后的性能状态发生变化。
10/100 ms 组也有离群的成对差值。未锁频且没有运行中 clock trace，因此不能仅凭这些差值归因于 B 或某种 DVFS 行为。
1 s 组的实际时长较稳定。单独运行之间的固定工作量可能不同，不能直接把不同数据集的同名 target 当作相同工作量比较。

## B 开始时间的估计

在成对实验中，10/100 ms 组共 40 个样本通过前后 CPU–GPU clock offset 区间的一致性检查；
1 s 组 20 个样本全部未通过，`start_est_valid=0`，估计列为 `nan`。
这说明当前恒定 offset 模型对该时间跨度不够可靠，不能用失效估计宣布 B 的真实启动时间。
原始 GPU `%globaltimer` 采样与 CPU launch/completion 时间均保留；通过检查的估计也仍受 README 中的时钟假设限制。

## 结论边界与下一步

已证明：两个 owner thread 各自使用独立 Context、预热 B 并在销毁 A 后成功完成新 kernel 的程序路径可以运行；
当前 GPU/driver/资源配置的串行销毁路径耗时很高，对 1 s 工作没有观察到提前接管优势。

没有证明：Context 销毁是快速 cancellation primitive，Driver 必然采用某种硬件机制，或此结论能推广到其他资源规模、驱动与调度环境。

下一步最小实验：在同一套线程、资源、计时条件下，只把 B 提交通知提前到 invalidate 时，让 B 与 destroy 并行。
它可以检验“必须等 destroy 返回”这一 host 顺序是否限制 B 的响应；这次没有运行该变体。
