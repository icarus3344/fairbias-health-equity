# AutoDL 当前批次安全优化记录

状态：**APPLIED_LIMITED_STARTUP_OPTIMIZATION；正式实验继续运行；GPU 原型未获准替换正式实现。**

用户授权优化，同时要求避免影响当前实验。核验时间为 2026-09-16 17:05:35 CST。

## 已实施的优化与保护边界

新增 `scripts/prepare_verified_bytecode.py`，仅对 registration 中源码哈希一致的 Python 文件生成 CHECKED_HASH 字节码缓存。验证缓存 magic、标志位及源码哈希，并在操作前后核对登记文件。新进程可以读取这些缓存；缓存不会改变 Python 源码或拟合逻辑。

正式批次 `/root/autodl-tmp/fairbias/artifacts/nhis/benchmark_autodl_20260916_075038Z` 的登记文件 SHA-256 仍为 `340cd60f6ab0d5919e2ac7bc22b62bdb3cc758faaa986650abdb3978b964732a`，70 个登记源码/配置文件全部一致。调度 PID 48932 保持运行，实时检查有 15 个非僵尸直接子进程。没有重启调度器、改动模型与候选配置、变更数据划分/种子/预算、覆盖既有结果或更换正式环境依赖。

优化前后各进行了三次独立进程的纯导入测量，未读取数据或训练模型：中位数从 **4.4635 秒降至 4.3219 秒**，相差 **0.1416 秒**。样本小，且前后顺序固定、后台负载持续变化，不能区分缓存预热、负载波动与字节码编译的贡献，也不能据此宣称整批实验显著提速。

## 独立 GPU 可行性诊断

新增 `scripts/probe_nhis_gpu_isolated.py`，在独立目录使用纯合成 4096×24 数据、固定种子、TabM k=4/d=64/2 blocks、100 epochs、batch=1024。关闭 TF32 和混合精度。CPU 使用现有 TabM 适配器，GPU 使用单独的对应训练循环。一次执行被限制为单 CPU 亲和性、单线程、低优先级和 180 秒进程上限，现已结束。短时诊断仍会消耗少量共享 CPU 资源，不能保证对同时在跑作业的墙时完全没有瞬时影响。

| 项目 | 实测 |
|---|---:|
| CPU 适配器 fit（含初始化） | 2.68897 秒 |
| GPU fit（含初始化、数据传输、预热和训练） | 1.99412 秒 |
| GPU 训练循环（不含初始化/预热） | 1.88894 秒 |
| CPU/GPU 事件概率最大绝对差 | 0.00472337 |
| CPU/GPU 事件概率平均绝对差 | 0.000116790 |
| 0.5 阈值分类不一致数 | 0/4096 |
| 各自在训练设备上新进程重载 | 均通过，atol=1e-7，rtol=0 |
| GPU 状态在 CPU 推断的最大概率差 | 1.19209e-7，仅作可移植性诊断 |

这是当前负载下的一对顺序测量，GPU fit 约少用 26% 时间；不能外推到 NHIS 全数据或整个 benchmark。GPU 原型没有建立方法等价性；相同阈值标签也不意味着校准、排序、其他阈值或公平指标一致。结果明确标记 `benchmark_equivalence=NOT_ESTABLISHED`，不会进入研究结果。

隔离与无 CUDA 契约：本地 **3 passed in 2.32s**；Linux **3 passed in 3.37s**。本地需按仓库惯例设置 `PYTHONPATH=src`；未设置时的首次调用出现导入错误，修正调用后通过。GPU 新进程重载另在服务器实测通过。

## 本轮决定与后续顺序

1. 保持当前 15 并发和正式 CPU 实现，采用上述已核验缓存；本轮不再增加并发或进行调度交接。
2. 优先对 EG+GBDT 重复拟合、BM+AE 搜索和重复表示变换进行独立剖析，再决定能否复用中间计算。现有耗时证据不足以认定重复变换是主要瓶颈。
3. 后续若采用 GPU，使用独立登记批次，验证概率契约、固定种子行为、重载、预测/公平性指标及真实吞吐；保存 CPU 基线。不能只因本次合成训练更快而在当前批次混用设备实现。
4. FRAPPE 独立环境依赖仍在下载，本轮尚未完成其运行准入，也没有启动 FRAPPE 正式任务。不能将环境准备记为方法完成。

## 证据

本地聚合证据：`artifacts/nhis/optimization_evidence_20260916/`，包含 `bytecode_report.json`、GPU `probe_manifest.json`、`result.json`、`reload_check.json`、`post_optimization_integrity.json` 及副本哈希清单。GPU 原型只写合成数据产物，没有读取 NHIS 微数据、选择集或测试集。

服务器原始目录为 `artifacts/nhis/optimization_bytecode_20260916T0856Z` 与 `artifacts/nhis/optimization_gpu_probe_20260916T0920Z`；后者目录名是预先选择的标签，不是测量时间，时间核验以 `post_optimization_integrity.json` 为准。未 stage/commit。
