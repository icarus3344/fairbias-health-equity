# AutoDL 并发提升与交接记录

状态：**RUNNING_15_WORKERS — 已完成交接，15 并发正式运行。**

后续核验：2026-09-16 16:35:23 CST 已生成 `drain_complete.json` 和 `resume_started.json`，新调度会话为 `9637354745d34fbe86ad974dec2c53b1`。12 项在途任务全部自然结束并封存，其中 7 项 VALID、5 项因 AE commit limit 退出；后者属于研究计算预算不足，不能称为有效结果。原 1800 秒墙时上限没有触发这些退出。已独立重新验证这 12 份回执，旧调度器退出，新实时状态确认 15 并发。

用户观察 CPU 约 76%、内存约 8%，要求提高资源利用率。核验实际配额仍为 16 CPU equivalents、80 GiB；原正式队列为 12 个单线程 worker。12/16 与观察相符。平台内存比例与 cgroup `memory.current` 的文件缓存口径不同；低内存占用不能推出 CPU 计算会因分配更多内存而加速。

## 新设置

- 目标并发 15；保持每 worker 单线程、4 GiB RSS、1800 秒上限。
- 总 RSS 上限 64 GiB，15 个任务的各自上限相加为 60 GiB，仍为系统留出内存。
- 新策略文件：服务器运行目录中的 `parallel_policy_15_workers.json`；原策略不变。
- 新策略 SHA-256：`8138f330d9ea588f9322730df977f4e97842a04ac1af4ce50762bf2fdddef393`。
- 理论并行容量增加 25%；尚未测得实际整体吞吐提升，不能将其称为 25% 实测加速。

## 交接方式与证据

原调度器没有动态并发或排空入口。新增 `scripts/handoff_nhis_parallel.py` 作为独立操作助手，不修改登记的科学源码、候选配置、数据或单任务预算。

助手在轮询边界暂停旧调度器派发，持续监控已经运行的 worker，读取 Linux 内核保存的实际退出状态并复用原回执校验。全部在途 worker 封存后，只终止已停止的旧调度器，再由原调度类以 15 并发恢复。已完成作业只核验，不重复运行或覆盖。交接期间 CPU 占用会暂时下降。

- 服务器批次：`/root/autodl-tmp/fairbias/artifacts/nhis/benchmark_autodl_20260916_075038Z`。
- 原 dispatcher PID：7174；交接助手 PID：48932。
- 启动证据：`handoff_15_launch.json`；日志：`handoff_15_workers.log`。
- 本次审计目录：`operational_handoffs/20260916T082148_756822Z/`。
- `captured.json` 固定旧 PID/starttime、原策略、新策略及源码身份；`kernel_exit_*.json` 保存原始 wait status 与解码后的返回码。
- `drain_complete.json` 出现表示旧在途作业已封存；`resume_started.json` 与随后实时状态用于确认 15 并发开始。`queue_exit.json` 才是新队列结束证据。
- 原 `full_cpu_queue_exit.json` 将记录旧 dispatcher 的有意终止，不能单独据此认定整个研究队列失败。
- `status.json=NEEDS_RECOVERY` 或日志异常时必须人工核查，不能盲目恢复旧 dispatcher 或启动第二个队列。

16:24 CST 附近的已核验快照：交接捕获 12 项，7 项已自然完成并封存，剩余 5 项为 FairBias BM+AE。此时 15 并发尚未实际开始；最终状态必须读取上述服务器文件，不应将本文件作为实时监控。

## 验证及限制

- Linux 纯合成交接测试：**2 passed in 17.69s**，覆盖 3→4 并发恢复、无重复、全部回执、正常退出与真实非零退出码保留。
- 部署前逐项核验 registration 中的科学源码哈希均未变化。
- 原始 2024 数据仍未迁入；没有启动选择或测试集评估。本地真实训练继续暂停，未 stage/commit。
- 另发现 `EG_EO + GBDT` 同一候选的 5 个种子在 C 分区触发决策概率越界契约。独立单线程 F/C 诊断已经完成（seed 0，207.62 秒）：q 最大值及混合权重之和均为 `1.0000000000000002`，10 项超过 1，最大幅度 `2.220446049250313e-16`；权重非负且基础分类器输出均为 `{0,1}`。这一复现确认机器精度舍入引起严格边界拒绝；其余种子没有追加拟合。原失败回执保留，不能当作 FairBias 优势；容差修复与重跑仍待独立登记，不进入本次资源调整。聚合诊断保存在服务器 `artifacts/nhis/diagnostics_eg_bounds_162731/eg_gbdt_bounds_aggregate.json`，副本已拉回本地迁移证据目录；未保存行级预测。
- GPU 尚未用于正式训练，当前增加的是 CPU 并发。

这是运行资源调整记录，不构成方法学验收或论文结果。
