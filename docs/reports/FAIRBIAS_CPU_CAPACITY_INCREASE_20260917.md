# CPU 并发扩容 — 2026-09-17 10:18 Asia/Shanghai

用户要求进一步利用新机剩余 CPU 和内存，尽快训练并分析结果。**已部署并启动扩容交接；快照时仍在交接，不能声称 30 个任务已经同时运行。**

| 项目 | 初始 CPU 调度 | 扩容调度 |
|---|---|---|
| worker 上限 | 24，初始实际准入 22 | 30，实际数量仍受内存准入约束 |
| 总 RSS 限额 | 52 GiB | 54 GiB |
| 准入每任务内存下限 | 2.25 GiB | 1.5 GiB |
| 观测峰值乘数 | 1.20 | 1.10 |
| 总 RSS 准入额外预留 | 2 GiB | 1 GiB |
| 主机内存预留 | 6 GiB | 6 GiB |
| 单任务科学预算 | 1,800 秒、4 GiB | 不变 |
| 方法、seed、参数、数据分区、运行时身份 | 修复版 FRAPPÉ | 不变 |

新机有效配额是 32 核、60 GiB。并发上限留 2 个核给调度和其他开销。54 GiB 总限额与 6 GiB 主机余量相容，准入仍检查实时 cgroup 内存；更高峰值会阻止后续新任务入场。不会为了提高内存百分比人为分配无用内存。

在调整前独立核验了 **22 个正式 VALID 回执**的文件哈希与运行时身份，最大观测任务峰值 **1.5822258 GiB**，没有查看公平性/预测性能排名来决定调整。对这一级峰值，30 个任务按 1.1 倍保守估计为约 52.21 GiB，低于 53 GiB 准入阈值。后续候选可能有不同峰值，因此 30 是上限，并非无条件承诺；现有 per-worker 和总内存保护继续生效。

新封装为 `scripts/run_nhis_cpu_capacity.py`，原运行时 worker、adapter、调度核心等 70 个注册文件未改。新操作策略为远端 `artifacts/nhis/cpu_sharding_20260917/runtime_cpu_capacity_policy.json`。源机的既定分工不受此次操作策略更新影响。

旧调度器 PID `11283` 不支持实时修改并发。于 **10:17:02** 启动新接管器 PID **209845**，将旧父进程停在安全派发边界，保留在途 worker 和它们已消耗的时间预算；核对内核退出状态、封存回执后，才结束旧父进程并取得运行锁，用更高并发继续原来的 300 个任务。不会重启已运行的训练或重复派发已完成任务。10:17:31 的快照为 **DRAINING，已封存 2 个、剩余 20 个**；过渡阶段并发会暂时下降，后续提升自动完成。

原运行清单绑定旧策略的 SHA。为复用经过验证的交接器，在新 run 中新增 `parallel_policy.json`，内容是原 `runtime_cpu_policy.json` 的逐字节副本，先验证与初始 manifest 一致。原 manifest 和旧策略没有被覆盖；新策略与新封装 SHA 由新的 launch、交接和 capacity activation 记录承接。

验证：新增 `tests/benchmark/test_cpu_capacity.py`，最终 Linux **4 passed in 3.69s**；覆盖第 30 个任务准入、总内存/主机内存压力拒绝、CPU 父进程真实交接、非零退出码保留以及继续执行尚未启动任务。主管审查时移除了测试草稿中可能沿源码符号链接写入的危险夹具代码，移除后才在 Linux 运行；本地原测试因平台条件跳过该部分，真实源码未被该草稿覆盖。首轮 Linux 测试又暴露合成环境缺少专用解释器路径，补齐临时解释器链接后通过。每轮 Linux 测试后核对 70 个注册源文件，均保持一致。未覆盖或丢弃失败测试日志。

证据见 `docs/reports/fairbias_cpu_replica_20260917/evidence/`：`cpu_capacity_launch.json`、`runtime_cpu_capacity_policy.json`、`capacity_linux_test_final.txt`、`capacity_precheck.json`、`capacity_transition_snapshot.json` 和 `capacity_release.json`。运行日志在新机 `artifacts/nhis/cpu_sharding_20260917/cpu_capacity.log`；真实交接状态在修复版 run 的 `operational_handoffs/*_cpu_capacity/`。

后续分析沿用部署报告的版本边界：完整 480 个修复版 FRAPPÉ 注册任务才形成该方法的主比较集合；旧版 FRAPPÉ 不混入。已有历史方法及其未解决失败也需要明确计入覆盖率和局限，不能以本次部署完成代替完整论文验收。一次性增加并发不会改变先验研究设计，但会影响墙钟时间，应在计算成本表中记录资源政策阶段。

已有 `fairbias` 定时任务更新为双节点验收与分析，采用低频唤醒；运行正常且无可做事项时立即结束，不在任务里待机。后续需检查空闲节点与未启动任务的分配机会；只有经过分工排他性和在途任务保护核验后才迁移未启动任务。分析完成后暂停定时任务。
