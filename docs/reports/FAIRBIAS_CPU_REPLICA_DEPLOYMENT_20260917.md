# CPU 第二节点部署与 FRAPPÉ 运行时修复验收

2026-09-17，Asia/Shanghai。结论：**第二节点已部署并于 10:01 启动真实 NHIS 开发任务；运行时修复通过限定范围验收。** 这不是完整 benchmark 完成或论文结果验收。以下资源和进程数来自约 10:03 的一次启动快照。

## 1. 两台机器实际在做什么

| 节点 | 实际可用配额 | 已启动/后续任务 | 并发与资源 |
|---|---|---|---|
| `fairbias-cpu` | 32 核、60 GiB 内存 | 修复运行时的 300 个 FRAPPÉ 注册 seed 任务 | 上限 24；实际准入 22 个；采样约用 23.2 核、33.4 GiB 内存 |
| `fairbias-autodl` | 16 核、80 GiB 内存 | 当前旧 FRAPPÉ 批次自然结束 → 原注册 LFR 剩余队列 → 修复运行时的另外 180 个 FRAPPÉ 任务 | 15 个单线程 worker；采样约用 16 核、36.9 GiB 内存 |

新机内存配额为 **60 GiB，不是页面宣传的 80 GB**，按容器 cgroup 配额设置保护。CPU 节点任务 RSS 总限额 52 GiB；准入按每任务至少 2.25 GiB、观测峰值的 1.2 倍估算，另外保留总 RSS 2 GiB 和主机 6 GiB 余量。因此初始实际并发是 22，上限 24 不等于必须启动 24 个任务。正式首批峰值尚未测完，暂不扩大并发。

CPU 调度器 PID `11283`，源机阶段控制器 PID `174423`。新机快照中的 22 个 worker 均已实际消耗约 92 CPU 秒，且全部携带正确的修复运行时身份和环境变量；不是仅创建了排队记录。此时修复版正式结果尚无完成回执。

源机旧调度器 PID `632570` 已停在安全派发边界，15 个在途 worker 继续计算，由接管器保留原预算并回收真实退出状态。第二次交接在快照时仍为 DRAINING，尚不能写成 LFR 已开始。服务端控制器负责后续两个阶段；关闭 Mac Terminal 不会中断已脱离终端的任务。

用户提供的单价为新机 ¥0.88/小时、原机 ¥1.88/小时；同时开机合计 ¥2.76/小时，不包含平台可能另计的存储等费用。没有根据小样本训练时间外推整批精确完工时间，也没有设置持续调用 AI 的进度轮询。

## 2. 数据、环境、任务如何保持可比

- 复制原机已经安装好的 Linux Python 3.11.16 环境及 FRAPPÉ 专用 TensorFlow 2.14 环境；没有重新求解依赖后宣称环境相同。压缩包约 4.55 GB，打包约 13 分钟、传输约 10 分钟。后续时间用于实际缺陷诊断和修复验证。
- 通过 SSH 传递复制包，在新机校验 SHA-256 后解包。Mac 只做流式转发，没有落地真实数据复制包。此 SHA 校验用于传输一致性，不等于上游软件发布者认证。
- 两端独立核对原注册的 **70 个源文件和 4 个 prepared 文件**，全部一致。原注册数据是 F/C/S 开发分区；此次没有新的 2024 测试评估，也没有根据 S/T 排名决定修复方案。
- 8,054 个已注册 seed job 的所有权形成完整、无重叠分区：源机 7,754，新机 300。新机只接收 FRAPPÉ；原 LFR 共享表示缓存始终归源机，不拆散共享缓存组。
- 新机 300 个和源机 180 个构成 **全部 480 个预注册 FRAPPÉ seed 任务**，不是只挑选旧版失败或性能较好的候选。超参数、seed、训练轮数、分区和科学计算预算保持原值。
- 原 14 个冻结根文件与保护 tag 一致；`.gitignore` 与本轮 HEAD 一致。历史源代码和结果未覆盖，无 stage/commit。

## 3. 部署时发现的 FRAPPÉ 问题及修复

旧实现即使固定随机种子、初始权重和基础模型输出，多次独立训练仍会得到不同的修正模型。诊断中，第一轮训练批次相同，但第二轮开始的 MinDiff 分组批次发生分歧；一个 32 行合成案例的重复预测最大差异为 0.00569。

单独设置 `TF_DETERMINISTIC_OPS=1` 不足以解决；单独关闭 oneDNN、只去掉预取、只限制私有线程池也不足以稳定复现。当前 Linux 正常路径确实执行了 TensorFlow 确定性开关，不能把本次问题直接归因于未执行这个开关。旧实现在线程配置抛出异常时可能跳过后续确定性设置，是另一个防御性问题，未在本次改动冻结代码。

修复为新增 `src/nhis_fairbias/benchmark/adapters/adapter_frappe_pipeline.py`：仅对 `pack_min_diff_data` 生成的训练数据集设置 `options.autotune.enabled = False`。保留 shuffle seed、逐轮重洗、repeat、batch、loss、优化器、epochs 和分区。使用局部代理对象，没有全局 monkeypatch TensorFlow 模块。正式修复环境同时显式设置 `TF_DETERMINISTIC_OPS=1`，oneDNN 开关保持未设置。

**已观察到的证据**是批次分歧和关闭 autotune 后的稳定重复；具体 C++ 调度/随机流消耗机制尚未被直接插桩证明，不能把机制推断写成已经完全证明的因果链。小规模跨机器一致性也不证明任意硬件、所有完整训练都逐位一致。

新增独立 worker `scripts/run_nhis_runtime_worker.py` 在读取数据前检查运行时版本、启动环境、三个相关文件 SHA 和运行时身份，再调用原生产 worker 路径。修复身份为 `frappe_pipeline_deterministic_v1_20260917`。原 `adapter_frappe.py` 未修改。

## 4. 验证覆盖和边界

| 验证 | 结果 | 说明 |
|---|---|---|
| Linux 调度/交接定向测试 | 24 passed，29.35 秒 | 修复 run-lock 释放竞态之后；随后另行验证最终运行时身份新增检查 |
| 最终运行时与 adapter 测试 | Linux 14 passed；本地 14 passed | Linux 5 条第三方依赖弃用警告；不把警告写成训练失败 |
| 原入口生产 worker 合成训练 | LR×2、GBDT×2，全部 VALID | 4 次重载全部验证；独立进程重复预测完全相同 |
| 最终 adapter 重复与重载 | LR、GBDT 各 3 次训练、各 3 次重载一致 | 保留逐轮重洗；合成数据，不读取 NHIS |
| 真实训练批次跟踪 | 3 次 LR、每次 64 个批次相同 | 与未插桩训练最终输出一致 |
| 跨机器复核 | LR 输入、预测和权重哈希一致 | 源机与 CPU 节点同一生成案例；GBDT 跨机未另测 |
| 分块预测 | 最大差异 2.98e-8 | 符合已有 1e-7 容差，不声称分块与整批逐位相同 |
| 正式启动核验 | CPU 22 个有效运行进程、源机 15 个在途进程 | 只检查身份、进程和资源，不宣称正式预测性能已验收 |

另外修复了交接中的真实 Linux 竞态：旧父进程已呈 zombie 状态，并不保证运行锁已释放。新封装在恢复派发前对实际锁做最多 10 秒的有界获取检查，避免误报“锁仍被占用”导致队列停转；合成测试覆盖成功与非零退出的交接，不吞掉 worker 退出码。

合成诊断摘要由子代理产出，主管另外检查最终源文件与哈希，实际运行生产入口四次，并独立完成源机跨机器核验。没有仅凭子代理的“通过”报告批准正式运行。

## 5. 结果版本与后续必须完成的工作

修复版输出目录，两台机器相同路径、不同文件系统：

`/root/autodl-tmp/fairbias/artifacts/nhis/benchmark_frappe_pipeline_20260917`

历史目录保持：

`/root/autodl-tmp/fairbias/artifacts/nhis/benchmark_autodl_20260916_075038Z`

原 FRAPPÉ 快照已经有 54 个 VALID 状态结果，另有在途任务。这些表示旧 worker 验证成功，**不表示解决了重复训练问题**。旧版记录作为历史保留；主比较必须使用这次完整 480 个任务的明确版本，不能混入旧版 FRAPPÉ 结果补齐数量。

本次部署尚未完成以下研究工作：

1. 收齐新机 300 与源机 180 的修复版回执，逐项验证退出、预算、模型重载、输出哈希和运行时身份。
2. 构建严格的跨节点、跨方法版本结果索引：原目录取未变更方法，新目录取明确批准的 FRAPPÉ 版本；处理缺失、重复与失败，不能把多个目录直接拼表。由于旧完整队列已改为分工，旧目录本身也不会自动生成全部 8,054 个最终回执。
3. 原 LFR 12 个预算失败保留，其余队列继续原预算。本次没有放宽预算或伪装为成功；如果后续仍失败，要先独立诊断再批准新版本。
4. EG 数值修复正式恢复、BM 中间几何失败、Joint 的后续验收仍按晨间报告推进，不能因为第二节点上线而声称这些已完成。
5. 所有有效性与版本问题关闭后，再做预注册选择、调查设计推断、时间外评估和论文结果表。不能由当前进程数推导 FairBias 优于其他方法。

本轮主管决定：**接受 CPU 节点部署和有界的运行时修复准入；保持正式 benchmark 结果待验收。**

## 6. 审计定位

全部证据位于 `docs/reports/fairbias_cpu_replica_20260917/evidence/`：

- `supervisor_verification.json`：主管独立核验、测试范围与未完成事项。
- `fairbias-cpu_formal_launch.json`、`fairbias-autodl_formal_launch.json`：实际启动命令、PID、时间、入口 SHA。
- 两份 `*_postlaunch_snapshot.json`：本次有限进程/资源/状态快照。
- 两份 `*_final_release.json`、`final_operational_sources.json`：最终部署版本哈希；早期 `*_wrapper_transfer.json` 仅描述早期版本，不能代替最终版本清单。
- `runtime_plan.json`、两份 activation、两份 policy 和 `source_phases_config.json`：不可重叠分工及自动阶段配置。
- `frappe_seed_diagnostics_summary.json`、`frappe_seed_diagnostics_manifest.json`：诊断摘要和 112 份远端诊断文件清单。
- `production_worker_synthetic_summary.json`、`source_pipeline_crosshost.json`、`validation_evidence_transfers.json`：主管生产入口与跨机证据、传输哈希。
- `source_first_drain_complete.json`：首次 15 个在途任务全部封存、失败 0；本轮第二次交接完成状态应查看原机对应 handoff 目录，不能复用首次完成证明冒充第二次已完成。

关键身份：

| 对象 | SHA-256 / 身份 |
|---|---|
| 原注册 | `340cd60f6ab0d5919e2ac7bc22b62bdb3cc758faaa986650abdb3978b964732a` |
| 原注册源身份 | `818f01dbe1fcaa119493efa78acabf165961d1fdeaabb8e23f36a86dce7d5891` |
| 修复运行时源身份 | `36593df7cadb740a0063d63729007b51c5bdc1d375281c2e12a4f74fd7b4ca04` |
| 最终修复分工计划 | `57a1fbce80a5c81be7d6a22f576936dbb4cb894b8e705bbad1067b76fd8c35c1` |
| Linux 复制包 | `e1780ce95087af9c65728ba9c3cba5d366ea8b886a93634a55ed806d2bcb4119` |

