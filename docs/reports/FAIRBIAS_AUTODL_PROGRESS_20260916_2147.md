# AutoDL 进度核验：2026-09-16 21:47–21:49 CST

本次只核查运行、资源、登记源码一致性和 FRAPPÉ 环境；未重启调度、变更预算/模型或启动新的真实数据拟合。正式队列继续运行。

## 21:47:13 运行快照

- 已封存 2,604 / 6,614 项当前 CPU 队列任务，约 39.4%；相对于完整 8,054 项矩阵约 32.3%。这是尝试结束比例，不是整项研究完成比例。
- VALID 2,542；FAILED 16；BUDGET_EXHAUSTED 17；TIME_LIMIT 29。
- PID 48932，15 个活跃 worker；最近 10/30/60 分钟分别封存 94/173/348 项。
- 全部登记源码/配置 SHA 检查一致，尚无 selection_freeze.json 或 completion_result.json。未进入本批次 2024 评价。

| 方法 | 已封存/计划 | VALID | 其他状态 |
|---|---:|---:|---|
| UNMITIGATED | 241/312 | 241 | — |
| FAIRBIAS_BM | 16/3520 | 16 | — |
| REWEIGHING | 84/96 | 84 | — |
| EG_DP | 642/768 | 632 | FAILED 10 |
| EG_EO | 633/768 | 627 | FAILED 6 |
| TO_EO | 83/96 | 83 | — |
| OXONFAIR_EO | 354/384 | 354 | — |
| FAIRRET_EO | 75/80 | 75 | — |
| FAIRGBM_EO | 411/480 | 411 | — |
| FAIRBIAS_BM_AE | 40/40 | 19 | BUDGET_EXHAUSTED 16，TIME_LIMIT 5 |
| FAIRBIAS_JOINT | 25/40 | 0 | BUDGET_EXHAUSTED 1，TIME_LIMIT 24 |
| 三种几何消融 | 0/30 | 0 | 在当前队列中等待 |
| FRAPPE_EO | 0/480 | 0 | 尚未开始正式拟合 |
| LFR_RECONSTRUCTED | 0/960 | 0 | 预算准入尚未解决 |

## 资源

三秒 cgroup 采样使用约 15.17 CPU equivalents，占16配额约94.8%；memory.current约19.81GiB/80GiB，包含缓存。memory.events没有OOM或OOM kill。数据盘约16/50GB；GPU利用率0%、显存0MiB，符合当前CPU实现。

## 需要关注的问题

1. **FairBias 主队列尚未展开。** 调度器 `discover()` 按 `(representation_key or "", candidate_id, seed)` 排序：没有共享表示键的普通/EG/AE/Joint等任务优先，BM和几何条件排在后面。16项来自前期已完成任务；不是证明BM卡死，也不能按外部方法完成比例推断BM已完成。源码位置：`src/nhis_fairbias/benchmark/parallel_execution.py:429`附近。
2. **Joint预算不可忽略。** 已结束25项均没有VALID结果，其中24项达到1800秒墙时，1项内部预算退出。BM→AE虽然40项都结束，但仅19项VALID，不能称消融成功完成；种子不完整的配置不能删除失败种子后正常选优。
3. **EG数值边界失败增至16项。** 本次这些FAILED记录的错误均为 `adapter decision output must lie in [0, 1]`。先前一个候选已用独立诊断确认机器精度舍入；本次没有重训逐一证明新增失败均同因，保留这一界限。
4. 17项内部预算退出中，14项为AE提交次数上限且停止准则未建立，3项为MDS迭代上限、收敛未验证。29项TIME_LIMIT均为1800秒墙时退出，不是OOM。

## FRAPPÉ环境的新进展

安装报告已于17:40 CST生成，安装日志显示成功，原安装进程已退出。本轮核查专用环境：pip check通过；TensorFlow 2.14.0及MinDiff导入通过，TFMR 0.1.7.1、numpy1.26.4、pandas2.1.1。

导入阶段有CUDA插件重复注册诊断信息，但进程正常返回并成功导入；这不是GPU训练准入。合成拟合、固定种子、新进程重载以及完整依赖来源准入仍未在本次服务器环境完成，不能据安装成功启动正式480项。

## 剩余工作与工期解释

当前队列还有4,010项未封存（含在途任务），完整矩阵还有FRAPPÉ480和LFR960待准入。之后才是完整回执核验、配置冻结、服务器调查统计验收、2024评价和论文图表。最近一小时吞吐不能直接用于后面的BM阶段；目前不作稳定的总完成时间承诺。

快照依据：服务器registration、parallel_scheduler_events.jsonl、live status、逐任务result.json、/proc、cgroup、依赖安装报告及本轮环境导入。未修改既有失败记录。
