# Joint / BM_AE 恢复入口：待独立审查

Gate:
2026-09-17 Joint/AE recovery readiness；仅新增代码、合成测试和失败元数据清单。未运行真实 F/C，未查看 S/T 比较指标。

Status:
实现已交付 parent 审查，尚未自行批准 gate 或正式研究重跑。原 Joint 40 项失败、BM_AE 21 项失败仍保持原状态。

Files changed:
- `src/nhis_fairbias/benchmark/adapters/adapter_joint_ae_recovery.py`
- `tests/benchmark/test_joint_ae_recovery.py`
- `docs/reports/fairbias_joint_ae_recovery_20260917/failed_tasks.json`
- 本报告。

入口复用已接受的 NumPy MDS exact kernel、原 FairBias adapter、scheduled controller 和 MDS retry。新的状态检查拒绝未验证的完整性，并阻止旧 AE 的 `except Exception` 隐藏几何错误。全局临时替换在异常及中断后恢复；一进程一拟合，禁止在线程内并发使用。

| 显式选项 | 保留及变化 | 可用返回状态 |
|---|---|---|
| `controller='strict'` | 原顺序和注册预算；仅 exact kernel 与错误传播修复 | `COMPLETE_FEASIBLE`；其余抛出错误 |
| `controller='strict', ae_cap_retry=40` | 仅 BM_AE：先原 cap10；确切 AE 提交上限退出后同种子从头 cap40；utility500/geometry20000/BM50 原值保留 | 有实际搜索耗尽证据才 `COMPLETE_FEASIBLE`；cap40仍耗尽为 `NO_MODEL_BUDGET_EXHAUSTED` |
| `use_mds_retry=True` | 显式数值预算扩展：符合已有规则时，MDS 从同种子重拟合至双倍 cap | 沿用上述状态；保留每个 MDS 尝试 |
| `controller='scheduled'` | 仅 Joint：改变 AE 搜索顺序与时间控制；单列算法敏感性 | `COMPLETE_FEASIBLE`、`FEASIBLE_BUDGET_LIMITED`、`FEASIBLE_GEOMETRY_INCOMPLETE` |

`search_order_changed`、`mds_iteration_budget_changed`、`ae_commit_budget_changed` 分别记录。任何返回状态都不自动构成正式 benchmark `VALID`。新预算/顺序的结果不能伪装成旧注册的严格路径。旧 `experiment_worker` 会继续执行 S 校准比较，因此不应直接把新 adapter 塞入旧 factory。

Commands executed:
读取协议、compute repair/scheduled review/morning delivery、已验证组件和原 adapter。通过授权 SSH 只读取原 run 的 registration/result/receipt 元数据；首次 `python3` 不存在，改用原 `.venv311/bin/python -B` 后成功。未加载数据、预测或模型。61 项清单最终由 parent 已获取的 `failure_inventory.json` 派生，与远程计数一致。运行下面的合成测试及哈希核验。

Permissions requested:
无新增权限请求；未自行启动远程训练；未 staging/commit。

Tests executed:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv311/bin/python -B -m pytest -q tests/benchmark/test_joint_ae_recovery.py`

Exact test results:
`21 passed in 14.08s`。覆盖 Joint/BM_AE 实际非 drop 提交与原路径相同、预测逐字节一致、跨进程模型重载、可行预算退出与不允许预测的失败、cap10→fresh cap40、cap40仍未完成、重复 fit 从10重启、其他预算/中断不误触重试、F/C角色及记录边界、来源哈希和加载位置。新增 cap 测试使用确定的合成 AE 提交序列验证真实原 adapter 的 cap 分支；不声称真实 AE 在40步内会完成。

Input hashes:
原 registration SHA256 `340cd60f6ab0d5919e2ac7bc22b62bdb3cc758faaa986650abdb3978b964732a`。核验当前原注册 70 个源文件全部匹配；`fairbias_joint_scheduled_20260916/verification.json` 中 13 个接受文件全部匹配。新入口固定7个关键组件的内容及加载来源；运行 worker 仍须核验全部注册来源、prepared SHA/identity 和加载模块来源。

Output hashes:
- adapter：`139060e7314174cafaedbd0f22bde3a4d3120d0ff71cbbe60e7392fdeabba10c`
- tests：`09ccd9cb3826932660cd177562568150326c20a23fce8150e67c60bd8495f1cb`
- failed tasks：`76f426299d525ff6cb35875dd41a8c5afeffbd381874764c9d5fc2d0c83803a9`

Row counts:
真实微数据读取/拟合：0；S/T指标：0。合成每例 F48/C48。失败任务元数据61项：14个AE提交上限、4个MDS迭代上限、43个超时；BM_AE原19个VALID不改动。清单含候选原配置、seed、job/result/source/data哈希及建议修复选项，全部 `execution_admitted=false`。

Assumptions:
保持 sklearn1.9.1 与现有精确内核支持环境。不同 NumPy/BLAS 环境只能沿已有环境证据解释；本次没有新增跨环境数值等价声明。cap40是明确的新计算预算，不是已证明足够的收敛参数。

Unresolved issues:
1. 原 `adapter_fairbias_ae.py` 的 `ae_step()` 在第10次提交后、下一次调用前就抛出上限错误，尚未检验第11次有无改进；14个失败因此不能靠加速本身恢复。新40预算也可能先触及原 utility/geometry预算，仍真实失败。
2. 先前 strict NumPy Joint600秒仍未完成；4.44倍只是在相同16次几何前缀的单次测量，不等于全拟合收益。先前 scheduled LR完成、GBDT预算内可行、arm002 retry预算内可行均仅支持各自F/C诊断，不能证明所有arms/seeds完成。
3. 61项是操作清单，不是可直接汇总的完整方法比较。若采用预算/顺序变体，正式研究需新注册、完整成对配置与种子设计，不能混合旧成功模型和修复模型后称同一方法。

Git diff summary:
本worker只新增上述四个文件；共享工作树其他既有更改未操作。未修改原70来源、13个已接受修复文件、旧结果或受保护baseline。

结束核验：14个继承文件与baseline tag无差异；`.gitignore` 与tag存在历史已提交差异（commit `b595e59` 新增presentation/build-script/PDF忽略项），当前working tree对它无diff。此历史差异已上报parent，不能称整个protected集合与tag完全一致。

Proposed next step:
Parent审查新入口和21项测试后，按现有外部 supervisor 在空闲CPU上顺序运行两个 F/C pilot，每个独立进程、单线程、nice19、4GiB sampled RSS、fresh外部回执/日志；先预登记硬时间，建议各1800秒（禁止运行中悄悄续时）。

- BM_AE：candidate `3c8badc6d7eeaa99adce`、arm001、GBDT、seed0、`controller='strict', ae_cap_retry=40`。旧记录明确为 AE commit cap10；保留原10尝试及fresh40完整回执。
- Joint：candidate `cc84d0d23c1bc5af275d`、arm003、LR、seed0、`controller='strict'`，原搜索预算不变。1800秒仍未完成则明确保留超时；后续3600秒须独立预登记，不能称原预算已完成。

最简已验证 F/C API：

```python
from nhis_fairbias.benchmark.adapters.adapter_joint_ae_recovery import fit_joint_ae_recovery_fc
adapter = fit_joint_ae_recovery_fc(config, seed, F, C,
                                   controller="strict", ae_cap_retry=40)  # BM_AE only
receipt = adapter.provenance_["recovery"]
```

需要在失败后持久化两次尝试的worker应先持有adapter，再执行拟合：

```python
from nhis_fairbias.benchmark.adapters.adapter_joint_ae_recovery import make_joint_ae_recovery_adapter
adapter = make_joint_ae_recovery_adapter(config, seed, controller="strict", ae_cap_retry=40)
try:
    adapter.fit_development(F.X_semantic, F.y, F.A, C.X_semantic, C.y, C.A,
                            metadata={"F_ids": F.record_keys, "C_ids": C.record_keys})
finally:
    receipt = adapter.provenance_["recovery"]
```

调用方先核验F/C role/year/arm、注册/输入hash和来源位置，并用原子回执保留 `receipt`。仅模型可用时做C预测/重载一致性检查。新worker不得进入S/T。scheduled敏感性可沿已有独立launcher另行登记；无需再复制一套搜索器。

STOP — waiting for Codex review.
