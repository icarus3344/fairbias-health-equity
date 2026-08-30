# FairBias Round 4.1 Repair-2：模式诚实命名、类别失败 scope、措辞清理与产物披露修复报告

Gate: REWORK round 4.1 REPAIR-2 — 响应 Codex 监督者 2026-08-30 对 Round 4.1 Repair 轮报告的 REPAIR 判定（四项：P1 模式名过度表述、P1 类别失败误记为幂流穷尽、P2 strict paper 残留措辞与 Decision 0005 旧名、P2 运行产物漏记；不进入 MEPS Gates 6–13，不暂存不提交，等待复审）

Status: COMPLETE — 四项判定全部落地：模式更名为 `official_code_derived_monotone_cursor_unweighted`（官方代码衍生·带终止安全扩展的变体，非官方代码行为等价声明）；类别失败按 `categorical_merge_chain` / 数值按 `official_power_stream` 分别记录，pipeline termination note 依真实 `search_scope` 生成，新增 6 项回归测试；全部 "strict paper" 残留措辞与 Decision 0005 旧名清除；Repair-1 报告补录完整六份产物，脚本默认预算改为 10 后一次执行只生成预声明的四份可比产物（新模式名下重新生成，数字与既往逐一一致）。

---

## 1. 监督者 REPAIR 判定逐项修复

### 1.1 [P1] "官方代码兼容/重实现"仍是过度表述 → 采用最小方案：诚实更名

**诊断确认**：Repair-1 报告 Assumptions #1 已承认单调游标是对官方实现的显式偏离，但模式名 `official_code_unweighted_reimplementation` 与 config.py / pipeline.py 的模式契约仍声称"官方代码兼容模式"。游标永久排除先前拒绝的幂次，而官方实现每次重访从头重搜——其他属性变化后这些幂次本可能重新可接受；两份数据轨迹不变只是经验观测，不能证明一般行为等价。

**修复（更名方案，监督者建议的最小方案）**：
- `ALGORITHM_MODE_OFFICIAL = "official_code_derived_monotone_cursor_unweighted"`（[src/fairbias/config.py](../../src/fairbias/config.py)）；模式契约改写为 **"OFFICIAL-CODE-DERIVED VARIANT WITH A TERMINATION-SAFETY EXTENSION（官方代码衍生·带终止安全扩展的变体）"**：继承官方代码的固定 dim=2 与交错幂流（inherited），但显式声明游标为 **DELIBERATE DEVIATION**（官方实现 restart-from-head，游标永久排除已消费幂次，一般行为可能分歧；轨迹不变是经验观测非等价保证），并声明**真正的官方行为参考模式（状态循环检测 + fail closed）日后另建为独立模式，本轮未实现**。
- 更名覆盖：[pipeline.py](../../src/fairbias/pipeline.py)（模块/类 docstring、`algorithm_mode_definition`、单终态键 `final_results_official_code_derived_monotone_cursor_unweighted`、`final_states`、`best_selection_reason`、state 字符串）、[bias_metric.py](../../src/fairbias/bias_metric.py)（固定维数注释）、[mitigation.py](../../src/fairbias/mitigation.py)（游标注释，含显式偏离声明）、两个脚本（[run_fairbias_benchmark.py](../../scripts/run_fairbias_benchmark.py) 帮助文本/输出标签、[empirical_rework_validation.py](../../scripts/empirical_rework_validation.py) docstring/输出标签）。
- 测试（[tests/test_fairbias_algorithm_modes.py](../../tests/test_fairbias_algorithm_modes.py)）：模块 docstring 记录两轮更名史与新边界；payload 键/`final_states`/`termination.algorithm_mode` 断言更新；负向断言同时钉住两个历史名（`final_results_paper_strict`、`final_results_official_unweighted_reproduction`、**`final_results_official_code_unweighted_reimplementation`**）均不得复现。

### 1.2 [P1] 类别属性失败被误记为幂流穷尽 → 按属性角色分别记录 scope

**诊断确认**：监督者已构造类别属性失败并复现——原 [mitigation.py](../../src/fairbias/mitigation.py) stop 分支只要 `preserve_exponent_order=True` 就无条件写 `search_scope="official_power_stream"`、`stream_positions_consumed=0`、"official power stream exhausted"，而该分支同样处理 `_search_categorical` 的失败。

**修复**：
- `mitigate_step` stop 分支按**属性角色**分流（`selected_attribute in cate_attrs` 或非数值 dtype）：
  - **数值**：`search_scope="official_power_stream"` + `stream_positions_consumed`（游标计数），reason 阐明幂流在单调游标下穷尽、无位置可重试、无迭代预算；
  - **类别**：`search_scope="categorical_merge_chain"`，**不含** `stream_positions_consumed`（类别搜索不产生游标状态），reason 阐明合并链（含被拒的终态 drop）穷尽、无迭代预算。
- [pipeline.py](../../src/fairbias/pipeline.py)：termination note 由新模块级函数 `_termination_note(termination_reason, is_official, non_convergence)` 依**真实记录的 `search_scope`** 生成——类别失败不再被描述为幂流穷尽；engineering 分支 note 不变。
- **回归测试 6 项**（新增两个测试类）：
  1. `TestCategoricalFailureScopeRecording.test_categorical_failure_records_merge_chain_scope`：构造类别属性失败（mock `_make_candidate` 全拒），断言 `search_scope="categorical_merge_chain"`、无 `stream_positions_consumed` 字段、reason 含 "categorical merge chain" 且不含 "power stream"、游标字典为空——修复前该场景必失败（scope 被误写为 official_power_stream）。
  2. `TestCategoricalFailureScopeRecording.test_numeric_failure_still_records_power_stream_scope`：对照断言数值失败仍记 `official_power_stream` 且 `stream_positions_consumed=len(stream)`。
  3–6. `TestTerminationNoteFollowsRecordedScope`（4 项）：note 函数对类别 scope（含 "categorical merge chain"、不含 "power stream"）、数值 scope（含 "power stream" 与 "monotone"）、official 非穷尽分支、engineering 穷尽分支的分别断言。

### 1.3 [P2] "strict paper" 残留措辞清理

- [config.py](../../src/fairbias/config.py) `transform_x_max` 注释：`None = strict paper mode` → `None = NO additional magnitude guard`（具体语义：唯一幅值界是论文 float32 溢出规则；数值 x_max 是非论文工程护栏）；`failed_attribute_mode` 注释：`"stop" (default, strict paper)` → `"stop" (default)`，语义改为"最高 d_phi 属性失败即停止"。
- [mitigation.py](../../src/fairbias/mitigation.py)：`__init__` 校验错误消息与两处注释、`mitigate_step` docstring 同步去除 "strict paper" 标签，改用具体语义。
- [transform.py](../../src/fairbias/transform.py)：`x_max` 注释同 config.py 改法。
- [Decision 0005](../decisions/0005-round4-commit-boundary-audit.md)：Consequences 段的旧模式名 `official_unweighted_reproduction` 更新为"历经两轮更名、现名 `official_code_derived_monotone_cursor_unweighted`（官方代码衍生·带终止安全扩展的变体，非官方代码等价声明）"。
- 两份既有报告（MODE_SEPARATION / REPAIR）的过度表述与旧名以修订注记 + 就地改写清除（见 §3）。

### 1.4 [P2] 实证运行产物记录不完整 → 完整披露 + 脚本默认预算改为 10

**诊断确认**：14:45 的重跑实际共生**六份**产物（监督者给出两份被漏记的哈希，本地逐一复核一致）：脚本一次执行生成 4 份（official × 2 + engineering × 2 默认 `max_iterations=3`），另以 `max_iterations=10` 单独补跑 engineering × 2；Repair-1 报告仅列四份。

**修复（双管齐下）**：
- [REWORK_FAIRBIAS_ROUND4_1_REPAIR.md](REWORK_FAIRBIAS_ROUND4_1_REPAIR.md)：顶部加 Repair-2 修订注记；Output hashes 补录完整六份（含两份预算 3 的产物及其 SHA-256：`740854e8…` / `c47b5b53…`，明确标注"非可比集合成员"）；§2 与 Commands executed 的口径同步改写。
- [empirical_rework_validation.py](../../scripts/empirical_rework_validation.py)：默认 `max_iterations=10`（预声明的可比预算），并注释说明——**一次执行恰好生成预声明的四份**（两数据集 × 两模式；official 无预算故参数无关）。
- 本轮以新模式名重新生成四份可比产物（§2），旧六份保留为不可变历史记录。

## 2. Repair-2 后双模式实证重跑（seed=0；一次执行、恰好四份；描述性观测）

`PYTHONPATH=src:scripts .venv311/bin/python scripts/empirical_rework_validation.py` 单次执行生成四份，终止状态、轮数、ACC/EO 与 round-4.1 / Repair-1 报告**逐一一致**（更名与 scope 修复不改变任何算法行为；engineering 路径代码零改动）：

| 运行 | 模式 | 终止 | test ACC | test EO |
|---|---|---|---|---|
| COMPAS | official-derived | converged=true · epsilon_reached · 4 轮（terminal_max_dphi=0.0009444） | 0.6397 | 0.0207 |
| Credit | official-derived | converged=true · epsilon_reached · 11 轮（terminal_max_dphi=0.0001659） | 0.8153 | 0.0142 |
| COMPAS | engineering（预算 10） | converged=true · epsilon_reached · 4 轮 | 0.6397 | 0.0207 |
| Credit | engineering（预算 10） | converged=false · iteration_budget_exhausted · 10 轮（terminal_max_dphi=0.0002168） | 0.8133 | 0.0238 |

（engineering COMPAS Pareto 选 iter 2：test ACC=0.6713/EO=0.2022；engineering Credit Pareto 选 iter 1：test ACC=0.8083/EO=0.0151——均与既往一致。四份运行均无 non-convergence 记录，故本轮 scope 修复不影响其实证输出。）

## 3. 报告修订（两份既有 + 本报告）

- **[REWORK_FAIRBIAS_ROUND4_1_MODE_SEPARATION.md](REWORK_FAIRBIAS_ROUND4_1_MODE_SEPARATION.md)**：新增 "Round 4.1 Repair-2 修订" 块（四项判定摘要）；Status/§1 P1-2/§3 各运行标注/表述边界/Assumptions #1 的旧名与"官方代码兼容/复现官方代码行为"表述改写为"官方代码衍生·带终止安全扩展的变体 + 显式偏离 + 不作等价声明"。
- **[REWORK_FAIRBIAS_ROUND4_1_REPAIR.md](REWORK_FAIRBIAS_ROUND4_1_REPAIR.md)**：顶部 Repair-2 修订注记（含历史名保留说明）；§2/Commands/Output hashes 六份产物完整披露。
- **本报告**（新增，`REWORK_FAIRBIAS_ROUND4_1_REPAIR_2.md`）。

## 4. 标准报告模板

Gate: REWORK round 4.1 REPAIR-2（如上）

Status: COMPLETE（四项判定全部落地；6 项新回归测试；四份可比产物以新模式名重新生成且数字逐一不变；未暂存未提交任何内容，等待 Codex 重新审阅）

Files changed（相对 HEAD=`b9cc9ca`，与 Repair-1 叠加的累计工作区改动；括注本轮新增触碰）:
- `src/fairbias/config.py` — 模式常量值 `official_code_derived_monotone_cursor_unweighted`、模式契约改写为"衍生变体 + 显式偏离 + 官方参考模式另建"声明、`resolved()`/错误消息/预设 docstring、strict-paper 措辞清理
- `src/fairbias/mitigation.py` — stop 分支按属性角色分流 scope（数值 `official_power_stream`+游标计数 / 类别 `categorical_merge_chain`）；游注释义加显式偏离声明；strict-paper 措辞清理
- `src/fairbias/pipeline.py` — 更名贯穿（单终态键/`final_states`/`algorithm_mode_definition`/`best_selection_reason`/state）；新增 `_termination_note()` 依真实 `search_scope` 生成 note
- `src/fairbias/bias_metric.py` — 固定维数注释更名（本轮触碰）
- `src/fairbias/transform.py` — x_max 注释 strict-paper 清理（本轮触碰）
- `scripts/run_fairbias_benchmark.py` — 帮助文本/输出标签更名
- `scripts/empirical_rework_validation.py` — 更名 + 默认预算 10（一次执行恰生成四份）+ non-convergence 打印按 scope 分述
- `tests/test_fairbias_algorithm_modes.py` — 更名断言 + 两个历史名负向断言 + 新增 `TestCategoricalFailureScopeRecording`（2 项）与 `TestTerminationNoteFollowsRecordedScope`（4 项）
- `docs/decisions/0005-round4-commit-boundary-audit.md` — Consequences 旧名更新（本轮触碰）
- `docs/reports/REWORK_FAIRBIAS_ROUND4_1_MODE_SEPARATION.md` — Repair-2 修订块与过度表述改写（本轮触碰）
- `docs/reports/REWORK_FAIRBIAS_ROUND4_1_REPAIR.md` — 修订注记 + 六份产物补录（本轮触碰）
- `docs/reports/REWORK_FAIRBIAS_ROUND4_1_REPAIR_2.md` — 本报告（新增）

Commands executed:
- `PYTHONPATH=src .venv311/bin/python -m unittest tests.test_fairbias_algorithm_modes tests.test_fairbias_mitigation`（`Ran 36 tests` / `OK`）
- `PYTHONPATH=src .venv311/bin/python -m unittest tests.test_fairbias_pipeline tests.test_fairbias_mds_dimension_golden tests.test_fairbias_clean_checkout`（`Ran 14 tests` / `OK`）
- `PYTHONPATH=src .venv311/bin/python -m unittest discover -s tests -p "test_*.py"`（混合工作区回归套件：`Ran 246 tests in 105.350s` / `OK`，EXIT=0；含未提交 MEPS Gates 6–13 测试文件，非拟提交提交自身的独立证据口径）
- 拟提交树验证：`git archive HEAD`（b9cc9ca）解压至 `runs/_repair_check_2/` → 叠加本轮 8 个 fairbias 修复文件 → 干净树内 `PYTHONPATH=src … unittest discover -s tests -p "test_*.py"`（`Ran 164 tests in 70.962s` / `OK (skipped=3)`，EXIT=0；3 项 skip 为 Git checkout 专属测试在无 .git 目录下的预期跳过；全部文件定稿后同步复跑再次 `Ran 164 tests in 15.722s` / `OK (skipped=3)`，EXIT=0）
- 双模式实证重跑：`PYTHONPATH=src:scripts .venv311/bin/python scripts/empirical_rework_validation.py`（单次执行，恰好生成预声明的四份：official × 2 无预算 + engineering × 2 预算 10）
- `shasum -a 256`（输入/输出取证）；`git diff --numstat` / 冻结文件零改动核验

Permissions requested: 无（无网络访问意图；混合工作区套件中既有未提交 MEPS 下载测试按其自身设计写入 stub 文件，非本轮行为）

Tests executed: 混合工作区回归套件（246 项）+ fairbias 定向（36 + 14 项）+ 拟提交树全部已跟踪测试（164 项，b9cc9ca + Repair-1/2 叠加的干净导出）。

Exact test results:
- 混合工作区回归套件：`Ran 246 tests in 105.350s` / `OK`（Repair-1 时点为 240 项；净增 6 = 本轮 scope/note 回归测试）
- 定向（algorithm_modes + mitigation）：`Ran 36 tests in 2.435s` / `OK`；定向（pipeline + golden + clean_checkout）：`Ran 14 tests in 12.998s` / `OK`
- 拟提交树（b9cc9ca + Repair-1 + Repair-2 叠加，干净导出）：`Ran 164 tests in 70.962s` / `OK (skipped=3)`（监督者复核的 Repair-1 时点为 158 项；净增 6 同上）

Input hashes:
- `data_COMPAS.csv` — `4b2bc1d55553f1c2061768bd245b725ee57b3f7e438f612b78c60f308848cb27`（与 round-4/4.1/Repair-1 一致）
- `data_Credit_Card.csv` — `303cf916663273a345671688c96e9be7a83c76e39f0a2727f8d0ed33b8d6df1e`（与 round-4/4.1/Repair-1 一致）
- fixture `tests/fixtures/official_uciadult/distance_matrix_step_0.csv` — `5f9fd4f81766ce724bd9866bd139503d252f68c9bc7e5065ff34ace0f57228e1`（与 PROVENANCE.json 一致）
- 代码状态：HEAD=`b9cc9ca`（未提交任何修复，等待重新审阅）

Output hashes（runs/rework_empirical/，唯一 run_id 不覆盖；Repair-2 后新模式名单次执行生成，恰四份）:
- official-derived COMPAS `fairbias_compas_seed0_20260830_181506/results.json` — `5a37ec70bea733206b687ef3628c3e3a46f33db0f64646516f587acb7a4a8f9e`
- official-derived Credit `fairbias_credit_seed0_20260830_181511/results.json` — `bce96e285e4a4918ba7760302b9150ae301ad019137b6e1a51cd4c104abc3a2a`
- engineering COMPAS（预算 10）`fairbias_compas_seed0_20260830_181513/results.json` — `baa2cf0d07f08588b8d970e39f3ff203b5378b72194e50877b63c8c1444bd0ee`
- engineering Credit（预算 10）`fairbias_credit_seed0_20260830_181524/results.json` — `4e186f945beef517e422ae29c0cee0ace48187cbf7b12d957bcff994c8928e95`
- （Repair-1 时点旧模式名的六份产物——含两份本轮补录的预算 3 产物——见 [REWORK_FAIRBIAS_ROUND4_1_REPAIR.md](REWORK_FAIRBIAS_ROUND4_1_REPAIR.md) Output hashes 完整清单，保留为不可变历史记录）

Row counts（observed，与 round-4/4.1/Repair-1 一致）:
- COMPAS：train=3950 (0.6400) / validation=987 (0.1599) / test=1235 (0.2001)，总 6172
- Credit：train=19200 (0.6400) / validation=4800 (0.1600) / test=6000 (0.2000)，总 30000

Assumptions:
1. 模式边界（承监督者裁定）：`official_code_derived_monotone_cursor_unweighted` = 官方代码衍生·带终止安全扩展的变体——继承固定 dim=2 与交错幂流，游标为显式偏离；**不作官方代码行为等价声明**（轨迹不变是经验观测）；亦非论文正文方法严格复现（论文正文模式未实现）。真正的官方行为参考模式（状态循环检测 + fail closed）**本轮未实现、未声称**，日后另建为独立模式。
2. 类别失败 scope 修复不改变算法行为（仅改失败记录字段与 note 文案）；四份可比产物数字逐一不变即为回归证据。
3. engineering 路径行为与 b9cc9ca 完全一致（本轮 engineering 路径代码零改动；`transform.py` 仅注释变更）。
4. 拟提交树口径：b9cc9ca + Repair-1/Repair-2 的 fairbias 修复叠加（未含未提交 MEPS 文件）；164 = 监督者复核的 158 + 本轮 6 项新测试。

Unresolved issues:
1. 官方行为参考模式（循环检测 + fail closed）未实现——监督者已裁定日后另建，非本轮范围。
2. 数值编码类别（承 round-3/4.1）：MEPS 接入时必须由变量字典显式指定 numeric/categorical。
3. MEPS 侧 P1/P2（study.json arm 过度宣称、meps pipeline.py 硬编码 136）：归入 MEPS Gates 6–13 整理 gate，本轮未触碰。
4. MEPS Gates 6–13 未提交文件：保持原状，未纳入本轮。
5. survey-weighted 实现（Gate D）：仍为预声明。
6. 多 seed 稳健性运行仍属论文实验协议差距（如实列为未做）。

Git diff summary:
- 本轮 + Repair-1 累计（相对 HEAD=`b9cc9ca`）：`src/fairbias/{config,mitigation,pipeline,bias_metric,transform}.py`、`scripts/{run_fairbias_benchmark,empirical_rework_validation}.py`、`tests/test_fairbias_algorithm_modes.py`、`docs/decisions/0005-round4-commit-boundary-audit.md`、`docs/reports/REWORK_FAIRBIAS_ROUND4_1_MODE_SEPARATION.md` 修改；新增 `docs/reports/REWORK_FAIRBIAS_ROUND4_1_REPAIR.md` 与本报告（未跟踪）。
- 14 个冻结基线根文件与 `.gitignore` 零改动（`git diff HEAD -- <冻结清单>` 为空）。
- 未提交 MEPS 工作区文件（`src/meps_fairness/`、`tests/test_gate*.py`、`tests/test_download_meps.py` 等）未触碰，不属于本轮 diff。
- 无暂存（`git status --porcelain` 无 `^[MA]` 行）、无新提交。

Proposed next step:
1. 监督者重新审阅：(a) 模式更名与诚实边界（含两个历史名负向断言）；(b) 类别/数值 scope 分离与 6 项回归测试（含监督者构造的类别失败情形）；(c) strict-paper 清理与 Decision 0005；(d) 六份产物完整披露 + 脚本默认预算 10 后单次执行恰生成四份；(e) 混合套件 246/246、拟提交树 164/164（skipped=3）、四份产物数字逐一不变。
2. 审阅通过后授权提交 Repair-1 + Repair-2 累计改动。
3. 之后再进入 MEPS Gates 6–13 整理 gate。

STOP — waiting for Codex review.
