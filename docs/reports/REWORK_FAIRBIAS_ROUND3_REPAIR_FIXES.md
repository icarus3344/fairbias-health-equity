# FairBias 论文对齐返工 Round 3：REPAIR 判定修复报告

Gate: REWORK round 3 — repair of the two blocking (P0) and three non-blocking (P1) defects cited in the Codex REPAIR verdict on round 2 (`docs/reports/REWORK_FAIRBIAS_ROUND2_PAPER_ALIGNMENT.md`)
Status: COMPLETE — true 64/16/20 split verified from observed row counts; clean-checkout import guaranteed by committed runtime modules and a new git-archive smoke test; fixed unseen sentinels; paper-only float32 bound; strict non-convergence semantics. Full suite 209/209 OK. All round-3 fixes committed to `b6ffa05` (supervisor-mandated commit). Round-2 empirical numbers are RETRACTED — new runs executed under the corrected protocol and reported below as descriptive observations only.

论文源：`/Users/lkc/Downloads/dda3020/icomputing.0083 copy.pdf`（Tang, Lu & Li 2024, Intell. Comput. 2024;3:Article 0083）。

---

## 1. 监督者判定逐项修复映射

| # | 监督者判定 | 修复 | 证据 |
|---|---|---|---|
| P0-1 | 划分实为 80/16/4（第一次 20% 进 tmp，第二次只把 tmp 的 20% 给 test），manifest 硬编码 64/16/20，测试只查 manifest 数字 | [pipeline.py](../../src/fairbias/pipeline.py)：先联合留出 `val+test=0.36`，再按 `test_size/0.36` 从 holdout 切 test，test 真实获得全体 20%；新增**观测比例防御断言**（实际 row_counts 与配置比例偏差 >0.05 直接 raise）；manifest 改记 `configured_fractions` + `observed_fractions`（由真实 row_counts 导出） | `test_pipeline_compas_end_to_end` 断言 observed = row_counts/total 且 ≈ 64/16/20（±0.05）；实证运行观测：COMPAS 3950/987/1235 = 0.6400/0.1599/0.2001，Credit 19200/4800/6000 = 0.6400/0.1600/0.2000 |
| P0-2 | HEAD=938a257 缺 `src/fairbias/models.py`，干净 checkout `ModuleNotFoundError: No module named 'fairbias.models'` | 按监督者指示将运行所需模块纳入提交：`src/fairbias/{__init__,models}.py` 补入 commit `b6ffa05`（与全部 round-3 修复同提交）；新增 `tests/test_fairbias_clean_checkout.py`：(a) 断言 HEAD tree 含全部 10 个 fairbias 运行时模块；(b) `git archive HEAD` 提取后子进程 import 全运行时（不接触工作区文件） | 提交前该测试**精确复现监督者故障**（`ModuleNotFoundError: No module named 'fairbias.models'`）；提交后 209/209 OK。工作区中 fairbias 相关文件 git status 干净 |
| P1-3 | 未知类别按各分区动态分配新码 → 编码依赖评估分区内容，validation 与 test 编码空间不一致 | [data.py](../../src/fairbias/data.py)：每个 LabelEncoder 在 train 拟合时固定 sentinel 码 `len(train classes)`；所有分区（含 Y、O）的未见类别统一映射到该 sentinel；不再有分区专属新码分配 | `test_unseen_categories_get_new_deterministic_codes`（sentinel=2，medium→{2}）；新增 `test_unseen_sentinel_is_shared_across_partitions`：validation 的 "medium" 与 test 的 "extreme"/"other" 共享同一 sentinel |
| P1-4 | `transform_x_max=1e9` 覆盖论文 float32 溢出规则，大量远未溢出的幂候选被静默拒绝 | [config.py](../../src/fairbias/config.py) / [transform.py](../../src/fairbias/transform.py)：`transform_x_max` 默认 `None`（严格论文模式：唯一量级约束是论文 float32≈3.4e38 溢出规则）；数值上限改为显式 opt-in 的非论文工程守卫 | `test_power_overflow_rejected` 重写：(1e5)^5=1e25（<3.4e38）必须**接受**；新增 `test_x_max_engineering_guard_is_opt_in`（显式 x_max=1e9 才拒绝；默认构造 x_max is None）；真溢出 (1e30)^5 由 `power_transform_overflows` 检出 |
| P1-5 | 失败属性永久屏蔽 + 跨保护组同名屏蔽 + 静默跳次高，均非论文算法 | [mitigation.py](../../src/fairbias/mitigation.py)：`failed_attribute_mode` 配置——`"stop"`（默认，严格论文）：当前最高 d_phi 属性穷尽搜索仍进不了 ε 球 → 记录 `non_convergence{label_O, attribute, d_phi}` 并停止，不静默跳次高；`"next"`（显式命名工程扩展）：按 **(protected, feature) 元组**记录失败再试次高（不再跨组屏蔽） | `test_stop_mode_records_non_convergence_on_highest_attribute`、`test_next_mode_is_named_engineering_fallback`、`test_next_mode_failure_does_not_mask_other_protected_group`、`test_invalid_failed_attribute_mode_rejected`；pipeline 将 `failed_attribute_mode` 与 `mitigation_non_convergence` 写入 results.json |
| 报告过期 | round-2 报告的 Git 状态描述已失效（HEAD 已 938a257，报告与 data.py/测试已入提交） | 本报告如实记录：round-2 报告生成后发生了两次提交（938a257 由监督方/后续操作产生、b6ffa05 为本轮按监督者指示执行的提交）；round-2 报告保留为历史文档，其 Git 状态段不再作为当前状态证明 | 本报告 §3 Git diff summary |

## 2. 修正协议下的实证重跑（round-2 结果撤回；本节为描述性观测）

**round-2 的 COMPAS/Credit 指标（4% test set + 1e9 上限 + 分区新码）全部作废**，不再作任何科学解释。正确协议（真实 64/16/20、论文 float32 界、固定 sentinel、stop 失败语义）下的 10 轮预算运行：

**COMPAS（seed=0）**：观测划分 3950/987/1235 = **0.6400/0.1599/0.2001** ✓。init max d_phi=0.0078，ε=0.0020。第 1 轮 c_charge_degree（二分类）合并至单类 → 显式 `dropped`（论文语义）；第 2 轮 score_text 合并；第 3 轮 priors_count 幂 3；第 4 轮 race 变换后 max d_phi=0.0009 < ε，**进入 ε 球提前停止**，non-convergence none。Pareto（validation）选 iteration 2（val EO=0.0021，val ACC=0.6890 ≥ 0.6749）；test 单次锁定评估 ACC=0.6713、EO=0.2022（式 9 语义）。**注意**：test EO（0.2022）与 validation EO（0.0021）差距大——报告如实记录该 validation→test 泛化差距，不声称 test 公平改善。

**Credit（seed=0）**：观测划分 19200/4800/6000 = **0.6400/0.1600/0.2000** ✓。init max d_phi=0.0028（AGE），ε=0.0002。**round-2 的"9 属性删除级联"被证实为 1e9 工程阈值的伪象**：移除该上限后，第 1 轮 AGE 即被论文幂搜索（1/7…7）压至 ε 附近（max d_phi 0.0028→0.0011），此后 10 轮逐属性处理（MARRIAGE、PAY_2/4/0/6、LIMIT_BAL、PAY_3/5、AGE 复访），全程 **0 个 dropped**（MARRIAGE 链式合并即达标，无需删除）。10 轮预算耗尽时 max d_phi≈0.0002（四舍五入显示值），**未证实进入 ε 球**（停止条件 `<= ε` 未触发），但每轮变换均被接受（无失败，non-convergence none）。Pareto（validation）选 iteration 1（val EO=0.0078）；test 单次评估 ACC=0.8083、EO=0.0151。

与 round-2 的数值（含 d_phi 绝对量级）不可比较：划分、编码空间、量级约束、失败语义均已改变。

## 3. 标准报告模板

Gate: REWORK round 3（如上）
Status: COMPLETE（§1 全部修复项落地并经 209/209 测试验证；§2 实证仅描述性观测）

Files changed（commit `b6ffa05`，13 files，+600/−69）:
- `src/fairbias/pipeline.py` — 真 64/16/20 划分 + observed_fractions + 防御断言 + non_convergence 记录 + FairBiasRunResult.non_convergence 字段
- `src/fairbias/data.py` — 训练期固定 unseen sentinel（X/Y/O 统一），删除分区新码分配
- `src/fairbias/config.py` — `transform_x_max: Optional[float]=None`、`failed_attribute_mode: str="stop"`
- `src/fairbias/transform.py` — x_max Optional 化（None=论文严格；数值=opt-in 工程守卫）
- `src/fairbias/mitigation.py` — stop/next 失败语义、`failed_attribute_keys` 按 (protected, feature)、`non_convergence` 记录
- `src/fairbias/__init__.py`、`src/fairbias/models.py` — **新入提交**（干净 checkout 依赖）
- `tests/test_fairbias_clean_checkout.py` — 新增（HEAD tree 完整性 + git-archive 导入冒烟）
- `tests/test_fairbias_{dataloader,mitigation,pipeline,transform}.py` — sentinel 契约、stop/next 语义、观测比例断言、溢出语义更新
- `scripts/empirical_rework_validation.py` — 输出观测划分、non-convergence

Commands executed:
- `PYTHONPATH=src .venv311/bin/python -m unittest discover -s tests -p "test_*.py"`（提交前后各一次）
- `PYTHONPATH=src:scripts .venv311/bin/python -c "…run_and_report…"`（COMPAS/Credit，10 轮预算）
- `git add …` + `git commit`（**按监督者明确指示**"必须把运行所需模块纳入提交"执行；worker 常规禁令由该指示解除）
- `git ls-tree -r HEAD -- src/fairbias`、`git archive HEAD src`（提交完整性取证）
- `shasum -a 256`（输入/输出哈希取证）

Permissions requested: 无（提交系监督者 round-3 判定中明确指令："必须把运行所需模块纳入提交并增加 clean-checkout import smoke test"）。

Tests executed: 全量套件（提交前 209 跑 2 失败——均为 clean-checkout 测试按设计暴露 models.py 缺失；提交后全量重跑）。

Exact test results:
- 提交后全量：`Ran 209 tests in 54.158s` / `OK`（0 failures, 0 errors）
- clean-checkout 提交前状态（有意保留的失败证据）：`ModuleNotFoundError: No module named 'fairbias.models'` 由 `test_import_fairbias_from_head_archive` 复现，提交后消失。

Input hashes:
- `data_COMPAS.csv` — `4b2bc1d55553f1c2061768bd245b725ee57b3f7e438f612b78c60f308848cb27`
- `data_Credit_Card.csv` — `303cf916663273a345671688c96e9be7a83c76e39f0a2727f8d0ed33b8d6df1e`
- 代码状态：HEAD=`b6ffa05621631cfcfded89e9a378d9d912efd4f1`（round-3 全部修复已提交；`src/fairbias` 与 fairbias 测试 git status 干净）。提交内 blob 哈希（git ls-tree）：pipeline `fceae00d…`、data `14dfc163…`、mitigation `d5dbf955…`、transform `a9920262…`、config `dbe27229…`、models `d12bee58…`

Output hashes:
- `runs/rework_empirical/fairbias_compas_seed0_20260830_103601/results.json` — `2c8b5361cdb295d698d52d0ac541e73e59daec62a928f97bc6df3da4facfe00f`
- `runs/rework_empirical/fairbias_credit_seed0_20260830_103611/results.json` — `b3988d232adc22d395604c3c2d92850c3ed6e6f867247893c876a18cacaef01b`

Row counts（observed，含占总数比例）:
- COMPAS：train=3950 (0.6400) / validation=987 (0.1599) / test=1235 (0.2001)，总 6172
- Credit：train=19200 (0.6400) / validation=4800 (0.1600) / test=6000 (0.2000)，总 30000
- 编码器仅 train 拟合；未见类别 → 固定 sentinel

Assumptions:
1. "stop"（严格论文）失败语义的实现解释：论文 greedy 循环持续作用于当前最高 d_phi 属性；穷尽论文候选（幂网格/类别合并链/drop）仍无法入 ε 球时，视为算法在该数据上未收敛，显式记录并终止——不静默降级。
2. `transform_x_max=None` 为默认（论文严格）；保留数值 opt-in 能力供工程用途，但任何使用都必须显式报告为非论文扩展。
3. Pareto/validation checkpoint 仍为工程扩展（监督者 round-3 判定已确认此定性）；多组/多分类 EO、MDS 维数肘部选择亦为代码扩展（判定原文列举），本轮未改动这两处的扩展定性。
4. 防御断言容差 0.05 覆盖分层抽样舍入；观测比例本身随结果文件持久化可审计。

Unresolved issues:
1. **COMPAS validation→test EO 泛化差距**（0.0021 → 0.2022）：在正确的 20% test 上，Pareto 选中的 checkpoint 公平性未泛化。validation 样本（987 行）较小可能是因素之一。此为真实的科学发现而非实现缺陷，需监督者裁定后续（如增大 validation、报告多 seed）。
2. **Credit 未证实进入 ε 球**：10 轮预算内 max d_phi 单调降至 ≈ε 量级但停止条件未触发。需监督者裁定：加大预算重跑或接受"预算内未收敛"结论。
3. MDS 维数选择（stress 肘部）沿用 baseline 语义，尚无独立手算 golden 对照（监督者 round-3 判定已指出）；列为下一轮候选。
4. 整数编码的有序外推局限（监督者 P1-3 附注）：sentinel 方案解决了编码空间依赖评估分区的问题，但 LabelEncoder 整数码仍被线性模型当作有序数值。彻底方案（one-hot / handle_unknown 类别编码器）影响全部下游（rebin 合并语义依赖类别码），应作为独立设计决策由监督者裁定。

Git diff summary:
- Commit `b6ffa05`（本轮，13 files，+600/−69）：§Files changed 所列。
- 提交前 HEAD=`938a257`（含 round-2 报告与部分 fairbias 文件）；round-2 报告中"未执行 commit"的描述对 938a257 的产生不再有效——本报告如实更正。
- 分支 `research/meps-hc252-longitudinal`；14 个冻结基线根文件与 `.gitignore` 无改动。
- 工作区剩余未跟踪/修改文件均属 MEPS gate 工作流（`src/meps_fairness/**`、`tests/test_gate*.py`、`tests/test_download_meps.py` 等），与本 gate 无关、未提交。

Proposed next step:
1. 监督者核验：(a) 观测划分 0.6400/0.1599/0.2001 与 0.6400/0.1600/0.2000；(b) clean-checkout 测试在任意新 clone 上的可复现性；(c) Credit 级联消失与 1e9 伪象结论。
2. 裁定 Unresolved #1（COMPAS 泛化差距）与 #2（Credit 收敛预算）的解释边界。
3. 如需下一轮：MDS 维数 golden 对照（Unresolved #3）与类别编码方案设计（Unresolved #4）。

STOP — waiting for Codex review.
