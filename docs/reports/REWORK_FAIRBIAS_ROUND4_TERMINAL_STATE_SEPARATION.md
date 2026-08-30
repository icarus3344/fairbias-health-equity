# FairBias 论文对齐返工 Round 4：终态分离与终止语义修复报告

Gate: REWORK round 4 — repair of the P0 (terminal-state rollback) and P1 (termination semantics, power-grid wording, MDS golden contrast, dual-state reporting) defects cited in the Codex verdict on round 3 (`docs/reports/REWORK_FAIRBIAS_ROUND3_REPAIR_FIXES.md`)
Status: COMPLETE — `paper_strict` 与 `pareto_engineering` 两个终态已显式分离、分别保存/评估/命名；`converged` + 完整 `termination_reason` 已记录（Credit 预算耗尽不再显示为无未收敛）；幂网格已改称"配置预算"并以 `candidate_grid_exhausted` 记录穷尽；MDS 维数选择已建立官方仓库 golden 对照（记录到一处偏离）；实证按双终态分别重跑。全量套件 217/217 OK。worker 未执行任何 git commit。

论文源：`/Users/lkc/Downloads/dda3020/icomputing.0083 copy.pdf`（Tang, Lu & Li 2024, Intell. Comput. 2024;3:Article 0083）+ 官方代码仓库 `github.com/zftang/MachineClassifer_BiasMitigation_beta`（论文 Data Availability 声明指定的公开仓库）。

---

## 1. 监督者判定逐项修复映射

| # | 监督者判定 | 修复 | 证据 |
|---|---|---|---|
| P0-1 | 默认管线返回 Pareto 选出的中间状态（COMPAS 回滚到 iter 2、Credit 回滚到 iter 1），非论文贪心算法终止状态 | [pipeline.py](../../src/fairbias/pipeline.py)：results.json 不再有单一 `final_results` 键；改为 `final_results_paper_strict`（贪心终止状态 = 最后一次被接受的变换，无 validation 回滚）与 `final_results_pareto_engineering`（validation Pareto checkpoint，显式标注 ENGINEERING 扩展）两个独立键，各自在 test 上单次评估、各自携带 changed_dict 与说明。`FairBiasRunResult` 拆分为 `paper_strict_metrics` / `pareto_engineering_metrics` 等字段 | 新测试 `test_paper_strict_state_is_greedy_terminal_state`：断言 paper_strict 的 changed_dict **等于最后一个被接受迭代的 changed_dict**、`best_iteration <= terminal_iteration`（Pareto 只能回滚不能超前）；`test_pipeline_compas_end_to_end` 断言 `final_results` 合并键**不再存在** |
| P1-2 | 迭代预算耗尽未记录为未收敛（Credit `mitigation_non_convergence=null` 却打印 none） | pipeline.py 新增顶层 `termination` 记录：`converged`（bool；mitigation 关闭时 null）、`termination_reason` ∈ {`epsilon_reached`, `candidate_grid_exhausted`, `iteration_budget_exhausted`, `mitigation_disabled`, `accuracy_threshold_reached`}、`terminal_iteration`、`terminal_max_dphi`、`epsilon_threshold`。循环各退出点显式追踪（epsilon 达标 / 无变换可接受 / 预算耗尽 / 工程精度阈值） | 新测试 `test_iteration_budget_exhausted_is_recorded_as_not_converged`（COMPAS 预算=3：`converged=False`、`reason=iteration_budget_exhausted`、terminal 状态仍在 ε 球外）、`test_epsilon_reached_is_recorded_as_converged`、`test_mitigation_disabled_termination_reason`（reason=`mitigation_disabled`，converged=null，terminal_iteration=0）。实证：Credit 现记录 `converged=False, reason=iteration_budget_exhausted, terminal_iteration=10, terminal_max_dphi=0.0002168089 > ε=0.0002` |
| P1-3 | 六值幂网格 (1/7,1/5,1/3,3,5,7) 被称作"穷尽论文候选"，但论文正文用 "e.g." 且无限定上界；须核实 Supplementary Algorithm 1 | **核实结果（如实报告）**：Supplementary Materials 无法获取——出版方页面（spj.science.org DOI 10.34133/icomputing.0083 及 suppl_file 直链）返回 HTTP 403，官方仓库不含补充材料 PDF。取得的最强替代证据是**官方代码仓库**主文件 `MachineClassifer_BiasMitigation.py` 第 198 行：`polynomial_arr = np.array([[i, 1/i] for i in range(3,2000,2)]).reshape(-1)`，即交错流 `[3, 1/3, 5, 1/5, …, 1999, 1/1999]`（对同一属性按 polynomial_i 递增搜索）。据此：config.py / mitigation.py 全部"Paper power grid / exhaustive"措辞更正为"**CONFIGURED candidate grid（有限实现预算）**"；网格穷尽失败记 `termination_reason="candidate_grid_exhausted"`，`mitigation_non_convergence` 增加 `search_scope="configured_grid"` 与 reason 字段，明确"非论文层面不收敛主张"。**是否将默认网格切换为官方交错流留待监督者裁定**（见 Unresolved #1） | config.py/mitigation.py 注释与字段；`test_stop_mode_records_non_convergence_on_highest_attribute` 通过（新字段不破坏原断言）；本报告 §2 证据引文 |
| P1-4 | MDS 维数选择（stress 肘部）无 supplementary/golden 对照 | 新增 `tests/test_fairbias_mds_dimension_golden.py` + `tests/fixtures/official_uciadult/distance_matrix_step_0.csv`（官方仓库发布的 UCI Adult step-0 距离矩阵，13 特征 + origin 节点 = 14×14；按协议原子下载 + SHA-256 校验 + `PROVENANCE.json` manifest）。**对照结果（记录性偏离，非对齐声明）**：官方实现仅将 stress 肘部图（dim 1..8）用于目视检查，实际嵌入**固定 dim=2**（源码注释 "here the MDS dimensition is fixed at 2"）；我们的 `_find_optimal_mds_components` 在同一矩阵上自动选择 **dim=3**（归一化 stress 在 2→3 处首次低于 0.01 斜率阈值，实现返回曲线变平处的维度）。该偏离由 `test_recorded_deviation_from_official_fixed_dimension` 显式钉住（断言二者**不等**，若将来对齐须同步更新记录） | 4 个新测试全部通过；`PROVENANCE.json` 记录 URL/日期/大小/SHA-256 与偏离说明 |
| P1-5 | 报告表述过宽："所有未见类别统一映射 sentinel"（实际仅非数值类别经 LabelEncoder 拟合；Credit 数值编码类别原样通过）；"真实的科学发现"过强 | 本报告更正表述：(a) sentinel 仅适用于**非数值类别列**（data.py 只对非数值列拟合 LabelEncoder）；Credit 的 EDUCATION/MARRIAGE/PAY_* 等数值编码类别不经 sentinel，test 中 train 未见过的数值码（如当前 seed 的 PAY_4=1）原样通过；整数码被 LR 当作有序数值的深层问题仍开放（round-3 Unresolved #4 不变）。(b) COMPAS validation→test EO 差距改述为：**单 seed、单划分、工程 Pareto checkpoint 下观察到的内部泛化差距**；本轮双终态数据进一步显示该差距与 checkpoint 选择相关（paper_strict 状态 test EO=0.0207，Pareto 状态 test EO=0.2022），但同样仅为单 seed 描述性观测，不构成任何科学结论。论文实验的 Optuna 超参优化、validation early stopping、中型数据 10 次 test 汇总均未复现，所有数字均非论文指标复现 | 本报告 §2/§3 |

## 2. 双终态实证重跑（10 轮预算，seed=0；描述性观测，非论文指标复现）

**COMPAS**（观测划分 3950/987/1235 = 0.6400/0.1599/0.2001；ε=0.0020）：
- 贪心轨迹：iter 1 c_charge_degree 合并至单类（显式 `dropped`）；iter 2 score_text 合并；iter 3 priors_count 幂 3；iter 4 race 变换后 **max d_phi=0.0009439 < ε，进入 ε 球**。
- `termination`：`converged=true, reason=epsilon_reached, terminal_iteration=4, terminal_max_dphi=0.0009439`。
- **paper_strict（论文贪心终止状态）**：changed_dict 含全部 4 轮变换（c_charge_degree dropped、score_text {1:0}、priors_count ^3、race {2:3, 0:3}）；test ACC=0.6397，EO=0.0207（式 9 语义）。
- **pareto_engineering（工程扩展）**：validation 选择 iteration 2（val EO=0.0021，val ACC=0.6890 ≥ 0.6749）；test ACC=0.6713，EO=0.2022。
- 监督者指出的"最终状态缺 iter 3/4 变换"已消除：paper_strict 状态完整包含 priors_count 与 race 变换。

**Credit**（观测划分 19200/4800/6000 = 0.6400/0.1600/0.2000；ε=0.0002）：
- 贪心轨迹 10 轮逐属性处理（AGE、MARRIAGE、PAY_2/4、LIMIT_BAL、AGE 复访、PAY_3/5/0/6），全程 0 个 dropped。
- `termination`：`converged=false, reason=iteration_budget_exhausted, terminal_iteration=10, terminal_max_dphi=0.0002168089 > ε=0.0002` —— 监督者指出的"预算耗尽被表示为无未收敛"已修复：不再出现 `non-convergence none` 式的误导输出。
- **paper_strict（预算耗尽终止状态）**：changed_dict 含全部 10 轮 9 属性变换（AGE、MARRIAGE、PAY_2、PAY_4、LIMIT_BAL、PAY_3、PAY_5、PAY_0、PAY_6）；test ACC=0.8133，EO=0.0238。
- **pareto_engineering（工程扩展）**：validation 选择 iteration 1（val EO=0.0078）；test ACC=0.8083，EO=0.0151。
- 监督者指出的"报告展示十轮缓解状态、实际返回只有 AGE^5"的混同已消除：两种状态现在分别保存、分别评估、分别命名。

**表述边界**：以上全部为单 seed、单划分下的描述性观测。COMPAS 的 val→test EO 差距是工程 Pareto checkpoint 下的内部泛化差距观察（且本轮显示其与 checkpoint 选择相关），不是科学发现；Credit 预算耗尽状态如实记为未收敛。论文的 Optuna 超参优化、early stopping、10 次 test 分类汇总未复现。

## 3. 标准报告模板

Gate: REWORK round 4（如上）

Status: COMPLETE（§1 五项判定全部落地；217/217 测试通过；实证按双终态重跑并如实表述边界）

Files changed（相对 `f57fe84`，未提交）:
- `src/fairbias/pipeline.py` — 双终态分离（`final_results_paper_strict` / `final_results_pareto_engineering`，删除合并键 `final_results`）；顶层 `termination` 记录（converged / termination_reason / terminal_iteration / terminal_max_dphi / epsilon_threshold）；`FairBiasRunResult` 字段拆分；bool 序列化修正（`converged` 序列化为 true/false 而非 1/0）
- `src/fairbias/mitigation.py` — 网格穷尽语义更正为 `candidate_grid_exhausted`；`non_convergence` 增加 `search_scope="configured_grid"` 与 reason 字段；全部"exhaustive/论文网格"注释更正
- `src/fairbias/config.py` — `transform_poly_exponents` 注释更正为"CONFIGURED candidate grid（有限实现预算）"，记录论文 "e.g." 措辞与官方交错流证据
- `scripts/empirical_rework_validation.py` — 输出 termination 记录 + 两种终态分别打印（各自命名）
- `scripts/run_fairbias_benchmark.py` — 同步双终态输出（该文件属上一轮未跟踪文件，本轮一并更新）
- `tests/test_fairbias_pipeline.py` — 更新断言（无合并键、双终态键、termination 字段）+ 新增 4 个终止语义测试
- `tests/test_fairbias_mds_dimension_golden.py` — 新增（官方矩阵 golden 对照，4 个测试）
- `tests/fixtures/official_uciadult/distance_matrix_step_0.csv` + `PROVENANCE.json` — 新增（网络下载 fixture，见 Commands）

Commands executed:
- `PYTHONPATH=src .venv311/bin/python -m unittest discover -s tests -p "test_*.py"`（最终：`Ran 217 tests in 57.286s` / `OK`）
- `PYTHONPATH=src .venv311/bin/python -m unittest tests.test_fairbias_pipeline tests.test_fairbias_mitigation tests.test_fairbias_mds_dimension_golden -v`（定向验证）
- `PYTHONPATH=src:scripts .venv311/bin/python -c "…run_and_report…"`（COMPAS/Credit，10 轮预算；共执行两遍——第一遍产物 110004/110014 因 bool 序列化修正作废，最终产物 110302/110313；runs 目录从不覆盖）
- 论文核实：本地 PDF 文本提取（pdftotext）；`spj.science.org` 文章页与 suppl_file 直链（均 HTTP 403，未获补充材料）；官方 GitHub 仓库 `MachineClassifer_BiasMitigation.py`、`params.py`、`Result/UCIAdult/`（取得幂搜索流与 MDS 固定维数的官方实现证据）
- fixture 下载：`curl -sL -o <target>.part https://raw.githubusercontent.com/zftang/MachineClassifer_BiasMitigation_beta/main/Result/UCIAdult/distance_matrix_step_0.csv` + SHA-256 校验后改名（原子下载协议）
- `shasum -a 256`（输入/输出/fixture 哈希取证）

Permissions requested: 网络 HTTPS 访问（spj.science.org 与 raw.githubusercontent.com）——用于监督者判定的第 3、4 项"核实 Supplementary / golden 对照"。出版方 403 未获补充材料；官方仓库（论文 Data Availability 指定）为替代证据源，下载仅限单个 4261 字节结果 CSV fixture，含完整 provenance manifest。

Tests executed: 全量套件（217 项）两遍 + 定向测试。

Exact test results:
- 最终全量：`Ran 217 tests in 57.286s` / `OK`（0 failures, 0 errors；round-3 为 209 项，净增 8：4 个终止语义测试 + 4 个 MDS golden 测试）

Input hashes:
- `data_COMPAS.csv` — `4b2bc1d55553f1c2061768bd245b725ee57b3f7e438f612b78c60f308848cb27`（与 round-3 一致，未变）
- `data_Credit_Card.csv` — `303cf916663273a345671688c96e9be7a83c76e39f0a2727f8d0ed33b8d6df1e`（与 round-3 一致，未变）
- fixture `tests/fixtures/official_uciadult/distance_matrix_step_0.csv` — `5f9fd4f81766ce724bd9866bd139503d252f68c9bc7e5065ff34ace0f57228e1`（4261 字节，URL/日期见 PROVENANCE.json）
- 代码状态：HEAD=`f57fe84`（round-3 报告提交）；本轮全部改动在工作区未提交（fairbias 相关文件 git status：5 个修改 + 3 个新增路径）

Output hashes:
- `runs/rework_empirical/fairbias_compas_seed0_20260830_110302/results.json` — `4ab6d2a40217a1d8e7b631b7f6db06a947be4b3a6b5c312c21cef4590eb01cc2`
- `runs/rework_empirical/fairbias_credit_seed0_20260830_110313/results.json` — `57f66e6b6c65e550490aaf9e7c055d24db95df1aaa6fe28c604a6d0954792742`
- （作废中间产物：`fairbias_compas_seed0_20260830_110004` / `fairbias_credit_seed0_20260830_110014`，内容与最终产物仅差 `converged` 的 JSON 布尔序列化与时间戳）

Row counts（observed）:
- COMPAS：train=3950 (0.6400) / validation=987 (0.1599) / test=1235 (0.2001)，总 6172
- Credit：train=19200 (0.6400) / validation=4800 (0.1600) / test=6000 (0.2000)，总 30000
- 编码器仅 train 拟合；**sentinel 仅覆盖非数值类别列**（更正 round-3 的过宽表述）

Assumptions:
1. paper_strict 终态 = 贪心循环最后一次**被接受**变换的状态（无变换被接受时为 iteration 0 原始数据）；预算耗尽时该状态即预算终止状态，不回溯、不外推。
2. 每个报告终态在 test 上各评估**一次**（共两次）：两个状态互不参与对方的选择；test 仍不参与任何模型/变换选择。"单次锁定评估"原则按状态粒度维持。
3. `converged` 仅在 `use_bias_mitigation=True` 时有意义（否则 null）；`accuracy_threshold_reached`（默认关闭的工程增强停止）作为扩展枚举值如实记录。
4. 官方仓库代码是论文 Data Availability 指定的公开实现，作为补充材料不可得时的最强实现证据；六值网格是否切换为官方交错流（[3,1/3,5,1/5,…,1999,1/1999]）**未擅改**，属监督者决策。
5. MDS 维数选择的偏离（自动肘部 dim=3 vs 官方固定 dim=2）仅**记录**，未修改肘部规则——修改会改变全部 d_phi 数值并作废既有 golden 公式测试，属独立设计决策。

Unresolved issues:
1. **默认幂网格 vs 官方交错流**：官方实现对同一属性按 polynomial_i 递增搜索交错流至 1/1999。切换可显著提高收敛概率（Credit 差 0.0000168 未入球）但增加每轮计算量，且 Credit 的 ε=0.0002 由自适应阈值产生、量级极紧。请监督者裁定：切换 / 维持六值预算 / 扩展中间档（如至 21）。
2. **MDS 维数选择偏离**：golden 对照已建立（dim=3 vs 官方 dim=2）。是否将肘部规则改为传统肘点（返回 i 而非 i+1）或对齐官方固定 dim=2，请监督者裁定（影响全部 d_phi 数值）。
3. 整数编码类别（EDUCATION/MARRIAGE/PAY_*）被 LR 当作有序数值、数值码无 sentinel 覆盖——沿袭 round-3 Unresolved #4，未在本轮处理。
4. 单 seed 描述性观测的边界已在本轮报告落实；多 seed、Optuna、early stopping、10 次 test 汇总复现仍属论文实验协议差距（如实列为未做，不声称复现）。

Git diff summary:
- 工作区（相对 HEAD=`f57fe84`，**未提交**）：`src/fairbias/{pipeline,mitigation,config}.py`、`scripts/empirical_rework_validation.py`、`tests/test_fairbias_pipeline.py` 修改（约 +359/−62）；新增 `tests/test_fairbias_mds_dimension_golden.py`、`tests/fixtures/official_uciadult/*`；`scripts/run_fairbias_benchmark.py`（上轮未跟踪文件）同步更新。
- 14 个冻结基线根文件与 `.gitignore` 无改动；工作区其余未跟踪/修改文件属 MEPS gate 工作流，与本 gate 无关、未触碰。

Proposed next step:
1. 监督者核验：(a) COMPAS paper_strict changed_dict 含 iter 3/4 变换且 test 单次评估；(b) Credit `converged=false / iteration_budget_exhausted` 与 terminal_max_dphi=0.0002168 > ε；(c) `final_results` 合并键已消失、双终态键存在；(d) golden 测试 fixture 哈希与 PROVENANCE 一致。
2. 裁定 Unresolved #1（幂网格切换）与 #2（MDS 维数规则）。
3. 如需下一轮：多 seed 稳健性运行与类别编码方案设计（Unresolved #3/#4）。

STOP — waiting for Codex review.
