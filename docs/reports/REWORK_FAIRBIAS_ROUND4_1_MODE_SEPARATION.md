# FairBias 论文对齐返工 Round 4.1：算法模式分离与提交完整性修复报告

Gate: REWORK round 4.1 — repair of the round-4 REPAIR verdict (four mandatory fixes; MEPS-side P1 study.json arms and P2 hardcoded event count explicitly deferred to the MEPS gates gate per the supervisor's minimal-scope ruling)
Status: COMPLETE — golden fixture 已纳入受控提交范围；`paper_strict` 命名废除，代之以两个显式定义的算法模式（历经两轮更名，REPAIR-2 后定名为 `official_code_derived_monotone_cursor_unweighted`（官方代码衍生·带终止安全扩展的变体：MDS 固定 dim=2、官方交错幂流、单调游标、无迭代预算、无 Pareto 回退，**不作官方代码行为等价声明**）与 `engineering_bounded`（自动维数、六值网格、有限预算、Pareto 扩展））；survey-weighted 扩展以可测试声明钉住；提交边界审计文档补录（Decision 0005）；混合工作区回归套件 237/237 OK（含未提交 MEPS Gates 6–13 测试文件；b9cc9ca 提交自身的独立测试证据为干净导出 `Ran 155 tests / OK (skipped=3)`）；双模式实证重跑完成（official 模式下 Credit 首次进入 ε 球收敛）。clean-checkout 验证基于暂存树（`git write-tree` + `git archive`）通过后方才提交。

> **Round 4.1 Repair 修订（2026-08-30 Codex REPAIR 判定，针对提交 `b9cc9ca`）**：本报告已按修复轮修订——(P0) "官方幂流有限且每属性幂次严格推进 ⇒ 终止有保证" 的说法不成立：原实现每次重访都从幂流头部重新搜索，存在 3→1/3→3 振荡风险；已改为持久化每数值属性的幂流单调游标（只向后推进），并新增回归测试；(P1) "论文方法严格复现" 属过度表述：固定 MDS dim=2 是官方代码行为，论文正文（第 5 页）规定 elbow plot 选维数；模式已更名为 `official_code_unweighted_reimplementation`（官方代码兼容模式），论文正文模式未实现；(P2) 237 项测试为混合工作区回归套件，不得作为 b9cc9ca 自身的全量测试证据，提交自身证据为干净导出 155 passed / 3 skipped。修复详情见 [REWORK_FAIRBIAS_ROUND4_1_REPAIR.md](REWORK_FAIRBIAS_ROUND4_1_REPAIR.md)；修复后重跑双模式实证，数字与本报告 §3 一致（无变化）。
>
> **Round 4.1 Repair-2 修订（2026-08-30 Codex 对 Repair 轮报告的 REPAIR 判定）**：(P1) "官方代码兼容/重实现" 仍是过度表述——单调游标是对官方实现（每次重访从头重搜）的显式偏离，游标永久排除先前拒绝的幂次，两份数据轨迹不变不能证明一般行为等价；模式再次更名为 `official_code_derived_monotone_cursor_unweighted`（官方代码衍生·带终止安全扩展的变体，非官方代码行为等价声明）；真正的官方行为参考模式（状态循环检测 + fail closed）留待日后另建。(P1) 类别属性失败曾被误记为幂流穷尽（`search_scope` 无条件写 "official_power_stream"），已按数值/类别分别记录（`official_power_stream` / `categorical_merge_chain`）并合入回归测试；pipeline termination note 改为根据真实 `search_scope` 生成。(P2) 残余 "strict paper" 措辞与旧模式名已清除。修复详情见 [REWORK_FAIRBIAS_ROUND4_1_REPAIR_2.md](REWORK_FAIRBIAS_ROUND4_1_REPAIR_2.md)。

论文源：`/Users/lkc/Downloads/dda3020/icomputing.0083 copy.pdf`（Tang, Lu & Li 2024, Intell. Comput. 2024;3:Article 0083）+ 官方代码仓库 `github.com/zftang/MachineClassifer_BiasMitigation_beta`（论文 Data Availability 声明指定的公开仓库）。

---

## 1. 监督者 REPAIR 判定逐项修复映射

| # | 监督者判定 | 修复 | 证据 |
|---|---|---|---|
| P0-1 | 提交不完整：`distance_matrix_step_0.csv` 未跟踪，干净 checkout 在 fixture 读取时 FileNotFoundError | fixture 已 `git add` 纳入本轮提交（与 PROVENANCE.json 的 SHA-256 一致：`5f9f…28e1`）；新增回归防护测试 `test_golden_fixture_files_are_tracked`（[tests/test_fairbias_clean_checkout.py](../../tests/test_fairbias_clean_checkout.py)）：`tests/fixtures/` 下磁盘上存在的文件必须全部被 Git 跟踪，工作区残留文件掩盖提交缺失这一失败模式被测试钉死；提交前以暂存树（`git write-tree` + `git archive`）在干净目录跑全套测试 | 新测试通过；clean-checkout 套件通过 |
| P1-2 | `paper_strict` 名不副实：自动 dim=3 vs 官方固定 dim=2；六值网格 vs 官方交错流 `[3,1/3,…,1999,1/1999]`；有限预算 | **模式架构**（[src/fairbias/config.py](../../src/fairbias/config.py)）：新增 `algorithm_mode` ∈ {`official_unweighted_reproduction`, `engineering_bounded`}（构造期校验）与 `resolved()` 单点解析；`official_power_stream()` 程序化生成官方交错流（与官方 L198 `np.array([[i,1/i] for i in range(3,2000,2)]).reshape(-1)` 逐元素一致，含顺序）；`mds_fixed_components` 支持固定维数短路（official 强制 2，非 2 显式拒绝）。official 语义（历经两轮更名，REPAIR-2 后定名 `official_code_derived_monotone_cursor_unweighted`，官方代码衍生·带终止安全扩展的变体，非论文正文方法严格复现、亦非官方代码行为等价声明）：MDS dim=2、官方幂流**保持交错顺序**（[mitigation.py](../../src/fairbias/mitigation.py) 新增 `preserve_exponent_order`，官方顺序 ≠ 升序排序——升序 (1/7,1/5,1/3,3,5,7) 与交错 (3,1/3,5,1/5,…) 是实质差异）、**无有限迭代预算**（循环仅由 ε 球或幂流穷尽终止；终止保证由 Round 4.1 Repair 引入的每属性幂流单调游标提供：每个被搜索过的幂次位置只消费一次、只向后推进，重访不会复用或振荡——原报告此处声称的"幂流有限且严格推进 ⇒ 终止有保证"不成立，已被修复；该游标是对官方 restart-from-head 搜索的**显式偏离**，REPAIR-2 已如实定名）、无 Pareto 回退（输出仅 `final_results_official_code_derived_monotone_cursor_unweighted` 单终态键）；同时拒绝 accuracy enhancement 与 failed_attribute_mode='next'（均为具名工程扩展）。engineering 语义 = 原 round-4 行为诚实命名：贪心终态键更名 `final_results_configured_greedy_terminal`（`paper_strict` 键与 `FairBiasRunResult.paper_strict_*` 字段全部移除），Pareto 键保留 | `tests/test_fairbias_algorithm_modes.py`（19 项新测试 + Round 4.1 Repair 新增 3 项单调游标回归测试）：幂流内容/顺序/长度/尾部、官方构造直译对照、resolved 各拒绝分支、固定维数跳过肘部（mock 断言 `_find_optimal_mds_components` 未被调用）、official 单终态且 payload 无任何 pareto 键、`max_iterations=1` 在 official 被忽略（终止原因必非 `iteration_budget_exhausted`）而同参数 engineering 必触发预算耗尽、顺序保留 vs 升序排序、多次重访不复用幂次/不振荡/游标只向后推进 |
| P1-3 | 报告与 Git 状态不一致：HEAD 已是 `4565487` 但无 Codex Accept 记录 | [docs/decisions/0005-round4-commit-boundary-audit.md](../decisions/0005-round4-commit-boundary-audit.md)：记录事实链（round-4 报告以 `f57fe84` 未提交状态发出 → `4565487` 无对应授权记录 → 2026-08-30 Codex REPAIR 审查已覆盖其内容并授权 round-4.1 完成即提交）；确立规则：今后每次 gate 提交必须引用当日 Codex 判定记录 | Decision 0005；本轮提交信息引用该判定 |
| P2-4 | 数值编码类别未处理：类别代码被当连续数值引入虚假顺序 | **列入 Unresolved（本轮范围裁定为最小 fairbias 侧）**，同时在报告中钉住规则：MEPS 接入时必须由变量字典显式指定 numeric/categorical（不得依赖 pandas dtype 自动判断）；fairbias 侧后续将增加强制列角色配置。survey-weighted 扩展预声明见 §2 | 本报告 §3 Unresolved #2 |

范围裁定（监督者 2026-08-30）：MEPS 侧 P1（study.json arm 过度宣称）与 P2（meps pipeline.py 硬编码 136）**不在本轮**，归入下一 gate（MEPS Gates 6–13 整理）。

## 2. survey-weighted 扩展预声明（Gate D 边界，本轮只钉声明不实现）

[src/fairbias/bias_metric.py](../../src/fairbias/bias_metric.py) 模块级常量 `SURVEY_WEIGHTED_EXTENSION_DECLARATION` + `compute_pairwise_divergences` 文档标注扩展点：

- **唯一计划偏离**：Eq. 2 的两个经验统计量替换为加权版本
  \(\hat\mu_{mg} = \frac{\sum_{i:O_i=g} w_i X_{im}}{\sum_{i:O_i=g} w_i}\)，\(\hat p_{mkg} = \frac{\sum_{i:O_i=g} w_i \mathbf{1}(X_{im}=k)}{\sum_{i:O_i=g} w_i}\)；
- 距离矩阵、MDS、属性排名、贪心变换搜索**全部不变**；
- **等权重必须严格退化**为未加权统计量（该退化为扩展实现的必过单元测试）；
- 声明本身由 `TestSurveyWeightedExtensionDeclaration` 钉住（scope、两个估计量、unchanged、degenerate EXACTLY 关键词断言）。

## 3. 双模式实证重跑（seed=0；描述性观测，非论文指标复现）

**COMPAS official 模式**（REPAIR-2 后定名 `official_code_derived_monotone_cursor_unweighted`；ε=0.0020，运行 0.7s）：
- 贪心轨迹 4 轮（c_charge_degree dropped → score_text → priors_count 幂变换 → race），第 4 轮后 max d_phi=0.0009444 < ε；`converged=true, reason=epsilon_reached, terminal_iteration=4`。
- 唯一报告终态（test 单次评估）：ACC=0.6397，EO=0.0207。无 Pareto 状态。
- 注：terminal_max_dphi=0.0009444（dim=2）与 engineering 的 0.0009439（dim=3 自动肘部）不同——同一变换状态下两种 MDS 维数的 d_phi 数值差异，证明固定维数确实生效；两者均 < ε。

**Credit official 模式**（REPAIR-2 后定名 `official_code_derived_monotone_cursor_unweighted`；ε=0.0002，运行 5.5s）——**本轮关键观察**：
- **11 轮后进入 ε 球**：`converged=true, reason=epsilon_reached, terminal_iteration=11, terminal_max_dphi=0.0001659 < ε=0.0002`。round-4 六值网格 + 10 轮预算差 0.0000168 未入球的问题，在官方幂流（无预算）下按监督者预测的方向解决。
- 唯一报告终态（test 单次评估）：ACC=0.8153，EO=0.0142。
- 无护栏中止：official 模式运行时间完全在正常范围（0.7s / 5.5s），未触发任何截断。

**COMPAS engineering_bounded**（10 轮预算，1.9s）：与 round-4 数字一致（4 轮 ε 收敛；configured_greedy_terminal ACC=0.6397/EO=0.0207；Pareto 选 iter 2，test ACC=0.6713/EO=0.2022）——键更名后行为回归一致。

**Credit engineering_bounded**（10 轮预算，9.4s）：与 round-4 数字一致（预算耗尽 `converged=false`，terminal_max_dphi=0.0002168 > ε；configured_greedy_terminal ACC=0.8133/EO=0.0238；Pareto 选 iter 1，test ACC=0.8083/EO=0.0151）。

**表述边界**：以上全部为单 seed、单划分下的描述性观测。official 模式（REPAIR-2 后定名 `official_code_derived_monotone_cursor_unweighted`）是**官方代码衍生的变体**：继承官方代码的公式、贪心控制流与实现裁决细节（如固定 dim=2、交错幂流），但带一处**显式偏离**——幂流单调游标（官方实现每次重访从头重搜，游标永久排除已消费幂次），用于为无预算循环提供终止保证；故**不作官方代码行为等价声明**（两份数据轨迹不变只是经验观测），更不是论文正文方法的严格复现——论文正文（第 5 页）规定以 elbow plot 确定 MDS 最优维数，固定维数 2 属官方代码实现行为；论文正文模式（elbow 维数选择）未在本代码库实现。该模式亦不复现论文实验协议（Optuna 超参优化、validation early stopping、中型数据 10 次 test 汇总）。

## 4. 标准报告模板

Gate: REWORK round 4.1（如上）

Status: COMPLETE（§1 四项判定按范围裁定全部落地；混合工作区回归套件 237/237 通过（含未提交 MEPS 文件，非 b9cc9ca 自身证据；b9cc9ca 自身独立证据为干净导出 `Ran 155 tests / OK (skipped=3)`）；双模式实证完成；clean-checkout 验证基于暂存树通过后提交。2026-08-30 Codex REPAIR 判定后按修复轮修订，见顶部修订说明与 [REWORK_FAIRBIAS_ROUND4_1_REPAIR.md](REWORK_FAIRBIAS_ROUND4_1_REPAIR.md)）

Files changed（相对 `4565487`）:
- `src/fairbias/config.py` — 模式常量与 `official_power_stream()`；`algorithm_mode` / `mds_fixed_components` 字段（构造期校验）；`resolved()` 单点模式解析（official：固定 dim=2、官方幂流、拒绝工程扩展组合）；预设工厂支持 `mode` 别名
- `src/fairbias/bias_metric.py` — `compute_bias_concentration` / `compute_dphi_matrix` 支持 `mds_fixed_components` 固定维数短路；`SURVEY_WEIGHTED_EXTENSION_DECLARATION` 预声明常量与扩展点文档
- `src/fairbias/mitigation.py` — `preserve_exponent_order` 参数（official 保留官方交错顺序，engineering 保留升序排序）
- `src/fairbias/pipeline.py` — 模式分支输出（official 单终态键 / engineering 双终态键，`paper_strict` 键与字段全部移除更名）；official 无迭代预算的 while 循环；termination 记录含 `algorithm_mode` 与模式化 note；payload 含 `algorithm_mode` / `algorithm_mode_definition` / 按模式的 `final_states`
- `src/fairbias/evaluator.py` — 透传 `mds_fixed_components`
- `scripts/run_fairbias_benchmark.py` — `--mode {official,engineering}` 参数与按模式输出
- `scripts/empirical_rework_validation.py` — 双模式重跑支持
- `tests/test_fairbias_pipeline.py` — 键/字段更名断言 + 模式记录断言（语义不变）
- `tests/test_fairbias_algorithm_modes.py` — 新增（19 项模式契约测试）
- `tests/test_fairbias_clean_checkout.py` — 新增 fixture 跟踪回归防护测试
- `tests/fixtures/official_uciadult/distance_matrix_step_0.csv` — **纳入受控提交**（P0 修复本体）
- `docs/decisions/0005-round4-commit-boundary-audit.md` — 新增（提交边界审计）
- `docs/reports/REWORK_FAIRBIAS_ROUND4_1_MODE_SEPARATION.md` — 本报告

Commands executed:
- `PYTHONPATH=src .venv311/bin/python -m unittest discover -s tests -p "test_*.py" -v`（混合工作区回归套件——工作区含未提交 MEPS Gates 6–13 测试文件：`Ran 237 tests in 40.198s` / `OK`；不得作为 b9cc9ca 提交自身的全量测试证据）
- `PYTHONPATH=src .venv311/bin/python -m unittest tests.test_fairbias_pipeline tests.test_fairbias_algorithm_modes tests.test_fairbias_mitigation tests.test_fairbias_mds_dimension_golden tests.test_fairbias_effectiveness tests.test_fairbias_evaluator tests.test_fairbias_dataloader tests.test_fairbias_transform tests.test_fairbias_golden_formulas tests.test_fairbias_clean_checkout`（定向：`Ran 82 tests in 15.907s` / `OK`）
- 双模式实证：`PYTHONPATH=src:scripts .venv311/bin/python -c "…run_and_report…"`（COMPAS/Credit × official/engineering；engineering 预算=10 与 round-4 可比，official 无预算）
- Clean-checkout 验证：`git write-tree`（暂存树）→ `git archive <tree>` 解压至干净目录 → 干净目录执行全量套件（详见 Commands executed 与本报告 Status）
- `shasum -a 256`（输入/输出/fixture 哈希取证）

Permissions requested: 无（本轮无网络访问；fixture 为 round-4 已下载文件的补提交，SHA-256 与既有 PROVENANCE.json 一致）。

Tests executed: 混合工作区回归套件（237 项，含未提交 MEPS Gates 6–13 测试文件）+ fairbias 定向（82 项）+ clean-checkout 暂存树全量套件（155 项，b9cc9ca 提交自身的独立测试证据）。

Exact test results:
- 混合工作区回归套件（含未提交 MEPS Gates 6–13 测试文件）：`Ran 237 tests in 40.198s` / `OK`（round-4 为 217 项，净增 20：19 个模式契约测试 + 1 个 fixture 跟踪防护测试）——该计数混合了未提交的 MEPS 工作区文件，不是 b9cc9ca 提交自身的独立测试证据
- b9cc9ca 提交自身的独立测试证据（干净导出）：`Ran 155 tests` / `OK (skipped=3)`
- 定向：`Ran 82 tests in 15.907s` / `OK`

Input hashes:
- `data_COMPAS.csv` — `4b2bc1d55553f1c2061768bd245b725ee57b3f7e438f612b78c60f308848cb27`（与 round-4 一致）
- `data_Credit_Card.csv` — `303cf916663273a345671688c96e9be7a83c76e39f0a2727f8d0ed33b8d6df1e`（与 round-4 一致）
- fixture `tests/fixtures/official_uciadult/distance_matrix_step_0.csv` — `5f9fd4f81766ce724bd9866bd139503d252f68c9bc7e5065ff34ace0f57228e1`（与 round-4 PROVENANCE.json 一致，4261 字节）
- 代码状态：HEAD=`4565487`（round-4 提交，Decision 0005 追溯确认）；本轮改动在提交前经暂存树 clean-checkout 验证

Output hashes（runs/rework_empirical/，唯一 run_id 不覆盖）:
- official COMPAS `fairbias_compas_seed0_20260830_114606/results.json` — `9c6b31cdecd0ac5f2dafb585b8d258e49a98f0ecae92372674b8f2c55f42f15d`
- official Credit `fairbias_credit_seed0_20260830_114611/results.json` — `c5885f1f1b434197f753ee7d1bbb3cf44a570ffc9c71735b7fed34cf48572e4a`
- engineering COMPAS `fairbias_compas_seed0_20260830_114613/results.json` — `b1e3e029eafa311440719cb25a680406292a2b61a14f14bfb8b9cf0e40e22052`
- engineering Credit `fairbias_credit_seed0_20260830_114623/results.json` — `2222c63fe9d7c90c663bfacf8c857da5d5ae13bcb4617cbd63dacd5d143e1938`

Row counts（observed，与 round-4 一致）:
- COMPAS：train=3950 (0.6400) / validation=987 (0.1599) / test=1235 (0.2001)，总 6172
- Credit：train=19200 (0.6400) / validation=4800 (0.1600) / test=6000 (0.2000)，总 30000

Assumptions:
1. official 模式（REPAIR-2 后定名 `official_code_derived_monotone_cursor_unweighted`）的边界 = 官方代码衍生·带终止安全扩展的变体：继承论文公式 + 贪心控制流 + 官方代码实现行为（含固定 dim=2，这是官方代码行为而非论文正文规定——正文规定 elbow plot 选维数），但带显式偏离——幂流单调游标（官方实现每次重访从头重搜），用于提供无预算循环的终止保证；**不作官方代码行为等价声明**，不是论文正文方法的严格复现，论文正文模式未实现；亦不复现论文实验协议（分类器超参、Optuna、多 seed、10 次 test 汇总）。
2. 官方幂流上限 1999 来自官方代码常量（`range(3,2000,2)`），不揣测论文正文含糊处以外的语义。
3. official 模式迭代无预算的终止保证（Round 4.1 Repair 修订）：由每数值属性的幂流单调游标提供——每个被搜索过的幂次位置（跳过/拒绝/接受均计入）只消费一次、游标只向后推进，同一属性的重访严格向前消耗有限幂流（1998 项），叠加类别合并链受类别数上界约束，故被接受的变换总数有界、循环必终止；ε 球内不接受任何变换（沿袭 round-3/4 语义）。原报告声称的"幂流有限且严格推进 ⇒ 终止有保证"在原实现（每次重访从头重搜）下不成立，已由游标修复并钉上回归测试。
4. engineering 模式行为与 round-4 完全一致（COMPAS/Credit 数字逐一复现），仅键名/字段名/措辞更名——这是回归证据而非新主张。
5. 本轮提交授权来自监督者 2026-08-30 REPAIR 判定随附的明确指示（"修复完成即提交"），Decision 0005 记录在案。

Unresolved issues:
1. **数值编码类别（监督者判定 P2-4，承 round-3 Unresolved #4）**：MEPS 接入时必须由变量字典显式指定 numeric/categorical，不依赖 pandas dtype；fairbias 侧应增加强制列角色配置。归入 MEPS 接入前的前置 gate。
2. **MEPS 侧 P1/P2（本轮明确排除）**：`configs/study.json` 的 `paper_informed_reconstruction` / `survey_weighted_mitigation_extension` arm 需标记 pending 或替换为真实实现名；`src/meps_fairness/pipeline.py` 硬编码 `136` 需改为从 power audit 结果生成并重构为可配置 development-panels 规则。归入 MEPS Gates 6–13 整理 gate。
3. **MEPS Gates 6–13 未提交文件**（约 30+ 路径，含 `src/meps_fairness/`、`tests/test_gate*.py`、报告与配置）：待监督者独立审查后另行提交，本轮未触碰。
4. **survey-weighted 实现（Gate D）**：本轮只钉声明与扩展点；加权 μ̂/p̂ 实现、等权重退化测试、多 panel 权重标准化属 Gate D。
5. 多 seed 稳健性运行仍属论文实验协议差距（如实列为未做，不声称复现）。

Git diff summary:
- 工作区（相对 HEAD=`4565487`）：`src/fairbias/{config,bias_metric,mitigation,pipeline,evaluator}.py`、`scripts/{run_fairbias_benchmark,empirical_rework_validation}.py`、`tests/test_fairbias_pipeline.py`、`tests/test_fairbias_clean_checkout.py` 修改；新增 `tests/test_fairbias_algorithm_modes.py`、`docs/decisions/0005-round4-commit-boundary-audit.md`、本报告；`tests/fixtures/official_uciadult/distance_matrix_step_0.csv` 由未跟踪转为暂存（P0）。
- 14 个冻结基线根文件与 `.gitignore` 零改动（提交前以 `git diff 4565487 -- <files>` 核验为空）。
- MEPS gate 工作流文件（Gates 6–13 未提交内容）未触碰、不纳入本轮提交。

Proposed next step:
1. 监督者核验：(a) 提交中 fixture SHA-256 与 PROVENANCE 一致且 clean-checkout 套件通过；(b) official 模式契约（dim=2、官方幂流顺序、无预算、无 Pareto 键）；(c) engineering 数字与 round-4 逐一一致（仅更名）；(d) Credit official 11 轮 ε 收敛记录。
2. 下一 gate（按监督者最优先顺序 #2）：独立审查并整理 MEPS Gates 6–13 未提交工作流（含 study.json arm 与硬编码 136 两项修复）。
3. 之后再冻结多 panel 统计分析计划 → 核实下载更早 longitudinal panels → 实现 survey-weighted FairBias（Gate D，含等权重退化测试）。

STOP — waiting for Codex review.
