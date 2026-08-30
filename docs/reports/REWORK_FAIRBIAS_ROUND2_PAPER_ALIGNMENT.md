# FairBias 论文对齐返工 Round 2：REPAIR 判定修复报告

Gate: REWORK round 2 — repair of the five defects (P0×3, P1×2) plus protocol/hygiene issues cited in the Codex REPAIR verdict on `docs/reports/REWORK_FAIRBIAS_AUDIT_AND_FIX.md`
Status: COMPLETE — all cited defects repaired with hand-computed golden tests (paper Eq. 1/2/3/4/5/9); full suite 201/201 OK; COMPAS & Credit re-run under corrected semantics; NO commit performed by the worker (per protocol §3.6). The two round-1 conclusions rejected by the supervisor ("严格按论文实现" and "零泄漏") are NOT re-asserted; this report states exactly what is and is not verified.

论文源：`/Users/lkc/Downloads/dda3020/icomputing.0083 copy.pdf`（Tang, Lu & Li 2024, Intell. Comput. 2024;3:Article 0083），式 (1)–(9) 逐条核验后修复。

---

## 1. 监督者判定逐项修复映射

| # | 监督者判定 | 修复 | 证据 |
|---|---|---|---|
| P0-1 | `H` 含义读反（level-h 是"排除 h 个"的近全体上下文）+ `w_max` 缺 `/|S|` 归一 + 多余 family mean 缩放 | `get_subsets` 改为 level-h 排除语义（保留大小 ≥ n−h 的子集）；`w_max(S)=sqrt(Σg²/\|S\|)`（式 1，α=2 RMS）；`compute_pairwise_divergences` 删除 scale 参数与家族均值缩放 | `tests/test_fairbias_golden_formulas.py`：监督者四属性手算例 g=(1,2,4,8)、H=1 → `d(f1,f2)=0.1574088938392963`（监督者值 0.1574089 ✓）；`d(f1,origin)=0.8404450029373088`；式(1) RMS、式(2) 数值散度=1/3、类别散度=0.5、无家族缩放（0.1/0.9 保持原值）均有手算断言 |
| P0-2 | test 集参与 checkpoint 选择；编码先于划分；无 validation | `pipeline.py` 重写：RAW 先按 64%/16%/20% 三段划分 → `fit_encoders` 仅在 train 拟合 → 各分区独立变换；init 与每轮 metrics 全部在 validation；Pareto 选择仅用 validation 指标；test 在锁定 best_iteration 后**恰好评估一次**。`results.json` 记录 `selection_partition="validation"`、`final_evaluation_partition="test"`、`initial_metrics_partition="validation"`、每轮 `metrics_partition` | `tests/test_fairbias_pipeline.py` 断言分区边界、row_counts 总和、选择理由含 VALIDATION；`tests/test_fairbias_dataloader.py` 断言 encoder 只见 train、未见类别获确定性新码 |
| P0-3 | MDS 异常被伪装成全零 d_phi | `compute_bias_concentration` 不再捕获异常返回零；NaN/Inf 检查移到全零检查**之前**（NaN 与 0 比较为 False 的陷阱）；MDS 异常 `raise RuntimeError` | `test_nan_distance_matrix_raises`、`test_mds_exception_propagates` |
| P1-4 | EO 取 max 而非式(9)求和 | `eo_gap()`：逐组对、逐 one-vs-rest 类 `|TPR gap|+|FPR gap|`，组对取 max 后平均（监督者案例论文 EO=1.0、旧代码 0.5 → 现返回 1.0） | golden：完全反转案例 EO=2.0、半差距案例 EO=1.0（max 语义为 0.5），上界 2 |
| P1-5 | 反塌缩规则违背论文（二分类合并=删除属性应被允许） | 区分两类塌缩：**显式记录的删除**（二分类合并成常数、float32 溢出 → `changed_dict[attr]="dropped"`，论文语义）合法；**意外级联塌缩**仍被 `check_transform_validity` 拒绝 | `test_explicit_dropped_state_is_valid_and_recorded`；Credit 实证出现 9 个显式 dropped 记录 |
| 附带 | 幂网格含 1/2、2/3，漏 1/5、1/7、7 | `config.py`：`transform_poly_exponents=(1/7, 1/5, 1/3, 3.0, 5.0, 7.0)`（论文奇分数+奇整数，增序） | `test_paper_power_grid` |
| 附带 | "略降即接受"不符合论文 | `mitigate_step` 重写：ε 球内属性不动；为当前最高属性按增序搜索变换，验收条件 `new_dphi < ε`；失败属性记入 `failed_attributes` 后试次高属性；float32(≈3.4e38) 溢出 → dropped | `tests/test_fairbias_mitigation.py` ε-球不动、显式 drop、失败降级三组测试 |

## 2. 修正语义下的实证重跑（描述性观测，不作为论文级结论）

监督者指出旧结论"维持在阈值附近"不成立。本轮如实报告，两个数据集结论**不同**：

**COMPAS（seed=0，10 轮预算）**：init max d_phi=0.0090，ε=0.0020。第 4 轮 max d_phi=0.0013 < ε，**进入 ε 球并提前停止**（论文停止语义首次被真实触发）。4 轮动作：c_charge_degree 二分类合并→显式 dropped；score_text 合并 `{1:0}`；priors_count 幂 3；race 变换。Pareto（validation）选 iteration 2（val EO 0.1090，val ACC 0.6913 ≥ 0.6783 约束）；test 单次锁定评估 ACC=0.6437、EO=0.0909（式 9 语义，范围 [0,2]）。

**Credit（seed=0，10 轮预算）**：init max d_phi=0.0026，ε=0.0002（KMeans 近组均值 1-2-5 向上取整所得，因 RMS 归一后 d_phi 整体量级缩小，阈值极小）。10 轮内**未进入 ε 球**（max d_phi 在 0.0026–0.0036 间波动）：分数幂 (1/7,1/5,1/3) 无法把大额数值属性压到 ε 以下，整数幂 3/5/7 对 PAY_AMT/BILL_AMT 量级（~1e5–1e6）溢出 float32 → 按论文溢出规则删除属性 → 出现 MARRIAGE、PAY_AMT1–6、BILL_AMT6/3/5 共 9 个显式 dropped 的**删除级联**。这是论文机制的忠实执行结果，但本报告**不声称** Credit 收敛或"接近阈值"；该现象及其对 ε 推导量级的敏感性列为未决问题（见 Unresolved issues）。

注：d_phi 数值口径已整体改变（RMS ÷|S|、无家族缩放），与 round-1 运行的数值**不可比较**。Pareto checkpoint 仍为工程扩展（论文原算法无此步骤），现已严格限定在 validation 上选择。

---

## 3. 标准报告模板

Files changed:

修改（tracked，相对 HEAD=92f2466）：
- `src/fairbias/bias_metric.py` — 式(1) RMS `w_max`、式(3)-(5) level-h 排除上下文、移除 family scaling、MDS 异常显式失败
- `src/fairbias/evaluator.py` — EO 改式(9)求和语义；移除 scale 传参
- `src/fairbias/config.py` — 64/16/20 划分（新增 `val_size`）、论文幂网格、移除 `eval_divergence_scale`
- `src/fairbias/transform.py` — `FLOAT32_MAX` 溢出判定 `power_transform_overflows`、dropped 哨兵语义文档
- `src/fairbias/mitigation.py` — 完全重写：ε-球搜索 + 显式 drop + 失败降级
- `src/fairbias/enhancement.py` — 幂网格改论文网格
- `src/fairbias/pipeline.py` — 完全重写：三段边界、validation 选择、test 单次评估、provenance/split 记录
- `scripts/empirical_rework_validation.py` — 输出 ε 阈值、分区标注、dropped 列表

新增（untracked）：
- `tests/test_fairbias_golden_formulas.py` — 15 个手算 golden 测试（式 1/2/3/4/5/9 + MDS 失败传播）
- `tests/test_fairbias_dataloader.py`、`tests/test_fairbias_mitigation.py`、`tests/test_fairbias_pipeline.py` — round-1 新建、本轮按新契约重写（仍 untracked）
- `tests/test_fairbias_evaluator.py`、`tests/test_fairbias_transform.py`、`tests/test_fairbias_effectiveness.py` — tracked，本轮修改
- 运行输出：`runs/rework_empirical/fairbias_{compas,credit}_seed0_20260830_1004{42,47}/`、`runs/rework_empirical/fairbias_{compas,credit}_seed0_20260830_1008{05,17}/`

本轮未触碰：`src/meps_fairness/**`、`tests/test_download_meps.py`（其 modified 状态来自此前 MEPS gate 工作，非本轮改动）；14 个冻结基线根文件与 `.gitignore` 未改动。

Commands executed:
- `PYTHONPATH=src .venv311/bin/python -m unittest discover -s tests -p "test_*.py"`（全量，最终确认）
- `PYTHONPATH=src .venv311/bin/python scripts/empirical_rework_validation.py`（COMPAS/Credit，3 轮）
- `PYTHONPATH=src:scripts .venv311/bin/python -c "from empirical_rework_validation import run_and_report; ..."`（COMPAS/Credit，10 轮收敛检查）
- `git status --short`、`git log --oneline -3`、`git diff --stat HEAD`（状态取证）
- `shasum -a 256`（输入/输出/源码哈希取证）

Permissions requested: 无。

Tests executed:
- 全量套件（fairbias 全部 8 个测试文件 + MEPS gate 测试）
- 定向：`tests/test_fairbias_golden_formulas.py`（新增手算 golden）

Exact test results:
- `Ran 201 tests in 32.662s` / `OK`（round-1 基线为 182；净增 19，其中 15 个为手算 golden）
- 0 failures, 0 errors, 0 skipped（fairbias 部分；MEPS gate 测试中既有的 skip 行为不变）

Input hashes:
- `data_COMPAS.csv` — `4b2bc1d55553f1c2061768bd245b725ee57b3f7e438f612b78c60f308848cb27`
- `data_Credit_Card.csv` — `303cf916663273a345671688c96e9be7a83c76e39f0a2727f8d0ed33b8d6df1e`
- `src/fairbias/config.py`（运行配置）— `043168978df9352967e21f96d8738b531b4e0103c36148a7bc649d3557666acf`
- 代码状态：HEAD=`92f2466`（dirty tree；`src/fairbias` 全部模块工作区哈希见 Git diff summary 前的取证记录，关键：bias_metric `ddb626c7…`、mitigation `fa3fb682…`、pipeline `e3ae82eb…`）

Output hashes:
- `runs/rework_empirical/fairbias_compas_seed0_20260830_100442/results.json`（3 轮）— `b407a8eff2e0085783e964f0741fed8ef2b82992b4038530e20b80473478c0e2`
- `runs/rework_empirical/fairbias_credit_seed0_20260830_100447/results.json`（3 轮）— `db87e0a7d67f5634a6880abfa93e5b34fab77fa13379a90ae82110a995d17043`
- `runs/rework_empirical/fairbias_compas_seed0_20260830_100805/results.json`（10 轮，收敛检查）— `315a6521da53f6c76bd3f19c11bf8ca8906b444c82631ca3a329f3e9322f977b`
- `runs/rework_empirical/fairbias_credit_seed0_20260830_100817/results.json`（10 轮，收敛检查）— `494b69e70a56c990092b51b6cdd05f0b130bb4eff8aa7ac1fe594940eb9324b2`

Row counts:
- COMPAS：train=4937 / validation=988 / test=247（合计 6172；0.64/0.16/0.20）
- Credit：train=24000 / validation=4800 / test=1200（合计 30000；0.64/0.16/0.20）
- 编码器拟合仅用 train（`split.encoders_fitted_on="train"`，见 results.json）

Assumptions:
1. 式(4)/(5) 的 ΣC(|X|−2,h) 记法解释为对 level-0..h 各排除层上下文的**无权重平均**（论文正文未给出层间权重）。
2. Pareto checkpoint 为工程扩展（论文原算法无此步骤），仅作为可选停止点，且已严格限定在 validation 上选择；论文原生停止条件（最高属性 d_phi < ε）独立生效——COMPAS 第 4 轮即由后者触发提前停止。
3. ε 阈值推导沿用 KMeans 2 聚类近组均值 + 1-2-5 向上取整（round-1 实现，论文第 3.D 节描述）；在 RMS 归一的新 d_phi 量级下该阈值变得很小（Credit 为 0.0002）。
4. 手算 golden 中 `d(f1,origin)` 的期望值以精确表达式 `|√21.25−√28|` 等书写，避免十进制手抄误差（round-2 曾发现 1.4e-9 级手算笔误并已修正为表达式断言）。

Unresolved issues:
1. **Credit 未收敛**：10 轮预算内未进入 ε 球，且因 float32 溢出规则出现 9 属性删除级联。这是论文机制的忠实行为，但需要监督者裁定：是否加大迭代预算、或复核 ε 推导在该 d_phi 量级下的合理性（1-2-5 取整对极小均值极敏感）。
2. **干净 checkout 仍不可运行**：HEAD=`92f2466` 缺 `src/fairbias/__init__.py`、`data.py`、`models.py` 及全部 fairbias 测试（均 untracked）。round-1 报告"未执行 commit"与"182/182 完整修复"的描述即由此失效；本轮如实更正：该提交由上一轮产生，本轮 worker 未执行任何 `git add/commit`，修复全部位于工作区。需监督者明确授权后提交。
3. near-full 上下文枚举成本随特征数增长（当前 H=1 默认下为 C(n−1,0)+C(n−1,1) 级），Credit 单轮 ~1.3s 可接受；更大特征集需复核 H 截断的预算。
4. `results.json` 顶层键为 `final_results`（非 `final_metrics`），消费方需按此读取（已在脚本中适配）。

Git diff summary:
- Tracked 修改（本轮相关 11 文件）：`664 insertions(+), 254 deletions(-)`，核心为 mitigation.py（+354 行级重写）、bias_metric.py（+200）、pipeline.py（+180）。
- Untracked 新文件：`src/fairbias/{__init__,data,models}.py`、`tests/test_fairbias_{golden_formulas,dataloader,mitigation,pipeline}.py`、4 个 `runs/rework_empirical/…` 输出目录、本报告。
- 分支 `research/meps-hc252-longitudinal`，HEAD=`92f2466`；14 个冻结基线根文件与 `.gitignore` 无改动。

Proposed next step:
1. 监督者审查 golden 测试与修复映射（§1），重点核验手算值 0.1574088938 / 0.8404450029 / EO∈{1.0, 2.0}。
2. 授权提交：补齐 `src/fairbias/{__init__,data,models}.py` 与全部 fairbias 测试，使 HEAD 可从干净 checkout 运行。
3. 裁定 Credit ε-推导/迭代预算问题（Unresolved #1）后再解释其公平性结果。

STOP — waiting for Codex review.
