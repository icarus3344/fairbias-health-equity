# FairBias 论文对齐返工：审计与修复报告

Gate: FairBias rework gate — paper-aligned bias mitigation (P0+P1+P2 full remediation)
Status: COMPLETE — all planned fixes implemented; full test suite 182/182 OK; COMPAS & Credit empirical validation executed with measurable d_phi reduction and a non-zero Pareto checkpoint on Credit.

---

## 1. 审计结论回顾（返工前证据）

审计阶段在 `.venv311` 环境（174/174 测试通过）下实证发现以下核心缺陷：

| 级别 | 缺陷 | 实证证据 |
|---|---|---|
| P0-1 | 数值特征线性缩放 `alpha_O·x+beta_Y` 在 epsilon（标准化均值差）与 min-max 归一化下双重不变 | 变换前后 epsilon 恒为 0.607171；min-max 后组间输入差 2.2e-16（机器噪声级） |
| P0-2 | 缓解引擎无接受准则，候选"恒通过"，同属性反复空转 | COMPAS 5 轮迭代零变化，`changed_dict` 每轮重复选同一特征且数值无任何改变 |
| P0-3 | `get_subsets` / `h_order` 为死代码，epsilon 用 TV/均值差代理而非论文的 Shapley 距离→MDS→d_phi | 代码内 `get_subsets` 无任何调用点 |
| P1-4 | 6 个公平性指标公式伪造：`BNC = fpr_gap/2`；`FDRP=FORP=NPVP=PPVP=(sp+eo)/2` | 源码直查 |
| P1-5 | 类别合并用同步映射覆盖，破坏链式语义：`{3:1}` 之后 `{1:0}` 不会传递到类别 3 | 源码直查 |
| P1-6 | rebin 候选在原始数据上搜索、却在变换后数据上排名，搜索空间与评估空间错位 | 源码直查 |
| P1-7 | NMI 信息损失约束（基线 `PARAMS_MAIN_THRESHOLD_PHI`）完全缺失 | 源码直查 |
| P2 | `Sequence` 未导入导致 `get_type_hints` 抛错；`cate_attrs or self.cate_attrs` 吞掉空列表；`current_alpha` 死分支 | 运行时复现 `NameError` |

---

## 2. 修复清单（逐项证据）

### 2.1 P0-1 / P0-3：新增论文级偏见浓度度量（新建 `src/fairbias/bias_metric.py`，325 行）

- **行为**：完整实现论文伪代码链路——
  1. `compute_pairwise_divergences`：逐组对特征散度。数值 `num-a`（组内 min-max 归一化后组间均值差）；类别 `cat-a`（归一化比例差绝对和 `/K`，K 取组对类别并集大小）；按基线 `mean` 家族缩放。
  2. 子集价值 `v(S) = sqrt(Σ divergence²)`（基线 `d1B` 口径），带缓存。
  3. `compute_shapley_distance_matrix`：特征 + `origin` 节点，`dist(a,b) = mean_S |v(S∪{a}) − v(S∪{b})|`；子集由 `get_subsets(features, h_order)` 生成（k∈[1,H]，H≤3 截断）——死代码正式接入。
  4. `compute_bias_concentration`：`sklearn.manifold.MDS(dissimilarity='precomputed')`，维度由 stress 归一化肘部法选取（`max_components=15`、`slope_threshold=0.01`，与基线一致；显式 `normalized_stress="auto"` 消除 FutureWarning）；坐标平移使 origin→0。
  5. `compute_dphi_matrix`：各特征嵌入点到原点的欧氏距离，即新 `calculate_epsilon` 返回值 `{p_col: {attr: d_phi}}`。
- **参照**：冻结基线 `eval.py` L873-1267、`config.py` L62-65 默认口径。
- **测试**：`tests/test_fairbias_evaluator.py`（d_phi 语义断言）+ `tests/test_fairbias_effectiveness.py`（合成注偏数据上 d_phi 非零且可被变换移动）。

### 2.2 P0-1：数值变换改为多项式幂（`src/fairbias/transform.py`、`enhancement.py`）

- **行为**：数值分支语义改为基线 `poly`：`sign(x)·|x|^p`（`apply_power_transform`），配 `x_max` 溢出守卫；`changed_dict` 数值格式改为 `{"power": p}`；alpha_O/beta_Y 线性缩放代码全部删除。`check_transform_validity` 同步校验 `power`（NaN/Inf/溢出/常数坍缩）。`enhancement.py` 数值分支同样改为幂网格搜索。
- **幂网格**：`cfg.transform_poly_exponents = (1/3, 1/2, 2/3, 3.0, 5.0)`（分数幂覆盖"压低长尾"，奇次幂保持基线 `d4A` 方向；p≠1 才有意义，纯缩放已被实证无效）。
- **测试**：`test_polynomial_power_changes_dphi`——p=1/3 变换后 `|Δd_phi| > 1e-9`（线性缩放反例在该断言下必失败，守护回归）。

### 2.3 P1-5：链式类别合并语义（`src/fairbias/transform.py`）

- **行为**：新增 `compose_category_mapping(existing, new_change)` 传递闭包合成——旧映射值若等于新合并源类别则改写为新目标；`{3:1}` + `{1:0}` ⇒ `{3:0, 1:0}`。`mitigate_step` 构建候选与写回 `changed_dict` 均经此合成。
- **测试**：`tests/test_fairbias_transform.py` 链式合成单测 + `test_fairbias_effectiveness.py` 迭代缓解用例依赖链式语义。

### 2.4 P0-2 / P1-6 / P1-7：缓解引擎三重接受准则（`src/fairbias/mitigation.py` 整体重写，253 行）

- **行为**：`mitigate_step(X, Y, O, nmi_org, changed_dict, current_epsilon, X_search=None)` 候选接受流程：
  1. 结构合法性（`check_transform_validity`）；
  2. **NMI 信息损失闸门**：`phi = (nmi_org[attr] − nmi_new[attr]) / (nmi_org[attr] + 1e-10)`，`phi > phi_threshold` 拒绝（基线 `module_BM.py` L292-300 语义，默认阈值 100 即基线默认不约束、可配置收紧）；
  3. **有效性准则**：候选必须使该特征 d_phi 严格下降（容差 1e-12）才接受（基线 L316 语义）；否则尝试下一个幂指数/下一个类别合并/下一个属性。`failed_attributes` 仅在所有候选均无效时记录，消除"恒通过"空转。
- **搜索空间一致性**：rebin 候选在 `X_search`（当前已变换训练帧）上搜索，与 epsilon 排名空间一致（`pipeline.py` Step A 先 `transform_data` 再传入）。
- **测试**：`test_nmi_gate_rejects_information_destroying_candidate`（`phi_threshold=0` + mock NMI 归零 → 全部拒绝；放宽阈值 → 可接受）、`test_mitigation_strictly_reduces_dphi`、`test_iterated_mitigation_lowers_max_dphi`。

### 2.5 P1-4：公平性指标对齐基线定义（`src/fairbias/evaluator.py`）

- **行为**：`compute_metrics` 从冻结基线 `eval.py` L359-747（基于 `_binary_confusion_counts` 的 `_calculate_bnc` 等 8 个函数）逐一移植为二分类 one-vs-rest 版本：
  - BNC / BPC：score-based，y_prob 按真实类别分组的组间均值差（`evaluate` 现透传 `y_prob`）；
  - CUAE = max(|PPV 差|, |NPV 差|)；FDRP = |FDR 差|；FORP = |FOR 差|；FNRB = |FNR 差|；FPRB = |FPR 差|；NPVP = |NPV 差|；PPVP = |PPV 差|；OAE = |ACC 差|；SP = |PPOS 差|；
  - 聚合模式"跨类别取 max、跨组对取均值"，与基线一致。伪造公式（`BNC = fpr_gap/2`、`FDRP=FORP=NPVP=PPVP=(sp+eo)/2`）全部删除。SP/EO/EOpp/OAE/CUAE 原有正确实现保留。
- **测试**：`tests/test_fairbias_evaluator.py` 更新断言 + 全量套件端到端校验。

### 2.6 P2 修复

- `evaluator.py` 补 `Sequence` 导入（`typing.get_type_hints` 不再抛错，由 `test_get_type_hints_resolves_public_functions` 守护）；
- `cate_attrs or self.cate_attrs` 改为 `is None` 判断，空列表不再被静默回退；
- `mitigation.py` 重写时清除 `current_alpha` 死分支；
- `pd.unique(list)` FutureWarning 修复（改 `pd.Index(...).unique()`）；MDS `normalized_stress="auto"`。

### 2.7 配置与接线（`config.py`、`pipeline.py`）

- `config.py` 新增 7 字段：`mds_max_components=15`、`mds_slope_threshold=0.01`、`phi_threshold=100.0`、`transform_poly_exponents=(1/3, 1/2, 2/3, 3.0, 5.0)`、`eval_divergence_num='num-a'`、`eval_divergence_cat='cat-a'`、`eval_divergence_scale='mean'`。
- `pipeline.py`：`mitigate_step` 传入 `X_search=transformed_X_train`；构造缓解引擎传入 `phi_threshold` / `poly_exponents`；早停条件维持 `max d_phi <= epsilon_threshold`；Pareto checkpoint 逻辑不变。

---

Files changed:
- 新增 `src/fairbias/bias_metric.py`（325 行）
- 重写 `src/fairbias/mitigation.py`（253 行）；新增 `tests/test_fairbias_effectiveness.py`（181 行）
- 修改 `src/fairbias/evaluator.py`（346 行）、`src/fairbias/transform.py`（197 行）、`src/fairbias/config.py`（96 行）、`src/fairbias/pipeline.py`（333 行）、`src/fairbias/enhancement.py`（136 行）
- 修改 `tests/test_fairbias_evaluator.py`、`tests/test_fairbias_transform.py`
- 新增 `scripts/empirical_rework_validation.py`（实证脚本）与本报告
- 冻结基线 14 个根文件：零改动（只读参照 `eval.py`、`module_BM.py`、`module_transform.py`、`config.py`）

Commands executed:
- `PYTHONPATH=src .venv311/bin/python -m unittest tests.test_fairbias_effectiveness -v`
- `PYTHONPATH=src .venv311/bin/python -m unittest discover -s tests -p "test_*.py"`
- `PYTHONPATH=src .venv311/bin/python scripts/empirical_rework_validation.py`（COMPAS 3 轮 + Credit 3 轮）

Permissions requested: 无（全部操作限于工作区内 `src/`、`tests/`、`scripts/`、`docs/reports/`，未越权）

Tests executed:
1. 全量套件：`unittest discover -s tests`（10 个测试文件，含新增有效性回归）
2. 有效性回归 5 项：`test_mitigation_strictly_reduces_dphi`、`test_iterated_mitigation_lowers_max_dphi`、`test_polynomial_power_changes_dphi`、`test_nmi_gate_rejects_information_destroying_candidate`、`test_get_type_hints_resolves_public_functions`
3. 实证：COMPAS / Credit 各一次 `run_fairbias_pipeline`（seed=0，3 轮）

Exact test results:
- 全量：`Ran 182 tests in 85.031s — OK`（182/182 通过，零失败零跳过；输出尾部 `[DOWNLOADED]` 行为 MEPS 下载测试的正常日志）
- 有效性回归单跑：`Ran 5 tests in 3.458s — OK`
- 实证输出（逐轮记录）：

```
===== COMPAS (seed=0, iterations=3) =====
[init] ACC=0.6728 EO=0.1216 max_dphi=2.6874
  init d_phi top3 [sex]: priors_count=2.6874, c_charge_degree=1.0413, score_text=0.7698
[iter 1] attr=priors_count              max_dphi=0.9730 ACC=0.6344 EO=0.1255
[iter 2] attr=days_b_screening_arrest   max_dphi=0.9712 ACC=0.6328 EO=0.1020
[iter 3] attr=juv_other_count           max_dphi=0.9882 ACC=0.6344 EO=0.0938
[pareto] best_iteration=0 (初始 ACC 0.6728 最高，Pareto 约束下 EO 0.1216 仍为可行域最优)
[time] 13.8s

===== Credit (seed=0, iterations=3) =====
[init] ACC=0.8027 EO=0.0032 max_dphi=6.5021
  init d_phi top3 [SEX]: AGE=6.5021, LIMIT_BAL=1.6381, MARRIAGE=1.1350
[iter 1] attr=AGE        max_dphi=4.6418 ACC=0.8030 EO=0.0018
[iter 2] attr=MARRIAGE   max_dphi=4.6435 ACC=0.8027 EO=0.0077
[iter 3] attr=PAY_2      max_dphi=4.6436 ACC=0.8063 EO=0.0078
[pareto] best_iteration=1 (EO 0.0018，ACC 0.8030 >= 0.7927)
[time] 78.5s
```

结论：COMPAS `max d_phi` 2.6874 → 0.9730（iter1 后降幅 63.8%，此后维持在阈值附近）；Credit `max d_phi` 6.5021 → 4.6418（AGE 幂变换），EO 0.0032 → 0.0018，**`best_iteration=1` 非零**，证明迭代产生真实改善且 Pareto checkpoint 可选中非初始状态。

Input hashes: 不适用（本次返工不修改任何数据文件；输入数据集为冻结基线 `data_COMPAS.csv`、`data_Credit_Card.csv`）

Output hashes: 不适用（输出为 `runs/rework_empirical/**/results.json` 运行产物，未纳入版本基线）

Row counts: `data_COMPAS.csv` 6173 行（6172 条记录 + 表头）；`data_Credit_Card.csv` 30001 行（30000 条记录 + 表头）；本次无数据增删

Assumptions:
1. 子集价值口径取基线默认 `d1B`（平方和开根），散度口径 `cat-a`/`num-a`/`mean` 缩放，与冻结基线 `config.py` 默认值一致。
2. 幂指数默认网格含分数幂以覆盖"压低长尾"方向；任何纯缩放（p=1）不可取。
3. `phi_threshold` 默认 100（等效基线默认不约束），闸门逻辑完整可用、可配置收紧。
4. MDS 每轮重算不做近似：COMPAS 10 特征 + origin、Credit 23 特征 + origin，H≤3 时子集规模可控（Credit 3 轮实测 78.5s）。
5. 测试与实证必须使用 `PYTHONPATH=src .venv311/bin/python`（系统 python3.13 缺 pandas/numpy）。
6. 合成注偏数据上单数值特征的 mean 缩放散度恒为 1 属基线同语义行为（家族内相对比值），有效性测试通过加入第二个数值特征验证幂变换可移动 d_phi。

Unresolved issues:
1. COMPAS 上 `best_iteration=0`：缓解使 ACC 从 0.6728 降至 ~0.634（超出 `tau=0.01` 可行域），Pareto 规则正确地回退到初始状态。若需非零选中，可放宽 `accuracy_tolerance_tau` 或启用 accuracy enhancement——属调参决策，未在本次范围内改动。
2. Credit 的 `AGE` 幂变换后 `max d_phi` 仍为 4.64（AGE 组间差异极强，单一幂次无法完全抹平），后续轮次转向次高特征，符合贪心逐特征缓解的论文语义。

Git diff summary:
- `src/fairbias/` 全部为工作区新增/未跟踪文件（仓库基线中不存在该目录），无对已提交代码的修改。
- `git status --porcelain` 中与本次返工相关的条目：`?? src/fairbias/*`、`?? tests/test_fairbias_*`、`?? scripts/empirical_rework_validation.py`、`?? docs/reports/REWORK_FAIRBIAS_AUDIT_AND_FIX.md`。
- 14 个冻结基线根文件 `git status` 干净，零改动。
- 未执行任何 `git add` / `git commit`（遵守协议第 3.6 条，等待监督者批准）。

Proposed next step:
1. Codex 审查本报告与文件差异，独立复跑 `PYTHONPATH=src .venv311/bin/python -m unittest discover -s tests -p "test_*.py"` 与 `scripts/empirical_rework_validation.py` 验证声明边界。
2. 批准后提交（建议单一提交覆盖 `src/fairbias/`、`tests/test_fairbias_*`、`scripts/empirical_rework_validation.py`、本报告）。
3. 后续可选：放宽 `accuracy_tolerance_tau` 或启用 accuracy enhancement，探索 COMPAS 上非零 `best_iteration` 的调参空间。

STOP — waiting for Codex review.
