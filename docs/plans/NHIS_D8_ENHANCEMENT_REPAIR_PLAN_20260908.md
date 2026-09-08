# NHIS D8 Accuracy Enhancement：Gemini 执行方案

日期：2026-09-08。制定者：Codex Supervisor。执行者：Gemini Implementation Worker。

目标：建立可信的候选评估与可解释变换搜索，判断 joint enhancement 是否在相同公平预算与计算预算下改善效用。不得将“Joint 必须胜出”设为验收条件。

**当前唯一放行的执行 gate：D8-R1（源码修复、证据留存、纯合成验证）。D8-R2 至 D8-R5 为后续路线，不是当前执行授权。**

先完整阅读 [AI_EXECUTION_PROTOCOL.md](../AI_EXECUTION_PROTOCOL.md)，再阅读本文件和 [预检指纹](NHIS_D8_ENHANCEMENT_PRECHECK_20260908.json)。本文件是本次 D8-R1 的具体 gate specification。用户授权 Codex 设计本方案并交 Gemini 实施；Gemini 不得自行批准下一 gate、提交或发布结果。

## 1. 已核对的工作区与证据边界

- 工作目录：`/Users/lkc/Downloads/code_v_0_3`。
- 分支：`research/nhis-fairbias`。
- 制定本规范时 HEAD：`b595e59b67d8781a3dca41199487909fdb9ed42d`。
- 保护 tag：`inherited-code-v0.3-baseline-20260828`；commit：`038897e9f751edac6e36445b7706eec5fdb15988`。
- 工作区已有未提交改动：`src/fairbias/enhancement.py`、`src/fairbias/pipeline.py`；已有未跟踪 D8 runner、CLI、测试和 `archive/baseline_v0.3/`。这些是用户已有工作，不得删除、覆盖为 HEAD 版本或擅自提交。
- 预检 JSON 记录 15 个保护文件以及 22 个源码／协议／汇总结果输入的 SHA-256、大小、mtime。此指纹绑定当前待修复版本，不代表代码验收或实验验证。
- 14 个继承文件匹配保护基线；`.gitignore` 相对基线已增加 presentation/build/PDF 忽略项，当前内容与 HEAD 一致。完整差异和 SHA 已写入预检 JSON。

### 已知 `.gitignore` 差异的监督处置

该差异已升级到 Codex Supervisor，作为 `PRE_EXISTING_BASELINE_DRIFT_RECORDED_NOT_RESOLVED` 记录。本 gate 仅允许在保持其当前字节不变的前提下开展新增目录内的源码修复和合成验证；这不追认过去改动、不授权恢复或编辑 `.gitignore`，也不构成完整基线 PASS 或真实数据实验放行。只有与预检完全相同的既有差异可按此处置继续；新增差异或其他保护文件不匹配必须停止并报告。

`docs/decisions/0004-continuous-gemini-batch-implementation.md` 的连续执行授权仅针对当时 MEPS Gates 6–13，不适用于本次 NHIS D8。不要套用其 batch token。

## 2. 对上一轮发现的准确表述

| 编号 | 当前证据 | 本轮任务 |
| --- | --- | --- |
| F1 | AE 将变换后的训练集与原始验证集配对；最终 D8 评估却一致变换 | 修复并用独立 oracle 验证训练／验证变换合同 |
| F2 | `_evaluate_utility` 吞掉异常返回 0；有删除列时可出现列不匹配 | 显式失败，禁止错误分数参与选择 |
| F3 | AE 使用 `epsilon + 0.02`，不是相对 BM 后状态的反弹上限 | 保留具名 legacy 语义供对照；实现显式预算字段及日志，不偷偷替换历史值 |
| F4 | Joint 001 达到本次训练 epsilon；002/003/004 未达到 | 完整输出可行性和停止原因，不把预算用完写成收敛 |
| F5 | D8 使用工程网格、二维 MDS、十轮限制，Canonical 来自 D6 论文参考路径 | 后续增设同配置 BM-only，历史 Canonical 只能作参考 |
| F6 | D8 丢弃权重和 strata／PSU；当前指标未加权 | R1 明确标注；人口推断留待加权 gate |
| F7 | 默认 NMI 相对损失阈值为 100，不能有效防止完整信息损失 | 报告单位与实际行为；不更改既有 BM 默认或论文路径 |
| F8 | AE 类别候选与阳性率基于原始类别，mapping 可能混用 JSON 字符串键和数字键 | 当前类别分区上生成候选，规范化与冲突检查 |
| F9 | 已尝试候选／耗尽特征被永久缓存，其他特征变化后不会重新评估 | 按完整状态记录候选，避免旧状态拒绝污染新状态 |

不能声称已从旧日志证明每个拒绝的原因，因为旧结果没有完整候选轨迹。F1/F2 已有源码与合成复现支持；零接受步不能再被当作公平门禁全部否决的证据。

幂变换无溢出／截断时可逆；AUROC 提升可能来自模型适配，不能写成互信息增加。低 DP Gap、正例数量回升、低几何偏差均不能单独代表全面公平、校准恢复或因果机制成立。

## 3. D8-R1 授权与禁止范围

### 允许

- 在下述文件范围内修复现有新增代码：
  - `src/fairbias/enhancement.py`：候选评估、类别状态、缓存与审计。
  - `src/fairbias/pipeline.py`：仅 AE 接入、BM 后状态刷新及其日志；`use_accuracy_enhancement=False` 的行为必须保持。
  - `src/nhis_fairbias/d8_enhancement_runner.py`、`scripts/run_nhis_enhancement_study.py`：依赖注入、同一评估合同、停止状态和输出保护。
  - `tests/test_fairbias_enhancement.py`、`tests/test_nhis_d8_enhancement.py`：修正过度 mock 和依赖真实数据的测试。
- 可以新增 `src/fairbias/enhancement_contracts.py`、`src/fairbias/enhancement_state.py`、`tests/test_fairbias_enhancement_contracts.py`、`tests/test_nhis_d8_synthetic_contracts.py`，以及本 gate 的 `docs/reports/` 报告。
- 读取源码、已有汇总 JSON/CSV、已有 D6 变换字典；读取 CSV 仅用于既有汇总证据，不需要重新计算个人级结果。
- 使用明确在内存生成的合成数据进行小规模模型拟合、错误注入与集成测试。
- 在唯一的新 `artifacts/nhis_d8_repair/<UTC时间戳>_<随机后缀>/` 中保存源码快照、差异、测试输出和合成审计产物。

### 禁止

- 不读取、训练、抽样或重新评估 NHIS/MEPS 真实微观数据；禁止运行真实 runner 的 `--smoke-test`。当前旧 smoke 测试会加载真实 parquet，并不是合成测试。
- 不调用真实 `NHISStudyAdapter()`，因为其构造器会读取包含 2024 的 parquet。R1 合成 runner 必须显式注入替代 adapter。
- 不读取 2024 个人级特征／标签／预测，不根据旧 2024 结果挑选超参数；已有汇总表只作问题背景。
- 不改变 BM、几何指标、通用变换、通用分类器工厂或已有模式默认：`mitigation.py`、`bias_metric.py`、`transform.py`、`evaluator.py`、`models.py`、`config.py` 本 gate 保持字节不变。需要修改共享模块才能修复时，先给出最小提案和失败证据，由 Codex 扩展 gate。
- 不改变既有 D0–D7 release、旧 D8 结果、保护文件、协议、已有配置或基线；不操作 `archive/baseline_v0.3/`。
- 不安装依赖、不联网、不上传数据、不修改 Git 配置、不 stage/commit/push/tag，不清理或切换工作区。
- 不引入 DAG 学习、Wasserstein、新调查加权目标、group-specific thresholds、神经网络或全面重写。这些会混入独立的研究问题。

## 4. D8-R1 执行步骤

### R1.0 — 先保存待修复版本

1. 核对分支、HEAD、保护 tag 与预检 JSON。对记录的输入重新计算 SHA-256；不一致先报告差异，不假定昨天的代码仍是当前代码。
2. 记录启动时 Git 状态、Python 解释器、已安装依赖版本。不得以安装包解决环境问题。
3. 以独占创建方式新建 artifact 目录。复制本 gate 涉及的现有源码／测试和旧 D8 汇总结果，记录路径、大小、mtime 与哈希；不复制微观数据。单独保存已有 tracked diff 和未跟踪文件清单。
4. 报告分别提供“相对 HEAD 的总差异”和“相对本轮启动快照的新增差异”，避免把用户先前改动归为本轮工作。
5. 列出拟执行测试的输入依赖。先排除所有真实数据入口，再运行测试；不得直接运行不加选择的全量 pytest。

### R1.1 — 建立唯一候选评估合同

评估器输入为原始建模特征、标签、明确的 fit/selection 分区与不可变变换状态。每个状态执行一次：

```text
原始 fit 特征 ──同一 T_state──> fit 表征 ──fit scaler──> fit model
原始 selection 特征 ─同一 T_state─> selection 表征 ─apply scaler/model─> utility
```

具体要求：

- 当前状态、候选状态、最终状态使用同一变换、缩放、分类器参数、正类约定和指标定义。
- 校验索引、列名、列序、删除列集合、有限值及标签长度。不能靠分别排序或自动取交集掩盖分区不一致。
- 每个候选使用独立模型和 scaler，不污染已接受状态；缓存必须绑定数据分区指纹、配置、变换状态和随机种子。
- scaler 仅 fit 在 fit 分区。候选生成中的 NMI、类别阳性率也只能使用 fit 标签。
- D8 必须显式提供 selection 分区。若兼容旧通用 API 的内部切分，应先在原始数据上固定切分，再拟合任何标签相关候选规则；该兼容路径不得用于 NHIS。
- R1 的选择指标保持 AUROC；不静默退回 accuracy。单类 selection、缺失概率、NaN/Inf、拟合失败等返回具名无效状态或抛出可定位异常。基准状态无效时终止该 arm，不继续搜索。
- 可预期的候选变换溢出属于 `candidate_invalid`；非预期程序异常应终止并报告。不要用宽泛 `except Exception: return 0`。
- 完整记录分数提升量。严格 `gain > min_gain`；R1 的兼容对照设 `min_gain=0`，不将其解释为统计显著性，也不在这一步改成新的经验阈值。
- AUPRC 明确记录算法：旧表是梯形 PR 曲线面积，与 average precision 分开命名，不混用。

### R1.2 — 当前状态上的候选与变换映射

- AE 类别阳性率、候选类别对均基于当前 `T_state(X_fit)` 的类别分区。保持“阳性率排序后相邻类别”这个既有提案规则，R1 不新增更复杂的候选策略。
- 从原始类别到当前分组的映射必须无歧义。将 JSON 的字符串键与内存数字键按特征 schema 规范化；不能把所有特征盲目转成 int。`"3"` 与 `3` 指向冲突目标时显式拒绝。
- 合并的是整个当前分组，原来已合并的类别不得因新 mapping 被意外拆开。允许通过回到父状态撤销整个操作；R1 不新增主动拆分候选。
- 规范化输出必须可 JSON round-trip，并保持对观测类别和未见类别的既定行为。真正的类别名称 `"01"` 与 `"1"` 不得因草率整数转换合并。
- AE 不主动恢复 `dropped` 特征，R1 保持这一搜索空间限制并记录 `skipped_dropped`。这不等于宣布原始数据中的信息不可恢复。
- 记录涉及 explicit_missing／structural_not_in_universe 的合并；R1 不自行改变既有允许集合，禁止把此类合并描述为生物学含义。
- 排名与候选耗尽缓存绑定完整变换状态；BM 或其他特征改变后重新生成适用候选。对同一完整状态的同一候选不重复评估。
- 用全局状态轨迹与明确预算保证终止。不要用永久禁用某个幂或特征的方式代替状态循环检查。

### R1.3 — 公平预算、事务状态与终止语义

- 每次 BM 后立即重新计算训练几何指标，再交给 AE；AE 接受后再更新，确保 `current_epsilon` 对应实际状态。
- 在新增合同里分开记录 `final_epsilon`、`legacy_absolute_slack`、`max_step_rebound`、`min_utility_gain`。禁止悄悄把绝对值 0.02 改成比例。
- R1 对旧行为只提供显式具名的 `legacy_epsilon_plus_absolute_slack` 策略，并在结果中打印实际上限。新工程约束政策仅实现可测试的合同，实际参数与研究运行在 R2/R3 冻结后启用。
- 公平门禁启用时，缺保护属性、缺特征维度、空结果、NaN/Inf 或无效预算必须失败。关闭公平门禁的合成测试必须明确标记 `fairness_guard_disabled`，不能输出“完全公平”。
- 合法删除列可以按已定义语义不出现在 active feature 集合中；其他缺失不能当作 0。所有特征删空时禁止正常 LR 评估，显式记为不可用状态。
- 候选深拷贝后评估，只在接受时替换当前变换。拒绝或失败后所有当前状态与已接受模型引用保持不变。
- 有 BM+AE 的一轮必须保存两条独立事件，不再用同一个 `selected_attribute` 覆盖其中一个。
- 新 D8 输出至少包含：`terminal_state`、`terminal_train_max_dphi`、`final_epsilon`、`fairness_feasible`、`termination_reason`、BM/AE 接受步数、候选数、模型拟合数、几何评估数。
- 区分 `epsilon_reached`、`candidate_exhausted`、`budget_exhausted`、`cycle_detected`、`evaluation_failed`。未达 epsilon 必须显式为 false；不能用 accuracy 达标替代公平达标。
- R1 报告 terminal 状态的真实可行性，不新增“最佳可行 checkpoint”选择来改变策略；该选择归 R2。
- 既有 `phi_threshold=100` 仅报告为既有宽松 NMI 安全项。本 gate 不改共享 BM 默认；后续使用新的相对损失字段，范围和单位明确为 [0,1]。

### R1.4 — 候选轨迹与不可覆盖输出

每个候选事件至少包含：

```text
run_id, arm_id, condition, iteration, engine, parent_state_hash,
candidate_state_hash, selected_feature, proposed_transform,
fit_partition_fingerprint, selection_partition_fingerprint,
utility_metric, utility_before, utility_candidate, utility_gain,
train_max_dphi_before, train_max_dphi_candidate,
final_epsilon, effective_candidate_cap, step_rebound,
accepted, rejection_reason, validity_status,
model_fit_count, geometry_eval_count, config_hash
```

- 无效候选的分数填 null，并配具名原因，不能填 0。
- 允许未计算的候选效用为 null（例如公平门禁提前拒绝），但写明计算顺序，不能声称该候选的效用也未改善。
- 日志不含个体行、ID、标签数组或个人预测。错误消息只含必要结构信息。
- CLI 改用唯一 run 目录。给定已存在输出目录必须拒绝；无参数时生成新的唯一目录，不写回 `runs/d8_enhancement_study/`。
- 运行开始即保存配置与输入指纹；结束保存完成／失败状态和输出 SHA。最终 manifest 不需要给自身做循环哈希。
- R1 可用合成数据验证 writer；不运行真实 CLI 进行“快速确认”。

## 5. D8-R1 必须通过的行为验证

验收看行为和证据，不以测试数量作为完成条件。

| 测试组 | 必须覆盖的反例／断言 |
| --- | --- |
| 分区合同 | identity、幂变换、类别合并、删除列四种状态，fit/selection 同变换、同列序；selection 极值不会进入 scaler 拟合统计 |
| 选择分数 | 小型合成真实 LR 的候选分数与独立手写变换/缩放/拟合 oracle 一致；不能所有成功测试都 mock 掉 utility |
| 列删除 | 初始状态已删除一列时仍能合法评估剩余特征候选；旧版错误由回归测试捕获 |
| 显式错误 | scaler/模型异常、单类选择集、缺失概率、NaN/Inf 都不变成有效 0；基准状态失败必须中止 |
| 索引合同 | 打乱标签 index、不同列序、列缺失被准确识别；不能错位算分 |
| mapping | 3→1 后再把当前 1 组并入 0，所有原组成员一起变换；JSON round-trip 一致；冲突键拒绝；真实字符串类别不被误合并 |
| 状态缓存 | 同状态同候选不重复；其他特征变换后原拒绝候选可重新评估；可构造的 BM/AE 循环有界终止 |
| 事务性 | 拒绝、异常、预算中止都不会部分修改当前 changed_dict；已接受状态不会引用可变候选对象 |
| 公平门禁 | cap 边界等号、超过 cap、非有限值、缺保护维度、空结果、合法删除与非法缺维度；BM 后刷新值确实被 AE 使用 |
| 终止 | 所有候选失败、零接受步、达到 epsilon、预算耗尽且不可行、循环终止分别有正确状态 |
| 双事件 | 同一轮 BM 和 AE 各自的操作、前后分数与接受结果均可重建 |
| 输出保护 | 两次默认运行不同目录；显式重复目录被拒绝；模拟中途异常有失败 manifest，旧输出 SHA 不变 |
| 数据隔离 | 合成测试使用 fake adapter 与 fake canonical transform provider；真实 adapter/读 parquet 入口设为触发即失败的哨兵 |
| 向后兼容 | AE=False 的 pipeline 路径及既有论文/官方模式拒绝 AE 的行为不变；相关源码外的冻结文件 SHA 不变 |

先在启动快照版本上运行能复现 F1/F2 的针对性测试，保存预期失败证据，再在修复版本上验证通过。不要为凑红测试人为引入失败；若旧依赖环境无法运行，准确报告。

执行建议：用当前可用解释器、`PYTHONPATH=src`、`PYTHONDONTWRITEBYTECODE=1`，只运行审核过依赖的具名合成测试。已有单元测试可纳入，但 `tests/test_nhis_d8_enhancement.py` 必须先移除真实数据依赖。输出中记录确切命令、通过/失败/跳过数及失败细节。

## 6. R1 交付与 Codex 审查

交付：修复文件、合成测试、启动源码快照、输入／输出 manifest、候选轨迹示例、前后测试证据和 `docs/reports/NHIS_D8_R1_REPAIR_REPORT_<run_id>.md`。

报告必须使用协议 Section 9 的完整标题，不用“全部通过”替代具体证据：

```text
Gate:
Status:
Files changed:
Commands executed:
Permissions requested:
Tests executed:
Exact test results:
Input hashes:
Output hashes:
Row counts:
Assumptions:
Unresolved issues:
Git diff summary:
Proposed next step:
STOP — waiting for Codex review.
```

Status 使用 `IMPLEMENTED_SYNTHETIC_VERIFIED_PENDING_CODEX_REVIEW` 或准确的失败／未完成状态。Row counts 只报告合成规模；真实数据标为未读取。Permissions requested 若没有则填 None，不为已在本 gate 授权的常规操作再次询问用户。

Codex 需要独立确认：F1/F2 已消除、分数 oracle 成立、循环和状态合同有效、测试没有打开真实 parquet、paper/BM 路径未改变、旧结果未覆盖、差异只在白名单内。**只有 Codex 明确 Accept 后，才可能发布下一 gate。**

## 7. 后续路线：现在不执行

### D8-R2 — 统一搜索控制器，仍先纯合成验证

保留 BM/AE 作为候选生成器，新增工程 controller，不改变论文参考 BM。以 `TransformState → proposals → common evaluator → decision → checkpoint` 为结构。

设计要点：

1. 每次候选经过同一个效用／几何评估器。保留 parent state，允许整体回退；第一版只用当前状态和最佳可行 checkpoint，不立即加入大型 beam search。
2. 定义并分别冻结：最终几何预算 epsilon、单步反弹 delta_step、全程效用损失 tau、最小接受增益 min_gain。相对改善基于同一原始特征宇宙的基准，不在跨 21/15 特征 arm 间混比绝对 dphi。
3. 从不可行初始状态搜索时，不要求每一步立刻全局可行；要求按预定策略降低约束违反量，同时守住效用下限。达到可行域后，以可行域内效用改进为主。禁止“所有候选必须已可行”造成搜索无法启动。
4. BM 提案在应用前检查完整模型效用；其内部为同一特征寻找合并链可以保留，但没有进入统一接受器前不能永久删除信息。
5. 最终只从满足 epsilon 和效用预算的 checkpoint 选择。无可行解输出 `NO_FEASIBLE_SOLUTION_FOUND_WITHIN_BUDGET`，另存终态诊断；原始 baseline 若不可行不能冒充公平 fallback。
6. 未达到 min_gain 不等于统计无效；超过 min_gain 不等于显著。固定 split/seed，避免把反复查询选择集所得微小增益解释为发现。
7. 类别候选可在单独扩展中加入按类别质量加权的条件熵损失；保护缺失／结构类别、one-hot、rollback 搜索分别作为独立开关和消融，不能全部一起改变后归功于 joint。
8. 显式处理单变量 NMI 不识别交互／冗余、正幂可逆但几何变化、MDS 全局上下文随删除变化等指标局限。增加独立敏感属性预测探针属于额外诊断，不视作新的公平证明。

数值预算由 R2 的 Codex gate 固定，依据计算试算和开发集设计，不能由 Gemini 根据旧 2024 表格调出好看结果。R2 首先证明控制器正确，不要求在真实数据上优于 joint。

### D8-R3 — 冻结同配置实验，再运行开发数据

在 R1/R2 Accept 后，由 Codex 签发具体运行矩阵、计算上限、输入指纹和数值参数。下表是必须保留的对照结构：

| 条件 | 比较目的 |
| --- | --- |
| baseline | 同一分类器、同一编码的原始特征参照 |
| D6 frozen canonical | 历史方法参照，单独标注其原始搜索与 MDS 配置，不作交替机制的唯一对照 |
| matched BM-only | 使用与其他 matched 条件完全相同的 BM、候选规则与几何设置 |
| matched AE-only | 判断单独的模型适配是否已经解释主要提升 |
| matched posthoc | 从同一个 matched BM-only 终态进入 AE，不从不同算法的 D6 终态进入 |
| matched joint | 同一 BM/AE，只改变固定交替调度 |
| unified constrained search | 检查统一候选决策、回退和可行解选择的贡献 |

编码对照另成区块：旧整数编码供兼容性分析；名义类别 one-hot 的结果在其自己的 baseline/matched 条件内比较，不能跨编码归因。网格、初始化、MDS 维度规则、分类器正则化、缺失处理、epsilon 算法和权重状态均需一致。

预算至少分别记录并限制：候选提案数、完整模型拟合数、几何评估数及总运行时间；相同迭代数不等于相同预算。无须为了耗尽预算重复无用候选。报告实际消耗和预算上限；未用完与耗尽均如实标注。

数据边界：

- 所有拟合与标签相关搜索均限定在 2022 训练数据。需要内部选择时，先在 2022 内按已核实的有效分组边界创建固定的 fit/search 分区；不要擅自假定存在可用家庭 ID。预处理也须遵守内部 fit 边界，不能复用在完整 2022 上拟合的监督选择状态。
- 2023 用作开发期外部验证，不能同时承担无上限候选搜索、校准拟合和无偏效果估计三种角色。具体复用方式须在运行前声明。
- 当前 adapter 的构造器会读取整份 2022–2024 parquet。下一 gate 必须先建立有验证证据的仅开发年份读取接口；不能仅因随后没有使用 test 标签，就声称实现了未打开 test 的隔离。
- 2024 旧测试结果已参与方法讨论；再用同一 2024 评估改进方案只能标为已使用测试上的探索性评估。R3 不默认打开它。新的确认性评价需要独立样本及单独授权。
- 不假设现有 D5 加权结果自动覆盖新 D8。未加权算法消融只能报告样本层面结果，不能称人口结论；完整调查推断在 R5 审查后开展。

每个 arm 输出独立可行性、排序效用、校准诊断、正例率、TPR/FPR/PPV 和公平差距。指标无分母时输出不可估计，不能填 0。不同条件的比较使用同一评估样本；四个 arm 不是四次独立重复实验。

### D8-R4 — 冻结表征后的校准／阈值诊断

保留表征与模型，单独评估：无校准、全局有正斜率的 logit 校准、isotonic；统一阈值根据预先声明的成本或资源目标确定。校准在开发数据内使用独立或交叉拟合的预测，禁止用模型拟合内预测替代校准数据。

报告 AUROC/AUPRC、校准截距/斜率、log-loss/Brier、可靠性图、固定政策下 Recall/PPV/FPR/选择率。严格递增校准保留排序；isotonic 并列分数可改变排名指标；正温度缩放不改变 0.5 决策。这些行为需合成验证。

Group-specific thresholds 是新的部署政策，涉及推断时使用保护属性，本路线不默认启用。正例数回到 baseline 不作为成功标准。

### D8-R5 — 调查加权与不确定性；因果路线另立项

优先复用当前 `bias_metric.py` 的加权群体均值／类别比例接口，并审核 BM 类别提案是否仍用未加权频率。加权几何、加权模型损失、加权候选排序、加权评估是独立开关，先按预注册结构逐一比较，再运行全加权版本。

- 区分样本目标和人口目标；训练权重若归一化，需要冻结其与 LR 正则化强度的约定。
- 使用 NHIS 官方设计变量处理估计不确定性。指标加权不等于有设计调整区间；不直接套独立同分布 DeLong 或逐人 bootstrap。
- 记录组内有效样本量诊断、最大归一化权重及事件数；正确处理小群体、缺失维度、单 PSU／无效设计情况。具体抑制规则在 gate 中事先确定。
- 固定模型的评估区间与重跑完整选择流程的不确定性是不同对象，分别命名；配对比较使用一致的设计重复权重／重采样。
- 权重截断、收缩和 MDS 精度加权改变估计目标，不能为压低 stress 自动启用。
- Wasserstein、新因果 DAG/路径特定约束、工具变量均为独立方法，不作为本轮修复的一部分。先明确预测“因费用推迟医疗”的用途和允许／禁止路径，后讨论可识别性。

## 8. Gemini 的停止条件

出现未记录的输入指纹变化、保护文件新差异、真实数据读取需求、共享模块改动需求、网络／新依赖需求、无法解释的指标差异或必须改变研究目标时，完成不依赖该问题的工作后提交具体阻塞证据，不自行扩大范围。

完成 D8-R1 即提交完整报告并结束本 gate，最后一行必须为：

`STOP — waiting for Codex review.`
