# NHIS 有限属性与扩展属性公平审计 v3：独立 Goal 审查备忘录

日期：2026-09-19（Asia/Shanghai）

角色：独立审查 Agent；不承担 Worker 执行或科学决策。

最终目标：形成证据边界清楚、方法可复核、能够提交同行评议的 paper package；本备忘录不承诺期刊录用。

## 1. 独立判定

总判定：**REPAIR REQUIRED；A3 NO-GO；A4 NO-GO；2025 LOCKED；PAPER SUBMISSION NO-GO**。

分阶段判定：

- A0 资产、哈希和已声明的 2024 元数据核验：`PASS_WITH_LIMITS`。该结论只覆盖当前 31 行 atlas 和已绑定资产，不等于“全 NHIS 候选宇宙已经系统盘点”。
- A1：`REPAIR_REQUIRED / SCIENTIFIC_FREEZE_NOT_READY`。原报告把它写为结构已接受是可理解的，但主 Q2 结论算子、`O_train`、seed policy、完整 family identity 和 panel 成员级合同仍未机器化。
- A2：从原来的 `ACCEPTED_FOR_SYNTHETIC_CONTRACTS` 下调为 `REPAIR_REQUIRED`。40 项测试确实通过，但存在测试未覆盖且可复现的语义错误，不能据此批准 A3。
- A3：`NOT_AUTHORIZED`。不得生成 2023 真实性能或读取 2023/2024 结果来补注册表。
- A3.5：尚未开始；只能在合格 A3 后生成 Methods 草稿、O 流程、首图和 claim–evidence 表。
- A4：尚无证据支持开启；必须等 A3.5 人类决定。
- A5/2025：继续 fail closed。当前 release registry 的所有 freeze/approval 字段仍为 false，状态正确。
- A6/投稿：当前只有研究设计和软件合同，没有新的经验结果，不能称为可投稿 paper package。

独立复核证据：

- 重新运行登记的主验收集合：`40 passed, 3 warnings in 3.14s`。
- Python 编译与两个 JSON 严格解析通过。
- `planning_verification_v3.json` 登记的 21 个输入/输出 SHA-256 全部与当前字节一致。
- 当前分支为 `research/nhis-fairbias`；受保护的 14 个根文件与 `.gitignore` 无当前 diff。
- 本审查未读取真实 NHIS 性能、未生成预测、未访问 2025、未训练、未联网、未提交。

## 2. Severity-ranked findings

### [P1] 主论文 estimand Q2 只有文档字段，没有可执行的 scope-level 结论规则

`RESEARCH_PLAN_V3.md:53-55` 把 anchor 与 expanded 的“判断是否改变”定为主结果，`A1_INTERFACE_AND_FAMILY_SPEC_V3.md:105-118` 只列出 `anchor_conclusion`、`expanded_conclusion` 等输出名称。实际模块和测试没有生成这五个字段的函数或断言；`generate_statistical_families` 仅产生逐 contrast 记录（`bias_attribute_audit_v3.py:177-270`）。

因此以下决定尚未预注册：

- scope 内多个 O、组、端点、seed 中，何时把一个方法整体判为 within、exceeds 或 insufficient；
- 出现一个 exceeds 和多个 insufficient 时的优先级；
- anchor 与 expanded 使用不同 family size 时，判断变化如何区分“新增 O 的内容”与“family 扩大导致区间变宽”；
- 不同 O 使用不同 respondent domain 时，scope-level 结论的分母、coverage 和 missingness 摘要如何聚合。

这是论文主轴的核心缺口，不是展示细节。A3 前必须冻结一个机器可执行的 Q2 aggregator，并至少同时报告：新增 O 带来的内容变化、anchor contrasts 在共同最大 family 下的稳定性、以及由 precision/multiplicity 单独造成的状态变化。

### [P1] `p_event` 与阈值后二元决策没有在 FNR/FPR 推断合同中锁死

主 panel 被定义为 `event_probability_p` 并复用冻结阈值（`PRE_FREEZE_REGISTRY_V3_DRAFT.json:169-176`），但 A1 输入合同只要求“预测及输出语义”，没有明确要求为错误率推断生成并绑定阈值后的 `y_hat`（`A1_INTERFACE_AND_FAMILY_SPEC_V3.md:21-30`）。现有 domain 合成测试直接把连续分数传给 `linearized_survey_inference`（`test_bias_attribute_audit_v3.py:213-234`）。被调用实现接受任意 `[0,1]` 向量（`survey_linearization.py:89-99`），然后将其组内均值命名为 TPR/FPR（`survey_linearization.py:162-179`）。对连续 `p_event`，这不是通常定义的 TPR/FPR/FNR。

修复要求：风险质量使用 `p_event`；FNR/FPR/BA 使用由每个模型冻结 threshold policy 产生的二元 `y_hat`。预测 hash、threshold hash、`y_hat` hash、输出语义及 order hash 必须分别绑定，并增加一个能区分“软均值”和“阈值错误率”的合成测试。

### [P1] 方法权衡分类会把“所有组都变差”误报成“目标 O 改善”

`classify_method_tradeoff` 首先检查 `gap_narrowed and any_other_O_worsened`，便返回 `TARGET_O_IMPROVED_OTHER_O_WORSENED`（`bias_attribute_audit_v3.py:491-503`）。因此当弱势组和较好组都恶化、gap 仅因较好组恶化更多而缩小时，只要另一 O 也恶化，就会被标成“目标 O 改善”。这与“不能只看 gap 缩小”的 v3 原则及互斥分类定义（`RESEARCH_PLAN_V3.md:93-110`）冲突。

独立复现实例：`gap_change=-0.03`、弱势组 `-0.02`、较好组 `-0.05`、`any_other_O_worsened=True`，当前返回 `TARGET_O_IMPROVED_OTHER_O_WORSENED`，而非“所有组都受损”。现有参数化测试只覆盖无重叠的四个例子（`test_bias_attribute_audit_v3.py:125-178`）。

修复要求：把连续输出保留为主证据；若使用互斥主类，先判定所有组受损，并且“目标 O 改善”必须要求预声明的目标改善规则，不得由 gap 缩小单独推出。为全部重叠情形增加真值表测试。

### [P1] `O_train`、seed estimand 与 panel 成员级合同没有进入冻结机器对象

计划明确要求每一结果行显式记录 `O_train` 与 `O_audit`（`RESEARCH_PLAN_V3.md:31-34`）。Arm001 的训练保护属性实际上是 `SEX_A`（`src/nhis_fairbias/benchmark/data_contracts.py:65-73`），但 A1 predictions 输入表没有 `O_train` 字段（`A1_INTERFACE_AND_FAMILY_SPEC_V3.md:21-28`），主 panel 各 member 也只登记 model IDs 和字符串 source binding（`PRE_FREEZE_REGISTRY_V3_DRAFT.json:184-244`）。

此外，A1 把 `seed_policy` 写入比较键（`A1_INTERFACE_AND_FAMILY_SPEC_V3.md:44-49`），但 registry 没有冻结方法级 seed-to-model 映射、seed 聚合 estimand 或训练随机性处理。family generator 也没有 year、panel hash、scope、seed policy、`O_train`、family ID/hash（`bias_attribute_audit_v3.py:177-270`）。多 seed 若被当作独立调查样本会构成伪重复；若先平均，又需要明确训练随机性是否进入区间。

修复要求：每个 panel member 至少绑定 `O_train`、seed→model ID→model SHA、训练/特征/预处理 hash、输出类型、阈值策略、是否 O-at-inference、预测生成源码 hash。每个结果键显式包含 `O_train` 与 `O_audit`。冻结 seed estimand 后再生成完整 family manifest。

### [P1] 当前 precision 输出把“区间半宽”误命名为 detectable effect，并可能给出误导性充分结论

`precision_diagnostics` 将 `critical_value × standard_error` 直接命名为 `detectable_effect_at_supplied_critical`，并用 `CI half-width <= delta` 定义 `precision_sufficient_for_delta`（`bias_attribute_audit_v3.py:440-469`）。前者是给定 critical value 下的 margin of error，不是带目标 power 的 minimum detectable effect；后者也不能单独回答“能否排除 delta 以上的差异”。例如点估计 0.30、区间 `[0.28,0.32]`、delta 0.05，当前仍返回 `precision_sufficient_for_delta=True`。

修复要求：若不做 power，重命名为 simultaneous margin of error，并以区间相对 `[-delta,+delta]` 的位置回答可排除范围；若报告 MDE，则必须登记 alpha、目标 power、备择方向、family critical value 和复杂调查/复制设计，并通过模拟或解析功效计算得到。

### [P1] 31 行 atlas 不是系统候选宇宙，七个新增字段的选择分母不清楚

当前 atlas 明确由 3 个 anchor、21 个原 X 和 7 个手工扩展组成（`VARIABLE_ATLAS_V3_DRAFT.csv:1-32`）。这比只看原 21 个 X 有进步，但没有记录 2024 Adult Codebook 中所有经资格筛选变量的纳入/排除分母，也没有说明为何是这 7 个而不是其他合格社会处境、临床背景或数据生成变量。若论文把结果写成“扩展审计发现”，审稿人无法区分预先系统筛选与选择性挑选。

修复要求：在读取性能前建立 codebook-wide eligibility ledger，至少记录变量、模块、构念角色、X/Y 邻近性、universe/routing、跨年可用性、嵌套 cluster、纳入/排除及理由；正式候选仍可限制至 30，但 downselect 规则和完整排除分母必须可审计。现有 31 行可作为 seed map，不能称全变量地图。

### [P2] “FROZEN” panel 的验证只证明 JSON 未变，尚未证明成员合同完整或 2023 可比域可行

`freeze_comparison_panel` 仅强制 panel 基本字段以及每个 member 存在 `model_ids` 和 `source_binding`（`bias_attribute_audit_v3.py:522-543`）。registry 静态测试对前三类方法核对 `p_event`，但对 completion 两类只核对 model ID 集合和数量（`test_bias_attribute_audit_v3_registry.py:76-104`）；没有成员级模型 SHA、seed 对位、`O_train`、threshold 或 2023 prediction-valid coverage 断言。

建议把当前状态表述为 `TECHNICALLY_BOUND_PENDING_MEMBER_CONTRACT_AND_COVERAGE_PREFLIGHT`。在不读取结局/性能的前提下，先核验所有五成员能否生成同序 2023 prediction、缺失率和 per-O 可比域；若某成员的非性能可用性会严重缩小主域，应在正式 freeze 前处理，不能等 A3 后再改变 panel。

### [P2] record-order hash 会把不同类型的键折叠成同一字符串

`_record_keys` 在哈希前把任意值转换为 `str`（`bias_attribute_audit_v3.py:62-85`），因此整数 `1` 与字符串 `"1"` 具有相同 order hash。若上游严格冻结“键必须为某一规范字符串 schema”，风险可消除；当前接口没有这个 schema。

修复要求：要么拒绝非字符串键并登记规范化版本，要么用带类型标签和字段边界的 canonical encoding；增加跨类型碰撞测试。

### [P2] simultaneous CI 保持正确的阻塞状态，但目前只验证了字段分离，不是调查设计覆盖率

实现已诚实标为 `A2_CANDIDATE_NOT_PRODUCTION_FROZEN`（`bias_attribute_audit_v3.py:314-385`），这是可接受边界。当前合成测试使用 500 行独立正态 replicate（`test_bias_attribute_audit_v3.py:92-110`），未验证 NHIS stratified PSU resampling 的 replicate scaling、有限设计自由度、NA pattern、强相关 contrasts 或 family coverage。

因此现阶段只能接受“ordinary/simultaneous/Holm 字段没有混用”的软件合同，不能接受同时区间方法本身。A3 前必须与独立参考实现交叉检查，并在多个不平衡、稀有事件、singleton/zero-contribution PSU 和 NA 场景下预先设定 coverage/失败标准。

### [P2] 可复现环境仍有未关闭缺口

独立测试确认主集合通过，但既有 `survey_batch.py:60` 仍产生 divide-by-zero、overflow 和 invalid 三类 warning；可选 catalog tests 因缺少 `aif360` 无法收集。这不否定当前隔离 v3 测试，但 paper package 前必须提供冻结环境清单，并明确哪些 optional 方法不属于主结论。不能在投稿复现说明中把未收集测试写成通过。

## 3. 可接受项

以下设计内容忠实覆盖了用户十项修订，可保留：

- 论文主轴已从“多找 O”转为“有限审计是否高估方法有效性”，并定义 anchor/expanded scope（`RESEARCH_PLAN_V3.md:7-25`）。
- `O_train/O_audit` 分离、逐 O 域、2024 已知回顾性、2025 先冻结后开放、非因果措辞、复杂调查设计、哈希/顺序对齐和 FairBias 不必获胜均被明确保留（`RESEARCH_PLAN_V3.md:29-41`）。
- Q1 `d=0` 与 Q3/Q4 `d_method-d_baseline=0` 在计划和 generator 中分离，确认性与描述性方法矩阵也有边界（`RESEARCH_PLAN_V3.md:43-63`；`bias_attribute_audit_v3.py:177-270`）。
- delta 不允许看结果后选择；三态规则没有把“不显著”当作公平通过（`RESEARCH_PLAN_V3.md:65-91`；`bias_attribute_audit_v3.py:413-437`）。
- Holm p 值、普通 CI 和 simultaneous CI 被分开；最大 gap 的投影拒绝单个事后最大 pair（`bias_attribute_audit_v3.py:273-410`）。
- 固定主 panel 与 supplemental parent-hash 的方向正确；晚加入方法不得反向改变主 panel（`RESEARCH_PLAN_V3.md:120-137`）。
- A3.5 提前写 Methods、O 流程、首图和 claim–evidence 表，以及 A4 非必需的门禁已写入（`RESEARCH_PLAN_V3.md:161-172`）。
- 执行验收和科学/伦理决定已在 decision log 中分开；2025 lock 函数按缺一即拒绝实现（`bias_attribute_audit_v3.py:569-589`）。
- 首篇最小证据包聚焦四块，没有无边界追加第二数据集（`RESEARCH_PLAN_V3.md:20-27`）。

## 4. A3 前必须修复并重新审查

按依赖顺序：

1. 完成系统变量 eligibility ledger、2022/2023 harmonization、正式 O/分组/参照/嵌套 cluster 草案；保持结果盲态。
2. 冻结应用场景与 delta 后，定义机器可执行的 Q2 scope-level 三态 aggregator，并分解新增内容效应与 multiplicity/precision 效应。
3. 修复 `p_event → frozen threshold → y_hat` 合同，保证 FNR/FPR/BA 与风险质量指标不混用。
4. 修复 trade-off 分类和 precision/MDE 语义，增加反例、重叠类、稀有事件与边界测试。
5. 补齐每个 panel member 的 `O_train`、seed/model/hash/threshold/output/feature/preprocessing 合同；先做不读结局的 2023 prediction availability/coverage preflight，再决定正式 panel freeze。
6. 将 family generator 升级为完整 manifest：year/evidence role、scope、panel hash、`O_train`、`O_audit`、seed policy、endpoint、delta rationale、family ID/hash、所有 NA slots 和明确的 confirmatory/descriptive 身份。
7. 修复 record-key canonical schema，并把源、O、Y、prediction、threshold decision 的同序 hash 全部纳入 replay receipt。
8. 对 simultaneous CI 做独立参考/覆盖率 QA；正式方法、replicate scheme、scaling、自由度、有效复制阈值和失败行为写入冻结 registry。
9. Worker 输出新的 A0–A2 repair report 后，由审查 Agent重新运行代码、反例测试、hash、panel/family manifest 和 fail-closed 检查。只有新的独立 `ACCEPT` 才能生成 A3 run specification。

## 5. 需要人类事实的集中清单

这些事项不能由 Worker 或审查 Agent替代决定：

- 真实用途：科研描述、筛查、资源分配或其他；模型输出实际影响谁，FNR/FPR 的现实代价是什么。
- 各确认性端点的 delta、单位、依据、批准人和批准时间；若没有充分领域依据，应把相应结论降为描述性。
- 哪些 O 可以承载公平规范性主张，哪些只能叫临床、社会处境或数据过程切片；参照组与分组合并的领域依据。
- 确认性 method–baseline 对，以及 seed 聚合/训练随机性的科学 estimand。
- 伦理审查或豁免、数据使用权限、2025 一次性 release 责任人、作者资格、AI 使用披露、资助与冲突事实。
- A3.5 后是否开启 A4；A4 不得因为当前结果“不好看”而自动开启。

## 6. GO / NO-GO 范围

当前允许继续（GO）：

- 纯元数据、跨年 codebook harmonization、eligibility ledger 和排除流程表；
- Q2 aggregator、family manifest、panel contract、hash/alignment 和 threshold-semantics 的纯合成修复；
- simultaneous CI 参考实现/覆盖率模拟，不读取真实 NHIS 性能；
- paper Methods/Introduction 的无结果骨架、相关工作创新矩阵、报告模板和 claim vocabulary；
- 独立环境复现与测试缺口修复。

当前禁止（NO-GO）：

- 2023/2024 真实性能扫描、2023 prediction replay 或任何按结果选 O/delta/参照/family 的操作；
- 读取或生成任何 2025 微观结局/性能；
- A4 针对性训练；
- 将 40 个通过测试写成真实调查推断已验证；
- 将当前计划、历史 2024 结果或合成测试写成新的论文发现；
- 对外宣称 paper ready、已验证公平改进、FairBias 获胜或期刊将录用。

## 7. 到可投稿 paper package 的门禁

paper package 至少需要全部满足：

1. 本备忘录 P1/P2 中与 A3 推断有关的修复全部独立验收，A3 获得新的书面授权。
2. A3 完整报告所有候选分母、负结果、不可估、precision、anchor/expanded 判断变化和绝对性能权衡；2024 始终标为已知回顾性复核。
3. A3.5 产出 Methods 草稿、候选 O 流程图、首版主结果图和逐条 claim–evidence 表，并作 A4 GO/NO-GO 决定。
4. 若使用 2025 作验证，必须在全部对象冻结后一次性 release 并完整报告复现、未复现与不可估。若最终不用 2025，则论文必须降格为回顾性/假设生成研究，不能使用独立未来验证措辞。
5. A6 完成主张—证据映射、相关工作/创新矩阵、复杂调查与缺失/域限制、绝对效用权衡、负结果、非因果边界、数据/代码可得性、伦理/作者/AI/资助/冲突声明。
6. 由独立审查 Agent 对冻结 run manifests、源/模型/预测/order hashes、表图数值、Methods 与代码路径进行最终交叉复核。

在上述门禁满足前，本项目是**有潜力形成论文的受控研究计划**，不是已形成可投稿结论的论文。
