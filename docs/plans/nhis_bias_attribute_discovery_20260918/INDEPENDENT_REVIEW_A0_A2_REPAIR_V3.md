# A0–A2 repair tranche 1 independent review v3

日期：2026-09-19  
角色：独立 Reviewer（非执行 Worker）  
审查对象：`A0_A2_REPAIR_TRANCHE_1_V3`

## 1. 独立判定

总判定：**REPAIR REQUIRED；A2 不能恢复为 `ACCEPTED_FOR_SYNTHETIC_CONTRACTS`；A3 NO-GO；2025 LOCKED。**

Worker 的第一轮修复关闭了 record-key 类型折叠、软概率/二元决策混用、方法权衡重叠和 precision/MDE 误命名等原始反例；registry 也诚实保留为 draft，没有补造未知 hash 或把 panel、A3、2025 错写成已开放。但是，独立对抗性复核又发现四类会让机器合同错误放行的缺口：Q2 只核验了部分 contrast identity、Q2 无法表示“没有冻结新 O”的负结果路径、panel verifier 只验自洽 hash 不验完整合同、family verifier 不绑定顶层 identity/size。现有 51 项测试未覆盖这些反例。

因此，本轮不是对 Worker 工作的全盘否定：软件合同已有实质进步，但尚不足以作为 A3 的 fail-closed 前门。软件修复完成也不会自动授权 A3；科学冻结、人类决定和 production inference 验证仍是独立门禁。

## 2. 审查范围与证据边界

我完整复读并交叉检查了以下当前文件：

- `A0_A2_REPAIR_REPORT_V3.md`
- `src/nhis_fairbias/bias_attribute_audit_v3.py`
- `tests/synthetic/test_bias_attribute_audit_v3.py`
- `tests/synthetic/test_bias_attribute_audit_v3_registry.py`
- `PRE_FREEZE_REGISTRY_V3_DRAFT.json`
- `A1_INTERFACE_AND_FAMILY_SPEC_V3.md`
- `A1_FREEZE_CHECKLIST_V3_DRAFT.json`
- 上一轮 `INDEPENDENT_REVIEW_V3_GOAL.md`

独立重跑：

```text
PYTHONPATH=src pytest -q \
  tests/synthetic/test_bias_attribute_audit_v3.py \
  tests/synthetic/test_bias_attribute_audit_v3_registry.py \
  tests/benchmark/test_survey_domain_contract.py \
  tests/benchmark/test_survey_linearization.py \
  tests/benchmark/test_survey_sufficient_totals.py \
  tests/test_nhis_survey.py

51 passed, 3 warnings in 1.51s
```

三个 warning 均来自既有 `src/nhis_fairbias/benchmark/survey_batch.py:60` 的数值 warning。另行执行的 `py_compile` 与 registry/checklist 严格 JSON 解析通过。Worker 报告列出的六个输出文件 SHA-256 均与当前字节重新计算值一致，registry 的 main-panel canonical binding hash 也独立复算为 `2875bc7c5961f70c95ad02d1d77adfa98e097d322b10e39a50ec45d161dd4b2e`。

本审查没有读取 NHIS 真实性能、2023/2024 结局结果或任何 2025 微观结局/性能；没有生成 prediction replay；没有修改 Worker 文件。全部反例均为内存中的合成对象。

## 3. 原 findings 的逐项复审

| 原问题 | 状态 | 独立证据 | 判定边界 |
|---|---|---|---|
| Q2 scope-level aggregator | **PARTIAL** | 三态优先级、multiplicity/precision 与 new-O 内容变化、逐 O coverage 已实现（模块 762–835）；原测试覆盖域漂移和优先级（测试 131–186）。但新反例表明同一 `contrast_key` 的 `endpoint`/`model_id` 可在 native 与 expanded-family 行间改变仍被接受；空 new-O 集合也被拒绝。 | 主 paper estimand 尚未 fail closed。 |
| `p_event -> threshold -> y_hat` | **CLOSED（隔离软件合同）** | 冻结阈值产生 hash，thresholding receipt 同时绑定 prediction、threshold、`y_hat` 和 record order（模块 125–223）；soft score 作为 FPR/BA 输入被拒绝（226–243）；独立重跑相应测试通过。 | 真实 panel 的阈值、预测和 receipt 仍为 `PENDING`，故 production binding 仍 OPEN。 |
| tradeoff 重叠分类 | **CLOSED** | `ALL_GROUPS_WORSE` 在 gap-improvement 分支之前判断（模块 915–928）；上一轮误分类反例现返回 `ALL_GROUPS_WORSE`，测试包含重叠真值表（测试 288–315）。 | 不等于真实模型已有权衡结果。 |
| precision / detectable effect | **CLOSED（术语与区间语义）** | `[0.28, 0.32]`、delta `0.05` 不再产生“precision sufficient”；MDE 显式为 `None/NOT_COMPUTED...`，`critical x SE` 命名为 margin of error（模块 838–893；测试 220–236）。 | 正式 power/MDE 方法尚未实现；若论文需要 MDE，仍须预先冻结 alpha、power、备择和复杂调查设计方法。 |
| record-key 类型碰撞 | **CLOSED** | 只接受非空唯一字符串键（模块 62–88）；`[1, "1"]` 及 `expected=["1"], observed=[1]` 均 fail closed（测试 37–49）。 | 上游仍须证明 canonical key 的来源与年度唯一性。 |
| `O_train` / seed / panel 成员机器合同 | **PARTIAL** | `freeze_comparison_panel` 已要求 member hashes、seed-to-model binding、阈值与 `O_train`（模块 946–1048）；registry 对未知值保持 `null/PENDING`。但 `verify_frozen_panel` 仅验 `status + self-hash`（1051–1057），可接受手工构造的空 panel；member validation 只要求 `O_at_inference` 是 bool，不要求与主 deployment contract 一致，也未结构化验证 `source_binding`。 | 尚不能把“可生成一个 frozen 对象”当成“任意收到的 frozen 对象已完整验证”。 |
| statistical-family 机器合同 | **PARTIAL** | Q1 与 Q3/Q4 null/family 已分开，family ID/hash 和 NA slots 已实现（模块 308–478）。但 verifier 只重算各 family 子对象（481–510）；修改顶层 `manifest_identity.year` 或 `family_sizes.q1_confirmatory` 后仍通过。 | 顶层 manifest 不是一个封闭、不可矛盾的机器对象。 |
| simultaneous CI 方法 | **OPEN，诚实阻塞** | Holm 与 ordinary/simultaneous CI 字段分离仍正确；spec 明确当前方法只是 A2 candidate，A3 前需参考实现/覆盖率验证。 | 51 tests 不证明 NHIS stratified-PSU 下 family coverage。 |
| codebook-wide eligibility / 候选 O 分母 | **OPEN** | 本 tranche 未完成上一轮要求的 2022/2023 官方 codebook harmonization 与系统 eligibility ledger，Worker 报告也将其列为 unresolved。 | 不能把 31 行 atlas 表述为完整候选宇宙。 |

## 4. 新发现（按严重度排序）

### [P1] Q2 允许以同一 key 比较不同 endpoint 或不同模型

`aggregate_q2_scope_conclusions` 先要求 native 与 expanded-family 的 key 集合相同，但之后仅比较 `O_audit`、`annual_rows`、`domain_rows`（模块 789–796）。我构造如下语义反例：两行共享同一 `contrast_key` 与覆盖数，但 native 行为 `model_id=m1, endpoint=fnr`，expanded-family 行改为 `model_id=m2, endpoint=fpr`。函数没有拒绝并返回 `SUPPORTED_WITHIN_TOLERANCE`。

这会把“换了 estimand”误写成“同一 anchor contrast 在扩大 multiplicity family 后仍稳定”。修复必须绑定完整 scientific identity，至少包括：year、evidence role、panel ID/hash、`O_train`、`O_audit`、group/reference、endpoint、model ID 或 method-baseline pair、seed-policy hash、delta/units/rationale，以及对应的预声明 contrast ID。允许变化的只能是预先声明为 family-dependent 的推断字段（例如 simultaneous interval/conclusion），不能靠一个自由文本 `contrast_key` 代替 identity。

### [P1] Q2 拒绝空 new-O 集合，无法表达关键负结果路径

`_validated_scope_rows` 对所有三套输入统一要求至少一行（模块 693–694），所以没有新 O 被正式冻结时，`new_O_under_expanded_family_rows=[]` 直接抛错。v3 计划明确要求：若没有稳定新 O，区分“已经排除重要大效应”和“数据精度不足”。空 new-O 不应被伪造成一条 contrast，也不应使整项 Q2 软件失败。

需要新增一个显式、哈希绑定的 no-new-O 状态，例如记录候选分母、排除/不可估计计数和 precision 解释；此时 expanded audit 可合法等于 anchor audit，并禁止产生“新增 O 改变结论”的肯定主张。如何定义 paper 级结论需在人类审阅的 decision log 中批准，但软件必须能表示该路径。

### [P1] `verify_frozen_panel` 可接受从未经过 freeze schema 的空 panel

独立反例：对 `{"panel_id":"forged","status":"FROZEN","members":[]}` 去掉 hash 后计算 canonical hash，再附为 `panel_sha256`；`verify_frozen_panel` 返回该 hash 而非拒绝。原因是 verifier 仅检查字符串状态与 self-hash（模块 1051–1057），没有重新执行 `freeze_comparison_panel` 的 required fields、至少两成员、baseline、seed、member/hash/threshold/`O_train` 等验证。`validate_supplemental_panel` 又依赖这一弱 verifier。

Verifier 必须在不改变对象的前提下重跑完整 schema/semantic validation，并核对 `panel_sha256`。同时主 panel 应明确要求成员 `O_at_inference == deployment_contract.O_at_inference == false`，逐成员 output/feature/backbone/threshold contract 与 deployment contract 一致，`source_binding` 为可解析且受 source hash 约束的对象，并对 model IDs 的允许重复规则作显式决定与测试。

### [P1] family verifier 可接受互相矛盾的顶层 manifest

在一个合法生成的 manifest 上，把 `family_sizes["q1_confirmatory"]` 改为 `999`，并把 `manifest_identity["year"]` 改为 `2099`，`verify_statistical_family_manifest` 仍返回 success。各 family 子 hash 没变，因此局部 hash 的确自洽；但调用者读取顶层 summary/identity 会得到与冻结 family 不一致的事实。

Verifier 必须检查 schema/version/status、精确 family-name 集合、顶层 `manifest_identity` 与每个 family/slot 的共享 identity 一致、`family_sizes` 与实际 slot/`slot_count_including_not_estimable` 一致，以及所有 contrast IDs/slot states/null hypotheses 属于注册规则。顶层对象本身也应有 manifest hash，或删除无法被验证的重复字段。

### [P2] 51 项测试覆盖了原反例，但没有覆盖 verifier/negative-path 语义

现有测试并非空壳：它们确实覆盖 record-key 跨类型拒绝、Q2 优先级和覆盖漂移、p-event/decision 分离、family slot 篡改、precision 反例、tradeoff 重叠、panel freeze 缺字段、调查域和 2025 lock。缺口在于：

- 没有 Q2 full-identity mismatch 反例；
- 没有 no-new-O 合法负结果路径；
- panel 测试只对 `freeze_comparison_panel` 的产物再 verify，没有直接向 verifier 提交伪造但 self-hash 自洽的对象；
- family tampering 只改受子 hash 保护的 slot，没有改未受保护的顶层 identity/size。

因此 `51 passed` 是有价值的软件证据，但不是当前 A2 的充分证据。修复后应把上述四种反例加入固定回归测试，而非仅在报告中描述。

## 5. Registry、hash 与冻结诚实性

这一部分判定为 **ACCEPTABLE AS DRAFT**：

- 顶层仍是 `DRAFT_NOT_FROZEN`，`real_performance_scan_authorized=false`（registry 3–6）。
- main panel 明确是 `TECHNICALLY_BOUND_PENDING_MEMBER_CONTRACT_AND_COVERAGE_PREFLIGHT`，seed policy 仍为 `HUMAN_DECISION_REQUIRED`（165–176）。
- 三个 parent comparator 的 seed/model/model-SHA 由 registry 静态测试逐项与 admission prediction refs 对照；两个 completion 成员的未知 model SHA 保持 `null` 并标为 `PENDING_MODEL_ARTIFACT_CONTRACT`（305–309、337–341）。全部成员未知 feature/preprocessing/training/prediction-source hashes 和 thresholds 也保持 `null/PENDING`。
- `panel_sha256` 与当前 technical-binding payload 的 canonical hash 一致，但 registry 没有把它等同于 `status=FROZEN`；coverage preflight 明确未运行（360–365）。
- 2025 lock 的所有 freeze/release 条件为 false，`micro_outcome_or_performance_read=false`（467–479）。A3 仍为 `BLOCKED...`，并明确列出未完成项（481–495）。
- checklist 为 38 项、18 complete、20 incomplete；`a2_repair_tranche_1.ready=false`，A3/2025 release flags 均为 false。没有发现 Worker 用测试通过替代独立验收。

因此没有发现伪造未知 hash 或错误冻结。需要注意的是，这份 registry 的“诚实 draft”不能弥补通用 verifier 的 fail-open 缺陷；两者必须分别判断。

## 6. A2 与 A3 门禁

### A2

**当前：REPAIR REQUIRED。不能恢复 `ACCEPTED_FOR_SYNTHETIC_CONTRACTS`。**

恢复 A2 synthetic-contract acceptance 的最低条件是：

1. Q2 对比绑定完整 contrast identity，并新增 identity-mismatch 回归测试；
2. Q2 能诚实表示 no-new-O 的负结果路径，并记录候选分母/precision 边界；
3. panel verifier 对任意输入重新验证完整 schema/semantics，拒绝 self-hash 的伪造空 panel，并锁住主 deployment contract；
4. family verifier 绑定顶层 identity/size/schema，拒绝本审查的两类顶层篡改；
5. 独立 Reviewer 重跑全部相关测试与上述反例并书面接受。

即使以上完成，A2 的软件验收也只能证明隔离合成合同；simultaneous-CI production validity、真实 panel binding/coverage、正式 scientific freeze 仍是 A3 的单独阻塞项。

### A3

**当前：NO-GO。不得运行 2023 真实性能扫描，不得以本轮软件修复为自动授权。**

A3 还至少需要：完整官方跨年 eligibility ledger；人类批准的应用场景与 delta；正式候选 O/分组/参照/嵌套；固定 confirmatory method-baseline pairs；冻结 seed estimand；全部主 panel 成员 artifact/source/feature/preprocessing/threshold/prediction/`y_hat`/order hashes；不读结局的 2023 availability/coverage preflight；simultaneous-CI 参考实现或预设覆盖率 QA；以及新的独立 A0–A2 acceptance。

2024 仍只能按预声明规则作已知结果的回顾性复核。2025 保持锁定，直到所有 O/模型/指标/family/源码与 order contract 冻结、source hashes 验证并有人类 release decision；本备忘录没有也不能提供该 release。

## 7. 人类事实与 paper 目标

仍需人类责任人集中决定：临床/政策应用场景；各 endpoint 的 delta、单位和领域依据；正式 O 与分组/参照/嵌套规则；seed estimand；少量确认性方法对；允许的主张；伦理与作者责任；受限数据权限；A3.5 后是否值得开启非必需 A4。

paper 目标仍合理，但当前只到“可审查的研究基础设施”，不是 paper-ready result。首篇最小证据包仍应围绕：审计覆盖与证据边界、扩展审计的增量发现（包括诚实负结果）、方法评价是否改变、冻结后的验证与绝对性能权衡。A3 合格后必须先做 A3.5 的 Methods 草稿、候选 O 流程表、第一版结果图和 claim–evidence 表，再由人类决定 A4；不得为追求阳性结果或篇幅而无边界增加第二数据集。

## 8. 最终 GO / NO-GO 范围

- 继续做隔离 v3 合同修复、合成反例测试、官方 metadata/codebook harmonization、非性能 source/hash inventory：**GO**。
- 把 registry 当作诚实 draft 继续补全，但不得改成 frozen：**GO**。
- 将 A2 标为 accepted：**NO-GO**。
- 生成 2023 prediction replay、读取真实性能、运行 A3：**NO-GO**。
- 读取 2025 微观结局/性能：**NO-GO / LOCKED**。
- 开启 A4 或声称 paper-ready/submission-ready：**NO-GO**。

结论标签：`INDEPENDENT_REVIEW_COMPLETE__REPAIR_REQUIRED__A2_NOT_ACCEPTED__A3_NO_GO__2025_LOCKED`。
