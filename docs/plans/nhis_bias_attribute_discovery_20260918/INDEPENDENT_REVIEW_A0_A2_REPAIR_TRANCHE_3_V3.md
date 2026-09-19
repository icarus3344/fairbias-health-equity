# A0–A2 repair tranche 3 independent review v3

日期：2026-09-19  
角色：独立 Reviewer（非执行 Worker）  
审查对象：`A0_A2_REPAIR_TRANCHE_3_V3`

## 1. 最终判定

总判定：**REPAIR REQUIRED；A2 synthetic-contract acceptance 仍不能恢复；A3 NO-GO；2025 LOCKED。**

Tranche 3 已经关闭上一份独立审查列出的具体缺口：new-O 与 anchor 的 year/panel/model/seed/delta shared-scope identity、每个 O 的 group-rule identity、逐 O panel domain、self-consistent hollow seed policy、完整 threshold provenance、canonical panel/model IDs，以及 no-new-O receipt 的结构化 provenance。93 项完整相关测试、当前 receipt/report/output hashes、main-panel draft hash、历史 receipt supersession 和 2025 lock 均可独立复现。

但是，本轮对 `audit_role`、family manifest 与 Q2 输入集合的关系继续做 fail-closed 检查后，发现两个会直接破坏论文主问题的新 P1：

1. family generator 接收 `AuditAttributeSpec.audit_role`，却不把它写入 slot 或验证 scope-membership；Q2 也不绑定冻结 family/预声明 contrast universe。因此，一个 expanded-only O 可以被合法生成到 `scope=anchor`，同一个 O 也可以同时作为 anchor 和 `new_O`，并被报告为“新增 O 改变结论”。
2. family generator/verifier 没有强制 `O_audit -> O_group_rule_sha256` 一对一。同一 O 的 FNR 与 FPR slots 可使用不同分组规则、重算全部 IDs/hashes 后仍通过 verifier，破坏 joint EO family 与跨 endpoint 解释。

这些不是尚待人类选择的正式 O 内容，而是当前通用机器合同可错误放行的结构性反例。因此，本 memo 不能因为上一轮指定反例已经关闭而恢复 A2。修复仍可全部使用纯合成对象完成；不需要也不得运行真实性能或读取 2025。

结论标签：`INDEPENDENT_REVIEW_TRANCHE_3_COMPLETE__REPAIR_REQUIRED__A2_NOT_ACCEPTED__A3_NO_GO__2025_LOCKED`。

## 2. 审查范围与证据边界

本轮完整检查：

- `A0_A2_REPAIR_TRANCHE_3_REPORT_V3.md`
- `repair_verification_v3.json` 与 superseded `planning_verification_v3.json`
- `src/nhis_fairbias/bias_attribute_audit_v3.py`
- `tests/synthetic/test_bias_attribute_audit_v3.py`
- `tests/synthetic/test_bias_attribute_audit_v3_registry.py`
- `PRE_FREEZE_REGISTRY_V3_DRAFT.json`
- `A1_INTERFACE_AND_FAMILY_SPEC_V3.md`
- `A1_FREEZE_CHECKLIST_V3_DRAFT.json`
- 上一份独立审查 `INDEPENDENT_REVIEW_A0_A2_REPAIR_TRANCHE_2_V3.md`

本审查没有读取 NHIS 真实性能、2023/2024 结局结果或 2025 微观结局/性能，没有生成 prediction replay，没有修改 Worker 文件。新增反例全部是内存合成对象。本文件是唯一新增文件。

## 3. 独立执行、receipt 与 hash 复核

完整相关集合独立重跑：

```text
PYTHONPATH=src pytest -q \
  tests/synthetic/test_bias_attribute_audit_v3.py \
  tests/synthetic/test_bias_attribute_audit_v3_registry.py \
  tests/benchmark/test_survey_domain_contract.py \
  tests/benchmark/test_survey_linearization.py \
  tests/benchmark/test_survey_sufficient_totals.py \
  tests/test_nhis_survey.py

93 passed, 3 warnings in 1.71s
```

三个 warning 与 Worker 报告一致，来自既有 `src/nhis_fairbias/benchmark/survey_batch.py:60` 的 divide-by-zero、overflow 和 invalid matrix multiplication。它们没有被隐藏，但当前 synthetic assertions 均通过；这不构成 production simultaneous-CI 或 survey coverage 接受。

另外独立重跑：

- tranche 3 指定反例相关的 23 个定向实例：`23 passed`；
- deployment/source/model-ID、forged panel 与 family top-level tamper 的 8 个实例：`8 passed`；
- 2025 lock、registry draft gate 与 receipt supersession 的 3 个实例：`3 passed`。

Worker 报告与当前字节逐项一致的 hashes：

- module：`c6903d786bb66e272f5580fff23da852b13c557bd5f8ee5226f57da0a0751273`
- main synthetic test：`773f66c256d8ccd0ce8208057ced1e263c6504ef0e42c7580a2d05595f7f624f`
- registry test：`8cf624d1a4fb33608ccbe5ee046d42b5f86a25923858e455b25f71896e2321cd`
- registry：`f39499904645a634557ac021a789b436b389f7b1e33b3ee54fd260f463a7b945`
- interface spec：`cd07533271ccd39a64e36c0bf96ed3c3fff70403335b051743a2905c1ffee28e`
- checklist：`d5f821deb4e37d7e70dee19d472e969984618cdd08eb52efe60558ba95d8cc6d`
- superseded planning receipt：`6813360d11a1edac8a7edd6756442f24f9c3c2656eaec0a8c754cb90fcf5d313`
- tranche-3 report：`a51b8593a0d992bccc12248e8b93fd7e74c0f62797fe27e507f4a8dd964cb99f`

`repair_verification_v3.json` 当前自身 SHA-256 为 `47f07b8ebbde53658df3529fe29ef038147b967435edfb1f90bf2a264dbbb7e0`；按声明不自哈希。receipt 内登记的全部 output/report hashes 均匹配，没有缺项或错配。registry 的 main-panel draft binding 独立复算为 `2a63adc9e43829af676570553d8dbedc7a9209ca59a832089e77479bfe6a8d0f`，与登记值一致。

历史 `planning_verification_v3.json` 仍明确为 `SUPERSEDED_BY_REPAIR_VERIFICATION_V3`、`historical_snapshot_only=true`；它过去的 `ACCEPTED_FOR_SYNTHETIC_CONTRACTS` 不能作为当前授权。当前 repair receipt 保持 `REPAIR_TRANCHE_3_WORKER_COMPLETE_REVIEW_REQUIRED`、A3 blocked、2025 locked。

## 4. 上一轮指定复现项：CLOSED / PARTIAL / OPEN

| 项目 | 状态 | 独立结果 |
|---|---|---|
| new-O 与 anchor 的 year/evidence role/panel/model/seed/delta shared identity | **CLOSED** | year、panel、model、seed-policy 与 delta 五类 cross-layer drift 均 fail closed；实现位于模块 913–924、1140–1186、1307–1334。 |
| native/expanded anchor 的完整 scientific identity | **CLOSED** | endpoint、model、panel、delta、`O_group_rule_sha256` 漂移均因 scientific identity 不同被拒绝（894–910、1091–1137、1308–1319）。 |
| 同一 new O 内 group-rule hash 一致 | **CLOSED（Q2 row 层）** | 同一 `O_audit` 两行使用不同 rule hash 被拒绝（1227–1249）。这不关闭第 5.2 节的 family 跨 endpoint 漏洞。 |
| panel domain false / missing / non-bool | **CLOSED** | `annual_design_rows_retained`、`global_all_O_complete_case_prohibited`、`all_panel_members_required` 的 false、整数替代布尔与缺字段反例均被拒绝；domain schema 为精确字段集（1649–1665）。 |
| self-consistent hollow seed policy | **CLOSED** | family verifier 与 panel freeze 共用精确 seed-policy validator；空/重复/bool/非整数 seed 与缺 estimand 均不能通过（118–147、608–615、1678）。 |
| partial/bare threshold policy | **CLOSED** | 缺 `policy_id`/`selection_role` 的 self-hashed partial policy 因精确 schema 被拒绝；role 必须为无结果访问状态（172–198、1599–1603）。 |
| non-string panel/member/model/threshold IDs 与 bool seed | **CLOSED（上一轮所指范围）** | panel ID、`O_train`、member ID、model ID、threshold policy ID 的非字符串以及 bool seed 均被拒绝（1506–1603、1623–1691）。 |
| deployment/source/model-ID 约束 | **CLOSED（结构合同）** | `O_at_inference`、output type、backbone/feature/preprocessing、结构化 source binding、seed→model binding 与跨 member/seed 全局 model-ID 唯一性均重验证。真实文件存在性和字节 hash 仍是 A3 前的 source preflight。 |
| no-new-O 合法路径 | **CLOSED（结构合同）** | 合法 receipt 得到 `expanded_equals_anchor_when_no_new_O=true`、`changed=false`、`affirmative_new_O_change_allowed=false`。 |
| no-new-O receipt/provenance tamper | **CLOSED（hash/schema/scope 层）；生产来源核验仍 OPEN** | 直接改 count/source 会触发 receipt hash mismatch；删除 O registry 后重哈希仍因精确 schema 拒绝；修改 shared year 后重哈希仍因 anchor scope mismatch 拒绝。receipt 已绑定 atlas/candidate/precision/selection/O-registry path+hash+schema/count；这些路径指向的真实字节尚未由此函数读取，必须留给 A3 source preflight，不能把结构闭合表述为端到端来源真实性。 |
| family top-level year/size/hash tamper | **CLOSED** | 未重哈希 year tamper、重哈希 year tamper、重哈希 family size tamper 均被拒绝。 |
| self-hash forged empty panel | **CLOSED** | verifier 重跑完整 panel validator，残缺对象被拒绝且输入不被修改。 |
| 93-test 声明 | **CLOSED AS EXECUTION FACT** | 数目、结果和 warning 来源均独立一致；现有测试未覆盖第 5 节反例，因此不能单独决定 A2。 |
| 2025 lock | **CLOSED / ACTIVE** | registry 九个 freeze/release 条件仍为 false，`micro_outcome_or_performance_read=false`；guard 与 registry tests 通过。 |

## 5. 新发现（仅列可复现的 gate-critical 问题）

### [P1] anchor/new-O membership 与预声明 contrast universe 没有机器绑定

规范把 anchor 固定为 `SEX_A`、`HISPALLP_A`、`DISAB3_A`，expanded 定义为 anchor 加冻结 new-O（spec 7–12），并要求正式 family manifest 在 A3 前显式列全 contrasts（62–73）。当前实现没有完成这个语义闭环：

- `AuditAttributeSpec` 有 `audit_role`（模块 340–356），但 generator 构造 slots 时只写 variable/group/rule hash，完全丢弃 `audit_role`（456–505）。generator 也没有检查 `scope=anchor` 时 attributes 必须全为 anchor。
- verifier 只验证 slot 自身 hash/ID/null/status，不验证 audit role 或 expected O universe（551–717）。
- Q2 仅要求 `audit_layer` 字符串等于调用方指定层、native/expanded anchor keys 相等、anchor 与 new-O 的 contrast keys 不重叠（1189–1256、1275–1344）；它不接收或验证 frozen family manifest，不检查 anchor/new-O 的 `O_audit` 集合互斥，也不检查输入 contrast IDs 是否完整等于预声明 slots。

独立反例 A：用唯一属性 `VISIONDF_A`、`audit_role="expanded_candidate"` 调用 `generate_statistical_families(..., scope="anchor")`。generator 成功，slot 中不存在 `audit_role`，`verify_statistical_family_manifest` 返回 success。即机器对象把 expanded-only O 合法伪装成 anchor。

独立反例 B：anchor native/expanded 只含 `SEX_A/WITHIN`，new-O 行仍使用 `O_audit="SEX_A"`、不同 contrast ID、`audit_layer="new_O"` 和 `EXCEEDS`。聚合器返回：

```text
expanded_conclusion = SUPPORTED_EXCEEDS_TOLERANCE
new_O_content_change.changed = true
new_limiting_O = ["SEX_A"]
per_O_coverage = [("SEX_A", "anchor"), ("SEX_A", "new_O")]
```

这会把同一 anchor 属性重新贴上 new-O 标签，产生虚假的“扩展审计增量发现”；也允许遗漏另外两个固定 anchor 或任意预声明 NA slot。它直接破坏首篇论文的主轴，不是普通 schema 美化。

最低修复要求：

1. family slot 和 manifest 显式绑定 `audit_role` 与 O-registry/version/hash；generator 对 `scope=anchor` 只允许完整冻结 anchor set，对 `scope=expanded` 要求同一 anchor set加冻结 new-O set；
2. verifier 重建并检查每个 O 的唯一 role、规则、groups/reference 和预声明 slot universe；
3. Q2 接收已验证的 native/expanded family manifests（或其 hash 与 exact expected contrast-ID sets），要求所有预声明 slots—including not-estimable slots—一一出现，并要求 anchor/new-O 的 O 集合互斥且与 registry role 一致；
4. 加入 expanded-only O 进入 anchor、缺少一个固定 anchor、同一 O 同时属于两层、遗漏一个 NA slot 四类反例。

### [P1] 同一 O 的 group-rule identity 可在 family 内跨 endpoint 漂移

Tranche 3 已在 Q2 rows 内检查同一 O 的 rule hash 一致，但 family generator/verifier 没有建立全 manifest 的 `O_audit -> O_group_rule_sha256` 唯一映射。模块 456–505 逐个 `AuditAttributeSpec` 直接生成 slots，没有禁止重复 variable；verifier 680 行只逐 slot 检查 hash 格式，未比较同一 O 的其他 slots。

独立反例：从合法单 O family 开始，把 `SEX_A` 的 FPR slot rule hash 从 `1…1` 改为 `2…2`，保留 FNR 为 `1…1`，随后重算 contrast IDs、family hash 和 manifest hash。`verify_statistical_family_manifest` 返回 success：

```text
[('fnr', '1…1'), ('fpr', '2…2')]
verify_statistical_family_manifest(...) = success
```

另一个更直接的 generator 反例是同时传入两个名为 `SEX_A`、但 groups/rule hash 不同的 `AuditAttributeSpec`；generator 和 verifier 均成功。这使构成同一 EO family 的 FNR/FPR components 实际使用不同分组/缺失/NIU 规则，不能解释为同一 O 的 max EO 或统一三态结论。

最低修复要求：generator 禁止重复 `AuditAttributeSpec.variable`；verifier 对全 manifest 重建唯一的 O role/group/reference/rule identity 并要求所有 family、endpoint、model/method slots 一致；该 identity 应与第 5.1 节的冻结 O registry hash 联动。加入“同 O 跨 endpoint rule drift 后重算全部 hashes”的固定反例。

## 6. Registry、checklist 与冻结诚实性

判定：**ACCEPTABLE AS DRAFT，但不能按当前 checklist 宣称 tranche 3 software contract 已关闭。**

诚实保留的状态包括：

- registry 是 `DRAFT_NOT_FROZEN`，`real_performance_scan_authorized=false`；
- main panel 是 `TECHNICALLY_BOUND_PENDING_MEMBER_CONTRACT_AND_COVERAGE_PREFLIGHT`，不是 frozen；
- panel seed estimand、feature/preprocessing/training/prediction-source hashes、completion model hashes、thresholds、prediction/`y_hat`/order hashes 与 coverage preflight 仍为 `null/PENDING`；
- A1 application context、delta、正式 O/group/reference/nesting、confirmatory pairs、allowed claims 与伦理/作者责任尚未由人类批准；
- simultaneous CI 明确是 A2 candidate，production reference/coverage validation 未完成；
- A3 与 2025 flags 均未开放，receipt 没有自批准。

但 checklist 的 A1-044（Q2 group rule/shared scope）只能视为关闭了其字面 cross-row drift 测试，不能覆盖 audit-layer membership/completeness；A1-037/A1-042 也不能被解释为完整 family semantic verifier。下一 tranche 应新增独立 checklist 项，或收紧这些项的定义后重新审查。

## 7. A2 / A3 / 2025 与 paper 门禁

### A2 synthetic contracts

**不恢复接受。** 上一轮指定的 tranche-3 修复均已关闭，但第 5 节两个 P1 仍允许机器生成或验证错误的 anchor/expanded universe 与 O group-rule identity。最小 tranche 4 只需修复这两个 family/Q2 语义闭环并加入合成回归，无需触碰真实性能。

### A3

**NO-GO。** 即使第 5 节完成，A3 仍需单独关闭：

- 2022/2023 官方 codebook harmonization 与 codebook-wide eligibility ledger；
- 人类批准的应用场景、endpoint delta/单位/领域依据；
- 正式候选 O、group/reference/nesting/correlation、role 与允许主张；
- 少量确认性 method-baseline pairs；
- seed aggregation/training-randomness estimand；
- 所有主 panel members 的 model/source/feature/preprocessing/threshold/prediction/`y_hat`/order hashes；
- 不读取结局/性能的 2023 prediction availability 与逐 O coverage preflight；
- simultaneous-CI 参考实现或预设复杂调查设计模拟覆盖率 QA；
- A3 source/run manifest 冻结、独立 review 与新的书面授权。

2024 仍只能按已声明规则作已知结果的回顾性复核，不能升级为前瞻验证。

### 2025

**LOCKED / NO-GO。** 所有 O、模型、指标、检验 families、源码、source hashes、record-order contract 与人类 one-shot release 决定冻结前，不得读取 2025 微观结局或性能。当前 registry、repair receipt 与 guard 均保持该边界。

### Paper package

**NO-GO。** 当前只能声称完成了计划、draft registry 和部分 synthetic contract hardening；不能声称发现新增不公平、有限审计高估去偏效果、方法排序改变、跨年验证成立或 paper-ready。投稿最小证据包仍需：

1. 可审计的覆盖/排除/不可估计边界与正式 O flow；
2. 2023 扩展审计增量发现，并区分“排除重要效应”和“精度不足”；
3. 固定 comparison panel 下 anchor vs expanded 对同一方法结论的改变；
4. gap、弱势组、其他组、总体 BA/风险质量和其他 O 的绝对权衡；
5. 2024 回顾性复核与冻结后的 2025 one-shot validation；
6. A3.5 Methods、首版图、candidate flow 和 claim-evidence table；
7. 伦理、作者责任、AI disclosure 与 data responsibility 记录。

## 8. 当前允许范围

- 修复第 5 节两个纯软件合同并增加合成反例：**GO**。
- 继续 metadata/codebook harmonization、eligibility ledger、非性能 source/hash inventory、Methods 骨架：**GO**。
- 把 tranche 3 或 A2 标为 accepted：**NO-GO**。
- 生成 2023 prediction replay 或运行真实性能扫描：**NO-GO**。
- 访问 2025 微观结局/性能：**NO-GO / LOCKED**。
- 开启 A4、声称 paper-ready 或投稿：**NO-GO**。
