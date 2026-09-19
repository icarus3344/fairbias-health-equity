# A0–A2 repair tranche 2 final independent review v3

日期：2026-09-19  
角色：独立 Reviewer（非执行 Worker）  
审查对象：`A0_A2_REPAIR_TRANCHE_2_V3`

## 1. 最终判定

总判定：**REPAIR REQUIRED；A2 synthetic-contract acceptance 尚不能恢复；A3 NO-GO；2025 LOCKED。**

Tranche 2 已经关闭上一份备忘录明确列出的四个反例：anchor native/expanded 的 endpoint、model、panel、delta identity drift；合法 no-new-O 路径及 receipt 篡改；self-hash forged empty panel；family 顶层 year/size/hash 篡改。66 项相关测试、修复收据版本链和当前文件 hashes 也都可以独立复现。

但对任意自洽输入继续做对抗性检查后，仍发现会破坏论文主 estimand 或 v2/v3 底线的语义逃逸路径：new-O 行没有绑定到同一 anchor 年度/panel/模型/delta；panel 可显式否定逐 O 调查域的关键约束仍被 freeze；family verifier 可接受缺失 seeds 与 seed estimand 的自洽 manifest；threshold policy 可缺失 policy ID/selection provenance，model ID 也可不是字符串。它们不是 production 精度问题，而是当前隔离软件合同仍可错误放行，因此不能用“原四个反例已关闭”自动恢复 A2。

本结论不否定 Worker 的实质修复，也不改变 registry 的诚实 draft 状态。下一轮只需做定向 contract hardening，无需运行真实性能或读取 2025。

## 2. 审查范围与证据边界

已完整检查：

- `A0_A2_REPAIR_TRANCHE_2_REPORT_V3.md`
- `repair_verification_v3.json`
- `planning_verification_v3.json`
- `src/nhis_fairbias/bias_attribute_audit_v3.py`
- 两份 v3 synthetic/registry tests
- `PRE_FREEZE_REGISTRY_V3_DRAFT.json`
- `A1_INTERFACE_AND_FAMILY_SPEC_V3.md`
- `A1_FREEZE_CHECKLIST_V3_DRAFT.json`
- 上一份独立审查 `INDEPENDENT_REVIEW_A0_A2_REPAIR_V3.md`

本审查没有读取 NHIS 真实性能、2023/2024 结局结果或 2025 微观结局/性能，没有生成 prediction replay，也没有修改 Worker 文件。全部反例是内存合成对象。默认 Homebrew `python3` 缺少 NumPy 的一次导入失败未执行任何测试、未计作证据；随后使用 pytest 所在 Python 3.13 环境完整重跑。

## 3. 独立测试与 hash 复核

完整相关集合：

```text
PYTHONPATH=src pytest -q \
  tests/synthetic/test_bias_attribute_audit_v3.py \
  tests/synthetic/test_bias_attribute_audit_v3_registry.py \
  tests/benchmark/test_survey_domain_contract.py \
  tests/benchmark/test_survey_linearization.py \
  tests/benchmark/test_survey_sufficient_totals.py \
  tests/test_nhis_survey.py

66 passed, 3 warnings in 1.70s
```

三个 warning 仍来自既有 `src/nhis_fairbias/benchmark/survey_batch.py:60`。针对四个原反例、deployment/source/model-ID、receipt supersession 和 2025 lock 的 16 个定向实例独立重跑为 `16 passed`。

以下 Worker 声明 hashes 与当前字节逐项一致：

- module：`0ad71d0cb31c375fa8195c80384d486dbf24f47c897b5eba7057debf93ef8749`
- main synthetic test：`f3314d7d1089668c021c82ba83ec808bf4e1ad1fae6b5617a1126f278afd96d6`
- registry test：`5f5fe644c94c8411ba7081804cd63669a2c7549b2672c88a366d50cb2ade0dea`
- registry：`fe9b971cba4cedf549a821ca7ca2a4f485a5d9849906b1bd745dead5c5f0958d`
- interface spec：`70922c8ccd1f5b285f548699eb05749a96c7e2ca1b4126ba797e5727f54b8b9c`
- checklist：`0e572c0e3f1dedb0f048fc1d215a6ec05eabc6299a17afe46314c410bbf27906`
- superseded planning receipt：`6813360d11a1edac8a7edd6756442f24f9c3c2656eaec0a8c754cb90fcf5d313`
- tranche-1 report：`34a40c571fcbb318818521c16a4577e0d7fbcecf8d45870c119a22724199d03d`
- tranche-2 report：`dcb9e7a276b88e8c5da3fefc9a20a6d54356e847b8be8ed310c479b8467aa700`

`repair_verification_v3.json` 自身当前 SHA-256 为 `f95d28ee9191efb2ffb111f046e76ee0e856db2e9ffd85003d27546392803905`；它按声明不自哈希，避免循环引用。receipt 内登记的全部 output/report hashes 均匹配。registry main-panel draft binding hash 也独立复算为 `d91130f79576c9f7d87d5b010ee563d6363045ca6b8b2a94a27d2de3900930ee`。

## 4. 指定复现项：CLOSED / PARTIAL / OPEN

| 复审项 | 状态 | 独立结果 |
|---|---|---|
| Q2 anchor native vs expanded endpoint/model/panel/delta mismatch | **CLOSED** | 四种漂移均抛出 `family expansion must not change the scientific contrast identity`。完整 identity 位于模块 832–999，成对比较位于 1111–1124。 |
| 合法 no-new-O 路径 | **CLOSED（局部合同）** | 合法 hash-bound receipt 使 `expanded_equals_anchor_when_no_new_O=true`，`new_O_content_change.changed=false`，且禁止肯定式 new-O change。 |
| no-new-O receipt 缺失、直接篡改、重新哈希但计数不闭合 | **CLOSED（局部合同）** | 分别因缺 receipt、hash mismatch、count reconciliation 失败而拒绝（模块 873–948、1131–1134）。端到端候选宇宙/precision evidence 绑定仍属 A1/A3 freeze，见下文。 |
| self-hash forged empty panel | **CLOSED** | `verify_frozen_panel` 对 `{"panel_id":"forged","status":"FROZEN","members":[]}` 重新执行完整 validator，因缺 required fields 拒绝且不修改输入（1478–1489）。 |
| family 顶层 year 未重哈希 | **CLOSED** | 以 `manifest_sha256 mismatch` 拒绝。 |
| family 顶层 year 重哈希 | **CLOSED** | 以 `family identity contradicts manifest_identity` 拒绝。 |
| family size 重哈希 | **CLOSED** | 以 `family_sizes does not match actual slots` 拒绝。 |
| deployment output/`O_at_inference`、结构化 source binding、跨 member/seed model-ID 重复 | **CLOSED（已测试部分）** | Worker 四个反例均 fail closed（模块 1295–1464；测试 578–607）。 |
| 历史 planning receipt superseded | **CLOSED** | 旧 receipt 明确为 `SUPERSEDED_BY_REPAIR_VERIFICATION_V3`、`historical_snapshot_only=true`；新 receipt 保持 review-required/A3-blocked/2025-locked。 |
| 66-test 声明 | **CLOSED AS EXECUTION FACT** | 独立重跑数目和 warning 来源一致；但现有测试未覆盖第 5 节的新反例，因此不能单独决定 A2。 |
| 2025 lock | **CLOSED / ACTIVE** | registry 的九个 freeze/release 条件均为 false，`micro_outcome_or_performance_read=false`；guard test 通过。 |

## 5. 新发现（按严重度排序）

### [P1] new-O contrasts 没有绑定到 anchor expanded 的固定年度、panel、模型或 delta

当前代码只逐行验证 new-O identity 的字段存在与类型（951–999），随后仅检查 anchor/new-O keys 不重叠（1125–1126）；没有要求 new-O rows 与 anchor-expanded rows 共享 year/evidence role、panel ID/hash、`O_train`、model 或 method-baseline pair、seed-policy hash、endpoint 对应的 delta/units/rationale。

独立反例把一个 new-O 行设为 2024 retrospective、不同 panel、不同 model、不同 delta，同时保留合法 self-contained identity。聚合器接受它，并把其 `SUPPORTED_EXCEEDS_TOLERANCE` 写入原 2023 anchor 的 expanded conclusion。这直接违反“同一固定模型集合、同一年度源只改变审计范围”的 paper 主轴。

修复要求：为一次 Q2 aggregation 建立显式 shared-scope identity，并要求所有 anchor-expanded 与 new-O rows 共享 year/evidence role、panel ID/hash、`O_train`、model 或 method-baseline pair、seed-policy hash，以及按 endpoint 映射的 delta/units/rationale。`O_audit`、group/reference、contrast ID 可以按预声明 slot 变化。新增至少 year、panel、model/method、seed 与 delta 的 cross-layer 反例。

### [P1] Q2 machine key 缺少 spec 已声明的 `O_group_rule_hash`

Interface spec 50–55 把 `O_group_rule_hash` 列为主对照键，但 `_Q2_SCIENTIFIC_IDENTITY_FIELDS`（832–847）、registry `required_scientific_contrast_identity` 和测试 helper 均未包含它。同一个 `O_audit/group/reference` 数值可以来自不同 missing/NIU 合并、分组或嵌套规则，却仍被当作同一 contrast。

修复要求：在 attribute registry 冻结后，把 `O_group_rule_sha256`（或项目统一命名）加入 family slot、Q2 scientific identity、结果行和对抗测试；native 与 expanded 同一 anchor slot 必须完全相同，new-O slot 必须绑定自己的已冻结 group-rule hash。

### [P1] panel freeze 可接受与逐 O 调查域底线显式矛盾的合同

`_validate_complete_panel_contract` 只要求 `common_domain_contract.scope == "per_O"` 和 all-members flag 为 true（1427–1435）。我在 otherwise-valid panel 上加入 `annual_design_rows_retained=false` 和 `global_all_O_complete_case_prohibited=false`，`freeze_comparison_panel` 仍返回成功。也就是说，一个对象可以同时写 `scope=per_O` 又显式允许全 O complete-case、删除年度 design rows，而 verifier 仍称其完整。Registry 当前值是正确的 true；缺口在通用 freeze/verifier。

修复要求：完整 panel contract 必须要求这两个布尔字段存在且为 true，并拒绝未知的矛盾别名；新增 false、缺失和非 bool 三类测试。这是 v2/v3 的不可退让底线，不能留到 A3 才解释。

### [P1] family verifier 仍未重验证完整 seed-policy 语义

Generator 正确要求非空唯一 seeds、`aggregation_estimand` 与 `training_randomness_inference`（365–372），但 verifier 只要求 `status=FROZEN` 和 seed-policy hash 自洽（551–554）。我把一个合法 manifest 的 seed policy 一致改为仅 `{"status":"FROZEN"}`，重算所有 contrast/family/manifest IDs 与 hashes 后，`verify_statistical_family_manifest` 返回 success。

这会让一个没有 seed 列表和 seed estimand 的手工 manifest 绕过 verifier，违反 A1 的 seed 冻结要求。修复要求：抽出 generator/verifier 共用的 seed-policy validator，并在 family 与 panel 路径复用；对空 seeds、重复 seeds、bool/non-integer seed、缺 estimand 做固定反例。

### [P2] threshold provenance 与 identifier 类型约束仍可绕过

两个独立反例均被 `freeze_comparison_panel` 接受：

1. self-hash threshold policy 仅含 `status`、`threshold`、`comparison_operator`，缺 `policy_id` 和 `selection_role`；`_verify_threshold_policy`（152–165）不检查它们。
2. member 的 `model_ids=[1]` 且 seed binding `model_id=1`；member validator 只检查 truthiness/相等/唯一（1349–1357），不要求 canonical nonempty string。member IDs 和 seed 类型也没有完整 canonical-type 约束。

修复要求：threshold verifier 使用精确 schema，`policy_id` 为非空字符串，`selection_role` 只能是预注册的无结果访问状态；panel/member/model IDs 为非空字符串；seeds 为非 bool 整数且顺序唯一。source `path+sha256` 结构、deployment output 和 cross-member uniqueness 本身已正确实现，不需推翻。

### [P2] no-new-O receipt 的端到端 provenance 尚未进入机器对象

局部 receipt 的 hash/计数/precision 枚举已经正确，但 receipt 本身只含一个不透明 `selection_rule_sha256`，没有直接绑定 year/evidence role、候选 registry/eligibility ledger hash、正式 O/group-rule version 或 precision-evidence manifest。一个来自其他年度/候选宇宙的合法 receipt 可被复用于当前 Q2 空 new-O 路径。

可在修复第一个 P1 时一并处理：receipt 绑定 shared-scope identity、candidate-registry/eligibility-ledger hash、selection-rule hash 和 precision-evidence manifest hash；A3 前再由 decision log 批准其科学内容。若项目选择让 `selection_rule_sha256` 的被哈希文档承载这些内容，则必须在 receipt 中登记该文档路径/schema，并由 verifier 复核，而不是只接受任意 64 位字符串。

## 6. Registry、receipt 与冻结诚实性

本部分判定为 **ACCEPTABLE AS DRAFT**：

- registry 仍为 `DRAFT_NOT_FROZEN`，`real_performance_scan_authorized=false`；
- main panel 仍为 `TECHNICALLY_BOUND_PENDING_MEMBER_CONTRACT_AND_COVERAGE_PREFLIGHT`，不是 frozen；
- seed policy 为 `HUMAN_DECISION_REQUIRED`，五个 members 均 `member_contract_complete=false`；
- completion model hashes 保持 `null/PENDING_MODEL_ARTIFACT_CONTRACT`；全部 thresholds 保持 `PENDING`；
- feature/preprocessing/training/prediction-source hashes 与 coverage preflight 未伪造；
- 旧 planning receipt 的历史 `ACCEPTED_FOR_SYNTHETIC_CONTRACTS` 已明确 superseded，不能再作为当前授权；
- repair receipt 没有自批准 A2，A3 为 blocked，2025 为 locked；
- checklist 为 43 项、23 complete、20 incomplete，`a1_freeze_ready=false`、`a3_authorized=false`、`validation_2025_authorized=false`。

没有发现未知 hash 被补造，也没有发现 registry 或 receipt 错误开放真实扫描。当前问题是通用软件 verifier 仍有语义缺口，不是 draft registry 造假。

## 7. A2 / A3 / 2025 门禁

### A2 synthetic contracts

**不恢复接受，维持 `REPAIR_REQUIRED`。** 最小下一 tranche 仅需：

1. Q2 为所有 expanded rows 强制 shared-scope identity，并加入 `O_group_rule_sha256`；
2. panel common-domain 两个底线字段强制存在且为 true；
3. family verifier 复用完整 seed-policy validator；
4. threshold policy 与 panel/model/seed identifiers 使用精确 schema/type；
5. no-new-O receipt 绑定当前 shared scope 与候选/precision provenance；
6. 把本备忘录所有 ACCEPTED 反例加入回归测试，再由独立 Reviewer 重跑。

这些均可使用纯合成对象完成，不需要运行 2023 replay 或接触真实性能。

### A3

**NO-GO。** 除上述 A2 修复外，仍需单独关闭：

- 2022/2023 官方 codebook harmonization 与 codebook-wide eligibility ledger；
- 人类批准的应用场景、endpoint delta/单位/领域依据；
- 正式候选 O、group/reference/nesting/correlation 规则及允许主张；
- 少量确认性 method-baseline pairs；
- seed aggregation/training-randomness estimand；
- 全部主 panel members 的 model/source/feature/preprocessing/threshold/prediction/`y_hat`/order hashes；
- 不读取结局/性能的 2023 prediction availability 与逐 O coverage preflight；
- simultaneous-CI 参考实现或预设复杂调查设计覆盖率 QA；
- A3 source/run manifest 冻结与新的书面授权。

软件合同即使在下一 tranche 被接受，也不会自动授权 A3。2024 仍只能作已知结果的回顾性复核。

### 2025

**LOCKED / NO-GO。** 所有 O、模型、指标、families、源码、source hashes、record-order contract 与人类 release 决定冻结前，不得读取 2025 微观结局或性能。当前 registry/receipt 与 guard 均保持该边界。

## 8. 允许继续的范围

- 定向修复上述纯软件合同并增加合成反例：**GO**。
- 继续官方 metadata/codebook harmonization、eligibility ledger、非性能 hash/source inventory：**GO**。
- 把当前 tranche 2 标为 A2 accepted：**NO-GO**。
- 生成 2023 prediction replay 或运行真实性能扫描：**NO-GO**。
- 访问 2025 微观结局/性能：**NO-GO / LOCKED**。
- 开启 A4、声称 paper-ready 或投稿：**NO-GO**。

结论标签：`INDEPENDENT_REVIEW_TRANCHE_2_COMPLETE__REPAIR_REQUIRED__A2_NOT_ACCEPTED__A3_NO_GO__2025_LOCKED`。
