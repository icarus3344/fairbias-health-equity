# A1 接口与统计 family 规范 v3

日期：2026-09-19。状态：**REPAIR TRANCHE 4 WORKER IMPLEMENTED；INDEPENDENT REVIEW PENDING；SCIENTIFIC CONTENT NOT FROZEN；A3 NOT AUTHORIZED**。

本规范把 anchor/expanded 审计、固定 comparison panel、逐 O 调查域以及 Q1 与 Q3/Q4 家族的机器接口固定下来。它不登记任何真实差异，也不替代 delta、正式 O、分组/参照与确认性方法对的科学审批。

## 1. 冻结对象与审计范围

- `anchor audit`：固定为 `SEX_A`、`HISPALLP_A`、`DISAB3_A`。
- `expanded audit`：anchor 加上发现阶段按预注册规则入选并冻结的 `new_O`；不是 atlas 所有条目的自动并集。
- `AuditAttributeSpec` 必须把 `audit_role`、`audit_role_sha256`、`O_group_rule_sha256`、O-registry version/hash、groups 与 reference 作为同一机器身份。role hash 必须绑定 O 名称、role 和 registry identity；同一 O 不得同时进入 anchor 与 new-O。
- 所有 family 和 Q2 输入必须绑定同一个 `FROZEN_WITHOUT_PERFORMANCE_RESULT_ACCESS` contrast-universe receipt。receipt 明列固定有序 anchor set、互斥 new-O set、每 O 的完整属性身份、selection-rule hash、registry hash、panel/family identity、以及包含不可估计 slot 的 exact expected contrasts，并以 `contrast_universe_sha256` 封存。
- 同一轮 anchor/expanded 对照必须复用同一 panel、同一模型字节、同一预测、同一阈值策略和同一年度源。
- 每个 O 在自己的有效域上分析。跨 O 比较不得改成所有 O 的全局 complete-case 样本。
- `main_risk_gbdt_arm001_v3` 当前仅为 `TECHNICALLY_BOUND_PENDING_MEMBER_CONTRACT_AND_COVERAGE_PREFLIGHT`，不是 frozen。三个 parent comparator 的 seed/model/model SHA 已从 admission manifest 绑定；两个 completion 成员的 model SHA、全部成员的 seed estimand、feature/preprocessing/training-source/prediction-source hash、threshold hash 和 2023 coverage 仍保持 `null/PENDING`。成员 `source_binding` 只接受结构化 `path + sha256`，且 backbone、feature/preprocessing contract、output type、`O_at_inference`、threshold operator 必须与主 deployment contract 一致。model ID 策略固定为跨成员与 seed 全局唯一。完成这些字段前 `freeze_comparison_panel` 会 fail closed。
- `verify_frozen_panel` 对任意收到的对象无副作用地重新执行完整 schema、成员、seed、deployment、source-binding、model-ID 唯一性与 threshold 语义验证；只提供自洽 self-hash 的空/残缺对象不能通过。
- frozen 主 panel 的 deployment/domain 合同不可降级：`event_probability_p`、`O_at_inference=false`、`scope=per_O`、`all_panel_members_required=true`、`annual_design_rows_retained=true`、`global_all_O_complete_case_prohibited=true` 均必须显式存在且为正确类型和值；false、整数替代布尔、缺字段或旧别名均 fail closed。
- panel/member/model/threshold-policy/`O_train` 等标识只接受非空字符串；seed 只接受非 bool 整数。冻结 threshold policy 必须使用完整精确 schema，并把 `selection_role` 固定为 `FROZEN_WITHOUT_EVALUATION_RESULT_ACCESS`；bare self-hash partial policy 无效。
- 后加低覆盖、不同输出、需要 O-at-inference 或不同 backbone/feature contract 的方法，只能建有 parent hash 的 supplemental panel，不能反向缩小主 panel 域。

## 2. A3 runner 的必需输入合同

每个年度运行单元至少包含：

| 输入 | 必需字段/约束 |
|---|---|
| annual source | 年度、源路径、bytes、SHA-256、source manifest hash；2025 在 release 前不得创建此运行单元 |
| full survey design | 与 annual row 一一对齐的 `WTFA_A`、`PSTRAT`、`PPSU`；domain 外记录保留为零贡献，不删除 PSU |
| record identity | 无重复稳定记录键、annual order SHA-256；不得把原始 ID 写入公开结果 |
| outcome | `MEDDL12M_A` 的有效性掩码和冻结编码；仅在获准年度读取 |
| predictions | 每个 panel 成员显式绑定 `O_train`、seed policy、seed→model ID→model SHA、feature/preprocessing/training-source/prediction-source hash、`p_event` prediction hash、threshold policy/hash、二元 `y_hat` hash、输出语义和 order hash；未知字段必须为 `null/PENDING`，不得补造 |
| O registry | `variable`、role、substantive groups、missing/NIU、reference、跨年映射、nesting cluster、group-rule hash |
| panel | 完整冻结 JSON 和 `panel_sha256`；所有成员在该 O 域上均须有有效预测 |
| inferential registry | endpoint、delta/units/rationale、Q1 模型、确认性方法-baseline 对、描述性方法对、alpha、family hash |

任何源、模型、预测、`y_hat` 和 O 向量必须先通过 exact-order alignment；记录键必须是上游规范化后的非空唯一字符串，整数 `1` 与字符串 `"1"` 不得在本层静默折叠。长度相同不构成对齐证据。模型输出语义不相同的成员不得进入同一主 panel。

`p_event` 仅用于 Brier、校准或其他预注册风险质量指标。FNR/FPR/TPR/TNR/BA 只允许输入 `binary_decision_y_hat`；`y_hat` 必须由 `p_event` 与冻结的 threshold policy 生成，且 prediction、threshold、`y_hat` 与 order hash 分别保存。连续概率的组内均值不得命名为 TPR/FPR/FNR。

## 3. 逐 O domain 与比较键

对年度 `t`、属性 `o`、panel `p`：

```text
D(t,o,p) = valid_Y(t)
           AND O_o in frozen substantive groups(o)
           AND valid_prediction(member) for every member in p
```

完整设计仍保留年度所有 strata/PSU；`D(t,o,p)=false` 的记录只在 estimating equation 中贡献零。输出必须记录 annual rows、domain rows、逐成员 prediction-valid rows、事件数、组别 n、加权量、ESS、PSU/strata 支持、lonely-PSU 处理和 domain/order hash。

主对照键为：

```text
(year, evidence_role, scope, panel_id, panel_sha256,
 O_train, O_audit, O_group_rule_sha256, group, reference_group,
 audit_role, audit_role_sha256, O_registry_version, O_registry_sha256,
 contrast_universe_sha256,
 endpoint, model_or_method_pair, seed_policy_sha256,
 delta, delta_units, delta_rationale_id, contrast_id,
 family_id, family_sha256)
```

anchor 与 expanded 结论变化必须以相同 panel/hash 为前提；不同 O 的数值不伪装成同一 respondent domain 上的直接差值。

## 4. 统计问题与 family 生成

`generate_statistical_families` 只接受已声明的 `AuditAttributeSpec`、同一 frozen contrast-universe receipt、Q1 固定模型 ID、确认性方法-baseline 对、描述性方法对、端点以及显式的 year/evidence role/scope/panel hash/`O_train`/seed policy/delta rationale。seed policy 未冻结、panel hash 非法或任何输入与 receipt identity 不一致时 fail closed。`scope=anchor` 必须精确等于固定三个 anchor；`scope=expanded` 必须精确等于相同 anchor 加互斥 frozen new-O，不能缺失、额外加入或重贴 role。

- Q1 confirmation：`d(model,o,g)=0`。用于固定模型的组间差异存在性；独立 family 和 contrast ID。
- Q3/Q4 confirmation：`d(method,o,g)-d(matched baseline,o,g)=0`。只包括少量预先指定方法对。
- Q3/Q4 description：相同相对 estimand，但只作描述性迁移矩阵，不与确认性 family 混合。
- 每个 slot 显式写入 `PREDECLARED_NOT_YET_ESTIMATED` 和 `estimate=null`；未估计/NA slot 保留在 family 分母中，不能在看到支持度后删除。
- manifest 顶层必须嵌入 hash-bound contrast-universe receipt 并有 `manifest_sha256`。verifier 要求固定 schema/version/status、精确的三个 family 名称、顶层 `manifest_identity` 与每个 family/slot 一致、`family_sizes` 与实际 slot 数一致，并重新计算 family ID/hash、contrast identity、null hypothesis 与 slot status。每个 scope 的 slot 集必须精确等于 receipt 的预声明 universe；只重算子 family hash 而不验证顶层 summary 的实现不得用于门禁。
- family generator 与 verifier 复用同一 seed-policy validator：精确字段、`status=FROZEN`、非空唯一的非 bool 整数 seeds，以及非空 `aggregation_estimand` 与 `training_randomness_inference`。即使重新计算全部 IDs/hashes，自洽但空洞的 seed policy 仍必须拒绝。每个 family slot 还必须绑定该 O 的 role/group-rule/registry identity；同一 O 跨 endpoint、group、model/method 与 family 不得漂移 group-rule hash。

正式 family manifest 在 A3 前必须显式列出所有 contrast，而不是只存生成规则。正式 O、参照组、确认性方法对或 endpoint 的任何改变都会产生新版本和新 hash，不能静默覆盖。

## 5. 推断与三态输出

每个确认性 contrast 的结构化输出必须至少包含：

```text
estimate, standard_error,
ordinary_ci_lower, ordinary_ci_upper,
simultaneous_ci_lower, simultaneous_ci_upper,
raw_p, holm_adjusted_p,
family_id, family_size_including_not_estimable,
delta, delta_units, delta_rationale_id,
conclusion_state,
ci_half_width, simultaneous_margin_of_error_from_point,
supplied_critical_margin_of_error,
largest_absolute_effect_compatible,
rules_out_effects_with_absolute_magnitude_above_delta,
rules_out_within_tolerance_region,
delta_exclusion_statement,
estimation_status
```

规则：

- Holm 只调整 p 值，不产生 simultaneous CI。
- 普通 CI 与 simultaneous CI 分字段存储和展示。
- 当前 max-standardized joint-replicate 区间仅是 A2 candidate；A3 前仍需参考实现/覆盖率验证。
- 最大 EO/gap 从完整预声明 FNR/FPR 组件 family 的 simultaneous intervals 投影；不能对事后最大组对套普通区间。
- 三态只能是 `SUPPORTED_WITHIN_TOLERANCE`、`SUPPORTED_EXCEEDS_TOLERANCE`、`INSUFFICIENT_EVIDENCE`；“不显著”不能自动转换为第一态。
- 最低 n、事件、ESS、PSU 是可估门槛，不是精度充分。每个冻结 O 必须报告当前区间能排除多大的差异。
- `critical × SE` 只能叫 margin of error，不叫 MDE。当前不计算 MDE；若以后计算，必须另行冻结 alpha、目标 power、备择方向、family critical value 和复杂调查设计功效方法。
- “区间较窄”不等于能够排除实质差异。例如点估计 0.30、区间 `[0.28,0.32]`、delta 0.05 会支持超出容忍范围，不能输出“precision sufficient for delta”。

## 6. 方法比较与绝对权衡输出

每个 method vs matched baseline 至少同时输出：

- gap 变化；
- 原表现较差群体的绝对性能变化；
- 其他/原表现较好群体的绝对性能变化；
- 总体 balanced accuracy 变化；
- 对 `p_event` 才适用的总体风险质量变化及其指标名称；
- 是否有其他 O 恶化；
- 互斥主分类优先级：先判定所有组受损；其次仅当 gap 缩小且弱势组有超过 tolerance 的实质改善、同时其他 O 恶化时，才叫“目标 O 改善但其他 O 恶化”；再判定弱势组改善或由较好组损失造成的 gap 缩小；否则为无预注册改善模式。连续输出始终保留为主证据。

若“原表现较差群体”在 seed/端点间改变，必须按预先声明的 baseline 定义规则处理，不能按方法结果重新命名。

## 7. anchor 与 expanded 的判断变化

`aggregate_q2_scope_conclusions` 对同一固定模型/方法执行三套输入：anchor native-family contrasts、同一 anchor contrasts 在 expanded family 下的区间/状态、以及 new-O contrasts 在 expanded family 下的状态。每次调用必须提供与 family 相同的 frozen contrast-universe receipt；anchor 两套 rows 必须逐一等于 receipt 的完整 anchor slots，new-O rows 必须逐一等于完整 new-O slots，不可估计 slot 也不能删除。native 与 expanded-family anchor 行必须绑定完全相同的 scientific contrast identity：year、evidence role、family/question、panel ID/hash、`O_train`、`O_audit`、audit role/hash、`O_group_rule_sha256`、O-registry version/hash、contrast-universe hash、group/reference、endpoint、model ID 或 method-baseline pair、seed-policy hash、delta/units/rationale 与预声明 contrast ID；只有注册的 family-dependent inference 字段允许改变，自由文本 key 不能替代该 identity。所有 anchor-expanded 与 new-O 行还必须共享 receipt/panel/family scope；new-O 只可使用 receipt 中冻结为 `expanded_candidate` 的互斥 O。scope 三态固定优先级为：

```text
SUPPORTED_EXCEEDS_TOLERANCE
> INSUFFICIENT_EVIDENCE
> SUPPORTED_WITHIN_TOLERANCE
```

因此一个 `EXCEEDS` 与任意数量 `INSUFFICIENT` 并存时，scope 结论为 `EXCEEDS`；没有 `EXCEEDS` 但存在 `INSUFFICIENT` 时不能判 within。至少保存：

```text
anchor_native_conclusion
anchor_under_expanded_family_conclusion
new_O_content_conclusion
expanded_conclusion
multiplicity_precision_change
new_O_content_change
overall_anchor_to_expanded_change
new_limiting_O
per_O_coverage
```

`multiplicity_precision_change` 比较同一 anchor contrasts 的 native family 与 expanded family 状态；`new_O_content_change` 在同一 expanded family 下比较 anchor-only 与 anchor+new-O；两者不得混称“发现了新内容”。不同 O 的 annual/domain rows、coverage fraction、NA contrasts 和 conclusion counts 分别输出，禁止全 O complete-case 聚合。

“changed”只表示按冻结规则得出的审计判断变化，不等同于发现歧视原因。若 expanded 中某 O 为 `INSUFFICIENT_EVIDENCE`，应解释精度边界，不得把它算作公平通过。

若冻结选择规则得到零个 new O，`new_O` 行可以为空，但 frozen contrast-universe receipt 的 `new_O_set` 也必须为空，并同时提供 hash-bound `FROZEN_NO_NEW_O_SELECTED` receipt。negative receipt 必须绑定同一个 contrast-universe hash、本次 Q2 shared scope、atlas 路径/hash/行数、候选 registry 路径/hash/分母、排除与不可估计数、`selected_new_O_count=0`、precision interpretation 与 evidence manifest、相同 selection-rule ID/hash、相同 O-registry version/hash 及 receipt hash；来源分母与计数必须闭合。receipt 必须明确 `PROJECT_ATLAS_NOT_CODEBOOK_WIDE_ELIGIBILITY_LEDGER` 且 `codebook_wide_eligibility_ledger_complete=false`，不能冒充全 codebook eligibility 证据。此路径下 expanded 明确等于 anchor，`new_O_content_change.changed=false`，并禁止任何肯定式“新 O 改变结论”主张；没有 receipt、scope/provenance 不匹配、receipt 被篡改或计数不一致均 fail closed。

模型选择变化属于增强实验，只允许在相同部署合同且有冻结共同域的可比集合上进行。主 panel 的科学结论不因后加入 supplemental 方法而重算。

## 8. 接受与剩余阻塞

repair tranche 1–4 已由 Worker 完成隔离实现和合成/静态测试。tranche 4 新增：hash-bound audit role/registry/group-rule identity；固定 anchor 与互斥 new-O 的 contrast-universe receipt；family/Q2 exact slot completeness；以及跨 endpoint/group/model/family 的同 O group-rule 一致性。该描述等待独立 Reviewer 重跑，不能自称 A2 accepted。

A1 尚未科学冻结，因为以下内容需要人类或统计审阅：

1. 应用场景以及各确认性 endpoint 的 delta、单位、依据与批准者；
2. 正式候选 O、分组、参照、嵌套/相关规则和允许主张的角色；
3. 少量确认性 method-baseline 对；
4. simultaneous CI 的参考实现或模拟覆盖率接受；
5. 所有 panel member 的完整 hash/threshold/seed 合同与不读结局的 2023 coverage preflight；
6. 2023 版本化 replay 的完整运行 manifest；
7. repair tranche 4 的独立 Reviewer verdict。

因此当前允许继续跨年元数据核对、family manifest 草拟和推断 QA，但不允许 A3 真实性能扫描，也不允许任何 2025 微观结局/性能访问。
