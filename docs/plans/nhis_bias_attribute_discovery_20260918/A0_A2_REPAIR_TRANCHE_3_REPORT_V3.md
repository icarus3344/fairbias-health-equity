# A0-A2 repair tranche 3 Worker report v3

Gate:
`A0_A2_REPAIR_TRANCHE_3_V3`

Status:
`WORKER_IMPLEMENTED__INDEPENDENT_REVIEW_REQUIRED__A2_NOT_SELF_ACCEPTED__A3_NOT_AUTHORIZED__2025_LOCKED`

Files changed:
- `src/nhis_fairbias/bias_attribute_audit_v3.py`
- `tests/synthetic/test_bias_attribute_audit_v3.py`
- `tests/synthetic/test_bias_attribute_audit_v3_registry.py`
- `docs/plans/nhis_bias_attribute_discovery_20260918/PRE_FREEZE_REGISTRY_V3_DRAFT.json`
- `docs/plans/nhis_bias_attribute_discovery_20260918/A1_INTERFACE_AND_FAMILY_SPEC_V3.md`
- `docs/plans/nhis_bias_attribute_discovery_20260918/A1_FREEZE_CHECKLIST_V3_DRAFT.json`
- `docs/plans/nhis_bias_attribute_discovery_20260918/repair_verification_v3.json`
- `docs/plans/nhis_bias_attribute_discovery_20260918/A0_A2_REPAIR_TRANCHE_3_REPORT_V3.md`

`INDEPENDENT_REVIEW_A0_A2_REPAIR_TRANCHE_2_V3.md`、旧 survey 实现、数据、模型、预测和受保护基线均未修改。`planning_verification_v3.json` 保留为已 supersede 的历史收据，本 tranche 未改写。

Commands executed:
- 完整读取 183 行 `INDEPENDENT_REVIEW_A0_A2_REPAIR_TRANCHE_2_V3.md`。
- 仅用 `apply_patch` 修改获准的隔离 v3 源文件、合成测试和 draft planning/receipt 文件。
- 定向运行 `PYTHONPATH=src pytest -q tests/synthetic/test_bias_attribute_audit_v3.py -x`。
- 运行 `PYTHONPATH=src pytest -q tests/synthetic/test_bias_attribute_audit_v3.py tests/synthetic/test_bias_attribute_audit_v3_registry.py -k 'not repair_receipt' -x`。
- 运行完整相关测试命令：`PYTHONPATH=src pytest -q tests/synthetic/test_bias_attribute_audit_v3.py tests/synthetic/test_bias_attribute_audit_v3_registry.py tests/benchmark/test_survey_domain_contract.py tests/benchmark/test_survey_linearization.py tests/benchmark/test_survey_sufficient_totals.py tests/test_nhis_survey.py`。
- 运行 Python compilation、严格 JSON 解析、canonical panel hash、receipt 文件绑定、SHA-256、空白和保护路径 diff 检查。

Permissions requested:
无。未联网、未下载、未启动收费服务器、未上传、未 stage、未 commit、未 reset 或 clean。

Tests executed:
- Q2 每行 `O_group_rule_sha256` 和 native/expanded-family 同一 slot identity；
- anchor 与全部 new-O rows 的 year/evidence role、panel ID/hash、`O_train`、model 或 method-baseline pair、seed-policy hash、delta units/rationale shared-scope identity；
- 跨年、panel、model、seed、delta 和 group-rule 漂移的对抗拒绝；
- no-new-O receipt 的 shared scope、atlas/candidate registry、selection rule、O registry 和 precision evidence provenance 绑定，以及缺失、篡改、计数不闭合反例；
- main panel 的 `event_probability_p`、`O_at_inference=false`、`per_O`、all-members、annual design-row retention 和禁止全 O complete-case 的严格合同；
- family generator/verifier 共用的 frozen seed-policy 语义，包括空/重复/bool/非整数 seed、缺失 estimand 和 self-consistent hollow manifest 反例；
- threshold policy 精确 schema、非空 policy ID、无结果访问 selection role，以及 panel/member/model/O identifiers 的字符串约束；
- tranche 1/2 的 Q2、panel、family、prediction-decision、record-order、precision、tradeoff、调查域、simultaneous-CI 隔离合同和 2025 lock 回归。

Exact test results:
- 最终定向主合成套件：`63 passed in 1.62s`。
- registry/主合成预检：`69 passed, 1 deselected in 1.52s`。
- 最终完整相关集合：`93 passed, 3 warnings in 1.57s`。
- 三个 warning 仍来自既有 `src/nhis_fairbias/benchmark/survey_batch.py:60` 的 divide-by-zero、overflow 和 invalid matrix multiplication；未修改或压制旧 survey warning。
- Python compilation：PASS。
- 四个 JSON 严格解析：PASS。
- canonical panel hash、receipt 文件 hash 和报告末行检查：PASS。
- `git diff --check`：PASS。
- 当前结果只证明 Worker 环境中的隔离软件合同；不等于独立 Reviewer acceptance、production simultaneous-CI validity 或 NHIS 科学结果。

Input hashes:
- `docs/AI_EXECUTION_PROTOCOL.md`: `1a3d38929d65493f8352d6b46d32f65fa836502d08a3c2d2704dd713e5c3beac`
- `INDEPENDENT_REVIEW_A0_A2_REPAIR_TRANCHE_2_V3.md`: `09d8cdc1a7a17e489c3fb0374b7e8725df808cff709825132112cadc93be22e7`
- tranche 2 report：`dcb9e7a276b88e8c5da3fefc9a20a6d54356e847b8be8ed310c479b8467aa700`
- tranche 2 audit module：`0ad71d0cb31c375fa8195c80384d486dbf24f47c897b5eba7057debf93ef8749`
- tranche 2 主 synthetic test：`f3314d7d1089668c021c82ba83ec808bf4e1ad1fae6b5617a1126f278afd96d6`
- tranche 2 registry test：`5f5fe644c94c8411ba7081804cd63669a2c7549b2672c88a366d50cb2ade0dea`

Output hashes:
- `bias_attribute_audit_v3.py`: `c6903d786bb66e272f5580fff23da852b13c557bd5f8ee5226f57da0a0751273`
- `test_bias_attribute_audit_v3.py`: `773f66c256d8ccd0ce8208057ced1e263c6504ef0e42c7580a2d05595f7f624f`
- `test_bias_attribute_audit_v3_registry.py`: `8cf624d1a4fb33608ccbe5ee046d42b5f86a25923858e455b25f71896e2321cd`
- `PRE_FREEZE_REGISTRY_V3_DRAFT.json`: `f39499904645a634557ac021a789b436b389f7b1e33b3ee54fd260f463a7b945`
- `A1_INTERFACE_AND_FAMILY_SPEC_V3.md`: `cd07533271ccd39a64e36c0bf96ed3c3fff70403335b051743a2905c1ffee28e`
- `A1_FREEZE_CHECKLIST_V3_DRAFT.json`: `d5f821deb4e37d7e70dee19d472e969984618cdd08eb52efe60558ba95d8cc6d`
- superseded `planning_verification_v3.json`: `6813360d11a1edac8a7edd6756442f24f9c3c2656eaec0a8c754cb90fcf5d313`
- main-panel draft binding hash：`2a63adc9e43829af676570553d8dbedc7a9209ca59a832089e77479bfe6a8d0f`
- `repair_verification_v3.json` 将在本报告后单向绑定上述输出与本报告；本报告不反向记录 receipt hash，以避免循环引用。本报告也不自哈希。

Row counts:
- 真实微观数据行：0。
- 新真实性能记录：0。
- 2023 prediction replay：0。
- 2025 微观结局/性能行：0。
- no-new-O receipt 只使用测试内合成 source 与计数；没有把 project atlas 误称为 codebook-wide eligibility ledger。
- checklist：48 项，其中 28 项有 Worker 技术证据、20 项未完成；不构成 gate acceptance。

Assumptions:
- 同一次 Q2 aggregation 只比较同一固定年度、evidence role、panel、`O_train`、endpoint、model 或 method-baseline pair、seed policy 与 delta rationale；new-O 只允许审计属性、组/参照、group-rule 和相应 contrast ID 改变。
- `O_group_rule_sha256` 绑定每个 O 的已冻结分组/缺失/NIU 规则；同一 O 内必须一致，不同 O 可不同。
- no-new-O receipt 绑定 project-atlas/candidate-registry 范围、selection rule、O registry 和 precision evidence；它明确不证明全 codebook eligibility ledger 已完成。
- 主 risk panel 只能输出 `event_probability_p`，不在 inference 使用 O，且所有成员必须遵守逐 O 域、年度 design-row 保留和禁止全 O complete-case 合同。
- 未知 feature/preprocessing、seed estimand、completion model hash、threshold 和 coverage 继续保持 `null/PENDING`；draft panel 未伪装为 frozen。

Unresolved issues:
- tranche 3 尚未由独立 Reviewer 验收；A2 不得自称 accepted。
- 应用场景、delta、正式 O、分组/参照/嵌套、确认性 method-baseline pair 与允许主张仍需人类决定。
- 2022/2023 官方 codebook harmonization 与 codebook-wide eligibility ledger 未完成。
- 主 panel 成员级 feature/preprocessing/training/prediction source、completion model、threshold、prediction/`y_hat`/order hash 和 coverage preflight 未完成。
- simultaneous CI 仍缺 production 参考实现或复杂调查设计覆盖率接受。
- A3、A4、2025 解锁与投稿仍为 NO-GO。

Git diff summary:
当前分支 `research/nhis-fairbias`，HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`。本 tranche 只修改隔离 v3 源码、合成测试、draft registry/spec/checklist 和版本化 repair receipt/report 文件。Reviewer memo、旧 survey、数据、模型、预测及受保护基线无差异。未执行破坏性 Git 操作。

Proposed next step:
由独立 Reviewer 重跑 93 项完整集合与 tranche 3 对抗反例，复算 output/receipt/panel hashes，并书面判定 Accept、Repair 或 Reject。即使隔离软件合同获接受，仍须完成 A1 人类科学冻结、panel/coverage 与 production simultaneous-CI 门禁并得到新的书面授权，方可考虑 A3；当前不得生成 2023 replay、读取真实性能或访问 2025 微观结局/性能。

STOP — waiting for Codex review.
