# A0-A2 repair tranche 4 Worker report v3

Gate:
`A0_A2_REPAIR_TRANCHE_4_V3`

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
- `docs/plans/nhis_bias_attribute_discovery_20260918/A0_A2_REPAIR_TRANCHE_4_REPORT_V3.md`

`INDEPENDENT_REVIEW_A0_A2_REPAIR_TRANCHE_3_V3.md`、旧 survey 实现、数据、模型、预测和受保护基线均未修改。`planning_verification_v3.json` 继续保留为已 supersede 的历史收据。

Commands executed:
- 完整读取 201 行 `INDEPENDENT_REVIEW_A0_A2_REPAIR_TRANCHE_3_V3.md`。
- 先检查共享目录中的未完成 tranche 4 修改，保留本轮正确部分并继续；未覆盖无关文件。
- 仅用 `apply_patch` 修改获准的隔离 v3 源文件、合成测试和 draft planning/receipt 文件。
- 定向运行 `PYTHONPATH=src pytest -q tests/synthetic/test_bias_attribute_audit_v3.py -x`。
- 运行 `PYTHONPATH=src pytest -q tests/synthetic/test_bias_attribute_audit_v3.py tests/synthetic/test_bias_attribute_audit_v3_registry.py -k 'not repair_receipt' -x`。
- 运行完整相关测试命令：`PYTHONPATH=src pytest -q tests/synthetic/test_bias_attribute_audit_v3.py tests/synthetic/test_bias_attribute_audit_v3_registry.py tests/benchmark/test_survey_domain_contract.py tests/benchmark/test_survey_linearization.py tests/benchmark/test_survey_sufficient_totals.py tests/test_nhis_survey.py`。
- 运行 Python compilation、严格 JSON 解析、canonical panel hash、receipt 文件绑定、SHA-256、空白和保护路径 diff 检查。

Permissions requested:
无。未联网、未下载、未启动收费服务器、未上传、未 stage、未 commit、未 reset 或 clean。

Tests executed:
- `AuditAttributeSpec` 的 audit role、role hash、group-rule hash 与 O-registry version/hash 绑定；
- 固定有序 anchor universe 与互斥 frozen new-O universe receipt；
- `scope=anchor` 仅接受完整三个 anchor，`scope=expanded` 仅接受相同 anchor 加完整 frozen new-O；
- VISIONDF expanded role 进入 anchor family、expanded scope 缺少一个成员、SEX_A 同时属于 anchor/new-O 的拒绝；
- family manifest 嵌入并验证同一 contrast-universe receipt，exact slots 包括 not-estimable slots；
- 同一 O 跨 endpoint/group/model/family 的 role/group-rule/registry identity 一致性；
- SEX_A FNR/FPR 使用不同 group-rule、重新计算 contrast ID、family ID/hash 与 manifest hash 后仍被拒绝；
- Q2 同一 receipt 的 exact anchor/new-O slots、缺失/额外 slot、anchor O 重贴 new-O、receipt 篡改的拒绝；
- no-new-O 路径要求 frozen new-O set 为空，且 negative receipt 绑定相同 universe、registry 与 selection rule；
- tranche 1–3 的 Q2、panel、family、prediction-decision、record-order、precision、tradeoff、调查域、simultaneous-CI 隔离合同与 2025 lock 回归。

Exact test results:
- 最终定向主合成套件：`70 passed in 1.99s`。
- registry/主合成预检：`76 passed, 1 deselected in 2.00s`。
- 最终完整相关集合：`100 passed, 3 warnings in 2.35s`。
- 三个 warning 仍来自既有 `src/nhis_fairbias/benchmark/survey_batch.py:60` 的 divide-by-zero、overflow 和 invalid matrix multiplication；未修改或压制旧 survey warning。
- Python compilation：PASS。
- 四个 JSON 严格解析：PASS。
- canonical panel hash、receipt 文件 hash 和报告末行检查：PASS。
- `git diff --check`：PASS。
- 当前结果只证明 Worker 环境中的隔离软件合同；不等于独立 Reviewer acceptance、production simultaneous-CI validity 或 NHIS 科学结果。

Input hashes:
- `docs/AI_EXECUTION_PROTOCOL.md`: `1a3d38929d65493f8352d6b46d32f65fa836502d08a3c2d2704dd713e5c3beac`
- `INDEPENDENT_REVIEW_A0_A2_REPAIR_TRANCHE_3_V3.md`: `c6ed5c21b3fa8de7f8a36480b8ed469619b92aad22ef2a3ddee3e5c90d2fca65`
- tranche 3 report：`a51b8593a0d992bccc12248e8b93fd7e74c0f62797fe27e507f4a8dd964cb99f`
- tranche 3 audit module：`c6903d786bb66e272f5580fff23da852b13c557bd5f8ee5226f57da0a0751273`
- tranche 3 主 synthetic test：`773f66c256d8ccd0ce8208057ced1e263c6504ef0e42c7580a2d05595f7f624f`
- tranche 3 registry test：`8cf624d1a4fb33608ccbe5ee046d42b5f86a25923858e455b25f71896e2321cd`

Output hashes:
- `bias_attribute_audit_v3.py`: `7d7961bb2697489ebbd5750d0426e1354fa034e232d38bdd11a970453703c4ea`
- `test_bias_attribute_audit_v3.py`: `c6beac8a74be0aec6abbbbe0e6fa2375122e35ae285f5936495e419a071bc64f`
- `test_bias_attribute_audit_v3_registry.py`: `f739202788f4e98befb0407c4c8a74dad2128db20d299f025855c87b88caa4b7`
- `PRE_FREEZE_REGISTRY_V3_DRAFT.json`: `ec0dd67fcc602b6337d425a742d1e9abf3d5b9245fc77ca38b6bffbfd00c7bdb`
- `A1_INTERFACE_AND_FAMILY_SPEC_V3.md`: `eea366962b7f9f6322c02de9c423bdb6970df6a10c72b95692dcbdd596cd03f6`
- `A1_FREEZE_CHECKLIST_V3_DRAFT.json`: `a6a470f61f18c82b38a69ca1688fba5fbc9648aa76665c90ac7e597208f7d0dd`
- superseded `planning_verification_v3.json`: `6813360d11a1edac8a7edd6756442f24f9c3c2656eaec0a8c754cb90fcf5d313`
- main-panel draft binding hash：`2a63adc9e43829af676570553d8dbedc7a9209ca59a832089e77479bfe6a8d0f`
- `repair_verification_v3.json` 将在本报告后单向绑定上述输出与本报告；本报告不反向记录 receipt hash，以避免循环引用。本报告也不自哈希。

Row counts:
- 真实微观数据行：0。
- 新真实性能记录：0。
- 2023 prediction replay：0。
- 2025 微观结局/性能行：0。
- 所有 contrast-universe receipts、family slots 与 no-new-O receipts 均为测试内合成对象；没有宣称真实 O/group/delta 已冻结。
- checklist：52 项，其中 32 项有 Worker 技术证据、20 项未完成；不构成 gate acceptance。

Assumptions:
- anchor universe 固定且有序为 `SEX_A`、`HISPALLP_A`、`DISAB3_A`；new-O universe 必须与其互斥并以变量名规范排序。
- audit-role hash 绑定 O 名称、role 和 O-registry identity；contrast-universe receipt 进一步把 role 与 group-rule、groups/reference、selection rule 及 panel/family identity 一起封存。
- family scope 与 Q2 rows 必须精确等于 receipt 为该 endpoint/model 或 method pair 声明的 slots；`NOT_ESTIMABLE` 不允许被删除。
- 同一 O 在全部 endpoint、group、model/method 和 family 中使用同一个冻结 role/group-rule/registry identity。
- 未知正式 new-O、groups/reference、delta、method pairs、feature/preprocessing、seed estimand、completion model hash、threshold 和 coverage 继续保持 `null/PENDING`；本轮未把合成 receipt 写成真实 frozen 决定。

Unresolved issues:
- tranche 4 尚未由独立 Reviewer 验收；A2 不得自称 accepted。
- 应用场景、delta、正式 O、分组/参照/嵌套、确认性 method-baseline pair 与允许主张仍需人类决定。
- 2022/2023 官方 codebook harmonization 与 codebook-wide eligibility ledger 未完成。
- 主 panel 成员级 feature/preprocessing/training/prediction source、completion model、threshold、prediction/`y_hat`/order hash 和 coverage preflight 未完成。
- simultaneous CI 仍缺 production 参考实现或复杂调查设计覆盖率接受。
- A3、A4、2025 解锁与投稿仍为 NO-GO。

Git diff summary:
当前分支 `research/nhis-fairbias`，HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`。本 tranche 只修改隔离 v3 源码、合成测试、draft registry/spec/checklist 和版本化 repair receipt/report 文件。Reviewer memo、旧 survey、数据、模型、预测及受保护基线无差异。未执行破坏性 Git 操作。

Proposed next step:
由独立 Reviewer 重跑 100 项完整集合与 tranche 4 对抗反例，复算 output/receipt/panel hashes，并书面判定 Accept、Repair 或 Reject。即使隔离软件合同获接受，仍须完成 A1 人类科学冻结、panel/coverage 与 production simultaneous-CI 门禁并得到新的书面授权，方可考虑 A3；当前不得生成 2023 replay、读取真实性能或访问 2025 微观结局/性能。

STOP — waiting for Codex review.
