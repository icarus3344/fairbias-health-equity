# A0-A2 repair tranche 2 Worker report v3

Gate:
`A0_A2_REPAIR_TRANCHE_2_V3`

Status:
`WORKER_IMPLEMENTED__INDEPENDENT_REVIEW_REQUIRED__A2_NOT_SELF_ACCEPTED__A3_NOT_AUTHORIZED__2025_LOCKED`

Files changed:
- `src/nhis_fairbias/bias_attribute_audit_v3.py`
- `tests/synthetic/test_bias_attribute_audit_v3.py`
- `tests/synthetic/test_bias_attribute_audit_v3_registry.py`
- `docs/plans/nhis_bias_attribute_discovery_20260918/PRE_FREEZE_REGISTRY_V3_DRAFT.json`
- `docs/plans/nhis_bias_attribute_discovery_20260918/A1_INTERFACE_AND_FAMILY_SPEC_V3.md`
- `docs/plans/nhis_bias_attribute_discovery_20260918/A1_FREEZE_CHECKLIST_V3_DRAFT.json`
- `docs/plans/nhis_bias_attribute_discovery_20260918/planning_verification_v3.json`
- `docs/plans/nhis_bias_attribute_discovery_20260918/repair_verification_v3.json`
- `docs/plans/nhis_bias_attribute_discovery_20260918/A0_A2_REPAIR_TRANCHE_2_REPORT_V3.md`

`INDEPENDENT_REVIEW_A0_A2_REPAIR_V3.md` 未修改。旧 survey 实现、真实性能、2025 微观结局/性能、模型、预测和受保护基线均未读取或修改。

Commands executed:
- 完整读取 149 行 `INDEPENDENT_REVIEW_A0_A2_REPAIR_V3.md`。
- 仅用 `apply_patch` 修改获准源文件、合成测试和 v3 planning/receipt 文件。
- 定向运行 `PYTHONPATH=src pytest -q tests/synthetic/test_bias_attribute_audit_v3.py -x`。
- 运行 `PYTHONPATH=src pytest -q tests/synthetic/test_bias_attribute_audit_v3.py tests/synthetic/test_bias_attribute_audit_v3_registry.py -k 'not repair_receipt' -x`。
- 运行完整相关测试命令：`PYTHONPATH=src pytest -q tests/synthetic/test_bias_attribute_audit_v3.py tests/synthetic/test_bias_attribute_audit_v3_registry.py tests/benchmark/test_survey_domain_contract.py tests/benchmark/test_survey_linearization.py tests/benchmark/test_survey_sufficient_totals.py tests/test_nhis_survey.py`。
- 运行 Python compilation、严格 JSON 解析、canonical panel hash、SHA-256、空白和保护路径 diff 检查。

初次定向运行得到 `2 passed, 1 failed`：篡改 slot 同时触发了更早的 `contrast_id mismatch`，而测试原先只接受稍后的 `family_sha256 mismatch`。实现已正确 fail closed；测试改为验证实际的更早拒绝点后，完整定向套件重跑通过。该失败未计为通过。

Permissions requested:
无。未联网、未下载、未启动收费服务器、未上传、未 stage、未 commit、未 reset 或 clean。

Tests executed:
- Q2 native 与 expanded-family anchor 行的完整 scientific contrast identity；
- endpoint、model、panel 与 delta 身份漂移的对抗拒绝；
- hash-bound no-new-O 负结果路径，以及 receipt 缺失、篡改、计数不闭合拒绝；
- 任意 frozen-panel 输入的完整无副作用重验证和 forged empty panel 拒绝；
- member/deployment 的 backbone、feature/preprocessing、output、`O_at_inference`、threshold 一致性；
- 结构化 source `path + sha256` 和跨成员/seed model ID 唯一性；
- family schema/status/精确 family 集、manifest identity、family sizes、slot count、contrast ID、null/status 与顶层 manifest hash；
- 顶层 year、family size 和未重哈希篡改反例；
- tranche 1 的 threshold、record-order、precision、tradeoff、调查域、simultaneous-CI 隔离合同与 2025 lock 回归。

Exact test results:
- 最终定向主合成套件：`36 passed in 1.49s`。
- registry/主合成预检：`42 passed, 1 deselected in 1.74s`。
- 最终完整相关集合：`66 passed, 3 warnings in 1.73s`。
- 三个 warning 仍来自既有 `src/nhis_fairbias/benchmark/survey_batch.py:60` 的 divide-by-zero、overflow 和 invalid matrix multiplication；未修改或压制旧 survey warning。
- Python compilation：PASS。
- 四个 JSON 严格解析：PASS。
- `git diff --check`：PASS。
- 当前结果只证明 Worker 环境的隔离软件合同；不等于独立 Reviewer acceptance、production simultaneous-CI validity 或 NHIS 科学结果。

Input hashes:
- `docs/AI_EXECUTION_PROTOCOL.md`: `1a3d38929d65493f8352d6b46d32f65fa836502d08a3c2d2704dd713e5c3beac`
- `INDEPENDENT_REVIEW_A0_A2_REPAIR_V3.md`: `829355f078401416864cdc019f51e32b3e339939ab662f99505a415dd3ef8030`
- tranche 1 report：`34a40c571fcbb318818521c16a4577e0d7fbcecf8d45870c119a22724199d03d`
- tranche 1 audit module：`e0496fb22ee8654f2ef81538cadb7171514a3b5b39b17437fd0680d73444b09b`
- tranche 1 主 synthetic test：`d5df6851b0af2be8b15e4cee69102c59c8546c7fe520d85bfb6f810a98479c00`
- tranche 1 registry test：`44d793d68d36b8f75902aee49e6381e8ff3581510627e39de4ab9088b373f1f6`

Output hashes:
- `bias_attribute_audit_v3.py`: `0ad71d0cb31c375fa8195c80384d486dbf24f47c897b5eba7057debf93ef8749`
- `test_bias_attribute_audit_v3.py`: `f3314d7d1089668c021c82ba83ec808bf4e1ad1fae6b5617a1126f278afd96d6`
- `test_bias_attribute_audit_v3_registry.py`: `5f5fe644c94c8411ba7081804cd63669a2c7549b2672c88a366d50cb2ade0dea`
- `PRE_FREEZE_REGISTRY_V3_DRAFT.json`: `fe9b971cba4cedf549a821ca7ca2a4f485a5d9849906b1bd745dead5c5f0958d`
- `A1_INTERFACE_AND_FAMILY_SPEC_V3.md`: `70922c8ccd1f5b285f548699eb05749a96c7e2ca1b4126ba797e5727f54b8b9c`
- `A1_FREEZE_CHECKLIST_V3_DRAFT.json`: `0e572c0e3f1dedb0f048fc1d215a6ec05eabc6299a17afe46314c410bbf27906`
- superseded `planning_verification_v3.json`: `6813360d11a1edac8a7edd6756442f24f9c3c2656eaec0a8c754cb90fcf5d313`
- main-panel draft binding hash：`d91130f79576c9f7d87d5b010ee563d6363045ca6b8b2a94a27d2de3900930ee`
- `repair_verification_v3.json` 将在本报告后单向绑定上述输出与本报告；本报告不反向记录 receipt hash，以避免循环引用。本报告也不自哈希。

Row counts:
- 真实微观数据行：0。
- 新真实性能记录：0。
- 2023 prediction replay：0。
- 2025 微观结局/性能行：0。
- no-new-O 合成 receipt 使用 atlas 31、候选 24、排除 20、不可估计 4，仅为测试计数。
- checklist：43 项，其中 23 项有 Worker 技术证据、20 项未完成；不构成 gate acceptance。

Assumptions:
- Q2 行的 `contrast_id` 必须与 `contrast_key` 相同；native 与 expanded-family 只允许注册的 family-dependent inference 字段变化。
- no-new-O receipt 的候选计数必须满足 `excluded + not_estimable + selected = candidate_denominator`，且 `selected=0`。
- 主 risk panel 的冻结合同要求 `event_probability_p`、`O_at_inference=false`、逐成员冻结阈值，并采用跨成员与 seed 的 model ID 全局唯一策略。
- registry 中未知 feature/preprocessing、seed estimand、completion model hash、threshold 和 coverage 继续保持 `null/PENDING`，draft panel 未伪装为 frozen。
- family generator 当前生成未估计 manifest，故 verifier 要求 `PREDECLARED_NOT_YET_ESTIMATED` 与 `estimate=null`；未来估计结果必须使用另行版本化的结果 schema。

Unresolved issues:
- tranche 2 尚未由独立 Reviewer 验收；A2 不得自称 accepted。
- 应用场景、delta、正式 O、分组/参照/嵌套、确认性 method-baseline pair 与允许主张仍需人类决定。
- 2022/2023 官方 codebook harmonization 与 codebook-wide eligibility ledger 未完成。
- 主 panel 成员级 feature/preprocessing/training/prediction source、completion model、threshold、prediction/`y_hat`/order hash 和 coverage preflight 未完成。
- simultaneous CI 仍缺 production 参考实现或复杂调查设计覆盖率接受。
- A3、A4、2025 解锁与投稿仍为 NO-GO。

Git diff summary:
当前分支 `research/nhis-fairbias`，HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`。本 tranche 只修改隔离 v3 源码、合成测试、draft registry/spec/checklist 和版本化 verification/report 文件。Reviewer memo、旧 survey、数据、模型、预测及受保护基线无差异。未执行破坏性 Git 操作。

Proposed next step:
由独立 Reviewer 重新运行 66 项完整集合和本轮对抗反例，复算 output/receipt/panel hashes，并书面判定 Accept、Repair 或 Reject。只有独立接受软件合同且 A1 人类科学冻结、panel/coverage 与 simultaneous-CI production 门禁另行完成后，主管才能考虑 A3；当前不得生成 2023 replay、读取真实性能或访问 2025 微观结局/性能。

STOP — waiting for Codex review.
