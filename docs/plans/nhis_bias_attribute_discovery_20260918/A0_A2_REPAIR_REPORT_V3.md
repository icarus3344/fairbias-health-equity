# A0-A2 repair tranche 1 Worker report v3

Gate:
`A0_A2_REPAIR_TRANCHE_1_V3`

Status:
`WORKER_IMPLEMENTED__INDEPENDENT_REVIEW_REQUIRED__A2_REPAIR_REMAINS_OPEN__A3_NOT_AUTHORIZED__2025_LOCKED`

Files changed:
- `src/nhis_fairbias/bias_attribute_audit_v3.py`
- `tests/synthetic/test_bias_attribute_audit_v3.py`
- `tests/synthetic/test_bias_attribute_audit_v3_registry.py`
- `docs/plans/nhis_bias_attribute_discovery_20260918/PRE_FREEZE_REGISTRY_V3_DRAFT.json`
- `docs/plans/nhis_bias_attribute_discovery_20260918/A1_INTERFACE_AND_FAMILY_SPEC_V3.md`
- `docs/plans/nhis_bias_attribute_discovery_20260918/A1_FREEZE_CHECKLIST_V3_DRAFT.json`
- `docs/plans/nhis_bias_attribute_discovery_20260918/A0_A2_REPAIR_REPORT_V3.md`

`INDEPENDENT_REVIEW_V3_GOAL.md` 未修改。旧 survey 实现、受保护根文件、数据、模型、预测和历史结果均未修改。

Commands executed:
- 完整读取 `INDEPENDENT_REVIEW_V3_GOAL.md`。
- 使用 `apply_patch` 修改上述获准文件。
- `PYTHONPATH=src pytest -q tests/synthetic/test_bias_attribute_audit_v3.py tests/synthetic/test_bias_attribute_audit_v3_registry.py`
- `PYTHONPATH=src pytest -q tests/synthetic/test_bias_attribute_audit_v3.py tests/synthetic/test_bias_attribute_audit_v3_registry.py tests/benchmark/test_survey_domain_contract.py tests/benchmark/test_survey_linearization.py tests/benchmark/test_survey_sufficient_totals.py tests/test_nhis_survey.py`
- `python3 -m py_compile src/nhis_fairbias/bias_attribute_audit_v3.py tests/synthetic/test_bias_attribute_audit_v3.py tests/synthetic/test_bias_attribute_audit_v3_registry.py`
- `jq empty` 检查 registry、checklist 和既有 planning verification JSON。
- 运行 SHA-256、branch/HEAD、保护文件 diff 和静态合同检查。

一次未设置 `PYTHONPATH=src` 的定向 pytest 在 collection 阶段因找不到 `nhis_fairbias` 失败，未执行测试；随后使用登记环境完整重跑。一次通过 Homebrew `python3` 导入 package 计算 panel hash 的只读辅助命令因该解释器缺少 NumPy 失败；随后使用标准库 canonical JSON 计算，同一算法由 registry 静态测试复核。这两个失败均未作为通过结果计数。

Permissions requested:
无。未联网、未下载、未启动或连接服务器、未上传、未提交。

Tests executed:
- record key 严格字符串 schema、跨类型碰撞拒绝与 exact-order hash；
- Q2 scope aggregator 的三态优先级、new-O 内容变化、multiplicity/precision 变化、逐 O coverage 和 anchor 域漂移拒绝；
- `p_event -> frozen threshold -> binary y_hat` 以及 prediction/threshold/`y_hat`/order hash；
- FNR/FPR/BA 与风险质量输入语义分离；
- Q1 与 Q3/Q4 family 分离，year/evidence role/scope/panel hash/`O_train`/seed policy/family ID/hash/NA slots；
- family hash 防篡改；
- precision margin-of-error 与 delta 排除语义；
- tradeoff 重叠真值表；
- 完整 panel member freeze schema 与 pending registry 状态；
- 逐 O 调查域、调查 linearization/sufficient totals 和 2025 fail-closed lock。

Exact test results:
- 定向 v3 测试：`28 passed in 1.62s`。
- 完整相关集合：`51 passed, 3 warnings in 1.74s`。
- 三个 warning 均来自既有 `src/nhis_fairbias/benchmark/survey_batch.py:60` 的 divide-by-zero、overflow 和 invalid matrix multiplication；测试通过，warning 未隐藏。
- Python compilation：PASS。
- JSON strict parse：PASS。
- 当前结果只证明 Worker 环境中的软件合同，不是独立 Reviewer acceptance，也不是 NHIS 科学结果。

Input hashes:
- `docs/AI_EXECUTION_PROTOCOL.md`: `1a3d38929d65493f8352d6b46d32f65fa836502d08a3c2d2704dd713e5c3beac`
- `INDEPENDENT_REVIEW_V3_GOAL.md`: `564835cf86a5334d382c8a0e40f166449b127fe09469d382d0320e3e027ce5c9`
- repair 前 registry：`fa7c9d34089058972f49e857b49135af605be8925f3ed32fce47bb9bfb1b012e`
- repair 前 audit module：`80772b23ee2de7ee123b74ea3f5821e12da3a2726a349a53f80cee11121230bf`
- repair 前主 synthetic test：`bc0ce66c61c64495040f6ccfa9ac7bfbf61308b9b94b3fa7b6dd63943909a0ca`
- repair 前 registry test：`b6ccaafcf63aab378cb723b1cee7eda82747eb1196236ba7e7cc51d88eae7b6b`

Output hashes:
- `bias_attribute_audit_v3.py`: `e0496fb22ee8654f2ef81538cadb7171514a3b5b39b17437fd0680d73444b09b`
- `test_bias_attribute_audit_v3.py`: `d5df6851b0af2be8b15e4cee69102c59c8546c7fe520d85bfb6f810a98479c00`
- `test_bias_attribute_audit_v3_registry.py`: `44d793d68d36b8f75902aee49e6381e8ff3581510627e39de4ab9088b373f1f6`
- `PRE_FREEZE_REGISTRY_V3_DRAFT.json`: `39c1ab86ef291ecca545a55c61ea411ace33f162fb846c365f64ffc2faf0ee74`
- `A1_INTERFACE_AND_FAMILY_SPEC_V3.md`: `57dceaab2bff3d4fcd05b4013c1e9d549bf3082ed1757cdaa989779f188dd924`
- `A1_FREEZE_CHECKLIST_V3_DRAFT.json`: `aa553140ff2a2d6fb59bc722f9828896da81ffee1937b2a30ef301234c253c7c`
- main panel binding hash：`2875bc7c5961f70c95ad02d1d77adfa98e097d322b10e39a50ec45d161dd4b2e`
- 本报告不自哈希，避免自引用。

Row counts:
- 真实微观数据行：0。
- 新真实性能记录：0。
- 2023 prediction replay：0。
- 2025 微观结局/性能行：0。
- Q2 合成反例使用的最大 annual rows 标签：100，仅为内存构造计数，不是 NHIS 记录。
- checklist：38 项，其中 18 项具备 Worker 技术证据、20 项未完成；该计数不构成 gate acceptance。

Assumptions:
- scope 优先级预注册为 `EXCEEDS > INSUFFICIENT > WITHIN`，其科学采用仍需 Reviewer 和责任人确认。
- `critical x SE` 是 margin of error，不是 MDE；本 tranche 不实现 power/MDE。
- 主 risk panel 的 `O_train=SEX_A` 来自 Arm001 源规范；三个 parent comparator 的 model SHA 来自 admission prediction refs。
- 两个 completion 成员的 model SHA，以及全部成员的 seed estimand、feature/preprocessing/training-source/prediction-source hash、threshold hash 和 2023 coverage 尚不能完整证明，故保持 `null/PENDING`。
- main panel 当前只是技术绑定，不是 frozen，也不授权 replay 或性能扫描。

Unresolved issues:
- 应用场景、delta、正式 O、分组/参照/嵌套、允许主张和确认性 method-baseline 对仍需人类决定。
- 2022/2023 官方 codebook harmonization 与 codebook-wide eligibility ledger 未完成。
- seed 聚合/训练随机性 estimand 未冻结。
- completion model artifact hashes、全成员 feature/preprocessing/training-source/prediction-source hashes、冻结 thresholds、2023 prediction/`y_hat`/order hashes和非性能 coverage preflight 未完成。
- simultaneous CI 仍需独立参考实现和设计覆盖率 QA；本 tranche 只保持其 production blocker。
- repair tranche 1 尚未由独立 Reviewer 验收；A2、A3、A4、2025 和投稿均不得开放。

Git diff summary:
当前分支 `research/nhis-fairbias`，HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`。本 tranche 只改上述隔离 v3 文件并新增本报告。受保护 14 个根文件与 `.gitignore` 的当前 diff 为空。未 reset、clean、stage、commit 或 upload。

Proposed next step:
由独立 Reviewer 重新运行 51 项集合，并检查 Q2 反例、threshold semantics、family/panel hash、防篡改、pending 字段和 2025 lock。若判 Repair，则继续限定修复；只有新的书面 Accept 且人类科学决定完成后，才可考虑 A3 run specification。当前不得生成 2023 replay 或读取真实性能。

STOP — waiting for Codex review.
