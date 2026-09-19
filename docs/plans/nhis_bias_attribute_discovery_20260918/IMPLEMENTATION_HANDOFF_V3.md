# 执行 handoff v3：NHIS 有限属性与扩展属性公平审计

日期：2026-09-19。当前 gate：**A0–A2；A3 BLOCKED UNTIL ACCEPTANCE；2025 LOCKED**。

## 先读与优先级

按顺序阅读：

1. `docs/AI_EXECUTION_PROTOCOL.md`
2. `docs/plans/nhis_bias_attribute_discovery_20260918/RESEARCH_PLAN_V3.md`
3. 本文件
4. `PRE_FREEZE_REGISTRY_V3_DRAFT.json`
5. `A0_ASSET_INVENTORY_V3.md`
6. `DECISION_LOG_V3.md`

v2 与旧计划必须保留。v3 取代 v2 的论文主轴、统计问题分家、delta/三态、固定 panel、precision、A3.5 和科学 decision log 规则；v2 的 O_train/O_audit、逐 O 域、2024 已知、2025 锁、调查设计、哈希/对齐、非因果措辞和 FairBias 不必获胜等底线继续有效。

## 本 gate 允许

- 只读盘点本地 2022–2024 源、处理后数据、manifest、模型/预测登记与 2024 官方元数据；
- 新增 v3 计划、registry、atlas、inventory、decision log、隔离代码与合成测试；
- 运行不读取真实微观数据/性能的合成和静态测试；
- 核验关键文件 SHA-256、Git branch/HEAD、受保护文件 diff 和记录顺序合同。

## 本 gate 禁止

- A0–A2 验收前运行 2023/2024 真实性能扫描；
- 读取或生成任何 2025 微观结局/性能；
- 修改旧四臂入口来迁就新 O、重训或覆盖历史预测；
- 把历史 2024 结果当未见验证；
- 根据结果选择 delta、类别切点、参照组、候选 O、模型或检验族；
- reset/clean/覆盖/提交/上传、启动或连接收费服务器、输出个体记录。

## A0–A2 交付

1. `RESEARCH_PLAN_V3.md` 与本 handoff；
2. `A0_ASSET_INVENTORY_V3.md`：分清当前验证事实、历史报告和缺口；
3. `VARIABLE_ATLAS_V3_DRAFT.csv`：官方 2024 元数据证据与跨年未决项；
4. `PRE_FREEZE_REGISTRY_V3_DRAFT.json`：anchor/expanded、候选 O、delta 状态、固定 panel、家族模板和 2025 锁；
5. `DECISION_LOG_V3.md`：执行验收与科学决定分离；
6. `A1_INTERFACE_AND_FAMILY_SPEC_V3.md`：anchor/expanded 输入输出、逐 O 域和 family schema；
7. `src/nhis_fairbias/bias_attribute_audit_v3.py`；
8. `tests/synthetic/test_bias_attribute_audit_v3.py` 与 registry 静态合同测试；
9. `A0_A2_GATE_REPORT_V3.md` 与 `planning_verification_v3.json`。

## A2 接受标准

- exact-order alignment 对重排、重复、缺失、预期哈希不符 fail closed，错误不泄露 ID；
- 每个 O 单独形成 panel 共同域，不存在所有 O complete-case 路径；
- 主 panel 一旦冻结不可添加成员；后加入方法必须是带 parent hash 的 supplemental panel；
- Q1 `d=0` 与 Q3/Q4 `d_method-d_baseline=0` 由生成器产生不同 family/contrast IDs；
- Holm 只生成调整 p 值；普通 CI 与 simultaneous CI 分字段存储；NA 保留家族分母；
- 最大 EO/gap 从完整预声明组件同时区间投影，拒绝单个事后最大对；
- 三态结论与 precision/detectable-effect 结果可独立复算；
- 方法权衡至少能区分四类必需情形并保留绝对性能变化；
- 调查域测试在完整设计上保留零贡献 PSU；
- 未满足全部 freeze 和人类 release 字段时，2025 access 始终 fail closed；
- 相关合成测试全部通过，且旧调查推断相关回归未受影响。

## A0–A2 后的停止点

主管可以自主验收软件/哈希/对齐。若任何合同失败，修复仍限制在 A0–A2。只有 A0–A2 通过、delta 与正式候选 O 的人类事实进入 decision log 后，才生成 A3 run specification。不得因“代码已写完”自动进入真实扫描。

Gate:
A0-A2_NHIS_BIAS_ATTRIBUTE_AUDIT_V3
Status:
A0_PASS_WITH_LIMITS__A1_SCIENTIFIC_FREEZE_PENDING__A2_ACCEPTED_FOR_SYNTHETIC_CONTRACTS__A3_NOT_AUTHORIZED
Files changed:
See current Git diff limited to v3 additive paths.
Commands executed:
Record exact commands in the gate report.
Permissions requested:
None; local-only, no paid server.
Tests executed:
Targeted v3 synthetic/static tests and relevant existing survey tests; optional catalog suites were also probed and stopped at collection because `aif360` is absent.
Exact test results:
40 passed, 3 warnings; optional catalog probes did not collect because `aif360` is not installed.
Input hashes:
See `A0_ASSET_INVENTORY_V3.md`.
Output hashes:
See `planning_verification_v3.json`.
Row counts:
No real microdata rows processed in A0-A2.
Assumptions:
2024 is known retrospective evidence; 2025 remains locked.
Unresolved issues:
Application-specific delta and human scientific decisions.
Git diff summary:
Additive v3 files only.
Proposed next step:
Obtain the concentrated human decisions in `DECISION_LOG_V3.md`, finish cross-year metadata/family freeze and reference-validation of simultaneous CI; then issue a new A3 gate decision.
STOP — A3 remains blocked; 2025 remains locked.
