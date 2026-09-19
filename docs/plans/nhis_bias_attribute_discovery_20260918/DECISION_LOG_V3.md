# NHIS 扩展公平审计 decision log v3

本表把可由主管依据代码/测试/哈希验收的执行决定，与需要领域、伦理、作者或数据权限事实的科学决定分开。未决科学决定不阻止独立的 A0–A2 合成开发，但会阻止相应 freeze、A3/A4 或 2025 release。

| ID | 类型 | 决定 | 当前状态 | 证据/需要的事实 | 权限/责任人 | 后续动作 |
|---|---|---|---|---|---|---|
| D-V3-001 | 已确认研究方向 | 主轴改为有限属性审计是否高估方法有效性；anchor 与 expanded 对同一固定模型集合比较 | APPROVED_BY_USER | 2026-09-19 用户指令与 `RESEARCH_PLAN_V3.md` | 项目负责人 | 作为 v3 主轴，不因结果改写 |
| D-V3-002 | 执行验收 | A0 关键路径、SHA-256、保护文件和 2025 锁状态盘点 | PASS_WITH_LIMITS | `A0_ASSET_INVENTORY_V3.md`；跨年全面元数据仍缺 | Codex supervisor | A1 补跨年 metadata，不触发性能 |
| D-V3-003 | 执行验收 | 隔离 A2 alignment/hash/domain/panel/family/CI/lock 合同 | ACCEPTED_FOR_SYNTHETIC_CONTRACTS | 40 项 v3/相关调查测试通过；3 个既有数值警告；不能推导真实科学结论 | Codex supervisor | 保持隔离；A3 前完成 simultaneous CI 参考/覆盖验证 |
| D-V3-004 | 科学决定 | FNR/FPR/BA/风险质量的实质 delta 与用途依据 | HUMAN_DECISION_REQUIRED | 需要应用场景、错误代价、受影响对象、依据与批准者；禁止看结果后选择 | 项目负责人 + 领域合作者 | 集中给出并签入 registry；此前不能 freeze/A3 confirmation/A5 |
| D-V3-005 | 科学决定 | 正式新 O、分组、参照、anchor 嵌套处理及 clinical/context/process 身份 | HUMAN_REVIEW_REQUIRED | 24 个正式种子和 7 个 atlas-only 2024 扩展已映射；2022/2023 跨年定义、领域规范与关联结构待审 | 项目负责人 + 领域合作者 | A1 审阅；正式候选最多 30、最多 5 新 O 验证 |
| D-V3-006 | 执行验收 | 固定 main risk panel 的成员和共同域合同 | TECHNICALLY_FROZEN | Arm001/GBDT/p_event 五类方法与 admission manifest 逐项匹配；panel hash `19ae854b...73c7f`；2023 预测与逐 O 实际域仍未生成 | Codex supervisor | 成员不可再缩减；后加方法只进 supplemental panel；此冻结不授权 A3 |
| D-V3-007 | 统计决定 | Q1 与 Q3/Q4 独立家族；Q1 优先确认性；少量方法对确认性 | STRUCTURE_APPROVED_CONTENT_PENDING | family generator 已实现；正式 O、参照、模型/方法对和 delta 未冻结 | 项目负责人 + statistician/author | A1 输出完整 family manifest 后 freeze |
| D-V3-008 | 统计执行 | simultaneous CI 方法 | A2_CANDIDATE_NOT_PRODUCTION_FROZEN | max-standardized共同复制实现与 Taylor/投影接口；需覆盖/参考交叉验证 | Codex supervisor + statistical reviewer | A3 前完成参考验证；Holm 不代替 simultaneous CI |
| D-V3-009 | 阶段决定 | 是否开启 A4 针对性新 O 训练 | DEFERRED_UNTIL_A3_5 | 需要 Methods 草稿、O 流程、首图和 claim-evidence 表 | 项目负责人/作者 | A3.5 审阅后记录 GO/NO-GO；A4 非必需 |
| D-V3-010 | 数据 release | 2025 微观结局/性能开放 | LOCKED | 所有 O/delta/model/panel/metric/family/source/hash/order 必须冻结；还需人类一次性 release | 项目负责人 + 数据责任人 + Codex supervisor | 未满足任一字段均 fail closed |
| D-V3-011 | 伦理/作者 | 伦理、作者责任、AI 披露、署名/资助/冲突和受限数据权限 | HUMAN_FACTS_REQUIRED | 不能由代码或历史报告推断 | 实际作者/机构/数据责任人 | 投稿/受限数据或 2025 release 前补齐 |

## 待负责人集中回答

1. 本模型预期用于何种情境；FNR/FPR 的实际代价是什么？
2. 对各主端点可接受的 delta、依据和批准者是谁？
3. 社会处境、临床背景和数据过程变量中，哪些允许形成正式公平性主张，哪些只作描述切片？
4. 是否有 2025 数据访问权限及允许的 release 流程？
5. 谁承担伦理、作者、AI 披露与最终主张责任？
