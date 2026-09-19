# NHIS paper Goal 主控文件 v3

日期：2026-09-19

Goal 状态：`ACTIVE`

最终目的：形成可正式投稿、以同行评议发表为最终外部目标的 NHIS 有限属性与扩展属性公平审计 paper package。项目可控制投稿就绪质量与实际投稿；不能保证期刊录用。

## 1. 固定角色

| 角色 | 当前执行体 | 权限与责任 |
|---|---|---|
| Supervisor | `/root` | 维护 Goal、划定 gate、批准 Worker 范围、裁决 Reviewer 意见、整合证据和判定投稿就绪；不得用软件通过代替科学决定 |
| Worker | `/root/nhis_worker` | 只执行 Supervisor 明确授权的实现、验证、分析和文稿资产；提交协议 Section 9 报告；不得自批 gate、A3/A4/2025 或投稿 |
| Independent Reviewer | `/root/paper_reviewer` | 独立重跑、设计反例、核对 hash/主张/门禁并给出 Accept/Repair/Reject；不修改 Worker 产物，不替代人类领域/伦理/作者决定 |
| Human scientific owners | 实际项目负责人、领域合作者、统计/伦理/数据责任人和作者 | 决定应用场景、delta、正式 O/分组/参照/嵌套、确认性主张、权限、伦理、作者责任、A4、2025 release 和最终投稿 |

固定闭环：`Supervisor 定义 gate -> Worker 执行 -> Reviewer 独立审查 -> Supervisor 裁决 -> 才能进入下一 gate`。

## 2. 当前已裁决状态

- A0：`PASS_WITH_LIMITS`。
- A1：`SCIENTIFIC_CONTENT_NOT_FROZEN`。
- A2：`ACCEPTED_FOR_SYNTHETIC_CONTRACTS`，依据 tranche 4 独立复审。
- A3 真实性能扫描：`NO-GO / NOT AUTHORIZED`。
- A4 针对性训练：`NO-GO`；只能在 A3.5 后由人类决定。
- 2024：仅限按预声明规则的已知结果回顾性复核；当前不得扫描。
- 2025：`LOCKED_NOT_READ`。
- paper-ready / submission-ready：`NO-GO`。

A2 接受只覆盖隔离软件合同。独立接受证据：

- `INDEPENDENT_REVIEW_A0_A2_REPAIR_TRANCHE_4_V3.md`
- SHA-256：`1763668580c58723725d35a7a08ce7886cf96741711d89abe1db57c73eb3d513`
- 独立完整测试：100 passed；3 个既有 survey 数值 warning 保留披露。

## 3. 已完成的软件合同

- anchor/expanded scope 与 hash-bound frozen contrast universe；
- `O_train`/`O_audit`、audit role、逐 O group-rule 和完整 scientific contrast identity；
- Q2 scope aggregator，分别报告 new-O 内容效应与 multiplicity/precision 效应，并支持诚实的 no-new-O 负结果路径；
- `p_event -> frozen threshold -> binary y_hat`，FNR/FPR/BA 与风险质量输入分离；
- Q1 与 Q3/Q4 独立 family、NA slots、顶层 manifest hash 和防篡改验证；
- 固定 panel 的完整 member/deployment/domain verifier 及 supplemental parent 约束；
- 三态 delta 结论、margin-of-error/precision 边界和方法绝对权衡；
- canonical string record keys、exact-order/hash alignment 和 2025 fail-closed lock。

这些能力尚未绑定真实 2023 predictions、真实 thresholds 或最终科学 registry。

## 4. 下一阶段：A1 completion 与 A2 production qualification

### Worker 可继续的无结果工作

1. 建立 codebook-wide eligibility ledger 和排除分母；现有 31 行只叫 seed atlas。
2. 在授权的官方 CDC/NCHS 来源下绑定 2022/2023 codebook，完成逐年构念、codes、missing/NIU、universe/routing 和 `URBRRL -> URBRRL23` harmonization。
3. 盘点主 panel 每个成员的 model artifact、feature、preprocessing、training/prediction source、threshold policy 和 seed-to-model hash；无法证明的保持 `PENDING`。
4. 在不读取结局/性能的前提下设计 2023 prediction availability/order/coverage preflight；真正生成 replay 仍需新的明确 gate。
5. 用独立参考实现与预注册模拟验证 simultaneous-CI 在 stratified PSU、稀有事件、不平衡组、零贡献/孤立 PSU、NA replicates 和强相关 contrasts 下的 coverage/failure 行为。
6. 准备无结果的 Introduction/Methods 骨架、文献创新矩阵、claim vocabulary 和 A3.5 模板；不得填入未经 A3 的发现。

### 必须集中取得的人类事实

1. 实际应用场景及 FNR/FPR 等错误的现实代价；
2. 各确认性 endpoint 的 delta、单位、领域依据、批准人和时间；
3. 正式候选 O、分组、参照、嵌套/相关 cluster 和允许主张的角色；
4. 少量确认性 method-baseline pairs 与 seed 聚合/训练随机性 estimand；
5. 伦理/豁免、数据权限、作者资格、AI 披露、资助/冲突及 2025 release 责任人。

## 5. A3 解锁条件

只有同时满足以下项目并经 Reviewer 新一轮书面接受，Supervisor 才能签发 A3 run specification：

- 2022/2023/2024 官方 metadata 和系统 eligibility ledger 完成并 hash-bound；
- delta、正式 O、分组/参照/嵌套、确认性方法对和 seed estimand 由人类批准；
- main panel 所有成员合同、threshold、source/model/feature/preprocessing hashes 完整；
- 2023 replay source/model/code/order/output receipt 设计冻结；
- simultaneous-CI production 方法与覆盖/失败标准通过独立 QA；
- registry、family、contrast universe 和源码形成不可静默覆盖的冻结版本；
- A3 明确仍只用 2023 发现，2024 已知回顾性，2025 继续锁定。

## 6. 面向 paper 的后续里程碑

- A3：有限 vs 扩展审计的冻结真实性能证据。
- A3.5：Methods 草稿、候选 O 流程表、第一版核心结果图、claim-evidence 表；据此决定 A4。
- A4：仅在 claim-evidence 必要缺口存在时开启，最多 3 个新 O；非必需。
- A5：全部对象冻结且人类一次性 release 后才可运行 2025；若不用 2025，论文诚实降为回顾性/假设生成范围。
- A6：全文、补充材料、可复现包、伦理/作者/AI/数据声明和内部独立审稿。
- A7：人类选择期刊并批准投稿；保存最终 hash 和真实投稿回执。
- A8：审稿回复与修订；任何新增分析重新过 gate。

首篇最小证据包始终只围绕四块：审计覆盖与证据边界、expanded audit 的增量发现或诚实负结果、方法评价是否改变、冻结验证与绝对性能权衡。第二数据集和无界模型扩展不是默认条件。

## 7. 当前下一动作

当前安全且高价值的下一动作是：在不读取真实性能的前提下，完成官方跨年 metadata/eligibility ledger、真实 panel 非性能合同盘点和 simultaneous-CI production QA，同时由人类集中回答 decision log。任何一项未完成都不影响其他独立工作，但 A3 保持关闭。
