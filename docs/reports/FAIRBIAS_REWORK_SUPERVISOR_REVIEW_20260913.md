# FairBias 第二轮返工的独立验收报告

出具日期：2026-09-13。验证于 2026-09-12 启动，收到继续指令后再次确认候选无漂移。

**决定：REPAIR。当前候选仍未满足边界与审计修复的验收条件。** 本轮未确认 P0 问题或足以作出 REJECT 的证据；主要阻塞为下述 P1 正确性与证据表述问题。依据 [执行协议第 3 节](/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md:24)，本决定不批准 staging、commit 或真实 NHIS 实验重跑。

审查对象是工作者最新报告所列的 13 个未提交候选文件，而不是重新审判历史 Gate。工作者主动延期的名义编码、总体调查推断、stress-elbow 稳定性改进及全特征全局排序，不作为本轮必须完成的算法重构。监督者只新增本报告与证据，未修改实现或测试源码。

## 1. 独立验证与证据边界

实际分支：`research/nhis-fairbias`。实际 HEAD：`67e6659fa65249a8842e34af5d8969629efe4bca`。

| 核查 | 独立结果 | 解释 |
|---|---|---|
| 工作者报告中的 13 个文件大小与 SHA-256 | **13/13 一致** | 上轮报告散列不匹配的问题已修正。 |
| 六个核心测试文件 + D6 指定 test_03 | **148 passed in 59.72s** | 包含新增 18 项测试；不是完整运行七个测试模块，也不是全库通过。 |
| 追加 11 项独立合成边界检查 | **3 PASS / 8 FAIL** | FAIL 表示验收断言不满足；8 个失败观察分属若干缺陷，不等于 8 个全新独立 Bug。探针脚本自身正常完成。 |
| 上轮证据目录 manifest 所列文件 | **15/15 散列一致** | 没有证据证明旧审查工件被覆盖。 |
| 三个历史 R4 主 JSON | **与原审查散列一致** | 仅证明工件未变，不能证明新代码复现历史数值。 |
| 14 个继承文件和 `.gitignore` 对当前 HEAD | **无未提交差异** | 不反推 `.gitignore` 自保护 tag 起从未变化；该历史差异已在前次审查记录。 |
| 验证前后及继续指令后的候选散列 | **无漂移** | 本结论绑定同一份工作区候选。 |

核心套件使用 Python 3.13.2、scikit-learn 1.7.1，并在独立进程的持续文件/网络访问 guard 下运行。其日志仅有一次真实 parquet 打开尝试，在打开前被阻断，对应 [D8 的预期 sentinel](/Users/lkc/Downloads/code_v_0_3/tests/test_nhis_d8_synthetic_contracts.py:1143)。新增独立探针的访问尝试日志为空。没有执行真实 NHIS/COMPAS 模型拟合，没有下载数据。

缓存与平分选择探针通过真实 `enhance_step` 路径运行，用 mock 效用/几何隔离控制流，不能将其人工评分当成真实 AUROC 或几何结果。预处理器注入探针只生成 89,802 行年份/角色元数据，替换 parquet 读取；没有加载个体数据，也没有 mock 掉 adapter 的年份、人数与角色验证。

工作者曾列出直接运行完整 D6 测试文件及旧探针的命令。其历史执行没有提供与每次命令绑定的持续访问日志，因此“此前所有执行均 0 字节读取”仍不可追溯认证。本次有 guard 的指定套件通过，不补证明此前运行；也不据此断言工作者实际读取了多少数据。

完整证据见 [证据说明](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_REWORK_SUPERVISOR_REVIEW_20260913_evidence/README.md)，其中保留工作者材料快照、候选散列、补丁、测试日志、探针脚本、JSON 结果及历史 schema 核查。

## 2. P1：缓存上下文仍不完整，且候选参数发生确定性碰撞（原 R1）

位置：[上下文构造](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:187)、[配置指纹 payload](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:220)、[候选签名](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:621)、[cycle history](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_state.py:172)。

`selection_fingerprint()` 的方法调用已经修正，隐藏验证标签变化的回归现在通过；新上下文也包含 fit、selection、有效 ε 与 slack。但它调用的配置指纹仍未包含 `h_order`、`mds_fixed_components`、`mds_max_components`、`mds_slope_threshold`、数值/类别散度选项等几何参数。加入 `cfg=` 字样不等于绑定完整配置。

三个独立反例均成立：

1. **配置变更复用旧判断。** 同一 engine 首次使用 H=0 拒绝候选，改为 H=1 后上下文键仍相同，候选不再接受几何评价。相同 H=1 配置的新 engine 则选择 `x`。探针固定其他条件并记录真实 engine 调用，确认问题发生在跳过逻辑。
2. **候选精度丢失。** 网格 `[3.00001, 3.00002]` 被 `power={power:.4f}` 映射成同一签名。引擎只评价第一个低增益候选；第二个高增益候选被跳过。只含第二个值的对照 engine 正确选择它。这是确定性的格式碰撞，不是 SHA 碰撞。
3. **cycle history 未随上下文隔离。** engine 已访问一个变换状态后，换用不同 selection 标签的分区，从同一基准启动搜索，仍因旧历史报 `CYCLE_DETECTED`；新 engine 对照则接受。当前 API 没有明确拒绝这种跨上下文复用。

这些反例证明通用搜索缓存的正确性问题，不证明冻结 R4 的单次运行实际触发了这些路径。默认六值网格本身也没有探针中的近邻指数。

**验收条件：** 将完整有效配置、数据与参数绑定到缓存/历史生命周期；可以在上下文变化时重新初始化，也可以明确拒绝复用，不能静默跳过。候选签名与状态序列化使用无损参数表示；[当前状态序列化](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_state.py:28)仍把浮点数舍入到 8 位，配置另有 6 位舍入，应一起处理。补充真实 engine 的配置变更、近邻指数及跨上下文历史对照测试，不能只比较人工构造的字符串。

## 3. P1：分区隔离和预处理来源仍有可复现缺口（原 R5）

位置：[EvaluationPartition.validate](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:115)、[预处理豁免](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:260)、[adapter 接纳预拟合对象](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:114)。

全同分区拒绝、缺少元数据时默认拒绝拟合均已生效。但新增交集检查仅在两侧索引都不是 `RangeIndex` 时执行。

**已复现：** 从同一份六行合成源数据切出 fit IDs `[0,1,2,3]` 与 selection IDs `[2,3,4,5]`，共享的两个真实源记录仍被接纳。两侧为 RangeIndex，或仅一侧为 RangeIndex，都能绕过。把完全相同的 ID 改为普通整数 Index，反而立即被拒绝。隔离结果不应取决于 pandas 索引类。

另一方面，预处理器通过 `allow_unspecified_source=True` 拟合后，记录的年份/角色均为 `unspecified`，但注入正式 `NHISStudyAdapter` 时仍被接纳：adapter 只判断 `is_fitted`，没有检查已拟合对象的来源。独立探针保留了 adapter 原始年份/人数/角色验证，只将 parquet 读取替换为生成的元数据。这证明默认 fit 的严格检查没有覆盖预拟合注入路径；未据此认定历史 D6/D8 已发生泄漏。

**验收条件：** 以带来源/年份命名空间的稳定记录 ID 判断同源交集，验证重复 ID，并覆盖部分交集和独立年度同号的正反例。不能把“数值相同”本身当作同一个人，也不能简单删除 RangeIndex 条件后误拒不同来源的同号记录。生产入口必须核验预拟合对象的年份、角色及适用配置；合成豁免应被生产入口明确拒绝。名义 one-hot 重构仍可延期。

## 4. P1：manifest 的平分规则与真实执行不符；对象驱动导出仍未接通（原 R3）

位置：[manifest 声明](/Users/lkc/Downloads/code_v_0_3/scripts/run_nhis_d8_r4_substantive.py:171)、[新参数方法](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:176)、[实际候选选择](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:775)、[manifest 输出路径](/Users/lkc/Downloads/code_v_0_3/scripts/run_nhis_d8_r4_substantive.py:304)。

默认多项式网格与 `minimum_utility_gain=0.0` 现在正确。这两项旧错误应关闭。剩余问题是新字典和 `get_active_parameters()` 都声称“效用降序，然后公平性恶化升序”，实际选择只在 `gain > best_gain` 时更新：平分时保留网格中最先出现者，没有公平性比较。

**已复现：** 同一特征的 p=3 和 p=5 候选效用同为 0.9，两者均可行，dφ 分别为 0.004 和 0.001。实际选择 p=3；声明的公平性平分规则应选 p=5。此问题是现有单特征内规则的错误描述，与已延期的全特征全局排序无关。

`get_active_parameters()` 已存在，但 R4 script 未调用它；实际仍从 `R4_FROZEN_CONFIG` 输出 manifest。当前默认值测试只比较两个字段，不能证明真实 runner 的参数传递与输出绑定。

**验收条件：** 若保持历史搜索行为，应准确声明“效用严格比较；平分保留网格先出现者”，无需为了修文档改变搜索算法。由实际构造的 engine/runner 生成或严格校验有效参数，并用非默认参数的合成执行验证输出。不要继续宣称另一份人工字典“100% 反映执行”。

### 同属 R3 的 P2：新增布尔字段没有保持 JSON 布尔类型，完整候选审计未落盘

位置：[CandidateAuditEvent.to_dict](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:302)、[runner 收集事件](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:888)、[R4 汇总输出](/Users/lkc/Downloads/code_v_0_3/scripts/run_nhis_d8_r4_substantive.py:518)。

新增严格/松弛字段已进入 dataclass 和实际候选事件，比上轮完整。但 `_clean_val` 先按整数处理，而 Python `bool` 属于 `int`。独立 JSON 往返后，`accepted`、`strictly_feasible`、`relaxed_feasible` 均变成整数 1，非布尔 true；false 同样会成为 0。原有真值比较测试不识别这个 schema 差异。

R4 script 仍没有导出 `runner.audit_events`。内存列表、dataclass 字段和 summary 行不等于完整候选审计持久化。

**验收条件：** 在整数分支之前处理布尔值，测试 true/false/null 的 JSON 类型；将完整候选事件写入新运行目录，并验证字段、上下文与事件数量。可用合成 writer 测试验收，不需要真实 NHIS 重跑，不能补写旧 run 来制造历史审计记录。

## 5. P1：勘误把当前新增字段误写成历史记录（原 R7）

位置：[Erratum 第 110 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/NHIS_D8_R4B_SUPERVISOR_ERRATUM.md:110)。

C3 的文字现已改为“观察到相同终态，机制未识别”，撤去了“预算耗尽导致相同终态”的因果断言，这项修正可以接受。

但第 110 行仍有两个可核实的错误：

- 声称历史候选日志和 runner 记录存在 `strictly_feasible` / `relaxed_feasible`。历史 R4 保存的 `enhancement_audit_summary.json` 实际使用 `fairness_feasible`；本轮新增 schema 不会追溯出现在历史日志中。
- 声称扁平 `condition_metrics.json` 同时保留 dφ 和冻结 ε。实际 48 行指标记录含 `max_dphi`，没有任何 ε/feasibility 字段。阈值需关联冻结参考工件读取。

核查对象为 `artifacts/nhis_d8_r4/20260909T080325Z_d8r4_substantive/`，其文件清单、键计数与八条历史 feasibility 汇总已保存在证据中。该目录没有完整 candidate audit 文件。当前 script 的完整 study-results 写出语句，也不能代替历史目录中实际存在的文件。

**验收条件：** 分别描述“历史实际记录”“可根据冻结阈值另行计算的诊断”“本次新增的未来输出字段”。历史记录引用 `enhancement_audit_summary.json` 的 `fairness_feasible`；dφ 与 ε 分别引用其真实来源。修订只改文档，不改历史 JSON。Arm 4 继续使用共同变更下的路径敏感性描述，不声称已识别单独的 MDS 因果效应；第 3.3 节与补充段落应使用同一限定。

## 6. 已确认修复、范围声明与 P2/P3 后续项

| 原问题 | 当前可接受的准确状态 |
|---|---|
| R1 指纹方法未调用 | 已修复；完整上下文、无损参数和历史隔离仍须返工。 |
| R2 缺失群体、未声明特征、单群体 DP | 本次核心套件的对应反例通过；下层不再对这些输入补零，DP/EOpp 返回 null 与原因。 |
| R3 默认网格与最小增益 | 当前值正确；排序描述、动态输出、完整审计持久化仍未完成。 |
| R4 非有限权重 | 核心 inf 反例通过；独立空样本和总量溢出探针也正确取消资格。 |
| R5 全同分区与默认缺失来源 | 已阻断；部分交集和预拟合注入仍有缺口。 |
| R6 HEAD/文件散列 | 本次 13/13 匹配，旧证据 15/15 未漂移；历史零读取断言需限定范围。 |
| R7 C3 机制断言 | 已降级为观察；历史字段归属仍错误。 |
| D6 test_03 生产前置依赖 | 本次指定单项在持续 guard 下通过；该测试验证 mock 前置条件下的 manifest 语义，不证明正式 release 前置检查通过。 |
| D8 guard 生命周期与被删除的 survey 测试 | fixture 已按模块启停；原类别/缺失桶测试已恢复。模块 guard 不能证明其他模块也持续受保护，应使用本轮外层 guard。 |

P2 范围说明还应修正两点：`survey_inference_eligible` 目前只能证明基础字段与权重检查通过，不能证明设计方差可估计，应明确命名/注释边界；工作者计划写“保持 D8-R4 锁定二维基线”不准确，R4 使用 stress-elbow，固定 2D 属于所比较的另一执行配置。

P3/后续方法学任务保持延期：名义编码重命名不变性、调查加权估计及设计方差、stress-elbow 稳定性诊断、预声明的新搜索策略。它们不能因本轮测试通过而宣称已经解决，也不要求本轮为完成验收立即实现。

## 7. 有限返工清单

1. 修复 R1 的完整配置/参数/历史边界，以独立新 engine 为对照测试复用行为；若不支持复用，明确拒绝。
2. 修复 R5 的 namespaced 记录独立性和生产预拟合来源检查；同时保留独立年度同号记录可用的正例。
3. 使 R3 的实际排序、参数导出和 JSON schema 一致，完成合成候选审计输出；无需引入全局排序或真实数据运行。
4. 修正 R7 的历史字段引用及工作者材料的范围表述；保留所有历史工件。
5. 在新目录记录持续 guard 下的核心套件、上述反例、原始输出和候选散列。工作者按协议第 9 节报告，结尾 `STOP — waiting for Codex review.`。

这轮已有实质进展，不要求回滚已经通过的修复。返工范围就是上述尚未闭合的正确性与证据问题。
