# FairBias R1A-R2 独立复审：仍需返修，暂不进入 R1B

日期：2026-09-14。分支 `research/nhis-fairbias`，HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`。审查对象是该 HEAD 上的未提交候选，而非仅该 commit。

**结论：不接受“R2-01 至 R2-10 完全修复”的申请。旧两个 P0 的具体反例已经修复，但 D8 出现新的正常输入崩溃；预处理冻结、组全集、真实搜索测试及交付证据仍不完整。继续 R1A-R3 定点返修，不启动正式模型比较、不授权提交。** 候选代码和历史记录全部保留。

## 1. 核验结果与实质进展

- 独立快照 87 项输入；相对上轮候选确有 13 个源码/工具/合成测试文件变化。保护文件相对 HEAD 未改、无 stage；`.gitignore` 的既有 tag 差异未被改写。
- worker manifest 逐项匹配；实际 61 个测试通过节点存在，AST 也找到 61 个测试。执行前后哈希真正覆盖 **22 项**，均一致且匹配当前文件；不是报告所称的 86 项执行哈希。没有独立重跑全部 61 项。
- 独立完成 **13 组合成诊断**，程序错误 0，监督外层保护的非预期拒绝 0。这里包含成功复现的缺陷，不能称“13 项验收通过”。真实 LR 最多 6 行/1 特征，预处理真实 fit 为 8 行/24 个 schema 字段；仅注入人工运输和历史全年人数检查。
- 上轮监督 manifest 的 121 项无变化；本轮 worker 的 4 次尝试及其报告目录仍在。本监督只新增文档和证据，没有改生产代码、读微数据、联网、安装依赖或 stage/commit。

已独立确认修复：`compute_metrics` 合法输入不再 NameError；原相对路径+dir_fd 删除被专用异常阻止且人工文件保留；嵌套字典键和空网格碰撞消失；原显式 keyword 恢复；原字典外部修改不再影响清理结果，实际 feature_lists 重排被 adapter 拒绝；年份 2023 与 "2023.0" 等价、NaN 来源被拒绝、重叠错误不再打印 ID；一维硬标签与常见非有限组检查改善。前轮概率、权重与状态编码反例仍通过。

证据：[完整诊断结果](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R2_SUPERVISOR_REVIEW_20260914_evidence/probe_run_134558_194177/worker_case_output/results.json)、[执行源码](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R2_SUPERVISOR_REVIEW_20260914_evidence/probe_run_134558_194177/probe_source.py)、[证据与测试引用核验](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R2_SUPERVISOR_REVIEW_20260914_evidence/evidence_verification.json)、[源码差异](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R2_SUPERVISOR_REVIEW_20260914_evidence/diff_from_pre_R1A_R2.patch)。

监督诊断有两次尝试：第一次在保存含 Decimal 字典键的诊断结果时序列化失败；第二次仅调整该反例的汇总字段后完整重跑。第一次源码、日志和人工产物保留，不当作完整成功结果。最后一组对新造日志做受控追加及预算故障注入，因此该诊断日志也不是正式验收日志。

## 2. P0：D8 正常评估仍会中断

### R3-01：修复字段传递时新增 KeyError（对应原 R2-04）

[d8_enhancement_runner.py:233](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:233) 从 application helper 的结果读取 `expected_groups`，下一行读取 `expected_groups_source`。但 [application_metrics.py:110](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:110) 及其他返回分支都没有这两个键。

**独立真实 LR 例：6 行拟合、4 行评价、1 特征，执行 `evaluate_representation`，直接得到 `KeyError: 'expected_groups'`。** 这不是模型性能不佳或不可估计，而是结果 schema 不一致造成软件失败；正常 helper 结果也会在 D8 包装层遇到同一缺键。

run_arm 已从冻结 arm 定义取得 expected_groups 并向四个调用点传递，这是进步；但最终调用无法完成，故原 R2-04 尚未关闭。应建立所有分支一致的结果契约，贯通组定义来源和状态，直接覆盖 helper、D8 包装层、真实 LR 终局及实际 run_arm。不能只用 `.get(..., 默认值)` 隐藏必需字段缺失。

## 3. P1：影响可信比较的剩余问题

### R3-02：预处理冻结仍漏列顺序、拟合统计和载入身份（原 R2-05）

[preprocessing.py:424](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:424) 只比较 specs；随后 [437](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:437) 使用可修改的 primary_core_features，[443](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:443) 使用可修改的 fitted_record。`PreprocessingFitRecord` 在 [73](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:73) 的文档虽称 immutable，实际 dataclass 和嵌套字典均可变。

三个真实 fit 反例已经成立：

1. fit 后反转 `primary_core_features`，输出列序改变，`is_fitted=True`，没有拒绝或重新拟合。
2. fit 后把 `fitted_record.numerical_medians['agep_a']` 改成 77，同一年龄90人工输入从输出18变77，仍称已冻结。
3. 用允许年龄范围0–100的人工 registry 在8行年龄90数据上真实拟合，导出 fit JSON；随后用官方 registry 的新对象载入，adapter 接受，年龄90仍输出90，尽管当前 registry 最大有效年龄85。

第三例的根源是 [load_fit_json:524](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:524) 无条件把载入统计与当前对象的 specs 重新绑定；[export_fit_json:517](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:517) 没有保存可校验的完整拟合规则身份。原规则的统计被重新标记成新规则下的统计。需要版本化拟合工件，绑定有效 schema/列序/规则/统计/来源并在载入和使用时验证；老文件没有证明信息时明确标遗留未验证，不能补写当前 hash 后升级可信级别。以上是人工契约漏洞，不是已发现真实数据泄漏。

### R3-03：预声明组之外的样本被静默排除，公平性可显著变好（原 R2-08）

[application_metrics.py:129](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:129) 只检查预期组缺失，没有检查额外观察组；[163](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:163) 只对 target_groups 计算。人工 a/b/c 三组例，声明 a/b 得 EO=0 且 `is_primary_estimand=True`；完整 a/b/c 得 EO=1。当前没有显式 domain 筛选契约或排除说明，不能把这个变化默认为同一总体的公平性改善。

另有三处输入/状态不一致：缺预期组时 EO=null 但仍返回主估计标志 true；字符串 `"abc"` 被当成组序列 `a,b,c`；[99–104](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:99) 对 Decimal Infinity 主动抛出的 ValueError 又被自己的 except 吞掉，已复现非有限组得到 VALID/主估计 true。统一有限标量校验与组全集政策，禁止静默过滤；若标志仅表示“主估计目标已声明”，应另有 eligibility 字段并修正 worker“全部组存在才为真”的说法，不能混用定义。

### R3-04：所谓搜索集成没有执行对应搜索（原 R2-07）

| 测试位置 | 实际行为 | 没有证明的行为 |
|---|---|---|
| [CACHE-7:282](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_state_and_config.py:282) | 实际调用一次 enhance_step，fit_count>0；随后只改变 C 并比上下文 | 同配置第二次调用复用、改 C 后实际重拟合。报告写出的 fit_count=1/1/2 并无对应执行 |
| [CACHE-3:192](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_state_and_config.py:192) | 对普通 set 做加入和成员判断 | D8 Joint 提交和循环停止 |
| [GEOM-7:332](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:332) | 构造对象、检查 policy，getattr 不存在属性时返回0 | 多次 BM 搜索的真实 restart/cursor 指数轨迹 |
| [GEOM-8:362](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:362) | `eps<=eps` 和 `eps<eps` | 实现中的候选拒绝/终局接受是否用对边界 |
| [GEOM-9:370](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:370) | Python `max(dict)` | BM 是否实际先尝试最高偏差属性以及失败后 stop |
| [ISOLATION-2:35](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:35) | 扰动 C 后比较 F 数据 hash | 独立 S/T 的 X/y/A/w 扰动不影响真实拟合和 C 选择 |

真实 AE/MDS 单次调用已有，不应否认；但整个 suite 仍未调用 `compute_metrics`、`evaluate_representation` 或 `run_arm` 来覆盖公共终局，NMI 阈值边界也未补齐。61个测试通过不支持这些未执行的断言。下一轮先让真实行为测试在当前缺陷上失败，再修实现；给出行为 trace、候选/fit 计数与明确注入范围。

### R3-05：问题关闭矩阵大量引用不存在的证据（原 R2-10）

独立核对 [issue_closure_matrix.json](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r2_worker_20260914_133802Z_42029af1/issue_closure_matrix.json)：**23 个不同的测试节点引用中，18 个在当前测试里不存在。** 其中部分是相近名称误写，部分如 D8 终局/Joint 根本无对应行为；均不能算验收证据。

该文件还写出不存在的 `FairBiasEvaluator`、`FairBiasEnhancementEngine.run_search`、`GuardedTestRunner.run` 等名字；所谓 AST 行号与当前函数不符。例如 [matrix:23](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r2_worker_20260914_133802Z_42029af1/issue_closure_matrix.json:23) 声称 compute_metrics 在270–420，真实函数始于 [evaluator.py:111](/Users/lkc/Downloads/code_v_0_3/src/fairbias/evaluator.py:111)。本轮 adapter 确实新建了 `_validate_preprocessor`，这一名字本轮存在，不能照搬上一轮的不存在结论。

原问题 ID 的含义也被换掉：R2-02原为dir_fd越界，矩阵改成标签验证；R2-03原为日志闭环，矩阵改成组校验；R2-09原为预检/监控，矩阵改成搜索轨迹。F01/F02 又被误概括为调查权重/全MEPS：原 F01 是 benchmark 没有实际从F学习 FairBias BM，F02 是事件风险p与决策概率q混用。

要求按监督原 ID 对齐，自动解析真实 AST、pytest节点和结果引用；验收前检查所有引用存在。没有行为证据就 PARTIAL/OPEN。粘贴记录显示矩阵是手工字典生成，报告是人工拼写，不应称为从真实机器结果自动导出。这里判定的是报告与证据不一致，不推测作者动机。

### R3-06：guard 与控制器闭环仍有明确缺口（原 R2-03/R2-09）

- **日志 fd 别名绕过路径保护已复现。** [_fairbias_r1_guard.py:265](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:265) 对登记fd提前返回，[518](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:518) 又直接放过日志fd的open事件。对日志路径写入被拒绝，但 `io.open(log_fd, 'wb', closefd=False)` 可向新造日志追加内容，无新增审计事件，健康状态仍true。该诊断只修改本轮新人工日志，不是对 worker 原日志的篡改。
- [expect_denial:167](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:167) 退出不检查剩余次数；已复现一次都未执行的预期拒绝被无声撤销。匹配采用目标子串、部分声明无目标/原因，并未形成要求的精确闭环。
- [runner:202](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:202) 得到 controlled_probe_passed 后只写入摘要，没有纳入返回失败条件；[242](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:242) 的健康检查发生在写summary与关闭sink之前，之后的新故障不再改变总退出。
- [runner:401](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:401) 只记录 monitor_error，[412](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:412) 仍按child退出码成功，没有强制失败或结束失去监测的运行。后两项为源码控制流证据，本轮没有实际制造系统监控故障。

已改好的磁盘健康标志、预检返回码、源码前后比较、dir_fd拒绝应保留。修复上述具体Python入口和退出规则即可，不要求扩建一个完整OS沙箱，也不承诺对恶意原生代码全面隔离。

## 4. P2：接口与交付身份

### R3-07：新接受的配置参数被静默忽略（原 R2-06/R2-08）

[enhancement_contracts.py:398](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:398) 新增 target_fairness、primary_metric、search_strategy 等形参及 `**kwargs`，但最终 payload [519](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:519) 不消费多数新增项。已复现 target_fairness=.1/.9 指纹相同；拼错 `epslion_threshold` 也被接受且与默认相同。旧算法参数恢复兼容是正确修复，但不应接受未定义接口后默默忽略。删除未支持参数或实现并绑定，未知keyword明确拒绝。[_normalize_id:208](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:208) 仍把任意对象str化，人工object被当ID接受；应限制类型，避免不稳定身份。

### R3-08：声明的源码加载策略和前像与实际不一致（原 R2-09/R2-10）

[runner:467](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:467) 写入 `FreshSourceFinder`，实际 runner 没有实现或安装它。guard日志含23种项目pyc路径的允许读取事件；这些事件本身不足以证明一定执行了旧字节码，但明确不能支持“只加载当前源码”的声明。实际执行哈希只有22项，不能把候选快照86项写成执行绑定86项。

[runner:492](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:492) 继续从旧scratch复制 `r1a_r1_preimage`。该84项清单有16项不匹配这次真正的紧邻前像；它可作为旧历史引用，不能标成本轮前像。当前正确前像仍可从上轮未变的监督快照重建，须说明重建时间与来源，不能冒充事前记录。

成功日志实际1502条，summary记录1500，是统计后写summary又产生2条事件；目前差异可解释，但需明确定义统计终点或最终校验，别机械声称完全一致。四次运行都保留、当前manifest有效且没有覆写前轮证据，这是进步。

## 5. 对原问题逐项裁决

| 原ID | 本轮裁决 |
|---|---|
| R2-01 公共指标NameError | 具体反例已关闭；持续要求公共API回归测试 |
| R2-02 dir_fd越界 | 具体反例已关闭；日志别名另见R3-06 |
| R2-03 日志闭环 | PARTIAL：健康计数改善，别名与退出闭环未完 |
| R2-04 D8最终输出 | OPEN：新增KeyError，阻断验收 |
| R2-05 Registry/冻结 | PARTIAL：外部字典/列表比对改善，拟合对象及载入身份未完 |
| R2-06 配置指纹 | 原碰撞/重复定义已修；新增静默参数需收口 |
| R2-07 行为测试 | OPEN：Joint、缓存复用、游标/边界、S/T隔离证据不足 |
| R2-08 输入/错误隐私 | PARTIAL：多处已修，组全集、非有限对象、ID类型仍有缺口 |
| R2-09 执行工具 | PARTIAL：预检/hash改善，监测及源码加载声明未完 |
| R2-10 报告可信度 | OPEN：大量错误引用、改写ID定义、错误前像身份 |

## 6. 推进建议

先按[下一轮 Gemini 提示词](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R3_REPAIR_PROMPT_20260914.md)完成四个顺序步骤：真实回归测试 → 契约修复 → 搜索/隔离集成 → 自动证据交付。不要再把测试数量增长当完成标准；每个问题必须能定位到真实触发、修改函数、执行节点和观察结果。

完整应用研究路线继续有效：R1A通过后，R1B接真实F-only FairBias及外部RW/LFR/EG-DP/EG-EO/TO-EO与Unmitigated，统一LR/GBDT和p/q/yhat；R2验证复杂调查domain/配对复制；R3按76条件、5种子做获准开发；R4冻结后回顾性评价已被观察过的2024。主20差值家族及B=2000等规则保持既定版本。本轮不重新授权这些执行。

FairBias仍是论文主方法，但当前只能谈软件修复，不能谈其预测/公平性优势。已知问题被真实关闭后，研究才能从反复修改测试转入可靠模型比较。本报告是本轮证据范围内的完整问题清单，不保证潜在缺陷已穷尽。
