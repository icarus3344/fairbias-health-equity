# FairBias R1A-R9 独立监督复核（2026-09-15）

**结论：REJECT 本次验收交付；保留有效修复，R1A尚未接受。** R7-03的内部模型/scaler拟合身份指定反例已关闭，R7-02的loaded来源缺口也已关闭；剩余为三类P1：执行与交付身份、日志闭合路径、生成器的语义及整体裁决。没有确认P0、真实数据泄漏或外传。本次撤回交付验收资格与推进状态，没有整体回滚源码或删除历史。

对象：[R9 worker报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r9_worker_20260915_093939Z_f8d57f0d/WORKER_REPORT.md)，run `r1a_r9_20260915_093939Z_f8d57f0d`；[用户附件副本](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R9_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt)。分支`research/nhis-fairbias`、HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`，无暂存。

## 已核实事实与关闭范围

| 项目 | 结果 |
|---|---|
| 当前worker测试 | 日志88 passed、0 failed、1 warning；13个哨兵、controlled probe有相应记录 |
| 执行前/后/当前 | 37项一致；loaded/compiled各34项，hash/bytes均匹配 |
| 前像/改动 | 114项前像匹配R8监督快照；3个既有脚本/测试变动；新增本轮生成器及scratch诊断脚本 |
| 当前run | 2个，1 FAIL、1 PASS；均保留；最终采样RSS184.81MB、3.06秒 |
| 当前日志 | 摘要2120、最终2122；两条尾记录均为test_summary.json的ALLOWED file_write |
| 旧材料 | R8监督MANIFEST的927项、上轮worker/run的606项指纹均未变化 |
| 当前交付清单 | 30项中29项匹配，**生成器源码1项不匹配** |
| 附件 | 27条命令，2次guarded入口、2次直接模型调试、6次scratch生成器诊断、1次删除probe_work |
| 监督独立执行 | 永久audit hook下4组诊断，退出0、执行异常0、意外拒绝0；未重跑全部88项pytest |

独立结果：[results.json](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R9_SUPERVISOR_REVIEW_20260915_evidence/probe_run_101411_521520/results.json)。人工模型每分区≤8行、2预测特征；生成器故障只改新的人工副本。监督没有生产/配置/测试修改、真实微数据读取、联网、安装、正式实验、暂存或提交。

| 原问题 | 本次状态 | 已证明的边界 |
|---|---|---|
| R7-01 | NOT ACCEPTED | 写一次机制保留，但当前又有直接模型调试、诊断删除及生成器hash不匹配 |
| R7-02 | PARTIAL | loaded缺/坏/空/字段/声明数反例全部拒绝；旧日志反例全部拒绝；尾事件仍未绑定真实目录 |
| R7-03 | **CLOSED（指定软件契约范围）** | 正常通过；原模型偏移和scaler偏移均在第一次错误fit被同一测试发现 |
| R7-04 | PARTIAL | 原7种生成器诊断行为正确；但字段值、实际node、来源/日志缺失及整体false仍可生成成功包 |
| R7-05/06、R8-01 | 维持CLOSED原范围 | 相关生产实现和JSON测试未变，不重复扩展验收 |
| F01/F02 | OPEN/R1B | 机器矩阵已恢复原含义；F02自然语言仍需一致，未授权接线或正式实验 |

R7-03的有效修复见[scaler.fit spy:242](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:242)、[model.fit spy:283](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:283)：按当前候选从固定F计算预期特征，核对scaler实际输入和min/max/scale，以及模型输入/y。监督分别执行正常、仅模型fit输入+0.125、仅scaler.fit输入+0.125三种情形：正常26次模型和26次scaler调用通过；两个故障均失败并经MODEL_FIT_FAILED传播，错误包含实际身份断言，而非fixture无法导入。scaler故障的实际learned min偏移为[0.125,0.125]。这次可以关闭上轮反例，不需要继续重写该测试。

R7-02的[父读取分支:1199](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:1199)已验证loaded声明数，[校验器:405](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:405)强制拒绝None及缺hash/bytes。监督执行当前父控制器原AST读取→调用片段：正常通过；缺loaded文件、坏JSON、空根对象且缺compiled enhancement.py，空sources、字段缺失、错误hash/bytes、错误声明数、单节点缺来源均拒绝。失败向退出码/判决的传播按实际父源码接线核验；未假称启动了篡改材料的完整父子进程。

## P1：R7-01 — 生成器与清单不一致，执行和诊断留存仍违背生效要求

**当前生成器不是MANIFEST记录的源码身份。** [MANIFEST](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r9_worker_20260915_093939Z_f8d57f0d/MANIFEST.json)内generate_r1a_r9_delivery.py记录：

| 身份 | SHA-256 | 字节数 |
|---|---|---:|
| 清单声明 | f89c2150c5a7b7c1fa28bf7101b2fa84f98b68c3e5695e2b7700364aa164aaf1 | 29746 |
| 当前实测 | d57bcbc70607c9b6ed02f7ca9d88b0d2474692da7e9e2c8dc6df219c83ef4286 | 30206 |

另外29项匹配。MANIFEST自身hash与用户回复一致，因此这次不只是回复抄错MANIFEST hash。不能凭现有材料断言源码何时、由谁变化，或清单如何取得旧hash；但其绑定关系确实不成立，不能接受“30项全部验证”的声明。监督后文生成器诊断针对**当前d57bcbc版本**，不能补证f89c215版本曾通过历史负例。

[附件149行](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R9_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt:149)直接执行sklearn MinMaxScaler.fit调试；[164行](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R9_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt:164)直接导入FairBias并准备AE/model/scaler调试。这两次仍违反[生效R9提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R9_TARGETED_REPAIR_PROMPT_20260915.md)的明确入口约束。只计为尝试，不断言均完成；即使因ModuleNotFoundError失败，也不是guard已安装或已阻止IO的证据。

scratch诊断脚本[23行](/Users/lkc/Downloads/code_v_0_3/scratch/test_generator_isolated_probes.py:23)使用固定probe_work，存在即rmtree；附件334、336、339、345、350、356行共6次调用，[357行](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R9_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt:357)又执行rm -rf scratch/probe_work。当前目录已不存在。新写scratch、重复覆盖/删除独占诊断证据均被明确禁止。脚本[末尾](/Users/lkc/Downloads/code_v_0_3/scratch/test_generator_isolated_probes.py:149)只打印简要成功列表；当前保留的源码不能恢复每轮实际退出、当时生成器版本和完整结果。不能用本轮监督新运行补称那些已删除的worker负例有持久证明。

history_recovery只列4条R9汇总记录，遗漏首次scaler调试、六轮诊断及删除；两条旧R8生成器命令又被填成exit0，而上轮要求无持久证据保持UNKNOWN。补记的历史不应重新制造确定性。报告“所有模型执行都走入口”与附件不符。

**处理：** 旧包不删不改，新版本重新建立源码→运行→诊断→清单链。明确旧身份和退出状态未能证明，不删除MANIFEST重建成匹配状态。本轮已修好的模型/来源代码可以保留。

## P1：R7-02 — 闭合检查仅看文件名后缀，错误目录仍通过

[563行](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:563)用`target.endswith("test_summary.json")`判定闭合目标，并用reason包含某句文字作为依据。它没有确认目标属于当前run，也未要求文件名精确相同。

独立在当前合法日志副本中保留事件数量、action、decision及reason，只修改最后两条目标：

| 目标变化 | 校验结果 |
|---|---|
| 两条均改成/unrelated/run/test_summary.json | **通过** |
| 两条均改成/unrelated/run/not_test_summary.json | **通过** |
| 删除1条或2条尾事件、改成无关action | 拒绝，旧反例已修复 |
| 错误登记原因、缺登记操作/目标、未知match_mode | 拒绝，旧反例已修复 |

这是**证据路径未绑定**，不是证明guard实际允许了这些写入。当前真实2120→2122日志没有被认定篡改。修复应由真实父控制器传入/绑定本run预期的精确summary路径，规范化后比较；不要再从文件名后缀或日志自述推断目录身份。人工副本必须显式携带被审计run的身份，正例不能仅因复制路径改变而失败。

另需纠正“pytest hook导致部分compile捕获无法实现”的叙述：[876行](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:876)已经将相同source_bytes用于hash、AST rewrite和compile，修复仍在。剩余限制是[759行](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:759)在child函数开始后重读runner源码，并非首次入口编译缓冲区的捕获。按前序授权保留准确PARTIAL范围，不以“runtime compensation”声称完全等价，也不为此扩大成通用沙箱工程。

## P1：R7-04 — 生成器仍不能验证结果含义，整体判决为false也会最终化

当前生成器确实修复了上轮明确反例：合法材料可生成；重复最终化拒绝且全包字节不变；显式FAIL、非空无关trace字典、重复同一node88次、资源FAIL、preflight FAIL均在写入前拒绝。监督在永久guard内重现了这7种行为。该正面结论适用于当前源码，不恢复已删除的worker执行历史。

剩余代码原因：

- [45行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r9_worker_20260915_093939Z_f8d57f0d/generate_r1a_r9_delivery.py:45)每个trace仅列少数键；[161行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r9_worker_20260915_093939Z_f8d57f0d/generate_r1a_r9_delivery.py:161)检查键存在，不检查类型/值、候选关系或矩阵trace_assertions。
- [139行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r9_worker_20260915_093939Z_f8d57f0d/generate_r1a_r9_delivery.py:139)只需≥88个不同字符串；没有核对实际node是否存在、是否覆盖要求的原测试。
- validate_run_artifacts不读取compiled来源或guard日志来确认必需证据完整，也不拒绝execution_verdict子项与总PASS矛盾。
- [550行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r9_worker_20260915_093939Z_f8d57f0d/generate_r1a_r9_delivery.py:550)整体判决现在是动态bool，然而[593行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r9_worker_20260915_093939Z_f8d57f0d/generate_r1a_r9_delivery.py:593)后没有以false阻止成功包，继续创建MANIFEST并打印SUCCESS。

监督执行**当前原main**，仅替换外部diff命令及改动文件列表以隔离副作用，没有替换任何校验。六种单独注入结果：

| 人工故障 | main / 最终包 |
|---|---|
| isolation.fit_all_from_f=false、fit_source=T_test、内部fit数=0，保留键 | 成功，FIXED_CANDIDATE，整体true |
| 将必须的isolation PASSED节点换成不存在的唯一节点，仍88条 | 成功，整体true |
| 删除executed_bytes_hashes.json | 成功，整体true |
| 删除guard_events.jsonl | 成功，整体true |
| execution_verdict.integrity_passed=false，外层仍PASS/overall_success=true | 成功，整体true |
| 增加第14个PASSED_BLOCKED哨兵，触发最终整体判决false | **仍成功创建MANIFEST、写FIXED_CANDIDATE并打印SUCCESS，overall_verification_passed=false** |

这些均属于前序要求的字段值/必需node/来源与日志/整体一致性检查，没有引入新的算法验收范围。真实worker日志88条PASSED不因此被认定虚假。源码的动态判决比写死True有进步，但判决必须控制是否最终化。

**处理：** 在任何成功写入前完成严格schema、字段值、必需来源/日志、唯一真实node、子项与总判决一致性验证；由同一个总裁决决定退出/状态/成功清单。未知或失败只输出新独占目录中的FAILED/INCOMPLETE诊断。保留现有7种已有效回归，新6种故障要被相应检查拒绝。

## P2：报告与矩阵位置需要由当前事实生成

这些归入R7-04交付修复，不新增缺陷编号：

- worker五文件计数写成14/19/17/16/22；真实PASSED按文件分别是**14/24/23/18/9**，总数都为88。可见总数对上不代表明细可靠。[元数据核验](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R9_SUPERVISOR_REVIEW_20260915_evidence/evidence_verification.json)。
- 矩阵48项中46项有AST位置，**11项已过期**。例如verify_guard_log_events实际432–638，仍写426–592；parent_controller实际1038–1372，仍写992–1309。当前生成器复制旧位置，只给R9-01重新计算。[位置核验v2](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R9_SUPERVISOR_REVIEW_20260915_evidence/metadata_verification_v2.json)。v1解析器未递归单名称嵌套函数，曾将check_dir_fd误记NOT_FOUND；v2修正为实际498–502，函数存在，旧位置仍过期。两个静态记录均保留。
- F01、F02机器含义已恢复，原47个ID未丢失；另新增R9-01汇总项不能替代旧缺口。F02自然语言仍写“notation unification”，应明确p/q/yhat改变的是BA/EO/DP评估对象。
- 附件5项hash声明中MANIFEST、candidate.patch、cumulative.patch匹配；history_recovery、issue_closure_matrix不匹配，但后二者当前内容与MANIFEST匹配。区别于生成器的包内不匹配，不能混为同一事件。

## 后续意见与研究边界

当前应冻结已通过的内部拟合、JSON、预处理、终局及BM/NMI/cache等修复，下一轮只收尾执行留存、精确日志路径和交付裁决。不要继续扩大模型测试范围，或用R10阶段名和更多测试总数掩盖未完成事项。

[Gemini R1A-R10收尾提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R10_CLOSURE_REPAIR_PROMPT_20260915.md)给出三批有界任务、固定故障表及中断续作规则。R1A尚未接受，不授权R1B、真实数据或提交。FairBias作为应用论文主方法的路线保持：R1B接通从F实际学习的BM/AE/Joint及竞争方法/LR/GBDT适配；R2调查domain与配对推断；R3冻结比较；已看过的2024作为冻结后的回顾性评价。同时评价预测、公平性和权衡，不预设FairBias胜出。

[静态/历史证据](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R9_SUPERVISOR_REVIEW_20260915_evidence/evidence_verification.json)、[命令与尝试快照](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R9_SUPERVISOR_REVIEW_20260915_evidence/delivery_verification.json)、[诊断源码与结果](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R9_SUPERVISOR_REVIEW_20260915_evidence/probe_run_101411_521520/results.json)及最终完整性检查/独立MANIFEST均保留。永久Python audit在所有项目/模型导入之前安装；输出路由替身不等于worker guard被安装，Python保护也不是OS全能力沙箱。CLOSED仅对应明确的软件契约，不能升级成整个benchmark零泄漏、复杂调查推断成立或论文优势结论。
