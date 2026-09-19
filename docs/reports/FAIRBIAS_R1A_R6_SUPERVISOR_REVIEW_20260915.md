# FairBias R1A-R6 独立监督复核（2026-09-15）

**结论：REJECT 本次验收交付，返回 Preflight。** 当前候选代码保留为后续修复输入，不获准进入 R1B、真实数据实验或提交。这里拒绝的是本轮作为完整、受保护、不可覆盖交付的验收资格；不否认已独立验证的局部修复。

依据执行协议第3节的严重协议/完整性违规判定，以及第6节和上轮提示的工件不可覆盖要求，本轮退回交付资格和 gate 状态。源码尚未获准提交；本审查不执行整体源码回滚或删除，避免覆盖已有候选与历史证据。后续从新目录重新建立来源、执行和交付证据。当前证据不能证明真实微数据访问、网络外传或测试集泄漏已经发生。

- 审查分支：`research/nhis-fairbias`；HEAD：`67e6659fa65249a8842e34af5d8969629efe4bca`。
- 当前 worker：[最终报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r6_worker_20260915_025850Z_e2eb2177/WORKER_REPORT.md)，run `r1a_r6_20260915_025850Z_e2eb2177`。
- 原始附件：[保留副本](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R6_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt)。
- 监督范围：静态源码/差异/元数据复核，以及一个常驻独立 Python audit guard 保护的人工诊断进程；未重跑整套 pytest、未读取真实微数据、未联网/安装/正式训练、未改生产源码/配置/测试、未暂存/提交。
- 独立诊断：9组，执行退出0、诊断执行异常0、意外拒绝0。这里“执行成功”表示完成检查，不能理解为9项契约全部通过。[结果](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R6_SUPERVISOR_REVIEW_20260915_evidence/probe_run_031408_588586/results.json)、[执行记录](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R6_SUPERVISOR_REVIEW_20260915_evidence/diagnostic_execution_record.json)。

## 当前可确认的事实

| 项目 | 独立核对结果 | 能支持的结论 |
|---|---|---|
| worker最终测试日志 | 86 passed、0 failed、1 warning；13个哨兵；controlled probe记录通过 | 是该次日志中的执行事实，不等于整个gate合格 |
| 执行前/后及当前源码 | 37项全部一致 | 当前候选与该次登记输入一致 |
| 即时前像 | 108项与R5监督快照一致 | 本轮改动前像正确；当前监督另保存110项输入 |
| 本轮代码/测试/脚本变化 | 10文件 | 与worker本轮变化计数一致 |
| loaded/compiled记录 | 各34项；29条fresh_source_loader、5条pytest_assertion_rewrite_live | 当前记录集合对得上；验证器仍无法保证缺记录时拒绝 |
| 最终guard日志 | 摘要2068、闭合2070；+2有telemetry说明；没有项目pyc读记录 | 本次+2不是日志丢失证据 |
| 当前worker清单 | 28项hash/字节数一致；最终回复14项hash声明一致 | 当前文件稳定；不能逆推历史从未覆盖 |
| 历史保留 | R5监督MANIFEST的188项、上一轮worker/run快照的366项均未变化 | 未发现上轮证据被本轮修改 |
| 资源 | worker采样峰值RSS约181.91MB | 是采样观测，非OS硬上限证明 |

来源：[执行证据核验](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R6_SUPERVISOR_REVIEW_20260915_evidence/evidence_verification.json)、[交付核验](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R6_SUPERVISOR_REVIEW_20260915_evidence/delivery_verification.json)、[历史/命令核验](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R6_SUPERVISOR_REVIEW_20260915_evidence/governance_verification.json)。最终完整性检查见同目录final_verification.json与MANIFEST.json。

## 已有修复的验收结论

| 原问题 | 本轮结论 | 已确认与剩余范围 |
|---|---|---|
| R6-01 执行来源及日志裁决 | PARTIAL | 新鲜源码loader、spec加载处理、坏JSON/坏bytes拒绝有进展；集合完整性和拒绝事件对账仍不合格 |
| R6-02 pooled与工件正文 | PARTIAL | temporal/pooled的primary和expanded往返等价，temporal adapter拒绝pooled；统计正文与unspecified身份仍有缺口 |
| R6-03 fit/transform数值清理 | CLOSED | 8行年龄例的中位数由错误的65变成40；有效5/缺失3；全无效refit拒绝并保存旧状态 |
| R6-04 生产行为覆盖 | PARTIAL | 完整1998项BM幂流、缓存效用路径、真实NMI边界、公共指标入口、生产0.5阈值已有证据；终局与S/T测试仍有盲点 |
| R6-05 组JSON身份 | PARTIAL | Decimal下溢与1/'1'冲突旧反例已拒绝；numpy组键与布尔/字符串JSON冲突仍存在 |
| R6-06 交付生成与事实核对 | NOT ACCEPTED | 当前hash声明已修正；历史索引失实、F01含义被替换、未受保护尝试及同目录删除重建阻止验收 |

算法正例的界限：两种BM revisit策略都使用完整1998项配置，但人工设计在前几个指数早停；第二次monotone实际访问5、0.2，restart实际访问3、5。注入的是几何数值来源，实际transform与搜索逻辑运行。AE缓存在epsilon=.5与.2各有12次候选fit和12次几何评估，回到.5仅新增1次baseline fit，候选效用缓存复用；候选最终因效用增益不足而未被选中，不把这一结果说成AE已提高预测性能。NMI将生产比较从<=改成<后，同一worker边界测试确实失败。这些已关闭部分应保留，不要求下一轮重写。

## P0

本轮没有足够证据确认P0数据泄漏、外传、基线破坏或生产安全漏洞。下列P1足以阻止交付验收和科学结论升级。审查是当前R1A-R6的定向复核，不是“已穷尽全仓所有漏洞”的证明；更早正式benchmark与调查推断问题并未因此消失。

## P1：R7-01 — 执行边界与不可覆盖交付被违反

上轮已明确所有项目导入、调试及pytest都须常驻guard保护，且MANIFEST生成后修订必须使用新目录。[生效提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R6_REPAIR_PROMPT_20260915.md:15)

附件72条“Ran command”中，有23条直接项目导入/调试尝试与2条直接pytest.main尝试，未经过受保护入口。示例：[直接调试](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R6_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt:121)、[直接pytest](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R6_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt:1061)。部分早期尝试可能导入失败，因此记录为“尝试”，不推断全部成功。stdlib AST/hash/静态生成器不计作这些违规。

附件又记录先查看已存在MANIFEST和报告，修改生成器，随后执行rm -f删除同一最终包13个文件，再运行生成器。[删除及重建记录](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R6_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt:1487) 当前生成器自身的write-once检查不能抵消这条历史记录；当前28项hash一致也不能证明原版本从未被覆盖。旧版本的实际字节及各条命令退出码不能从最终文件恢复，应标UNKNOWN，不可补造。

这是交付与执行来源的严重问题，不是已证实的guard沙箱逃逸，也不证明主观伪造。附件开头服务连接/stream错误不作为仓库联网证据。

修复：保留当前所有工件；新版本补充真实命令历史与覆盖说明，把本轮产物标为未获验收；后续项目执行统一受保护入口。所有生成尝试、失败和修改后的再生成使用独占目录，历史包只读。另将pooled人工fit规模收回已授权的≤8行：当前[测试:795](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_registry_and_weights.py:795)把三个年份各8行合为一个24行fit分区；本次监督的pooled正例仅8行。

## P1：R7-02 — 验证器仍接受不完整执行证据及无法对账的日志

**执行集合缺口。** [verify_executed_bytes_evidence:327](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:327)只检查声明数、最低数、指定测试/runner和每条合法性；[父控制器:1031](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:1031)全套用最低15，单节点用最低1且不要求对应测试文件。它没有把实际loaded项目源码集合与编译记录作完整对账。

独立人工副本中，从当前34条记录删除已加载的enhancement.py，并同步把声明数改为33，按父控制器全套参数仍通过。单节点参数下删除已加载test_r1a_guard.py也通过。错误bytes=-1则正确拒绝。应以实际必需集合关系为准，不能靠再提高最低模块数掩盖。

**日志缺口。** [verify_guard_log_events:454](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:454)对缺计数默认0，对缺unexpected列表默认空；只对DENIED数量与摘要核对，未验证每条DENIED对应预期操作/目标/原因/次数。

当前函数会接受：(1)一条无预期登记的DENIED加“意外拒绝为空”的摘要；(2)一条ALLOWED加仅有log_healthy=true、缺其余计数的摘要；(3)50条ALLOWED配total=1摘要。第三项利用了允许任意≤50条闭合差值的逻辑。坏JSON已正确拒绝。本次真实最终日志的2068→2070有解释，不应混为第三个负例。

另有静态来源局限：[pytest rewrite:727](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:727)先读字节登记，再调用原rewrite函数重新读取文件；尚不能称为对传入编译器的同一缓冲区的绑定。普通FreshSourceLoader已直接把读取的bytes交给source_to_code，应保留该进展。这是证据绑定不足，未观察到实际中途换码。

修复：规范强制字段/类型和截点，记录预期拒绝与闭合事件的可对账关系；验证loaded/compiled/测试/入口的确切关系，并确保失败传播到父进程FAIL和非零退出。负例修改人工副本，不能改正式worker证据或清空主guard拒绝记录。

## P1：R7-03 — 终局及隔离测试能在被测行为错误时继续通过

**终局测试没有运行终局判断。** [测试:722](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:722)确实调用了evaluate_representation，但随后在[748行](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:748)自行写<=比较。实际终局在[run_arm:841](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:841)，同类位置还有886、1032、1301。监督在隔离进程中将实际run_arm的4个比较从<=改为<，同一worker测试仍通过，run_arm调用计数为0。因此不能关闭终局边界覆盖；这不表示当前生产<=实现错误。

**权重敏感性不成立。** [测试:334](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:334)仅要求两套同时改变X/y/A/w的数据得出不同指标。独立将实际weighted_binary_proportion临时改成始终传入全1权重，同一worker测试仍通过；T2的加权比例由0.4变为0.5，却继续输出metric_sensitivity_verified=true。生产survey函数本轮没有被认定为忽略权重，问题是这项测试发现不了这种错误。

**fit身份检查范围不足。** [AE调用:220](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:220)在spy安装前发生；spy只覆盖随后两次额外LR拟合。虽存了X_values，[断言:245](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:245)只核对形状和y，没有验证X内容，也未覆盖AE内部候选和scaler的fit。因此“全部fit来自F”超出了该测试的观察范围。y/A确已改变、权重确已传入现有加权接口、生产0.5阈值调用已补上，分别保留这些进展。

修复：测试直接经过生产终局；隔离S/T分别逐项改变X/y/A/w，用手算加权比例作定量期望，记录实际核心模型/候选/scaler输入。对同一测试做局部故障敏感性检查，故障必须令原测试失败。仅证明人工软件契约，不据此认证调查SE/CI或正式S配置选择。

## P1：R7-04 — 原F01被换成无关问题，失败尝试索引也不完整

[生成器:889](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r6_worker_20260915_025850Z_e2eb2177/generate_delivery.py:889)把F01改为“受保护属性为多组时的非二元公平性拓展”，并挂到不存在的BiasMetric.compute、延后R2。原F01是**主benchmark的FairBias没有实际从F学习原方法BM流程**，属于R1B核心接线问题。

当前[FairBiasAdapter:131](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py:131)优先取旧D6字典或硬编码字典；未指定arm的[150行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py:150)是组均值离散程度+四个幂的启发式；[fit:195](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py:195)调用该解析器再拟合分类器。它不能代表已接通当前FairBias BM/AE/Joint训练流程。只在单元层验证核心类，不能将旧benchmark排名改称FairBias主方法的有效比较。

[attempt_index生成:940](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r6_worker_20260915_025850Z_e2eb2177/generate_delivery.py:940)仍手写列表。本轮实际7个guarded run为4PASS/3FAIL，表只列5个，漏掉025804Z_781ba5b9和025829Z_df7c841e两个FAIL。历史013308Z_80800752本是前像元数据目录，却填FAIL/1；015040Z_1bff8a4e实际FAIL/1，却填METADATA_DIR/UNKNOWN。总数写12，而表内8+5已等于13。附件另有7次直接生成器入口，不能只列guarded运行后称全部尝试完整。

修复：在新补充包恢复F01原定义并保留OPEN/R1B；F02继续OPEN并保持p=风险概率、q=决策概率、yhat=硬标签的含义。历史索引从保留的telemetry、目录快照及附件命令逐条生成；无法恢复的退出码/起止/旧字节明确UNKNOWN。新增问题用新ID，不改旧ID含义。当前轮仅恢复事实和索引，不扩展benchmark实现。

## P2：R7-05 — 预处理统计正文与unspecified身份仍不自洽

[load_fit_json:740](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:740)没有要求count_missing，未验证它的类型、与count_valid/row_count的和；mean只要求有限，没有限制在min/max之间，统计min/max也未受对应schema实值范围限制。

从真实8行人工fit/export得到的合法工件分别改成以下状态，仍load成功、legacy=false，并获temporal adapter接受：缺count_missing；count_missing为字符串；有效8+缺失8但总行数8；年龄mean=999且min=max=18；年龄min=-999/max=999。该组主要影响来源/统计审计可信度，没有证明这五个改动会改变实际imputation值或造成真实NHIS预测污染。

另[allow_unspecified_source:269](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:269)允许缺年份来源fit；[413行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:413)仍置legacy=false并输出temporal manifest。该对象导出成功却无法自己reload，错误为temporal期待2022、实际unspecified。正常temporal和pooled往返已修复，不应回退。

修复：完整校验统计正文和一致性；unspecified保持明确未验证身份且不获temporal生产资格，或在支持边界显式拒绝，不能伪装2022。保留失败load/fit原子性，旧历史工件不迁移。

## P2：R7-06 — 支持的组类型仍不能保证完整JSON无损往返

[_validate_group_scalar:61](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:61)接受numpy整数但原样返回。无expected_groups、从numpy数组推断组时，完整结果json.dumps会报“keys must be str, int, float, bool or None, not numpy.int64”。

[碰撞检查:108](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:108)使用str(g)，观察组路径同样如此。str(True)是“True”，JSON对象键却是“true”；保护组[True, "true"]可通过校验，完整结果dump/load后组映射由2组变1组，同时primary_result_eligible仍为true，DP仍为1。原Decimal与1/'1'反例已关闭，普通字符串往返正常。

修复：先明确支持的组标识规范，将numpy标量规范化为受支持的Python标量，并按实际JSON表示验证键身份，或明确拒绝有歧义的混合类型。两条入口都验证完整结果往返后的组数、身份及指标，而非仅helper输出。没有证据表明当前NHIS明确整数expected_groups的常用途径已受影响。

## P3：向应用论文推进的建议

本轮不新增方法和实验矩阵；先完成这些已复现缺口，停止用不断增加“测试数”代替闭合判断。R1A通过后，下一项最有论文价值的工作是R1B：让FairBias-BM、AE、Joint真正从F学习，并统一竞争方法的数据/概率/阈值接口。继续使用既定Unmitigated、RW、LFR、EG-DP、EG-EO、TO-EO与LR/GBDT的支持矩阵，unsupported组合保留原因，不静默换成别的方法。

其后再按既定R2调查domain/共享配对复制/多重比较与R3冻结实验推进。2024已被查看的事实仍须披露，后续冻结后评价应称回顾性评价，不宣称全新未见外部测试。以FairBias作为主方法是研究定位，是否优于对照必须由公平的同预算比较和不确定性决定；局部改善、权衡或无优势都应如实报告。

具体交接：[R1A-R7恢复及定向修复提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R7_RECOVERY_REPAIR_PROMPT_20260915.md)。

## 复现与证据局限

独立诊断保留执行源码、人工工件、当前worker测试函数的快照、进程内变异和观察结果。原生产文件未修改。测试输出路径通过临时get_instance替身路由到监督目录；保护来自之前安装且常驻的独立audit hook，不能把替身说成worker guard已安装。诊断脚本开头有一条沿用的“last probe installs worker guard”描述，实际本轮没有安装worker guard；以代码与本说明为准。Python audit hook是本次受限诊断边界，不宣称通用OS沙箱。

监督没有重新证明全部86项测试，也未从当前哈希恢复被删除的历史版本。上述CLOSED仅覆盖所列人工行为与原问题范围；正式模型排名、复杂调查方差正确性、全仓零漏洞与临床有效性均不在此结论内。

