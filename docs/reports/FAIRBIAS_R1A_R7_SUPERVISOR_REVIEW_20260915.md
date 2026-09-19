# FairBias R1A-R7 独立监督复核（2026-09-15）

**结论：REJECT 本次验收交付，返回 Preflight；保留当前有效修复作为下一轮输入。** 本轮重复出现明确禁止的未受保护项目调试，交付生成器又退化为可覆盖、可对失败材料写“通过”的实现，符合执行协议第3节的严重执行/完整性问题。本次撤回交付的验收资格和gate推进状态；没有执行源码整体回滚、删除历史材料或Git变更。

算法与接口层已有进展：R7-05预处理的指定反例、R7-06原numpy/JSON碰撞反例已独立验证修复；生产终局边界和忽略权重的故障现在能被同一测试发现。不能把这些进展与整轮交付验收混为一谈。未取得真实数据泄漏、网络外传或P0事件的确认性证据。

审查对象：[worker最终报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r7_worker_20260915_063925Z_4d5feb4f/WORKER_REPORT.md)，run `r1a_r7_20260915_063925Z_4d5feb4f`；[用户附件保留副本](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R7_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt)。分支仍为`research/nhis-fairbias`，HEAD仍为`67e6659fa65249a8842e34af5d8969629efe4bca`，无暂存。

## 本轮实际证据

| 核查项目 | 已确认结果 |
|---|---|
| worker最终pytest日志 | 86 passed、0 failed、1 warning；13个哨兵；controlled probe记录通过 |
| 即时前像 | 110项与R6监督快照一致 |
| 执行前/后及当前源码 | 37项一致；loaded/compiled各34项，当前记录hash均与源码一致 |
| 本轮源码/测试/脚本变化 | 8项；另新增scratch生成器不在授权目录 |
| 最终日志/资源 | 摘要2095，闭合2097，+2有telemetry说明；采样RSS约183.33MB，非OS硬上限 |
| 当前worker清单 | 29项内容hash和字节数一致；附件7项hash声明中6项一致、MANIFEST声明错误 |
| 历史保留 | R6监督MANIFEST的185项、上轮worker/run快照519项均未变化 |
| 当前尝试 | 磁盘5个R7 run，4FAIL/1PASS；附件开始于首个run之后，包含59条命令、4次guarded入口、6次直接项目调试尝试、1次生成器入口 |
| 监督独立诊断 | 1个常驻audit guard进程、7组诊断、退出0、诊断执行异常0、意外拒绝0 |

监督只读生产源码/配置/测试，新增本报告、下一轮提示及证据目录；没有真实微数据读取、联网、安装、正式训练、暂存或提交。人工诊断每模型分区≤8行、≤2预测特征，完整预处理schema≤8行。没有重跑整套86项测试。“7组执行完成”不表示7个契约都合格。

证据：[静态与历史核验](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R7_SUPERVISOR_REVIEW_20260915_evidence/evidence_verification.json)、[命令与交付核验](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R7_SUPERVISOR_REVIEW_20260915_evidence/delivery_verification.json)、[独立诊断结果](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R7_SUPERVISOR_REVIEW_20260915_evidence/probe_run_070015_438338/results.json)。最终输入与历史完整性检查另存同目录final_verification.json及MANIFEST.json。

## 原问题闭合状态

| 原ID | 结论 | 边界 |
|---|---|---|
| R7-01 执行与不可覆盖交付 | NOT ACCEPTED | 已补述R6历史，但当前又出现6次直接项目调试；生成器可重写最终包 |
| R7-02 执行来源及日志对账 | PARTIAL | loaded完整时能拒绝漏编译模块、单节点对应测试也必需；loaded证据自身缺/坏时仍接受，日志对账仍有反例 |
| R7-03 终局与隔离测试 | PARTIAL | 终局<=生产函数及权重故障敏感性已确认；AE内部fit身份及T逐项扰动仍未完成 |
| R7-04 原问题身份与历史索引 | PARTIAL | F01原含义和5个R7 run恢复；F02被弱化，旧ID丢失/换义，仍引用不存在测试及无依据退出码 |
| R7-05 预处理统计/unspecified | CLOSED（本轮指定范围） | 五类坏正文全部拒绝且旧状态保留；temporal/pooled往返等价；unspecified标未验证并拒绝导出 |
| R7-06 原组JSON问题 | CLOSED（两个原反例） | numpy整数两入口往返、True/'true'及1/'1'碰撞拒绝均通过；另有R8-01过度拒绝的兼容性回归 |

已确认的正例不要求重写：年龄中位数仍为40；完整1998项BM幂流、AE缓存效用、NMI及公共指标入口沿用上轮结论，相关生产实现没有本轮变化。终局函数抽取后有4个run_arm调用位置；本轮独立将实际函数<=改为<并在测试模块导入时绑定变异函数，同一原测试失败。权重函数改为始终用全1权重，同一隔离测试失败。终局接线按源码核验，未执行包含真实数据流程的完整run_arm。

## P1：R7-01 — 当前仍绕过项目执行入口，生成器没有不可覆盖保障

附件记录6次直接导入项目/调试尝试，位于103、139、179、237、270、298行，其中多条主动注入site-packages和src路径后运行AE/LR人工拟合。[首个例子](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R7_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt:103)、[路径注入例子](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R7_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt:139)。这违反[生效R7提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R7_RECOVERY_REPAIR_PROMPT_20260915.md)中所有项目导入/调试必须走常驻guard的要求。不能由“-I -S -B”推断已受guard保护；也不能因输入是人工数据便改变执行边界。

这是6次尝试，不断言每次均成功；单独第三方源码路径检查及stdlib日志/AST/hash读取未计入这6次。附件没有证明本轮再次删除旧MANIFEST，不应重复指控上轮行为。

生成器被放在明确禁止新增的[scratch路径](/Users/lkc/Downloads/code_v_0_3/scratch/generate_r1a_r7_delivery.py:24)，且没有纳入worker MANIFEST。[main:83](/Users/lkc/Downloads/code_v_0_3/scratch/generate_r1a_r7_delivery.py:83)直接mkdir(exist_ok=True)，后续覆盖trace/patch/report，[MANIFEST:793](/Users/lkc/Downloads/code_v_0_3/scratch/generate_r1a_r7_delivery.py:793)用w打开，无既有最终包检查。监督在新人工目录执行同一个main两次，第二次仍成功，MANIFEST内容改变。

修复应限定在新的报告目录和新生成器版本，不能移动/删除当前scratch文件来隐藏历史。最终包生成前检查存在性；部分生成失败也保留尝试目录；需修订则创建新版本。当前源码正例可以保留，执行来源与交付保障必须重新建立。

## P1：R7-02 — loaded来源文件缺失时检查失效，日志只是字段存在而未完整对账

[父控制器:1087](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:1087)将actual_loaded_sources初始化为空；文件缺失、JSON解析异常或缺executed_sources时保持空字典，随后传入验证器。[包含关系:356](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:356)只遍历字典的键；空字典令必需模块检查自动通过。

独立使用父控制器原AST中“读取loaded文件→调用验证器”的真实片段，在人工副本中观察：

- loaded完整时删除enhancement.py的compiled记录：正确拒绝。
- 同时缺失loaded文件、把它改为坏JSON或空对象：compiled仅剩33条仍接受。
- loaded记录里enhancement.py的sha256改为wrong、bytes改为-1：仍接受，因为其值不参与对账。
- 单节点缺对应测试文件：本轮已正确拒绝，应保留修复。

这是实际父函数片段和实际验证器的执行结果，未启动被篡改的完整子进程；不把它描述成已经复现整条进程退出0。

日志验证器[472行](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:472)检查整数类型，[500行](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:500)把容许差值从50缩成2，但未检查完整数量关系与闭合事件身份。对当前合法日志/摘要的人工副本：

| 变动 | 当前结果 |
|---|---|
| 删除mandatory allowed_events | 正确拒绝 |
| 清空consumed_expected_denials但保留22条拒绝事件 | 正确拒绝 |
| total_events=0、allowed_events=999999，保留真实cutoff | 接受 |
| 删除unexpected_denials字段 | 接受（默认空列表） |
| log_healthy设为字符串“false” | 接受（非空字符串为真） |
| 改消费记录的expected_action/expected_target_pattern为无关操作和目标 | 接受 |
| 删掉两条实际闭合事件 | 接受（差值变0仍在允许范围） |

[517行](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:517)只比较消费记录抄下的action/target/reason，没有验证其对应原登记约束。guard本身的登记匹配逻辑并未被认定存在同样的错误；这是证据裁决漏洞，不能据此声称实际发生越权访问。

pytest rewrite现已把同一source_bytes用于AST和compile，[768行](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:768)是实质修复。入口初始源码记录仍在函数开始后read_bytes生成，[655行](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:655)不等于捕获首次入口编译缓冲区，完整覆盖声明应继续受限。

修复：对两份执行来源文件都做必需结构/类型/集合/hash/bytes检查；失败明确传播。日志强制布尔和必需列表，校验total/allowed/denied/cutoff关系及闭合事件；将消费记录和原登记约束关联。不要再通过调小容差常数代替事实对账。

## P1：R7-03 — 新“前置spy”只看到了AE参数，仍没检查内部fit

新增[SpyFairAccuracyEnhancement:218](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:218)确实核对了传入enhance_step的X/y。但这不是模型或scaler.fit的spy；[249行](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:249)的LR spy仍仅覆盖AE运行结束后另做的两次拟合，并且只断言X形状和y。

监督在同一worker测试中，仅让AE内部baseline/candidate模型fit实际收到的特征整体+0.125，形状与y不变。**26次真实fit的输入均被改变，测试仍通过，trace仍写fit_all_from_f=true。** 这是测试盲点，不是当前生产已发生数据泄漏的证据；生产[模型fit:777](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:777)本轮没有修改。

此外，逐项扰动只新增在S分区，[360行](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:360)开始明确标“S partition”；T仍为多变量同时改变。单独0.75/0.25手算权重例已有效，监督的忽略权重变异现在会使测试失败。应关闭权重函数数值敏感性子项，继续完成内部模型/scaler真实输入与T逐项扰动。

修复：在真实内部fit处安装记录/断言，核对当前changed_dict下由F产生的转换和缩放输入，记录实际baseline/candidate数量；不能拿入口参数或后置两次拟合代替。冻结后S/T评价不得重新fit。故障必须使原测试失败，trace布尔值应由这些观测计算。

## P1：R7-04 — 交付生成器会为失败材料写成功报告，缺陷身份再次漂移

[生成器:640](/Users/lkc/Downloads/code_v_0_3/scratch/generate_r1a_r7_delivery.py:640)只填AST行号，没有验证节点是否实际存在/执行、trace字段是否满足要求；也不读取execution_verdict决定是否允许最终化。[报告文本:694](/Users/lkc/Downloads/code_v_0_3/scratch/generate_r1a_r7_delivery.py:694)把86通过、0失败等写成常量。

监督运行原生成器main，在新人工副本中提供FAIL verdict、exit=1、0通过的测试日志和7个空trace，生成仍成功，报告写FIXED_CANDIDATE、86通过、0失败。为隔离副作用，仅将外部diff命令替换为空结果并禁用改动文件列表；没有绕过任何校验或报告逻辑。正常材料作为正例也成功。实际worker最终86条PASSED日志存在，不能因生成器漏洞反过来声称本次86条日志虚假。

当前机器缺陷表还含3个不存在且未执行的node：

- R7-06：[两条JSON node](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r7_worker_20260915_063925Z_4d5feb4f/issue_closure_matrix.json:93)。对应identity测试文件本轮没有改变。
- R5-02：test_r5_02_adapter_rejects_pooled_preprocessor；实际源文件不存在此函数。

因此R7-06软件修复的本次关闭来自监督自己的诊断，不能归功于不存在的worker测试。

旧问题表29个ID没有被带入新表；R6-03由“fit/transform数值清理”变成“GEOM-7游标推进”，[生成器:587](/Users/lkc/Downloads/code_v_0_3/scratch/generate_r1a_r7_delivery.py:587)重新分配了编号。F01已正确恢复原义，但F02被描述成全仓符号统一；其原问题是p/q/yhat混用导致BA/EO/DP实际比较对象不一致，不只是重命名。

历史索引对5个R7 run、两个R6漏列FAIL、两个R5目录性质的恢复正确，应保留。但[259行](/Users/lkc/Downloads/code_v_0_3/scratch/generate_r1a_r7_delivery.py:259)把7条R6生成器尝试直接填为6次exit0、1次exit1；上轮证据明确这些逐条退出码未能恢复，应UNKNOWN，不能由“预期失败”推断实际退出。R7当前6条直接项目调试和生成器入口没有作为当前完整尝试记录纳入表。也不应仅凭脚本名称把更早其他gate的直接运行追认为未经授权。

附件MANIFEST哈希声称`1460395780d6b6fb5404d80fc4ef2713f8c85ae7d995c76db3158c545300e843`，当前实际为`a7b4a25279325734454327e92aa37032147f825d6b9bc8f88040b516135fb6a3`。其内部29项均匹配；这说明回复的hash声明不对，不能仅凭此断言文件被偷偷修改。

修复必须调用实际生成器校验路径来拒绝坏输入；生成器与校验结果应入清单。恢复全部原ID含义，通过明确引用保存旧状态；不必复制所有旧报告，但禁止省略后宣称完整。事实计数与hash从同一机器材料生成，未知证据保持UNKNOWN。

## P2：R8-01 — JSON键检查过度规范化普通字符串

[_json_group_key:84](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:84)把字符串strip并lower后再比较。因此合法的两个字符串组“True”和“true”也被拒绝；实际JSON对象保留大小写，它们本来是不同的键。同样可能误拒绝带空格的组名。

独立反例已复现；这是兼容性/错误拒绝回归，不是丢组或错误公平性数值。原True布尔值与“true”字符串碰撞已经修复。

可在同一小函数内保留普通字符串原值，只对布尔值使用实际JSON键编码；验证完整结果两入口往返。若有意限制字符串组名，应声明单独的输入规范而不称其为“exact JSON object key representation”。不涉及正式NHIS特征或estimand改变。

## P3：对后续项目推进的意见

当前阻塞已经从大量算法问题收敛为四类：遵守执行边界、来源/日志拒绝坏输入、内部fit身份验证、交付自动核验。继续增加测试数量或重新手写报告生成器不能解决这些问题。下一轮应修现有失败路径并把已关闭内容锁为回归，不再重新定义旧问题或重写整套算法。

以FairBias作为应用论文主方法的路线不变：R1A软件契约接受后，R1B接通从F学习的BM/AE/Joint，统一竞争方法、LR/GBDT、F/C/S/T、p/q/yhat及权重能力；R2验证调查设计与配对推断；R3执行冻结比较；已看过2024的数据评价仍按回顾性结果披露。公平性和预测性能都比较，不预设FairBias胜出。

下一轮：[Gemini R1A-R8定向修复提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R8_TARGETED_REPAIR_PROMPT_20260915.md)。本轮未获准进入R1B或真实数据实验。

## 证据边界

独立诊断保留完整源码、当前测试函数/元数据快照、人工副本、过程内故障和结果。项目导入前安装常驻独立Python audit hook；输出目录替身仅用于路由trace，不能说worker guard因此已经安装。Python级保护不是通用OS沙箱。

当前附件从首个R7 run之后开始，不能假装掌握缺失部分的完整命令历史。监督没有重复全套pytest，也没有从最终hash恢复旧生成器版本或未知退出状态。CLOSED仅对应表中已验证软件反例，不能升级为零泄漏、调查推断有效性、论文复现或预测排名结论。

