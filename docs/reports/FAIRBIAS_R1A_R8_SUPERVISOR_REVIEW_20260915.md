# FairBias R1A-R8 独立监督复核（2026-09-15）

**结论：REJECT 本次验收交付，返回 Preflight；保留已验证有效的修复，R1A 尚未接受。** 本轮仍有3次被生效提示明确禁止的直接模型调试尝试；证据裁决、内部拟合身份和报告生成器仍存在可复现反例。撤回的是本交付的验收资格与推进状态，本次没有整体回滚源码、删除历史或修改 Git。没有确认真实数据泄漏、网络外传或 P0 事件。

审查对象：[R8 worker 报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r8_worker_20260915_075612Z_fa2feedc/WORKER_REPORT.md)，run `r1a_r8_20260915_075612Z_fa2feedc`；[用户附件副本](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R8_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt)。分支 `research/nhis-fairbias`，HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`，无暂存。

## 实际核验与已修复内容

| 项目 | 已确认结果 |
|---|---|
| worker 最终测试日志 | 88 passed、0 failed、1 warning；13个哨兵及 controlled probe 有对应记录 |
| 执行身份 | pre/post/current 37项匹配；loaded/compiled 各34项、当前 hash/bytes 匹配 |
| 前像与改动 | 112项前像匹配R7监督快照；4个既有文件变化，另新增本轮目录内的生成器 |
| 交付清单 | 31项内容 hash/bytes 均匹配；生成器已经纳入清单 |
| 历史保留 | R7监督清单305项、上轮worker/run指纹492项均未变化 |
| 可见尝试 | 当前6个run完整保留：5 FAIL、1 PASS；附件29条命令含6次guarded入口、3次直接模型调试和2次生成器入口 |
| 日志与资源 | summary 2095、闭合2097；+2在telemetry有解释；采样RSS约183.77MB，时长约3.55秒；采样不是OS硬上限 |
| 监督执行 | 两个有永久audit hook的进程，均退出0；首轮5组，生成器组因精确只读路径漏配重检1次；未重复全套pytest |

本次可关闭的子项：

- **R8-01 CLOSED**：普通字符串保留大小写/空格；布尔/字符串及数值/字符串的真实JSON键碰撞仍拒绝。新建的两个JSON测试确实存在于源码及88条PASSED记录，监督直接调用同一测试也通过。[_json_group_key:80](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:80)、[真实测试:571](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_identity_and_probabilities.py:571)、[numpy测试:626](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_identity_and_probabilities.py:626)。
- R7-02：loaded中错误hash/bytes、空sources，缺必需计数、错误计数关系、缺unexpected列表、字符串健康标记、无关登记操作/目标、删除两条闭合事件，现均能拒绝。pytest同缓冲区AST/compile修复保留。
- R7-03：原“仅模型内部fit输入+0.125”反例现在失败；T的X/y/A/w单变量扰动已加入并随正常隔离测试通过。[T逐项测试:477](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:477)。仍未证明内部scaler/model的完整输入身份，见下文。
- R7-01/04：生成器已迁入本轮报告目录并入清单；合法人工材料首次可最终化，二次最终化抛FileExistsError且全部原文件字节不变；明确FAIL材料在写入前被拒绝。旧“可以覆盖”和“明确FAIL仍成功”的具体反例已关闭。
- 原问题ID数量恢复至47个；当前矩阵引用的测试均存在且出现在PASSED列表，R6-03数值清理含义恢复。但F01再次换义，不能仅按ID数量判定身份完整。

R7-05预处理、R7-06原JSON反例及先前已关闭的BM完整1998幂流、AE缓存效用、NMI、终局边界、公共指标/权重敏感性保留原限定结论；相关生产实现本轮未改。没有把旧结论升级为调查推断有效或整个benchmark零泄漏。

## P1：R7-01 — 当前执行仍越过明确入口，来源声明不完整

[生效R8提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R8_TARGETED_REPAIR_PROMPT_20260915.md)明确要求“所有项目导入、AE/LR调试、pytest及单节点一律走常驻guard入口”，包括人工数据调试。

附件[69行](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R8_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt:69)通过路径注入直接导入项目并调试AE/LR；[102行](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R8_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt:102)和[116行](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R8_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt:116)直接执行sklearn LR/scaler拟合调试。后二者未导入项目，也仍属于明确受限的模型调试。

这是3次尝试，不声称每次都执行完成或访问了真实数据；退出码没有持久证据时保持UNKNOWN。`-I -S -B`并不安装guard。生成器[522行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r8_worker_20260915_075612Z_fa2feedc/generate_r1a_r8_delivery.py:522)生成的“所有命令均受保护”声明不成立。当前索引恢复了旧R7的6次尝试，却漏记当前3次及两条生成器调用的逐条执行证据。

本轮未发现再次写scratch、删除旧清单或重写最终包的确认性证据；应保留已修好的排他最终化机制。附件两次生成器入口本身与“一次成功、第二次拒绝”的测试兼容，不能单凭重复调用判违规。

**修复边界：** 在新目录补充真实来源和UNKNOWN，不改旧记录。所有下一轮模型调试只能从入口执行；需要新诊断时先静态增加白名单节点。不得用再次手写“全部受保护”替代事实。

## P1：R7-02 — loaded校验在None路径失效，日志闭合仍未核对事件身份

[父控制器:1148](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:1148)把`actual_loaded_sources`初始化为None。缺文件、坏JSON、空根对象均保留None；[校验器:359](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:359)只在不是None时验证loaded证据。因此，换成“非空字典必需”的检查仍没有覆盖实际父路径。

监督执行父控制器原AST的读取→调用片段及当前真实校验器，结果如下。acceptance仅指证据检查返回值，未伪称运行了完整父子进程。

| 人工材料变动 | 实际结果 |
|---|---|
| 原始完整材料 | 通过 |
| 仅删除compiled中的enhancement.py | 拒绝，正确 |
| 删除loaded文件/坏JSON/空根对象，同时删除上述compiled项 | 三种均通过；仅剩33项compiled |
| loaded的enhancement.py项改为空字典 | 通过；hash/bytes缺失均未拒绝 |
| loaded声明数量改为99999 | 通过；父路径丢弃了声明数量 |
| loaded.sources为空、hash改wrong且bytes=-1、单节点缺测试来源 | 均拒绝，保留修复 |

[376行](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:376)对sha256/bytes使用“存在才检查”，不能约束必需字段。入口源码[714行](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:714)仍在child函数已开始后重读，`entrypoint_initial_load`标签不等于首次编译缓冲区捕获；此覆盖限制继续PARTIAL。

日志计数关系和严格类型已改好，但[闭合事件:535](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:535)依旧允许1至2条任意ALLOWED尾事件，[登记关联:570](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:570)仍可略过缺字段或未知匹配模式。实际反例：

- 删除真实两条闭合事件中的一条，仍通过；两条均删则正确拒绝。
- 把两条尾记录替换成无关action/target/reason，保留ALLOWED，仍通过。
- 原登记原因改为从未发生的文本，仍通过。
- 删除消费记录中的expected_action/expected_target_pattern，仍通过。
- target改为无关路径，同时match_mode改为未知值，仍通过。

这些是**证据裁决盲点**，不是证明guard实际放行越权IO。当前真实2095→2097日志保持完整，两个尾事件确为test_summary写入，不能反向指控它被篡改。

**修复边界：** 强制两份来源证据结构、必需字段、类型、声明数量、集合及hash/bytes一致，None不得在父裁决中跳过。登记与事件按实际语义关联，未知模式拒绝；闭合过程依据实际应发生的事件，而非容许常数。失败需传播到真实父控制器的退出码和execution_verdict。入口覆盖若未实现，应明确PARTIAL，不冒称完整。

## P1：R7-03 — 模型概率和条件不能证明F来源，26次错误scaler拟合仍获通过

当前测试新增了内部spy，这是进展；但[scaler spy:233](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:233)只断言形状、记录raw_X/scaled_X，没有核对它们与当前候选由F产生的预期内容，也没有校验缩放器实际学到的min/max。

[模型spy:243](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:243)以`abs(sum(p)-sum(y))<0.05`作为拟合身份判据。这个单标量条件不能识别输入来源；保持该条件成立的错误缩放仍能通过。[trace:513](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:513)仅从记录数量及上述verified标记计算`fit_all_from_f`。

监督在同一现有worker隔离测试中分别执行：

| 情形 | 实际结果 |
|---|---|
| 正常F/C学习及S/T扰动 | 通过，26次内部模型fit、26次scalerfit |
| 原模型fit输入+0.125故障 | 首次fit被检测，错误经MODEL_FIT_FAILED传播；原反例关闭 |
| 只让AE内部MinMaxScaler.fit实际拟合X+0.125 | **26次实际fit均改变，所有学到的data_min偏移0.125，测试仍通过并写fit_all_from_f=true** |

第三行没有改变模型源码、y或输入形状；fit_transform spy入口仍看到原始候选X，实际缩放器却学了另一份数据。诊断保留了每次learned_min_shift。这证明“scaler真实拟合来源”要求没有满足，不能以捉住一个模型偏移反例代替。不是生产当前已发生泄漏的证据；生产enhancement及其contracts本轮未修改。

**修复边界：** 从固定F、当前changed_dict及预期缩放规则建立参照，核对实际scaler.fit输入和学到的状态、实际模型fit特征/y/适用索引或权重。候选可合法变换F，不要求都等于原始未变换F。保留已有效的模型偏移故障，并使同形状错误scaler拟合使同一原测试失败。trace由真实内容断言得出，不只由计数/形状/概率和产生。

## P1：R7-04 — 生成器仅检查壳层，失败证据仍可得到成功包；F01再次换义

[validate_run_artifacts:85](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r8_worker_20260915_075612Z_fa2feedc/generate_r1a_r8_delivery.py:85)新增的显式FAIL、空trace拒绝是有效修复；但trace只需非空字典。[177行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r8_worker_20260915_075612Z_fa2feedc/generate_r1a_r8_delivery.py:177)统计PASSED行数，没有验证完整唯一node集合与必须执行的测试。[614行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r8_worker_20260915_075612Z_fa2feedc/generate_r1a_r8_delivery.py:614)整体通过被写死True；traces_verified只判断TRACE_FILES长度等于7。矩阵里的trace_assertions没有被执行。

监督在新的人工副本中执行**同一个生成器main**，仅隔离外部diff调用和改动文件列表；未替换任何校验或最终化逻辑。最终重检无audit拒绝：

| 情形 | main及最终包结果 |
|---|---|
| 原始合法材料 | 成功 |
| 同一最终目录第二次生成 | 拒绝，全部既有文件不变 |
| FAIL verdict、exit1、0通过日志、7空trace | 拒绝，尚未写入；原反例关闭 |
| 7个trace均换成`{"unrelated": true}` | **成功，FIXED_CANDIDATE及overall_verification_passed=true** |
| 仅一个真实node重复写88条PASSED | **成功，计为88通过，其他必需node缺失未发现** |
| telemetry报告资源监测失败、overall_success=false | **成功；子项execution_verdict=false，整体仍true** |
| preflight.status=FAIL，其他材料不变 | **成功** |

这些故障分别注入，证明检查缺口独立存在；当前真实88条PASSED不因此被认定虚假。生成器仍未完整校验输入身份、源符号、trace字段与实测值、已冻结历史等；get_ast_lines返回UNKNOWN也不阻止成功。应从校验结果合成整体判决，不能在报告末尾固定写True。

问题身份方面，[矩阵F01:726](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r8_worker_20260915_075612Z_fa2feedc/issue_closure_matrix.json:726)变成“多组非二元公平性扩展，R2”，原因是[生成器:350](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r8_worker_20260915_075612Z_fa2feedc/generate_r1a_r8_delivery.py:350)复制R6 worker旧矩阵而没有应用监督纠正。正确F01是**主benchmark没有从F学习FairBias BM，OPEN/R1B**。worker自然语言F01较正确，但机器矩阵矛盾。F02机器标题已恢复p/q/yhat区分，报告仍弱化为符号统一；其关键是BA/EO/DP实际评估对象会不同。

附件MANIFEST声明为`a4f89d38c64c76b92f9e4ea2c2ea55755104ec035f29c4baefae93f7da21867c`，实测为`e37bf74e81e295bfc1d7822daffcf84acb22943ff79400845908c0824672278e`。另外7项附件hash及清单内31项均匹配，故只认定回复hash错误，不据此断言隐蔽覆盖。cumulative.patch来自git diff HEAD，须明确tracked-only，不能代表新增未跟踪文件的完整差异。

**修复边界：** 对上述每类人工坏材料都要求实际main拒绝成功最终化。必须从当前证据生成唯一node、字段值、源符号、日志/资源/前像裁决；未实现保持PARTIAL。保留正确写一次机制。原ID含义以监督纠正为准，不从旧worker报告恢复错误定义。

## 后续推进与证据边界

本轮没有新增未关闭P2问题；R8-01已关闭。剩余工作仍是R7-01至R7-04四类P1，不能用测试总数88或新阶段名称代表完成。下一轮采用固定正常/故障验收表，保留已关闭项，避免反复扩张检查范围或重写已有效算法。

[Gemini R1A-R9定向修复提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R9_TARGETED_REPAIR_PROMPT_20260915.md)已给出可顺序执行的完整范围。R1A接受后才另授权R1B接通实际BM/AE/Joint及竞争方法的LR/GBDT适配；R2验证调查domain和配对推断；R3正式执行冻结比较；已看过的2024继续作为冻结后的回顾性评价。FairBias作为应用论文主方法的定位不变，同时比较预测、公平性及二者权衡，不预设胜出，不用旧失效排名调参。

证据入口：[静态核验](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R8_SUPERVISOR_REVIEW_20260915_evidence/evidence_verification.json)、[交付与命令核验](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R8_SUPERVISOR_REVIEW_20260915_evidence/delivery_verification.json)、[第一轮诊断](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R8_SUPERVISOR_REVIEW_20260915_evidence/probe_run_082250_625846/results.json)、[生成器重检](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R8_SUPERVISOR_REVIEW_20260915_evidence/probe_run_082352_603198/results.json)。

监督第一轮对旧生成器源码的5次精确只读请求被audit阻止，并由get_ast_lines转换为UNKNOWN；记录未清空、工件未删除。前四组不涉及该读取，生成器组另以精确路径白名单重检，退出0、执行异常0、audit拒绝0。此处是监督fixture漏配，不计为worker违规；不能把第一轮称为全程零拒绝。诊断执行源码/元数据副本与每次输出均保留，Python audit不是OS全能力沙箱；worker输出路由替身也不等于安装worker guard。

监督本轮只新增报告、提示与证据；模型诊断每分区≤8人工行、2特征，没有真实微数据读取、网络、安装、正式训练、生产/配置/测试改动、暂存或提交。未重跑全部88项测试。源码/历史最终复核及本轮独立清单保存在同目录final_verification.json和MANIFEST.json。CLOSED仅适用于明确验证的软件反例，不代表整条benchmark、复杂调查估计或论文结论已获认可。
