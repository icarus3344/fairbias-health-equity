# FairBias R1A-R10 独立监督复核（2026-09-15）

**结论：REPAIR，R1A当前交付尚未接受。** 已确认本轮源码身份与清单一致、实际日志路径缺口关闭、此前13种生成器用例均按预期处理。剩余是交付层的两类P1：完整内容校验不足，以及诊断证据未形成不可覆盖的清单链。下一轮只修交付工具，复用已核实的R10运行；不新增模型实验或重新改写已关闭的契约。

本次不延续“当前又有直接模型调试或删除”的结论：本附件没有提供这种新证据。此前严重问题的历史仍保留，但不能把旧违规重复计作本轮事件。未确认P0、真实微数据泄漏或网络外传。

审查对象：[worker报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r10_worker_20260915_105822Z_d582f745/WORKER_REPORT.md)，完整run `r1a_r10_20260915_105822Z_d582f745`；[附件副本](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R10_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt)。分支`research/nhis-fairbias`、HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`、无暂存。

## 已确认进展

| 核查 | 结果 |
|---|---|
| 完整worker测试 | 88 passed、0 failed、1 warning；逐文件14/24/23/18/9正确 |
| 执行身份 | pre/post/current 37项一致；loaded/compiled各34项，hash/bytes匹配 |
| 前像/变化 | 117项前像与R9监督快照一致；仅runner及日志测试变化；另新增交付生成器/诊断器 |
| 已关闭模型契约 | test_isolation_f_c_and_s_t_perturbation函数正文与R9完全一致 |
| 当前清单 | 31项全部hash/bytes匹配；附件32项hash声明（含MANIFEST）全部匹配 |
| 当前两个run | 单节点和完整suite均PASS，目录均保留 |
| 历史 | R9监督清单916项、上轮worker/run 221项指纹均未变化 |
| AST与测试引用 | 46项AST位置全部匹配；矩阵引用的node均存在且有PASSED记录 |
| 当前日志/资源 | summary 2130、闭合2132；采样185.41MiB，总run 6.11秒 |
| 监督独立诊断 | 永久audit hook下2组，退出0、执行异常0、意外拒绝0；**没有模型fit，没有重跑88项suite** |

附件实际执行轨迹38条命令；末尾报告另复述4条命令，不重复计为执行。未发现新的模型/scaler直接调用、scratch写入或删除命令。两次诊断脚本调用与一次手动生成器复查属于纯stdlib交付操作，不误计为模型绕过入口。[静态证据](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R10_SUPERVISOR_REVIEW_20260915_evidence/evidence_verification.json)、[命令快照](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R10_SUPERVISOR_REVIEW_20260915_evidence/delivery_verification.json)。

R7-02实际日志路径子项可关闭：[父控制器:1270](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:1270)根据真实run_dir构造expected_summary_path并传入，[校验器:560](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:560)优先用该可信参数、规范化后比较完整路径。监督调用同一校验器并传入父路径所用的明确来源路径：正常2132条日志通过；错目录同名、not_test_summary后缀以及同时伪造summary自述路径和日志均拒绝；原计数/登记/尾事件反例也均拒绝。未伪称启动了篡改材料的完整父子进程。

R7-03内部拟合及R7-05/06、R8-01等既有关闭范围保持；相关生产和模型测试不改。入口首次编译缓冲区仍按前序授权保留准确PARTIAL，不归咎于已经完成的pytest同buffer rewrite/hash/compile，也不要求为本轮交付修复扩大沙箱工程。

生成器的原13种行为已独立确认：正常最终化成功，重复最终化抛FileExistsError且原文件不变；显式FAIL、缺trace字段、重复node、telemetry FAIL、preflight FAIL、isolation错误来源值、缺必需node、缺compiled文件、缺日志、子项integrity=false、额外哨兵全部拒绝成功MANIFEST。这些是实质修复，应保留。

## P1：R7-04 — 生成器仍以数量代替来源与日志内容校验

[来源检查:285](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r10_worker_20260915_105822Z_d582f745/generate_r1a_r10_delivery.py:285)仅要求loaded/compiled文件中executed_sources长度≥15，没有对账声明数、条目hash/bytes、集合、pre/post或当前执行身份。[日志检查:305](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r10_worker_20260915_105822Z_d582f745/generate_r1a_r10_delivery.py:305)仅要求至少100个非空行，未解析JSON或调用与runner等价的日志契约。[trace校验:342](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r10_worker_20260915_105822Z_d582f745/generate_r1a_r10_delivery.py:342)新增了isolation若干布尔及joint终止原因，但仍未执行全部trace值/关系要求。

监督执行**当前原main**，只在新的人工副本内注入故障；外部diff调用和改动文件列表被隔离，没有替换校验。结果：

| 人工故障 | 当前生成器结果 |
|---|---|
| loaded与compiled声明数改99999；enhancement.py的hash=wrong、bytes=-1，保留条目数量 | **成功，MANIFEST存在，FIXED_CANDIDATE、整体true** |
| loaded仍含enhancement.py，compiled删该项并更新数量 | **成功** |
| 删除pre_execution_hashes和post_execution_hashes | **成功** |
| guard日志替换为100行not-json | **成功** |
| 真实JSON日志仅把两条尾目标改成其他目录 | **成功**；同样材料传runner的真实校验器会拒绝 |
| 其余trace语义同时置坏，保留所查键 | **成功**，详见下文 |

最后一例分别把BM人工流类型改为invalid、正式流长度改0、cache复用标记false、公共指标入口改invalid且keys_present为空、isolation选择来源改T_test且阈值999、joint事件改成字符串、NMI终局equal_feasible改false。所有材料同时存在仍获成功，说明只修上一条isolation组合反例没有覆盖已有矩阵的trace契约。这不是新增算法要求。

[独立结果](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R10_SUPERVISOR_REVIEW_20260915_evidence/probe_run_122524_176197/results.json)、[实际生成报告状态解释](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R10_SUPERVISOR_REVIEW_20260915_evidence/delivery_details.json)。原探针的fixed_candidate_written字段只匹配单行`Status: FIXED_CANDIDATE`；新模板分两行，因此旧字符串标记为false。静态重读实际报告均确认状态为FIXED_CANDIDATE且最终整体true；原探针输出未覆盖，也未因此重复执行。

这些反例证明**交付校验缺口**，不表示当前真实loaded/compiled或日志已损坏。当前真实R10运行材料通过独立静态核验；因此可冻结该run作为下一轮交付修复输入，无需重新跑模型。

**修复要求：** 使用一套完整的纯stdlib证据校验逻辑，验证pre/post/loaded/compiled的必需schema、严格类型、声明数量、集合/hash/bytes及可信run身份；逐条解析日志并核对实际计数、拒绝登记和精确闭合路径；按已存在的trace契约核对字段值及关系。校验结果必须同时决定退出、报告状态和最终MANIFEST，不能另写一个较弱的包装检查。

## P1：R7-01 — 诊断结果虽保留，但没有覆盖两次执行及最终清单链

本轮清单内的生成器源码身份已修复；诊断结果记录的generator_sha256也与当前9c0ee426版本一致。R8未知退出码与R9丢失/源码身份不一致均已用新补充记录承认，这是有效进展。

剩余问题明确限定为：

- [DIAG_ROOT:27](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r10_worker_20260915_105822Z_d582f745/diagnose_r1a_r10_delivery.py:27)固定为diagnostics；[case目录:82](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r10_worker_20260915_105822Z_d582f745/diagnose_r1a_r10_delivery.py:82)固定命名并允许exist_ok，输入copy和结果写入均可复用。附件[203行](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R10_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt:203)、[230行](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R10_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt:230)两次调用之间编辑了生成器和诊断器，另有204行手动复查；当前只有一份diagnostic_results.json，没有分别保存两次完整来源/结果。不能据此确定首次是否完成成功最终化，但不能声称两次独占证据已保留。
- 子进程stdout/stderr虽在内存捕获，[汇总写入:245](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r10_worker_20260915_105822Z_d582f745/diagnose_r1a_r10_delivery.py:245)只保留部分观察和错误末行，没有保存诊断生成器的完整stdout/stderr、逐次命令/起止和前后文件指纹。目录里的child_stdout/stderr是原pytest输入副本，不是诊断子进程日志。
- diagnostics下当前278个文件，**顶层MANIFEST覆盖0个**。[生成器:822](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r10_worker_20260915_105822Z_d582f745/generate_r1a_r10_delivery.py:822)只遍历顶层文件，没有递归清单或对整个诊断子清单的hash引用；正常case的独立MANIFEST不能替其他失败case或总结果提供覆盖。
- 诊断器没有在外层成功包已含MANIFEST时拒绝启动，故后续调用仍可改包下诊断结果；这与“最终包及诊断永久不可变”的声明不符。本次没有擅自执行这种覆盖测试，结论来自实际代码路径。

**修复要求：** 新诊断attempt目录排他创建；每轮保存输入、源码hash、实际命令/时间/退出、完整stdout/stderr及case结果，失败和中断也留存。最终清单递归覆盖诊断，或以可核验hash链引用每个独占子清单。外层包最终化后，诊断器和生成器都必须拒绝在其下写入。旧R10目录不改、不删除；无法恢复的首轮身份/结果明确UNKNOWN，不能用新运行补填。

## P2：报告仍有模板事实错误

逐文件数、AST位置及全部回复hash已改对，但[worker报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r10_worker_20260915_105822Z_d582f745/WORKER_REPORT.md)还写“sklearn未关闭文件”警告和pytest 1.48秒；真实日志是numpy极端权重测试的overflow RuntimeWarning，pytest **3.60秒**，总run为6.11秒。用户回复说2048MB限制，而当前[监测代码:688](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:688)使用4GiB采样预算。不能把标准Python权限或`-I -S -B`本身称为guard；纯stdlib交付操作被允许，也应与受保护模型执行准确区分。

生成器仍有固定的“所有13诊断通过”文案，而没有读取并验证diagnostic_results。下一轮应由事实清单生成这些字段；缺证据就写未核验，不要复制上一轮文字。此项归入现有R7-01/04，不新增算法缺陷编号。

## 后续安排与证据边界

| 原ID | 当前结论 |
|---|---|
| R7-01 | PARTIAL：当前清单源码身份、历史纠正已改善；诊断独占留存和覆盖待修 |
| R7-02 | 指定来源/日志反例CLOSED；入口首次编译覆盖保留已允许的PARTIAL边界 |
| R7-03、R7-05、R7-06、R8-01 | 维持已关闭的软件契约范围 |
| R7-04 | PARTIAL：13种原行为已过；实际来源/日志/全部trace内容校验待修 |
| F01/F02 | OPEN/R1B，未授权实际benchmark接线或真实数据 |

[Gemini R1A-R11交付层修复提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R11_DELIVERY_ONLY_PROMPT_20260915.md)将范围缩至新交付目录中的工具和证据。保持R10 runner、tests和模型源码不变；复用已验证R10 run，明确“R11交付审计、R10执行来源”，不制造不存在的R11训练运行。这样可修完当前问题，同时避免反复训练和重复改phase。

FairBias作为应用论文主方法的路线不变：R1A接受后另授权R1B接通从F学习的BM/AE/Joint与竞争方法/LR/GBDT；R2调查domain和配对推断；R3冻结比较；已看过的2024按回顾性评价披露。不能从本次软件契约推导出预测或公平性优势。

监督本次只新增报告、提示和证据，120项审查输入及历史工件最后再次核验；没有生产/配置/测试改动、模型fit、微数据读取、网络、安装、暂存或提交。完整独立工件清单、最终检查及诊断输入/源码均保存在本报告同名_evidence目录。永久Python audit在runner导入和生成器执行前安装；它不是OS全能力沙箱。本次没有确认当前工件遭恶意篡改，也没有将清单匹配夸大成科学有效性。
