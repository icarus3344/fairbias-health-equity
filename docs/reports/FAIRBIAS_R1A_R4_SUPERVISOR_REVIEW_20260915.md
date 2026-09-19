# FairBias R1A-R4 独立监督审查

日期：2026-09-15；目录：`/Users/lkc/Downloads/code_v_0_3`；分支：`research/nhis-fairbias`；HEAD：`67e6659fa65249a8842e34af5d8969629efe4bca`。审查对象为 worker 的 `r1a_r4_20260914_161802Z_d97c1578` 交付及当前未提交候选。

**裁决：REPAIR / NOT_ACCEPTED。已有明确进展，但不能按“仅F01/F02遗留”验收R1A。暂不进入R1B或正式对比实验，不授权stage/commit。** 本次确认5项P1、3项P2剩余问题，没有确认新的P0。所有候选和历史记录保留；下一轮只修具体剩余项。

## 已验证的进展与证据边界

- worker日志有77个PASSED节点，AST也有77个测试；矩阵引用31个不同节点，均存在并出现在通过日志。源码符号现在能解析，原R2/R4编号恢复原意。不能再沿用“矩阵大量引用不存在函数/节点”的旧结论。
- 37项pre/post执行输入相同且匹配当前文件；34项执行后模块记录、28项加载字节记录均匹配当前代码。19项worker MANIFEST全部匹配；上一轮监督162项MANIFEST未变。89项重建前像这次真正来自旧source_snapshot且全部匹配，candidate清单另行保存，旧前像问题的具体实现已修复。
- 本轮相对紧邻监督快照实际改动11个既有文件；enhancement.py及enhancement_contracts.py此次未变。不能把累计相对HEAD的改动清单当作本轮13项diff。
- 独立人工例确认：legacy直接export被拒绝；拟合后specs变更的export被拒绝；未知版本、主特征列序错配被拒绝；合法往返一致；合法重新fit会清除legacy状态。
- 适配器公开列清单移除agep_a后，实际get_cohort保持21列且矩阵相同；旧21→20列反例关闭。frozenset组也被正确TypeError拒绝。
- 实际运行D8 run_arm的人工例中，baseline/canonical/posthoc/joint四条路线都完成终局评价并传递冻结expected_groups及来源。非零ps错误的实际监测函数会立即调用终止/等待，旧“监测错误处理放在循环外”的问题已修复。最终日志1711事件与summary截点1709在telemetry中已有解释，不是丢失两条日志。

本次没有独立重跑77项全套测试；运行了10组有界独立诊断，包括真实fit/export/load、adapter、真实D8/LR、监测函数、人工故障判定及已知坏实现的测试敏感性检验。它们包含复现缺陷，不能称“10项验收通过”。未读取真实微数据、联网、安装、运行正式benchmark或改动生产/测试/配置。

## P1：影响结果与验收可信度

### R5-01：失败的load会提升旧对象资格；工件内容校验仍不完整

位置：[load_fit_json:575](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:575)、[标记赋值:589](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:589)、[from_dict:100](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:100)。延续R4-01、R3-02、R2-05。

`is_legacy_unverified=False` 在正文完整解析之前赋值。若后续from_dict失败，旧 `_fitted_record` 仍存在，但资格标记已经改变。

独立反例：按年龄0–100规则、8行年龄90真实fit，构造人工legacy工件并载入官方规则对象。adapter最初拒绝该对象。再向同一对象load一个规则身份正确但缺row_count的人工文件，调用抛出KeyError；**异常之后legacy变false，adapter接受原先年龄中位数90的旧统计，export也成功**。没有修改私有状态，没有使用真实历史fit文件。

载入器还接受NaN中位数、超出当前年龄范围的90中位数、缺失primary_core_features身份、反转expanded_features身份、错误categorical_features身份。前三类数值/身份问题均可到达adapter；NaN会在transform中继续输出NaN。v2缺整个rule_identity时会退为legacy并被adapter拒绝，这一例没有提升主分析资格，应与前述已接受问题区分。

修复：先在局部临时对象中验证版本、必需身份字段、所有列序/类别归属、有限统计、来源及一致性，再一次性替换全部拟合状态。失败后保留完整旧状态或整体失效，不能出现混合状态。旧工件不能靠失败加载获得新身份；不需要迁移真实历史工件。

### R5-02：为兼容人工字段新增的fallback放宽了研究结局定义

位置：[adapter OUTCOME_MAP:72](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:72)、[字段fallback:299](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:299)、[过滤及标签:317](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:317)。本轮新增回归，与R4-02测试适配有关。

当已知harmonized字段不存在时，代码按小写和后缀猜测任意列；之后只过滤notna，直接astype(int)。实际人工例中：

- 正确meddl12m输入返回{0,1}，原正常路径成立。
- 只有meddl12m_a、取值为1/2/7/8/9的输入，全部8行被保留，标签仍为{1,2,7,8,9}。
- 调用 `outcome='agep_a'` 也被接受，年龄变成结局标签。

这会让错误编码或未声明结局进入下游，至少破坏二分类契约；不能自动解释为已得到医学结局。触发这一改动的[测试fixture:477](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_registry_and_weights.py:477)使用meddl12m_a和1/2，却只检查特征列数，没有检查y。

建议恢复明确支持的harmonized结局入口，修正人工fixture为既有meddl12m和0/1。若另需原始编码适配，应由明确的源数据清理层按registry处理1/2及缺失码，再交给adapter；不要在此层按名称猜测或静默改变研究目标。保持已修好的冻结列接口。

### R5-03：关键回归测试吞掉自己的失败断言，旧漏洞回来仍会通过

位置：[test_r4_01_lifecycle_legacy_unverified_export_and_adapter_denial:361](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_registry_and_weights.py:361)，特别是370–372行。

测试把export、reload和assert放在同一try中，再用 `except (ValueError, Exception)` 捕获所有异常；只要消息含legacy/unverified/export即视为符合要求。自己的 `AssertionError('Reloaded legacy artifact must retain unverified status')` 同样会被吞掉。

独立敏感性检验在临时进程内，把export方法替换为上轮已知有缺陷的真实实现，其他当前load/adapter保持不变，再执行这一个原样worker测试。观测：首次load legacy=true，重新导出后load legacy=false，**测试仍正常返回、没有失败**。生产文件没有改动。

这是77/77不能支撑当前关闭结论的直接证据。将异常断言限制在允许拒绝的单一调用上，成功路径的assert放在捕获范围之外。关键负例应证明对已知坏实现会失败，再证明修复后通过；不以异常字符串中出现问题名称代替行为判定。

### R5-04：run_arm已接通，但循环、边界和隔离的声称仍超过实际测试

延续R4-03、R3-04、R2-07。以下主要是证据缺口，不据此断言生产Joint错误或真实数据已泄漏。

| 项目 | 现在已执行的内容 | 仍缺什么 |
|---|---|---|
| D8/Joint | [测试:391](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_state_and_config.py:391)实际run_arm；独立同例四路终局成功 | Joint为candidate_exhausted，BM/AE接受步数都是0。459–469行的A→B→A仍是普通set，未触发真实候选提交与cycle_detected |
| BM游标 | [GEOM-7:332](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:332)现在同official_stream、每个engine两次内部搜索，已消除两个参数同时变化 | 只断言cursor不减/默认0，没有实际指数轨迹或公开提交路径；第二次空遍历也可满足断言 |
| epsilon/NMI | [GEOM-8:388](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:388)真实候选及高/零信息NMI gate | 终局仍是435–437行Python数值比较；未触发终局判定，也未构造实际phi_threshold等值边界 |
| AE缓存 | [CACHE-8:335](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_state_and_config.py:335)真实执行F/C数据改变，已有C/scaler测试保留 | 函数标题声称epsilon与返回旧配置，但正文没有这两个调用 |
| S/T隔离 | [isolation:146](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:146)真实两次AE、真实变换和预测，较上轮进步 | y_s、O_s、w_s从未用于后续评价；没有独立T对象、fit来源轨迹或生产阈值状态。把局部变量设0.5再断言等0.5不能证明生产阈值契约 |

下一轮需要有意触发生产A→B→A循环和阈值边界，记录状态/候选/fit trace。对核心F/C阶段验证S/T不能反向影响冻结状态；不应错误要求正式S配置选择阶段对S数据变化不变。数据适配和正式四分区仍在R1B验收。

### R5-05：执行字节证据缺失或损坏时，完整性判断仍可PASS

位置：[compiled记录读取:746](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:746)、[最终判定:779](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:779)。延续R4-05、R3-08、R2-09。

缺失文件、JSON解析失败或空executed_sources，都会得到空字典；随后只遍历存在的条目比对，空字典没有不匹配项，integrity仍为true。对实际父控制器判定代码的AST片段，在其他前提固定成功、仅改变人工证据文件时，absent/empty/malformed三例均得到PASS、模块数0；这是代码片段的故障隔离验证，没有启动真实控制器或修改worker运行文件。

当前完整模块记录有34项，但compiled bytes只有28项，差的是5个测试文件和runner本身。37项pre/post已经覆盖这些路径，且现有哈希全部一致，不能据此声称实际执行了被改写代码；问题是声明的执行字节契约未覆盖它们，也没有对缺失证据作强制失败。pytest测试加载路径与自定义loader并非同一条路径，需要明确处理。

修复需验证工件存在、结构/非空、预期与实际模块集合、必需测试/入口身份及各条目hash，再把校验结果纳入最终verdict。日志文件的存在、解析和合理计数也应校验，不能仅依赖child summary的健康声明。不要继续用“只比较已提供条目”代替完整性检查。

## P2：收口接口、工具和报告

### R5-06：默认匹配仍非精确，监测无读数可被忽略，故障测试修改主guard记录

位置：[guard匹配:200](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:200)、[监测:293](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:293)、[guard测试:123](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:123)。延续R4-06。

旧前缀A_DIFFERENT例已关闭，但默认仍允许basename相同或目标子目录匹配。独立人工事件中，声明一个确切文件目标，实际目标是其 `/DIFFERENT` 子路径，expectation被消费；只声明basename时其他父目录也匹配。不是文件访问逃逸，而是预期拒绝记账范围不明确。区分exact-file、explicit-directory等明确模式，默认exact；不要因哨兵用相对路径就开放隐式范围。

实际监测函数对非零ps返回已会终止；但模拟“进程仍运行、ps返回0且无RSS”时返回无错误、不终止。应区别退出竞态与无法采样，失败策略明确。无需制造真实资源超限。

测试对主guard注入两条非预期拒绝后用 `unexpected_denials.pop()` 删除（128、134行）。日志仍保留人工事件，不是证据文件被删除；但最终“0 unexpected”是测试修改列表后的值。受控故障应隔离到独立状态/人工sink，不修改主运行的异常清单以换取PASS。

### R5-07：有限Decimal被接受为主分析组，但完整结果不能JSON序列化

位置：[_validate_group_scalar:44](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:44)。延续R4-07。

frozenset旧反例关闭；有限Decimal则因存在is_finite方法被直接接受，组率字典仍使用Decimal键。独立例primary_result_eligible=true，但json.dumps完整返回值抛TypeError。采用明确支持的类型及统一输出编码，或明确拒绝本轮无需求的Decimal；不要把任何带is_finite方法的对象都当合法标量。若做转换，要避免不同组身份合并。

### R5-08：交付现在能验证引用存在，但仍未从行为证据生成结论，尝试记录不全

位置：[generate_delivery.py:465](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r4_worker_20260914_161802Z_d97c1578/generate_delivery.py:465)、[WORKER_REPORT:20](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r4_worker_20260914_161802Z_d97c1578/WORKER_REPORT.md:20)。延续R4-04/R4-08、R2-10。

这次实际引用可解析，是进步。但生成器仍手填FIXED_CANDIDATE、actual_result_fields和notes，仅核对符号/节点；不存在的符号只Warning，未失败。它没有检验trace是否记录了声称的循环/状态。原公共指标项仍引用只测group helper的节点。不能说“代码及节点存在”就已验证其所声称的全部行为。

用户操作记录有[7次受控入口命令](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R4_SUPERVISOR_REVIEW_20260915_evidence/worker_submission.txt:172)，磁盘保留6个运行目录（5 FAIL、1 PASS），最终报告只列3次。其中两次哨兵失败没有test_summary，但有FAIL verdict。没有证据表明旧尝试被删除；第7条调用与无目录的具体原因未能仅凭提交记录完全还原，应明确缺证据而非只列成功附近的三次。

本轮未在提交记录发现裸pytest或新scratch写入；不能继续指控上一轮的权限偏离仍在发生。但历史更正、全尝试索引仍未交付完整。报告“全部人工测试≤8特征”也漏掉已授权的24字段预处理例；13改动文件、36闭包等计数与本轮实际11、37不一致。统一由机器材料生成，并区分本轮diff、累计未提交diff、schema字段与分类器特征。

## 对上一轮8项的裁决与下一步

| 原问题 | 当前状态 |
|---|---|
| R4-01 工件生命周期 | 原直接export、specs、版本、主列序、refit例关闭；失败load及完整内容校验仍PARTIAL，见R5-01 |
| R4-02 适配器列冻结 | 原21→20例关闭，保留修复；新增结局fallback回归另见R5-02 |
| R4-03 真实集成 | run_arm与部分AE/BM执行已接通；真实循环、边界与完整隔离仍PARTIAL，见R5-04 |
| R4-04 关闭矩阵 | 原ID/符号错误改善；关键负例会误通过、行为证据不足，见R5-03/R5-08 |
| R4-05 身份/前像 | 89项真实前像关闭，37项pre/post改善；缺失compiled证据仍可PASS，见R5-05 |
| R4-06 工具闭环 | 非零监测故障及时停止、日志截点已修；默认匹配与空采样等见R5-06 |
| R4-07 组类型 | frozenset例关闭；有限Decimal序列化见R5-07 |
| R4-08 执行与披露 | 本轮调用边界改善，尝试和历史披露仍PARTIAL，见R5-08 |

原R2/R3含义不变；F01仍是主benchmark尚未实际从F学习BM，F02仍是风险p与决策概率q混用，按计划在R1B处理。q不是“决策阈值”。

建议停止用新增测试数量推动验收，先让关键测试对已知坏实现明确失败，再修实现并补实际trace。[下一轮Gemini提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R5_REPAIR_PROMPT_20260915.md)直接授权三批有限工作：数据契约和负例敏感性 → 剩余生产行为 → 证据强制失败与完整交付。已经关闭的反例只做必要回归，不再次重写对应模块。

应用研究路线保持：R1A验收后，R1B接FairBias-BM/AE/Joint及Unmitigated/RW/LFR/EG-DP/EG-EO/TO-EO的LR/GBDT接口、F/C/S/T和p/q/yhat能力；R2验证全年复杂调查domain及共享配对复制；正式R3按冻结76条件×5种子获准开发；正式R4做已被观察2024的回顾性评价。20个主差值家族、B=2000及权重/不支持策略以已冻结计划为准，本轮不启动。FairBias为文章主方法，但优势必须由合格比较产生。

## 审计工件

[输入快照与差异](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R4_SUPERVISOR_REVIEW_20260915_evidence/verification.json)、[哈希/引用/尝试核验](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R4_SUPERVISOR_REVIEW_20260915_evidence/evidence_verification.json)、[8组行为诊断](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R4_SUPERVISOR_REVIEW_20260915_evidence/probe_run_010045_412117/worker_case_output/results.json)、[2组敏感性诊断](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R4_SUPERVISOR_REVIEW_20260915_evidence/probe_run_010357_370648/results.json)。主诊断中NaN是故意构造的损坏统计值，保留供Python读取核验，不是可发布的模型结果JSON。

两个诊断进程均在常驻独立访问策略下完成，非预期拒绝为0；真实分类器最多20行/分区、2预测特征，预处理例8行/分区、完整schema。监测和父控制器故障使用人工进程对象或原样AST片段，未启动/终止真实子进程。已知坏export只在一个临时进程中替换，磁盘源码未变。

静态核验首版误把矩阵指向的Markdown当Python解析，后改成按文件类型处理，失败脚本保留；不作为生产缺陷。最终化将再次核验106项初始输入、19项worker清单、162项前轮证据及所有文件链接。冻结文件和.gitignore没有本次工作树改动，既有历史.gitignore差异保持。本报告是本轮候选和指定契约范围内的完整已确认问题清单，不保证全库潜在缺陷已穷尽。
