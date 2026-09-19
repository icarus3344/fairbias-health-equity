# FairBias R1A-R5 独立监督审查（2026-09-15）

**结论：REPAIR。R1A尚未验收，不能进入R1B、正式比较实验或论文结果定稿。** 本轮确实修好了若干问题，但“R5-01…08全部解决、只剩F01/F02”不成立。以下是当前交付范围内确认的剩余问题：4项P1、2项P2；未发现新的P0。没有声称穷尽整个项目的所有缺陷，也没有重新验证真实NHIS结果。

审查对象为分支`research/nhis-fairbias`、HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`上的未提交工作树，worker运行`r1a_r5_20260915_015117Z_7bac5330`。以[上一轮监督快照](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R4_SUPERVISOR_REVIEW_20260915_evidence/verification.json)为前像，核验本轮108项输入并保存源码快照。本轮实际改变10个已有源码/测试文件，worker列出的13个中，enhancement.py、enhancement_contracts.py、d8_enhancement_runner.py均未相对上一轮改变。

**哪些成果已经有效**

| 上轮问题 | 本次裁定 | 独立证据及边界 |
|---|---|---|
| R5-01：失败加载原子性与完整工件 | PARTIAL | 缺row_count的加载失败后，legacy标记和旧record保留，export继续拒绝；NaN中位数拒绝。剩余内容校验和兼容性见R6-02。 |
| R5-02：任意结局fallback | 已核验修复 | 真实get_cohort对harmonized 0/1返回正确；未知agep_a、仅原始1/2/7/8/9字段、非二元harmonized标签均拒绝。 |
| R5-03：测试吞掉自己的断言 | 已核验修复 | 当前原测试通过；临时注入“仅移除legacy拒绝”的坏export实现，**同一原测试失败**，报DID NOT RAISE。磁盘生产文件未改。worker新增的单独mutant示例不等同于敏感性证明，但本次监督已补足这一证据。 |
| R5-04：实际搜索、缓存与隔离 | PARTIAL | 实际run_arm接受A→B后拒绝回到A，终止cycle_detected，bm_steps_accepted=1，保留B；四路线LR正例继续通过。BM提议与epsilon函数被注入，提交、cycle和终局控制流是真实的；不是完整算法保真度验收。其余见R6-04。 |
| R5-05：执行证据不完整仍PASS | PARTIAL | 缺文件、空字典、坏JSON等已拒绝；遗漏部分模块和坏日志仍可PASS，见R6-01。 |
| R5-06：拒绝目标、主guard污染、空RSS | 已核验修复 | 默认exact不再匹配子路径/异父目录同名文件；显式directory仍可用。故障事件进入独立guard实例，未再pop主unexpected列表。运行中进程返回空RSS会触发MONITOR_ERROR和终止。 |
| R5-07：组序列化 | PARTIAL | 常见有限Decimal可序列化，未知is_finite对象被拒绝；无损身份尚未完成，见R6-05。 |
| R5-08：交付与实际一致 | OPEN | 包内manifest正确，但关闭矩阵、最终答复hash、尝试清单及文件数仍失真，见R6-06。 |

证据：[独立诊断结果](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R5_SUPERVISOR_REVIEW_20260915_evidence/probe_run_020237_895625/results.json)、[guard/监测补充诊断](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R5_SUPERVISOR_REVIEW_20260915_evidence/probe_run_020330_656155/results.json)。首轮10组诊断中，监测探针因监督脚本漏传两个参数而发生TypeError；保留了失败工件，并以正确调用单独重跑该组。该TypeError不是生产缺陷。最终10个问题组均获得观察结果，不能把含有反例的诊断组称为“10项验收通过”。两次运行均无独立guard意外拒绝。

**P1 / R6-01：执行完整性验证仍会放过不完整或无效证据（延续R5-05）**

- [verify_executed_bytes_evidence](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:306)只遍历提交的条目，没有与实际加载集合及应执行测试集合做完整性核对。人工清单应有两个模块、只提供其中一个，返回True；伪造executed_module_count=999及bytes=-1也被接受。这里不能简单要求37个pre文件全被执行，因为其中包括配置及可能未执行的源码；应分别定义pre、loaded、compiled、选定测试与runner之间的集合关系。
- [602–626行](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:602)在测试结束后重新read_bytes补入runner和5个测试文件，却标为entrypoint_main/pytest_assertion_rewrite。实际是**28个自定义loader记录 + 6个事后文件hash**，不是34个均在加载/编译时捕获的字节身份。运行日志还存在runner的.pyc读取；`-B`只禁止写缓存，不能证明未读缓存。[测试中的spec_from_file_location](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:299)也绕过了普通meta-path查找路径。没有证据证明这次实际执行了坏缓存，但现有材料不足以排除。
- [日志验证](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:845)只数非空行，不解析JSON或核对事件内容、截点及计数。对实际父控制器裁决段做AST原样抽取，其他条件固定成功，输入一行“not json”，仍得到PASS；一条DENIED记录配合声称0意外拒绝的摘要、1行日志配合声称999条事件，也PASS。此为人工工件上的真实裁决片段验证，未修改worker日志或启动整个控制器。

影响：损坏/遗漏的执行证据可被标为合格，后续所有“代码确实按所述运行”的结论缺少可靠入口。修复应让缺失集合、加载来源不明、错误长度/计数、坏日志都导致最终FAIL及非零退出；捕获实际执行时的来源，不能事后补名目。

**P1 / R6-02：工件加载破坏既有pooled往返，且正文仍未完整校验（延续R5-01）**

[fit的pooled分支](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:289)仍合法生成`fit_year=2022-2024_pooled, fit_study_role=pooled_train`；[新加载器](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:656)却只接受2022/development_train。8行人工数据实际fit→export成功，随后同schema reload被拒绝。这是本轮新增兼容性回归。temporal adapter应拒绝pooled对象，但通用preprocessor不应生成自身无法读回的既有模式工件。

[705–721行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:705)只查numerical_stats/categorical_categories的外层键，不查内部内容；[from_dict](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:100)还做宽松类型转换。独立反例均获“已验证”身份并被adapter接受：数值统计mean=NaN且count_valid=-8、类别999、错误rules_manifest、row_count=1.9被截为1。版本改为v1_legacy但保留v2 identity，也被认定非legacy（[589–633行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:589)）。

这些正文异常多数直接破坏来源/审计元数据，而非全部直接改变预测值，应区分影响。修复应按明确的regime和版本校验完整对象，保持局部验证后提交；不放宽temporal adapter的2022训练限制，不给矛盾的legacy版本自动授予当前身份。pooled修复仅恢复既有软件兼容性，不授权pooled真实数据实验。

**P1 / R6-03：fit与transform对数值有效性的规则仍不一致**

[fit 315–335行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:315)仅dropna后计算中位数，再检查中位数是否越界；[transform 485–495行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:485)先把越界值视为缺失。8行人工年龄`[20,30,40,50,80,997,997,997]`得到中位数65，并将后三项填为65；有效值的中位数应为40。检查“最终中位数在范围内”无法防止异常码参与估计。

边界：[上游harmonize](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/features.py:166)会清理声明缺失码并拒绝意外码，正常经过该层的输入有额外保护；本次没有证明正式NHIS结果已受污染。缺陷存在于preprocessor自身可接受的直接输入/大写fallback路径。修复可选择先按同一schema清理，再仅用F的有效值拟合，或者明确拒绝非harmonized输入；必须与transform约定一致。全无有效值明确失败，保留原拟合状态，不填任意常数或在此轮决定正式特征删除政策。

**P1 / R6-04：部分测试仍不能证明它们声称的生产行为（延续R5-04）**

| 子项 | 已确认事实 | 需要补充的验收 |
|---|---|---|
| AE缓存 | [CACHE-8](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_state_and_config.py:395)传epsilon但不传O_train。独立spy发现epsilon=.5和.2各产生12个`MISSING_PROTECTED_DATA_WITH_ENABLED_GUARD`，候选模型拟合均为0；返回.5时无新事件，仅增加一次baseline fit。 | 返回配置确有拒绝记录的复用，但不是成功候选效用缓存的证明。须提供保护组，使候选通过真实公平性门；分别记录候选评估、模型fit、拒绝缓存、效用缓存及选中状态。[生产拒绝逻辑](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:241)当前行为正确，不能为了测试删掉它。 |
| BM幂流 | [GEOM-7第337行](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:337)仍传六值`1/7,1/5,1/3,3,5,7`并标official_stream。两次实际搜索现已能区分游标行为。 | 只能视为人工流控制测试。必须从现有官方序列配置取完整1998项`3,1/3,5,1/5,…`，只改变revisit策略，借早停控制成本，不用AE六网格冒充。 |
| 终局/NMI边界 | [515–520行](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:515)仍是测试自定义`check_terminal_feasibility`。NMI名义“等值”实际为0.049999999990000044，小于.05。独立将真实NMI门的`<=`临时改为`<`，该测试仍通过。 | 调用生产终局[feasibility赋值](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:1301)；NMI使用与生产同样浮点计算的真实边界，并让不等号故障变异确实失败。候选严格epsilon门原有正例保留。 |
| S/T隔离 | [隔离测试](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:177)新增独立T并消费y/A，但S1/S2及T1/T2的y/A值并未改变；4个w向量从未传入任何评价调用。阈值仍在[229–231行](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:229)本地设.5再断言自身；fit_source是构造时标签，非实际拟合调用证据。 | 分别扰动X/y/A/w，实际调用冻结变换、分类器和已有加权评价；用fit/scaler spy记录输入身份。核心冻结状态应不受评价扰动，评价值应对适当扰动敏感。生产固定.5可通过实际调用验证；正式S调参职责留给R1B，不要求S选择对S不变。 |

这里指出的是验证覆盖与叙事缺口，不能据此反推所有核心生产算法都错误。Joint循环已关闭，不应再要求重写。

**P2 / R6-05：组身份在浮点归一化和JSON往返中仍会丢失（延续R5-07）**

[Decimal→float](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:51)不是无损映射。两个有限不同Decimal `1e-1000`和`2e-1000`都归一为0，原本两组变成SINGLE_GROUP。另在未提供expected_groups的显式object数组中，整数1与字符串"1"可形成两个内存字典键；json.dumps后同名键"1"重复，json.loads只保留一组。碰撞检查仅放在[expected_groups分支](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:100)，未覆盖观察组推断分支。

这些反例的primary_result_eligible为False，未发现其直接变成有效主结果；但兼容性/探索性输出仍会损坏。最小修复是明确拒绝不需要的Decimal及存在JSON身份碰撞的输入；若支持更多类型，完整结果应有可逆的类型化组表示。不要只检查json.dumps不报错。

**P2 / R6-06：交付生成器没有验证行为，最终答复与实际工件不一致（延续R5-08）**

- [生成器613–645行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r5_worker_20260915_015117Z_7bac5330/generate_delivery.py:613)仅核对符号位置及PASSED节点，UNKNOWN仍只Warning；并未检验trace字段和所声称的值。状态、零失败、健康标记等多处写死。R2-01公共FairEvaluator入口仍映射到helper测试；“有同名字段”不足以证明公开调用成功。F02说明还把q称decision threshold（[609行](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r5_worker_20260915_015117Z_7bac5330/generate_delivery.py:609)），应为决策概率，阈值另列。
- 本轮7个guarded入口对应7个保留运行（6 FAIL、1 PASS），另有1个改动前元数据目录；最终文件报告只列最终运行和生成器。附件含28条命令记录，其中生成器2次。缺少完整尝试索引及上轮7次调用/6个目录的未决对应说明。不能删除失败，也不能补造无法恢复的当时证据。
- 附件最终答复中**18项SHA-256声明全部与当前对应文件不符**；包内MANIFEST的24项则全部匹配。两种事实必须分开，hash差异不等于已证明篡改。实际源码10项改变；“13 files”“clean working tree”不准确，实际是未暂存且工作树有改动。24字段schema测试也不应写成一律≤8特征；≤8限制针对分类器输入。
- 生成器使用`w`覆盖报告、矩阵和MANIFEST，重复运行不会拒绝已有最终包。附件显示两次生成器调用，但没有两次完整退出输出，故不推断每次都成功覆盖。程序本身没有实现承诺的最终化只写一次。

证据：[交付与全部尝试核对](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R5_SUPERVISOR_REVIEW_20260915_evidence/delivery_verification.json)。应由同一份机器结果生成报告和最终答复；缺行为证据的条目保留PARTIAL/OPEN，严格校验缺字段、错节点和错误trace后才最终化。

**验证范围与项目意见**

worker日志确有81个PASSED节点，37项pre/post一致，106项前像与上一轮监督快照一致，包内24项manifest一致；上一轮170项监督manifest未变。最终日志1911条与telemetry闭合计数一致，1909是写摘要时截点，并非本次计数漏洞。独立诊断在常驻外层访问限制下执行，最多8行/分区、24个schema字段；实际LR/D8最多20行/分区、2个预测特征。没有读真实微数据、安装、联网、正式训练或修改生产文件，也未stage/commit。全套81项未再次运行；本次执行了针对缺口的生产调用、原测试和敏感性变异，避免把同一套有盲点的绿色结果当验收。

建议按[下一轮可直接交给Gemini的提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R6_REPAIR_PROMPT_20260915.md)完成三个有限批次：真实接口修复、实际行为覆盖、执行/交付证据闭合。保留已关闭项，禁止为让测试通过而弱化保护组、版本或标签契约。

应用论文路线保持：R1A基础契约通过→R1B FairBias-BM/AE/Joint与Unmitigated、RW、LFR、EG-DP、EG-EO、TO-EO在统一LR/GBDT和F/C/S/T协议下适配→R2调查设计与配对推断验收→冻结后的正式比较。FairBias是主方法，不预设胜出。已见过的2024结果只能回顾性评价，不能重新称为盲测。本轮P3建议是继续沿用既定对比矩阵及阶段验收，不增加方法数量或新数据集来掩盖尚未修好的基础问题。
