# Gemini：执行R1A-R6，补齐可被反例检验的契约与证据

你是implementation worker，Codex负责独立验收。项目为`/Users/lkc/Downloads/code_v_0_3`，目标是以FairBias为主方法发表医疗公平性应用文章，并可靠比较预测与公平性表现，允许FairBias不占优。

**本提示直接授权下述三个有限批次，按依赖顺序执行，无需先交计划等许可。R1A-R5未通过。本轮不进入R1B、不读真实数据、不跑正式benchmark、不stage/commit/push。保留全部历史工件及已关闭修复。** 每批内先用反例确认测试敏感性，再修实现/测试；不通过修改生产语义迁就错误fixture。

先阅读[执行协议](/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md)、[AGENTS](/Users/lkc/Downloads/code_v_0_3/AGENTS.md)、[GEMINI](/Users/lkc/Downloads/code_v_0_3/GEMINI.md)、[本轮监督报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R5_SUPERVISOR_REVIEW_20260915.md)和[上轮提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R5_REPAIR_PROMPT_20260915.md)。未被本提示调整的算法和边界继续适用。

读取监督报告同名`_evidence/`中的verification、evidence_verification、delivery_verification、source_snapshot、diff、probe_payloads、两次probe_run结果、final_verification和MANIFEST。首轮监测诊断有监督脚本参数遗漏，第二次已更正；不要当作生产bug。不要把结果中execution=OK理解为该契约通过。

**起点和权限**

- 确认分支`research/nhis-fairbias`、HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`、无暂存。核对本轮108项输入及前像；意外差异按路径报告，不重写预期hash掩盖。保护14个继承文件与.gitignore；既有.gitignore历史差异不处理，不读继承CSV。
- 允许修改：`src/nhis_fairbias/preprocessing.py`、`adapter.py`、`d8_enhancement_runner.py`；`src/fairbias/application_metrics.py`；仅实际行为反例发现具体缺陷时改`enhancement.py`、`enhancement_contracts.py`、`mitigation.py`、`evaluator.py`；两个既有guard/runner脚本；5个`tests/synthetic/test_r1a_*.py`及必要conftest。其余生产/benchmark/配置/旧测试/治理/scratch均只读。
- 新输出分别为独占`runs/r1a_r6_<UTC>_<id>/`、`docs/reports/fairbias_r1a_r6_worker_<UTC>_<id>/`。每次入口、节点调试、失败和生成器修订都保留；最终化后不可覆盖。stdlib静态AST/hash/交付生成器放在本轮新报告目录。
- 所有项目导入、调试和pytest必须在常驻guard下，以Framework Python `-I -S -B`执行。添加r1a-r6 phase和白名单单节点入口；不裸pytest、不临时PYTHONPATH、不跑全旧tests、不联网、不安装。
- 保持每分区≤200人工行、分类器≤8预测特征；preprocessing/adapter可用24字段完整schema但≤8行/分区，不在宽表上训练分类器。保留15分钟/4GiB采样监测，区分采样RSS与OS硬上限。
- 保留H=1、作者1998项BM幂流、AE六网格及既定算法模式，不改变论文统计计划。

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -S -B scripts/run_fairbias_r1_guarded_tests.py --phase r1a-r6 --output-root /Users/lkc/Downloads/code_v_0_3/runs
```

已关闭并应保留的行为：结局fallback移除与二元标签拒绝；失败load不清除legacy；坏export使原负例测试失败；真实Joint A→B→A cycle_detected并保留B；exact拒绝目标不匹配子路径/其他父目录；故障事件不清空主guard；活进程空RSS失败；缺/空/坏JSON执行证据拒绝；前像与candidate分离。不要重新实现这些已合格部分。

**第一批：修真实接口问题（R6-02、R6-03、R6-05）**

1. 按工件声明的既有regime建立完整加载验证。人工temporal fit→export→load→primary/expanded transform应等价；人工`fit(regime='pooled')`往返也应等价。temporal NHISStudyAdapter仍拒绝pooled来源。仅恢复既有软件兼容性，不授权pooled真实数据实验，不扩大其他来源可用范围。`allow_unspecified_source`的工件若保留，应有明确未验证身份且不能进入temporal生产adapter；不要静默改成2022。
2. 校验完整必需正文及类型：版本/identity/regime/year/role一致；row_count为真正的正整数，不截断小数、不接受bool；数值统计字段有限且彼此一致（median同numerical_medians、min≤median≤max、合法计数与row_count一致）；类别符合对应schema/特殊employment语义；employment分布与规则manifest一致。对有不同schema的legacy反例可显式拒绝，但不能自动认证。v1_legacy带v2 identity应拒绝矛盾或保留legacy，不自动升级。已验证对象连续失败load/fit后保持完整旧状态或明确整体失效；验证完成才提交。仅新人工工件，不迁移旧结果。
3. 统一fit与transform数值清理：8行年龄`[20,30,40,50,80,997,997,997]`不能静默产生65作为有效训练中位数。选择与当前预处理接口一致的清理后仅用有效F值拟合，或在边界明确拒绝非harmonized输入，并给出理由。验证NaN、正负Inf、schema非响应/越界码、全无有效值以及大写fallback；全无有效值失败并保留旧状态。正常上游harmonized输入不变。不得在本轮增加任意填充值、改变正式特征集合或决定新的缺失估计政策。
4. 组JSON身份：最小方案可拒绝Decimal及会产生JSON键冲突的异类型标识。若支持Decimal，必须无损、无上下溢出；完整结果json.dumps→loads后身份、组数及指标一致。覆盖有限Decimal `1e-1000/2e-1000`和object数组`1/'1'`，有/无expected_groups两条入口均验证。常见字符串/数值组及frozenset/未知对象拒绝继续回归。不要把“能序列化”当“无损”。

每个反例取真实8行fit得到的工件再修改目标字段，不用手填私有record替代合法正例。说明哪些异常影响变换、哪些仅影响来源或统计记录，不宣称真实NHIS已被污染。

**第二批：只补未完成的生产行为证据（R6-04）**

每项输出独占trace，含实际调用、参数、注入边界、人工行/列数、状态、候选/模型fit数及观察结果。trace字段由调用返回或spy记录生成，禁止手填预期“true”。

1. AE缓存：给同一engine提供实际O_train及合适人工组，让epsilon=.5→.2→.5时至少有候选通过真实公平性门并进入utility模型fit。固定F/C后记录baseline fit、候选fit、几何评估、tracker/cache命中类型、rejection_reason与选中changed_dict。返回旧配置究竟复用有效效用、复用拒绝记录还是重算，由真实实现证明；不要规定错误的“最多1次fit”来迫使实现迎合测试。保留F/C、分类器参数与scaler的已有缓存正例。禁止删除MISSING_PROTECTED_DATA_WITH_ENABLED_GUARD检查。
2. BM游标：保留已有六值人工流测试但诚实标为artificial stream；另从既有配置取完整1998项官方序列，两个engine仅revisit不同，每个至少两次实际搜索。记录transform实际指数、起始/结束cursor、接受位置。设计前几个指数早停；不要用截短poly_exponents或穷尽1998次模型训练控制成本。证明第二次不是空遍历，两种策略实际访问顺序不同。配置原件不改。
3. 终局：通过真实run_arm或保留原逻辑的生产调用覆盖`max_dphi < / == / > epsilon`的fairness_feasible；可注入几何数值来源，不能mock终局比较、返回值或整个run_arm。候选严格小于epsilon的既有测试保留。
4. NMI：生产公式分母有`+1e-10`，0.475并非phi=.05的浮点等值。使用明确记录的before/after及同一浮点计算所得phi_loss建立真实相等测试，必要时将该人工测试的phi_threshold设为计算出的边界；另保留名义.05邻域。临时将生产门的`<=`替换为`<`时，**同一边界测试必须失败**；正常实现通过。变异只在隔离测试进程内，不改磁盘生产文件。记录同一测试的正常/故障结果，修复后也不能吞掉AssertionError。
5. S/T：固定F和核心selection，独立S/T各自逐项扰动X、y、A、w。y/A值要真正改变；w要传入现有接受权重的评价接口，不能仅定义变量。不要为无权重API硬塞参数；现有survey加权函数可用于冻结预测后的评价，但这不是复杂调查SE/CI验收。真实transform/scaler/model的fit spy记录输入内容身份和来源，证明全部来自F，核心候选选择只用指定selection。分别检查评价前后冻结状态及适当指标敏感性。阈值从生产调用行为验证（目前D8固定.5），不能本地设.5自证。正式S用于配置选择的职责留R1B，不要求正式选择对S改变不变。
6. 公共指标：针对原R2-01，实际调用FairEvaluator.compute_metrics，核对返回字段和缺组/多组行为；helper测试不能单独关闭公开入口问题。把此调用的运行节点、返回字段和工件对应起来。

Joint循环已经有独立接受证据，作为回归保留即可。若需白名单外生产代码，精确列OPEN并完成无依赖部分，不扩展正式benchmark来制造覆盖。

**第三批：执行证据与交付都必须能被坏输入判失败（R6-01、R6-06）**

1. 让父控制器实际使用强制完整性检查：预执行清单、loaded路径、编译/执行捕获记录、选定pytest节点及runner建立明确集合关系。37个pre文件不应无条件全部要求执行；配置和未执行源码另列。必须能识别只保留一个合法模块的非空子集、漏一个已加载测试、额外未登记项目模块、错误类型/hash/bytes/声明计数。验证记录来源，不接受仅写一个loader_source字符串。
2. 真实捕获pytest文件和runner入口的加载/编译来源；不能在pytest结束后read_bytes补作“实际执行字节”。处理spec_from_file_location绕过meta-path、潜在pyc读取；`-B`不等于禁止读pyc。可使用经校验的启动加载方式、受控loader/编译钩子等局部办法；不新建通用OS沙箱或升级依赖。凡未能捕获的部分如实标明覆盖不足且不能宣布完整。
3. guard日志必须存在、逐行可解析，结构/决策合法，计数和最终截点一致。用人工日志验证：坏JSON、1行配999条摘要、DENIED配零意外拒绝摘要、截断/丢失记录等。预期拒绝需有足以对照操作/目标/原因/次数的机器证据；不能把所有DENIED都当故障，也不能只相信child_summary的clean布尔值。保留本轮合法1909摘要截点→1911闭合计数语义。
4. 对实际控制器裁决函数/路径跑合法完整正例和上述负例，检查最终FAIL及非零退出传播。缺/空/坏JSON旧负例继续保留。故障实例单独sink，主guard意外拒绝不得pop/clear。记录预检/解析/哨兵早期失败，不能只有成功pytest才创建记录。
5. 建立机器可验证的问题表：保留R2/R3/R4/R5全部原ID含义，新增R6-01…06严格对应监督报告。每条含真实符号位置、触发条件、执行节点、production call、trace路径/JSON指针、观察结果与注入边界；缺符号、节点未执行、字段不存在或值不满足要求均应使生成失败，不能Warning后FIXED_CANDIDATE。可以人工定义验收条件，不能人工写死运行结果。未知或不完整保持PARTIAL/OPEN。
6. 静态生成器也要有负例：临时副本去掉trace字段、改节点/符号、提供失败verdict，应拒绝最终化；正例生成一次后再次指向同一最终包应拒绝，不覆盖MANIFEST。所有副本限本轮人工目录，失败生成不伪装成成功包。
7. 完整attempt_index包含本轮改动前记录、每次guarded/单节点/失败/变异、正常测试及生成器尝试的命令、起止、exit、目录、源码身份。补充R5的28条附件命令、7个运行（6FAIL1PASS）、1个前像目录和2次生成器调用；无法恢复退出证据时标UNKNOWN。上轮R4的7调用/6目录差异也只做可证实的对应，不补造、不改旧包。新增补充说明即可。
8. 从机器材料生成当前diff与累计diff的区别、文件数、pre/post/loaded/compiled数、schema字段数/模型特征数、测试和guard资源结果、全部hash。用户答复只复制这些已生成事实；修正R5附件18个错误hash声明，保留旧附件不改。q始终表示决策概率，硬标签yhat和阈值另外命名。
9. 最终MANIFEST只生成一次、排除自身，覆盖报告、生成器、trace、输入身份、测试/日志、资源和attempt_index。定稿后需要修订则新版本目录。核对108项起始身份、预期diff、冻结文件、无暂存、历史manifest及链接。F01/F02保持OPEN，其他仍未完成的R1A问题也必须列出。

**交付格式与后续路线**

完成以上获授权工作后，依协议第9节使用精确标题：Gate:、Status:、Files changed:、Commands executed:、Permissions requested:、Tests executed:、Exact test results:、Input hashes:、Output hashes:、Row counts:、Assumptions:、Unresolved issues:、Git diff summary:、Proposed next step:。这是提交监督复核，不是worker自批gate。最后一行必须是：

`STOP — waiting for Codex review.`

R1A接受后另行授权R1B方法适配：FairBias-BM/AE/Joint真正从F学习，对比Unmitigated、RW、LFR、EG-DP、EG-EO、TO-EO，统一LR/GBDT、组类型、权重能力、F/C/S/T及p/q/yhat；不支持组合明确保留。R2验收全年调查domain、共享配对复制、20个主差值家族及B=2000等已定规则；R3获准后运行冻结76条件×5种子；R4对已见过的2024做冻结后的回顾性评价。当前不执行这些正式阶段，也不增加方法/数据集或预设FairBias胜出。
