# Gemini 执行提示：FairBias R1A-R7 来源恢复与定向修复（2026-09-15）

你是Gemini Implementation Worker。Codex对R1A-R6的判定是**REJECT本次验收交付，返回Preflight**。继续保留当前候选源码作为修复输入；旧交付、失败运行和监督证据全部只读。此提示授权下面三个批次顺序执行，无需每个批次再向用户请求许可；任一批次发现越界需求或输入身份异常时，记录并完成无依赖工作，再按协议报告。不得自批gate或进入R1B。

先读[执行协议](/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md)、[AGENTS](/Users/lkc/Downloads/code_v_0_3/AGENTS.md)、[GEMINI](/Users/lkc/Downloads/code_v_0_3/GEMINI.md)、[R6监督报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R6_SUPERVISOR_REVIEW_20260915.md)、[生效前序提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R6_REPAIR_PROMPT_20260915.md)。当前提示只收敛剩余缺口，不撤销已定算法/数据边界。

读取R6监督同名_evidence目录的verification.json、evidence_verification.json、delivery_verification.json、governance_verification.json、probe_run_031408_588586/results.json、diagnostic_execution_record.json、final_verification.json与MANIFEST.json。诊断9组execution=OK只说明执行完成，里面accepted=true有时恰好表示坏输入被误收。不要把这些值自动转成FIXED。

**固定权限及起点**

- 分支`research/nhis-fairbias`，HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`，无暂存。核对本轮110项监督输入与当前源码；前像必须是R6结束时的candidate，不是67e6659或更早worker版本。异常按路径报告，不改预期hash。
- 14继承文件及.gitignore保持原状；既有.gitignore历史差异不处理。不读取继承CSV，不修改治理文件、配置、benchmark代码、旧实验和历史工件。
- 可改生产：`src/nhis_fairbias/preprocessing.py`、`src/fairbias/application_metrics.py`；仅确有本轮已授权行为缺陷时改`src/nhis_fairbias/adapter.py`、`d8_enhancement_runner.py`及`src/fairbias/enhancement.py`、`enhancement_contracts.py`、`mitigation.py`、`evaluator.py`。优先正确调用现有生产入口来补测试，不能为制造覆盖新建另一份终局逻辑。
- 可改两个既有guard/runner脚本及5个`tests/synthetic/test_r1a_*.py`，必要conftest只作人工fixture/guard支持。新静态审计/生成脚本和材料放本轮新报告目录；不写scratch或根目录工具。
- 新run：`runs/r1a_r7_<UTC>_<id>/`；新报告：`docs/reports/fairbias_r1a_r7_worker_<UTC>_<id>/`。每次入口/单节点/失败/变异/生成器修订独占新尝试目录；已创建的最终包不覆盖、不删除文件后重建。
- 所有项目导入、调试、pytest和单节点都经过常驻guard，Framework Python `-I -S -B`。不裸python导入、不直接pytest.main、不通过sys.path/PYTHONPATH绕过入口。stdlib纯静态AST/hash/生成器允许直接执行；不得在其中动态import/exec项目源码。
- 只用人工数据；模型每分区≤200行、≤8预测特征；preprocessing/adapter的完整schema例外仍为24字段、每次fit分区≤8行，不在宽schema上训练分类器。pooled测试也按合并后的fit分区计算，不能把3×8=24称为≤8。
- 保留15分钟/4GiB采样监测，说明其为采样控制；不扩大到通用OS沙箱。网络/安装/真实数据/正式模型实验/暂存/提交/push均未授权。
- 保留H=1、完整1998项BM幂流、AE六网格、既定revisit/几何策略和正式统计计划。禁止为追求FairBias胜出更改模型选择目标、阈值或测试数据。

先静态添加r1a-r7 phase和对应白名单后，唯一项目执行入口为：

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -S -B scripts/run_fairbias_r1_guarded_tests.py --phase r1a-r7 --output-root /Users/lkc/Downloads/code_v_0_3/runs
```

单节点也只能走这个入口并使用完整、白名单允许的node ID；每次有独立run_id。完整套件通过且反例验收满足后停止重复测试；无新修改/失败不继续扩大检查。

**第一批：恢复历史记录和原问题身份（R7-01、R7-04）**

1. 从现有监督110项快照建立R7起始前像、当前Git状态与候选身份。按路径登记此前10项变化，不整体回退源码；本轮重新登记后再动授权源码。
2. 新建HISTORY_RECOVERY.md和机器表，明确R6附件记录的25个未受保护尝试（23项目导入/调试、2pytest）与13文件删除再生成。仅补充说明，不“修复”旧附件或重写旧MANIFEST。不能恢复的旧bytes、退出码、准确起止时间标UNKNOWN，附证据来源。不能声称这些尝试全部成功，也不能从服务stream错误推断仓库外联。
3. 当前R6必须对齐7个guarded run（4PASS/3FAIL）、7次直接生成器入口以及附件72条命令；命令类型分别统计，不能加总重复计为独立模型运行。补上025804Z_781ba5b9及025829Z_df7c841e两个FAIL。恢复R5的013308Z_80800752=前像元数据目录，015040Z_1bff8a4e=FAIL/1。以原telemetry/监督快照/附件为证据，禁止编退出状态。
4. 原F01恢复为“主benchmark没有真正从F学习FairBias BM”，状态OPEN、后续R1B；指向真实FairBiasAdapter方法。多组扩展若保留建议，另设新ID，不挤占F01。F02仍OPEN：p为风险概率、q为决策概率、yhat为硬标签，threshold单列。新测试中硬标签变量也用yhat，别继续把0/1阈值结果命名q。
5. R2/R3/R4/R5/R6原ID含义继续保留；按R6监督closure表分清CLOSED、PARTIAL和OPEN。新增R7-01…06严格对应监督报告。已关闭的R6-03、BM官方流、缓存效用、NMI边界、公共compute_metrics、既有Joint循环及旧原子性修复作为回归保留，不重写实现。
6. 同一已最终化目录二次生成必须拒绝；拒绝后可以写独立attempt记录，但不能删除MANIFEST解锁。生成器源码修改后使用新版本目录，可引用同一次未改变代码的受保护run；报告版本和执行run不要求一对一，必须明确关系。

这一步是如实恢复来源，不可能抹去过去的协议违规。R7的验收依据是新的合规执行与候选行为，不宣称R6历史已经变成全程受保护。

**第二批：仅修复明确剩余反例（R7-02、03、05、06）**

A. R7-02：执行证据完整性和日志判定

- 父控制器实际调用的验证器必须接收并对账运行时loaded项目源码、捕获的编译字节、预执行登记、所选测试文件/节点及入口。配置和未执行源码单独列示，不强迫37个pre文件全都加载；但任何实际加载的项目模块缺编译记录必须失败。最低15/1不能充当集合完整性的标准。
- 完整合法材料通过；复制材料后删除enhancement.py并同步减少声明数必须失败；单节点材料删掉对应测试文件必须失败。继续覆盖缺/空/坏JSON、额外未登记模块、hash/bytes/类型错误、测试节点与记录不对应。测试通过真实裁决路径，证据错误必须传播成最终FAIL及非零进程退出。
- pytest和入口捕获必须绑定实际编译所用源码缓冲区。当前“读取fn记录→原rewrite再次读取”不够；采用局部受控加载/编译包装，保留source_to_code使用同一bytes的正确实现。覆盖spec_from_file_location与项目pyc；不升级pytest/依赖，不新建泛化框架。无法做到的覆盖诚实PARTIAL，不写字符串loader_source就算完成。
- guard摘要所需字段必须存在且类型正确，计数为非负真整数而非bool/字符串；日志每条结构、decision、操作/目标/原因可对账。
- DENIED应有独立的预期登记/消费证据：事件身份、操作、规范化目标、原因、次数与阶段相符。不只信unexpected_denials=[]。故障测试使用人工副本或独立实例，主guard异常不能pop/clear。
- 摘要截点、闭合末尾及具体事件数量可精确对齐；不要用任意允许多50条替代解释。合法2068→2070类型的实际闭合例应继续通过；人工50条配total1、缺mandatory计数、未登记DENIED、丢行/重复/截断应失败。不要因正常预期哨兵拒绝使合法运行全失败。
- 保留预检、解析、哨兵及早期失败记录，不能只在pytest成功后生成来源证据。

B. R7-03：让同一测试能发现真实终局/隔离故障

- 直接运行真实D8.run_arm终局路径，或将原有生产终局判定抽成被run_arm实际调用的单一函数后测试该函数及接线；不得测试里另写<=。注入人工几何/小数据/昂贵外围来源可以，不能mock目标终局比较、fairness_feasible返回或整段run_arm。若局部抽取，保持原行为并覆盖所有接线位置。
- 覆盖max_dphi<、=、>epsilon，实际等于时应可行。对生产目标比较做进程内<=变<故障，同一正常边界测试必须失败，正常实现通过。记录生产调用计数与返回，禁止用本地mutant函数单测自证。
- S/T固定F及指定核心selection，在冻结后对S、T各自分别只变X、只变y、只变A、只变w，其他输入保持不变。不同变量对应适当指标：y影响错误率、A影响组指标、w影响加权指标，X变化在人工非恒定预测fixture下影响预测。不是要求每个指标在每个扰动下都变化。
- w-only例用非恒定二元值并手算比例，例如yhat=[0,1]、w=[1,3]时0.75，w=[3,1]时0.25。调用真实weighted_binary_proportion并断言定量期望；生产函数忽略权重的隔离变异必须使同一测试失败。此处不修改survey.py，也不声称完成复杂调查方差验收。
- spy在核心AE/模型/scaler执行前安装，覆盖baseline与候选fit；核对实际X内容/索引、y、必要weight以及由F变换得到的预期输入，不能仅断言形状。记录C仅参与核心候选评价，S/T冻结后不触发任何fit/重新选择，评价前后模型/变换/阈值状态一致。真实无fit的组件按实际接口检查冻结参数，不虚构fit方法。
- 注入一个同形状非F输入或错误fit来源的故障后，同一身份测试必须失败。不要扩大到正式benchmark配置选择；正式S用于选配置的职责留R1B，不能把“对S永远不敏感”写成正式协议。
- 生产0.5阈值、实际y/A变化、有效w调用、NMI真实边界、完整BM幂流、缓存候选效用已获独立正例；保留并复用，不需重新发明测试。

C. R7-05：工件统计及来源身份

- 从≤8行真实人工fit→export合法工件出发，每个负例只改目标字段。要求count_valid、count_missing为合法整数，与row_count相加一致；missing字段/字符串/bool/负数/矛盾计数必须拒绝。mean在min/max之间，数值统计与对应schema合法实值范围一致；median与冻结中位数仍一致。保持正常temporal/pooled primary+expanded往返。
- 当前五个负例必须拒绝：缺count_missing；字符串count_missing；valid8+missing8但row8；mean999且min=max18；年龄min=-999/max999。不据此宣称历史imputation或真实结果已经污染。
- allow_unspecified_source若保留，应fit/export/load身份一致，始终明确未验证并被temporal adapter拒绝；不得标成已验证2022。也可在明确的软件接口边界拒绝该不支持导出用法，说明兼容性影响，不能悄悄生成自身不能加载且自称verified的工件。
- 失败load/fit保持原完整状态或明确整体失效；旧legacy不自动升级；历史工件只读。将pooled测试合并后的fit规模改为≤8行，正常往返不退化。

D. R7-06：完整结果的组JSON身份

- 规范化numpy.integer/numpy.bool_/numpy.floating到明确支持的Python标量，有限性与二元/多组检查保留。不能只在json.dumps临时加default=str而改变组语义。
- 按实际JSON对象键编码识别[True,"true"]等冲突，可明确拒绝混合类型或采用明确无损表示；不能用str(True)的大小写差异判为安全。
- 显式expected_groups和从数据推断两入口都覆盖：numpy整数、普通字符串/数值、True/"true"、1/"1"、Decimal极小数、非有限/容器。支持类型的完整返回结果dump→load后组身份/数量/指标一致；不支持或有歧义类型清楚拒绝，不能两组落盘变一组却primary_result_eligible=true。
- 保留既有Decimal拒绝和1/"1"冲突拒绝，不新增Decimal精度计算功能。

**第三批：受保护验证与一次最终化**

1. 每个开放R7问题对应实际源码符号、完整node、production call、trace路径/JSON指针、输入身份、实际输出和故障注入边界。行为trace由实际返回或spy计算，禁止手填fit_all_from_f=true等结论。闭合表的术语与原ID一致；合格代码仍只是FIXED_CANDIDATE等候监督。
2. 为实际生成器验证：合法材料可最终化；移除必需trace字段、改生产符号、改为未执行节点、给失败verdict均拒绝。二次指向同一最终目录拒绝。测试调用实际校验/生成路径，不在self-test中复制一个if再称主入口已测；只能向新人工子目录写。
3. 由实际材料生成测试/哨兵/guard/退出/资源计数，不把0失败、13哨兵等写死到报告。单项AST+PASSED不能代替行为证据；未满足仍PARTIAL/OPEN。生成器本身可登记所要求的条件，但观察值必须来自运行。
4. R7 attempt_index从每次run和生成器attempt收集，保留命令、阶段、起止、退出状态、源码/生成器身份、输出目录及父子关联。负例“预期失败”仍记录实际非零或异常结果；不是把所有尝试都标PASS。报告最终化后另一次拒绝测试结果可放外部独占attempt并由新封装目录引用，不能回写最终包。
5. 精确区分R7本轮candidate.patch、相对既有历史基准的累计差异；若累计diff只含git已跟踪文件，明确标tracked-only。需要完整候选差异时补未跟踪新增源码/测试清单及内容身份，不能把普通git diff当全仓快照。
6. final_verification核对110项起始输入的预期差异、保护文件、分支/HEAD/无暂存；R6监督MANIFEST及当前所有历史worker/run指纹应不变。输出新增/修改白名单内路径并说明原因，确认没有范围外源码变化。
7. 定稿MANIFEST只写一次、排除自身，覆盖最终报告、生成器、trace、来源快照、测试日志/资源/裁决和attempt_index；必要外部工件以明确路径hash纳入。需修订就新版本；绝不删除旧文件绕过write-once。最终回复的hash直接取机器材料，不手抄其他版本。

**验收边界与后续论文路线**

本次只完成来源恢复与有限软件反例。R1A接受后再由Codex授权R1B：FairBias-BM/AE/Joint从F实际学习，比较Unmitigated、RW、LFR、EG-DP、EG-EO、TO-EO，统一LR/GBDT、数据适配、组类型/权重能力、F/C/S/T、p/q/yhat、拟合预算和阈值冻结；unsupported组合显式保留。R2再验收全年调查domain、共享配对复制、20个主差值家族及B=2000等已定规则；R3获准后运行冻结76条件×5种子；R4对已见过2024做冻结后回顾性评价。旧master/registry的具体条件定义保持有效，顶层数量不是授权随意替换条件。

以FairBias为论文主方法，不预设它获胜；比较公平性、预测及两者权衡，所有对照按统一预定选择和评估规则。不要在R7新增模型、扩大真实数据测试或复用旧失效排名证明优势。

**标准交付**

按协议第9节使用精确标题：Gate:、Status:、Files changed:、Commands executed:、Permissions requested:、Tests executed:、Exact test results:、Input hashes:、Output hashes:、Row counts:、Assumptions:、Unresolved issues:、Git diff summary:、Proposed next step:。

回复先给真实结果、未完成问题、最终报告/清单/attempt路径。F01/F02必须OPEN，R7未解决项如实列出；不得把本轮技术缺口改成R2统计问题后删除。不要输出ACCEPT或授权提交。最后一行：

STOP — waiting for Codex review.

