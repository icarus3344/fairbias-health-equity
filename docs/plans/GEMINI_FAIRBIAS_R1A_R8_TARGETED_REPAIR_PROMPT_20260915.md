# Gemini 执行提示：FairBias R1A-R8 定向修复（2026-09-15）

你是Gemini Implementation Worker。R1A-R7本次交付判定为**REJECT，返回Preflight**。当前候选代码保留，已独立关闭的修复保留；本轮只完成下面有限剩余事项，不重写算法、不进入R1B。

先读[执行协议](/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md)、[AGENTS](/Users/lkc/Downloads/code_v_0_3/AGENTS.md)、[GEMINI](/Users/lkc/Downloads/code_v_0_3/GEMINI.md)、[R7监督报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R7_SUPERVISOR_REVIEW_20260915.md)及[前序R7提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R7_RECOVERY_REPAIR_PROMPT_20260915.md)。未被本提示收敛的边界继续有效。下面批次已授权顺序执行，不需反复请求用户确认。

读取R7监督_evidence目录中的verification.json、evidence_verification.json、delivery_verification.json、probe_run_070015_438338/results.json、final_verification.json及MANIFEST.json。诊断7组执行完成不代表7组契约通过。特别注意：生成器“失败材料仍写86通过”和“二次覆盖”是实际main在人工副本中的观测；不得忽略。

**起点、白名单与不可重复的错误**

- 分支`research/nhis-fairbias`，HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`，无暂存。核对R7监督112项输入及其快照，记录R8初始身份；不能把重读旧快照的hash说成已经证明更早缺失命令的执行身份。
- 可改：`scripts/_fairbias_r1_guard.py`、`scripts/run_fairbias_r1_guarded_tests.py`、`tests/synthetic/test_r1a_guard.py`、`tests/synthetic/test_r1a_identity_and_probabilities.py`、必要人工fixture支持；`src/fairbias/application_metrics.py`只修R8-01普通字符串JSON键过度规范化。
- 其他生产源码、配置、benchmark代码、旧测试、治理文件及scratch本轮只读。若在真实内部fit断言中发现具体生产bug，先记录触发与路径并完成无依赖工作，不擅自改白名单外文件。无需修改enhancement_contracts.py就可用spy发现错误输入。
- 14继承文件及.gitignore完全保持；既有.gitignore历史差异不处理。所有R7及更早run、报告、附件和监督证据只读，不移动/删除当前scratch生成器，也不覆盖历史错误报告。
- 新run目录`runs/r1a_r8_<UTC>_<id>/`；新报告/生成器/静态验证材料放`docs/reports/fairbias_r1a_r8_worker_<UTC>_<id>/`。生成器必须在本轮目录并纳入最终清单；不要再写scratch。
- **所有项目导入、AE/LR调试、pytest及单节点一律走常驻guard入口。** 不直接python -c导入fairbias/nhis_fairbias，不注入sys.path或PYTHONPATH绕过入口。人工数据、-I -S -B不等于guard已安装。入口不能支持调试时先静态补白名单节点，随后从入口运行；不能先裸运行再补日志。
- stdlib纯静态AST/hash/日志分析和交付生成器可直接执行；不得借此动态import/exec模型项目源码。生成器人工负例仅在新独占目录，不需项目模型导入。
- 模型每分区≤200人工行、≤8预测特征；完整preprocessing/adapter schema仍为24字段、每fit分区≤8人工行。本轮不需新增预处理实验。保留15分钟/4GiB采样监测并准确说明局限。
- 不联网、不安装、不读取真实微数据、不跑旧全量tests或正式模型实验、不暂存/提交/push。保持H=1、完整1998项BM幂流、AE六网格、几何/revisit策略、正式实验与调查估计计划。

先静态添加phase及白名单，所有项目执行使用：

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -S -B scripts/run_fairbias_r1_guarded_tests.py --phase r1a-r8 --output-root /Users/lkc/Downloads/code_v_0_3/runs
```

单节点从同一入口传完整node ID，每次新run。不要更改已关闭的数值清理、工件正文/unspecified、numpy标量、终局函数、NMI/BM/cache/公共指标等实现以迎合报告。保留原回归，新增针对下列反例的必要测试即可。

**批次一：补全来源并修复实际裁决路径（R7-01、R7-02）**

1. 新补充记录当前R7附件中的6次直接项目调试尝试（103、139、179、237、270、298行）和scratch生成器创建/执行；保留UNKNOWN退出码。附件从首个run之后开始，注明可见范围，不能宣称掌握全部历史。R7磁盘5个run为4FAIL/1PASS，附件仅包含其后4次guarded入口，分别对账。
2. R6的7条生成器调用逐条实际退出码未恢复，改在新补充表标UNKNOWN；不能保留无依据的6次0/1次1。旧记录不改。对更早其他gate直接执行是否违规，只有明确授权与事件证据才作判断。
3. 父控制器应强制存在合法的loaded与compiled两份证据。对loaded的缺文件、坏JSON、空对象/空sources、错误结构/声明数/hash/bytes/类型都拒绝，不能except后把空字典送去包含检查。
4. 用实际集合关系绑定预执行、loaded、compiled、入口与选定测试；loaded条目的hash/bytes也须校验，不只比较键。保留完整loaded时拒绝漏enhancement.py、单节点拒绝缺对应测试的已有效代码。37个pre文件中未执行源码/配置不用硬凑进loaded。
5. 在实际父裁决路径证明：完整材料通过；缺/坏/空loaded同时删compiled enhancement.py仍必须FAIL；loaded的sha=wrong、bytes=-1必须FAIL。失败传播到非零退出和execution_verdict，不能只测helper默认参数。故障用人工副本或受保护注入，正式run材料不被覆写。
6. 日志mandatory字段必须存在且类型正确：log_healthy必须真正bool且为true，unexpected_denials必须真正list且为空；不接受字符串“false”或字段缺失。计数非负整数只是必要条件，必须同时满足total、allowed、denied、cutoff的真实关系。
7. 实际事件与预期登记/消费记录按操作、规范化目标、原因、次数、阶段/身份关联；不能只比较消费表抄下的三字段而忽略expected_action/expected_target_pattern。记录定义应简单且可核验，不新建通用安全框架。
8. 合法摘要→闭合差值按实际事件逐项解释；不再用“≤2”充当闭合证明。保留合法2095→2097类型正例；删除实际尾部两事件、总数改0/allowed改999999、缺unexpected列表、log_healthy字符串、消费记录指向无关原登记时均拒绝。
9. pytest同缓冲区AST/compile修复保留。入口初始读取仍不能冒称首次编译捕获；用局部受控启动绑定实际读取与编译字节，或明确未覆盖并保持PARTIAL，不能靠标签entrypoint_initial_load自证。禁止依赖升级或扩大为OS沙箱工程。
10. 主guard异常不清空，日志坏输入和故障用独立实例/sink；预检、解析及早期失败也保留独占attempt。

**批次二：补内部fit身份和剩余小接口回归（R7-03、R8-01）**

1. 新AE入口spy已经验证入口X/y，但不代表内部fit。直接在实际模型/MinMaxScaler等拟合调用处记录并断言输入；AE执行前安装，覆盖baseline及所有候选。核对由当前changed_dict从F变换、再按F拟合缩放得到的实际X内容、y/索引和适用weight，不只看形状、fit_source字符串或入口数据。
2. 建立同一测试的正常/故障两次结果：正常通过；仅AE内部模型fit特征整体+0.125、保持形状/y不变时，原身份测试必须失败。监督已观察这种故障改变26次fit仍通过，必须关闭此反例。spy观测不能替换被测逻辑，trace结论由真实记录计算。
3. 同样覆盖scaler实际fit来源，能够发现同形状非F输入。候选允许对F做当前合法变换，不能错误要求所有候选fit_X字节都等于未经变换的F。
4. S和T各自逐项扰动X/y/A/w；S已有有效的分项测试，补T分项即可。冻结后不触发fit/重新选择，保持变换、模型及阈值状态。每个变量选择恰当的指标，不要求所有指标都变化。正式S用于选择配置的职责留R1B。
5. 终局<=生产函数与忽略权重的原测试故障敏感性已独立通过，不要重写。现有手算0.75/0.25例保留。模型内部fit身份、测试边界与复杂调查方差验收分别表述。
6. R8-01只修改_json_group_key对普通字符串的处理：字符串保持原值和大小写/空格；布尔值才转换成实际JSON对象键true/false。True布尔与“true”字符串仍拒绝，但“True”与“true”两字符串应可区分。用显式expected_groups和数据推断两入口检查完整结果往返。numpy整数、1/'1'、Decimal和非有限拒绝继续回归。
7. R7报告声称的两个JSON node当前不存在，不要只改报告说已执行。复用真实存在且覆盖要求的测试，或在允许的identity测试文件新增实际测试，再从真实PASSED记录引用。

**批次三：修复交付校验器，不再手写成功结果（R7-04及R7-01交付部分）**

1. 新生成器置本轮报告目录，可复用前轮已审阅的纯stdlib校验逻辑；不要再次从零写只拼Markdown的版本。实际main在任何复制/写入之前检查目标最终包不存在，最终MANIFEST用排他创建；失败/修订另开新目录。
2. 实际main读取并校验preflight、execution_verdict、test_summary、日志/资源核验、输入身份和真实节点列表；任一FAIL、缺失、矛盾或未执行必需测试，不允许写FIXED_CANDIDATE成功包。允许在独占失败目录输出FAILED/INCOMPLETE诊断记录。
3. 每个闭合条目核对真实source/symbol、完整node存在且实际执行、production call、trace JSON指针及值。未知符号/不存在node/空trace/少字段必须拒绝。不把trace里手填true当行为证据。
4. 测试实际main：合法材料能一次最终化；FAIL verdict+exit1+0通过日志+7个空trace应拒绝；每类故障还应逐项注入以确认对应检查生效。第二次指向同一最终包应拒绝并保持所有原字节不变。外部失败attempt保留实际退出记录，不覆盖已完成包去增加“自检通过”字段。
5. 从实际机器材料生成86或将来的新测试数、失败数、哨兵、计数、资源及hash，不能在字符串模板中写死本次期望值。用户回复直接读取最终事实清单，尤其MANIFEST hash不要手抄其他版本。
6. 恢复原问题身份：带入原ID和原含义，可在新矩阵用明确引用继承旧状态，不必复制旧报告。R6-03是数值fit/transform清理，不是BM游标；当前漏掉的29个旧ID按监督列表恢复。已关闭项保留CLOSED及依据，未完成PARTIAL/OPEN；新增兼容性问题只用R8-01。
7. F01保持“主benchmark未从F学习FairBias BM”OPEN/R1B。F02保持“p/q/yhat混用导致BA/EO/DP比较对象不同”OPEN/R1B，不能弱化成变量改名。旧排名不因此获得效力。
8. R7三条不存在node必须在新矩阵改为真实证据；若没有证据就OPEN/PARTIAL，不能改成另一个问题后关闭。历史逐条退出码无证据就UNKNOWN。
9. attempt_index分别记录运行、单节点、故障、生成器和静态分析，命令/起止/退出/目录/源码身份和父子关联可追溯；当前附件缺失的命令历史范围明确保留。不能只给5个run列表就说完整。
10. candidate.patch按R7监督112项前像生成；累计git diff只覆盖tracked文件时明确tracked-only，另列新增未跟踪源码和测试身份。生成器本身是本轮文件变化并纳入清单，不能漏算。
11. final_verification核对112项输入的预期变化、白名单、分支/HEAD/无暂存、保护文件、R7监督MANIFEST及历史worker/run指纹。最终MANIFEST覆盖报告、生成器、实际负例测试证据、trace、来源、资源/裁决及attempt_index，排除自身，只写一次。外部独占工件以明确路径/hash引用，不把不同版本混装。

完成必要测试且没有新修改/失败后，停止重复测试。若仍有要求未实现，准确报告PARTIAL/OPEN，不用新文案把缺口改成“已修复”。

**论文与正式实验路线保持**

本次不推进正式实验。R1A接受后由Codex另授权R1B：FairBias-BM/AE/Joint真正从F学习，对比Unmitigated、RW、LFR、EG-DP、EG-EO、TO-EO，统一LR/GBDT的数据适配、组类型/权重能力、F/C/S/T、p/q/yhat、调参预算和阈值冻结。R2再验收全年调查domain、共享配对复制、20个主差值家族和B=2000等既定规则；R3获准后运行冻结76条件×5种子；2024已被查看，R4做冻结后的回顾性评价。保持master/registry具体定义，不用总数不变掩盖换条件。unsupported组合显式列原因。

以FairBias为应用论文主方法，比较公平性、预测性能及其权衡；不预设胜出，不用旧失效benchmark结果倒推调参。

**标准交付**

严格使用协议第9节标题：Gate:、Status:、Files changed:、Commands executed:、Permissions requested:、Tests executed:、Exact test results:、Input hashes:、Output hashes:、Row counts:、Assumptions:、Unresolved issues:、Git diff summary:、Proposed next step:。

提交最终报告/清单/attempt路径、真实结果、已关闭与未关闭事项。不能自批ACCEPT、不能授权提交。最后一行：

STOP — waiting for Codex review.

