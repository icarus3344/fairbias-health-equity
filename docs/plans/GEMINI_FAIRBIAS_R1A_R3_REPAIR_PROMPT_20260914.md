# Gemini：执行 R1A-R3，先建立真实反例，再完成修复和证据闭环

你是 implementation worker，Codex 是独立监督者。用户要以 FairBias 为主方法完成医疗公平性应用论文，比较预测表现与外部公平性方法。你的目标是实现可信流程，不能把 FairBias 获胜当作通过条件。

**本提示直接授权下面四步的实际工作，按顺序完成，不再只提交实施计划等待许可。R1A-R2 未验收；本轮只做 R1A-R3 基础契约返修，不是正式实验阶段 R3，不进入 R1B。保留候选和所有失败记录，不回滚、不删除、不 stage/commit/push。**

## 0. 必读和范围

先读 `docs/AI_EXECUTION_PROTOCOL.md`、`AGENTS.md`、`GEMINI.md`，然后读：

- `docs/reports/FAIRBIAS_R1A_R2_SUPERVISOR_REVIEW_20260914.md`。
- 同名 `_evidence/verification.json`、`evidence_verification.json`、`final_verification.json`、`MANIFEST.json`、`source_symbols.json`、`diff_from_pre_R1A_R2.patch`。
- `_evidence/probe_run_134558_194177/worker_case_output/results.json` 与该 probe_run 下的 `probe_source.py`。13组是诊断，不是13项验收通过；监督第一次序列化失败不是生产缺陷。
- `docs/plans/GEMINI_FAIRBIAS_R1A_R2_REPAIR_PROMPT_20260914.md`。其未取消的算法保真、人工数据规模、网络/文件边界与原问题ID继续适用。本提示明确改变处优先。

分支须为 `research/nhis-fairbias`，HEAD 为 `67e6659fa65249a8842e34af5d8969629efe4bca`，无stage。核对本监督 verification 的87项输入身份，先在新审计目录保存紧邻本轮的前像、hash及Git状态。哈希不一致报告具体路径，不改预期值掩盖。上轮监督保存的前像可只读引用，但不能把更早的scratch清单冒充本轮前像；重建材料标生成时间和 `RECONSTRUCTED_FROM_SUPERVISOR_SNAPSHOT`。

保护14个继承文件及.gitignore；只读Git元数据核验，不读原始CSV来算hash。既有.gitignore的b595e59历史差异保留。只允许改：

- `src/fairbias/{enhancement_state,enhancement,enhancement_contracts,bias_metric,evaluator,config,mitigation,prediction_contracts,application_metrics}.py`。
- `src/nhis_fairbias/{adapter,preprocessing,survey,d8_enhancement_runner}.py`。
- `scripts/_fairbias_r1_guard.py`、`scripts/run_fairbias_r1_guarded_tests.py`。
- 五个 `tests/synthetic/test_r1a_*.py` 和必要的 `tests/synthetic/conftest.py`。

不要为了改完名单而修改无缺陷文件。禁止改benchmark包、旧tests、历史脚本、配置原件、治理、旧计划/报告、release/既存runs。禁止读真实微数据、网络、安装依赖、旧103测试或全tests。新增输出仅限独占 `runs/r1a_r3_<UTC>_<id>/` 和 `docs/reports/fairbias_r1a_r3_worker_<UTC>_<id>/`；不写scratch，不覆盖最终化文件，失败尝试全部保留。

所有项目导入、调试和pytest均经常驻guard，父子解释器用Framework Python `-I -S -B`；新增phase `r1a-r3`。若支持单模块/节点调试，只允许从五个测试文件中枚举选择，同样经过哨兵和完整记账。只读stdlib AST/hash及文档生成无需导入项目。不要为这些工作运行任意入口外项目命令。

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -S -B scripts/run_fairbias_r1_guarded_tests.py --phase r1a-r3 --output-root /Users/lkc/Downloads/code_v_0_3/runs
```

## 第一步：建立能在当前候选上失败的真实回归测试

先增加下面真实反例，保留一份受控的“修复前预期失败”运行。该运行不宣称验收通过；失败不意味着可以关闭guard或扩大读文件范围。随后修实现，使同一行为测试通过，记录测试与源代码各自身份。

| 编号 | 真实调用与判定 |
|---|---|
| T-D8 | `compute_application_group_fairness`→`compute_group_fairness_gaps`→真实LR `evaluate_representation`。覆盖正常两组、缺预期组、无正/负支持和未声明组；每个分支都有完整字段。当前应复现KeyError。再用人工provider实际执行run_arm，核实四条评估路线传递组全集与来源。 |
| T-FROZEN | 实际NHISPreprocessor.fit后，分别修改输出列序、嵌套拟合中位数、registry/specs，必须拒绝修改或使用真实冻结副本，结果不得静默改变。保留此前外部字典不变性、列表错配拒绝的正例。 |
| T-LOAD | 真实用年龄0–100规则在8行年龄90人工数据上fit并export；官方规则对象load该工件必须拒绝不匹配，不能被adapter重新认证。合法同schema往返结果一致；缺拟合规则身份的旧格式明确遗留/不合格。 |
| T-GROUP | a/b/c三组、仅声明a/b：不得静默给主EO=0；完整声明EO=1。缺组返回null与资格/原因；拒绝字符串冒充序列、Decimal Infinity及不支持组对象，合法标量类型有正例。 |
| T-CONFIG | 旧合法keyword实际影响指纹；未知keyword/拼写错误明确拒绝。target_fairness等本轮新增但未消费参数不能静默接受。任意object不能成为记录ID。 |
| T-LOG | 用本轮新造日志验证路径和整数fd别名都不能从普通io.open等入口写入；预期拒绝没发生/次数错误/目标错误必须失败。保留原dir_fd真实哨兵的正确阻断。 |

预处理例可完整24字段、每分区≤8行，不在宽表上训练classifier；其他真实classifier/AE/BM≤8预测特征、每分区≤200行。仅可注入人工运输/历史全年人数检查，不能伪造拟合统计当真实fit正例。真实LR正例至少一个。错误要在规定层抛出指定类型；不能接受任意异常作为正确拒绝。

## 第二步：修生产契约，保持原来已修好的内容

### 2.1 D8返回schema与公平性组定义（R3-01/R3-03，原R2-04/R2-08）

统一所有helper返回分支：DP/EOpp/EO值、独立estimable/status、expected_groups、expected_groups_source及主估计资格/原因均有明确定义。`expected_groups_source` 必须对应实际来源：冻结arm/schema或observed-only；不能把任意调用方输入自动标成已预注册。D8消费必需字段时不能用默认值掩盖缺项。保留legacy EO与应用EO不同公式/名称。

预期组是该评价域内的完整组全集。出现额外观察组应明确拒绝或通过已声明的domain机制处理，不能无声忽略。R1A本轮采用拒绝非预期组即可，不为此扩展研究估计目标。需要研究域改变时另报具体问题。

分清 `primary_estimand_declared`（目标已声明）、完整组/标签支持与 `primary_result_eligible`（该结果能否用于主比较）。可以兼容旧is_primary_estimand，但定义要单一，报告与代码一致；缺组结果不能被下游当可用主结果。不要让总体单一布尔值替代各指标支持判定。

统一有限标量/组类型校验；不要在except中吞掉自己的拒绝。字符串不能自动拆成多个组；若允许无序输入必须确定性规范化，否则明确拒绝。y/yhat严格1D二分类原值验证，q小数拒绝。合法多受保护列的FairEvaluator公共入口必须有回归。

### 2.2 冻结完整拟合对象与版本化载入（R3-02，原R2-05）

冻结对象包括实际列顺序、类别/数值归属、有效schema/清理规则、所有拟合统计与provenance，不能只冻结specs。可用不可变内部结构、受控副本或使用前完整指纹验证；外部访问不能改动内部拟合状态。仅dataclass frozen或浅复制不够。

新fit工件保存版本、真实有效规则身份、列序、统计与来源绑定；export时核验当前拟合状态，load先验证文件中的身份与当前schema再接受。不能load后用当前specs重新算一个hash给来源不明的统计背书。旧格式没有身份时不得升级为可信主分析工件；可保留显式legacy只读兼容，但production adapter拒绝其主分析资格。不要读取/改写任何历史fit文件来迁移，本轮只造人工文件。

### 2.3 收窄配置接口（R3-07，原R2-06/R2-08）

保留旧合法keyword及已修好的typed键/值、空网格、单一函数定义、真实model/scaler参数。移除没有实现语义的新增形参和宽泛**kwargs，或对明确需要的参数实现并纳入有效配置；不要为通过“接受参数”的测试伪装支持。任何会影响执行的参数都应有可验证绑定，未知参数显式TypeError。

ID限制为已声明支持的稳定标量；保留前导零、来源/年份规范化和重叠拒绝，不把任意对象str化。错误日志不输出行级ID/标签数组。

### 2.4 工具闭环与实际加载身份（R3-06/R3-08，原R2-03/R2-09）

在登记fd分支识别日志对象及方向，普通open/io.open/fdopen等不能重新包装日志fd进行读写或关闭；logger只通过受控原始sink写入。保留dir_fd拒绝、路径保护、日志预算和部分写健康检查。只修本报告具体Python路线，不扩建完整OS沙箱，不声称阻止所有恶意原生调用。

预期拒绝声明绑定操作、明确目标、原因/异常和次数；退出必须消费完，不得计数差或目标子串代替精确匹配。真实网络/进程不发出，所有哨兵用本轮人工目标并有独立兜底。

总成功条件必须包含：pytest成功、哨兵成功、controlled failure probe成功、最终日志健康、无非预期/未消费拒绝、完整执行身份一致、资源监测合格。summary写入/日志关闭之后的新失败不能被忽略。监控错误应可靠记录为失败并结束失去监测的子运行，正确处理已退出竞态；15分钟/4GiB监测策略与采样RSS措辞保留。

给故障判定设计一个被实际控制器调用的集中状态函数或等效可测试控制流，用人工故障验证每一失败项都会改变最终退出。故障实例与主guard隔离，不把真实主运行失败catch后伪PASS。

真正实现只加载项目源码的策略并记录实际模块origin/源码哈希；若用FreshSourceFinder，必须在代码中创建并安装，不能只在environment里写名字。项目导入前禁止pyc回退，第三方依赖按已安装环境记录版本。执行hash覆盖实际导入闭包、测试/fixture、工具和两个已授权配置，缺文件或变化均失败；计数由机器生成，不能把87候选快照说成87执行源码。

## 第三步：补完真实搜索与隔离行为，不能用Python基本操作代替

这部分是原R2-07持续未完成项。不得把普通set、max(dict)、eps比较或仅hash变化标成实现验证。

- **AE缓存**：同一engine实际连续执行原状态/同配置、改C、改实际scaler、改epsilon等路线，按候选ID记录真实fit增量和缓存命中/失效。基线效用可能另需fit，区分baseline与candidate次数，不硬写1/1/2。不变配置重复不得误复用不相同数据，变化必须使相关候选重算；返回旧配置的策略明确。
- **Joint**：实际D8 run_arm的Joint候选提交与循环停止；可人工provider/阈值/效用控制A→B→A路径并披露注入，但必须观察真正committed_state/stop_reason。另保留真实LR预测集成。
- **BM游标与停止**：实际两次以上mitigate_step，捕获transform候选指数顺序，证明restart与monotone_cursor不同；实际失败最高dphi属性的stop行为。H=1、作者1998幂流与AE六网格规则保持，人工例早停或预算停止，不能截短作者流冒充完整模式。
- **epsilon/NMI边界**：注入可控几何/NMI数值进入真实接受/拒绝函数，覆盖阈值下、等、上；候选严格<epsilon与终局<=epsilon分开。NMI使用实际phi_threshold。实际小型MDS正例继续保留，不把控制流注入称完整论文数值复现。
- **F/C与S/T隔离**：固定人工F/C真实拟合和候选选择，将独立人工S/T的X/y/A/w逐项扰动；比较实际拟合对象、变换/选择/阈值与fit调用来源，保持不变。可在测试中编排当前核心F/C→冻结应用路径，明确这只是核心契约；主benchmark四分区在R1B另验。不能只比F数据hash，也不能把C改称S/T。

每项给调用路径、有效参数、注入点、样本/特征数、候选/fit计数、状态/停止轨迹。上轮不存在的D8/Joint测试必须真实补出；不要仅改矩阵里的名字来声明已覆盖。发现生产控制流问题就在授权文件内修复，遇到确实超范围的依赖精确标OPEN，不编造结果。

## 第四步：由真实工件自动生成交付并停止

1. 保持监督原R2-01…R2-10的含义；新R3-01…R3-08另列映射，不能换ID标题后声称全部关闭。F01仍是主benchmark未真实从F学习BM，F02仍是风险p/决策q混用，不是MEPS或调查方差的简称。
2. 从最终AST和真实pytest采集/执行结果生成矩阵；每个函数/类/行号/节点必须存在，且引用实际结果字段或trace路径。生成器在发现缺引用时失败。不能使用当前不存在的FairBiasEvaluator、FairBiasEnhancementEngine.run_search或GuardedTestRunner.run来填表。
3. 每项只标FIXED_CANDIDATE/PARTIAL/OPEN，记录执行证据与能力边界；测试计数从实际节点得出。不能以测试名称、注释或手填字符串充当fit计数/搜索结果。
4. 保存全部受控调试、修复前预期失败、正式运行的唯一目录/退出原因；前像为本轮紧邻候选，旧scratch索引只标历史引用。既存4次运行与前轮证据不变，旧恢复索引无需重做。
5. 报告从机器矩阵/结果生成并做一致性检查；定稿后独占创建一次MANIFEST，包含报告、source/test/config身份、trace、日志、资源及所有尝试引用，排除自身。报告末尾用协议第9节精确标题与停止行；如需修订已最终化交付，新增独占版本，不能覆写旧manifest。

最终状态可为 `FIXED_CANDIDATE — awaiting independent review`，但必须如实列未完成项；worker不能自批gate。只要仍缺实际终局、Joint/缓存/隔离证据，就不能写“全部问题完全解决”。最后一行：

`STOP — waiting for Codex review.`

## 后续应用研究路线保持，当前不执行

R1A验收后再做R1B真实FairBias-BM/AE/Joint与Unmitigated、RW、LFR、EG-DP、EG-EO、TO-EO适配，统一LR/GBDT及p/q/yhat能力；随后R2复杂调查全年设计/domain/共享配对复制，R3的76条件与5种子获准开发，R4冻结后的回顾性2024评价。保留20个主差值家族、B=2000等既定统计规则；2024已被观察，不能叫新的盲测。不要本轮训练真实数据或生成优势结论。允许FairBias不占优，预测与公平性权衡必须如实报告。
