# Gemini：执行 R1A-R4，完成剩余生命周期、集成与交付契约

你是 implementation worker，Codex 是独立监督者。项目根目录是 `/Users/lkc/Downloads/code_v_0_3`。用户要以FairBias为主方法发表医疗公平性应用论文，同时比较预测能力和公平性。目标是可信比较，不能把FairBias获胜设为验收条件。

**本提示直接授权以下四步实际实施、受控验证和交付，不需要先只写计划再请求许可。R1A-R3未验收；本轮是R1A-R4基础契约返修，不是正式实验R4。禁止进入R1B、真实数据实验、stage/commit/push。保留全部候选及失败记录，不回滚、不删除、不覆盖既存工件。**

## 0. 权限与起点

先阅读：

- [执行协议](/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md)、[AGENTS](/Users/lkc/Downloads/code_v_0_3/AGENTS.md)、[GEMINI](/Users/lkc/Downloads/code_v_0_3/GEMINI.md)。
- [本轮监督报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R3_SUPERVISOR_REVIEW_20260914.md)及其同名 `_evidence/` 内 `verification.json`、`evidence_verification.json`、`final_verification.json`、`MANIFEST.json`、`source_symbols.json`、源码快照和逐次诊断记录。
- [上一轮提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R3_REPAIR_PROMPT_20260914.md)。其中未被本提示改变的算法、预算和边界要求继续适用。

核验分支 `research/nhis-fairbias`、HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`、无暂存，核验本轮监督保存的89项输入身份。保留14个继承文件和.gitignore，既有b595e59的.gitignore历史差异不修。只读Git元数据，不读原始CSV算hash。若输入发生意外变化，记录路径和来源，不能自行替换预期值。

允许修改的相对路径（均在上述根目录内）：

- `src/nhis_fairbias/preprocessing.py`、`adapter.py`、`d8_enhancement_runner.py`。
- `src/fairbias/application_metrics.py`、`enhancement.py`、`enhancement_contracts.py`、`mitigation.py`、`evaluator.py`。只有具体反例需要时才改后三类核心模块；已经通过的几何和缓存语义保留。
- `scripts/_fairbias_r1_guard.py`、`scripts/run_fairbias_r1_guarded_tests.py`。
- 五个 `tests/synthetic/test_r1a_*.py`，必要时增加 `tests/synthetic/conftest.py`。

其他生产文件只读。禁止修改benchmark包、配置原件、旧tests/历史脚本、治理、旧计划/报告、release、旧runs和scratch。`scratch/build_r1a_r3_delivery.py`仅保留作历史证据，不继续编辑或执行。新增静态生成器/计划/报告可放在本轮独占 `docs/reports/fairbias_r1a_r4_worker_<UTC>_<id>/`；运行输出在 `runs/r1a_r4_<UTC>_<id>/`。若一个逻辑步骤有多次尝试，每次新目录，并在最终包索引全部尝试。

所有项目导入、调试和pytest通过常驻guard，父子均用已安装Framework Python `-I -S -B`；给现有入口加 `r1a-r4` phase。单文件/单节点调试只从五个测试文件中明确枚举，仍有哨兵和完整记账。禁止裸pytest、临时PYTHONPATH绕过guard、全tests/旧103测试、网络、安装依赖及真实NHIS/MEPS输入。stdlib AST/hash、只读Git核验和不导入项目的文档生成可以独立执行并记录。

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -S -B scripts/run_fairbias_r1_guarded_tests.py --phase r1a-r4 --output-root /Users/lkc/Downloads/code_v_0_3/runs
```

真实分类器、AE、BM每分区≤200人工行、≤8预测特征；预处理/adapter例允许完整schema约24字段、每分区≤8行，宽表上不训练分类器。至少保留真实LR正例。只注入人工数据运输、历史全年人数约束以及明确披露的几何/效用控制点。不得mock掉被验证的fit、工件身份检查、候选提交或循环停止。保留15分钟/4GiB采样RSS策略，不把采样监控称OS硬限制。

## 第一步：一次修完拟合对象的完整生命周期和调用接口

先建立下面反例测试，保存受guard的修复前预期失败记录，再修代码并运行同一测试。反例用真实fit产生统计量，不能伪造私有 `_fitted_record` 当正例。只造人工JSON，不迁移或读取历史fit文件。

| 需要固定的行为 | 测试与观察 |
|---|---|
| 错配规则直接load | 0–100年龄规则、8行年龄90真实fit/export；官方规则对象必须拒绝。保留已有正例 |
| 旧格式生命周期 | 人工legacy工件load后明确未验证；export→load仍未验证或明确拒绝export，adapter始终拒绝主分析资格 |
| 拟合后公开specs修改 | transform与export必须一致拒绝，或始终使用真实冻结身份；不能把旧统计和当前specs绑定成新可信工件 |
| 版本与列序 | 未知版本明确拒绝；v2记录的列序、feature family、规则、统计和来源须一致，错配不能忽略后重建为当前值 |
| 合法往返 | 同schema真实fit/export/load后的输出、列序、类别/数值归属和身份一致 |
| 合法重新fit | legacy对象重新在合法2022人工F数据上fit成功，建立新的完整身份并清除旧状态；失败的fit/load不得把对象留下半可信状态 |
| 适配器最终矩阵 | 真实构造adapter并调用get_cohort；对公开列清单remove/reverse、类别列表及返回统计副本逐项变更，最终X/特征名/family/身份须保持一致或明确拒绝，不能21列静默变20列 |

采用单一的拟合状态校验/访问接口，让transform、export、load、family getter、adapter.get_feature_names/get_cohort用同一完整冻结身份。无需建立新的安全框架或承诺抵御Python私有对象任意篡改；需要覆盖真实公开生命周期。身份应来自成功拟合时的有效规则和统计，不能仅计算当前config hash给来源未知的统计背书。验证数值统计有限且与对应feature/schema兼容，验证fit_year/role等现有provenance；不能靠NaN/缺项fallback蒙混。

把合法组类型定义成有限的稳定标量集合，拒绝frozenset等容器和任意object；不要float转换失败后默认合法。保留字符串、整数及已声明支持类型的正例，明确bool与数值混用规则；非有限值、重复/空组、额外观察组继续拒绝。无序组集合要拒绝或确定性规范化，输出须可安全序列化。不要为此改变主研究组定义。

## 第二步：补齐真正经过生产路径的集成测试

此前68个节点和部分真实LR测试保留；验收数量由执行结果生成，不设人为目标数字。以下必须给真实调用路径、参数、注入点、规模、fit/候选计数和trace。找不到既有API先读源码，禁止凭印象编造类或函数名。

1. **D8四路线和Joint。** 用人工provider实际执行 `D8EnhancementRunner.run_arm`。逐条确认Unmitigated/BM/AE/Joint最终输出传递冻结arm的expected_groups/source，覆盖正常组、缺预期组、无正/负支持及observed-only诊断入口。保留真实LR终局；控制流测试可以注入小型几何/效用候选，但保留真实run_arm提交和停止。构造A→B→A，断言实际committed_state、stop_reason、最终保留状态及循环预算；普通set查询不算此项证据。
2. **BM游标与边界。** 保持同一 `power_sequence_policy=official_stream`，只改变 `power_revisit_policy`，对每个engine实际至少两次mitigate_step或其公开编排路径，记录transform收到的指数顺序。构造可解释的早停/预算例区分restart与monotone_cursor，不同时改sequence来“制造差异”。H=1、作者1998幂流和AE六网格按既定模式保持，不截短作者流后声称完整模式。保留最高dphi属性失败stop的已通过真实测试。
3. **epsilon与NMI边界。** 候选严格 `<epsilon` 与实际终局 `<=epsilon` 分别覆盖下/等/上；通过实际NMI gate测试phi_threshold边界，不能直接把gate mock为true。注入几何值的控制流测试与真实小MDS数值测试明确区分。
4. **AE缓存余项。** 保留监督证实的同engine重复、C变化、实际scaler变化；补epsilon、F/C实际数据改变与返回旧配置的策略。按candidate ID区分基线与候选fit计数，不强迫所有例都符合13/0/13/13。验证缓存上下文确实改变行为，不只比较hash。
5. **核心fit/selection与S/T隔离。** 建立F、核心selection分区及两个独立S/T人工对象，真正执行当前允许的FairBias核心学习/候选选择→冻结应用路径，逐项扰动S/T的X/y/A/w。检查拟合来源、实际变换/选择状态、模型参数和已有阈值保持不变；不能只重复fit两个相同LR，也不能把selection分区改名S/T。若现核心API阈值固定0.5，则按固定策略验证并披露，不能编造“学得阈值”。

第5项是R1A已授权核心契约测试，不重定义正式benchmark的C校准、S配置选择、T评价职责。正式S阶段本来会选择配置，不要求其选择对S数据变化不变；这里保证S/T不反向进入此前应冻结的核心拟合阶段。正式四分区适配仍在R1B单独验收。

若必须修生产控制流，在上述白名单内修复；确实需要改白名单外依赖则列明具体接口和缺项为OPEN，先完成其余独立工作。不以宽泛异常或降低断言替代正确行为。

## 第三步：完成现有工具的有限闭环

不重做已经修好的日志fd保护、未消费拒绝和FreshSourceFinder；完成以下具体缺口：

1. **精确拒绝契约。** 声明目标A时，A_DIFFERENT不能消费同一条期待；操作、规范化目标、原因/异常和次数绑定。正确拒绝、未发生、次数错、目标错、原因错各有反例。确实需要模式时使用明确类型且有限匹配语义，不能默认子串。只用人工对象，独立兜底阻断真实网络/进程/越界操作。
2. **监测状态。** 把监测错误处理放入有效循环路径，失去监测时终止仍在运行的子任务；已退出竞态单独处理。集中verdict继续消费所有失败项，并以实际控制器使用的状态函数/监测决策函数做故障测试，不单写一个与控制器无关的测试函数。无需真的制造大内存或15分钟等待。
3. **实际源码身份。** 预执行清单覆盖允许的潜在项目导入闭包、测试/fixture、工具及两个配置；实际loader记录其读取并编译的源码字节hash及origin，执行后与预期和当前身份比较。缺文件、采集错误、额外未绑定项目来源均使整体失败，不得except后静默少记模块。不安装依赖、不扫描微数据。控制器自身/guard的启动身份也纳入核验。项目pyc回退禁止，第三方只记录已安装环境。
4. **前像准确命名。** 首次改动前捕获实际工作树；若重建前像，必须读监督source_snapshot保存的字节并验证其hash，标重建时间与来源。分别输出preimage、post-change candidate、pre-execution、executed bytes、post-execution身份；不要把修改后87/89项扫描称修改前像。意外差异报错，预期变更用候选diff记录。
5. **最终日志截点。** 正式汇总在日志关闭后由父控制器读取/计数，或明确保存截点字段及完整最终计数。summary写入、日志关闭故障必须反映到最终exit/verdict。预检失败等早期失败也要有独占尝试记录；不能为了生成PASS覆盖失败记录。

只验证已知Python入口和研究运行治理，不扩展成抵御恶意原生代码的完整OS沙箱。保持整体任务可完成。

## 第四步：自动生成一致交付，保留全部尝试

建立下面不可换义的问题主键，然后附加R3及本轮R4问题映射：

| ID | 固定含义 |
|---|---|
| R2-01 | 公共指标NameError与公共入口回归 |
| R2-02 | dir_fd越界 |
| R2-03 | 日志闭环 |
| R2-04 | D8最终输出 |
| R2-05 | Registry与完整拟合冻结 |
| R2-06 | 配置指纹 |
| R2-07 | 真实搜索与隔离行为测试 |
| R2-08 | 输入校验与错误隐私 |
| R2-09 | 执行工具 |
| R2-10 | 报告可信度 |

R3-01…08保留前轮监督报告中的含义；R4-01…08对应本轮监督报告。F01仍指主benchmark没有实际从F学习BM；F02仍指风险p与决策q混用。不要把这些ID改成权重、MEPS或其他已测项目。

静态生成器在本轮新报告目录中创建；从最终AST、pytest真实节点结果及结构化行为trace生成矩阵。每项至少有：原ID、标题、状态、实际文件及限定符/行号、真实触发、已执行节点、trace字段、结果、注入与能力边界。逐项验证：

- 文件/符号/行号和测试节点真实存在；嵌套函数使用实际作用域，属性不冒充函数；引用跨文件要明确真实文件。
- 被引用节点真的调用被验证路径，且trace中有所声称的状态/fit计数/结果字段；节点名或注释不构成执行证据。
- 每项仅 `FIXED_CANDIDATE`、`PARTIAL`、`OPEN`，无法验证的项保持开放。文档生成器对缺引用、错误ID、错误计数、与trace矛盾的结果必须失败。
- 前后/source/测试/config/hash、报告和最终用户答复由同一机器材料生成。不得手工拼64位哈希；已知R1A-R3六个答复hash错误列入历史更正，不覆盖原答复或工件。
- 完整列出修复前预期失败、节点调试、正式运行、静态生成、失败尝试和所有退出状态。上轮三条guard外pytest与scratch写入作为历史权限偏离说明，不能补造当时guard日志，也不能宣称已证明那些尝试零访问。
- 最终MANIFEST独占生成一次、排除自身，覆盖报告、代码/测试/配置身份、trace、日志、资源和尝试索引。需要改最终化内容则生成新版本；旧worker三次运行和旧监督证据保持原样。

最后核验初始输入、冻结文件、未暂存状态、历史证据身份与授权diff。任何“全部解决”结论必须能逐项由证据支持，否则准确列剩余问题，不为了通过改原标准。

最终报告必须使用执行协议第9节精确标题：Gate、Status、Files changed、Commands executed、Permissions requested、Tests executed、Exact test results、Input hashes、Output hashes、Row counts、Assumptions、Unresolved issues、Git diff summary、Proposed next step（各带冒号）。状态只能是候选等待独立复核。最后一行：

`STOP — waiting for Codex review.`

## 本轮之后的论文路线：保留规划，当前不执行

| 阶段 | 后续交付与放行条件 |
|---|---|
| R1A验收 | 完成本提示具体生命周期、生产路径和证据契约 |
| R1B适配 | FairBias-BM/AE/Joint真实从F学习；对照Unmitigated/RW/LFR/EG-DP/EG-EO/TO-EO；统一LR/GBDT、特征语义、F/C/S/T角色及p/q/yhat能力。逐方法核对组类型、权重和预测接口，不支持则显式标注，不能静默降级 |
| R2统计 | 全年复杂调查设计与domain估计；共享配对复制；权重点估计和调查不确定性分开；20个主差值家族及B=2000按既定计划 |
| 正式R3开发 | 获准后按冻结的76条件×5种子、预算、阈值和选择规则执行；超时/失败/不支持保留，不能只挑FairBias有利结果 |
| 正式R4评价与文章 | 冻结后回顾性2024评价，2024已看过不能称新盲测；报告预测与公平性权衡、各亚组支持、不确定性、适用范围及负面结果 |

不本轮改研究估计目标、添加方法或训练真实模型。FairBias主方法地位保持，优劣由后续合格比较决定。
