# Gemini：执行R1A-R5，先验证负例有效，再完成剩余契约

你是implementation worker，Codex是独立监督者。项目目录 `/Users/lkc/Downloads/code_v_0_3`。研究目标是以FairBias为主方法完成医疗公平性应用论文，可靠比较预测表现与其他论文的公平性方法；FairBias不占优也是允许的研究结果。

**本提示直接授权以下三批有限工作及验证，不需要再先交计划等待许可。R1A-R4未验收；本轮R1A-R5仍是基础契约返修，不是新的正式实验。保留已经关闭的修复及所有历史工件，不回滚、不删除、不stage/commit/push，不进入R1B、不读取真实数据或训练正式benchmark。**

## 起点、边界与已关闭项

先读[执行协议](/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md)、[AGENTS](/Users/lkc/Downloads/code_v_0_3/AGENTS.md)、[GEMINI](/Users/lkc/Downloads/code_v_0_3/GEMINI.md)，然后读：

- [R1A-R4监督报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R4_SUPERVISOR_REVIEW_20260915.md)。
- 同名 `_evidence/` 中的verification、evidence_verification、final_verification、MANIFEST、source_snapshot、diff、diagnostic_execution_record，以及两个probe_run的结果和源码。10组诊断包含缺陷，不是10项验收通过；AST首版解析Markdown失败是监督脚本错误，不是生产缺陷。
- [R1A-R4提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R4_REPAIR_PROMPT_20260914.md)。未被本提示改变的算法和执行要求继续适用。

确认分支 `research/nhis-fairbias`、HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`、无暂存；核对监督106项当前输入身份，改动前保存本轮前像及Git状态。意外差异如实报告路径，不能重写预期hash掩盖。保护14个继承文件和.gitignore；既有.gitignore历史差异不处理，不读继承CSV算hash。

以下已经有独立证据，保留并做必要回归：legacy直接export拒绝、specs修改export拒绝、未知版本和主列序错配拒绝、合法refit清除legacy、adapter列清单移除后仍保持21列、frozenset拒绝、D8四路线正常终局、非零ps错误立即终止、89项真实前像与candidate分离、日志最终计数截点。不要为了扩大改动重新实现这些模块。

允许修改的相对路径（均以项目根目录为基准）：

- `src/nhis_fairbias/preprocessing.py`、`adapter.py`、`d8_enhancement_runner.py`。
- `src/fairbias/application_metrics.py`；只有真实行为测试发现具体缺陷时，允许改`enhancement.py`、`enhancement_contracts.py`、`mitigation.py`、`evaluator.py`。
- `scripts/_fairbias_r1_guard.py`、`scripts/run_fairbias_r1_guarded_tests.py`。
- 五个`tests/synthetic/test_r1a_*.py`和必要的`tests/synthetic/conftest.py`。

其余生产、benchmark包、旧tests、治理、配置原件、旧报告/计划/release/runs、scratch全部只读。新静态生成器和报告放在独占 `docs/reports/fairbias_r1a_r5_worker_<UTC>_<id>/`，执行输出放在独占 `runs/r1a_r5_<UTC>_<id>/`。每次尝试新目录，最终化后不覆盖。不要编辑上轮生成器来改上轮结论。

所有项目导入、调试和pytest都经过常驻guard，使用Framework Python `-I -S -B`。给现有入口加入r1a-r5 phase；单节点调试使用已实现的白名单入口，同样有哨兵和记账。stdlib静态AST/hash/文档生成可独立执行。禁止裸pytest、临时PYTHONPATH绕过、全tests/旧103测试、网络及安装依赖。

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -S -B scripts/run_fairbias_r1_guarded_tests.py --phase r1a-r5 --output-root /Users/lkc/Downloads/code_v_0_3/runs
```

规模不变：真实分类器/AE/BM每分区≤200人工行、≤8预测特征；preprocessing/adapter允许完整schema约24字段，每分区≤8行，不在宽表上训练分类器。保留真实LR正例、H=1、作者1998幂流、AE六网格与既定模式语义；人工几何用早停/预算限制，不截短作者流冒充完整模式。保持15分钟/4GiB采样监测，区分采样RSS与OS硬限制。

## 第一批：先证明负例有效，修好工件状态与结局标签

### A. 修测试敏感性（R5-03）

`test_r4_01_lifecycle_legacy_unverified_export_and_adapter_denial`会捕获自己的AssertionError。先修它：

1. 只用精确的`pytest.raises`检查单个允许拒绝的操作；成功路径的reload/资格断言放在异常捕获之外。不使用`except Exception`吞掉断言，不按异常消息包含legacy/export就判合格。
2. 在测试进程内临时注入监督快照中的已知坏export实现，或者等效且明确披露的故障替身，证明该测试确实失败；随后恢复当前实现并验证正常行为。不得改磁盘生产文件来制造变异，不读真实工件。
3. 把“原实现失败、修复后通过、关键故障变异仍被识别”的证据分别记录，不把预期失败直接改成PASS。故障测试不可清空主guard的异常记录。

### B. load/fit完整状态转换与内容校验（R5-01）

采用局部解析和验证完成后一次性提交状态的实现。不要先修改legacy标记、再解析正文；失败时要保持原状态完全不变，或明确整体失效。对已有对象连续操作做测试，而非每个反例都新建对象。

必须覆盖：

- 合法旧legacy对象→load规则正确但缺row_count的工件→异常之后仍未验证、adapter仍拒绝、export仍拒绝。统计量用0–100年龄规则、8行年龄90真实fit生成，不能手填私有record。
- 已验证对象→失败load；前后拟合统计、规则、列序、family、来源、资格必须一致或整体失效，不能混用新旧内容。
- 失败fit的相同原子性；成功重新fit建立完整新身份，保留已修好的legacy清除行为。
- v2必需字段缺失、版本与身份不一致、primary/expanded列序和类别/数值归属错配、缺失/多余统计键、NaN/Inf、越出有效schema范围的统计、错误fit_year/role等必须在边界拒绝。不能把缺字段当当前值重新补齐并认证。
- 合法同schema往返，包括expanded和family，结果及身份一致；旧格式保持显式legacy，不自动提升。

规则是完整对象有效，不是“存在rule_identity键即可信”。若某真实fit在全无效数值列上只能产生未定义或不合法中位数，应明确失败并保留原状态；不要放宽load检查或填任意常数来迁就测试，也不在本轮更改正式benchmark的特征删除/缺失估计政策。所有反例只用新人工工件，不迁移历史结果。

### C. 恢复明确的结局契约（R5-02）

保留adapter的冻结列访问，但移除本轮新增的任意字段名fallback。该层消费既有harmonized二分类结局，只接受明确支持的结局及已定义别名。修正R4-02人工fixture：使用现有meddl12m、0/1编码，禁止通过放宽生产代码来接受错误fixture。

验收应实际调用get_cohort：正确输入返回{0,1}；未知outcome如agep_a拒绝；仅存在原始1/2/7/8/9字段时明确拒绝；缺harmonized字段、非二元标签、别名歧义拒绝。不要把7/8/9当阴性或直接astype(int)后流入模型。若未来需要原始编码适配，应在已定义的源清理层另行实现，本轮不新增估计目标或数据来源。

## 第二批：补完仍未触发的实际生产行为（R5-04）

新测试应产生独占结构化trace工件，记录生产调用、有效参数、注入点、样本/特征数、候选与fit计数、状态和停止原因；不输出真实个体数据。既有测试正常调用不删除，不将增加测试数当验收标准。

1. **Joint循环。** 人工provider实际调用D8EnhancementRunner.run_arm，控制几何/候选效用使生产控制流接受A→B，再提出A。保留真实候选提交、committed集合和stop判断，断言实际cycle_detected、最后保留的状态、接受步数和预算。不能mock整个run_arm或stop函数，也不能在函数调用后再用普通set模拟循环。已通过的真实LR四路线作为独立正例保留。0个接受步骤、candidate_exhausted不是循环验证。
2. **BM游标。** 同一official_stream下只改变revisit策略；每个engine至少两次进入实际搜索/公开编排，spy记录transform实际收到的指数顺序和搜索起点。构造能区分restart和cursor继续行为的例子，不能让第二次空遍历只满足cursor不减。保留真实小MDS正例和最高属性失败stop。
3. **终局/NMI边界。** 使用现有生产接受/终局逻辑分别覆盖严格`<epsilon`和终局`<=epsilon`下/等/上。按真实phi_threshold公式构造边界，记录NMI before/after及结果；Python数值比较和高信息/零信息两点不能代替等值边界。
4. **AE缓存余项。** 在同一engine实际改变epsilon，并实际返回旧配置，记录baseline/candidate fit计数和选中状态。既有F/C、C参数、实际scaler测试保留。按照实现明确返回旧配置时重算还是有效复用，不硬写固定拟合次数，不只看hash变化。
5. **核心隔离。** 固定F及核心selection，建立独立S和T人工对象，逐项改变其X/y/A/w；进入真实冻结变换、模型和评价调用，再比较之前已冻结的拟合/选择状态及fit来源记录。标签、组、权重不能只是定义后未使用；阈值应取实际生产配置/对象，而非测试中定义0.5后断言自身。若核心生产策略固定0.5，明确验证这个事实。

这里验证核心阶段不得反向使用S/T；不改变正式benchmark的C校准、S配置选择、T最终评价职责，也不要求正式S配置选择对S变化不变。正式四分区比较继续在R1B验收。确实需要白名单外代码时精确列OPEN，先完成无依赖部分，不降低断言或编造trace。

## 第三批：证据不完整必须失败，交付与实际结果一致

### A. 完整性与工具（R5-05/R5-06）

- 给实际控制器使用的证据验证函数覆盖absent、empty、malformed、错误类型、缺模块、多模块、hash不符等反例；必须使最终FAIL/非零退出。不能只遍历已提供条目，缺失就得到空字典然后PASS。合法完整工件是独立正例。
- 对预执行源清单、实际模块来源、编译字节、执行后身份建立完整集合关系。特别处理5个pytest测试文件和runner入口：当前37项pre/post、34项执行模块、28项compiled bytes是不同计数，不能把28称完整34。选择适合pytest实际加载路径的记录方法，并注明验证覆盖；不默默依赖未执行的自定义loader分支。依赖环境不升级。
- 日志存在、可解析、最终计数和健康状态也要纳入强制检查；预检、早期解析/哨兵失败等每次入口尝试均有记录。保存真实前像与candidate分离的已修实现，不再回归到扫描工作树当作前像。
- 预期拒绝默认精确目标。确需目录范围或basename模式时显式声明类型，不能隐式把文件A匹配A/child或任意父目录下同名文件。按真实操作、目标、原因、次数验证。只造人工对象，不执行真实网络或越界行为。
- ps返回0但没有有效RSS时，区分进程已退出与仍运行；后者按明确的监测失败策略处理。timeout/kill竞态可用人工process对象测试，不制造真实内存占用或长等待。
- 主guard的unexpected_denials不能在测试中pop/clear。负例注入到独立matcher/状态及人工sink，或者使用有审计标识的隔离故障上下文；主运行健康和故障探针结果分开。不要建立新的通用OS安全框架，只修已列Python运行路径。

### B. 组序列化（R5-07）

有限Decimal和任意带is_finite方法的对象不能绕过类型规范。可以明确拒绝当前研究不需要的Decimal；若保留支持，完整返回对象需JSON可序列化且组身份稳定，不因浮点化/字符串化造成组碰撞。覆盖完整json.dumps及组定义往返，不只提取gap数值测试。保留frozenset/Infinity及合法常见标量正例。

### C. 生成交付（R5-08）

原R2-01…10、R3-01…08、R4-01…08的含义保持不变。本轮R5-01…08使用监督报告中的含义，不能通过换ID标题消失未完成项。F01主benchmark实际从F学习BM、F02风险p与决策概率q的区分继续OPEN并延后R1B；q不是阈值。

从最终AST、实际pytest节点结果、生产行为trace和所有运行工件生成：

1. 按原问题逐项记录FIXED_CANDIDATE/PARTIAL/OPEN、真实源码位置、触发条件、调用路径、执行节点、trace路径及字段、观察结果、注入边界。查不到符号/字段/节点须失败，不能只Warning。
2. 有节点不等于有行为证据：Joint必须引用实际循环trace；公共指标必须引用实际FairEvaluator.compute_metrics；缓存和隔离必须引用实际状态/fit记录。不能手填不存在的terminal_predictions、committed_state等字段充数。
3. 完整尝试索引：本轮每个命令、节点调试、修复前失败、敏感性故障、正常运行、生成器尝试均独占记录。补充上轮7条入口调用与6个保留目录、报告只列3次的对应关系；找不到证据的调用明确UNKNOWN，不补造当时记录，不覆盖上轮报告。
4. 计数与hash由机器生成，包括本轮diff与累计diff、schema字段与分类器特征、pre/post/loaded/compiled模块数。最终用户答复使用同一材料，禁止手工编哈希或写“仅F01/F02遗留”而忽略未完成R1A项。
5. 定稿后独占生成MANIFEST一次，排除自身，覆盖报告、生成器、代码/测试/配置身份、trace、日志、资源及尝试索引。发现需修订则新版本，不覆盖已最终化工件。

最后核对106项初始输入的预期diff、冻结文件、无暂存、历史证据及链接。报告必须使用协议第9节精确标题（含冒号）：Gate、Status、Files changed、Commands executed、Permissions requested、Tests executed、Exact test results、Input hashes、Output hashes、Row counts、Assumptions、Unresolved issues、Git diff summary、Proposed next step。worker不能自批gate。最后一行：

`STOP — waiting for Codex review.`

## 后续应用论文路线保持，当前不执行

R1A通过后再执行R1B：FairBias-BM/AE/Joint真实学习与Unmitigated、RW、LFR、EG-DP、EG-EO、TO-EO适配，统一LR/GBDT、组类型、调查权重能力、F/C/S/T及p/q/yhat；不支持的组合显式保留，不静默降级。R2验证全年复杂调查domain、共享配对复制、20个主差值家族与B=2000等既定统计规则。正式R3按冻结76条件×5种子获准开发，正式R4冻结后回顾性评价已看过的2024；不能称新的盲测。结果同时报告预测、公平性、亚组支持和不确定性，不挑选仅对FairBias有利的条件。
