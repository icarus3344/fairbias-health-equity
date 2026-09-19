# FairBias 审查修复的独立监督者验收

日期：2026-09-11。审查对象：工作者所称“GPT-6 Independent Audit Verification and Systematic Remediation”的未提交改动。

**决定：REPAIR — 当前候选不通过验收，不批准 staging、commit 或真实数据实验。** 本轮仅独立检查和新增报告、证据，没有修改工作者实现，没有撤销有价值的修复，也未改变历史 Gate 的结论。

## 1. 范围、版本与测试结论

实际分支为 `research/nhis-fairbias`，HEAD 为 `67e6659fa65249a8842e34af5d8969629efe4bca`。候选包含 11 个已跟踪文件修改（+200/-58）和一个新增测试文件。当前工作区还包含上轮审查报告/证据与 `scratch/`；后者未纳入本次实现审查或修改。

先后读取了执行协议、用户贴出的工作者报告、其 implementation_plan/walkthrough，逐项对照实际 diff。使用与 R4 一致的 Python 3.13.2 / scikit-learn 1.7.1 环境，不安装依赖，不联网，不重跑真实模型。测试和探针在独立持续数据访问 guard 下运行；真实数据尝试均在打开文件前被拦截。

| 独立核查 | 结果 | 可支持的结论 |
|---|---|---|
| 工作者六文件核心套件 | **136 passed，64.52 s** | 在本次候选上可复现核心测试通过；不代表所有审查问题修复。 |
| 19 项独立合成/配置/路径核查 | **5 PASS / 14 FAIL** | 14 个未满足项包含同一缺陷的不同观察，不等于 14 个独立新增 Bug。 |
| 原 D6 `test_03_repeated_cross_sectional_manifest_semantics` | **失败，在真实 parquet 打开前拦截** | 该合成测试仍进入生产前置检查；原问题没有闭环。 |
| 被列入“合成验证”的默认 COMPAS pipeline 路径 | **尝试打开 data_COMPAS.csv，被拦截** | 该测试路径不是纯合成。没有读取 CSV 内容或拟合其模型。 |
| 14 个冻结继承文件和 `.gitignore` 对 HEAD | **无本轮修改** | 不能据此反推 `.gitignore` 自保护 tag 起从未变化；该历史差异上轮已记录。 |
| 三个 R4 primary JSON | **散列与上轮一致** | 确认这三个主工件保持不变，不等于验证修改后代码会产生相同数值。 |
| 本轮开始/结束的 12 文件候选散列 | **无漂移** | 本报告绑定到同一份候选快照。 |

核心套件的数据拦截日志有一次 NHIS parquet 尝试，是 [D8 契约:1131](/Users/lkc/Downloads/code_v_0_3/tests/test_nhis_d8_synthetic_contracts.py:1131)故意验证阻断的 sentinel；它不是成功读取。独立 D6 进程另有一次生产前置检查尝试。首次 D6 检查结束时，guard 还对 pytest 旧临时目录清理产生误拦截；随后使用独立 `--basetemp` 重复该单项，仍在同一生产读取路径失败，避免将清理副作用混入主结论。

证据位于 [本轮证据说明](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_REMEDIATION_SUPERVISOR_REVIEW_20260911_evidence/README.md)。保留脚本、机器输出、日志、工作者材料快照及文件散列；没有以文件时间戳证明“从未读取”。

## 2. 阻止验收的问题

### R1 / P1：缓存修复漏掉方法调用，仍会跳过必须重新评估的候选

**位置：** [enhancement.py:591](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:591)、[831](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:831)。

新代码是 `f"sel={partition.selection_fingerprint};eps={epsilon_threshold}"`，但 `selection_fingerprint` 是方法，需要调用 `selection_fingerprint()`。目前缓存键实际包含 bound-method 的字符串表示，而该表示又包含 EvaluationPartition 的 pandas 截断显示内容。

**已独立复现：** 200 行纯合成验证集，只改变显示省略区内一个标签。真正的内容指纹不同，但新缓存键相同（长度 939，含 `bound method`）。真实 `enhance_step` 在 mock 效用下，第一次候选 0.5 低于基准 0.6；第二次候选应为 0.9，却根本没有调用候选评价，只重新计算了基准，返回未选择候选。

现有新增测试 [test_audit_remediation_probes.py:105](/Users/lkc/Downloads/code_v_0_3/tests/test_audit_remediation_probes.py:105)只把两个手写字符串传入 tracker，未测试 engine 如何生成上下文，因此无法发现这个错误。

**返工要求：** 修正调用只是第一步；键还需绑定 fit/selection/protected、完整解析配置、有效参考 ε（包括 `current_max_epsilon` 回退）、权重和无损候选参数。配置变化应重置或区分排名缓存、访问缓存及 cycle history。必须通过真实 engine 路径验证隐藏标签、训练数据和配置变化，不能只测 tracker 的字符串区分。

### R2 / P1：不可估计公平性仍被制造为零，新标志甚至将其认定为严格可行

**位置：** [bias_metric.py:224](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:224)、238、273、303–307 行；[d8_enhancement_runner.py:191](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:191)。

只修复了 `compute_dphi_matrix` 的单群体分支。某群体某数值/类别特征全缺失、未列入 num/cat 的预测变量，仍返回零。补充 `isfinite(max(...))` 无法识别已经被下层补成零的缺失信息。

**已复现：** 数值或类别整组缺失都返回 dφ=0；未声明变量返回 dφ=0；AE 在 ε=0.0005、slack=0 下接受整组缺失候选，新增 `strictly_feasible=True` 和 `relaxed_feasible=True`。单群体 DP 仍返回 0，虽然 EOpp 已改为 null。

**返工要求：** 从散度入口保留不可估计状态并拒绝不完整的特征/群体覆盖；补充数值、类别、加权零有效分母及公共入口的真实反例。单群体 DP/EOpp 使用 null + 原因。不要只增加更上层的 NaN 判断。

### R3 / P1：修改后的执行 manifest 依然与实际算法不一致

**位置：** [run_nhis_d8_r4_substantive.py:170](/Users/lkc/Downloads/code_v_0_3/scripts/run_nhis_d8_r4_substantive.py:170)、[enhancement.py:32](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:32)、47 行。

新的声明网格为 `[0.5, 2.0, 3.0]`，实际 AE 默认仍为 `[1/7, 1/5, 1/3, 3, 5, 7]`；声明 minimum gain=0.001，实际为 0。D8 构造 AE 时没有覆盖这些默认值。两项不一致均由对象值比较确认。

计划承诺的 `runner.get_active_parameters()` 动态内省未实现，只替换了一组硬编码。排名也并非全特征候选的统一全局效用排序，而是先选择特征、再在该特征内选候选。新增严格/松弛布尔字段尚未完整进入持久化 candidate audit schema。

**返工要求：** 构造单一有效运行配置，由实际对象生成 manifest，并与 runner 参数传递做一致性测试。补齐候选审计字段，保持历史文件原样；不能再用另一份人工字典宣称“100% 反映执行”。

### R4 / P1：新增调查推断资格标志会放行无穷权重

**位置：** [survey.py:60](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/survey.py:60)。

`survey_inference_eligible` 只检查可解析、非缺失、正权重与设计列非缺失，没有检查有限性。四行样本中一行权重为 `+inf`，其余为 1，且分层/PSU 非缺失时，得到 `status=PASS`、`survey_inference_eligible=True`、`wtfa_a_sum=None`。

**返工要求：** 验证有限权重和有限正总量、非空有效样本，明确字段究竟表示“基础字段可用”还是“可以进行设计推断”。仅有设计字段存在也不足以证明一般设计方差可估计；不应在没有设计诊断时使用过强名称。新增合法设计列 + 非有限权重的组合测试。

### R5 / P1：名义编码与分区独立性未修复，来源缺失只改了标签

**位置：** [enhancement_contracts.py:84](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:84)、[420](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:420)、[preprocessing.py:246](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:246)。

计划中的 one-hot 和标签置换测试没有落实；本次同一名义成员关系的整数编号重命名仍使实际 LR 验证 AUROC 从 0.5 变为 1.0。EvaluationPartition 的分区验证没有修改，完全相同记录作为 fit 和 selection 仍通过。

预处理缺少年份/角色时改记 `unspecified`，比冒充 2022 更诚实，但仍允许拟合；这没有实现计划承诺的生产 fail-closed。D6 的下游年份检查可拒绝这种标记，但通用接口及 D8 注入路径不能由此自动获得同样保证。

**返工要求：** 生产入口要求可验证来源；合成用途另设显式入口，验证预先拟合的注入对象。分区独立性基于带年份/来源命名空间的记录 ID，不能仅比较不同年度重复的 RangeIndex。

名义编码修改会改变研究模型，应在新版本预声明并单独验证；如果本 Gate 只修边界与报告，允许明确延期这项工作，不能写成已修复。调查推断也可以保留为独立后续方法学任务。

### R6 / P1：测试/报告证据没有绑定当前候选，纯合成声明也不成立

工作者报告给出的完整 HEAD 是 `67e6659c256086f6d506941eb0579e27c08c9038`，实际为 `67e6659fa65249a8842e34af5d8969629efe4bca`；不能只比较七位前缀。

| 文件 | 工作者声称 SHA-256 | 本轮实际 SHA-256 |
|---|---|---|
| `tests/test_audit_remediation_probes.py` | `0c0a5dc4a42b1ca4c3b62f43d3b76ea4db100cbcf40232ee963a2a466bf784ae` | `aa14bb630e88511e3ef354221a7a5a534fe2a4b857321b593b65b536a4f1ad9c` |
| `docs/reports/NHIS_D8_R4B_SUPERVISOR_ERRATUM.md` | `8e9e4f5068213601ce8f4955fd081076f7f2b1d3d052be1bb3cf12d4a65a3962` | `5295991d3f3017a25e29b44f75f1dd52fc4afedcfad55b5987d1bf6300189a22` |

这些差异证明报告未准确绑定当前候选；不据此推断是何时变化或主观伪造。walkthrough 列举的测试名称也与实际八项不符，声称存在的 feasibility 专门测试实际不在该文件中。累计 232 次运行可以包含重跑，不能当作 232 个唯一测试或完整修复证明。

[test_fairbias_pipeline.py:27](/Users/lkc/Downloads/code_v_0_3/tests/test_fairbias_pipeline.py:27)直接使用 `compas_default()`，后续读取真实继承 `data_COMPAS.csv`。本轮实际调用同一路径并在打开前拦截。这不证明工作者读取过 NHIS 行，但足以否定“列出的全部验证都是纯合成”。文件时间戳没变也不能证明零读取。

[test_fairbias_clean_checkout.py:104](/Users/lkc/Downloads/code_v_0_3/tests/test_fairbias_clean_checkout.py:104)导出并测试的是已提交 HEAD，不能证明这份尚未提交补丁的 clean-checkout 正确性。

**返工要求：** 自动生成真实 HEAD、补丁/文件散列、运行命令、环境与输出日志；区分当前补丁、旧 HEAD 和重复运行。所有新增验收在无真实文件的临时合成环境或持久 guard 下执行，记录预期 sentinel 与非预期访问尝试，提交前重新核对文件散列。

### R7 / P1：新的补充说明又将待验证的路径解释写成因果结论

**位置：** [Erratum:118](/Users/lkc/Downloads/code_v_0_3/docs/reports/NHIS_D8_R4B_SUPERVISOR_ERRATUM.md:118)。

新文字称 C3 终态相同“is a consequence of step budget exhaustion ...”。相同起点、相同预算和较宽约束是可能解释，但预算耗尽本身不能保证不同几何下候选选择一致。本次没有新增首个候选分歧分析或受控机制实验，不能从条件同时出现推断因果。

另在 [110 行](/Users/lkc/Downloads/code_v_0_3/docs/reports/NHIS_D8_R4B_SUPERVISOR_ERRATUM.md:110)声称 primary condition metrics 记录 `fairness_feasible`；实际 `condition_metrics.json` 的扁平指标记录没有该字段，需引用正确的审计汇总或 runner 结果位置。

**返工要求：** 明确新说明替代原第 3.3 节相应强结论；写成“在共同起点、给定松弛和五步预算的两组执行中观察到相同终态，机制未被识别”。不要以新的因果断言修复旧的因果断言。

## 3. 测试隔离与遗漏项（P2）

- [D8 测试:35](/Users/lkc/Downloads/code_v_0_3/tests/test_nhis_d8_synthetic_contracts.py:35)仍在导入/收集阶段安装全局 hook；teardown 只是把开关设 False。导入到 teardown 之间仍影响其他测试，结束后也不能再据此声称后续套件有数据保护。应采用独立进程或精确生命周期隔离，而不是以解除防护换取通过。
- 原 D6 test_03 仍访问生产 parquet 做 checksum，独立 guard 将其拦截。当前 136 项套件没有包含两个 D6 文件，故不能验证上轮最关键的合并测试失败已经解决。
- 原有 `test_protected_category_audit_keeps_categories_and_missing_bucket` 被删除并替换成新测试；应恢复其分类及缺失桶语义覆盖，而不是用新增边界测试抵消旧覆盖。
- `multigroup_aggregation` 已正确加入低层几何函数，但尚未接入 FairBiasConfig、FairEvaluator、runner 及配置指纹。作为低层显式选项可保留；不能称研究入口已完整支持作者模式。
- stress-elbow 的选维、收敛记录与稳定性问题没有修改；walkthrough 把原 P2-04 改称“加权加性偏置边界”，不是对原问题的解决。
- C4 停止原因、最近可行 incumbent、完整候选审计持久化及研究环境锁定仍未落实；候选列表移除完整 DataFrame 的内存改动有价值，但不代表 P2-06 的停止语义也已处理。

## 4. 对原 13 类审查项的准确状态

| 原审查项 | 本轮可确认状态 |
|---|---|
| P1-01 多类别几何 | 低层选项与披露部分完成；研究入口配置未接通。 |
| P1-02 严格/松弛约束 | 两布尔值计算正确；历史松弛已披露；不是实现严格可行 AE，持久化与完整测试仍不足。 |
| P1-03 调查推断 | 已补充样本层范围说明；总体加权推断未实现，应明确延期。 |
| P1-04 名义编码 | 未修复，置换反例仍成立；可作为新版本方法学任务延期。 |
| P1-05 不可估计公平性 | 单群体矩阵和无阳性 EOpp 部分修复；缺失群体/未声明特征/单群体 DP 未修复。 |
| P1-06 来源与隔离 | allow_real_data 公共 setter 已保护；来源缺失、分区独立性、注入来源验证仍不足。 |
| P1-07 学术叙述 | 方向改善，但新增了未经识别的预算因果解释，须再修订。 |
| P2-01 缓存 | 未通过，新代码包含未调用方法的错误，真实 engine 反例失败。 |
| P2-02 调查边界 | 零事件、正无穷 mask、数值零权重离群值部分修复；新增资格标志仍放行 inf。 |
| P2-03 测试隔离 | 未闭环；原 D6 生产依赖仍在，且删掉了旧分类/缺失桶测试。 |
| P2-04 stress-elbow | 未实现相关改进，应明确延期。 |
| P2-05 执行来源 | 未通过：manifest 仍错、报告散列不符、旧 HEAD 测试不能证明当前补丁。 |
| P2-06 内存与停止 | 候选矩阵保留减少；停止语义及可行 incumbent 部分未改。 |

已确认有价值的修改不必回滚：单群体矩阵异常、EOpp 未定义返回 null、零事件比例、非有限正权重 mask、allow_real_data 只读属性、低层多类别选项、严格/松弛标志及候选矩阵保留减少。但它们必须与仍未解决的问题同时报告。

## 5. 下一轮返工与验收边界

1. **锁定候选与真实证据。** 修正完整 HEAD 和散列，保存全部相关源码/测试的快照与原始运行日志；声明本轮只完成边界/审计修复，清楚列出延期方法学项。
2. **修复已复现的正确性问题。** 缓存内容指纹与上下文；不可估计状态；survey 资格误判；生产预处理来源与 namespaced split 独立性；对象驱动 manifest。每项均须在实际调用链上有独立反例。
3. **重做隔离验证。** 禁止真实 NHIS 和继承真实 CSV 读取或模型拟合；采用新建临时目录，避免覆盖旧 run。将 D6 的生产前置条件完整替换为可验证合成依赖，不得放宽正式 release 的科学代码边界。保留 sentinel 和跨套件生命周期检查。
4. **修正文档范围。** 旧实验输出不变；结果不变性与代码行为不变性分别表述。名义编码、调查推断、自动选维及新搜索策略若未实现，标记延期，禁止 `Unresolved issues: None` 和“全面修复”。
5. **再次提交审计材料。** 按执行协议第 9 节报告实际测试、散列、被阻断访问和全部未解决项，结尾 `STOP — waiting for Codex review.`。本决定不授权工作者提交 Git、不授权真实数据重跑，也不要求为了消除方法学风险立即改变冻结研究定义。

这是一份明确的返工决定，不是要求用户再次确认已授权的独立代码审查。
