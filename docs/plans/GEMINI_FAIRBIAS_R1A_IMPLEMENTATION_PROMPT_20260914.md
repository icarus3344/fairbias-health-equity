# Gemini：执行 R1A 核心修复，不再重写 R0 规划

工作目录 `/Users/lkc/Downloads/code_v_0_3`。你是 implementation worker，Codex负责独立验收。用户目标是以FairBias为主方法发表应用研究，并公平比较预测能力与其他论文的方法。

**Codex已完成RESULTS-R0.1监督补正，现仅激活R1A。请实际修改下列获准核心实现并运行隔离的合成验证，提交报告后停止。不得自动继续R1B，不得训练真实NHIS或执行benchmark。** 原worker的 `R1_EXECUTION_SPEC.md` 和 `CORRECTIONS.md` 含错误公式及不存在的函数，以本文件、监督报告和机器定位为准；不用再创建R0.2规划或重复请求已有范围许可。

## 1. 先读并建立本轮快照

读取 `docs/AI_EXECUTION_PROTOCOL.md`、`AGENTS.md`、`GEMINI.md`、`docs/reports/FAIRBIAS_RESULTS_R0_1_SUPERVISOR_REVIEW_20260914.md`，以及其同名 `_evidence/` 下的 `verification.json`、`source_symbols.json`、`REVIEWED_CONDITION_REGISTRY.json`。再读下面列明的真实函数全文及直接调用方，不能仅看规格或函数名。

保持branch=`research/nhis-fairbias`、HEAD=`67e6659fa65249a8842e34af5d8969629efe4bca`。核验输入与监督证据一致，保存当前dirty文件的前像与hash。已有dirty变化是基线，不得撤销。14个继承文件和.gitignore不动；旧报告、计划、release、运行、图表不覆盖。禁止stage/commit/push、网络、依赖安装和微数据读入。

### 获准修改的已有源码（每次修改需关联下表问题）

`src/fairbias/enhancement_state.py`、`enhancement.py`、`enhancement_contracts.py`、`bias_metric.py`、`evaluator.py`、`config.py`、`mitigation.py`；`src/nhis_fairbias/adapter.py`、`preprocessing.py`、`survey.py`、`d8_enhancement_runner.py`。

允许新增最小共用模块 `src/fairbias/prediction_contracts.py`、`src/fairbias/application_metrics.py`（确有需要才新增）；不新增虚构名称的空壳类，不重构无关模块，不改 `src/nhis_fairbias/benchmark/`、旧tests或 `scripts/run_nhis_d8_r4_substantive.py`。该脚本及旧测试可读，若其额外代码修改成为必要条件，提交具体差异并停止对应工作。

允许新增测试和工具：

- `tests/synthetic/test_r1a_state_and_config.py`
- `tests/synthetic/test_r1a_identity_and_probabilities.py`
- `tests/synthetic/test_r1a_registry_and_weights.py`
- `tests/synthetic/test_r1a_geometry_contracts.py`
- `tests/synthetic/test_r1a_guard.py`
- `scripts/run_fairbias_r1_guarded_tests.py`、`scripts/_fairbias_r1_guard.py`

允许在唯一 `runs/r1a_contracts_<UTC>_<id>/` 保存合成证据，新增唯一 `docs/reports/fairbias_r1a_worker_<UTC>_<id>/` 审计包。结果目录由工具内部以UTC+随机ID生成，独占创建；重复/失败运行不覆盖。没有必要修改的文件不要动。

## 2. 必须在真实调用路径修复的内容

行号为本次监督快照位置，实施后以AST重新导出；不得复制旧行号冒充新源码位置。

| ID | 真实入口 | 必须实现的行为 |
|---|---|---|
| C01 | `enhancement_state.py:21 canonical_json_dump`、`:53 hash_transform_state`、`:210 StatefulCandidateTracker`；`enhancement.py:110 configuration_fingerprint`、`:187 _build_candidate_cache_context_fingerprint`、`:421 enhance_step`；`enhancement_contracts.py:216 compute_configuration_fingerprint`；D8 `run_arm` 的1045/1071/1137附近Joint状态记录 | 已有v2无损序列化，不重造一个未接入版本。新运行时所有父/候选/提交/循环/缓存身份明确走v2；包括别名changed_dict_hash。旧历史比较显式走v1。配置身份包含实际estimator类型、get_params(deep=True)、实际预处理/缩放配置、seed、epsilon和slack、数据身份；修改有效值使缓存失效，未变化复用。非有限状态/不支持的参数类型明确拒绝，不静默把NaN与None合并成有效身份。 |
| C02 | `NHISStudyAdapter.__init__`（adapter.py:92，预拟合分支约131）；`NHISPreprocessor.__init__/_validate_registry`（preprocessing.py:159/173） | 比较真实 `.registry`、实际影响清理/编码/范围的schema字段，与adapter的feature_registry一致；未知或缺失必需元数据拒绝。仅比较名称集合不够。不改变历史数据拟合规则，不读取真实数据。 |
| C03 | `EvaluationPartition.validate`、`_extract_record_ids`、`_calc_fit_fingerprint/_calc_selection_fingerprint`（enhancement_contracts.py:87/147/168/177） | source/year/ID结构化规范。年份有限且整数值，2023与2023.0按明确数值来源规范一致；bool、NaN/Inf、2023.9拒绝。通用核心不硬编码2022–2024；应用研究年度范围留在NHIS层。ID字符串前导零保留，不能通用int()吞掉；数值1/1.0身份按来源一致。显式来源和年度不能靠子串判断包含关系。缺失来源、重复、部分交集、不同来源同号、同特征不同记录均有明确策略；不让普通特征相同变成泄漏证据。 |
| C04 | `evaluate_candidate_utility`（enhancement_contracts.py:375，概率提取约537）；`evaluate_representation`（d8_enhancement_runner.py:237，概率提取约320）；`FairEvaluator.fit_and_predict`（evaluator.py:51） | 统一正类概率校验：二分类classes_恰好覆盖声明标签，按正类位置取列，形状(n,2)、长度、有限、[0,1]、行和（显式容差）正确。不存在predict_proba/单类/错误标签不能伪造风险概率。候选与终局均实际调用；错误输出产生INVALID/明确异常，不能给VALID高分。核心现有AUROC效用在R1A不改为BA；应用AE的C-BA目标留待R1B/R3。 |
| C05 | `compute_pairwise_divergences`（bias_metric.py:119，类别支路约263/293） | K使用实际正支持的经验类别；仅增加未观察Categorical level不改变散度。无权重/有权重、零权重level、合并后保留空level均一致。不改成另一个归一化目标。 |
| C06 | `FairEvaluator.compute_metrics`（evaluator.py:110）；`compute_group_fairness_gaps`（d8_enhancement_runner.py:185）；需要时新application_metrics共用模块 | 保留显式标注的历史mean-pair输出作为兼容路线，应用预测DP/EO使用统一max−min定义和全expected_groups可估计性。DP需组总支持；EO需各组正/负支持，某组缺失不能按剩余组给主指标。新应用指标不能与内部d_phi混名。单群体预测公平性返回None+状态，而不是0。历史和应用公式在输出标识及调用接口可区分。 |
| E01 | `weighted_binary_proportion`（survey.py:116）、`weighted_category_proportion`（:140）、`validate_survey_weights`（:168） | 有限单项仍可能总和溢出。比例计算可先除最大正权重再累加；validator不得静默返回未标注的重标权重改变“总量”含义，不可用可能溢出的mean作为首个缩放量。全局及组内分母、形状、对齐、范围明确校验。两个1e308给0.5或明确拒绝，绝不0或伪VALID。保持现有缺失代码语义。 |
| G01 | `FairBiasConfig`（config.py:97/resolved:223）；`FairEvaluator.calculate_epsilon`（evaluator.py:264）；`compute_dphi_matrix`（bias_metric.py:555） | 显式配置multigroup_aggregation，并真正传到底层，纳入有效参数与配置身份；非法模式拒绝。历史默认值如需保留须有命名兼容，application-v1显式选择已声明的author_max_pair分支，不以默认mean_pair冒充。不要用外部max−min TPR/EO替代内部几何。 |

R1A中完整benchmark和新F-only BM适配仍为未完成事项；不得因上述核心修复就关闭F01/F02。C06若旧接口因兼容仍提供旧数值，明确标记legacy公式、测试路由隔离；不能声称历史公式被“修成了”新指标。

## 3. 算法规则锁定：禁止按原worker错误描述改写

以下是本地实现核验，不是新增论文保真认证：

1. `get_subsets(['a','b','c'],1)` 的集合应为 `{a,b}`、`{a,c}`、`{b,c}`、`{a,b,c}`。H=1表示最多排除一项，不是最多保留一项，也不是“holding one constant”。
2. BM幂序列是 `[3,1/3,5,1/5,...,1999,1/1999]`，1998项且顺序不变。AE六值网格是 `(1/7,1/5,1/3,3,5,7)`。不要缩短作者幂流换取通过；合成例让其早停或通过预算显式中止。
3. `_search_numerical` 的restart是每次该属性搜索start_index=0；monotone_cursor是另一策略。失败最高d_phi属性时主模式stop，不能临时切next。只纠正文档或必要配置传播，不新增“停滞后重启其他属性”的行为。
4. `compute_shapley_distance_matrix` 的author_max_pair是每个上下文 `abs(max(v1)-max(v2))` 后在上下文间平均。对 v1=(.9,.1)、v2=(.1,.9)，其值0，而max逐分量差为.8；同一向量的max组对差与max-min相等，不能用它们作为区分反例。group-pair向量与受保护组预测率不是同一个对象。
5. `_nmi_gate_ok` 保留属性级 `(before-after)/(before+1e-10) <= phi_threshold`，导出实际phi_threshold，包括当前默认100的含义；不得改成未定义tau全矩阵NMI式，不据gate存在声称已保证预测效用。
6. `_make_candidate` 当前要求该属性d_phi严格小于epsilon；终局可行性检查允许<=时须单独记录。有效epsilon、实际不等式和停止原因入账。达到约束只标FEASIBLE/NO_TRANSFORM_REQUIRED，不标全局最优。无候选、非有限、未收敛、预算耗尽与合法no-op分开。
7. paper模式继续拒绝直接启用AE；应用AE/Joint在R1B/R3使用独立命名的配置构造，显式保留其BM几何策略、epsilon_candidate和slack=0。R1A不放松paper模式禁令以让测试通过。只验证有效配置传播及旧保护仍生效。

## 4. 明确合成验收矩阵

把下列ID写入测试名或参数ID，按问题分组报告。测试数量不是验收目标，必须验证实际行为。每类至少正例与对应拒绝/不变性例；不编造历史pass数量。

- **STATE-1/2/3**：3.000000001与3.000000002运行时不碰撞；已存在v1在历史显式模式仍兼容；min_utility_gain的细小变化改变有效配置身份。
- **CACHE-1/2/3**：真实LR C=1→2、缩放配置变化使同一engine缓存失效；参数未变保持一致。真实AE入口和D8 Joint提交/循环检测路线都验证，不只测试hash函数。人工效用可用于隔离循环判据，但另有真实对象集成，分别计数。
- **REGISTRY-1/2/3**：真实合成拟合preprocessor匹配接受；修改AGEP_A有效范围/编码拒绝；缺registry/provenance拒绝。只看key集合的实现必须失败此测试。
- **IDENTITY-1…6**：等价数值年份/ID规范化；2023.9/bool/缺失拒绝；显式source/year不靠子串；部分交集与重复拒绝；不同source同号区分；同特征不同ID不误判。generic核心可接受研究范围外的合法整数年份。
- **PROBA-1…6**：classes_=[1,0]正确正类；合法两列可计算；-1/2、Inf/NaN、错误行和/维度、单类/未知标签拒绝；候选与终局两路径分别验证。行和用声明的容差，不要求浮点逐位等于1。
- **CATEGORY-1…4**：c=[0,1]跨两组对应散度1；增加level2不变；零权重类别不扩大K；合并后空level不改变几何。
- **WEIGHT-1…4**：1e308两项比例0.5或拒绝；公共尺度倍乘不改变率；负/非有限/全零/错索引拒绝；组内支持不足和混合大小权重有稳定状态。
- **METRIC-1…4**：单组不可估计；三组阳性率0/0/1的应用DP=1，legacy mean-pair=2/3但命名不同；七组缺一组主EO不可估计；DP支持足够但EO缺负类时分别处理。
- **GEOM-1…6**：H1上下文；两种非等价双向量聚合反例；配置实际到达compute_dphi_matrix（spy证据）；真实小型MDS/BM调用证据；restart与failed_attribute_mode分别覆盖；NMI边界、严格epsilon边界及无效几何拒绝。
- **ISOLATION-1/2/3**：主合成调用不读D6/真实数据；改合成selection之外的S/T不得改变F/C状态；非法变更不能通过测试阶段临时换实现或禁用guard。

合成实际模型例每个分区最多200行、语义特征最多8个、MDS低维；专门权重溢出和类别反例不受实际NHIS取值范围限制。禁止训练整个真实benchmark或用旧D6字典代替真实BM集成。D8的依赖注入只提供人工合成对象和常量，日志标注注入范围，不能把它称为新主FairBias适配器已验证。

## 5. 先实现guard再导入项目；具体运行方式

使用 `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`。本轮元数据：Python3.13.2/numpy2.2.6/pandas2.3.1/scipy1.16.1/sklearn1.7.1/pytest8.4.1。运行前重新确认，不安装缺包。fairlearn/aif360缺失不妨碍R1A；不导入benchmark适配器包。

工具完成后执行以下完整入口（创建ID由工具内部负责，无需替换占位符）：

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 scripts/run_fairbias_r1_guarded_tests.py --phase r1a --output-root /Users/lkc/Downloads/code_v_0_3/runs
```

该入口仅允许三段流程：标准库preflight与独占创建输出 → guard sentinel → 上述五个明确合成测试文件。禁止 `pytest tests` 全量收集。父控制器可用固定argv启动同一已核验解释器的隔离子进程，用于超时/峰值内存监视；父进程不导入项目、不读数据，子进程中拒绝所有再生子进程。此有限授权不允许shell、下载或任意外部命令。若无需父进程，可以同进程验证，但不得声称没有实施的强制资源上限已经成立。

guard的具体要求：

- 在pytest、项目和第三方库导入之前安装，禁用pytest插件自动加载、用户site和用户自定义导入；禁写pyc，pytest cache和临时fixture均放唯一输出目录。
- 在解析resolve后的路径上控制访问，使用sys.addaudithook及必要拦截覆盖builtins.open、io.open、Path.open、os.open；文件描述符和路径逃逸有明确策略。拦截真实socket事件或socket.socket.connect/connect_ex，不能访问不存在的模块级socket.connect。
- 项目读入只允许本轮列明源码、被导入的src/*.py源码、五个新测试文件、明确列入manifest的非个体配置文件；运行时允许当前解释器的stdlib、site-packages和必要平台库。全项目目录和docs/plans/不作为数据白名单。preflight已存的聚合常量与runtime数据访问分开。
- 禁止data目录、所有真实parquet、根目录两个数据CSV、docs/releases、旧runs/outputs/artifacts；不得用文件名伪装或将真实数据复制进fixture。写只允许本次输出目录中的明确子目录。绝对路径、软链接解析和路径前缀边界必须自检。
- 用本轮新造、不含敏感内容的sentinel测试四种open路线、路径逃逸、网络、子进程拒绝；每种拒绝记录机制和事件。不要尝试打开真实数据来证明guard。sentinel预期拒绝不等于测试失败；其他未预期拒绝必须完整保留并停止相关运行。
- 全程有效，包括收集/fixtures/模型拟合/预测/序列化；不能只在单个test里mock。不得把Python guard描述成覆盖所有C扩展的操作系统沙箱；本轮禁止项目使用原生文件读入/额外进程，不包含操作系统隔离未验证的保证。

preflight检查源码前像hash、受保护文件Git差异名、无stage；不为了hash读取原始微数据。源哈希异常不能更新预期值蒙混。controller对子进程最多15分钟、4GiB，记录单位/实际测量；超出即保留错误与失败结果，不降格伪PASS。guard未通过只修本轮guard并用新run_id重试，不得启动项目测试。

## 6. 交付与验收停止点

提交实际patch、问题→函数→测试→输出的矩阵；未实现项明确OPEN。提供：源码修改前后hash、有效参数、调用/状态证据、guard前置与全程日志、所有测试节点及退出码、命令argv/cwd/解释器、峰值内存/耗时、预期和非预期异常、旧兼容范围、独立模型集成与mock控制流分开的结果。

报告数值直接从最终机器输出渲染，不手抄散列。最终MANIFEST在报告定稿后生成，包含报告但排除自身；拒绝覆盖旧manifest。若某实际兼容测试必须额外读取历史工件，则先提交精确聚合文件列表和理由，继续其他独立工作；本prompt不允许直接读取D6。

正式报告使用协议第9节全部标题，结尾为 `STOP — waiting for Codex review.` 不自批准、不提交Git。停止时说明R1A已做与未做，不能宣称论文公平性改善、FairBias优于对照、R1B完成或2024可重跑。

## 7. 保留的后续研究计划（现在不执行）

R1B按已监督修正的接口要求落实真实F-only FairBias、p/q/yhat、数据适配和所有对照。阈值器显式表示全阳性/全阴性策略；不能把概率=1当成一定存在全阴性边界。新R1B规格不得继承原稿虚构类、H1/restart误读、核心年度硬限制或几何/EO混名。

R2完成全年设计+domain、配对共享复制、独立方差参考、非光滑gap覆盖检查、B=2000/95%有效复制/完整df与SE、零SE退化状态；S按method×arm×backbone选完整配置，不选最佳seed；失败不能缩小20家族。主对照为RW/LFR/EG-DP/EG-EO/TO-EO，Unmitigated仍保留强预测参考及描述性差值。

R3按监督修正版registry覆盖76条件、LR/GBDT、AE/Joint、加权下游训练及Arm004路径；算法seed=[0,7,19,37,73]，随机拟合包括GBDT同收益分裂，确定性结果不冒充独立样本。LFR/EG预算、已安装库版本、完整默认参数、模型fit次数和资源须入账。未定义的数据支持数字不得填造。先合成演练，后按精确授权做2022/2023支持审计和开发。

R4在冻结后评价2024，披露其既往已被观察，仅称回顾性跨年评价。完整报告预测/公平/组覆盖、失败与资源。Arm004的路径解释需要固定其他因素和首个分歧证据。以上均未激活，依赖与真实数据授权不在本轮范围。
