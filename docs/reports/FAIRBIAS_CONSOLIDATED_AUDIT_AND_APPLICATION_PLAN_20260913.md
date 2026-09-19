# FairBias 总审查与应用论文实验规划

日期：2026-09-13。目标：比较 FairBias 与其他论文的公平性方法，同时评价预测能力、公平性及跨年度稳定性，形成医疗健康应用论文。

## 1. 总体判断与证据范围

**目前适合推进“可复现的公平性方法应用比较”，尚不适合直接启动最终论文实验。决定仍为 REPAIR。** 需要先修复接口正确性，再建立所有方法共用的评价层。不能把“增加几个模型、输出 AUROC 与 DP”当作完整应用研究。

本报告合并历次已核实问题，并针对外部方法接入补查边界。**没有声称穷尽所有可能漏洞**；未覆盖全部依赖漏洞、全部数据分布和所有生产入口。下文是截至本次审查的已知问题总表，明确区分仍存在、已修复和新研究要求。历史缺陷不自动归因于当前代码，接口反例也不自动证明历史主结果已受影响。

审查对象是本地 `research/nhis-fairbias`、HEAD `67e6659fa65249a8842e34af5d8969629efe4bca` **加 15 个未提交候选文件**，不是声称 GitHub 上该提交已经包含这些修复。候选散列见[第四轮证据](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_FOURTH_REVIEW_20260913_evidence/final_verification.json)。

已完成的独立验证：

- 持续数据/网络 guard 下的指定核心套件：**163 passed，63.68 秒**；复制的既有探针 **13/13 PASS**。
- 第四轮调用路径检查：**7 项，3 PASS / 4 FAIL**，对应运行时哈希、真实注册表兼容及年份身份三个问题。
- 本次扩展矩阵：**12 项，1 PASS / 11 FAIL**。这是边界断言数量，不是 11 个独立新增漏洞；多个断言分别属于缓存、概率、类别、权重和记录身份问题。
- 另外重验了通用/D8 指标差异及真实 LR 的类别编号置换反例。所有新模型输入均为合成数据；未读取 NHIS 微数据、未重跑真实模型、未改实现或历史结果、未 staging/commit。
- 四臂历史状态比较通过，缺少参考臂时正确失败；15 个候选散列匹配，既有 9/60/33 项证据清单匹配，三个历史 R4 主 JSON 未变。

[第四轮详细报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_FOURTH_REVIEW_20260913.md)保留当时有限返工边界；本报告承接用户新增的“全面审查与应用文章规划”，新增研究工作不追溯改写旧 Gate。

本次扩展反例、来源快照与复现说明见[总审查证据目录](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_CONSOLIDATED_AUDIT_AND_APPLICATION_PLAN_20260913_evidence/README.md)。

## 2. 先统一研究对象

| 项目 | 应使用的定义 |
|---|---|
| FairBias | 特征间散度 → 特征及原点的 MDS 几何 → 贪心幂变换/类别合并。不是把受访者投影为 `XW` 的端到端交叉熵模型。 |
| `d_phi` | 表示层几何偏差代理；不等于预测 DP、EO 或健康权益。不能用它当作所有外部算法的共同胜负标准。 |
| HISP 臂 | `HISPALLP_A` 的七类别，不能静默改成 Hispanic/non-Hispanic 二分类。 |
| Arm 3/4 | Arm 4 从预测特征中去除六个残疾构成变量；不是仅分析某个残疾受访者子域。它应作为特征敏感性比较。 |
| 年度 | NHIS 重复横断面：2022 开发训练、2023 选择、2024 年度迁移评价；不是追踪同一批人。 |
| 结局 | `MEDDL12M_A` 为过去 12 个月因费用延迟就医；`MEDNG12M_A` 为因费用未获得所需医疗。不能称作未来个体患病风险。 |

代码依据：[adapter 定义](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:15)、[特征排除](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:176)、[结局注册表](/Users/lkc/Downloads/code_v_0_3/configs/nhis/features.json:1029)、[年度披露](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d6_temporal_runner.py:193)。

## 3. P0：本轮未确认

未确认规范 D6 路径把 2024 标签用于拟合，也未发现本轮篡改冻结结果。不能据此保证整个仓库零泄漏或不存在严重缺陷。下面的共享记录检查缺口是可复现的接口风险，尚无证据证明冻结年度主运行实际发生该类混入。

## 4. P1：当前实现中的六类重要正确性缺口

| ID | 位置 | 已核实问题与影响 | 完成条件 |
|---|---|---|---|
| C01 状态及配置缓存 | [state:21](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_state.py:21)、[AE:528](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:528)、[配置:239](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:239)、[配置:289](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:289) | 默认仍为八位小数 v1；幂 `3.000000001/3.000000002` 被误判循环，微小 minimum-gain 变化不刷新缓存。此外实际 LR 的 `C=1→2` 也不改配置指纹：复用 engine 跳过，新 engine 接受。后者使用人工效用隔离控制流。 | 所有运行时身份无损，历史比对显式 v1；绑定实际 estimator 类型和 `get_params(deep=True)`、编码器/缩放器配置、数据及约束。若不支持运行间复用，明确禁止。 |
| C02 预拟合注册表兼容检查失效 | [adapter:131](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:131)、[preprocessor:164](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:164) | 读取不存在的 `.feature_registry`；真实属性为 `.registry`，真实 schema 又没有顶层 `features`。真实合成已拟合对象修改 AGEP_A 有效编码仍被接纳。 | 比较真实属性和影响预处理的语义配置；必要项缺失拒绝。用真实对象做匹配与不匹配双向测试。 |
| C03 记录身份不规范 | [contracts:146](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:146) | 年份 `2023/2023.0`、ID `1/1.0` 可表示同一记录却生成不同身份；缺失和非整数年份也被接受。 | 年份必须有限整数；来源与 ID 使用明确结构及类型规范。覆盖 dtype、缺失、重复、部分交集、独立来源同号、显式来源与年度的组合；普通特征值相同不等于同一人。 |
| C04 概率契约不完整 | [candidate utility:537](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:537)、[terminal:320](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:320) | 默认取第 2 列，未绑定正类；合成 `classes_=[1,0]` 的正确预测被算成 AUROC=0。假 `predict_proba` 输出 -1/2 也能在 AE 中获得 VALID、AUROC=1。默认 sklearn LR 不触发这些反例，但外部包装器容易触发。 | 验证二分类标签、`classes_`、形状、有限性、[0,1] 与行和；按正类定位。风险分数、事件概率、决策概率和硬标签应采用不同输出类型。 |
| C05 未出现类别改变几何 | [bias_metric:263](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:263)、[weighted:293](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:293) | 两条观测 `c=[0,1]` 的散度为 1；仅设 pandas Categorical 的 levels=[0,1,2]，散度变为 2/3。未出现的 level 进入 K；有/无权重均复现。当前 NHIS 整数输出不代表此公共 API 边界已安全。 | 明确 K 是有效经验支持集，或明确采用固定理论支持并对所有 dtype 一致实现；零计数/零权重 level、合并后保留 level 均做不变性测试。 |
| C06 多条评价路径口径分裂 | [generic evaluator:153](/Users/lkc/Downloads/code_v_0_3/src/fairbias/evaluator.py:153)、[generic rate:169](/Users/lkc/Downloads/code_v_0_3/src/fairbias/evaluator.py:169)、[D8:185](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:185) | D8 单群体已返回 null，但通用 evaluator 仍返回 SP/EO=0。三组阳性率 0/0/1，通用 SP=2/3，D8 DP=1；前者是平均组对差，后者是极差，不能拼成一列。历史兼容定义本身可保留，应用层混用才会制造伪比较。 | 新 benchmark 统一评价服务，显式记录公式、聚合和可估计性。历史兼容输出加清晰命名，不能将其当作新应用主指标。 |

其中 C01/C02/C03 承接本轮返工；C04/C05/C06 属于扩展审查发现的公共接口和跨模块问题。旧的“未声明特征补零/单群体几何补零”在 D8 对应路径已修复，不能把 C06 写成 D8 修复没有发生。

## 5. P1：应用论文必须处理的六类方法学风险

### M01 名义编码任意性会改变“预测增益”

[预处理:461](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:461)保留整数类别，[效用:454](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:454)将其转 float 后缩放进入 LR。本次重新用真实 LR 验证：90 行合成分类数据，仅将类别 `[0,1,2]` 重命名为 `[0,2,1]`，AUROC 从 **0.5 变成 1.0**。成员关系与标签没有改变。

新应用版本应将语义清理、FairBias 几何表示和分类器编码分开。名义变量采用训练期拟合的 one-hot；有序变量另作预声明。LFR 的连续表示不再 one-hot。改变编码会改变 AE 选择，应新建应用基线，不覆盖历史 C1–C4，也不称作纯工程无影响修复。

### M02 调查权重与设计推断尚未形成完整评价层

[D8 cohort:639](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:639)丢弃权重与元数据，[fit:313](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:313)及指标没有使用 WTFA_A。[survey:206](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/survey.py:206)已正确披露这些边界。PSTRAT/PPSU 是方差设计信息，存在于文件中不等于已经进入推断。

应用文章建议以**调查加权评价为主、无权重评价为敏感性分析**。训练是否使用调查权重单独声明，首轮可保留各论文标准训练目标；不要因训练器支持 `sample_weight` 就宣称总体公平约束已实现。Reweighing 的公平权重、reduction 的内部成本权重、WTFA_A 分别保存。

均值/率的设计方差、模型间成对差值、非线性 AUROC/AP 的方差应分别验证。CDC 对年度比较、合并及方差变量有具体规定；本研究先逐年评价，不必把三年合成一个总体。域内比例过滤与设计方差问题分开，缺失 Y/A 后的可观测亚总体也要披露。[CDC 2024 Survey Description](https://ftp.cdc.gov/pub/health_statistics/nchs/dataset_documentation/NHIS/2024/srvydesc-508.pdf)

### M03 算法版本、几何目标与可行性不能互换

三件事需同时说明：

1. `d_phi` 与 DP/EO 是不同量，降低前者不推出改善后者。
2. 七群体的 `mean_pair` 与 `author_max_pair` 不等价。底层已提供两种分支，但 [FairEvaluator:294](/Users/lkc/Downloads/code_v_0_3/src/fairbias/evaluator.py:294)没有传递 aggregation，当前主路径仍走默认 mean-pair。新增函数选项不代表主流程完成作者版本切换。
3. [AE:359](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:359)及 [D8:894](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:894)使用 `epsilon+0.02`。已披露不等于严格约束已实现；SEX 的 0.0005 放宽到 0.0205，不能称为数值容差。

历史结果保留；新应用版本预声明聚合与 slack，严格可行版本记录最近可行解，任何不可行结果都保留标记。不要用外部方法的 DP 阈值替换 FairBias 的几何 ε，也不能把两个算法的参数数值相同视为“公平预算相同”。

### M04 2024 已被研究过程观察

[D6 disclosure:193](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d6_temporal_runner.py:193)明确 2024 已用于 D4/D5，其后还有 D8 分析。新实验可以做诚实的**回顾性跨年度比较**，不能再包装成全研究从未触碰的最终盲测。

从现在起冻结新协议，所有方法一次性完成选择再统一评价 2024；不得按其结果挑 seed、阈值、臂或方法。若要更强的外部验证，另选未观察且获准使用的年度/队列；先核对可用性，不预设某新年度已经发布。重分一个已经看过结果的数据集不能恢复原始独立性。

### M05 固定 0.5 阈值可制造“几乎人人阴性”的低差距

[D8:330](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:330)固定阈值。历史 [R4 表:78](/Users/lkc/Downloads/code_v_0_3/docs/reports/NHIS_D8_R4_PRIMARY_SUBSTANTIVE_EXECUTION_20260909T080325Z.md:78)中 SEX C3 在训练/验证几乎不预测阳性，测试仅一例预测阳性。DP/EOpp 接近零不能据此称作有用的健康公平改善。

所有方法需同时报告各组 TPR、FPR、PPV、阳性率和分母。新论文采用预声明服务资源/敏感度场景选择阈值，0.5 保留为历史敏感性对照。不存在已确认服务场景时，不凭空把某个 5% 差距或 80% 敏感度宣称为临床标准。

### M06 健康结局、测量时间与适用性

[结局配置](/Users/lkc/Downloads/code_v_0_3/configs/nhis/features.json:1029)测量过去 12 个月医疗可及性障碍，预测特征多在同次访问收集。年度外推不等于个体未来预测。建议研究问题为“识别自报因费用延迟就医者时，公平性方法的预测—公平权衡及年度迁移”。如果要定位成提前干预的预测工具，必须先证明特征在行动时点可获取。

Y/A 缺失者会在 [adapter:235](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:235)排除。抽样权重不会自动消除 item nonresponse、自报误差或群体标签测量差异。需报告加权缺失比例、纳入流程、各组有效事件数；改善统计差距不是已证明改善健康结局或因果权益。

## 6. P2：六类代码、数值与证据问题

| ID | 问题及位置 | 处理建议 |
|---|---|---|
| E01 有限单项权重仍可合计溢出 | [validate:201](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/survey.py:201)、[proportion:128](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/survey.py:128)、[category:152](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/survey.py:152)。两项 1e308 被 validator 接受，事件比例应为 0.5 却输出 0。 | 对总量和组内总量检查有限性，或先按公共尺度正规化再求比率；最终输出有限/范围校验。该反例是极端公共接口边界，不证明实际 NHIS 权重达到这一量级。 |
| E02 stress-elbow 稳定性与信息不足 | [bias_metric:431](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:431)。扫描单初始化、相邻平台判定，不提供稳定肘点或优化收敛保证。 | 保存完整 stress 曲线、初始化、迭代与所选维数；对种子/维数做预声明敏感性。改变选维规则另标算法版本。 |
| E03 搜索停止与最优性边界 | [AE 排名:154](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:154)、[joint:1185](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:1185)、[BM:434](/Users/lkc/Downloads/code_v_0_3/src/fairbias/mitigation.py:434)。特征级贪心、有限网格、首次可行停止，不能称全局 Pareto 搜索；外层未完整传播 BM non-convergence。 | 分离候选耗尽、预算耗尽、数值失败、策略停止、首次可行。全特征排序可作以后消融，非论文主实验的必需前置。 |
| E04 测试不能统一称“纯合成/全套通过” | [pipeline test:27](/Users/lkc/Downloads/code_v_0_3/tests/test_fairbias_pipeline.py:27)仍调用 COMPAS 默认数据路径；163 项是指定套件，不是全仓库。D6 指定 test_03 的 mock 前置通过不证明生产 release 当前可运行。 | 单元/合成集成/数据集成/冻结重放分别标记并分进程；CI 默认禁数据，持续 guard；生产检查保持失败保护。不要恢复真实数据读取来追求绿灯。 |
| E05 文档与证据仍可能误导 | [guide:7](/Users/lkc/Downloads/code_v_0_3/docs/GPT6_CODEBASE_AUDIT_AND_IMPROVEMENT_GUIDE.md:7)、[guide:69](/Users/lkc/Downloads/code_v_0_3/docs/GPT6_CODEBASE_AUDIT_AND_IMPROVEMENT_GUIDE.md:69)仍写 longitudinal、HISP 二分类、Arm4 子域和 C3 一般鲁棒。工作者把 60 项旧 manifest 写成 33 项。 | 统一方法词典与有效配置输出；清单计数自动生成。已修改候选的 source/环境/schema/状态哈希方案均绑定运行记录。保留历史 `.gitignore` 差异的事实，不新改继承文件。 |
| E06 扩展规模与资源不可见 | [subset enumeration:98](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:98)、[joint cost:1250](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:1250)。高 H 可退化为全部子集，BM 几何成本仍为 null。外部 reductions 的一次 fit 也包含多次基础模型训练。 | 限制组合数量，先以合成规模标定；记录实际基础模型 fit 次数、几何次数、墙钟时间与内存。当前 AE 已只保留候选摘要并重建胜者，不再重复列“所有候选保留完整矩阵”旧问题。 |

## 7. 已修复事项与 Arm 4 叙事

下列具体修复接受：getter 缺参；seed/科学配置散列及 smoke 预算；旧的 bound-method 指纹误用；部分来源交集/重复检查；未知来源预拟合对象拒绝；D8 单群体及缺失几何失败保护；非有限权重基础资格；布尔 JSON 类型；候选日志 writer；四臂历史哈希兼容；勘误历史字段归属。它们的通过范围由第四轮证据限定，不外推为全部接口安全。

现行 [Erratum:118](/Users/lkc/Downloads/code_v_0_3/docs/reports/NHIS_D8_R4B_SUPERVISOR_ERRATUM.md:118)已经承认 MDS、BM 网格和阈值来源共同改变。这比旧因果表述严谨。文章中建议进一步统一为：

> 在已比较的配置中，C3 的预算终点相同，C4 的轨迹和终点不同。Arm 4 的探索性效用增益未在主配置中复现。比较同时改变多个因素，不能识别单独的几何因果效应，也不证明 C3 普遍路径鲁棒。

路径现象可作解释性补充，不应取代外部公平性方法比较成为应用论文唯一卖点。严格阈值可行性与相对 C1/C2 的效用分别报告；相对 C2 恢复一些 AUROC，不意味着超过未处理 C1。

## 8. 外部公平性方法：建议首轮组合

研究主线是比较公平性方法；**预测能力和公平性同时评价**。先固定 LR 骨干，再以一个预声明 GBDT 做稳健性检查。不要从外部论文拷贝最终分数横比，必须在同一数据协议下重新训练。

| 方法 | 用途 | 接入与论文归属 |
|---|---|---|
| C1：未处理 LR/GBDT | 每个骨干各自的预测基线 | 所有相对增益均同时相对 C1 和 FairBias-BM，防止基线选择偏差。 |
| FairBias-BM、BM→AE、Joint | 项目内部三种方法 | 历史锚点单列；新 one-hot/严格约束等应用配置写清变体名，不能混称作者原版。 |
| Reweighing | 简单、有解释性的预处理对照 | Kamiran & Calders, 2012。优先核验 AIF360 sklearn API 的多群体权重矩阵及零频单元；不要使用只分 privileged/unprivileged 的包装器把 HISP 改成二分类。[官方实现说明](https://aif360.readthedocs.io/en/stable/modules/generated/aif360.sklearn.preprocessing.Reweighing.html) |
| LFR | 与 FairBias 最接近的表示学习对照 | Zemel et al., ICML 2013。主对照为学习表示后接同一骨干，标明“LFR representation + LR”；原生预测头可附录展示。[论文](https://proceedings.mlr.press/v28/zemel13.html)、[实现](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.LFR.html) |
| Exponentiated Gradient—DP | 明确优化预测 DP 的处理中方法 | Agarwal et al., ICML 2018。基础模型需支持内部样本权重。[论文](https://proceedings.mlr.press/v80/agarwal18a)、[API](https://fairlearn.org/main/api_reference/generated/fairlearn.reductions.ExponentiatedGradient.html) |
| Exponentiated Gradient—EO | 控制 TPR/FPR 差距的处理中方法 | 与 DP 版本分别报告，不能在 2024 上择优只留下一个。 |
| ThresholdOptimizer—EO | 强后处理对照 | Hardt et al., 2016。基础评分器先冻结，阈值仅用开发校准数据；预测时需群体信息，必须单列部署条件。[论文](https://arxiv.org/abs/1610.02413)、[API](https://fairlearn.org/main/api_reference/generated/fairlearn.postprocessing.ThresholdOptimizer.html) |

共九种条件（C1–C4 加五个外部条件），并非九种完全独立算法。LFR 官方实现只接受一个 privileged 与一个 unprivileged group，因此首轮放在二群体臂；HISP 七群体该格明确为 NOT_SUPPORTED，不能为了填满表改问题。[LFR 源码](https://github.com/Trusted-AI/AIF360/blob/main/aif360/algorithms/preprocessing/lfr.py)

资源充足时增加 **LAFTR** 作为深度表示学习补充，并配同结构、同优化预算的无公平惩罚神经网络；它是单独模型族，不能当作固定 LR 的纯方法效应。[Madras et al., ICML 2018](https://proceedings.mlr.press/v80/madras18a.html)

DisparateImpactRemover 暂不作为跨年度主对照：官方文档明确提示训练/测试的条件分布假设，不能分别对各年 `fit_transform` 后称严格 inductive 无泄漏。只有训练期拟合、未来数据固定映射的接口被核验后才纳入。[官方限制](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.DisparateImpactRemover.html)

本组合旨在覆盖表示学习、重加权、约束优化及后处理，不宣称代表 2026 全部最先进方法。投稿前补查目标期刊近三年同类应用工作，再决定是否需要一项较新且有可复现实验代码的方法。Fairlearn 本次查阅页面为 `main/0.15.0.dev0`，用于核对接口语义；正式实现需锁定经验证的发布版本和 commit，不能直接依赖漂移的 main。

## 9. 公平比较的共同协议

### 9.1 数据与选择

推荐主结局保持 MEDDL12M_A；MEDNG12M_A 作为预声明的次要迁移结局。三类保护轴为 SEX、七群体 HISP、DISABILITY，full/exclude 是残疾特征敏感性，不能当成四个独立人群证据。

年度仍为 2022/2023/2024。保持同一臂内所有方法相同纳入样本、X/Y/A、缺失规则和 ID 清单；跨臂有效样本可以不同，必须报告。2022 用于所有预处理与模型/表示学习。**2023 区分 calibration 与 selection 两个固定角色**，后处理只在 calibration 拟合，在 selection 选择；其他方法也使用同一 selection。

该开发内部划分属于新 benchmark 协议，不调用历史 adapter 禁止的随机跨年度拆分。划分应预声明并考虑 PSTRAT 内 PSU 边界及各组事件覆盖；如果样本不足，选用事先设计的开发期交叉拟合替代，不临场将 calibration 和 selection 混在一起。全部 2024 标签隔离到最终评价器。

2024 已曝光的限制不能消除：新设计是冻结后的回顾性年度评价。测试期指标可作为结果报告，不能再反馈修改方法清单/阈值/种子；后续修改另编号探索实验。

### 9.2 预测输出必须区分两种含义

**风险分数/事件概率**用于 AUROC、AP；只有有意义的事件概率才解释 Brier 与校准。**决策概率** `q_i=P(decision=1|X_i,A_i)` 用于随机化分类器的期望混淆矩阵。Exponentiated Gradient 原算法返回随机化决策，不能反复 `predict` 挑一个好 seed，或强行 `q>=0.5` 后继续声称原公平保证。[官方随机化说明](https://fairlearn.org/main/api_reference/generated/fairlearn.reductions.ExponentiatedGradient.html)

若由 mixture 得到 q，明确其构造及接口版本；q 的 AUROC/AP 可以作为决策排序指标另报，不能冒充校准后的疾病/医疗需要概率。ThresholdOptimizer 的基学习器分数 AUROC与后处理决策公平性分别列出，不应把基模型 AUROC加后处理 DP 拼成一个含义不明的模型点。

跨全部方法的主决策比较可以使用期望 TP/FP/TN/FN、balanced accuracy、TPR/FPR/PPV 与服务覆盖；真正的风险评分器再比较校准。无分数方法填 NOT_AVAILABLE，绝不能用硬标签计算 AUROC来补齐列。

### 9.3 共同指标与权重

| 层次 | 必报项 |
|---|---|
| 样本与缺失 | n、加权总量、Y阳性数、每组正/负事件数、Y/A缺失、分析域、权重分布；Kish effective n 仅作权重不均诊断，不代替设计自由度。 |
| 预测排序 | 加权 AP（明示 average precision）、AUROC；无权重版本并列。不要将 AP 与梯形 PR-AUC 混称。 |
| 概率质量 | Brier、总体与分组校准图；只用于事件概率输出。校准拟合仍限开发期。 |
| 决策表现 | sensitivity/TPR、specificity/FPR、PPV、balanced accuracy、阳性预测率、期望/实际事件数。 |
| 公平性 | DP极差、TPR极差（EOpp）、FPR极差、EO=max(TPR极差,FPR极差)；七群体同时给逐组和21组对估计。 |
| FairBias机制 | train/selection/evaluation 的 d_phi、严格/松弛可行性、改变特征、首个路径分歧、接受/拒绝理由。作为内部机制指标，不作跨表示空间的统一量尺。 |

各组加权率如 `TPR_g=sum(w*y*q*I_g)/sum(w*y*I_g)`；确定性分类器的 q 为 0/1。分母为零返回不可估计并保留原因。风险模型的阈值按开发期服务预算或敏感度目标冻结；新数据的实际预算偏移另外报告，不能在 2024 重新调阈值以“维持相同表现”。

**训练公平权重不用于最终评价总体。** Reweighing 若乘 WTFA_A，必须同时说明因子基于何种分布计算，作为单独训练策略；EG 内部 oracle 权重也不意味着公平约束已按调查总体加权。

### 9.4 调参和选择规则

建议每方法最多 8 个预声明配置作为首轮上限，先在合成数据测试资源可行性。主模型骨干的超参数在共同开发规则下确定；公平方法强度按本身参数含义建网格，记录总基础模型 fit 次数和时间。资源上限不足时统一缩减或报告超时，不能只给 FairBias 更多搜索机会。

DP、EO方法的“允许偏离总体均值”不必等于报告中的最大组间差。Fairlearn 的 `difference_bound` 与优化器 `eps` 也不应混淆。最终按共同 selection 指标判断实际可行性，保留每个方法内部约束含义。[DP 定义/API](https://fairlearn.org/main/api_reference/generated/fairlearn.reductions.DemographicParity.html)、[EO 定义/API](https://fairlearn.org/main/api_reference/generated/fairlearn.reductions.EqualizedOdds.html)

预声明一条主选择规则，例如“达到开发期敏感度/服务预算底线，在共同公平差距容许范围内最大化 balanced accuracy；不存在可行候选则报告不可行”。AP/AUROC作为并列预测指标。具体公平界限和应用底线需由研究问题与开发数据支持，不能预设成通用临床阈值。除选定模型外保留全部验证候选图；最终评价只展示预先选定的候选集合，不依据测试结果挑 Pareto 点。

### 9.5 不确定性与比较

每一对方法在相同受访者上产生预测，因此差值必须配对计算。调查分析先建立完整 PSTRAT/PPSU/WTFA_A 设计，再以分析域作子集；不将行级 iid bootstrap 或普通 DeLong 直接作为 NHIS 设计推断。

线性率先与成熟调查统计实现的手算/参考例子核对；非线性 AP/AUROC及方法差值用与该设计相容且经验证的复权/重采样方案。**最大组间差具有非光滑性**，尤其多个群体并列时，应考虑组对同时区间或保守界，不能无检查地把普通 percentile bootstrap 称作严格95%覆盖。

固定模型的抽样不确定性、训练种子不稳定性、开发选择不稳定性分别报告。随机方法建议预声明 5 个种子；确定性 LR 不用五份相同结果制造重复证据。不能把 bootstrap 副本或 seeds 当成独立受试样本做显著性检验。确定主对比家族，多重比较采用预声明校正或同时区间；缺乏精度时报告不确定，不强行宣布优胜。

## 10. 一次验收整个问题族：必须覆盖的矩阵

| 组件 | 关键正反例 |
|---|---|
| 身份/分区 | 年份与ID整数/浮点等价，字符串政策，空/缺失/非法年份，重复，部分交集，独立来源同号；测试标签置换不能改变已冻结模型。 |
| 配置/缓存 | 实际 model/scaler/encoder 参数逐项变化，新对象与复用对象结果一致；微小参数；history v1与runtime v2；重启顺序与候选顺序。 |
| 语义编码 | 类别重命名不变性；空level；未知类别；缺失哨兵；合并代表编号不改变模型含义；categorical与numeric互斥且覆盖完整。 |
| 输出/指标 | classes_反序；越界概率；非有限值；单组/单类/零分母；空样本；概率与决策类型；二群体手算和七群体21对；同一 metric registry。 |
| 权重/调查 | 全1退化为无权重；共同倍乘不改变率；零权重不改变支持；极端但有限权重；错索引；单PSU层/设计缺失；域估计保留设计。 |
| 外部方法 | fit只用训练；后处理仅开发calibration；inference不读取y；不支持的群体/权重/评分能力必须显式拒绝；原论文目标与包装实现一致。 |
| 持久化/复现 | 有效参数自动导出；异常/超时保留事件；selected与committed分开；bool/null正确；来源/schema/算法/依赖版本散列；旧结果不覆盖。 |

矩阵是有限的验收要求，不是宣称这些场景当前全部通过。提交前完整执行，而不是每次只针对最后一个报错加一条镜像测试。

## 11. 项目架构与分阶段工作

建议新增独立应用 benchmark 模块，复用已经审查的数据语义与 FairBias 算法；不要继续把所有外部方法塞入历史 D8 runner。

数据边界保存 X、Y、A、调查权重、设计与记录身份；语义预处理、方法适配器、公共评价、推断和报告导出分别负责自己的工作。每个适配器声明：是否改 X、是否改训练权重、是否需要 Y/A、预测时是否需要 A、是否随机化、输出是风险还是决策、是否支持多群体及调查训练权重。测试阶段数据只能交给冻结预测器与评价器。

| 阶段 | 交付物 | 完成门槛 |
|---|---|---|
| G0 正确性封口 | C01–C06及E01修复、有限边界矩阵、有效配置和证据自动生成 | 真实对象路径的合成正反例通过；旧历史锚点仍一致。保留第四轮已经通过的修改。 |
| G1 应用协议与共同评价 | 结局/臂/缺失/时间角色、one-hot应用版本、指标词典、权重估计、设计推断、评分类型 | 方法选择前冻结主次指标与应用场景；手算调查例子及配对差值通过。原先延期的编码/调查任务在此成为新阶段工作。 |
| G2 外部方法集成 | RW、LFR、EG-DP、EG-EO、TO-EO适配器，LR主骨干与GBDT敏感性，锁定依赖 | 各方法在同一合成数据验证训练隔离、七群体能力、输出语义、参数和随机性；无支持的格子明确NA。 |
| G3 冻结比较执行 | 签名配置、方法×臂×种子清单、候选预算、一次性的年度评价及独立聚合核查 | G0–G2通过，研究协议及真实数据执行边界确认后运行。历史D6/R4不覆盖、不重命名。 |
| G4 稿件与复现包 | 表/图/补充材料、局限性、运行说明、可共享聚合工件 | 所有文章数字由可追溯结果自动生成；比较包含失败/不可行结果，无测试集择优。 |

实施量可按 G0约2–4工作日、G1约3–5日、G2约3–5日、G3约2–4日、G4约5–10日预留。这是单人专注工作量的规划范围，不是完成承诺；设计推断与外部实现兼容性可能主导耗时，应以验收门槛而非日期放行。

控制规模：首轮仅主结局+LR完成全部支持的格子；再扩展GBDT与次结局。9条件×4臂×2骨干×5种子最多360个最终条件格子，是规模上界，**不是360次模型fit**；LFR不支持格和确定性重复应扣除，内部调参/AE/EG成本另计。

## 12. 论文组织与 P3 拓展

建议主题：**Predictive performance and fairness trade-offs in identifying cost-related delayed care: a temporal evaluation of bias-mitigation methods using NHIS**。若加权推断通过，可在题目/摘要明确 survey-weighted；此前不提前宣称。

文章回答三个问题：公平性方法改变了哪些群体的识别表现；改善是否伴随可接受的总体/最差群体预测代价；结论是否随年份、分类器和残疾构成特征改变。FairBias即使不是最优，可靠的失败条件、差距缩小但实际识别恶化、年度迁移中的约束失效，也有应用研究价值。

建议正文：纳入流程图；样本和缺失表；主预测/公平性/配对区间表；逐组TPR/FPR及服务覆盖图；经过开发选择的公平—效用候选图；校准和年度稳定性图。图中分开 deterministic与randomized、风险与决策，不把异义分数拼成漂亮前沿。Arm4路径因子消融放补充，避免主线变成算法证明。

报告可对照 [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378) 的预测研究透明度项目，说明结局时点、群体评价、局限、代码与协议。它是报告指导，不能作为统计有效性认证；本研究为可及性障碍识别时也需明确与临床诊断/预后任务的差别。

P3后续价值较高的扩展是：真正未观察队列的外部评价；LAFTR/较新表示方法；非预定义保护轴的公平泛化；少数交叉群体的估计可靠性；冻结因素的几何×幂网格×阈值来源消融。可微优化或全特征全局排序不是当前应用文章必须完成的任务。

本次交付为审查、证据和规划；没有实施这些方法或产生新的真实比较结果。不能保证投稿录用，也没有预设 FairBias 应当优胜。
