# FairBias 主方法应用论文：完整对比实验实施方案

日期：2026-09-13。性质：供独立审查后逐 Gate 实施的研究方案；不是实验结果，也不是已放行的真实数据运行许可。

## 1. 研究定位与当前起点

研究主线是：**将 FairBias 应用于 NHIS 的费用相关医疗延迟识别，评价其预测能力、公平性及跨年度稳定性，并与代表性公平性方法比较。** FairBias 是本文主方法；Reweighing、LFR、Exponentiated Gradient 和 ThresholdOptimizer 是外部对照。LR、GBDT 是各方法使用的预测器，不能把“FairBias vs XGBoost”当作充分的公平性方法比较。

目前没有证据证明 FairBias 优于这些方法。实验应能得出优势、代价、适用条件或无显著差异，不得以出现优势为停止条件。应用文章可以主推 FairBias，但不能将应用与适配包装为首次提出 Tang 等人的算法。

已核实的工作区基点：分支 `research/nhis-fairbias`，HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`。存在尚未通过最终验收的 worker 修改，因此 HEAD 不代表当前完整源码。上一轮综合结论仍为 **REPAIR**，不能据已有测试通过数直接启动实证。

必读依据：

- [执行协议](/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md)。遵守受保护文件、数据访问、网络、独立验收与报告要求。
- [综合审查及初步研究规划](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_CONSOLIDATED_AUDIT_AND_APPLICATION_PLAN_20260913.md)。本方案将其 C01–C06、M01–M06、E01–E06 转为实施门槛。
- [第四轮独立审查](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_FOURTH_REVIEW_20260913.md)。用于区分已修复问题与剩余问题。
- [历史结果更正](/Users/lkc/Downloads/code_v_0_3/docs/reports/NHIS_D8_R4B_SUPERVISOR_ERRATUM.md)。历史结果保留，不用新实验覆盖。
- [现有特征注册表](/Users/lkc/Downloads/code_v_0_3/configs/nhis/features.json)及 [D8 runner](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py)。用于复用语义和辨认实现，不自动等于新协议。

本方案的数值网格、种子、主次指标是明确的**拟定默认值**，B0–B4 可以在不查看实证性能的条件下修订并记录原因；B5 启动前必须冻结。不能留有“到时看结果再定”的关键项。

## 2. 固定方法身份，避免主方法随结果变化

### 2.1 主方法与消融

默认主方法定为 **FairBias-BM application-v1**：已审查的 BM、语义特征几何、训练期拟合变换、统一分类器编码及统一评价。这个名称必须与历史 D8、作者原始代码、论文公式以及本项目工程扩展区分。

主版本从当前 `tang2024_paper_faithful` 配置所表达的路线出发：stress-elbow 选维、H=1、作者实现的交错幂序列、restart revisit、highest-d_phi 贪心、失败停止；保留实际存在的 NMI gate 并明确其来源。代码中有这个名称不构成“严格复现论文”的证明。B0 必须形成逐项来源表，B1/B3 必须核验真正生效的参数。

主几何不使用 WTFA_A；主预测训练不使用调查权重；主外部评价使用调查权重。调查加权几何是另一种统计目标，只能作为命名清楚的敏感性扩展。几何 epsilon 和 EO/DP 阈值属于不同空间，不得替代或比较数值大小。

FairBias-BM→AE 与 FairBias-Joint-BM/AE 是两个预声明的扩展/消融条件。不得看完 2024 后，从 BM、BM→AE、Joint 中挑最好的一个重新命名为“FairBias”。如以后决定主推 Joint，必须在 B5 前修改主方法登记，并说明其为本项目扩展。

内部 d_phi 主版本采用显式声明的多群体聚合，拟定为作者公式对应的最大组对差；必须先核实原文并贯通 evaluator。若来源不支持这一设置，B0 提交确切差异及版本修订，不能静默改用当前默认 mean-pair。外部预测公平性始终使用本方案第 6 节公式。几何的不可估计值、负误差、MDS 未收敛都不能被替换为“零偏差”。

### 2.2 必须完成的核心对照

| 方法 ID | 论文/角色 | 同一预测器下比较什么 | 支持的 NHIS 臂 | 推断时 A | 主输出 |
|---|---|---|---|---|---|
| UNMITIGATED | 无公平干预 | 基准预测性能、原始差距 | 001–004 | 不需要 | 事件概率 p、冻结阈值决策 |
| FAIRBIAS_BM | Tang et al., 2024 的应用适配 | 本文主方法 | 001–004，须七群体合成验收 | 预期不需要；验证冻结映射 | p、决策、独立几何诊断 |
| REWEIGHING | Kamiran & Calders, 2012 | 训练样本公平重加权 | 001–004，使用支持多组的实现 | 不需要 | p、决策 |
| LFR_RECONSTRUCTED | Zemel et al., 2013 | 官方 LFR 重构特征接同一预测器 | 001、003、004 | 所选官方 API 需要 | 下游预测器的 p、决策 |
| EG_DP | Agarwal et al., 2018 | DP 约束下的 reduction | 001–004，须七群体验收 | 标准推断不需要 | 随机决策概率 q |
| EG_EO | Agarwal et al., 2018 | EO 约束下的 reduction | 001–004，须七群体验收 | 标准推断不需要 | q |
| TO_EO | Hardt et al., 2016 | 冻结预测器后的 EO 阈值优化 | 001–004，须各组正负类支持 | 需要 | q；底层 p 单独标记 |

LFR 的 HISP 七群体格子为 `NOT_SUPPORTED`，不能偷偷把七组改成白人/非白人，也不能把 one-vs-rest 拼接称作原 LFR。主表保留这个格子及原因。多群体 fairness 方法与二群体版本必须分别验收。

LR 为主预测器，sklearn GBDT 为必须完成的预测器稳健性检查；先完成 LR 全链，再复用适配器运行 GBDT。首篇不强制增加 MLP/LAFTR、对抗学习或大批新模型。可在核心方案完成后，以新的预登记协议扩展，不能为追求“最新”牺牲可比性。

方法原始依据：[LFR 论文](https://proceedings.mlr.press/v28/zemel13.html)、[reduction 论文](https://proceedings.mlr.press/v80/agarwal18a.html)、[Hardt 等论文](https://arxiv.org/abs/1610.02413)、[Reweighing 官方实现文档](https://aif360.readthedocs.io/en/stable/modules/generated/aif360.sklearn.preprocessing.Reweighing.html)。库实现与原论文不等同，版本、偏离及接口约束另行登记。

## 3. 数据对象、纳入与时序角色

### 3.1 固定任务和四个臂

主结局：`MEDDL12M_A`，过去 12 个月因费用延迟医疗。`1→1`、`2→0`；拒答、不清楚及不适用按注册表处理为不可用结局。次结局 `MEDNG12M_A` 为扩展，首篇核心交付不依赖它。

这是一项重复横断面中的既往医疗可及性障碍识别研究。2022、2023、2024 不是同一批人的纵向随访；并发健康特征不能支持“预测某人下一年发病/延迟医疗”的表述。Y=1 指障碍存在，算法正决策只是研究中的识别结果，不代表已经实施临床分诊或资源分配。

| Arm | 保护轴 A | 预测特征 | 用途 |
|---|---|---|---|
| 001 | SEX_A，注册表规定的两组 | primary_core 21 个语义变量 | 主要公平性轴 |
| 002 | HISPALLP_A，七组 | 相同 21 个语义变量 | 多群体适配、少数组估计 |
| 003 | DISAB3_A，两组 | 相同 21 个语义变量 | 包含残疾构成特征 |
| 004 | DISAB3_A，两组 | 去除六个残疾构成特征后 15 个 | 同人群的特征敏感性臂 |

Arm 004 **不是只分析残疾人群**。003/004 的纳入规则、记录集合、时间分区必须完全一致，仅特征集合不同；存在缺失 X 的人保留并按训练期规则处理。003 与 004 的差异是同人群的配对特征实验。

保护变量、Y、WTFA_A、PSTRAT、PPSU、年份、来源 ID 均不进入常规预测特征。残疾构成特征在 003 中是明确保留的代理信息，不能宣称该臂已“去除一切敏感信息”。不同保护轴可因 A 缺失产生不同合格人群，不得直接按跨臂总体准确率高低解释公平性优劣。

### 3.2 新 benchmark 的唯一时间划分

| 分区 | 年份与方式 | 允许用途 | 不允许用途 |
|---|---|---|---|
| F：模型拟合集 | 2022 年，约 80% PSU | 语义预处理拟合、编码/缩放、FairBias BM、RW/LFR/EG、预测器训练、train epsilon reference | 使用 C/S/T 统计量补全变换 |
| C：校准及内部开发集 | 2022 年，约 20% PSU | 所有概率模型的全局阈值拟合；TO 的公平后处理拟合；AE 内部效用与停止选择 | 拟合基础 imputer/scaler；用于宣称独立评价 |
| S：配置选择集 | 完整 2023 年合格样本 | 按固定准则选择超参数/候选；开发结果 | 拟合 TO；逐候选 AE 内层反馈；拟合新特征编码 |
| T：冻结后评价集 | 完整 2024 年合格样本 | 仅冻结预测与统一评价 | 调阈值、调 epsilon、换种子/特征/主方法 |

F/C 划分先于分析臂筛选和任何拟合，在 2022 设计主表上按 `(PSTRAT, PPSU)` 整组分配。拟定 seed=20260913，每个 PSU 以 0.2 概率进入 C，否则进入 F；对有序稳定键使用明确版本的随机算法。不得为获得好性能或有利类别比例重抽种子。这是约 80/20 的 PSU 级随机分配，不保证每层两侧均有 PSU，也不保证记录数恰为 80/20。

保留划分概率、主表 hash、PSU/记录映射 hash；其受保护明细只存本地忽略目录。跨臂复用这份主划分，不能四个臂分别随机切分。F/C 的描述性人口量若报告，使用相应纳入概率修正；这不改变主训练使用无调查权重的约定。不为 F/C 半样本出具完整年度调查推断。

各方法必需的训练/校准 `A×Y` 单元为空时，按能力契约返回 `NOT_ESTIMABLE`。不按性能改分区，不把缺失单元补成 0%。若问题使核心设计不可执行，在 B5a 仅看支持度的检查阶段出具方案修订，经审查后整体重新冻结；不边看模型性能边补救。

这是对之前“2023 年再拆校准/选择”的初步建议的明确细化：本版将校准留在 2022 内部，使 2023 完整保留为选择集。C 是开发训练数据的一部分，不是测试集。选定后默认**不在 F+C+S 上重新拟合**，避免已选阈值、变换与最终模型失配；如另做重拟合，须新协议规定重新校准与再评价边界。

2024 已在 D4/D5/D8 的研究过程中被观察，故新流程只能确保“从本次冻结起不再用它选模型”。文章如实称回顾性跨年评价，不声称第一次打开的独立锁箱。真正新的未观察年份/数据源是后续验证，不自动并入当前计划。

### 3.3 数据适配契约

一个分区对象至少包含 `record_key, X_semantic, y, A, survey_weight, stratum, psu, year, role, eligibility_mask, schema_hash, source_hash`。所有分量同一索引，不能分别 reset_index 后凭行号拼接。

记录键使用有来源定义的稳定调查记录 ID 和年份；年份必须有限整数，缺失/小数拒绝；数字型 ID 的等价及带前导零字符串政策由来源 schema 定义，不能全局强行转 float/string。原数据没有经核验的稳定 ID 时不得假称完成记录隔离。重复、部分重叠、同号不同来源都要有测试。

Y 或当前保护轴 A 不可用者按固定规则排除，X 缺失保留。报告每年、每臂各步骤的聚合人数、事件数和加权量；本方案不填造真实纳入人数。类别/特殊缺失编码逐年对照注册表与官方资料。2022 raw=27651、2023 raw=29522、2024 raw=32629 是历史原始记录数，不能当作本次各臂最终样本量。

## 4. 表示与适配器：每种方法如何拿到正确的数据

### 4.1 三层表示，禁止一个矩阵贯穿所有接口

1. 语义层：按官方代码清理；3 个数值变量为 agep_a、pcnt18uptc、pcntlt18tc；数值缺失用 F 中位数；全缺失变量采用预声明删除规则并记录。类别缺失使用显式 `MISSING` 水平。
2. FairBias 几何层：保留每个语义特征及类型，不将几十个 one-hot 列直接作为原论文的特征实体。观察支持、类别合并与变换输出按已审查的语义执行；数值变成分箱类别时必须更新输出类型 schema。
3. 预测层：主应用版本对全部非数值变量 one-hot，包括有序类别；数值按 F 拟合 MinMaxScaler。每个 FairBias 候选在其**变换后的 F** 上拟合自己的预测编码器并冻结；baseline/RW/EG/TO 的同一输入可共享等价的 F 编码。类别重命名不能通过数字顺序改变结果。

2023/2024 的新类别不能扩充词表。固定未知类别策略为显式 `UNKNOWN` 保留位；缺失与未知不同；记录其出现率。数值超出 F 范围允许按冻结线性规则外推，不按 S/T 最值重新缩放；方法如要求截断，必须提前命名为独立配置，不能暗中 clip。LFR 的连续重构输出直接接下游预测器，不能再 one-hot，也不能叫作 k 维潜在表示。

### 4.2 特别容易出错的适配点

| 方法 | fit / transform / predict 必须遵守的契约 |
|---|---|
| FairBias | F 学得的同一映射作用于 C/S/T；终局模型与 candidate evaluator 使用同一编码器规范、预测器参数、正类和权重政策。S/T 的 y 或 A 不得决定映射；测试几何诊断放在冻结预测之后，不反馈训练。 |
| RW | F 内估计 `P(A)P(Y)/P(A,Y)`；主实验从单位基础权重产生公平训练权重，使用多群体实现。AIF360 sklearn `fit_transform` 返回 `(X, weights)`，不能按普通 sklearn transformer 处理。若 A 位于 DataFrame index，临时 wrapper 必须保留 record_key 映射。零单元及非有限权重拒绝，不静默平滑。 |
| LFR | A 单独作为 protected metadata，防止 BinaryLabelDataset 同时把 A 混入 features。官方 `transform` 会重写 labels/scores：下游训练和全部评价使用原始 y 的独立副本，绝不使用变换产生的标签。推断对象可用不含真值的占位标签，必须验证输出与这些占位值无关。二群体、行顺序、连续重构与原 y 对齐逐项验收。 |
| EG | oracle 必须真实接收 reduction 的 `sample_weight`。权重路由不能传给 scaler 或被 pipeline 丢弃。使用官方 Moment，DP/EO 分开；`eps` 与 Moment `difference_bound` 不是同一个参数。q 为混合器对基分类器**硬决策**的加权和，不是基分类器风险概率的加权平均，也不能再阈值化 q 改写方法。 |
| TO | 基础预测器仅在 F 拟合；后处理仅在 C 拟合；显式 `prefit=True`、`predict_method='predict_proba'`，正类 wrapper 必须保证列含义。EO 不支持的 `tol` 不放入网格。预测需要 A，未知组/缺失组不能退回任意阈值。记录底层 p 与最终 q 的不同身份。 |

LFR 的 API 行为来自 [AIF360 官方源码](https://raw.githubusercontent.com/Trusted-AI/AIF360/main/aif360/algorithms/preprocessing/lfr.py)；其所示实现不使用调查权重。EG 的权重接口与参数语义见 [官方 API](https://fairlearn.org/main/api_reference/generated/fairlearn.reductions.ExponentiatedGradient.html)；TO 的拟合/预测参数见 [官方 API](https://fairlearn.org/main/api_reference/generated/fairlearn.postprocessing.ThresholdOptimizer.html)。这些在线页面可能是开发版本；实施必须锁定实际安装发行版并重新核对，不能将 main 分支当作可复现依赖锁。

EG 主对照保留原生 ErrorRate 目标；S 上的公共 BA 选优不把它改写为 BA 优化算法。Moment 对“群体相对总体的差”的约束与外部 max−min gap 也不一定数值相同，adapter 必须导出实际约束定义。LFR 的 privileged/unprivileged 代码只是 API 分组设置，按注册表记录实际组名及映射，不据此做本研究人群优劣判断。

LFR 自带 scores 可另作 `LFR_NATIVE_SCORE` 补充，明确与“重构特征+共同预测器”不同，不能只挑两者较优的结果。选择后处理和需要 A 的表示方法时，同时报告部署时是否必须获取 A；不将这个差异隐去。

## 5. 训练、调参与计算预算

### 5.1 固定主实验训练制度

主实验各方法遵循其标准的无调查权重学习目标，公共外部评价与 S 上的配置选择使用 WTFA_A。分别保存 `survey_weight`、`fairness_reweighting_factor`、`reduction_cost_weight`，禁止混名；不能因为内部有 sample_weight 就声称学习了调查加权公平约束。

训练权重敏感性至少包括 unmitigated 与 FairBias 的下游预测器使用均值归一化 WTFA_A，其他条件相同。RW 若加入此敏感性，必须从调查加权的组别-结局分布推导因子并标记新版本，不能随手与无权重 RW 因子相乘。EG/TO/LFR 未实现并验证总体加权目标的格子为 `NOT_SUPPORTED`；不要伪装成完整的加权方法榜单。

LR 网格：`C=[0.01,0.1,1,10]`，solver=lbfgs、max_iter=1000、class_weight=None。GBDT 网格：`n_estimators=[100,200] × max_depth=[2,3]`，learning_rate=0.05、subsample=1.0，其余参数在 B3 从实例完整导出。每种方法与相同预测器网格组合，在同一 S 上按第 6 节选优；不只给 FairBias 调参，也不靠 class_weight='balanced' 暗改对照。

| 方法 | 公平性配置网格（拟定） | 每个 arm/backbone 的配置上界 |
|---|---|---|
| Unmitigated | 无公平性超参数 | 4 |
| FairBias-BM | F 计算的有效 epsilon reference × `[0.25,0.5,0.75,1,1.5,2,4,8]` | 32 |
| RW | 原方法标准重加权，不人工添加可挑结果的“强度” | 4 |
| LFR | `k=[5,10] × Az=[0.1,1,10,50]`；Ax=0.01、Ay=1 | 32 |
| EG-DP / EG-EO | 相应 Moment 的 `difference_bound=[0.005,0.01,0.02,0.05,0.075,0.1,0.15,0.2]`；EG eps=0.01、max_iter=50 | 每种 32 |
| TO-EO | objective=balanced_accuracy_score、grid_size=1000、flip=False、prefit=True、tol=None | 4；每个基础预测器一套后处理 |

原论文本身只有一个强度设置的方法不必人为凑够 32 个配置。“候选预算相同”不等于计算量相同；必须报告每种方法真实拟合次数、内层 oracle 调用、几何计算次数、用时、内存与失败情况。

FairBias 的 reference 只从本版 F 的语义表示计算，并保存算法、聚合、seed 与数值。禁止盲用旧 D6 的 0.0005/0.002/0.005，禁止计算失败后回退常数。若 reference 非正/不可估计，标记并停该条件；若 F 本来满足几何约束，则保留显式 no-op 状态。

F/C 的角色在全部条件中相同：p 型方法在 C 上选全局阈值，最大化无权重 balanced accuracy；候选为相邻不同 p 的中点及边界，平局选最接近 0.5，再选较大阈值。保留 t=0.5 的描述性敏感性。TO 用 C 的原生无权重目标拟合，EG 使用原生混合决策。上述选择都不接触 S/T。风险概率校准曲线属于评价；本版不额外拟合 Platt/isotonic 作为隐含干预。

### 5.2 种子、预算与失败

拟定算法种子 `[0,7,19,37,73]`，分区种子固定不随算法种子变化。适用于 MDS/LFR/存在随机训练的模型；完全确定的重复不制造五份“独立实验”。确定性拟合的 EG 通过 q 评价无需抽样预测五次；支持随机训练的 oracle 才增加训练种子。禁止选最佳种子。LFR 修改全局 RNG 的实现要做进程隔离或安全恢复，避免污染后续方法。

拟定计算上限：FairBias BM 最多 50 次提交变换、每次 fit 总几何调用最多 20000；AE 最多 10 次外层提交、每次 fit 候选效用评估最多 500；LFR maxiter=maxfun=5000；EG max_iter=50。所有条件每个 fit 壁钟上限 30 分钟，单 worker 峰值内存目标不超过 4 GiB；默认一次运行一个重型拟合。B4 可根据纯合成和复杂度分析统一修订，在 B5 前冻结。

预算限制是应用实现的计算边界，不是论文收敛保证。BM/AE 超预算返回 `BUDGET_EXHAUSTED`，不得假称满足 epsilon。主选择只使用成功完成且契约有效的配置；被截断的最后状态仅作标记清楚的探索性工件，不能以“完成”身份参赛。真实数据若全配置超时，要报告不可执行，不得悄悄缩窄幂序列或降低对照预算。

核心每个预测器为 7 方法 × 4 臂 − 1 个 LFR 不支持格 = **27 个方法-臂条件**；这是条件数，不是模型拟合数。两预测器 54 个条件；两种 AE 消融再加 8 个条件/预测器。矩阵枚举器必须展开配置与有效训练种子，并去重可证明相同的 BM 表示/底层预测器缓存；预算和缓存均绑定完整数据、配置及版本身份。

主候选若某个预登记训练种子出现算法失败，该配置不能只删掉失败种子后按剩余种子选优；按不完整配置处理，保留失败率和已有有效结果。纯基础设施中断允许同配置同种子恢复，必须记录 attempt_id；收敛警告、退化单类预测、算法预算停止分别登记，不能统称“成功”。

## 6. 公共评价、选优与“比别人好”的判据

### 6.1 同时看预测和公平，区分输出含义

风险输出契约 `event_probability`：p 表示 Y=1 的模型事件概率；验证 classes_、正类映射、二维形状、有限值、[0,1] 与行和。`ranking_score` 只可用于排序指标；`decision_probability` q 表示随机政策作出阳性决策的概率；`hard_decision` 为 0/1。禁止缺少 p 时用硬标签冒充概率。

全部方法共同的主要评价是 **调查加权 balanced accuracy 与调查加权 EO gap 的二维权衡**。风险概率模型同时报告 weighted average precision（AP，明确不是梯形积分 PR-AUC）、AUROC、Brier、校准曲线，以及各组 AP/AUROC。EG/TO 的 q 不填入“事件风险 Brier/校准”列；它们可报告 q 的排序诊断，但不能与 p 列混名。TO 的底层 p 标为 base-score，不能宣称公平后处理提升了底层 AUROC。

令 q 对确定性政策等于冻结的 0/1 决策，对随机方法为其精确决策概率。以 WTFA_A 为 w，在相同合格域计算：

- TP=sum(w*y*q)，FP=sum(w*(1-y)*q)，FN=sum(w*y*(1-q))，TN=sum(w*(1-y)*(1-q))。
- TPR=TP/(TP+FN)，FPR=FP/(FP+TN)，BA=(TPR+1-FPR)/2。
- selection_rate=sum(w*q)/sum(w)，DP_gap=max_g SR_g−min_g SR_g。
- EO_gap=max(max_g TPR_g−min_g TPR_g, max_g FPR_g−min_g FPR_g)。同时单列 TPR gap / FNR gap。
- PPV、F1 等随机政策指标注明由期望混淆计数的比值计算，不声称是所有随机实现该指标的数学期望。

q 优先从锁定实现的有限混合精确计算。若只能依赖私有 PMF 接口，必须有薄包装、锁版本与官方随机预测的大样本一致性测试；接口漂移应失败，不能无声改为 threshold(q,0.5)。年度评价不靠一次随机掷币决定谁更公平。

每个组报告未加权 n、事件数、非事件数、权重和、Kish 有效样本量诊断、TPR/FPR、阳性覆盖率、置信区间与可估计状态。单类、缺失组、零分母返回 `NOT_ESTIMABLE`，不能补 0；七组主 gap 要求预登记七组全部可估计。可另给 observed-groups 描述性结果，但不能替换七组主指标。小样本精度标记与 CDC 正式发布标准分开，不伪造一个“20 例就是有效”的官方阈值。

### 6.2 选择规则在看 2024 前固定

每个方法、臂、预测器在 S 上形成候选集。对随机训练，先按预登记种子聚合指标，再选配置；不挑种子。拟定主操作点的外部公平预算为 `EO_gap≤0.10`，敏感性点为 0.05 和 0.20。这是研究比较刻度，**没有临床安全阈值或公平保证的含义**，也不是 EG 的 eps 或 FairBias 的 epsilon。

每个操作点的规则：在有效且满足该 S 上预算的配置中最大化加权 BA；平局先取较小 EO gap，再取较低计算复杂度，最后固定 config_id 排序。种子聚合使用各种子指标均值；同时保留每个种子是否满足预算，不把均值满足写成每个种子均满足。

没有配置可行时，返回 `NO_FEASIBLE_CONFIGURATION`；同时保留最小 EO gap、再最大 BA 的失败边界配置供描述。若 gap 不可估计，不进入此排序。主表不删除未达预算的方法，也不把“在 S 可行”写成“在 T 保证可行”。

无公平干预方法同样经历这套共同操作点选择，并明确这是“无公平训练 + 公共配置选择”。另保留一个强预测参考：在 S 上按 weighted AP 最大选定的 unmitigated 配置（平局 BA、复杂度、config_id），不施加公平预算。这个参考必须单列，防止共同选择规则削弱普通预测基线后再宣称 FairBias 胜出。

所有三个操作点和预测参考在 B5b 一次冻结。T 上只评价这些已登记产物，不扫描剩余超参数、不根据 T 画出一个重新挑选的最优包络。公平—效用完整候选前沿来自 S；T 图展示 S 已冻结的点及区间。

### 6.3 结论尺度

对每个对照报告配对的 `ΔBA=BA_FairBias−BA_other`、`ΔEO=EO_FairBias−EO_other`，以及兼容风险模型的 ΔAP/ΔAUROC。预测更好与公平更好是不同结论。点估计 ΔBA≥0、ΔEO≤0 是经验支配；区间跨越零时不能升级为确定优势。公平改善但 BA 下降属于权衡。

不设置未经健康场景论证的“BA 降 2% 可接受”并宣称临床非劣效。若需要正式非劣效结论，必须在 B5 前另行确定有依据的界值及检验方案。本版默认效应量、配对区间与探索性比较。

主表保留四臂完整结果，但 inferential 主对比家族拟定为主操作点、LR 的 SEX 与 DISAB-include 两轴上 FairBias-BM 对所有五个外部方法的 ΔBA/ΔEO，即预登记 20 个差值。HISP 七组、Arm 004、其他操作点和预测器是预声明扩展。主家族拟用基于复制方差的 t 区间及 Bonferroni 调整（每个双侧区间 alpha=0.05/20）；不会因为某方法失败而缩小分母。未调整的 95% 配对区间同时给出并注明描述性质。非光滑 gap 的覆盖表现必须通过 B2/B4 合成验证，不能把 Bonferroni 当作修复错误方差的工具；未通过则保留描述性结果，禁止正式“显著优于全部方法”的表述。应用文章无需强行制造一个总排名。

## 7. 调查设计推断必须独立成为一个模块

年度主估计使用 WTFA_A；分层 PSTRAT，PSU 键为 `(PSTRAT,PPSU)`。PSU 编号不是全表唯一。先保留完整年度设计主表，再以分析域指标形成结局/亚组估计；不能先删除其他组和零贡献 PSU 再构建设计。003/004 的比较还要共享同一设计复制权重。

主 T 的配对不确定性以固定模型为条件，使用同一组分层 PSU 复制权重评价所有方法和种子。建议实现分层 rescaled PSU bootstrap，B=2000、seed=20260914：在每层 m_h≥2 时有放回抽取 m_h−1 个 PSU，记录次数 k_hi，复制因子 m_h*k_hi/(m_h−1)。这是本项目选择的调查近似法，不声称为 CDC 提供的官方复制权重；忽略 FPC 的理由、适用设计与方差归一化要写清。

B2 必须用非敏感合成调查设计与独立成熟实现核对率、差值和方差；可参考 [R survey 的复制设计说明](https://r-survey.r-forge.r-project.org/survey/html/as.svrepdesign.html)。正态近似/百分位区间、多群体 max/min 非光滑统计量的覆盖表现及同时推断方法需要合成覆盖实验验证，不能仅证明自写函数和自写期望一致。

拟定默认方差为有效复制统计量围绕其复制均值的样本方差，使用 B_eff−1 分母；t 自由度为完整年度设计的 PSU 数减分层数，必须大于零。复制统计量每次重算各组率、max/min 和配对差值，不能固定原样本的“最差组”代替。该归一化需与采用的复制权重构造一并验证，不能借用其他 bootstrap/BRR 的 scale/rscales。正式区间的构造及适用条件在 B4 冻结，不根据 T 上哪种区间显著再选择。

若原始完整设计有 singleton PSU 层，默认 `DESIGN_NOT_ESTIMABLE`，不无声设方差为零。B2 预声明并验证需要的处理；B5a 发现真实设计不适用时，仅报告聚合诊断并请求审查设计修订。bootstrap 某次复制缺少组内正/负类时记录无效复制；不得补零或不断重抽到“好看”。有效复制不足预登记比例 95% 时不出该指标的正式区间，并保留比例。

需要至少两种不确定性报告：调查抽样不确定性（固定各个模型后，对种子平均指标及配对差值使用同一复制权重）和训练种子离散性（SD/range）。前者不包括完整开发选择/重新训练的不确定性，后者也不是人群抽样 SE；不将 5 seeds 或 2000 replicates 当成独立病人做 t-test。

AUROC/AP 计算采用排序或分块算法，禁止构造全样本正负例 O(N²) 矩阵。权重须有限且非负，零权重不扩大有效支持，公共比例缩放不改变率；极端有限权重的求和溢出必须拒绝或稳定正规化。报告加权与无权重结果，不能只报告对 FairBias 有利的那一种。

NHIS 年度设计与可比性依据需在 B0/B2 核对 [2024 官方调查说明](https://ftp.cdc.gov/pub/health_statistics/nchs/dataset_documentation/NHIS/2024/srvydesc-508.pdf) 及已有本地官方文档。不自动合并年度权重，也不武断给 PSU 加 year 前缀后假设年度独立；本版主要推断为 2024 单年，跨年点估计变化作为描述，不进行未经验证的跨年独立样本检验。

## 8. AE、Arm 004 与稳健性分析的边界

必须完成：LR/GBDT 两预测器；加权主评价/无权重敏感性；003/004 同人群特征消融；t=0.5 与 C 校准阈值；所有随机训练的固定种子；FairBias BM→AE 与 Joint 的独立版本；至少一组基线/FairBias 调查加权预测器训练敏感性。

AE 的选择效用拟定与应用主目标一致，使用 C 上的 BA；这是应用扩展，不能继续声称完全遵循原来 ACC 目标。对 p 型 AE 候选，先按共同 C 阈值规则取决策，再计算效用；C 的复用属于内部开发，不能把其增益当作验证发现。S 只选完整预定义运行的配置，不能反向驱动下一次 AE 搜索。

AE 主扩展设 strict geometric feasibility，slack=0，max outer=10，候选族/顺序从实际 runner 导出并冻结；记录 rejected/eligible/selected/committed、每次 BM/AE 的几何与预测变化及停止原因。旧 slack=0.02 或其他松弛单列敏感性，不与严格结果混合。

Arm 004 路径研究优先在固定行、F/C/S、预测器、epsilon 来源、幂网格、候选顺序及预算下，**仅改变几何维数策略**（stress-elbow vs fixed=2）。如果同时更改幂网格，则为第二个因素，用 2×2 因子比较并增加独立命名，不能把所有差异都归因于“几何”。epsilon 固定与重新计算是两个不同敏感性问题，分开报告。

保存可核验的路径证据：首个分歧步骤、所选特征、幂/类别合并、候选接受理由、当前 d_phi、有效维数、stress 曲线、效用与状态 hash。只看到最终分数不同不能证明路径机制；若没有这些证据，叙述限定为“与路径敏感性一致”。路径消融先看 S，预登记少量条件后再做 T 的冻结展示，不能重开测试集找解释。

## 9. 新架构和标准产物

建议新建 `src/nhis_fairbias/benchmark/` 下的 data_contracts、preprocessing、adapters、prediction_contracts、metrics、survey_inference、selection、runner、artifacts 模块，以及 `configs/nhis/benchmark_v1/`、`tests/benchmark/`、`scripts/` 中独立入口。它们是拟建位置，不是声称当前已经存在。避免继续把外部方法塞入历史 D8 runner。

每个 adapter 必须声明：输入/输出类型、supports_multigroup、requires_A_fit、requires_A_predict、requires_y_transform、supports_survey_training_weight、stochastic_fit、stochastic_decision、native_objective、resolved_params。任何能力缺失明确拒绝，不落回 unmitigated。

结果状态至少分为：`VALID`、`NOT_SUPPORTED`、`NOT_ESTIMABLE`、`NO_FEASIBLE_CONFIGURATION`、`BUDGET_EXHAUSTED`、`NUMERICAL_FAILURE`、`CONTRACT_VIOLATION`。运行状态与选优状态分字段；真正发生的异常保留 traceback 摘要和审计事件，不伪装成 candidate_exhausted。JSON bool/null 保留原类型，禁止把 NaN/Infinity 作为有效数值写出。

每次唯一 run_id 下至少有：环境锁/包 hash、源码及 dirty 文件 hash、数据/分区/schema hash、完整配置与实际参数、枚举矩阵、训练账本、候选表、选择 manifest、冻结预测模型 hash、逐方法运行状态、聚合指标、复制权重种子/算法及区间、图表数字出处。个体预测/ID/设计明细仅留本地忽略目录，公开包只含允许的聚合与代码，不打印微观行、不覆盖历史结果。

当前系统 Python 3.13.2 的只读依赖检查发现 sklearn 1.7.1、numpy 2.2.6、pandas 2.3.1、scipy 1.16.1、xgboost 3.4.1；该解释器未安装 fairlearn/aif360。这不是对整台机器所有环境的结论。B0 先做只读环境清单，B3 在专用环境锁定兼容发行版；禁止改根 requirements.txt 或全局 Python。确需安装时，Gate 明示 `pypi.org`、`files.pythonhosted.org` 等所需 HTTPS 域与原子下载/散列规则；未获该阶段权限不能自动联网安装。

## 10. 完整 Gate 路线与验收

本次给 Gemini 的首次指令只激活 **B0**。以下是完整路线，不是一次性授权全部阶段。协议第 3 节明确要求每个 Gate 经 worker 报告和独立 Codex 审查；因此分段停止源于仓库既有治理要求，而不是需要用户逐个选择普通编码细节。

| Gate | 工作与交付 | 数据/运行边界 | 放行条件 |
|---|---|---|---|
| B0 现状与设计落表 | preflight、来源映射、问题闭环表、四臂 schema、方法能力表、依赖计划、实验矩阵、后续 Gate 规格 | 只读代码/配置/既有聚合报告；仅新增计划文档；不运行测试/训练、不读微观数据、不安装 | 关键决定无歧义；每个核心条件有数据接口与指标；范围与当前修改一致 |
| B1 正确性修复 | 关闭 C01–C06/E01；同步应用需要的防御契约，保留已通过修复 | 只在获准 src/tests/configs/docs 范围内修改；持久 guard 的纯合成验证 | 问题族正反例、真实对象路径、旧锚点、状态/配置身份均通过独立检查 |
| B2 数据与公共评价 | F/C/S/T 合约、语义/预测编码、统一指标、设计推断和选择模块 | 合成数据；不读取实际 NHIS/COMPAS/MEPS | 手算权重例、独立调查基准、空组/单类/极端权重与隔离测试通过 |
| B3 外部方法适配 | FairBias、RW、LFR、EG-DP/EO、TO-EO；LR/GBDT；版本锁 | 合成验收；只在 Gate 所列域允许依赖下载；不接真实数据 | 每种方法与官方对象行为一致、七群体/权重/输出能力如实、预测不读 y |
| B4 全链合成彩排与协议冻结 | 模拟三年和多 PSU 的整矩阵；资源估算；实际配置、矩阵、推断/选择规则签名 | 全链仍为合成；无 T 微观数据 | 测试标签置换不改变模型/选择；序列化重载一致；模拟失败不丢行；主次对比全冻结 |
| B5a 真实数据支持度审计 | 2022/2023 纳入流、PSU 分区、缺失/编码/组别支持；schema 与来源散列 | 单独授权后仅指定 NHIS 文件；不看模型性能；2024 尚不读取 | 不更换人群、不悄悄改变方法；全部不可估计与设计例外处理已审查 |
| B5b 真实开发与选择 | 仅 F/C 拟合、S 选优；完整候选与预算账本；冻结方法/阈值/seed/模型 | 2022/2023；禁止 2024 读取；独立审查后冻结 | 所有核心条件成功或有完整失败证据，不能挑有利结果；预测包签名可重载 |
| B6 冻结跨年评价 | 一次冻结批次的 T 预测、指标、同复制权重配对区间；之后做必要公开聚合 | 单独授权 2024 预测/评价；禁止任何 fit/选优调用 | 报告可由冻结产物重算；若评价程序缺陷需修复，记录事故并对全部方法同规则重算 |
| B7 论文与复现包 | 主表/图/补充结果、方法段、局限性、软件/协议说明 | 仅获准聚合产物；不自动投稿/发布 | 每个数字可追溯；失败/不可行保留；文章身份、时间与调查推断表述准确 |

B5b 可按“LR 核心 27 条件 → GBDT 核心 27 条件 → AE 与权重/路径敏感性”分批提交进度与恢复，不要求一次长任务跑完；批次执行顺序与完整待运行矩阵提前登记。各批次使用同一个已冻结协议，不能因前一批 FairBias 输赢而取消不利方法或增加有利条件。所有计划进入正文/补充的 T 条件在 B6 前一起冻结。

B1–B4 不是只补最后一个报错。至少覆盖七类有限验收矩阵：身份与分区；完整配置/缓存；语义编码与类别重命名；概率/决策/多组指标；权重/设计域估计；外部方法真实接口；持久化/异常/重启。沿用综合审查第 10 节的正反例，并新增 LFR 标签保存、TO prefit、EG 权重路由及 q 语义、同 PSU 不跨 F/C、T 标签置换独立性。

每个修复必须关联 issue_id、现有证据、根因、代码位置、正例/反例、guard 命令与精确结果。不能用测试数量代替覆盖。测试集合明确区分纯合成与会读取 COMPAS/真实数据的旧测试；不无差别执行全库 pytest 后再声称 synthetic-only。

## 11. 论文交付与完成标准

拟定标题方向：**FairBias for identifying cost-related delayed medical care: predictive performance and fairness across NHIS survey years**。调查推断通过后可以明确 survey-weighted；没有验证前不预写“显著提升健康公平”。

正文至少包含：纳入流程与样本表；主方法及对照、输出/部署条件表；LR 的预测与公平双维主表及配对差值；逐群体识别率/误报率/覆盖率；S 上候选前沿与 T 上冻结点；概率模型校准；跨预测器和 003/004 消融。补充包含完整失败矩阵、参数/种子/计算量、AE/几何路径、无权重与加权训练敏感性。

方法段依次说明：任务与数据年份、纳入及缺失、A 与设计变量、F/C/S/T、防泄漏、算法身份与偏离、预测器/阈值/选优、调查推断、输出类型、多重比较及公开复现边界。参考 [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378) 整理报告完整性，但不能将清单勾选称为研究有效性认证。

完成标准是：本方案必需矩阵全部得到可审计结果或可解释的不可执行状态；FairBias 的相对优势/代价可由相同评价规则复算；每个主张有对应数字和适用范围；一位不了解本次对话的研究者能按环境锁、协议和入口重现合成链与获授权的数据分析。不是“FairBias 必须第一”，也不是“跑出一张表即可”。

本次仅新增规划文档，没有修复源码、安装方法、运行真实实验或授权 Git 提交。实际 worker 指令见 [Gemini 分阶段执行 Prompt](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_BENCHMARK_STAGED_PROMPT_20260913.md)。
