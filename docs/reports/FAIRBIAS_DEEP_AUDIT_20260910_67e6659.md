# FairBias 医疗公平性代码库独立深度审查

审查日期：2026-09-10。对象：`icarus3344/fairbias-health-equity`，`research/nhis-fairbias`，完整提交 `67e6659fa65249a8842e34af5d8969629efe4bca`。本地 HEAD 与 GitHub 指定提交已核对。本报告评价该提交的代码、契约和可支持的科学结论，不重新批准或撤销历史 Gate。

**结论：未确认 P0；发现 7 类 P1 风险和 6 类 P2 问题。** 核心流程已有实质性的隔离、状态审计和非有限值防护，但“严格作者实现”“复杂调查推断”“C3 始终可行且路径鲁棒”等表述超出了当前证据。若以这些主张投稿或对外发布结论，应先处理相应 P1；这不等于已有聚合数值全部无效。

## 1. 审查边界与证据

- 遵循 `docs/AI_EXECUTION_PROTOCOL.md`；审查前工作区干净。未修改算法、旧报告、配置、冻结结果或基线文件，未提交或推送。
- 阅读核心几何、BM/AE、预处理、调查统计、D6/D8 调度与指定报告；用现有聚合 JSON 核对 R4 结果，用 Git 历史与源文件散列绑定 R3 实现。
- 对照 Tang et al. 主文及作者公开代码。论文主文是出版社提供的全文；本轮未完整取得补充材料，因此不宣称完成所有补充算法和参数的逐项认证。作者仓库文件快照 blob SHA 为 `de75af978a6344b83807a596594cea8173cdcf8a`，不是声称它就是出版当日版本。
- 对照 CDC 2024 NHIS Survey Description，并提取、渲染检查印刷页 45、68，查阅跨年合并说明。来源、下载散列、合成探针输出与测试日志保存在本报告同名 `_evidence` 目录。
- 没有查看受访者记录、重跑真实 NHIS 模型或使用真实数据进行候选搜索。独立 D6 测试触发了生产前置检查，包含对本地数据文件的散列读取，随后在科学代码边界检查处失败；因此本轮不声称“零真实文件字节访问”。合成探针的模型拟合只使用人工构造数据，授权绕过探针使用 mock adapter。

证据强度区分：**实际结果已核实**、**合成反例已复现**、**静态条件风险**、**待实验验证的解释**。下文不会用其中一种替代另一种。

## 2. 首先纠正审查指引的科学前提

[审查指引](/Users/lkc/Downloads/code_v_0_3/docs/GPT6_CODEBASE_AUDIT_AND_IMPROVEMENT_GUIDE.md:63)有数处事实错误，会把审查带向不存在的实现：

| 指引表述 | 当前代码及应使用的表述 |
|---|---|
| C2 是 `Z = XW` 投影，联合几何损失和交叉熵梯度 | MDS 的节点是“特征及原点”；随后对原始特征做贪心幂变换或类别合并。分类器没有使用一个由 MDS 学出的 `W` 投影受访者数据。 |
| `d_phi` 是 demographic parity | 它是表示空间中的几何偏差浓度代理。DP、EO/EOpp 是基于预测结果的其他指标，数值及约束不可互换。 |
| HISP 为 `HISP_A` 的 Hispanic 二分类 | 实际是 `HISPALLP_A` 七类别；见 [adapter.py:18](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:18)。 |
| Arm 4 是“残疾限制人群”子域 | 它从预测变量中排除六个残疾构成变量；Arm 3/4 使用相同的结局与保护属性有效样本条件。见 [adapter.py:172](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:172)、[adapter.py:216](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:216)。 |
| D6 是 longitudinal，2022 train/val → 2023/24 test | 当前是重复横断面：2022 训练、2023 验证选候选、2024 时序评估，不是追踪同一批个体。 |
| C3 收敛且从不违反 ε；C4 搜索 Pareto frontier | C3 有五步预算和非零约束松弛；C4 是有预算的交错贪心启发式，没有全局最优或 Pareto 前沿保证。 |

论文描述的基本流程确实是特征间距离、MDS 与贪心数据变换，且分类器与偏差评估相分离。[Tang et al., 2024 主文](https://www.researchgate.net/publication/379141194_Metric-Independent_Mitigation_of_Unpredefined_Bias_in_Machine_Classification)。当前 RMS、level-H 排除上下文、原点平移等核心公式在已检查的二分类情形下未发现新的明显公式错误；这不覆盖下面的多类别扩展和全部数值实现。

## 3. P0：严重漏洞

**本轮未确认 P0。** 未发现当前规范 D6 流程将 2024 数据拟合进训练变换的直接证据，也未发现这次审查破坏冻结原始结果的情况。P0 未检出不是不存在缺陷的证明；尤其不能把合成测试通过解释为科学有效性认证。

## 4. P1：方法学及重要正确性风险

### P1-01：多类别几何与作者实现不等价，直接涉及 HISP 臂

**位置：** [bias_metric.py:338](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:338)、[bias_metric.py:376](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:376)；作者 `MachineClassifer_BiasMitigation.py` 约 417–463 行。[作者源码](https://github.com/zftang/MachineClassifer_BiasMitigation_beta/blob/main/MachineClassifer_BiasMitigation.py#L417)。

当前实现对每个上下文计算 `mean_pair(abs(w_pair(S+a) - w_pair(S+b)))`；作者代码计算 `abs(max_pair(w_pair(S+a)) - max_pair(w_pair(S+b)))`，然后才跨上下文平均。最大值、差与平均不可交换。代码注释已把当前规则称为扩展，但模式名和报告中的 fidelity 主张没有充分体现这个差异。

**合成反例已复现：** 三个保护群体 `o=[0,1,2]`，数值特征 `a=[0,0,1]`、`b=[0,1,0]`，`H=0`。三个群体对的 `(g_a,g_b)` 分别是 `(0,1),(1,0),(1,1)`。本库 `D(a,b)=2/3`，按作者这段代码为 `0`。这是进入 MDS 之前的确定性差异，不能归因于 MDS 随机性。

**后果：** 七类别 HISP 的阈值、特征排序和变换路径可能不同；目前不能把该臂称为作者代码等价复现。两种聚合没有简单的大小关系，也不能据此宣称作者规则一定更公平。二类别时只有一个群体对，不触发这一差异。

**建议：** 显式区分 `author_max_pair` 与 `mean_pair_extension`；新增二类别退化等价、三类别手算、多类别作者对照测试。保留历史 mean-pair 结果和原阈值，先修订方法名称；改变聚合后的真实实验与阈值必须另立版本，不能覆盖旧结果。

### P1-02：AE 执行的是 ε + 0.02，不能宣称严格保持冻结 ε

**位置：** [enhancement.py:308](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:308)、[enhancement.py:319](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:319)、[d8_enhancement_runner.py:813](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:813)、[d8_enhancement_runner.py:957](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:957)。

候选接受上限为 `epsilon_threshold + max_fairness_degradation`，D8 两条 AE 路径都传入 `0.02`。对于冻结 ε 为 `0.0005 / 0.002 / 0.005` 的臂，上限分别是 `0.0205 / 0.022 / 0.025`，即原阈值的 **41 / 11 / 5 倍**。这是绝对松弛，不是浮点容差。

以下值直接从 R4 `condition_metrics.json` 的 **训练集**记录重算可行性；不是测试集阈值判定：

| 臂 | 冻结 ε | C3 最终 max dφ | C3 ≤ ε | C4 最终 max dφ | C4 ≤ ε |
|---|---:|---:|:---:|---:|:---:|
| SEX | 0.0005 | 0.001511309 | 否 | 0.000954864 | 否 |
| HISP | 0.002 | 0.004691630 | 否 | 0.002673050 | 否 |
| DISAB-full | 0.005 | 0.005249348 | 否 | 0.004967521 | 是 |
| DISAB-exclude | 0.005 | 0.003096720 | 是 | 0.004753174 | 是 |

**实际影响已确认。** 最终 `fairness_feasible` 仍按严格 ε 判断，见 [runner:899](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:899)，因此问题不是该布尔字段谎报，而是算法接受规则与“C3 永不违反 ε”的叙述不一致。

**建议：** 明确声明搜索松弛约束和最终可行约束。若新版本要实现严格可行 AE，候选检查使用零松弛，并保留最近的严格可行 incumbent 以便预算耗尽或失败时返回；若保留松弛搜索，则必须展示不可行结果，不得据 AUROC 提升称为受约束改善。不得静默把历史 `0.02` 改成 `0` 后继续沿用旧结论。

### P1-03：主结果不是复杂调查加权估计，也没有设计校正推断

**位置：** [d8_enhancement_runner.py:560](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:560)丢弃 adapter 返回的权重和设计元数据；[runner:285](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:285)拟合无权重 LR；[runner:383](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:383)计算无权重指标；[evaluation.py:46](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/evaluation.py:46)接口不接收权重。

[survey.py:174](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/survey.py:174)已经诚实标注 `classifier_weighted=False`、`evaluation_weighted=False`、`complex_survey_inference=False`、PSTRAT/PPSU 未用于方差。D8 primary 又显式选择未加权的 paper geometry。因此，本轮确认的是**对外方法描述与实际 estimand 不一致**，不是说所有无权重机器学习实验都无效。

WTFA_A 应用于目标总体的点估计，PSTRAT/PPSU 则用于反映复杂抽样设计的方差估计。不能把无权重受访者样本 AUROC、组间差距直接解释成美国成年总体的公平性及显著性。[CDC 2024 Survey Description，pp.45–49、68](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHIS/2024/srvydesc-508.pdf)。

**具体判断与建议：**

- 加权几何的群体均值/频率比率表达式本身未发现简单的漏除权重和错误；但 [mitigation.py:153](/Users/lkc/Downloads/code_v_0_3/src/fairbias/mitigation.py:153) 的类别合并频率仍未加权，这是已声明的 R1 设计，不能把“局部散度加权”等同于全流程 survey optimization。
- 首先声明研究对象：样本层算法基准，还是总体健康公平性。分类器是否加权是另一个需预声明的设计选择；总体评价加权不要求强制所有分类器都加权。
- 增加独立的总体评价层：加权 AUROC/AP/Brier、校准及混淆矩阵率；对同一受访者上 C1–C4 差值采用保留分层和 PSU 的成对设计方差方法。不要直接采用普通 iid bootstrap 或未加权 DeLong；不能在缺少合法复权构造时直接声称 BRR 已实现。
- 子域分析应保留设计信息并用 domain indicator 计算。正确权重下，删去域外记录不必然改变比例点估计，但可能破坏方差；“过滤必然造成点估计偏差”同样不严谨。Arm 4 本身并不是子域过滤。
- 结局或保护属性缺失者在 [adapter.py:216](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:216) 被排除；普通调查权重不自动解决这些 item nonresponse。应报告可观测亚总体、有效样本与加权缺失率。
- 若合并年份，按 CDC 对所选年份的说明处理年度平均权重及设计标识；不要想当然给所有 PSTRAT/PPSU 加年份前缀。2023–2024 合并说明要求保留相应设计标识并按合并年数调整权重。

### P1-04：名义类别编号被当成连续数值，AE 效用可能反映编码任意性

**位置：** [preprocessing.py:440](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:440)、[enhancement_contracts.py:378](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:378)、[enhancement_contracts.py:420](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:420)、[runner:285](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:285)。

预处理保留官方整数类别和 `-1/-2` 特殊值；之后全部转 float、MinMax 缩放并进入 LR。对 REGION 等名义变量，这赋予了没有科学依据的顺序和间距。类别合并后选择哪个数字作为代表，也可能改变线性模型的可表达关系。

**合成反例已复现，使用真实 LR 拟合：** 训练集和独立索引验证集各 90 行，类别 `[0,1,2]` 重复 30 次，标签 `[0,1,0]`。类别成员关系完全不变，只将编号 `{0:0,1:2,2:1}` 重命名，候选效用评估 AUROC 从 `0.5` 变成 `1.0`。这不证明已有 NHIS 增益全部由编码产生，但证明效用评价不具备名义标签置换不变性。

**建议：** 几何层保留类别语义；分类器层对名义变量使用仅在训练集拟合的 one-hot 及冻结的未知类别策略。序数变量须逐一说明编码，缺失哨兵另行处理。将“标签任意置换”和“相同合并分区、不同代表代码”作为新版本效用稳定性测试。更换编码后需要新的研究版本，不能称为原结果的无影响工程修复。

### P1-05：不可估计的公平性仍可能被转成零，绕过有限值契约

**位置与机制：**

- [bias_metric.py:533](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:533)：保护属性不足两个有效群体，直接返回全部零。
- [bias_metric.py:216](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:216)、230、263、275 行：某群体特征全缺失或没有有效权重，散度设零。
- [bias_metric.py:293](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:293)：缺失散度和未列入 num/cat 清单的预测变量补零。
- [enhancement.py:230](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:230)、255 行：检查列存在及已声明活动特征的几何结果，不足以识别上述人为生成的有效零。
- [d8_enhancement_runner.py:191](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:191)、203–207 行：不足两个群体或可比较 TPR 时，DP/EOpp 返回零。

**合成反例已复现：** 单群体 `O=[0,0,0,0]` 的候选被严格 ε=0.0005、零松弛 guard 接受，dφ=0；一群体数值全部 NaN 得到零散度；未声明的预测变量得到零。`y=[1,1,0,0]`、分组 `[0,0,1,1]` 时第二群体没有阳性，EOpp 仍返回 `0`。

**后果：** 上层 `isfinite` 防御无法检测下层已抹去的“不可估计”。这是公共 API 的条件性正确性问题，本轮没有证据说 R4 主样本实际上缺少全部比较群体。

**建议：** 返回结构化 `NOT_ESTIMABLE`、分母及群体覆盖；有效类别集合由研究契约规定。全部预测变量必须恰好属于一个语义类型。候选检查遇到不可估计状态拒绝；对外统计保留 null 与原因，不能包装成“公平为零差距”。

### P1-06：规范时序路径基本隔离，但通用接口没有严格兑现零泄漏/授权契约

**已确认的正面实现：** [adapter.py:118](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:118)只在 2022 拟合默认预处理；D6 的 [878 行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d6_temporal_runner.py:878)验证预处理角色；[1084 行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d6_temporal_runner.py:1084)训练缩放与模型；[1111 行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d6_temporal_runner.py:1111)先冻结状态。测试释放模块的 [1485 行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d6_temporal_test_release.py:1485)、[1694 行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d6_temporal_test_release.py:1694)设置全部臂冻结屏障后才评分。未发现规范代码把测试标签送入 AE 选择。对测试集重新计算描述性 dφ 不等于重新拟合预测变换。

**三个缺口已用合成/mock 检查：**

1. [preprocessing.py:246](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:246)、253 行只在元数据列存在时检查年份/角色；缺少两列仍拟合并记录 `fit_year=2022, fit_role=development_train`。这是把默认假设写成已验证来源。Adapter 对已拟合预处理器还会跳过 fit，见 [118 行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:118)。
2. [enhancement_contracts.py:84](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:84)仅检查每个分区内部 X/y 与列/索引对齐；完全相同样本同时充当 fit 和 selection 能通过。内容散列证明快照一致，不证明来源独立。需要稳定且带年份/来源命名空间的记录 ID；不能简单因不同年份的 RangeIndex 数字重复就报泄漏。
3. [d8_enhancement_runner.py:436](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:436)的 `allow_real_data` 可修改；模式/授权组合只在构造时检查。[adapter 属性:525](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:525)不会重查。创建 exploratory、allow=False 的 runner，随后设 allow=True，mock 证明它会调用真实 adapter 工厂分支。此次没有打开真实数据。这是防误用的授权检查缺口，不是把 Python 对象当作能抵挡任意恶意代码的安全沙箱。

**建议：** 生产预处理必须具备来源字段；合成入口显式区分。使用不可变运行配置，在数据访问点再次核对模式、授权、规范 release 与预处理来源；对注入 adapter 也声明 synthetic/production 能力。验证 namespaced source IDs 的唯一性和分区不交叠。

**历史暴露限制：** [D6 disclosure:193](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d6_temporal_runner.py:193)已明确 2024 曾进入 D4/D5；其后又有 D8 探索/主运行。可以称“D6 特定流程先冻结后评估”，不能称“整个研究从未见过的独立最终测试集”。本轮没有证据证明研究者确实用 2024 调参，亦不能仅凭 API 隔离排除研究层面的适应性选择。

### P1-07：Erratum 收回因果归因是正确的，但“几何鲁棒/敏感”的强度仍过高

**位置：** [NHIS_D8_R4B_SUPERVISOR_ERRATUM.md:59](/Users/lkc/Downloads/code_v_0_3/docs/reports/NHIS_D8_R4B_SUPERVISOR_ERRATUM.md:59)、67–72 行。

**能够支持：** Arm 4 C4 在 R3 中的约 `0.7520` AUROC 没有在 R4 复现。R4 C4 为 `0.6865516853`，相对 C2 `0.6669734185` 提高 `0.0195782668`，但仍低于 C1 `0.7352557123`，差 `-0.0487040271`；R4 终点训练 dφ 满足冻结阈值。这些是聚合工件可验证的描述，不是显著性检验结果。

**仍不能支持：**

- 单独把变化归因于 MDS 几何。R3 历史源码绑定至 `cab6b63:src/nhis_fairbias/d8_enhancement_runner.py`，其中 426–436 行选择 engineering 与固定二维，455 行计算自适应阈值，715 行传入 engineering 幂网格。R4 除 stress-elbow 外还改变了 BM 幂序列与阈值来源。实际两个阈值是否恰好相等未在此重新估计，不能使用错误的静态 manifest 数值代替执行证据。
- 把 C3 相同终态上升为一般“路径鲁棒”。四臂都接受五步后 `budget_exhausted`，并非证明候选耗尽或算法收敛，前三臂还不可行。相同 canonical C2 起点、相同效用计算与候选族、较宽 guard，可以解释某些运行终态相同，但尚无逐候选证据确认这是唯一机制。
- 仅凭某终态可行/不可行判定 AUROC 增益由不公平导致，也不能把一次终态对比当作 Pareto 前沿或“显著”变化。

**建议替代表述：** “在已执行的两套配置中，C3 的预算终点相同；C4 对配置与搜索路径表现出敏感性。Arm 4 的探索性增益未在冻结阈值的主配置中复现。现有比较同时改变多个算法因素，尚不能单独归因于几何，也不构成 C3 一般路径鲁棒性的证明。”

若未来另获真实实验授权，最小机制设计为几何 × BM 幂序列的因子比较，固定阈值、数据、编码、随机种子及预算；逐候选记录排名、效用、dφ、约束余量与拒绝原因，定位首个分歧。先用合成数据建立识别逻辑，研究设计不得继续根据 2024 表现选择。

## 5. P2：代码正确性、测试与可复现性

### P2-01：候选缓存不绑定验证分区和完整配置

[enhancement_state.py:180](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_state.py:180)按 `parent_state_hash + candidate_sig` 记已评估；[enhancement.py:575](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:575)还把幂四舍五入到四位小数。[enhancement_contracts.py:204](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:204)的配置指纹没有包括 H、MDS 参数、实际参考 ε 等全部会改变接受结果的上下文。

**已复现：** 同一 AE 对象在相同变换状态上换用不同 selection_y；第一次模拟候选效用 0.5，低于 baseline 0.6；第二次该候选应为 0.9，但被 tracker 跳过，没有调用候选效用函数。改变 MDS 固定维数的配置指纹也相同。此缓存探针使用 mock 效用以隔离调度逻辑。D8 当前为各臂/条件新建对象且分区固定，未证明此问题影响既有 R4。

建议：缓存键包含 fit/selection/protected/weight 内容指纹、完整已解析配置、参考 ε 与无损候选参数，或保证每次运行上下文变化自动重置所有缓存。分别检查排名缓存、候选访问缓存与 cycle history，不能只修其中一处。

### P2-02：survey 边界处理存在三个可复现错误

- [survey.py:109](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/survey.py:109)、132 行：有效分母存在但某事件从未发生时，空分子 `sum(min_count=1)` 为 NaN，比例应为 0。`codes=[2,2], weights=[1,1], code=1` 已复现。仅无有效分母时应返回 null。
- [survey.py:57](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/survey.py:57)：PASS 只检查能否数值化；`[inf,0,-1,nan]` 且 PSTRAT/PPSU 均缺失仍 PASS。[positive_weight_mask:91](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/survey.py:91)还把正无穷视为有效。应区分字段可解析与可用于调查分析的校验结果。
- [bias_metric.py:208](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:208)：允许零权重的低层模式仍让零权重记录决定 min/max。`x=[0,1], o=[0,1], w=[1,1]` 的散度 1，加入 `x=100,o=0,w=0` 后变为 0.01。零权重类别还可能影响类别并集和 K。应在定义加权经验分布前去除零权重记录，或禁止该模式并收紧文档；未证明严格正权重的 NHIS 主运行受此影响。

### P2-03：60 个 D8 契约通过，但测试隔离和覆盖不足

[test_nhis_d8_synthetic_contracts.py:20](/Users/lkc/Downloads/code_v_0_3/tests/test_nhis_d8_synthetic_contracts.py:20)在导入时安装进程级 `sys.addaudithook`，永久拦截 data 路径与所有 `.parquet`；包括临时合成 parquet。它保护了单独的 D8 测试，却会污染同进程其他套件。部分 D6 合成测试又未经完整注入便走生产 provenance 检查。

本轮测试环境为 Python 3.13.2、scikit-learn 1.7.1、NumPy 2.2.6、pandas 2.3.1、pytest 8.4.1，与 R4 环境工件核对。

| 实际运行范围 | 结果 | 解释 |
|---|---:|---|
| D8 契约 + AE 核心 + AE 契约，独立进程 | **103 passed**，56.96 s | 其中 D8 文件确有 60 个 test 方法；不是 103 个 D8 专属契约。 |
| weighted geometry，独立进程 | **21 passed**，3.36 s | 不能覆盖上述零权重经验分布反例。 |
| 上述四文件 + 两个 D6 文件，同一进程 | **223 passed / 75 failed**，246.78 s | 75 个失败均位于 D6：42 + 33。包括全局数据拦截及生产授权/提交绑定问题；不把它们描述成 75 个算法数值错误。 |
| D6 temporal 独立进程，首错停止 | **2 passed / 1 failed**，1.62 s | test_03 仍在生产前置检查失败：冻结 base `f63f589…` 与当前 `67e6659…` 的科学代码已变化。 |

另尝试过 `.venv`，其无 pytest，且 sklearn 为 1.9.0；一次 unittest 运行被主动中止，不计入通过证据。没有因此安装或修改环境。

**建议：** 纯合成、生产前置检查和历史 release 验证分进程执行；保留真实数据禁读约束。生产 D6 拒绝漂移的行为应保留，不能为了让测试通过而放宽科学边界。测试应完整注入临时受控仓库及 release，历史重放使用指定版本。新增 P1/P2 所列反例，尤其测试“看起来合法的零”而非仅 mock NaN；断言真正的 evaluator/adapter 路径被调用。

### P2-04：stress-elbow 不是收敛或防止过度平滑的保证

[bias_metric.py:398](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:398)逐维扫描使用单初始化，而最终 MDS 使用四初始化；归一化 stress 曲线的第一个相邻差小于 0.01 即选中。曲线可能有局部波动，早期平台不等于可靠肘点；代码未提供完整 stress、迭代次数或候选排名余量证据。

作者公开脚本约 500–526 行画 stress 曲线，之后固定二维；这与主文的选维描述又不完全相同。因此应分清“论文思想实现”“作者脚本复现”和“项目自动肘点启发式”。本轮未完整核对论文补充选维算法，不把当前精确扫描参数认定为论文唯一规定。

建议记录各维 stress、初始化、收敛信息、所选维数及 dφ 对初始化的敏感性；验证最终坐标有限。改善选维稳定性属于新算法配置，保留原语义基线。另外 [config.py:268](/Users/lkc/Downloads/code_v_0_3/src/fairbias/config.py:268)允许 paper 模式保留非空 fixed dimension；若外部只看模式名会误判，manifest 应记录具体几何参数。R4 主路径显式传 None，未命中此例外。

### P2-05：事后勘误正确，但执行配置与来源仍有可复发缺口

- [run_nhis_d8_r4_substantive.py:165](/Users/lkc/Downloads/code_v_0_3/scripts/run_nhis_d8_r4_substantive.py:165)仍硬编码了并未执行的 AE 家族、`[2,3]` 网格、minimum gain 0.001、slack 0、C3 十步。实际 runner 为 slack 0.02、C3 五步及其他已解析参数。R4B 的事后 manifest 能解释历史错误，但执行脚本仍可能再次输出错误元数据。
- 通用 CLI [run_nhis_enhancement_study.py:228](/Users/lkc/Downloads/code_v_0_3/scripts/run_nhis_enhancement_study.py:228)保存逐候选 audit events；R4 脚本没有保存 `runner.audit_events`。已有 joint events 记录已提交路径，不足以还原全部被拒候选。历史 candidate count 应维持 `null/NOT_RECORDED`，不能用 0 代替，更不能事后编造。
- legacy JSON 的 26,751 字节与 SHA-256 `944df70bd75a8c1a1bb0c6758540936123582fcab08bc64d5806583f09e4da11` 本轮吻合，也与 R1 的预修复记录相符。Erratum 将其排除 R4 主输出的修正有依据。三个主工件的独立散列见 evidence，不能把事后派生表和主输出混称。
- 14 个继承根文件仍与保护 tag 相同；`.gitignore` 却早在历史提交 `b595e59` 已增加五行，见 HEAD 与 tag 的 diff。因此“`.gitignore` 始终完全 immutable”的当前治理描述与历史事实不一致。本轮没有更改它；不据此推断当时是否得到授权。

建议由实际构造出的对象生成唯一执行配置，启动前检查声明值与对象一致；运行中追加审计日志并在异常退出时保全。新增研究依赖锁文件固定 R4 环境，避免修改冻结的继承 requirements。治理例外应另记历史说明，不能回写冻结证据掩盖差异。

### P2-06：交错停止语义与候选内存保留值得改进

[d8_enhancement_runner.py:979](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:979)调用单步 BM；BM 的失败/非收敛状态没有完整进入外层停止原因。[1104 行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:1104)先把双空选择解释成 candidate_exhausted，随后才检查严格可行，首次可行又立即终止。应明确区分 BM 数值失败、策略停止、循环、候选耗尽、预算终止和“已达到可行但未优化完效用”；不要宣称 C4 效用逐步单调或全局 Pareto 最优。

[enhancement.py:711](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:711)、984 行在有效候选列表中保留完整变换 DataFrame，候选数扩大时有显著内存成本。建议仅保留候选状态/评分与当前最佳矩阵，缓存可复用的训练统计，先测量几何与拟合耗时。并行化必须固定候选选择次序与随机状态；没有基准数据前不承诺“快一个数量级”。

## 6. P3：可形成研究贡献的拓展

1. **把效用恢复与可行性恢复分开。** 预声明并列展示相对 C1、相对 C2 的 AUROC/AP 变化与训练/验证/测试 dφ。先判是否可估计，再判是否达到约束，再讨论效用。增加“最近可行解”和完整可行轨迹，避免只展示最佳终点。
2. **从几何代理走向健康决策评价。** `MEDDL12M_A` 等指标反映医疗可及性障碍，不应直接等同疾病状态或临床获益。说明标签的报告误差、群体差异和目标使用场景；公平代理降低不能自动推出改善健康权益或因果效应。
3. **防止稀有事件下的退化公平。** 极低预测阳性率可以使 DP/EOpp 看似很小。报告各组事件数、有效样本量、TPR/FPR、PPV、校准、PR 曲线和预测阳性率；决策阈值仅在验证阶段按预声明目标选择。不要为了改善测试公平差距反复调阈值。
4. **将多群体目标显式化。** mean-pair、max-pair、population-weighted disparity 是不同社会目标，不应由实现方便隐式决定。可将最差群体和交叉群体评价作为独立结果，但小样本交叉组必须设置可估计性与不确定性门槛。
5. **给路径敏感性一个可检验问题。** 分解“首个候选分歧 → 可达状态变化 → 效用与约束差异”，先用可手算合成数据说明，再做冻结因素的实验。分开估计对已训练模型的测试抽样不确定性与训练/选择过程的不稳定性；它们不能由同一条置信区间替代。

## 7. 建议执行次序与本轮交付

优先顺序为：补充方法与报告勘误（P1-01/02/03/07）→ 增加独立反例并修复 fail-closed 契约（P1-05/06）→ 解决编码与完整缓存上下文（P1-04、P2-01）→ 修复调查边界与测试隔离 → 在新版本中预声明设计推断和机制实验。

本轮交付仅新增这份审查报告和证据目录。未执行建议的真实数据实验，未修改旧结果，也没有把历史 Gate 的通过当成新审查问题的否定证据。
