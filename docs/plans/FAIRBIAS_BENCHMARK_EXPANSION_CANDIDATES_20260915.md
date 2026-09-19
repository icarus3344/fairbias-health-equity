# FairBias 应用论文：收尾边界与近期 benchmark 候选规划

日期：2026-09-15。状态：研究与适配候选草案，等待 R11 交付后确定下一步执行规格。

本文件回应用户关于返工、耗时及近五年对照方法的请求。它不改变已封存的 R10 监督证据、R11 提示或旧实验登记，不要求 Gemini 在 R11 增加任何任务。本次查阅论文和公开作者实现，未安装依赖、下载模型权重、训练、读取微数据或启动 benchmark。公开网页核验不等于本地实现验收。

## 1. 离收尾还有多少

当前软件关卡只剩两类 P1：R7-04 的完整交付内容校验，R7-01 的诊断独占保存与最终清单覆盖；报告事实错误归入这两项修复。具体证据见 [R10 监督报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R10_SUPERVISOR_REVIEW_20260915.md:32)。

目标是 **R11 一次集中修复，再一次独立验收**。这是工作目标，不是“保证再一轮就没有漏洞”。仅凭历史轮数不能估算整个项目完成百分比。R1A 通过后，仍有已经登记的 benchmark 接线和调查推断工作；不能把“当前只剩两项”说成“整篇论文只差两项”。

R11 已有清楚出口：

1. 可信 R10 正常材料通过；既有负例及来源、日志、七类 trace 的错误分别被真实总校验路径拒绝，失败不产生成功包。
2. 每次诊断保留独立来源、命令、退出、stdout/stderr；最终清单覆盖全部材料；已封存包不可被生成器或诊断器覆写。
3. 报告从实际事实生成；120 项冻结输入及已有模型契约不变化。

指定 runner 首次编译捕获的 PARTIAL 边界已经允许保留，不为本轮新建通用安全平台。文档排版、可选性能优化和未授权新研究想法不应成为 R11 新阻塞项。若出现影响结果真实性或数据隔离的新证据，明确其触发路径和必要性，不隐瞒；若仍因同一根因失败，先合并校验架构再修，不继续无限追加单个字符串反例。

## 2. 为什么反复修复，为什么时间变长

历轮记录显示了不同层次的问题，而非每一轮都发现一个全新的 FairBias 算法错误：

| 阶段 | 当时仍存在的主要问题 | 当前认识 |
|---|---|---|
| R5–R8 | 内部模型/scaler 拟合身份未真正被测到；执行证据检查不完整；某些调试越过规定入口 | 属于实质契约与可复核性缺口，不是纯文案返工 |
| R9 | 指定内部拟合反例关闭；生成器与清单身份不符、日志只看后缀、失败材料仍可获成功包 | 模型层有进展，问题集中到证据交付 |
| R10 | 源码清单一致、真实日志路径已修复、旧 13 种生成器行为有效 | 剩余两类交付问题；无需重复模型测试 |

依据：[R8](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R8_SUPERVISOR_REVIEW_20260915.md:3)、[R9](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R9_SUPERVISOR_REVIEW_20260915.md:3)、[R10](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R10_SUPERVISOR_REVIEW_20260915.md:3)。这些是已完成审查的结论，没有因此重新执行旧实验。

返工的共性是局部修补：检查“存在/数量足够”不等于检查内容有效；在 runner 修好的规则未被交付生成器完整复用；手写报告和复用诊断目录产生新的身份问题。监督流程也需要承担改进责任：应更早统一完整验收契约和结束条件，避免每轮只给一批反例、让 worker 继续针对反例打补丁。

用户报告的单轮 30–40 分钟可以理解，但不能据此认定研究计算更难或产出更多。R10 的 pytest 为 3.60 秒，完整 run 为 6.11 秒，说明这次耗时主体并非该测试计算。阅读累积历史、编写交付工具、生成证据、排错和模型服务延迟都是可能因素；缺少全过程分阶段计时，不能给各项杜撰百分比，也没有验证 Gemini 是否变慢。

之后采用一份短的当前状态表、一个校验入口、一次修复后必要验证；已关闭项仅在相关源码变化时做相应回归。研究计算与报告整理分别记录时间，失败尝试留下原因。R11 不重跑 88 项 suite，且不新增 phase。新方法的训练可能确实更贵，但要依据后续合成规模检查和真实获准运行的计时判断。

## 3. 论文定位与近期方法筛选范围

FairBias-BM application-v1 继续作为主方法，BM→AE 和 Joint 作为预登记扩展。论文研究 NHIS 费用相关医疗延迟的识别、公平性及跨年稳定性；它是重复横断面的回顾性研究，不能表述成追踪同一人预测下一年的临床结局。主方法身份不随结果重新挑选。

近期方法检索以 2021 年以来、截至 2026-09-15 为候选窗口；下列近期候选实际发表于 2022–2026，处于近五年范围。它们是有来源支持的适配候选，不声称穷尽所有论文或已证明适配 NHIS。传统对照仍保留 RW、LFR、EG-DP、EG-EO、TO-EO，其中 EG 的两个目标属于同一方法家族的两种配置，不把工具库的新发表年份当成算法的新年份。

## 4. 近期公平性候选与适配顺序

“优先”指优先核验可执行性，是否纳入最终主表由来源、适配和资源规则决定，不能由 2024 表现决定。

| 候选 | 年份与原始来源 | 官方实现 | 计划中的角色与配对 | 必须解决的适配点 |
|---|---|---|---|---|
| FairGBM | ICLR 2023：[论文](https://openreview.net/pdf?id=x-mXzBgCX3a) | [feedzai/fairgbm](https://github.com/feedzai/fairgbm/blob/main-fairgbm/README.md) | 优先；树模型内处理；普通 LightGBM / FairBias＋LightGBM / FairGBM 三方比较 | 固定实际 LightGBM 系列与容量；确认 FNR/FPR 约束和全局目标；单独核验多组、训练权重、平台编译；不把 FNR 单约束写成完整 equalized odds |
| OxonFair | NeurIPS 2024：[论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ab6022d3d669b5baafa24c91d7c407a6-Abstract-Conference.html) | [oxfordinternetinstitute/oxonfair](https://github.com/oxfordinternetinstitute/oxonfair) | 优先；相同冻结基础模型后的决策调整，先 LR/树模型 | 后处理在 C 拟合；真实 A 和预测 A 版本分别命名；辅助群体模型只在 F 训练；外部验证约束是否可行，不将库返回的最近解当成满足约束 |
| fairret | ICLR 2024：[论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/63943ee9fe347f3d95892cf87d9a42e6-Abstract-Conference.html) | [aida-ugent/fairret](https://github.com/aida-ugent/fairret) | 优先；可微公平正则；同一个 MLP 的无干预 / FairBias＋MLP / fairret＋MLP | 固定具体 statistic、loss 和正则系数，不能只登记库名；A 的独热编码与缺失类别；批内 A×Y 缺失支持；验证所选公平定义对应 DP、TPR 或完整 EO |
| FRAPPÉ | ICML 2024：[论文](https://proceedings.mlr.press/v235/tifrea24a.html) | [论文链接的实现目录](https://github.com/google-research/google-research/tree/master/postproc_fairness) | 优先候选；将正则化内处理转为后处理框架；同一基础模型上与 TO/OxonFair 比较 | 具体实例、模块输入 X/score、损失和训练样本量必须固定；拟合用 C；不能用 S/T 标签。论文 PDF 确认代码链接，但本次实现目录抓取失败，接口尚待核验 |
| FairSHAP（Zhu/Bian/You） | AISTATS 2026：[论文](https://proceedings.mlr.press/v300/zhu26b.html) | [ZhuMuMu0216/FairSHAP](https://github.com/ZhuMuMu0216/FairSHAP) | 第二批重点；与 FairBias、RW、LFR 的预处理路线比较，优先二群体臂 | 作者流程含匹配、归因、训练数据增强；不能强塞为同一种 transform。背景数据/匹配/归因只用 F，生成行不进入 C/S/T 或调查人口估计；七群体实现和计算规模另验。其文献 EO 指 equality of opportunity，不能直接等同本项目 equalized odds |
| FairProjection | NeurIPS 2022：[论文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fd5013ea0c3f96931dec77174eaf9d80-Abstract-Conference.html) | [HsiangHsu/Fair-Projection](https://github.com/HsiangHsu/Fair-Projection) | 第二批；概率分布投影、多组能力候选，适合考察 HISP 七群体 | 明确输入概率/所需辅助模型、ADMM 停止与约束；C 拟合后处理；确认输出为预测分布还是决策分布，不自动解释为校准风险 |
| FairBiNN | NeurIPS 2024：[论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bef7a072148e646fcb62641cc351e599-Abstract-Conference.html) | [yazdanimehdi/FairBiNN](https://github.com/yazdanimehdi/FairBiNN) | 第二批；双层优化，与匹配神经网络及 FairBias＋相同网络比较 | 不包装成 LR/GBDT 插件；核验实际公平损失、上下层步数、数据分区和停止条件；仓库 README 尚有占位安装地址，不能认为开箱可用 |
| LinearPost | 2024 预印本：[论文](https://arxiv.org/abs/2405.04025) | [uiuctml/fair-classification](https://github.com/uiuctml/fair-classification) | 储备；多组后处理及部署时 A 是否可用的对照 | 可能需 P(Y|X)、P(A|X) 或 P(A,Y|X)；辅助预测器用 F，LP 用 C；固定求解器。当前核验来源列为预印本，不能冒称已录用会议 |

LinearPost 当前代码与同仓库 `icml.23` 标签下的 [ICML 2023 方法](https://proceedings.mlr.press/v202/xian23b.html)要区分；2023 版本主要针对 DP，不能用 2024 代码却登记为原 2023 算法。所有仓库最终锁定 commit/tag、依赖和算法实例后才形成参赛方法。

建议首篇先做到“4 个传统方法家族＋4 个近期方法家族＋FairBias 主方法及消融”的完整比较，再决定是否增加 FairSHAP/FairProjection/FairBiNN/LinearPost。FairSHAP 值得优先做适配核验，但不以其年份新就直接跳过验证。不是每个候选都要进入主要推断家族。

暂不纳入纯图公平、推荐排序、无监督异常检测和 LLM 文本公平方法；它们的输入、目标或数据生成假设与当前有标签 NHIS 表格分类不同。不能仅因标题含 healthcare/fairness 就列作同任务竞争者。

## 5. 预测模型规划

| 预测器 | 角色 | 适配要求 |
|---|---|---|
| Logistic Regression | 保持现有主要预测器，便于解释和同骨干比较 | F 拟合编码器/缩放；统一正类；全部方法相同容量和阈值政策 |
| 现有 sklearn GBDT | 保留原登记的必要稳健性对照 | 不用新 LightGBM 替换后还叫同一旧条件 |
| LightGBM | 为 FairGBM 提供匹配基础模型，也作为强树模型预测对照 | 新增命名条件；匹配版本系列、树数/叶数/学习率与预算；具体版本待锁 |
| 小型 MLP | fairret/FairBiNN 的匹配无干预对照 | 相同层宽、优化器及有效数据；早停使用 F 内部预划开发子集或预登记 C，绝不使用 S/T 逐轮反馈 |
| TabM | ICLR 2025；近期神经表格预测对照 | 固定模型容量、集成数和预处理；先普通模型，再 FairBias＋TabM。后者需新适配验收，不能假称已经支持 |
| TabICL v1 | ICML 2025；近期表格基础模型候选 | 优先纸面版本、确切 checkpoint 与集成配置；训练上下文只能来自 F；资源适配后纳入独立预测扩展 |
| TabPFN v2（可选） | Nature 2025；基础模型敏感性对照 | 原论文重点范围为 ≤10,000 样本、≤500 特征；实际 F 的合格规模和资源未验证，不能承诺全量适配，更不能为了容纳它悄悄缩小所有主实验 |

来源：[TabM 官方论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html)、[TabM 官方实现](https://github.com/yandex-research/tabm)、[TabICL 论文](https://proceedings.mlr.press/v267/qu25d.html)、[TabICL 官方实现](https://github.com/soda-inria/tabicl)、[TabPFN v2 论文](https://www.nature.com/articles/s41586-024-08328-6)、[TabPFN 官方实现](https://github.com/PriorLabs/TabPFN)。

版本漂移已经是可见风险：当前 TabICL 仓库默认 v2，仍列出论文 v1 checkpoint；当前 TabPFN 默认已不是 Nature 2025 的 v2。只写 `pip install` 和默认构造器会混淆论文身份。因此候选表不等于“装最新版就跑”。较新版本可以另列探索条件，但不能暗中替换。

基础模型仅使用获准本地运行，不将 NHIS 记录送到远程预测 API。 checkpoint 来源、训练数据说明、许可证、初次下载和离线运行边界后续单独核验。现代框架的支持不等于当前 Mac 和 4 GiB 预算足够；无法满足时列资源不支持，不能用降低对照配置、隐性下采样或单个超时结果证明 FairBias 更优。

## 6. 比较矩阵如何控制规模并保留可比性

采用三类互补比较，避免全笛卡尔积：

1. **相同骨干上的方法效果**：LR/原 GBDT 上比较 FairBias 与可兼容 RW/LFR/EG/TO、OxonFair、FRAPPÉ；每种方法都有对应无干预参考。原有主对比仍完整报告。
2. **原生架构匹配**：FairGBM 对普通 LightGBM 与 FairBias＋LightGBM；fairret/FairBiNN 对匹配 MLP 与 FairBias＋MLP。不同架构的系统比较可以展示，但不能把架构差异全部归因于公平方法。
3. **近期预测能力**：在同样年度、合格人群和语义变量下比较无干预 LR/树/TabM/TabICL；资源及接口允许后增加 FairBias＋TabM/TabICL。每种组合是独立适配条件，不强迫 EG 使用不支持其权重 oracle 的基础模型。

同骨干条件优先共享可证明相同的输入表示；跨骨干允许官方推荐且仅从 F 学得的模型专属预处理，报告为完整预测系统比较。不能为追求“同一矩阵”破坏原生模型接口，也不能让不同系统拿到不同语义信息。特征构造和预测器同时变化时，分别登记因素。

在 R11 后先按理论/API/合成行为/资源准入决定入围，保持失败理由完整；不能先看 2024 结果再挑有利竞争者。近期方法未准入时保留候选表与原因，不冒称已比较。

## 7. 每个适配器的必填卡片

新增方法先交一张有源码依据的能力卡，再接 benchmark：

- 方法的确切论文、版本/commit、许可证、算法实例、目标、与作者实现的偏离。
- 输入：语义/独热/原生类别/连续重构/score；二群体或多群体能力；A 与 y 在 fit、transform、predict 各阶段是否必需。
- 训练/校准：F、F 内开发子集、C 的职责；早停、辅助 A 或 A×Y 模型、后处理和阈值各自来源；从不把 S 当 epoch 验证集。
- 输出：事件概率 p、一般排序分数、随机决策概率 q、硬决策 yhat；不能相互替代。被公平后处理修改的分布不能只因在 [0,1] 就冒称校准风险。
- 权重：原生训练权重支持及真实使用位置；调查权重与算法重加权分别登记；七群体约束和零支持如何处理。
- 资源：执行平台、线程/GPU、候选和迭代预算、计时、峰值内存；随机拟合与随机决策分别标记。
- 审核：有效路径上调用真正算法；确定性/种子；F/C/S/T 扰动、未知类别、A×Y 缺失、输出语义、失败处理及可重载产物。

方法文献中的 EO 缩写必须展开。项目主 EO gap 为 equalized odds（同时考察 TPR 和 FPR），不是单独 equal opportunity。各方法可保留其原生目标，外部统一报告；没有优化某指标不代表不能评估它，但解释时必须写明目标不同。

对于 TabICL/TabPFN，训练上下文和任何跨行统计只能从规定开发数据建立；查询时固定单条计算与跨查询批次适应应区分。核验改变其他测试行是否改变当前人的输出，若使用测试分布做适应则单列为 transductive 扩展，不能纳入原来的严格冻结预测主表。

FairSHAP 等增强训练行的算法必须保留训练来源/生成行角色，调查评价只用原始年度受访者。新生成行不虚构 WTFA_A/PSTRAT/PPSU，不改变真实评价人数。不能通过行数扩张制造人群样本量。

## 8. 数据、指标、选优和论文叙事保持一致

继承 [主方案](/Users/lkc/Downloads/code_v_0_3/docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md)的 2022 F/C、2023 S、2024 T；2024 已被研究过程观察过，继续披露为回顾性评价，不能改称全新锁箱。Arm 003/004 保持同人群特征消融，Arm 004 不是仅残疾人队列。

主几何与主预测训练不加调查权重，外部评价用 WTFA_A；支持的调查加权训练另列敏感性，不能给某个对照默默乘调查权重。设计变量不作普通 X。多组最差差距、调查 domain、配对复制和种子离散性按独立 R2 工作验证；原始支持不足时报告不可估计。

全部方法共享加权 BA、equalized-odds gap、TPR/FPR gap、DP gap、各组人数/事件数/支持度，以及无权重敏感性。AP、AUROC、Brier、校准只在输出含义可比的风险/排序模型之间比较；q 不混入事件风险校准栏。还要展示最差组的绝对表现，辨别差距缩小是否仅来自较好组变差。

调参预算不能只数最终 fit：BM/AE 内部候选、EG 多次 oracle、FairGBM 迭代、神经模型训练、后处理网格和辅助群体模型都计入。使用共同外部选择规则和明确各自搜索空间，报告预算耗尽和不可行配置。先保留强预测参考，避免刻意弱化无干预基线。S 上选择，T 上评价；不按 T 改 epsilon、种子、主方法或竞争方法名单。

最终报告效应量与配对不确定性，区分同时改善、公平改善伴预测代价、无明确差异和不适用。FairBias 作为主方法意味着研究问题围绕它展开，不意味着必须战胜每一个对照。

**扩展会改变条件数和比较家族。** 原登记的 76 条件及 20 个主要差值适用于原矩阵；新增方法不能塞进去后继续沿用原总数。默认保留原主要推断家族，把新增方法作为完整报告的预声明扩展；若论文需要把近期对照升级为主要统计结论，在任何新增 T 评价前形成版本化登记，重算比较家族和校正方案。该选择等 R11 后统一定，不让 Gemini 在当前轮自行改。

## 9. R11 之后的路线，当前不执行

| 工作包 | 目的 | 完成定义 |
|---|---|---|
| R11 验收 | 结束当前软件交付修复 | 两项完整契约关闭，允许保留的已知边界准确报告；不以新增可选功能延长 |
| R1B＋候选登记修订 | 真正接通 FairBias 与竞争者，固定输出含义；审查近期方法卡片 | 先关闭 F01（从 F 学得真实 BM）与 F02（p/q/yhat 混用），然后逐方法合成适配通过、名单/版本/预算可冻结 |
| R2 | 调查 domain 和配对推断 | 独立参考与合成行为核验完成，支持不足和复制失败有明确定义 |
| R3 | 冻结并运行比较 | 先完成获准的可行性检查，冻结扩展矩阵、种子、候选和选优规则；分批串行运行并保留失败 |
| R4 | 分析与应用论文 | 完整公平—预测权衡、跨年/亚组/消融、预算和局限；不选择性报告、不重开 T 调参 |

这些是工作包，不承诺每包一次 Gemini 对话即可完成。首篇完成依赖适配和计算，而不是再补固定数量的报告。用户先把现有 R11 提示交给 Gemini；收到交付后，Codex 先验收 R11，再结合实际修复结果发布具体下一步任务。当前不创建自动监控，不发送新的 worker 任务，也不把本候选附录混进 R11。
