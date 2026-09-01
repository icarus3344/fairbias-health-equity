# MEPS Insurance Fairness：Project Learning Guide

这不是 API 手册，也不是项目宣传稿。它是一份“研究者如何自己把问题搭起来”的学习路线：每一章都按 WHAT → WHY → HOW → WHAT COULD GO WRONG 展开，并把抽象概念落回当前仓库里真实存在的函数。

## 先读这三条边界

1. 当前结果的标签是 PRELIMINARY_SINGLE_PANEL_DEVELOPMENT_ONLY。
2. 本指南只讨论已授权、已使用的 HC-244 / Panel 26 开发材料及其已有汇总 artifact；不读取、不使用 Panel 27 / HC-252。
3. Panel 26 有 136 个阳性事件。它适合学习 cohort、加权建模、校准、capacity decision、suppression 和 OOF 稳健性工程；不应被描述为 temporal validation、multi-panel evidence、final fairness evidence、population-level final inference 或 publication-ready result。

这里的“当前结果”指已经存在的运行汇总，不表示本次重新运行了模型。阅读本指南时，把代码行为、配置意图、已有 artifact 和研究结论分开看。

## 一页 mental model

我们要回答的是：

> 对 Panel 26 中在 Year 1 已连续有保险、年龄 18–64 岁的人，能否只使用 Year 1 的人口、社会经济、健康、功能、就医和 coverage 信息，预测其在 Year 2 是否出现至少一个 uninsured month？当模型被用于固定 10% capacity 的干预分配时，预测性能、校准和不同群体之间的 TPR disparity 是什么？

这句话已经包含了五个不同问题：

| 问题 | 它问什么 | 不能冒充什么 |
|---|---|---|
| Prediction | 谁更可能发生 outcome | 不是因果效应 |
| Calibration | 预测的 0.10 是否对应约 10% 的发生概率 | 不是 ranking ability |
| Operational allocation | 固定容量下选谁 | 不是把所有人都诊断成 positive |
| Fairness | 不同群体的错误和收益是否有差异 | 不是“一个公平数字” |
| Inference | 这些数字在抽样设计下有多大不确定性 | 不是仅凭一个点估计就代表总体真值 |

从零搭建时，最重要的顺序是先锁定 estimand，再锁数据时间边界，再写 cohort 和 outcome；最后才是模型。否则很容易得到一个“代码跑通、科学问题却变了”的系统。

## 第 1 章：研究问题、estimand 与整条证据链

### WHAT：我们到底在估计什么

每个研究对象是一个符合 cohort 条件的人，而不是一行任意问卷记录。令：

- \(X_i\)：Year 1 结束时或 baseline 可获得的 predictors；
- \(Y_i\)：Year 2 的保险状态 outcome；
- \(w_i\)：MEPS longitudinal survey weight；
- \(G_i\)：用于 fairness audit 的群体标签；
- \(S_i\)：固定容量规则产生的 selection indicator。

主要 outcome 是：

\[
Y_i = I(\text{Year 2 有至少一个 uninsured month})
\]

模型输出的是 \(\hat p_i=P(Y_i=1\mid X_i)\) 的估计或 ranking score。当前开发问题可拆为：

1. 模型能否把阳性事件排在前面（AUROC、AUPRC）？
2. 概率是否有意义（Brier、calibration intercept/slope、ECE）？
3. 在只能干预约 10% 人群时，实际能找到多少事件（TPR、PPV、capacity metrics）？
4. 固定决策规则下，不同群体的错误和收益是否有差异？
5. 在 complex survey design 下，这些统计量的不确定性如何表达？

这五个 estimand 彼此有关，但不是同一个 estimand。尤其是“预测得准”和“公平”不能用一句话替代。

### WHY：为什么这是 longitudinal prediction

Year 1 的信息先于 Year 2 的 outcome。这个时间顺序使研究问题具有预测意义：我们不是用整个两年期间的变量解释同一期间的 insurance loss，而是模拟“在较早时点进行风险分层”。如果把 Round 3–5 的后续变量放进 baseline，就可能把 outcome 发生之后的结果当成 predictor，得到虚假的好性能。

它适合 fairness research，是因为同一 operational policy 可能对群体产生不同的漏检率、选中率和阳性预测值；而 survey design 又提醒我们，样本中的人不是简单 iid 抽样。

### HOW：把科学问题翻译成 pipeline

~~~text
研究问题
  ↓
Panel 26 cohort + Year 1 predictor boundary
  ↓
Year 2 target construction
  ↓
DUID-grouped partition / grouped OOF
  ↓
train-only preprocessing
  ↓
survey-weighted model fitting
  ↓
validation model selection
  ↓
validation-only Platt calibration + frozen 10% threshold
  ↓
untouched development-test 或 outer OOF evaluation
  ↓
performance + calibration + operational utility + fairness audit
  ↓
明确写出 uncertainty、suppression 和 research boundary
~~~

研究上的“成功”不是生成一个最高的 AUROC，而是每一条箭头都能说明：输入是什么、为什么允许、输出回答哪个问题、哪里可能失效。

Relevant code: src/meps_fairness/data/cohort.py :: extract_meps_cohort；src/meps_fairness/pilot.py :: run_panel26_pilot；src/meps_fairness/crossfit.py :: run_panel26_oof_robustness；src/meps_fairness/pipeline.py :: run_pipeline。

## Checkpoint

### I should now be able to explain

- prediction、calibration、operational allocation、fairness、inference 的区别。
- 为什么 Year 1 → Year 2 的时间顺序是研究问题的一部分，而不只是代码约定。
- 为什么一个很高的 ranking metric 不能自动证明公平、可部署或因果有效。
- 一个完整结果至少要同时说明 cohort、target、split、权重、决策规则和不确定性。

### Without looking at the code, try

1. 用三句话写出 \(X_i\)、\(Y_i\)、\(w_i\) 和 10% capacity rule。
2. 给出一个例子：模型 AUROC 不变，但 calibration 或 TPR gap 发生变化。
3. 说出一个“看起来像预测研究、实际上发生时间泄漏”的做法。

### Common misconception

“模型能预测”不是“模型能公平地分配资源”，更不是“模型发现了导致失保的原因”。

## 第 2 章：MEPS、panel structure 与 survey design

### WHAT：MEPS 不是普通 CSV

MEPS（Medical Expenditure Panel Survey）是复杂抽样的纵向调查。当前项目使用的入口是 HC-244 / Panel 26 的已准备数据路径：

data/interim/meps/h244/h244.dta

Panel 可以理解为一批被追踪的受访者及其多个时间点记录；Year 1、Year 2 是纵向年度概念，Round 1–5 是访谈轮次概念。它们不是五个独立的年度 outcome。代码用 baseline 的 Year 1 / Round 1 变量构造 predictors，再用 Year 2 的保险变量构造 outcome。

### 关键 ID 和设计变量

| 字段 | 学习时的工作理解 | 研究风险 |
|---|---|---|
| DUID | dwelling / family 层面的分组标识；本项目的 split unit | 同一家庭成员落入不同 partition 会造成信息相关性泄漏 |
| DUPERSID | 个人的纵向标识 | 不能当成家庭 split unit 的替代品 |
| PID | 个人/调查记录相关标识 | 不能作为 predictor；否则可能记住人而不是学习风险 |
| LONGWT | longitudinal survey weight | 决定 weighted prevalence、weighted metrics 和 fitting influence |
| VARSTR | 方差估计的 strata | bootstrap / survey inference 时要尊重它 |
| VARPSU | primary sampling unit | 同 PSU 记录并非 iid；推断时按 PSU 重采样 |

Longitudinal weight 不是“这个人有多少倍重要”的随意分数，而是把样本参与者映射到目标调查总体的设计/调整信息。它可能高度不均等，因此 nominal row count 与 information content 不同。

### WHY：为什么不能 random row split

普通 random row split 假设行之间大体独立。如果同一 DUID 的家庭成员共享保险、收入、coverage 变化或未观测家庭因素，把一人放 train、另一人放 test，test 表现可能含有家庭信息的“回声”。这不一定是显式复制，但会让泛化误差偏乐观。

同理，直接把一个 panel 的多个时间记录当作 iid 行，也会破坏 outcome 与 predictor 的时间结构。正确的研究对象是一个符合 cohort 的 person-level longitudinal unit，同时保留 household grouping 和 survey design。

### HOW：artifact boundary

本项目的学习边界是：

- 读取和解释已经存在的 Panel 26 汇总结果；
- 解释已有代码如何从入口文件抽取 cohort、构造 target、分割 DUID、训练和审计；
- 不在本次任务中读取新的微观数据、不运行新模型、不改 runs/；
- Panel 27 / HC-252 只保持锁定状态，不作为训练、比较或“未来验证结果”。

configs/ 里的研究协议和脚本中的 preflight 是治理证据，不等于本次重新执行。已有 artifact 的行数、汇总表和代码静态行为也必须分别标记。

Relevant code: src/meps_fairness/data/cohort.py :: DESIGN_COLUMNS；src/meps_fairness/data/split.py :: split_panel26_duid_grouped；src/meps_fairness/evaluation/inference.py :: stratified_psu_bootstrap_inference；scripts/run_panel26_preliminary_pilot.py :: preflight。

## Checkpoint

### I should now be able to explain

- Year、Round、Panel 在 MEPS 中扮演的不同角色。
- DUID、DUPERSID、PID 不能互换。
- LONGWT、VARSTR、VARPSU 分别影响什么。
- 为什么复杂抽样设计会影响 prevalence、模型拟合和不确定性。
- 为什么当前学习只覆盖 Panel 26 development boundary。

### Without looking at the code, try

1. 用一个三人家庭例子说明 row split 的信息泄漏。
2. 解释为什么 LONGWT 不是可任意删除的普通列。
3. 说出“读了已有汇总 artifact”和“访问新的微观数据”有什么差别。

### Common misconception

有 train.csv、test.csv 两个文件并不代表 split 合理；关键是 split unit、时间边界和抽样结构是否与 estimand 一致。

## 第 3 章：从原始记录到研究 cohort

### WHAT：cohort 是 scientific population，不是清洗剩下的行

extract_meps_cohort 把原始记录变成一个有明确资格条件的研究人群。当前已有汇总显示：原始记录 6,741，最终 eligible 2,882，阳性 136，阴性 2,746。这个数字是已有 artifact 的开发快照，不是本次重新读数据得出的新结果。

核心筛选可写成：

~~~python
eligible = (
    (YEARIND == 1)
    & (ALL5RDS == 1)
    & (LONGWT > 0)
    & (18 <= AGEY1X <= 64)
)
eligible &= baseline_insurance_months_all_equal_1
~~~

### WHY：每个条件解决什么问题

| 条件 | 它解决的科学问题 | 如果去掉，可能发生什么 |
|---|---|---|
| YEARIND == 1 | 确认记录处在项目定义的 Year 1 baseline | 不同时间位置混在一起；predictor 时间点含义变模糊 |
| ALL5RDS == 1 | 确认五轮访谈链条完整 | follow-up 缺失与真实保险状态混淆，纵向可比性下降 |
| LONGWT > 0 | 只保留有可用 longitudinal weight 的对象 | weighted estimand 无法定义；零/无效权重可能使结果无意义 |
| 18 <= AGEY1X <= 64 | 锁定研究年龄人群 | 目标 population 改变；不同年龄的 insurance mechanism 混在一起 |
| Year 1 12 个月连续有保险 | 排除 baseline 已失保者，定义“从 coverage 到 loss”的风险起点 | 目标变成混合的 prevalence/transition 问题，无法清楚解释为 Year-2 loss |

这里的连续 baseline coverage 还承担 target definition contamination 控制：如果某人在 Year 1 已经失保，Year 2 又失保，就不再是本项目定义的 coverage-loss transition。

### HOW：资格、target 和完整性要分层

一个好的 cohort extractor 先做资格 mask，再做 target construction，再保留设计字段和 audit 字段，最后检查 predictor 的时间、ID、design 和 audit leakage。不要把 dropna() 当作 cohort definition；那会把“缺失”这一研究现象悄悄改成选择规则。

validate_leakage_and_integrity 会拒绝 temporal features、ID、design vars 和 audit attrs 进入 predictor 集合，并检查 weight、strata、PSU 的完整性。这样做的价值是把“研究禁区”变成可失败的 gate。

### IMPLEMENTATION NOTE

当前 extract_meps_cohort 使用 expected predictor columns 与实际列名的交集。如果输入文件缺少某些预期变量，函数可能静默跳过它们，最终 predictor 数量减少；已有运行 artifact 记录了 74 个 predictors，但这不应被理解为任何未来输入都自动满足 74 列 schema。独立 schema gate 仍然是实现债务。

另一个重要实现选择是 derive_composite_disability：六个 baseline screener 全为 1 时为 1，全为 2 时为 0；混合 nonresponse 的剩余情况最后被填成 0。这是可审计的当前实现，但“无法确认 disability”被并入无 disability 可能带来测量偏差，应在 sensitivity analysis 中单独讨论。

Relevant code: src/meps_fairness/data/cohort.py :: extract_meps_cohort；src/meps_fairness/data/cohort.py :: validate_leakage_and_integrity；src/meps_fairness/data/cohort.py :: derive_composite_disability。

## Checkpoint

### I should now be able to explain

- cohort inclusion 条件如何定义研究 population 和 estimand。
- 为什么 baseline 连续 coverage 是 transition 研究的起点条件。
- ALL5RDS 解决的是纵向完整性，不是普通缺失值填补。
- 为什么最终行数和阳性数应被当作 gate evidence，而不是随手打印的统计。
- 74 predictor 这个数字为什么还需要 schema 验证。

### Without looking at the code, try

1. 从零写出五个 inclusion 条件，并为每个条件配一个“去掉后果”。
2. 构造一个 baseline 已失保、Year 2 仍失保的人，说明为什么他不属于当前 transition cohort。
3. 解释把 mixed nonresponse disability 填 0 可能改变哪一个科学结论。

### Common misconception

“样本越多越好”不是 cohort 原则。加入不属于 target population 的人可能增加 nominal n，却破坏 estimand。

## 第 4 章：三个 target phenotype

### WHAT：同一个“保险流失”可以有不同 operational definition

当前设计使用三个 Year 2 phenotype：

1. Primary：any uninsured month。Year 2 的月度保险状态中只要有一个 uninsured month，Y_any = 1。
2. Sensitivity：至少三个 uninsured months。Y_≥3 = 1 当 Year 2 uninsured 月数 ≥ 3。
3. Sensitivity：year-end uninsured。Year 2 年末保险状态为 uninsured，Y_end = 1。

代码对 follow-up insurance code 先要求每个值属于允许集合 {1, 2}，然后才构造 target；全是 1 才是 primary negative，全是 2 才是 positive，混合状态按定义计算。

### WHY：primary 为什么可能 noisy

“至少一个月失保”对短暂事件很敏感。例如：

- 一个人换工作，旧雇主保险先结束，新工作保险下月生效：any-month 是 1，但未必是 persistent hardship。
- 行政 churn 或 enrollment timing 造成短暂空档：记录上是失保，福利风险却不同。
- 一个人连续四个月失保：any-month、≥3 months、year-end uninsured 可能都为 1，更接近 persistent uninsurance。

因此 primary target 可能提高事件数、便于学习，但也把不同机制混到一个标签里。Sensitivity target 可能更少、更难预测，却更接近 persistent risk；year-end target 关注时间点，可能漏掉年中短暂或已恢复的人。

### HOW：target 改变会改变三类东西

| 变化 | 可能影响 |
|---|---|
| prevalence | 阳性比例改变，AUPRC baseline 改变，power 改变 |
| predictability | 短暂 churn 可能比 persistent loss 更难从 Year 1 预测 |
| fairness | 某一群体的短期 churn 与持续失保比例不同，TPR gap 可能改变 |

这不是“哪个 target 才是真实”的简单选择，而是先声明 primary estimand，再用 sensitivity phenotype 检查结论是否依赖标签定义。

### IMPLEMENTATION NOTE

当前 target 构造的合法码检查是 fail-closed 的；但“any month”仍是较宽的 phenotype。不要把它写成“长期失保”或“因失去保险而遭受健康后果”。代码预测的是定义好的保险状态，不是 causal hardship。

Relevant code: src/meps_fairness/data/cohort.py :: extract_meps_cohort（follow-up insurance validation 与 y_any、y_prolonged、y_yearend construction）。

## Checkpoint

### I should now be able to explain

- 三个 target 的精确定义，以及它们的时间窗口。
- 为什么 any-month 可能包含 temporary job transition 和 administrative churn。
- 为什么 prevalence、predictability、fairness 会随 phenotype 改变。
- 为什么 sensitivity analysis 不是“再跑一个数字”，而是在检查 estimand robustness。

### Without looking at the code, try

1. 给出三个 Year 2 月度序列，分别只触发 any、同时触发三个、只触发年中 any。
2. 解释从 any 改为 ≥3 months 为什么可能降低阳性数。
3. 用一句话区分 target construction 和 causal explanation。

### Common misconception

一个更严格的 target 不一定“更科学”；它可能减少噪声，也可能丢失真实但短暂的 coverage disruption。

## 第 5 章：baseline predictor engineering

### WHAT：74 个 predictors 代表一个时间受限的信息集合

已有 cohort artifact 的 predictor contract 为 74 个变量：19 个 continuous、55 个 categorical。它们来自 Year 1 baseline，并按领域组织如下。

| 领域 | 当前变量（代码真实 contract） | 为什么可能相关 |
|---|---|---|
| Demographics / household | continuous: AGEY1X、EDUCYR、FAMSZEY1；categorical: HIDEG、MARRY1X、MARRY2X、MARRYY1X、REGION1、REGION2、REGIONY1 | 年龄、教育、家庭结构和地区可能影响 employment、public/private coverage 和 eligibility |
| Socioeconomic / employment | continuous: POVLEVY1、FAMINCY1、TTLPY1X、WAGEPY1X、HOUR1、HOUR2、NUMEMP1、NUMEMP2；categorical: POVCATY1、EMPST1、EMPST2、OFFER1X、OFFER2X、HELD1X、HELD2X、CHOIC1、CHOIC2、SELFCM1、SELFCM2、UNION1、UNION2 | 收入、贫困、工作时长、雇佣状态和 employer offer 可能连接到 coverage stability |
| Health status | RTHLTH1、RTHLTH2、MNHLTH1、MNHLTH2、HIBPDXY1、DIABDXY1_M18、ASTHDXY1、CHDDXY1、ANGIDXY1、MIDXY1、OHRTDXY1、STRKDXY1、EMPHDXY1、CHBRON1、CHOLDXY1、CANCERY1、ARTHDXY1、JTPAIN1_M18 | 健康需要可能关联 employer coverage、public eligibility、医疗支出和 enrollment continuity |
| Functional limitation | ACTLIM1、ADLHLP1、IADLHP1、WLKLIM1、SOCLIM1、COGLIM1 | 功能限制可能影响 employment、行政稳定性和 coverage pathway |
| Utilization | continuous: OBTOTVY1、OBDRVY1、OPTOTY1、ERTOTY1、IPDISY1、RXTOTY1、TOTEXPY1、TOTSLFY1；categorical: HAVEUS2、LOCATN2、PROVTY2_M18 | baseline utilization 可能是 health need、coverage type 和 care access 的 proxy |
| Insurance / coverage | INSCOVY1、PRIEUY1、PRINGY1、PUBY1X、MCAIDY1X、MCAREY1X、TRICRY1X、VAPROGY1 | 当前 coverage 类型和 payer structure 直接关联未来 insurance transition |

上述“可能相关”是研究假设，不是 feature importance 结论。预测信号最可能来自 baseline insurance、employment/income 和 coverage-related variables；健康和就医变量可能提供增量信号，也可能只是在 proxy 某些社会经济因素。低频疾病、稀疏类别和噪声 utilization 不一定贡献稳定 signal。

### WHY：时间点比“变量名字看起来合理”更重要

合法 predictor 要满足：

1. 在定义的 baseline 时间可获得；
2. 不含 Year 2 outcome 之后的信息；
3. 不只是 ID、survey design 或 audit attribute；
4. 含义与目标人群一致；
5. 缺失处理不会偷看 validation/test 分布。

例如 INSCOVY1 是 baseline coverage 描述，可作为风险起点的一部分；但 Year 2 insurance month 变量不能进入 X。Round 3/4/5 的访谈可能已经接近 outcome 发生期，把它们放进 baseline 会让模型看到未来。

### 一个 leakage example

假设某人在 Year 2 第三轮已经发生失保。若把 Round 3 的 insurance 或 employment 状态编码成 predictor，模型可能“预测”得很好，只因为 predictor 已经承载 outcome 后的状态。部署时在 Year 1 并没有这个信息，于是 offline score 和真实 deployment score 不可比。

### HOW：变量 contract 先于模型

从零实现时应先写出允许变量清单和禁止规则，再让 extractor 返回：

- predictors；
- target；
- DESIGN_COLUMNS；
- AUDIT_COLUMNS；
- cohort counts 和校验结果。

代码的 is_prohibited_temporal_feature 会根据后缀和轮次识别 Round 3/4/5、Y2 等时间泄漏风险，并允许少数定义上必要的例外。这个规则层比事后看 feature importance 更重要：一个泄漏变量即使“重要”，也不能进模型。

### IMPLEMENTATION NOTE

当前 predictor 集合是按实际输入列与 expected contract 的交集构成；这是对缺列较宽容的工程行为，不是严格 schema validation。变量名末尾的 2 也不能机械解释为 Year 2：当前 contract 中有些 2 是 baseline window 内的后续 round 字段；必须结合 dictionary、YEARIND 和 leakage rule 判断时间。另一个容易忽略的实现点是 prior-round inheritance：HOUR2、NUMEMP2、CHOIC2、SELFCM2、UNION2 的 -2 会从 Round 1 对应变量补回。这是当前 data interpretation 的一部分，必须在方法中说明。

Relevant code: src/meps_fairness/data/cohort.py :: ALL_BASELINE_PREDICTOR_COLUMNS；src/meps_fairness/data/cohort.py :: is_prohibited_temporal_feature；src/meps_fairness/data/cohort.py :: resolve_prior_round_inheritance；src/meps_fairness/data/cohort.py :: derive_age_band。

## Checkpoint

### I should now be able to explain

- predictor contract 与“把所有列喂给模型”的差异。
- 为什么 baseline coverage、employment/income 可能有信号，但不等于因果变量。
- 为什么 Round 3/4/5 可以造成 temporal leakage。
- 为什么 ID、survey design、audit attr 即使能提升分数也可能被禁止。
- prior-round inheritance 和缺失处理为什么是数据语义，而不只是技术清洗。

### Without looking at the code, try

1. 把一个变量分类为 predictor、design column、audit column 或禁止列，并说理由。
2. 写一个 Round 3 变量造成 leakage 的时间线。
3. 猜测：低频疾病变量可能为什么在一个只有 136 events 的开发数据里贡献不稳定？

### Common misconception

“变量在 baseline 文件里出现”不代表它在 baseline 时点可用于预测；必须检查其 round、year 和语义。

## 第 6 章：从零设计 train / validation / development test

### WHAT：三份数据承担三种不可互换的职责

当前单次开发 split 大致为：

- train：拟合 preprocessing 和候选模型；
- validation：model selection、Platt calibration、capacity threshold freezing；
- development test / calibration partition：在固定决策规则下做一次留出的开发评估。

这里的“test”要谨慎命名：它是 Panel 26 内的 development partition，不是未来 panel 的 temporal holdout。

### WHY：DUID leakage

构造一个家庭：

~~~text
家庭 DUID = 9001
  A：Year 1 baseline 信息，Year 2 outcome = 1
  B：Year 1 baseline 信息，Year 2 outcome = 0
  C：Year 1 baseline 信息，Year 2 outcome = 0
~~~

若 random row split 把 A 放 train、B 放 test，模型可能学到家庭层面的收入、coverage arrangement 或未观测家庭因素。B 的 test error 不再代表遇到新家庭时的 error，而代表“同一家庭另一个人”的 error。以 DUID 分组可以避免同一 household group 跨 partition。

这不是说同一家人的 predictors 一定相同，也不是说 DUID grouping 能消除全部依赖；它只是对可见 household grouping 做了必要的隔离。

### HOW：60 / 20 / 20 的角色

split_panel26_duid_grouped 先把数据压缩到 DUID 层面，按 group 是否有 positive 和 total weight 的分箱做受约束的 stratification，再在 group 层面打乱、分配约 60% / 20% / 20%。最终检查 partition 间 DUID disjoint，并记录 assignment hash。

模型选择只能看 validation，因为如果同时看 development test 选模型，test 就变成训练流程的一部分。Platt calibration 也只能用 validation，因为它是在学习 raw score 与 outcome 之间的映射。capacity threshold 只能在 validation 冻结，因为它是一个 policy parameter；冻结后才能把 development test 当作“应用已确定规则”的评估。

### 一个错误 threshold implementation

错误做法：

1. 在 train 拟合模型；
2. 在 validation 选模型；
3. 计算 validation + test 的预测；
4. 在 test 上重新找到“正好 top 10%”的 probability threshold；
5. 报告 test TPR/PPV。

第 4 步已经让 test outcome 参与政策确定。它通常会让 selection rate 看起来很整齐，并且让测试集的 rank 分布适配 threshold；但它不再是对预先冻结 policy 的估计。

### IMPLEMENTATION NOTE

当前项目有两个 threshold 语义：pilot 的 fixed_threshold_subgroup_audit 接受外部冻结 threshold；evaluation.metrics.subgroup_audit_metrics 则会基于自己收到的 subgroup probabilities/weights 重新计算 threshold。因此 generic helper 若被直接用于 primary audit，可能回答的是“每个 subgroup 自己 top 10%”而不是全局 frozen policy。当前 pilot/OOF 路径使用 fixed-threshold helper；这是需要在 review 中持续防止的实现债务。

Relevant code: src/meps_fairness/data/split.py :: split_panel26_duid_grouped；src/meps_fairness/pilot.py :: fit_validation_calibrator_and_freeze_threshold；src/meps_fairness/pilot.py :: fixed_threshold_subgroup_audit；src/meps_fairness/evaluation/metrics.py :: subgroup_audit_metrics。

## Checkpoint

### I should now be able to explain

- train、validation、development test 的职责为何不可交换。
- DUID grouped split 解决了什么、没有解决什么。
- 为什么 model selection、calibration、threshold freeze 都不能依赖 test outcome。
- “全局 frozen threshold”和“每个 subgroup 自己重新 top 10%”是两种不同政策。
- assignment hash 为什么有助于 provenance。

### Without looking at the code, try

1. 给 A/B/C 家庭例子画出合法和非法的 partition。
2. 说明如果 validation 只用于 model selection、test 只用于 fixed policy evaluation，哪些信息流被阻断。
3. 设计一个 test leakage 的错误报告，并指出它会夸大哪种结论。

### Common misconception

“threshold 只使用了预测概率，没有直接使用 y”仍可能泄漏：为了选择 top 10% 而在 test 上调 threshold，已经让 test 的分布参与了决策规则。

## 第 7 章：train-only preprocessing

### WHAT：预处理也是一个会学习参数的模型

当前 MEPSPreprocessor 对连续和 categorical predictors 分开处理：

- 连续变量：识别特定 item nonresponse codes（如 -7、-8、-15），按变量语义处理 legitimate negative；用 train median 填补，并用 train mean/std 标准化；
- 连续缺失：对训练期间观察到缺失的变量建立 missing indicator；
- categorical：从 train 中建立合法 categories，排除 item nonresponse codes 和 NaN，但保留有语义的 -1；未知或缺失值在 one-hot 中表现为 all-zero；
- transform 只使用 fit 阶段保存的统计量和类别集合。

当前源码还明确列出变量级语义：TTLPY1X、FAMINCY1、POVLEVY1 的负值可能是 legitimate negative；HOUR1、HOUR2、NUMEMP1、NUMEMP2、WAGEPY1X 的零值可能是 employment-related structural zero。把所有负值或所有零值统一当 missing，会改变真实的收入/就业含义。

fit 不是无害的准备动作：median、mean、std、category vocabulary 都是从数据估计出来的参数。

### WHY：没有 y 也会 leakage

假设 test 中某个收入变量极端偏高。若在全体数据上 fit scaler，train 的 mean/std 已经被 test 分布改变。即使没有读 y，train 表示空间也被 test 信息影响。模型参数因此是在“知道未来输入分布”的表示上估计，test performance 不能再代表严格的 out-of-sample workflow。

类似地，全数据 fit categorical encoder 会把 test 独有类别放进 train 的 feature space；全数据 imputer 会让 test 的缺失模式影响训练填补值。无标签 leakage 仍是 leakage。

### 一个小例子

~~~text
train income = [10, 12, 14]
test income  = [1000]
只在 train fit：median=12，mean=12，std≈2
全数据 fit：mean 被 1000 拉高，scale 改变
~~~

对于树模型，scale 的影响通常较小；对 Logistic 的线性系数和 penalty ratio 可能很重要。无论模型类型如何，原则仍是：每个 outer/inner train 单独 fit。

### IMPLEMENTATION NOTE

构造函数接受 variable_contracts 参数，但当前主要 fit/transform 逻辑并未真正消费它来强制变量语义；不能因为参数名称存在，就说 contract 已经发挥了完整 schema gate 作用。另一个需解释的行为是 categorical 的未知/缺失可能落成 all-zero，而不是显式的 Unknown 指示列；这会把多个状态压到同一表示。

check_survey_design_quality 会检查权重、strata、PSU 等设计质量，但它不是一个自动修复器；通过检查只说明设计字段可用于后续步骤，不代表样本具有足够 event power。

Relevant code: src/meps_fairness/data/preprocess.py :: MEPSPreprocessor.fit；src/meps_fairness/data/preprocess.py :: MEPSPreprocessor.transform；src/meps_fairness/data/preprocess.py :: _prepare_continuous_series；src/meps_fairness/data/preprocess.py :: check_survey_design_quality。

## Checkpoint

### I should now be able to explain

- continuous、categorical、item nonresponse 与 legitimate negative 的区别。
- 为什么 median、mean/std、category vocabulary 都必须 train-only fit。
- 为什么没有使用 y 也可以发生 preprocessing leakage。
- missing indicator 和 categorical all-zero 表示的解释风险。
- 为什么 preprocessor 的通过不等于 study 的 statistical validity。

### Without looking at the code, try

1. 用三行伪代码写出 train-only fit / val transform / test transform。
2. 解释全数据 fit imputer 如何改变研究问题，即使 outcome 没有被读入。
3. 构造一个 categorical unknown 与 missing 被 all-zero 合并的例子，并说出可能后果。

### Common misconception

“只有在 feature selection 使用 y 才叫 leakage”是错误的；任何让评估分布反向影响训练表示的步骤都需要审查。

## 第 8 章：Survey weights 与 Kish effective sample size

### WHAT：一个 MEPS respondent 不等于一个普通 observation

普通机器学习常把每一行看成一票。复杂调查不是这样：不同 respondent 被抽中的概率、nonresponse adjustment 和 longitudinal follow-up eligibility 可能不同。LONGWT 让样本统计量更接近研究目标总体的 weighted estimand。

例如样本中有四个人，outcome 为 0、0、1、1，权重为 1、1、8、10：

~~~text
普通 sample mean = (0+0+1+1)/4 = 0.50
weighted prevalence = (1×0+1×0+8×1+10×1)/(1+1+8+10)
                   = 18/20 = 0.90
~~~

0.50 只回答“样本行中有多少阳性”；0.90 回答的是按当前权重定义的 sample-weighted prevalence。到底哪个 estimand 合适，取决于研究问题；当前 MEPS workflow 明确使用 survey weights。

### WHY：LONGWT 影响的不只是 prevalence

权重至少会影响：

- target prevalence 和 AUPRC 的 baseline；
- AUROC、AUPRC、Brier 和 calibration 的 weighted evaluation；
- model fitting 中每条观测对 loss 的贡献；
- capacity rule 中“覆盖 10% 总体质量”而非简单 10% 行数；
- survey-aware uncertainty，例如按 VARSTR 内 VARPSU 重采样。

如果把 LONGWT 当成普通 feature，它会丢失这些作用；如果完全不用它，回答的就可能是 unweighted sample estimand。

### Kish effective sample size

当权重相等时，nominal sample size 大致等于 information size；权重不均等时，少数高权重对象可能主导结果。Kish effective sample size 为：

\[
n_{\mathrm{eff}}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
\]

两个简单例子：

~~~text
equal weights: [1, 1, 1, 1]
sum = 4，sum of squares = 4
n_eff = 4²/4 = 4

unequal weights: [1, 1, 1, 10]
sum = 13，sum of squares = 103
n_eff = 13²/103 ≈ 1.64
~~~

第二个例子仍有 4 行，但一个单位承担了大量 weighted mass，独立信息量更接近 1.64 而不是 4。Kish n_eff 不是所有 survey variance 问题的完整答案，但它是快速发现“名义样本够大、有效信息很少”的直觉工具。

### HOW：把 weights 与 design variables 分开

LONGWT 用于 weighted estimand；VARSTR 和 VARPSU 用于反映抽样结构的 uncertainty。它们不是三个可以互换的权重列。当前汇总 artifact 中，eligible cohort 有 2,882 条 person-level records、weighted prevalence 约 0.0526、Kish effective n 约 1,785.7、105 个 strata 和 357 个 PSU clusters；其中还有 single-PSU stratum。这些数字支持“设计字段存在并已审计”，不自动保证 inference 稳健。

### IMPLEMENTATION NOTE

当前 inference helper 是 stratified PSU bootstrap：在每个 VARSTR 内抽 VARPSU，并按抽样次数调整 weights，再计算 paired metrics。它是 survey-aware 的工程近似，但不能把一个 apparent-fit 的 calibration 结果变成外部验证，也不能用大 nominal n 消除 136 events 带来的 power 限制。

Relevant code: src/meps_fairness/evaluation/metrics.py :: weighted_auroc；src/meps_fairness/evaluation/metrics.py :: weighted_auprc；src/meps_fairness/evaluation/inference.py :: stratified_psu_bootstrap_inference；src/meps_fairness/data/split.py :: PartitionData。

## Checkpoint

### I should now be able to explain

- 为什么 weighted prevalence 不一定等于普通 sample mean。
- LONGWT 如何同时进入 prevalence、model fitting、evaluation 和 capacity。
- Kish effective sample size 的公式和直觉。
- VARSTR / VARPSU 为什么与 LONGWT 的功能不同。
- 为什么 n_eff 大于某个阈值仍不能自动证明 event power 足够。

### Without looking at the code, try

1. 对权重 [1, 1, 8, 10] 和 outcome [0, 0, 1, 1] 算 weighted prevalence。
2. 对 [1, 1, 1, 10] 算 Kish n_eff，并解释为什么小于 4。
3. 说明“按 weighted mass 选 10%”和“按行数选 10%”何时会不同。

### Common misconception

survey weight 不是“为了让模型更重视困难样本”的机器学习技巧；它首先是 estimand 和抽样设计的一部分。

## 第 9 章：为什么 LONGWT 的 scale 会改变 penalized Logistic

### WHAT：raw weight 与 mean-1 weight

当前 robustness experiment 比较两种 fitting weight：

- LEGACY_RAW_LONGWT：直接使用 LONGWT；
- MEAN1_NORMALIZED_LONGWT：用 LONGWT 除以其均值，使平均 fitting weight 约为 1。

两种方式都可以在 evaluation 时保留原始 survey weights；“fit weight 怎么缩放”和“结果怎么按 survey design 评价”是两个问题。

### WHY：无正则化时的等比例缩放

设 logistic 的 weighted negative log-likelihood 为：

\[
L(\beta)=\sum_i w_i\,\ell_i(\beta).
\]

如果所有权重同时乘以常数 \(c>0\)，则：

\[
L_c(\beta)=\sum_i(cw_i)\ell_i(\beta)=cL(\beta),
\]

最小化位置通常不变，因为整个 objective 只是乘了同一个正数。这是“等比例 scaling 通常不改变 unregularized optimum”的直觉来源。

### HOW：L2 penalty 破坏了这个不变性

sklearn 当前 Logistic baseline 使用带 L2 regularization 的目标，直觉上可写为：

\[
J(\beta)=\sum_i w_i\,\ell_i(\beta)+\lambda\lVert\beta\rVert_2^2.
\]

缩放 weights 后：

\[
J_c(\beta)=c\sum_i w_i\,\ell_i(\beta)+\lambda\lVert\beta\rVert_2^2.
\]

惩罚项没有同时乘 c，所以相对比例改变：

- raw LONGWT 很大时，data loss 相对于固定 penalty 更占优势；
- mean-1 后 data loss 被缩小，固定 penalty 相对更强；
- 系数、概率、ranking 甚至 model selection 都可能变化。

这不是“归一化只改变单位”的纯数学现象，因为当前模型有 penalty。对 Random Forest 和 Histogram Gradient Boosting，weight scale 也可能影响加权分裂或叶节点统计，但它们没有同样的显式 L2 coefficient penalty，因此已观察到的变化较小。

### 当前 robustness evidence

已有 OOF candidate 汇总如下；这是已存在 artifact 的汇总，不是本次新跑：

| fitting weight | Logistic AUROC / AUPRC | RF AUROC / AUPRC | GB AUROC / AUPRC |
|---|---:|---:|---:|
| raw LONGWT | 0.494 / 0.0581 | 0.649 / 0.0772 | 0.633 / 0.0756 |
| mean-1 LONGWT | 0.598 / 0.0796 | 0.645 / 0.0759 | 0.633 / 0.0756 |

绝对数字不应脱离 split、OOF 方案和 event count 解读；但方向很有教学意义：Logistic AUPRC 约从 0.0581 变为 0.0796，RF/GB 变化小得多。当前 weight-sensitivity summary 还显示 selected pipeline 的 OOF AUPRC 差异约 0.000063，但候选模型最大差异约 0.0215，主要来自 Logistic。也就是说，“最终选中的流水线看起来稳定”不能掩盖候选模型层面的 scale sensitivity。

### IMPLEMENTATION NOTE

这里的 mean-1 是 fitting sensitivity mode，不是把 survey weights 从 estimand 中删除。写 methods 时必须明确：weight normalization 改变了 penalized optimization 的相对尺度；evaluation 和 prevalence 是否仍用原始 LONGWT 另行说明。

Relevant code: src/meps_fairness/crossfit.py :: fitting_weights；src/meps_fairness/models/baseline.py :: WeightedLogisticClassifier；runs/panel26_oof_robustness_20260901T040204171693Z_20260828/weight_scale_sensitivity.csv。

## Checkpoint

### I should now be able to explain

- 为什么无正则化 weighted likelihood 对整体 weight scaling 通常有不变性。
- L2 penalty 为什么让 raw 与 mean-1 的 Logistic optimum 不再等价。
- 为什么 weight sensitivity 可能改变 model selection，而不只是四舍五入。
- 为什么应把 fitting weights 与 evaluation weights 分开记录。

### Without looking at the code, try

1. 从 \(J(\beta)=\sum w_i\ell_i+\lambda\lVert\beta\rVert^2\) 推出整体缩放后哪一项比例变了。
2. 预测一个只有 tree model 的流程是否会完全不受 weight scale 影响，并说明你为什么不能直接断言“完全”。
3. 用当前表格写一句 evidence-bounded 结论，不使用“最终模型已确定”。

### Common misconception

“平均权重变成 1，所以 survey weighting 消失了”是错误的；它只描述 fitting objective 的 scale 处理，不能替代 survey-weighted estimand。

## 第 10 章：三个 baseline models 与 exploratory mitigation

### WHAT：模型是研究设计中的一个组件

当前 baseline suite 有三个候选：

#### Logistic Regression

\[
p_i=\sigma(\beta_0+X_i^\top\beta),\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
\]

它假设 log-odds 与输入表示近似线性。优点是可解释、训练稳定、适合先建立方向性 baseline；缺点是对非线性、复杂交互和稀疏/错测变量敏感。当前实现使用 sklearn LogisticRegression，solver 为 lbfgs、C=1.0、max_iter=10000，并传入 sample_weight。

#### Random Forest

Random Forest 建很多决策树，每棵树通过 bootstrap/随机特征子集寻找分裂，再平均树的预测。它能表达 threshold effect 和 interaction，通常不要求连续变量先标准化；但树间相关、叶节点稀疏、概率 calibration 可能有限。当前参数是 100 棵树、max_depth=6、min_samples_leaf=10，并把 sample_weight 传入 fit。

#### Histogram Gradient Boosting

Gradient boosting 逐轮拟合前面模型的残差/损失方向，每一轮修正当前 ensemble。它能高效表达非线性，常比单个浅树或某些 RF 更会利用弱 signal；但 sequential correction 也可能追逐噪声，尤其在事件少、类别稀疏时。

### WHY：为什么不争论“理论上谁最好”

当前结果 RF 与 GB 的 OOF AUROC/AUPRC 接近：raw 下约 0.649/0.0772 与 0.633/0.0756，mean-1 下约 0.645/0.0759 与 0.633/0.0756。它告诉我们的是：在当前 74 predictors、136 events、固定配置和 grouped OOF 定义下，两个非线性 baseline 提供了相近但不完美的 signal。它没有告诉我们某个算法在所有 target、panel 或 deployment context 都最好。

### HOW：FairBias/mitigation track 要单独理解

src/meps_fairness/models/mitigation.py 中的 ExploratoryGroupAwareCenteringMitigation 是探索性 group-aware centering extension：它在 representation 或 score 上使用 protected attribute 信息，要求 inference 时也能取得该属性。代码明确把它标成 exploratory，不是 Tang/FairBias paper 的严格 reproduction。baseline 仍是 group-agnostic。

这有两个科学后果：

1. “公平改善”可能来自改变 decision/representation，而不是预测 signal 变强；
2. 如果部署时拿不到 protected attribute，方法就不具备当前形式的 operational deployability。

### IMPLEMENTATION NOTE

部分配置/协议文字把 Logistic 描述成 elastic-net，但当前 baseline.py 的实际构造是 penalty="l2"。应以当前执行代码为准，并在 methods 或审计中保留这个不一致标记；不能把 L2 结果写成 elastic-net 结果。

Relevant code: src/meps_fairness/models/baseline.py :: WeightedLogisticClassifier；src/meps_fairness/models/baseline.py :: WeightedRandomForestClassifier；src/meps_fairness/models/baseline.py :: WeightedHistGradientBoostingClassifier；src/meps_fairness/models/mitigation.py :: ExploratoryGroupAwareCenteringMitigation。

## Checkpoint

### I should now be able to explain

- Logistic、RF、GB 各自能表达什么、不能表达什么。
- sample_weight 如何进入三个 baseline 的训练接口。
- 为什么 RF≈GB 是当前设计下的观察，而不是算法排行榜。
- 为什么 exploratory mitigation 与严格 paper reproduction 不能混写。
- 为什么 protected-attribute-at-inference 是部署边界。

### Without looking at the code, try

1. 给一个非线性 employment effect，说明 Logistic 可能需要什么额外表示。
2. 说出 RF 和 boosting 的一个共同风险与一个不同风险。
3. 解释为什么 group-aware mitigation 若减少 disparity，不等于提高 causal fairness。

### Common misconception

“模型越复杂，scientific evidence 越强”是错误的。模型复杂度增加的是函数空间，不是自动增加事件数、外部有效性或公平性证据。

## 第 11 章：AUROC、AUPRC 与弱 signal

### WHAT：两个 ranking metric 回答不同的视角

AUROC 是随机抽取一个 positive 和一个 negative 时，模型把 positive 排在前面的概率的样本估计。它对 threshold 不敏感，适合观察整体 ranking，但在低 prevalence 场景可能看起来比实际 intervention experience 乐观。

AUPRC 关注 precision-recall 曲线。其坐标更直接地询问：当我找回更多事件时，选中的人里有多少是真的 positive？

### WHY：当前 prevalence 约 5.26%

如果一个无信息模型把人随机排序，PR 曲线的 expected precision 大致等于阳性 prevalence。因此当前 random baseline 的 AUPRC 约为 0.0526。已有 candidate OOF AUPRC 约 0.075–0.080，说明比随机排序有一些 signal，但不是“接近完美”。

例如：

\[
\frac{0.077}{0.0526}\approx1.46.
\]

可以把它理解为一个粗略 ranking lift：平均 PR 表现约是 prevalence baseline 的 1.46 倍。它不能意味着：

- 选 top 10% 就有 46% 的阳性率；
- 模型解释了 46% 的 risk；
- 每个 subgroup 都有同样的 lift；
- 这个 lift 在新 panel 或总体中一定保持。

当前 candidate raw/mean-1 表格中，Logistic raw AUPRC 约 0.0581，RF 约 0.0772，GB 约 0.0756；mean-1 Logistic 约 0.0796。这种低 absolute PR 不是指标失败，而是低 prevalence + 弱 signal 的如实表达。

### HOW：不要用 accuracy 替代 PR

若 1000 人中只有 50 个 positive，全部预测 negative 的 accuracy 为 95%，但 TPR 为 0，无法帮助资源分配。AUPRC、TPR、PPV 和 capacity metrics 更贴近本研究的 intervention question。

### IMPLEMENTATION NOTE

当前 weighted_auprc 以 prediction 排序、累计 weighted true positives，再用加权 recall/precision 做 trapezoidal integration；它不是把每个 row 当等权的 unweighted average。解释结果时应说明 weighted metric、target prevalence 和评估 partition。

Relevant code: src/meps_fairness/evaluation/metrics.py :: weighted_auroc；src/meps_fairness/evaluation/metrics.py :: weighted_auprc；src/meps_fairness/crossfit.py :: _probability_metrics。

## Checkpoint

### I should now be able to explain

- AUROC 和 AUPRC 各自评价什么。
- 为什么随机 classifier 的 expected AUPRC 大致是 prevalence。
- 0.077 / 0.0526≈1.46 可以支持什么，不能支持什么。
- 为什么 95% accuracy 可能对应 TPR=0。

### Without looking at the code, try

1. 构造 20 人、1 个 positive 的例子，说明 accuracy 为什么危险。
2. 用一个排序结果解释 precision 随 recall 上升可能下降。
3. 说出一个必须同时报告 prevalence 才能解释的 PR 结果。

### Common misconception

AUROC=0.65 不是“65% 的人预测正确”；它是 pairwise ranking probability，且不直接说明 top-capacity selection 的 PPV。

## 第 12 章：Operational utility 与 10% capacity threshold

### WHAT：threshold 是 policy，不只是数学截断点

当前 capacity target 为 10%。如果总体有 1000 人、50 个 positive、只能干预 100 人：

- selection rate = 100 / 1000 = 10%；
- 如果选中的 100 人中有 20 个 positive，TPR = 20 / 50 = 40%；
- PPV = 20 / 100 = 20%；
- FPR = 被选中的 negative / 全部 negative；
- F1 = \(2\cdot PPV\cdot TPR/(PPV+TPR)\)。

这几个分母不同，所以“选中 10%”不等于“找到 10% positive”，也不等于“precision 10%”。

### WHY：capacity threshold 要在 validation freeze

代码的 find_weighted_capacity_threshold 按预测概率从高到低排序，用 cumulative survey weight 找到达到 target_capacity × total_weight 的 crossing。它模拟“最多覆盖总体 weighted mass 的 10%”。如果每次在 test 上重算 crossing，就把 test 分布用于设定 policy，造成 leakage。

一个直观比较：

~~~text
总共 50 个事件，能干预 100 人
理想 top-100 全是事件：TPR=100%，PPV=50%
随机 top-100：期望找到约 5 个事件，TPR≈10%，PPV≈5%
当前开发证据：selection≈9–10%，TPR≈10–13%，PPV≈6–7%
~~~

当前数字表明：规则确实选出了约 10% 的 weighted capacity，但事件捕获和阳性率都只有有限提升。现有 pilot development partition 的 selection 约 7.75%、TPR 约 11.44%、PPV 约 6.62%；OOF selected pipeline 的 raw/mean-1 selection 约 9.44%/9.12%，TPR 约 13.27%/9.69%，PPV 约 7.40%/5.59%。这些不是外部验证结果。

### HOW：capacity、rank、probability 要分开

如果只关心 top 10%，排序比 absolute probability 更重要；如果要把输出解释为风险概率，calibration 又不可省略。capacity threshold 在 calibrated probability 上可以有 policy 解释，但在 fixed-capacity ranking 里，单调变换可能保留排序而改变 threshold 数值。

### IMPLEMENTATION NOTE

由于 threshold 按 LONGWT 的累计质量计算，row-level selection rate 可能不是精确 10%。报告时要同时写 weighted capacity、row count、selection rate 的定义，避免把它们混成一个数字。

Relevant code: src/meps_fairness/evaluation/calibration.py :: find_weighted_capacity_threshold；src/meps_fairness/pilot.py :: fixed_threshold_metrics_with_fpr；src/meps_fairness/evaluation/metrics.py :: capacity_metrics。

## Checkpoint

### I should now be able to explain

- selection rate、TPR、FPR、PPV、F1 的分母。
- 10% capacity threshold 为什么是 operational policy。
- 为什么 threshold 用 weighted cumulative mass，而不一定按行数。
- 为什么在 test 上重新选 threshold 会泄漏。
- 当前 selection≈9–10%、TPR≈10–13%、PPV≈6–7% 的实际含义。

### Without looking at the code, try

1. 在 1000 人、50 positives、100 capacity 的例子中，若选中 8 positives，计算 TPR 与 PPV。
2. 给出一个排序相同但 probability threshold 不同的例子。
3. 解释为什么“10% capacity”不能直接改写成“模型找到 10% positives”。

### Common misconception

选中率固定只控制资源量，不保证 precision、recall 或 fairness。

## 第 13 章：Calibration、Platt scaling 与低 prevalence

### WHAT：raw score 不是自动可信的 probability

模型可能很会排序，却把 0.8 说得过于自信。Platt calibration 先把 raw probability \(p\) 转为 clipped log-odds：

\[
z=\log\left(\frac{\mathrm{clip}(p)}{1-\mathrm{clip}(p)}\right),
\]

再拟合：

\[
\Pr(Y=1\mid z)=\sigma(a+bz).
\]

其中 \(a\) 是 calibration intercept，\(b\) 是 calibration slope。当前实现把 validation raw probabilities、outcomes 和 survey weights 交给 weighted LogisticRegression（L2，C=10），然后在同一 validation 上得到 calibrated probabilities，并从 validation 冻结 capacity threshold。

### WHY：四个 calibration quantity

- **Calibration intercept**：整体预测风险偏高还是偏低。理想情况下接近 0。
- **Calibration slope**：风险排序的斜率/极端程度。理想情况下接近 1；小于 1 常提示 raw score 过于极端，虽然具体解读还依赖估计方式。
- **Brier score**：\(\sum_i w_i(p_i-y_i)^2/\sum_iw_i\)，越小越好，但受 prevalence 影响。
- **ECE**：把预测分箱，比较每箱平均预测与加权事件率的差异，再加权汇总。

### HOW：为什么 ECE≈0.007 不能叫“完美”

当前 OOF selected pipeline 的 ECE 约为 0.0071（raw fitting）和 0.0078（mean-1 fitting），看起来小。但至少有五个限制：

1. low-prevalence 下绝对差异可能看起来小，尤其很多 bin 事件很少；
2. ECE 依赖 bin 数量和 binning rule，不是一个无争议的 population parameter；
3. 当前 metrics helper 用 equal-width bins（np.linspace 0 到 1），虽然注释有“risk deciles”字样，实际不等同于 equal-frequency deciles；
4. 一个漂亮的 aggregate ECE 可能掩盖 subgroup miscalibration；
5. apparent-fit calibration 或同一阶段调参得到的数字不能替代真正 holdout calibration。

Calibration 不等于 discrimination：一个模型可以 slope/intercept 较好但 AUROC 接近随机，也可以 ranking 好但概率系统性偏高。

### 当前结果如何读

OOF selected aggregate 的 raw/mean-1 calibrated AUROC 约 0.626/0.625，AUPRC 约 0.0729/0.0697，Brier 约 0.0498/0.0498；calibration intercept 约 -0.67/-0.75，slope 约 0.76/0.73。这里的 calibrated AUROC 与 raw AUROC 不应简单当作“校准提高了 ranking”：对单调 Platt mapping，理论上排序通常保持；差异反映当前实现、折叠预测和汇总定义，需以 artifact 为准。

### IMPLEMENTATION NOTE

主 pipeline 的某些 calibration diagnostics 在同一 calibration partition 上拟合并诊断，因此是 apparent-fit evidence；pilot 通过 validation fit、development partition evaluation 的路径更接近隔离，但仍是单 Panel development。OOF 路径在 inner validation fit calibrator，并把它应用到 outer test，解释更接近 out-of-fold，但仍没有时间外部验证。

Relevant code: src/meps_fairness/evaluation/calibration.py :: SurveyWeightedPlattCalibrator；src/meps_fairness/evaluation/metrics.py :: weighted_brier_score；src/meps_fairness/evaluation/metrics.py :: weighted_calibration_stats；src/meps_fairness/crossfit.py :: calibration_bins。

## Checkpoint

### I should now be able to explain

- raw score、log-odds、Platt calibrated probability 的转换。
- intercept、slope、Brier、ECE 的不同含义。
- 为什么 ECE≈0.007 不能证明 calibration 完美。
- low prevalence、binning 和 apparent fit 如何影响 calibration 解读。
- 为什么 monotone calibration 与 ranking metrics 的关系需要谨慎验证。

### Without looking at the code, try

1. 一个模型所有概率都乘 2（并截断）时，为什么 calibration 会变差但排序可能不变？
2. 写出 calibration intercept 偏负的直觉解释。
3. 设计一个“总体 ECE 小、某个 subgroup ECE 大”的例子。

### Common misconception

“Brier 小”或“ECE 小”都不能单独证明模型能在现实中做出公平且有效的 intervention decision。

## 第 14 章：Fairness、confusion matrix 与 suppression

### WHAT：先从 decision rule 再谈群体差异

给定冻结 threshold，定义：

- TP：\(Y=1\) 且被选中；
- FN：\(Y=1\) 但未被选中；
- FP：\(Y=0\) 但被选中；
- TN：\(Y=0\) 且未被选中。

对群体 \(g\)：

\[
TPR_g=\frac{TP_g}{TP_g+FN_g},\qquad
FPR_g=\frac{FP_g}{FP_g+TN_g},
\]

\[
PPV_g=\frac{TP_g}{TP_g+FP_g},\qquad
selection_g=\frac{TP_g+FP_g}{n_g}.
\]

当前 primary fairness endpoint 是 unsuppressed groups 之间的 maximum pairwise TPR gap：

\[
\Delta_{\mathrm{TPR}}=\max_{g,h}|TPR_g-TPR_h|.
\]

### 三组小例子

假设冻结同一个全局 threshold 后：

~~~text
Group A: TP=8, FN=2  -> TPR=0.80
Group B: TP=5, FN=5  -> TPR=0.50
Group C: TP=2, FN=8  -> TPR=0.20
max pairwise gap = 0.80 - 0.20 = 0.60
~~~

这不是说 C “被算法故意歧视”，也不是一个因果效应；它表示在当前 outcome、sample、weight、threshold 和 suppression rule 下，C 的事件捕获率与 A 相差很大。机制可能来自 prevalence、signal、measurement、sample size、policy interaction 或数据缺失，不能由 gap 单独决定。

### WHY：suppression 保护谁，也限制我们能说什么

当前 subgroup estimability 规则要求：

- n ≥ 100；
- positive count ≥ 20；
- negative count ≥ 20；
- Kish effective n ≥ 50。

原因是比例的分母、分子和 weighted mass 太小时，TPR/FPR/PPV 会非常不稳定；报告小 cell 的精确 disparity 还可能暴露隐私或制造过度确定的叙事。suppression 是“不可可靠估计”，不是把该组设为 0，也不是证明该组公平。

### HOW：FULL、PARTIAL、NOT ESTIMABLE

- **FULLY_ESTIMABLE**：该 fairness dimension 的所有预定义 groups 都通过 suppression，才可以报告完整 maximum gap。
- **PARTIALLY_ESTIMABLE_UNSUPPRESSED_GROUPS_ONLY**：部分 groups 通过，只能在 unsuppressed subset 内计算；不能把它叫完整 race maximum。
- **NOT_ESTIMABLE_SUPPRESSED**：没有足够有效 groups 支撑该 endpoint，应报告不可估计，而不是猜一个数字。

当前已有 OOF audit 中，SEX 是 fully estimable；RACETHX 是 partial。RACETHX 的某些类别（已有汇总中 category 4、5）因为阳性数或 Kish/样本条件不足被 suppressed，所以剩余类别的 gap 只能称为 unsuppressed-subset gap。primary fairness endpoint 因 race 没有完整估计而不能写成完整的 race+sex fairness conclusion。更早的单次 pilot development partition 中 Race/Sex cells 均 suppressed，primary 为 NOT_ESTIMABLE_SUPPRESSED。

### IMPLEMENTATION NOTE

OOF 的 audit helper 输入的是每个 outer-fold 的 frozen binary decisions，然后汇总 full/partial coverage；它没有为每个 subgroup 重新选择 threshold。与此同时，generic evaluation.metrics.subgroup_audit_metrics 的实现会从传入 subgroup probability/weight 自己找 threshold。两者的 policy semantics 不同，使用者必须确认调用路径。

Relevant code: src/meps_fairness/pilot.py :: fixed_threshold_subgroup_audit；src/meps_fairness/crossfit.py :: audit_oof_binary_decisions；src/meps_fairness/evaluation/metrics.py :: subgroup_audit_metrics；src/meps_fairness/pilot.py :: _kish_effective_n。

## Checkpoint

### I should now be able to explain

- TPR、FPR、PPV、selection rate 的分母。
- maximum pairwise TPR gap 在回答什么问题。
- n、positive、negative、Kish 四个 suppression 条件的直觉。
- FULLY、PARTIALLY、NOT ESTIMABLE 三种 coverage status。
- 为什么 RACETHX partial gap 不能冒充完整 race maximum。

### Without looking at the code, try

1. 用三组 confusion counts 手算一次 maximum TPR gap。
2. 构造一个 n 很大但 positive<20 的 subgroup，解释为什么仍 suppress。
3. 把“某组 suppressed”改写成一条不会过度解释的 research sentence。

### Common misconception

suppression 不是公平性结论为 0；它是对估计可靠性和隐私风险的控制。

## 第 15 章：为什么 single split 不够

### WHAT：一次留出评估只能给一个 realization

已有第一次 preliminary pilot 的流程是：train 拟合候选模型，validation 选择 RF，validation 拟合 calibrator 并冻结 threshold，最后在 Panel 26 内的 development partition 评估。已有结果约为：

| 指标 | 单次 development partition |
|---|---:|
| selected model | RF |
| frozen threshold | 0.0878 左右 |
| AUROC | 0.5729 |
| AUPRC | 0.0513 |
| weighted selection | 0.0775 |
| TPR | 0.1144 |
| PPV | 0.0662 |

这个数字很重要，但它只告诉我们这一次 DUID-grouped split、这一次样本构成和这一次 selected model 的结果。

### WHY：三个解释无法靠一个数字区分

当 test AUPRC 约 0.0513，而后来的 grouped OOF aggregate 约 0.07 左右时，至少存在三种解释：

1. split unlucky：一次 development partition 恰好包含更难的对象；
2. model weak：signal 本来就很弱，OOF 也只比 prevalence baseline 有限提高；
3. model-selection instability：validation 在有限 events 下选择出的 model 不稳定，某次刚好选到 RF。

单次 split 没有提供足够重复结构来分解这三个来源。它也没有把当前 Panel 26 变成 temporal validation。

### HOW：cross-fitting 的目的

5-fold grouped OOF 让每个对象轮流成为 outer test，且每次 preprocessing、model fitting、model selection、calibration 和 threshold 都只用对应 development data。这样可以把多个 out-of-fold predictions 拼起来，得到一个更稳定的 development-only aggregate，并记录每个 fold 的 selected model、threshold、metrics 和 fairness coverage。

它解决的是 resampling / internal validation 的 uncertainty 观察，不是：

- 增加阳性事件；
- 产生新的 panel；
- 证明未来时间泛化；
- 代替正式 survey inference；
- 让 fairness small cells 自动变大。

Relevant code: scripts/run_panel26_preliminary_pilot.py :: execute；src/meps_fairness/pilot.py :: run_panel26_pilot；src/meps_fairness/crossfit.py :: run_panel26_oof_robustness。

## Checkpoint

### I should now be able to explain

- 单次 development test 结果为什么只能是一个 split realization。
- 0.0513 与 0.07 的差异为什么不能直接归因于“第一次运气差”。
- cross-fitting 解决的是哪类不确定性，不解决哪类 external validity。
- 为什么 single split 的 model selection 仍需要稳定性检查。

### Without looking at the code, try

1. 用三种解释解释 test AUPRC 较低，而不做过度归因。
2. 画出一个 5-fold OOF 中“对象只在一次 outer test 出现”的逻辑。
3. 说明为什么 OOF 不能创造 Panel 27 证据。

### Common misconception

更多 folds 或 OOF 能减少内部 split 偶然性，但不能修复 target noisy、event 少、测量偏差或时间外部有效性缺失。

## 第 16 章：Nested grouped OOF cross-fitting

### WHAT：两层分割各自回答不同问题

把 Full Panel 26 cohort 想成下面的结构：

~~~text
Full Panel 26 development cohort
|
|-- Outer Fold 1 test
|      +-- remaining development
|             |-- Inner train (约 3/4)
|             +-- Inner validation (约 1/4)
|
|-- Outer Fold 2 test
|      +-- remaining development
|             |-- Inner train
|             +-- Inner validation
|
|-- ... Outer Fold 5 test
~~~

Outer fold 模拟“当前开发样本中的未见对象”；inner split 负责在不触碰 outer test 的情况下做训练流程选择。

### WHY：inner validation 必须隔离 outer test

每个 outer fold 中：

- inner train：fit preprocessing、fit candidate models；
- inner validation：选择 candidate、拟合 Platt calibrator、冻结 capacity threshold；
- outer test：只接收已确定的 preprocessor、model、calibrator 和 threshold，产生 prediction 和 binary decision。

如果在 outer test 上重新选模型或 threshold，OOF prediction 就不再是 out-of-fold。它仍可能有一个漂亮的数字，但不能代表预先定义的 workflow。

### HOW：当前实现的关键细节

当前 run_grouped_oof 使用 outer 5 folds、inner 4 folds。inner validation 为一个 fold，inner train 为其余三个 folds。每个层级都检查 DUID disjoint，并在 inner train 上 fit preprocessor。candidate model 在 inner train 训练，使用 inner validation AUPRC（再以 AUROC、模型名作 tie-break）选择。

随后：

1. selected model 在 inner validation 上产生 raw probabilities；
2. weighted Platt calibrator 在 inner validation fit；
3. capacity threshold 从 inner validation 冻结；
4. 同一个 fitted preprocessor/model/calibrator 应用于 outer test；
5. outer test 保存 calibrated probability 和 thresholded binary decision；
6. 五个 outer predictions 拼成一份 aggregate OOF prediction。

每个人恰好有一个 OOF prediction，是因为 group assignment 对每个 eligible record 只指定一个 outer fold，且记录了 duplicate/missing prediction 检查。若一个人出现两次，aggregate 会把未知的测试对象泄漏进自己；若一次都没有，aggregate coverage 就不完整。

### 权重语义

当前 OOF 允许 raw LONGWT 与 mean-1 normalized fitting weights 两种 mode。模型拟合可以使用 normalized weights，但 outer evaluation 保留原始 LONGWT 来计算 weighted metrics；这能把 penalty scale sensitivity 与 survey estimand 拆开。

### IMPLEMENTATION NOTE

OOF 的每个 outer fold 可以选不同 model、不同 calibrator、不同 threshold。把拼接后的 predictions 叫“一个固定模型在未来数据上的表现”是不准确的；更准确是“预先定义的 grouped OOF selection-and-evaluation workflow 的 development robustness evidence”。compare_weight_modes 的阈值差异是 descriptive sensitivity，不是 formal hypothesis test。

Relevant code: src/meps_fairness/crossfit.py :: assign_outer_folds；src/meps_fairness/crossfit.py :: run_grouped_oof；src/meps_fairness/crossfit.py :: fitting_weights；src/meps_fairness/crossfit.py :: summarize_grouped_oof。

## Checkpoint

### I should now be able to explain

- outer test 与 inner validation 的不同职责。
- 为什么 inner validation 负责 model selection、calibration、threshold freeze。
- 为什么每个 eligible person 应当恰好有一个 OOF prediction。
- 为什么 OOF aggregate 不是 temporal holdout。
- 为什么每个 fold 可能选择不同 model 和 threshold。

### Without looking at the code, try

1. 给 outer fold 1 写出 train/validation/test 的信息流。
2. 找出一个会让 OOF prediction 失去 out-of-fold 性质的错误步骤。
3. 解释 normalized fitting weight 与 original evaluation weight 为什么可以同时出现。

### Common misconception

“5-fold”本身不保证无泄漏；必须检查 group disjoint、preprocessing fit、model selection、calibration 和 threshold 是否都留在正确层级。

## 第 17 章：当前 robustness result 应该怎样读

### WHAT：先看 candidate-level，再看 selected-pipeline-level

已有 OOF candidate 汇总：

| fitting mode | Model | AUROC | AUPRC | Brier |
|---|---|---:|---:|---:|
| raw LONGWT | GB | 0.6334 | 0.0756 | 0.0519 |
| raw LONGWT | Logistic | 0.4943 | 0.0581 | 0.0719 |
| raw LONGWT | RF | 0.6486 | 0.0772 | 0.0495 |
| mean-1 LONGWT | GB | 0.6334 | 0.0756 | 0.0519 |
| mean-1 LONGWT | Logistic | 0.5977 | 0.0796 | 0.0555 |
| mean-1 LONGWT | RF | 0.6454 | 0.0759 | 0.0495 |

这些是每个 candidate 在 OOF 方案下的汇总。另一个 selected-pipeline OOF summary 是“每个 fold 按 inner validation 选中谁，再拼接这些 selected predictions”，所以 selected pipeline 的 raw/mean-1 AUPRC 约为 0.0667/0.0667，并不应直接和 candidate RF 的 0.0772 当成同一个 estimand。

### WHY：五个核心结论

#### 1. 第一次 0.051 是 somewhat pessimistic，但不是纯 split accident

相对 OOF 的约 0.07，第一次 pilot test 的 0.0513 看起来偏低；但 OOF 也只比 prevalence 0.0526 高有限，candidate AUROC 约 0.63–0.65。合理结论是：一次 split 可能偏难，但整体 signal 仍弱。不能说 OOF“证明”第一次错误，也不能把差异写成 temporal improvement。

#### 2. 不能再说 RF 是 universally best model

RF 在 raw candidate table 的 AUROC/AUPRC 较高，但 mean-1 Logistic AUPRC 反而最高；model-selection frequency 也不稳定：raw mode 中 GB 被选 3/5 folds、RF 2/5，mean-1 中 RF 3/5、GB 2/5，Logistic 0/5。这个证据支持“RF/GB 是相近的非线性 candidates，selection 随 mode/fold 波动”，不支持一个最终普适冠军。

#### 3. model-selection instability 本身是研究结果

当有限 events 下不同 outer folds 选择不同模型，说明 model identity 对 partition 有敏感性。它提醒我们报告候选范围、fold choices 和 stability，而不是只报告一个被选名称。

#### 4. weight normalization 对 Logistic 有实质意义

Logistic raw AUPRC 约 0.0581、mean-1 约 0.0796；RF/GB 较稳定。这个差异与 L2 objective 中 data loss/penalty 相对尺度改变相符，说明 weight handling 是 model specification 的一部分。

#### 5. 相近 predictive performance 仍可有不同 fairness disparity

selected-pipeline OOF 的 raw/mean-1 calibrated performance 相近，但 SEX TPR gap 约 0.1528/0.0780；RACETHX unsuppressed-subset gap 约 0.1078/0.0683，且 race 仍是 partial。预测性能相近不意味着错误分配结构相同；fairness audit 必须在同一 frozen-policy 语义下单独做。

### 当前 evidence boundary

这些结果支持：

- Panel 26 单 panel development 内的弱但非零 ranking signal；
- candidate model 与 fitting-weight sensitivity；
- OOF selection stability、calibration 和 subgroup estimability 的工程诊断。

这些结果不支持：

- 对未来 panel 的时间泛化；
- 完整 race fairness evidence；
- population-level final inference；
- 发表级部署结论；
- 因果或机制解释。

Relevant code: src/meps_fairness/crossfit.py :: summarize_grouped_oof；src/meps_fairness/crossfit.py :: compare_weight_modes；runs/panel26_oof_robustness_20260901T040204171693Z_20260828/oof_summary.json；runs/panel26_oof_robustness_20260901T040204171693Z_20260828/model_selection_stability.csv。

## Checkpoint

### I should now be able to explain

- candidate-level metric 与 selected-pipeline metric 为什么不是同一个表。
- 为什么 pilot AUPRC 0.0513 可以偏悲观，却仍与整体弱 signal 相容。
- model-selection frequency 如何成为 stability evidence。
- weight normalization 为什么主要改变 Logistic。
- 为什么相似 performance 仍可能有不同 fairness gap。

### Without looking at the code, try

1. 用一段不夸大的话总结当前 Panel 26 evidence。
2. 解释为什么不能把 candidate RF 最高写成“RF 是最终最佳模型”。
3. 看到 race partial 和 sex full 时，写出各自允许的 claim。

### Common misconception

OOF 有更多 predictions 不等于有更多独立 events；136 个阳性事件的事实仍然决定 power 和不确定性。

## 第 18 章：如果从零重建，researcher 应按什么顺序

下面的顺序比“先找一个模型跑起来”更接近可靠的研究工作。每个 stage 都要有可检查 artifact。

| Stage | Objective | Concepts I need | Minimum implementation | Validation test | Common mistake | Expected artifact |
|---|---|---|---|---|---|---|
| 1 | 写清 research question | estimand、prediction vs causal、time origin | 一页 protocol | 每个变量能回答“何时可得” | 用模型目标替代科学问题 | study protocol |
| 2 | 理解 MEPS dictionary | panel/year/round、ID、survey design | 变量字典表 | 标出 person、family、design 字段 | 把代码名当成语义 | data dictionary notes |
| 3 | 锁 data boundary | authorized file、hash、no new data | 输入 manifest/preflight | 路径、hash、panel tag 一致 | 先下载/探索再补协议 | data manifest |
| 4 | 写 cohort extraction | target population、eligibility | eligibility mask | 每个条件计数、正负数 | 用 dropna 定义人群 | cohort summary |
| 5 | 写 target construction | phenotype、invalid code | primary + sensitivity targets | 合法码、月数、边界例子 | 把 any month 叫 persistent | target QA table |
| 6 | 写 predictor contract | baseline boundary、leakage | allowed/forbidden lists | future suffix、ID、design rejection | 把所有字段喂给模型 | predictor manifest |
| 7 | 处理继承和缺失 | nonresponse、structural zero | domain-aware resolver | 手工验证几个 code path | 把 -2/-7/-8 当普通数字 | cleaning audit |
| 8 | 先做 descriptive design audit | weights、strata、PSU、Kish | quality report | weight>0、strata/PSU nonmissing | 只报告 row count | design summary |
| 9 | 设计 grouped partition | household dependence | DUID split | DUID disjoint、hash、counts | random row split | partition manifest |
| 10 | 写 train-only preprocessor | fit/transform、representation | imputer/scaler/encoder | test 改动不影响 train fit | 全数据 fit | preprocessing audit |
| 11 | 建立 weighted baselines | objective、regularization | Logistic/RF/GB | 结果可复现、sample_weight 传递 | 未记录 weight mode | candidate metrics |
| 12 | 定义 validation selection | selection rule、tie-break | AUPRC→AUROC→name | test 不参与排序 | 用 test 选冠军 | selection manifest |
| 13 | 建 calibrated policy | Platt、Brier、capacity | validation calibrator + threshold | threshold 来源只含 validation | test 重算 threshold | calibration/policy artifact |
| 14 | 做 fixed-policy audit | confusion matrix、suppression | TPR/FPR/PPV/fairness | subgroup coverage status | suppressed 当 0 | fairness audit |
| 15 | 做 uncertainty analysis | PSU bootstrap、effective n | paired inference/limitations | strata/PSU resampling trace | 把 bootstrap CI 当外部验证 | inference report |
| 16 | 做 grouped nested OOF | internal robustness、OOF | outer 5 + inner 4 | exactly one OOF prediction | outer test 参与调参 | OOF bundle |

每完成一个 stage，先冻结 artifact 再进入下一步。这样即使后面发现模型弱，也能知道是 research question、data contract、label、split、preprocessing 还是 model 导致的。

### IMPLEMENTATION NOTE

当前仓库已经有这些组件，但“代码存在”不等于每一层都达到 publication-ready。尤其要把 inherited baseline、single-split pilot、OOF robustness、exploratory mitigation 和 formal temporal validation 分开命名。

Relevant code: src/meps_fairness/data/cohort.py；src/meps_fairness/data/split.py；src/meps_fairness/data/preprocess.py；src/meps_fairness/models/baseline.py；src/meps_fairness/evaluation/calibration.py；src/meps_fairness/evaluation/inference.py；src/meps_fairness/pilot.py；src/meps_fairness/crossfit.py。

## Checkpoint

### I should now be able to explain

- 为什么 research question、cohort、target、split 必须在 model 之前。
- 每个 stage 的最小 artifact 和 validation test。
- 哪些实现是 internal robustness，哪些才可能支撑 external validation。
- 为什么“代码已经存在”不等于“证据已经成立”。

### Without looking at the code, try

1. 从 Stage 1 到 Stage 16 说出每一步的输入和输出。
2. 任选一个 stage，设计一个 fail-closed test。
3. 找出一个如果顺序颠倒就会引入 leakage 的 stage pair。

### Common misconception

重建研究项目的核心不是打字速度，而是每一步都留下可审计的中间定义和失败条件。

## 第 19 章：Hands-on reconstruction exercises

这些练习按难度递增。先只看任务和验收条件；参考解答放在 Appendix A，不要一开始抄实现。

### Level 1：weighted primitives

**任务**

1. 自己实现 weighted prevalence。
2. 自己实现 Kish effective n。
3. 给 binary predictions 和 outcomes 实现 TPR/FPR/PPV。

**验收条件**

- [1,1,8,10] 与 [0,0,1,1] 的 weighted prevalence 为 0.90；
- [1,1,1,10] 的 Kish n_eff 约为 1.64；
- 全部 negative 时 TPR=0；没有 predicted positive 时 PPV 明确返回缺失/不可定义，而不是除零得到假数字；
- 至少写 5 个 edge-case tests。

### Level 2：DUID grouped split

**任务**

输入一个小表，包含 DUID、y、LONGWT。先聚合到 DUID，再实现近似 60/20/20 的分配。

**验收条件**

- 同一 DUID 只出现在一个 partition；
- partition index 覆盖每一行且无重复；
- positive DUID 和 weight mass 不会全部落在同一份；
- 固定 seed 得到同一 assignment hash。

### Level 3：train-only preprocessing

**任务**

用 train/validation 两个小表实现 continuous median imputation + standardization 和 categorical vocabulary。

**验收条件**

- 改变 validation 的极端值不会改变 train 的 median/mean/std；
- validation 新类别不会改变 train 的 columns；
- item nonresponse 与 legitimate negative 不被同样处理；
- transform 后 train/validation columns 对齐。

### Level 4：weighted Logistic baseline

**任务**

用一个有 sample_weight 的 binary 数据拟合 Logistic baseline，并比较 raw weights 与 mean-1 weights。

**验收条件**

- 记录 penalty、C、solver、seed/iteration；
- 确认 sample_weight 确实传入 fit；
- 对无 penalty 的 toy objective 解释整体 scaling；
- 对 L2 model 观察并解释 coefficients/probabilities 可能变化，而不是只报告变化。

### Level 5：Platt calibration + frozen capacity

**任务**

先用 train 拟合模型；只用 validation fit Platt calibrator 和 10% weighted capacity threshold；最后把 threshold 原样用于 holdout。

**验收条件**

- holdout outcome 不被 threshold function 读取；
- threshold 的来源被记录为 validation；
- 报告 raw/calibrated Brier、intercept/slope、selection、TPR、PPV；
- 尝试在 holdout 上重算 threshold，并写出为什么那是错误 workflow。

### Level 6：简化 nested grouped OOF

**任务**

实现 5 个 outer group folds；每个 outer development 内再做 4-fold inner split，选择两个候选模型，fit calibrator，freeze threshold，再对 outer test 预测。

**验收条件**

- 每个 row 恰好一次 OOF prediction；
- preprocessing 和 selection 不接触 outer test；
- 输出每-fold model、threshold、event counts；
- aggregate metrics 只使用 OOF predictions；
- 对至少一个 group attribute 产生 suppression status。

### IMPLEMENTATION NOTE

练习的目标是掌握信息流和 evidence contract，不是让你在本次任务中运行新模型或修改 runs。真实项目的练习应使用独立 toy data 或已授权的既有 fixture，并遵守当前 gate。

Relevant code: tests/；src/meps_fairness/data/split.py :: split_panel26_duid_grouped；src/meps_fairness/evaluation/calibration.py :: SurveyWeightedPlattCalibrator；src/meps_fairness/crossfit.py :: run_grouped_oof。

## Checkpoint

### I should now be able to explain

- 每个 Level 练习在重建 pipeline 的哪一层。
- 每道题的验收条件是在检查什么 scientific risk。
- 为什么先做 toy data 能帮助区分算法 bug 与数据问题。
- 为什么 Level 6 的 exactly-one OOF test 比“代码跑完”更关键。

### Without looking at the code, try

1. 只用纸笔完成 Level 1 的三个公式。
2. 解释 Level 3 为什么要测试“validation 改变不影响 train fit”。
3. 给 Level 6 写一条最重要的 assertion。

### Common misconception

能写出一个函数不等于掌握 research workflow；掌握的标志是你能说明它的输入边界、输出 estimand 和失败后果。

## 第 20 章：Oral exam / interview questions

请遮住其他章节，逐题口头回答。回答不应只说 API 名称，而应说 research reason、information flow 和 limitation。

### Basic

1. 当前研究预测的 outcome 是什么？
2. 为什么它是 longitudinal prediction？
3. Year 1 predictor 和 Year 2 outcome 的时间顺序是什么？
4. DUID、DUPERSID、PID 大致分别是什么？
5. LONGWT 的作用是什么？
6. VARSTR 和 VARPSU 的作用是什么？
7. 为什么不能把 MEPS 当普通 iid CSV？
8. YEARIND == 1 解决什么问题？
9. ALL5RDS == 1 解决什么问题？
10. 为什么要求 18–64 岁？
11. 为什么要求 baseline 12 个月连续有保险？
12. primary target 与 ≥3-month target 有什么区别？

### Intermediate

13. temporary job transition 为什么可能触发 any-month target？
14. 为什么 Round 3/4/5 变量可能造成 leakage？
15. 为什么 predictor contract 要排除 ID 和 design columns？
16. 为什么 random row split 会让家庭成员泄漏？
17. train、validation、development test 各自负责什么？
18. 为什么 model selection 只能使用 validation？
19. 为什么 Platt calibration 只能在 validation fit？
20. 为什么 threshold 不能在 test 上重新计算？
21. 为什么全数据 fit scaler 即使不看 y 也会 leakage？
22. weighted prevalence 的公式是什么？
23. Kish effective n 的公式是什么？
24. 当 weights 极不均等时，nominal n 为什么会误导？
25. 为什么整体 scaling 对无正则化 likelihood 通常不改 optimum？
26. L2 Logistic 为什么打破这个不变性？
27. Logistic、RF、GB 各自的一个优势和一个风险是什么？
28. 为什么 AUPRC 对当前低 prevalence 项目很重要？

### Advanced

29. random classifier 的 expected AUPRC 为什么大致等于 prevalence？
30. 0.077 / 0.0526≈1.46 能解释成什么、不能解释成什么？
31. capacity threshold 按 row count 和按 weighted mass 有什么区别？
32. selection rate、TPR、FPR、PPV 的分母分别是什么？
33. calibration intercept 和 slope 的理想值及直觉是什么？
34. 为什么 ECE≈0.007 不能证明 calibration 完美？
35. 为什么 aggregate calibration 可能掩盖 subgroup miscalibration？
36. max pairwise TPR gap 的公式是什么？
37. suppression 的 n、positive、negative、Kish 条件各自保护什么？
38. RACETHX partial gap 为什么不能当完整 race maximum gap？
39. SEX fully estimable 允许我们声称什么？
40. single split 的低 AUPRC 可能由哪三个来源造成？
41. nested grouped OOF 中 outer 和 inner 两层分别做什么？
42. 为什么每个人必须恰好有一个 OOF prediction？
43. 为什么 OOF 不是 temporal validation？
44. 当前 evidence 为什么只能叫 PRELIMINARY_SINGLE_PANEL_DEVELOPMENT_ONLY？

### IMPLEMENTATION NOTE

口试中若你发现 prose 与代码有差异，应先说“当前实现是什么”，再说“理想设计是什么”。例如实际 Logistic 是 L2，不应为了迎合配置文字回答成 elastic-net；generic subgroup helper 会重算 threshold，也不应把它和 pilot/OOF fixed-threshold path 混为一谈。

Relevant code: src/meps_fairness/pilot.py；src/meps_fairness/crossfit.py；src/meps_fairness/evaluation/metrics.py；src/meps_fairness/models/baseline.py。

## Checkpoint

### I should now be able to explain

- 44 个问题覆盖了数据、统计、模型、policy、fairness 和 governance。
- 口试回答应包含 definition、reason、implementation 和 limitation。
- 代码存在的行为与理想方法文字不一致时，如何诚实标记。

### Without looking at the code, try

1. 随机抽取 Basic 3、Intermediate 20、Advanced 42，限时各回答 90 秒。
2. 对 Advanced 38 写一条不超过两句的证据边界回答。
3. 解释一个“会背 metric 名称但还没有掌握项目”的错误回答。

### Common misconception

能准确复述函数名不等于理解 estimand；导师或面试官更可能追问“如果去掉这一步，scientific error 是什么”。

## 第 21 章：Researcher ownership checklist

如果你要向导师汇报、写 CV、面试、写 methods 或 defense 中说“I worked on / developed this project”，至少要能承担下面这些解释责任。

### MUST KNOW

- 用一句话写出 target population、baseline information boundary、Year 2 outcome。
- 解释三种 target phenotype 及 primary/sensitivity 的关系。
- 说出 YEARIND、ALL5RDS、LONGWT、年龄和 baseline continuous coverage 的作用。
- 解释 DUID split 与 random row split 的差别。
- 说出 train-only preprocessing 的 leakage 机制。
- 写出 weighted prevalence 和 Kish n_eff 公式。
- 解释 raw 与 mean-1 LONGWT 对 L2 Logistic 的影响。
- 说明 candidate model、model selection、calibration、threshold freeze 的信息流。
- 从 confusion matrix 解释 TPR/FPR/PPV 和 10% capacity。
- 解释 suppression、FULL、PARTIAL、NOT ESTIMABLE。
- 诚实报告 136 events、弱 signal 和单 Panel development boundary。
- 说清楚当前不能声称 temporal validation、final fairness evidence 或 publication-ready。

### SHOULD KNOW

- 能从 cohort contract 找到 74 predictors 并按领域解释其可能机制。
- 能解释 Round 3/4/5 temporal leakage 和 prior-round inheritance。
- 能读懂 RF、GB 与 Logistic 的主要超参数。
- 能推导 Platt log-odds calibration 并解释 intercept/slope/Brier/ECE。
- 能复述 nested grouped OOF 的 inner/outer information flow。
- 能解释为什么 selection stability 和 weight sensitivity 是 evidence，而不是 nuisance。
- 能读懂 execution manifest、assignment hash、model selection table 和 fairness audit。
- 能指出 generic subgroup helper 的 threshold semantic risk。
- 能区分 survey-aware bootstrap、internal OOF robustness、formal external validation。

### NICE TO KNOW

- 能审阅 mitigation.py 的 exploratory group-aware centering，并解释 protected attribute availability 的部署问题。
- 能重新实现 toy weighted PR、capacity threshold 和 grouped OOF。
- 能设计 target、missingness 和 weight-scale sensitivity analysis。
- 能提出如何在有授权时进行真正的外部/时间验证，而不提前把 Panel 26 说成它。
- 能把每个 claim 链回 code、manifest、test 和 artifact。

### 最终 ownership test

你真正掌握项目的标志不是“能让 pipeline 运行”，而是遇到一个新结果时能依次问：

1. 这个结果对应哪个 estimand？
2. 数据和时间边界是什么？
3. 哪些人被纳入、哪些人被排除？
4. weights 和 design 如何进入它？
5. 有没有 leakage 或 threshold reuse？
6. fairness cell 是否 estimable？
7. uncertainty 和 model-selection stability 怎么办？
8. 这条证据允许我说到哪一步，不能说到哪一步？

Relevant code: src/meps_fairness/pipeline.py :: evaluate_development_power_gate；src/meps_fairness/pipeline.py :: locked_panel27_status；docs/AI_EXECUTION_PROTOCOL.md；runs/ 下的既有 Panel 26 aggregate artifacts。

## Checkpoint

### I should now be able to explain

- MUST KNOW 是不能外包给 AI 的研究责任。
- SHOULD KNOW 是能独立审阅和修改研究设计的能力。
- NICE TO KNOW 是扩展、复现和向导师 defend 的能力。
- 每条 scientific claim 都应该有 estimand、code path、artifact 和 limitation。
- 为什么当前项目的最诚实 ownership statement 必须带上 preliminary single-panel development qualifier。

### Without looking at the code, try

1. 用两分钟向导师解释当前项目，不使用“final”“validated”“fair”这类没有限定的词。
2. 看到一个新 AUROC，按最终 ownership test 的八个问题逐项审问。
3. 自己给出一个可以写进 methods、但不能写进 abstract conclusion 的句子。

### Common misconception

“AI 写了代码、我知道代码大概在做什么”还不等于 ownership。能够预测错误后果、拒绝越界 claim、独立重建关键步骤，才接近研究者掌握。

## Appendix A：Solution Notes（先做题，再打开）

### A.1 Weighted primitives

weighted prevalence：

~~~text
weighted_mean = sum(w_i * y_i) / sum(w_i)
~~~

Kish：

~~~text
n_eff = sum(w_i)^2 / sum(w_i^2)
~~~

confusion metrics：

~~~text
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
PPV = TP / (TP + FP)
~~~

当分母为 0，应返回不可估计状态并记录原因；不能静默改成 0。

### A.2 Grouped split pseudocode

~~~text
groups = summarize_each_DUID(y_presence, total_weight)
strata = (has_positive, weight_bin)
shuffle_groups_with_seed(strata)
assign_whole_groups_to_train_val_test()
assert disjoint(DUID_train, DUID_val, DUID_test)
assert union_of_indices_is_all_rows()
record_assignment_hash()
~~~

先在 group level 分配，再把 group assignment 映射回 rows。直接对 rows 打乱再“事后去重”不等价。

### A.3 Train-only preprocessing

~~~text
fit(train):
  learn train medians, means, scales
  learn train categorical vocabulary
  record train missing indicators

transform(new):
  apply saved statistics and saved columns
  never refit on new
~~~

验收的关键不是 output 看起来合理，而是改变 validation/test 后，train-fitted state 完全不变。

### A.4 Weighted Logistic intuition

把每个 row 的 loss 乘以 \(w_i\)，再加入 L2 penalty。将 weights 全部乘 c 后，data term 乘 c，penalty 不乘 c，所以“相对惩罚强度”变化。你不需要先背 sklearn 内部实现，但要能从 objective 解释为什么 raw/mean-1 可能不等价。

### A.5 Platt + frozen capacity

~~~text
raw validation probability
  -> clip
  -> logit
  -> weighted logistic calibration
  -> calibrated validation probability
  -> sort by calibrated probability
  -> cumulative validation LONGWT
  -> freeze threshold at 10% weighted capacity
holdout:
  -> transform/predict/calibrate with saved objects
  -> apply saved threshold
~~~

holdout 只被读取来评价，不被读取来重新确定最后两步的 policy 参数。

### A.6 OOF and partial fairness

~~~text
for each outer fold:
  outer_test = held out groups
  inner_train, inner_val = split development groups
  fit preprocessing/model on inner_train
  select and calibrate on inner_val
  freeze threshold on inner_val
  predict outer_test once
concatenate outer_test predictions
audit decisions by group
report full/partial/not-estimable coverage
~~~

若某些 race cells 被 suppress，剩余 groups 的 gap 是 subset gap；不能把 suppressed groups 当成 TPR=0、也不能默认为与其他组相同。

## Appendix B：代码地图

### 数据与设计

- src/meps_fairness/data/cohort.py：cohort、targets、predictor contract、temporal leakage、inheritance、disability composite。
- src/meps_fairness/data/split.py：DUID grouped split、partition validation、assignment hash、Kish summary。
- src/meps_fairness/data/preprocess.py：train-only fit/transform、missing codes、categorical encoding、design quality。

### Models and evaluation

- src/meps_fairness/models/baseline.py：weighted Logistic、Random Forest、Histogram Gradient Boosting。
- src/meps_fairness/models/mitigation.py：exploratory group-aware centering extension；不是严格 FairBias reproduction。
- src/meps_fairness/evaluation/calibration.py：Platt calibrator、weighted capacity threshold。
- src/meps_fairness/evaluation/metrics.py：weighted AUROC/AUPRC/Brier、calibration stats、capacity、generic subgroup audit。
- src/meps_fairness/evaluation/inference.py：stratified PSU bootstrap inference。

### Orchestration

- src/meps_fairness/pilot.py：Panel 26 preliminary single split pilot、validation selection、fixed-threshold audit、stop condition。
- src/meps_fairness/crossfit.py：grouped nested OOF、weight-scale comparison、fold stability、fairness coverage。
- src/meps_fairness/pipeline.py：更完整的 development pipeline、power gate 和 lock status；阅读时保留其 apparent-fit 与 development-only 限制。
- scripts/run_panel26_preliminary_pilot.py：已有 pilot 的 preflight/execute wrapper。
- scripts/run_panel26_oof_robustness.py：已有 OOF aggregate artifact 的执行 wrapper。

### Verification and protocol

- tests/：数据、split、preprocess、metrics、pilot、OOF 的单元与集成验证入口；测试通过只证明实现行为，不证明 science claim。
- configs/study.json：研究协议和参数的配置来源；与执行代码不一致处必须记录。
- configs/cohort_and_variables.json：cohort/predictor contract 的配置来源。
- docs/AI_EXECUTION_PROTOCOL.md：当前仓库的 gate、角色边界、数据和提交规则。

## Appendix C：诚实限制清单

1. 事件数只有 136；不能把 2,882 eligible rows 当作 2,882 个独立阳性信息。
2. primary any-month target 可能混合 temporary disruption、administrative churn 和 persistent uninsurance。
3. overall predictive signal 弱：AUPRC 约 0.075–0.080，只是相对 prevalence 有限提升。
4. single split 的 test AUPRC 约 0.0513，不能独立解释 split luck、weak model 和 selection instability。
5. Panel 26 是 single-panel development；OOF 仍是 internal robustness。
6. 没有 temporal/multi-panel external validation，因此不能声称未来 panel 泛化。
7. race audit 只有 unsuppressed-subset partial gap；不是完整 race fairness evidence。
8. 小 subgroup suppression 会限制 TPR gap 的可估计性，不应为了完整表格而撤销规则。
9. ECE 依赖 binning；当前 equal-width implementation 不能随配置注释写成 risk deciles。
10. 一些 calibration diagnostics 是 apparent-fit；不能用小 ECE 证明完美 calibration。
11. raw/mean-1 scale 对 penalized Logistic 有 material sensitivity；weight handling 是 specification。
12. model selection 随 fold/mode 变化；不能写成 universal best model。
13. generic subgroup helper 与 pilot/OOF fixed-threshold helper 的 threshold semantics 不同。
14. survey-aware PSU bootstrap 是 uncertainty 工具，不是外部验证，也不保证小 event cell 足够稳定。
15. predictor intersection、unused variable_contracts 和 mixed nonresponse disability fill 是实现债务，应在 methods/audit 中标记。
16. mitigation track 是 exploratory、需要 protected attribute at inference，不是 paper-faithful FairBias claim。
17. aggregate artifacts、source code、tests、config 和 formal protocol 各自提供不同证据；不能用任一类替代全部。
18. 当前总边界保持 PRELIMINARY_SINGLE_PANEL_DEVELOPMENT_ONLY；Panel 27 / HC-252 保持 LOCKED。

这份限制清单不是对项目的否定。它是让你在下一步真正有能力改进设计、解释结果，并拒绝把工程完成误报成科学完成。
