# FairBias 应用比较：Codex 接手验收与执行状态

2026-09-16，Codex Supervisor。用户已授权完整推进和阶段验收，不再逐轮移交 Gemini 或请求确认。**新应用 benchmark 已开始完整 F/C/S 计算；本报告不是完成的论文结果，也不宣称 FairBias 优于其他方法。**

当前研究分支为 `research/nhis-fairbias`，HEAD 仍为 `67e6659fa65249a8842e34af5d8969629efe4bca`，实际执行对象包含本地未提交修复。没有提交、推送或改动14个继承文件及 `.gitignore`。完整训练登记在 [registration.json](/Users/lkc/Downloads/code_v_0_3/artifacts/nhis/benchmark_codex_20260916_014500/registration.json)，69项训练来源保持一致，source identity 为 `1054c31d2764076e2f8f70aed6c47786a5a63266c9ad45b1e7ba7ae9d0fce0d2`。

## 1. 阶段决定

| 阶段 | 决定与证据边界 |
|---|---|
| R12交付 | 不接受旧封存器为通用验收工具。独立核实3733项清单和上游22份证据后接收限定的R10软件证据；实际51个CLI情形中49符合、2个缺口保留。见 [R12决定](fairbias_codex_takeover_20260915_154733Z/R12_DECISION.md)。没有把旧封存器的失败改写成通过。 |
| 核心与数据适配 | 接受新的应用路径用于本轮实验。F学习表示/编码、C校准决策、S选择完整配置，T只评价冻结模型；真实FairBias已有非恒等学习和保存后预测见证。 |
| 近期方法 | FairGBM、FRAPPÉ、OxonFair、fairret均完成实际本地依赖和真实F/C/S首批拟合；TabM紧凑预测器也已跑通。仅代表接口/数值准入，全部配置和种子仍须执行。 |
| 调查统计 | 接受完整年度PSU域估计、配对BA线性化与保守gap投影用于有明确限制的应用推断。R `survey` 独立对照和合成覆盖实验通过相应检查，不能外推为所有稀有组上的精确覆盖保证。 |
| 正式实验 | 已登记1922个配置、8054个种子任务（含4个明确不支持的配置条目，不对应拟合任务）。完整开发进程串行运行；表示缓存已实际复用。 |
| 冻结评价与论文 | 实现和合成验证已接通；只有全部登记任务结束且证据核验通过才允许2024评价。最终比较数值、图和解释尚未产生。 |

## 2. 本轮解决的主要问题

| 严重性 | 问题及处理 | 具体实现 |
|---|---|---|
| P1 | 预测对象混淆：严格分开事件概率p、决策概率q和硬决策。AP/AUROC/Brier仅用p，TO/OxonFair原始风险单列，禁止用q伪装风险改进。 | [predictions.py](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/predictions.py)、[risk_metrics.py](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/risk_metrics.py) |
| P1 | 公平表示与预测器脱节/泄漏：每个表示只在F学习，拟合端和查询端重用同一字典与编码；C阈值、S调参、T冻结彼此分开。缓存绑定F、参数、种子、源码身份和模型SHA。 | [FairBias适配器:153](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py:153)、[worker:51](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/experiment_worker.py:51)、[预处理:29](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/preprocessing.py:29) |
| P1 | 多组作者公式没有到达应用入口：BM与AE显式采用两侧各取最大组对值的 `author_max_pair`。锁定作者独立源码与冻结继承版均值分支有别，不能拿继承版证明作者公式。 | [BM配置:167](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py:167)、[AE配置:190](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_fairbias_ae.py:190) |
| P1 | 调查域过滤导致零贡献PSU丢失：保留完整年度设计、域外贡献置零；WTFA人口指标与训练目标分开。主训练无调查权重，另列下游预测器WTFA训练敏感性。 | [data_contracts.py:119](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/data_contracts.py:119)、[survey_linearization.py:116](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/survey_linearization.py:116)、[survey_batch.py](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/survey_batch.py) |
| P1 | 普通极差bootstrap不能直接保证EO差值覆盖：BA使用Taylor设计区间；EO/DP使用所有冻结模型×率坐标的同时区间投影，bootstrap仅作描述性敏感性。20项主要家族固定，不因方法失败缩小分母。 | [线性化:132](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/survey_linearization.py:132)、[冻结评价:125](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/experiment_evaluation.py:125) |
| P1 | 缺少种子、超时、不可估或不可行配置可能冒充赢家：必须完成全部已登记尝试并核对receipt；完整seed均值选择，失败seed不删掉后求最佳结果；无可行结果另标边界。 | [配置选择:14](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/experiment_selection.py:14)、[冻结入口:42](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/experiment_selection.py:42) |
| P1/P2 | OxonFair 0.3忽略grid_width，七组实际超内存；其lambda又不能普通pickle保存。限定作用域转接让官方grid_search收到真实steps，finally恢复；保存采用可复现的序列化。修复后64项LR配置含七组均通过开发准入。 | [OxonFair适配器](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_oxonfair.py) |
| P2 | nullable整数列无法填入20.5等中位数；有限权重合计可能溢出；MDS没有收敛证据。数值列先转float，比例先归一化；逐次保存MDS stress/维数/迭代，达到上限拒绝当作成功。 | [数值准入](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/geometry_audit.py)、[权重指标](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/metrics.py) |
| P2 | Shapley反复pandas切片耗时；源码字符串测试容易随包装函数变化失效。保留同运算顺序的NumPy缓存并与原实现240例逐项比较；BM/AE顺序改为实际入口调用记录测试。 | [bias_metric.py](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py)、[AE实际顺序契约](/Users/lkc/Downloads/code_v_0_3/tests/benchmark/test_ae_integration_evidence.py) |

旧D8合成测试的8项失败共同来自0/1性别编码与冻结NHIS 1/2定义冲突，修正测试夹具后103项通过；生产组别校验没有放宽。另一个历史探针使用0–3岁作为成人合法训练数据，修正为注册有效数值后33项通过。这些是测试前置条件修复，不是删除失败断言。旧D8的部分joint失败分支仍缺少原始exception详情，列为兼容路径诊断改进，不在新实验中使用。

## 3. 验证与实际资源

- 新benchmark、TabM及几何公式定向套件：**161 passed，44.90秒**，完整日志和返回码见 [benchmark_tests.log](fairbias_codex_integration_20260916/benchmark_tests.log) 与 [receipt](fairbias_codex_integration_20260916/benchmark_tests_receipt.json)。后加完成链入口的2项测试实际证明：登记seed未完成时不能进入T；批准文件变化时不能启动后续进程。后续聚合/完成链定向10项也通过。
- 旧D8、AE契约、survey定向套件：**103 passed，49.50秒**；历史审查探针：**33 passed，0.93秒**。它们不是全仓库所有测试，测试数量也不等于统计有效性。
- R4.6.1 + survey4.5：48行完整合成设计的ratio协方差最大差约 `6.51e-19`，配对BA点估计/SE与Python匹配。见 [R对照](fairbias_codex_takeover_20260915_154733Z/luna_survey_linearization_report.md)。稀有组/小自由度模拟仍可能不可估或低覆盖，不能冒称精确保证。
- 八类既有冻结模型在统一TensorFlow环境按S前32行复核：q完全一致；两类float32模型p最大差约`2.98e-8`，其余为零。见 [运行时核验](fairbias_codex_takeover_20260915_154733Z/luna_frozen_prediction_runtime_admission_report.md)。这些是2022/2023开发证据，不是2024结果。
- 单个重型拟合串行、4 GiB、30分钟；普通LR及复用表示任务约2秒，实际重新学习FairBias表示约170–190秒。开发准入中的FRAPPÉ约65秒、BM→AE约214秒、Joint约530秒。240例矩阵构造中位加速约3.7倍，不代表整体BM同倍加速。
- 开发预备批次中的序列化、数值类型、OxonFair内存错误及源码变更停止记录全部保留，未拼接成正式结果。完整运行预计跨越多小时甚至更长，不能用此前单轮代码修补的10–40分钟估计全实验时长。

## 4. 比较矩阵与数据含义

| 方法 | 配置条目 | 种子任务 | 范围 |
|---|---:|---:|---|
| 普通预测器 | 88 | 312 | LR、GBDT、匹配MLP、FairGBM同源普通预测器、紧凑TabM；含登记WTFA敏感性 |
| FairBias-BM主方法 | 704 | 3520 | 匹配骨干、八点ε比例；有合法表示复用 |
| Reweighing | 32 | 96 | LR/GBDT |
| LFR重构表示 | 194 | 960 | 七组两条明确不支持，无隐蔽2000人截断 |
| EG-DP / EG-EO | 各256 | 各768 | LR/GBDT；原生q |
| ThresholdOptimizer-EO | 32 | 96 | F基础预测器、C后处理 |
| OxonFair-EO (2024) | 128 | 384 | 二组grid6，七组grid2，实际传参核验 |
| fairret-EO (2024) | 16 | 80 | 匹配MLP |
| FairGBM (2023) | 96 | 480 | 六点FPR/FNR共同slack，不冒充同值EO上限 |
| FRAPPÉ (2024) | 98 | 480 | 二组；七组两条明确不支持 |
| BM→AE / Joint | 各8 | 各40 | 固定容量和ε的阶段消融，不称全面调参 |
| Arm004几何三条件 | 各2 | 各10 | elbow、二维同ε、二维自身参考ε |

近期方法与现代预测器来源见 [能力卡](fairbias_codex_takeover_20260915_154733Z/RECENT_METHOD_CAPABILITY_CARDS.md)。TabM为2025论文的紧凑CPU适配；TabICL-v1和论文默认容量TabM不在本轮已运行家族，未伪装完成。

2022按完整PSU分配F/C，2023选择S，2024冻结评价T。Arm001性别二组，Arm002种族/族裔七组，Arm003残疾二组21特征，Arm004与Arm003同人群、移除六构成特征。结局为过去12个月因费用延迟就医的自报状态；这是重复横断面的回顾性年度迁移评价，不是个体未来风险或临床干预效果。2024此前已被历史研究观察，不称研究全程盲测。

人群、分母、设计PSU和语义缺失等开发汇总已生成：[F/C/S人群表](fairbias_codex_integration_20260916/cohort_FCS.json)、[年度纳入条件](fairbias_codex_integration_20260916/eligibility_FCS.json)。F/C的权重总和是分区描述，不是未经调整的整年人口总量估计。最终T会独立报告完整年度设计与可观察Y/A亚总体；WTFA不会自动消除item nonresponse或自报误差。

## 5. 自动衔接与剩余事项

本轮完成链是一次本地执行，按用户授权由Codex预先验收代码与条件：全部F/C/S任务结束 → 核实全部receipt和seed → 冻结选择与分析 → 固定2024预测/调查推断 → CSV/Markdown与PNG/PDF。代码、依赖版本或证据改变，或任务未完成，会在继续前失败停止。不会按T排名补方法、改阈值或重选种子。

当前状态以 [live_status.json](/Users/lkc/Downloads/code_v_0_3/artifacts/nhis/benchmark_codex_20260916_014500/live_status.json) 为准；完整记录见同目录 `completion_events.jsonl`。只有出现 `completion_result.json` 的 `COMPLETED_FIXED_STUDY` 才表示该固定研究计算完成。生成的结果稿仍须按数值内容解释，不能从流程完成推导FairBias胜出或文章可录用。

剩余科学边界明确保留：主方法是有名称的应用适配而非所有论文细节逐行复现；非凸几何/贪心搜索不保证全局最优；风险指标暂不提供设计区间者只报告点估计及seed变异；EO投影较保守；稀有组可估性、失败率和跨年变化必须进入论文；Arm004路径差异需看固定配置的实际轨迹，不能只凭终点差异归因于几何。未来增加独立新年度/外部医疗队列属于另行登记的研究扩展。
