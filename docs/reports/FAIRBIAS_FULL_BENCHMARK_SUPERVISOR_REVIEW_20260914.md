# FairBias 最新全量 benchmark 独立审查（2026-09-14）

**结论：当前结果不能作为正式方法优越性或确认性统计证据；保留为探索性运行记录。实现需要 REPAIR，不能据此验收 B1–B6。** 这不意味着 FairBias 方法已被证明无效，而是当前程序没有完成所声称的算法比较。

审查对象：分支 `research/nhis-fairbias`，HEAD `67e6659fa65249a8842e34af5d8969629efe4bca` 下的未提交 benchmark 实现，以及运行 `sequential_full_benchmark_20260914_103657Z`。读取了源码、测试、既有聚合来源记录和用户提供的执行文本；没有运行 pytest、导入项目、训练模型或打开个体微数据。结果数字是对既有 JSON 的独立核算，不是重新跑出的性能。公开 API 文档仅用于核对定义，不代表核验了本次运行的依赖版本。

可复查证据：[静态核验结果](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914_evidence/verification.json)、[源码及汇总快照](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914_evidence/source_snapshot)、[用户提交记录](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914_evidence/worker_submission.txt)。证据快照固定本次审查边界；工作区之后的变动需另外核验。

## 已经形成的成果与边界

- 已有七类方法的适配器框架；接入了语义表格、F 拟合预测编码、调查加权率、共用复制权重配对比较等模块。这些结构可以继续使用。
- 汇总 JSON 有 **27 个 VALID 方法—臂条目**，另有一个 LFR 多组 `NOT_SUPPORTED` 条目。不是 76 条件完整计划，更不是所有 grid×seed 拟合已经完成。
- 静态解析发现 **19 个测试函数**。用户记录称测试通过，本轮没有重跑，不能将静态计数表述成独立验证的 `19 passed`。
- 先前审查的 15 个候选文件 hash 全部保持一致，意味着旧的未闭环问题并未因新增 benchmark 自动解决。暂存区为空。
- 14 个继承文件与保护 tag 的 Git 比较无差异；`.gitignore` 与保护 tag 有历史提交差异，但与当前 HEAD 无差异。该差异最近对应 `b595e59`，不能归责本轮，也不能声明其仍与原 tag 完全一致。

## P0：会使主研究结论失效的问题

### F01 — 被命名为 FairBias 的主路径没有执行 FairBias BM 学习

位置：[adapter_fairbias.py:124](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py:124)、[adapter_fairbias.py:195](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py:195)、[实际 runner:116](/Users/lkc/Downloads/code_v_0_3/scripts/run_sequential_full_benchmark.py:116)。

实际 runner 指定 `arm_id`；适配器随即加载历史 D6 `frozen_changed_dict.json`，缺文件则使用源码内硬编码字典。这个分支直接返回，与当前 F 的 X/y/A、epsilon、MDS、NMI 和搜索预算没有拟合关系。真正执行的是“历史变换规则 + 新 LR”，不能称为本次 F 上学习的 FairBias-BM。

未指定 arm 的分支也不是 BM：它按各组数值均值差排序，尝试 `(0.5, 2, 0.33, 3)`，使用另设的 0.8 NMI 比例，没有流形几何和应有的候选规则。[adapter_fairbias.py:150](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py:150)。更严重的是数组路径虽然生成字典，随后仍直接 `clf.fit(X,y)`，推断也原样返回 X，根本不应用字典。[adapter_fairbias.py:214](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py:214)。通用 [BenchmarkRunner:101](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/runner.py:101) 仍走这条数组路径。

**修复要求**：主方法必须通过真实 BM 引擎仅在 F 学习并保存路径、有效参数、epsilon、输入 fingerprint 和退出状态。D6 规则若保留，单列为 `FROZEN_D6_TRANSFER` 历史迁移对照；不计入主方法结果。不得通过“必须非空 changed_dict”强迫制造改动：本来已经满足条件时，identity/no-op 可以是合法结果；非空字典也可能全部映射到不存在的列或没有改变任何取值。

### F02 — 混用事件概率 p 与决策概率 q，BA/EO/DP 比较的实际对象不同

位置：[base.py:47](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/base.py:47)、[unmitigated:54](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_unmitigated.py:54)、[FairBias:260](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py:260)、[metrics.py:50](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/metrics.py:50)、[runner:225](/Users/lkc/Downloads/code_v_0_3/scripts/run_sequential_full_benchmark.py:225)。

Unmitigated、RW、LFR 和 FairBias 把 `predict_proba[:,1]` 原样作为 q；EO/BA 计算使用 `sum(w*y*q)`。EG 和 TO 则提供真实随机策略的正决策概率。这样普通 LR 实际被当作 Bernoulli(p) 随机决策器评价，而不是按 C 上冻结阈值产生决定的分类器。类型字段虽然存在，runner 没有据其分流；只有 TO 得到了 C 阈值优化。

反例：两个样本 y=(1,0)，p=(0.9,0.1)，阈值 0.5 时 BA=1；当前把 p 当 q 得到 BA=0.9。这是另一个决策策略的期望 BA，不是同一个分类器的 BA。概率拉近常数还可能使这种“软 gap”缩小。因而表中 TO 的高 BA、LFR 的接近 0.5，以及所谓偏差下降率都不能按原叙事解释。[scikit-learn BA 定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) 明确使用预测类别计算各类召回率。

**修复要求**：拆开 `p_event`、`q_decision`、`yhat`。概率模型在 C 按统一规则确定 t，冻结 `yhat=I(p>=t)` 后计算决策指标；EG/TO 用策略 q 的期望混淆矩阵。p 用于 AUROC、AP、Brier、校准等概率指标。禁止把 q 直接当医疗事件风险。修复后所有结果表及统计推断重算，不能只替换列名。

## P1：方法学、数据隔离与结果解释

| ID | 位置与已核实问题 | 影响及所需修改 |
|---|---|---|
| **F03 历史训练状态与新 F 不兼容** | D6 Arm 003 的 [来源记录:488](/Users/lkc/Downloads/code_v_0_3/docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_003/input_provenance.json:488) 为 2022 train_n=27,451；新汇总 F=21,871、C=5,580，两者相加正是 27,451。新适配器加载旧规则而不核验当前 F 记录身份、预处理状态、配置和字典 hash。 | 不能把旧全年度训练得到的规则当成新 F-only 规则，也不能证明它排除了新 C。本轮未逐记录复核重叠，不虚报重叠人数。主方法重学 F；历史迁移对照明确来源范围、额外信息和任务差异。已加工输入是否含预先拟合的填补值，也必须追溯，不能凭再 fit 一个 scaler 证明全流程 F-only。 |
| **F04 T 已参与开发诊断，没有冻结再评价的证据链** | [提交记录:50](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914_evidence/worker_submission.txt:50) 开始多次使用 T 比较 D6/D5 规则，之后修改适配器，再执行 [T 冒烟和全量:539](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914_evidence/worker_submission.txt:539)。runner 在 S 选择之前已经加载、变换并预测 T：[runner:230](/Users/lkc/Downloads/code_v_0_3/scripts/run_sequential_full_benchmark.py:230)。 | 仅提前计算 T 预测不自动等于标签泄漏；但用户提供的开发记录确实包含先查看 T 指标、再改管线的路径。本次不能称“独立盲测”。如实登记为回顾性探索；之后冻结模型、代码、阈值、选择规则再评价，仍须披露旧 T 已被观察。新样本/新年份的独立验证另立方案。 |
| **F05 对照欠调优且实际预算不等** | [LFR:108](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_lfr.py:108) 在 F>2,000 时抽 2,000 行学表示，maxiter=50/maxfun=100；下游 LR 才用全部 F。[runner:111](/Users/lkc/Downloads/code_v_0_3/scripts/run_sequential_full_benchmark.py:111) 只用 C=1、单种子42、EG max_iter=15/eps=.05、LFR k=5/Az=50。 | 不是完整 grid×seed 比较，也不是全部方法均在相同全量 F 上学习全部组件。不能把有限配置/短预算结果解释为方法理论失败。恢复预登记调参预算；有限子样本 LFR 若保留必须独立命名、记录抽样、优化退出与对照预算。EG 的 difference_bound 和优化 eps 需要分开设置。 |
| **F06 S 选择对象错误，完整实验计划未执行** | [runner:271](/Users/lkc/Downloads/code_v_0_3/scripts/run_sequential_full_benchmark.py:271) 把“方法名→一份预测”一起选全局 winner；没有每方法参数搜索。选择结果仅打印；随后所有方法直接评价。没有双预测器、五种子、AE、权重训练和几何路径矩阵，也没有 AP/AUROC/Brier 输出。 | 应为每个方法×臂×预测器，先按完整种子聚合配置，再于 S 选优并保存。不要把方法间 winner 当超参数选择。恢复 54核心+16AE+4加权训练+2路径的条件登记；这些是条件数，内部还要展开配置和种子。当前只能称 LR 默认配置探索。 |
| **F07 缺失组被从主 gap 悄悄删掉** | [metrics.py:111](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/metrics.py:111) 只将非 NaN 率放入列表，最后仅要求列表长度≥2（132、137行）。 | expected_groups=7 而只剩2组，仍可输出有限 EO/DP；TPR 与 FPR 还可能分别来自不同组集。原样本和 bootstrap 都受影响。必须逐指标验证完整 expected_groups 的相应分母；DP 要组权重和，EOpp 要组阳性权重，EO 还要组阴性权重。不足则 null+reason，observed-only 另名。补二进制标签、数组维数、概率范围、非负有限权重及求和溢出契约。 |
| **F08 调查设计不可估计时仍出区间** | [survey_inference.py:92](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/survey_inference.py:92) 将 df 强行设≥1；105–107行 singleton 保持原权重，相当于不给该层复制变异；[173–179行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/survey_inference.py:173) 两次有效复制就出区间。runner 只传筛选后的臂设计。 | 完整年度设计与分析域应分开保存；不应因域筛选丢 PSU 或把缺失 PSU 当不存在。原始 singleton/df≤0 按预登记规则返回 DESIGN_NOT_ESTIMABLE；不能静默给零方差。正式区间须达到有效复制比例门槛（master 为95%）并输出实际 B_eff/df/状态。七组缺失率问题必须先修复。 |
| **F09 B=30、未校正推断与丢失的 df 不能支持确认性显著性** | [runner:73](/Users/lkc/Downloads/code_v_0_3/scripts/run_sequential_full_benchmark.py:73) 默认 B=30；实际命令也是30。survey 只用 alpha=.05；当前按所有方法、所有臂比较，未实现预登记20对比校正。[runner:352](/Users/lkc/Downloads/code_v_0_3/scripts/run_sequential_full_benchmark.py:352) 导出时丢弃 df、SE、B_eff；JSON 不含 worker 声称的 df=20。 | B=30 可以是调试预算，不符合已规划正式 B=2000；不能根据漂亮 p 值事后接受低复制数。核验复制构造、域分析、非光滑 gap 的适用性，记录独立参考及容限。固定20对比家族、同时给未调整与调整区间。df=20 在图文中被写死，本轮无法从汇总 JSON 证实；也不据此断言内部真实 df 就是20。 |
| **F10 身份和 PSU 主映射尚未满足零泄漏契约** | [data_contracts.py:144](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/data_contracts.py:144) 直接 int(year)，接受2023.9截断和bool；[323–331行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/data_contracts.py:323) 先按臂过滤再抽主 PSU；[387–388行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/data_contracts.py:387) 缺 HHX 就用筛选后位置造 ID。 | 缺组造成 PSU 集变化时，共同 PSU 对应的随机数序位会改变，不保证跨臂共享一次分区。应从全年完整设计一次生成映射、再套臂掩码；ID 要来源定义及规范化，不用位置替代。补重复/部分重叠、重排、非法年度、数组齐长、不可变状态校验。dataclass frozen 不会冻结其内部 ndarray/DataFrame。 |

F08/F09 并不否定代码里正确的部分：目前确实共用复制权重、每次重算 gap、用配对复制差和 ddof=1；这些可以保留。负的 gap 下界是未约束 Wald 区间的现象，不是单独证明 bootstrap 错误，也不能靠事后截为0解决其他统计缺陷。零复制 SE 时却将 t_stat 设0、p=1（[survey:218](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/survey_inference.py:218)）也应返回明确退化状态；极端尾概率用 sf，避免 `1-cdf` 下溢成0。

## F11（P1）：当前学术叙事与其自身数字相矛盾

下面只复核现有[汇总 JSON](/Users/lkc/Downloads/code_v_0_3/runs/sequential_full_benchmark_20260914_103657Z/integrated_full_benchmark_summary.json)；因 F01/F02，不能据此给出有效方法排名。

| 臂 | FairBias BA | RW BA | FairBias EO | RW EO | 当前点估计关系 |
|---|---:|---:|---:|---:|---|
| 001 性别 | 0.52335 | 0.54820 | 0.01530 | 0.00473 | RW 同时更高 BA、更低 EO |
| 002 七组 | 0.53208 | 0.54605 | 0.09498 | 0.09445 | RW 同时更高 BA、更低 EO |
| 003 残障含分量 | 0.54386 | 0.54618 | 0.02666 | 0.00654 | RW 同时更高 BA、更低 EO |
| 004 残障去分量 | 0.52954 | 0.53939 | 0.02425 | 0.00436 | RW 同时更高 BA、更低 EO |

因此“全局 Pareto 最优”即使按当前 BA/EO 两轴也不成立。RW 也不需要预测期 A；仅强调这个特性无法排除 RW 的比较。Arm 004 若额外重视 DP，应明确多目标取舍，不能把该单项优势改写为全局最优。

- Arm 003 相对 baseline 的 EO 下降57.0%算术成立，但它不等于优于其他公平性方法，也不等于消除57%的实际医疗不平等。BA 从0.54816降到0.54386，下降0.00429；“效用保持”需事先定义可接受损失或非劣效界限。
- Arm 001 的 EO 相对 baseline 上升约375%，BA下降0.02472；Arm 004 的 EO上升约79.8%，BA下降0.01055。原叙事分别改写成“绝对值极小”和“DP骤降”，属于指标选择不对称。应所有臂统一呈现绝对差、方向和区间；极小基线不宜主要用相对百分数。
- Arm 003 对 RW 的当前 delta_ba=-0.002315，delta_eo=+0.020121。这里 p=.0011 对应 EO 差异，方向是 RW 的 gap 更小；一列不标效应方向的“vs FB p值”容易被误读。
- 89,091 是三个年份的合计，不是2024测试人数。各臂 T 为32,350、32,355、32,354、32,354；文章标题/图题应分别列 F/C/S/T，而非给 T 表挂8.9万样本。
- 本代码 EO 是 equalized odds（同时涉及TPR/FPR），不是只看TPR的 equal opportunity。NHIS 年度重复横断面调查也不是患者纵向临床队列；任务是识别既往12个月成本相关延迟就医，不是已验证的临床未来事件预警。
- “TO 多组指标变差是小组校准过拟合造成”目前只是机制假说，需要 C/S/T 组支持、阈值、分布变化及对照实验证据。LFR `NOT_SUPPORTED` 是本次适配范围，不等于算法运行崩溃，更不等于多组公平在理论上不可能。
- [ThresholdOptimizer 官方接口](https://fairlearn.org/main/api_reference/generated/fairlearn.postprocessing.ThresholdOptimizer.html) 确实需要预测期 sensitive_features；这只支持部署依赖说明。代码没有法域、用途或法律评估证据，不能推出“违法差别待遇”“临床合规性为零”，也不能推出 FairBias 自动符合法律。
- “首次揭示”“顶刊级别”“信息论门控保护了本次效用”等结论应撤下：本次主路径没有运行 NMI gate，亦没有相应文献优先权、临床效用或投稿判断证据。

## P2：代码与工程问题

| ID | 证据 | 所需处理 |
|---|---|---|
| **F12 语义/输入契约不足** | [preprocessing.py:54](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/preprocessing.py:54) 将全缺失数值填0；[71行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/preprocessing.py:71) 先 astype(str)，使 NaN/None 成为 `nan`/`None` 普通类别；未按字段字典清理特殊缺失，类型按固定列名猜测。 | 按字段、来源阶段区分原始码、结构性缺失、显式缺失与 UNKNOWN，保留原语义；F-only 全缺失规则与主计划一致。变换输出类型必须随状态更新。新读入原始数据与读已清理数据需要不同 schema，不能双重处理或漏处理。 |
| **F13 隐藏 API Bug 与测试过浅** | [TO fit:91](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_threshold_optimizer.py:91) 调用 `fit_base(..., sample_weight=...)`，实际参数叫 `sample_weight_F`，统一 fit 会 TypeError；现有测试直接调 fit_base/calibrate，未覆盖。EG 接收 sample_weight 却不使用；TO 将所有 ValueError 都改称组支持不足。FairBias 合成[测试:134](/Users/lkc/Downloads/code_v_0_3/tests/benchmark/test_adapters.py:134) 用真实 arm 自动读历史规则，断言“非空字典”即可通过。 | 修正参数；不支持的权重明确拒绝；保留异常类别和原因。测试要验证调用了真实 BM、无历史读取、概率语义、完整组、非法输入、状态冻结、独立数组/语义路径一致性和真实参数生效。19个通过不能覆盖这些条件，也不能替代旧核心与独立反例回归。当前新增语义测试不是完全无历史文件依赖。 |
| **F14 产物与运行边界未兑现** | [runner:145](/Users/lkc/Downloads/code_v_0_3/scripts/run_sequential_full_benchmark.py:145) exist_ok=True，437行覆盖JSON；实测运行目录只含汇总JSON，没有模型、配置、数据来源hash、阈值、S候选表、seed账本和设计诊断。JSON有8个NaN字面量。fb_adapter/to_adapter额外引用在 del adapter 后仍保留。 | 独立run_id且拒绝覆盖，保存不可变模型/配置/来源/设计/选择证据；JSON必须标准null+reason。内存和壁钟要实际测量及限额，GC不是<500MB保证。B=2000、T=32,354时，仅dense复制权重就约517.7MB（493.7MiB），应分批统计。运行成功与可估计/选优/论文可用分别编码，不能全部赋VALID。 |

## P3：以 FairBias 为主方法的应用论文路线

主方法地位可以通过研究问题、详细机制分析、消融与复现深度体现，不需要预设所有指标都领先。建议研究定位为“FairBias 在美国健康调查中识别成本相关医疗可及性障碍的公平性—预测效用权衡及跨年稳定性”。

1. **先修算法真实性与指标语义**：F01/F02 是最先验收的阻塞项。用小型、可手算或可追踪的合成例验证，不查看更多 T 结果来决定修法。完成旧 C01–C06/E01 的依赖核验，不能靠新包装绕过问题。
2. **再冻结数据与统一对照**：稳定ID、完整设计主表、F/C/S/T、字段字典、同骨干、同C阈值规则、各方法适当参数网格与同等计算边界。LR先完成，再加固定的 GradientBoostingClassifier；RW/LFR/EG-DP/EG-EO/TO-EO全部有明确权重、组别、输出接口和来源版本。
3. **完成合成矩阵后做2022/2023开发**：每方法单独调参，多种子不挑优。保存 S 上 BA–EO权衡、预测性能、失败率、耗时。先验登记54核心+16AE+4加权训练+2几何路径；分批跑，但不把未跑条件写成完成。
4. **锁定后做回顾性2024评价**：B=2000为拟定正式预算，设计推断先验收；20主对比固定，报告缺失/失败。充分披露2024已经参与历史与本次开发。若追求独立验证，另选未观察数据并先核验可比性，不能宣称重新划一次分区就恢复独立性。
5. **论文同时报告优势和代价**：主方法与最强预测基线、RW等比较；主要给绝对差与不确定性，补AP/AUROC/Brier、组TPR/FPR/覆盖率。Arm004严格视为特征消融；路径差异需首个分歧步骤、几何、候选和固定epsilon证据，不能只看终点归因。

**当前可用的结论**：“已构建多方法探索性基准接口，并发现历史规则迁移、概率接口及调查推断中的关键实现问题；现有性能表需要在修复和冻结协议后重新生成。”不对尚未进行的正式实验预判胜负。

本次只新增监督文档与证据，未修改研究实现、未暂存或提交。当前记录中的审批历史不完整，本报告不推断 worker 是否另获用户运行许可；科学有效性按可见代码与证据独立判断。先前 B0-R1 静态检查中途被最新任务接续，其未完成目录有显式 STATUS，不作为任何 gate 的通过证据。
