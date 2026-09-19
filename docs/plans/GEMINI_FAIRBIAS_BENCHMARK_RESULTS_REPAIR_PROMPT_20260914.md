# Gemini：最新 benchmark 审计响应与分阶段修复交接

这是针对最新运行成果的交接，不是要求一次重跑所有真实数据。FairBias 保持主研究方法，但不得预设其必然胜出。先完成本轮静态审计响应，再按下列阶段逐一接受 Codex 审查。

工作目录：`/Users/lkc/Downloads/code_v_0_3`。

先读以下文件：

- `/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md`
- `/Users/lkc/Downloads/code_v_0_3/AGENTS.md` 和 `/Users/lkc/Downloads/code_v_0_3/GEMINI.md`
- `/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914.md`
- `/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914_evidence/verification.json`
- `/Users/lkc/Downloads/code_v_0_3/docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md`
- 旧 C01–C06/E01 的综合审查和第四轮审查；新包装不能替代底层缺陷修复。

## 本次立即执行：RESULTS-R0 静态审计响应

允许读取源码、测试、配置、文档和本次列明的聚合汇总；只在新的唯一 `docs/plans/fairbias_benchmark_results_repair_<UTC>_<id>/` 下新增文档和标准库静态核验材料。禁止覆盖旧计划/运行/图表，禁止项目导入、pytest、模型训练、个体数据读取、网络、依赖安装和 Git stage/commit/push。此阶段不修改实现。

1. 核验当前 branch、HEAD、dirty文件hash，建立本轮输入manifest。逐项响应监督报告 F01–F14：承认、纠正或给出源码反证；不能用19 passed或“用户批准过”代替方法学证据。
2. 新增 `RESULT_STATUS_ADDENDUM.md`：明确 `sequential_full_benchmark_20260914_103657Z` 为探索性记录，撤回其“全局Pareto最优”“高保真BM”“独立盲测”“临床违法/合规保证”“全量LFR学习”等不成立声明。保留原始结果，勘误通过新文档表达。
3. 从当前代码与JSON自动导出方法/参数/训练组件样本政策/输出类型/状态表、F/C/S/T人数、BA/EO有方向的差值和测试函数清单。不得读取原始数据重算分数；不得填造缺失的df、B_eff、版本、hash或命令。
4. 写 `REPAIR_IMPLEMENTATION_SPEC.md`，落实下面R1–R4的文件范围、依赖、验收反例、输入输出、明确命令、解释器与访问guard。R1优先修真实性与口径，不得先完善美化图表。所有正式运行必须唯一run_id、失败不覆盖、保存完整错误。
5. 输出完整协议第9节报告。当前测试状态为 `NOT RUN — static audit response only`。停止等Codex审查；后续阶段在对应规格接受和激活后执行。

## R1：算法身份、p/q与公共契约（后续合成实现阶段）

主要实施范围为 `src/nhis_fairbias/benchmark/adapters/`、`metrics.py`、`preprocessing.py`、`data_contracts.py`、对应合成测试；需要底层FairBias改动时，逐项关联旧C01–C06/E01并列出最小源文件范围，经Codex确认为同一阶段范围，不能自己假装旧问题已通过。

- 主 `FairBiasAdapter` 必须调用真实BM引擎，在F学习：语义特征、真实H=1排除上下文、正确幂序列与restart、NMI gate、明确epsilon来源/倍率、MDS策略和有预算的停止。记录实际调用、有效参数及状态。`DEFAULT_POLY_GRID`是AE的六值网格，不能与BM作者幂流混用。多组公式和生效路线必须核验，不能用外部EO替代d_phi。
- 删除主路径的“读取D6/硬编码字典即可当fit成功”行为；若需历史迁移对照，独立命名且默认不进入主矩阵。未知arm不得降级成均值差幂启发式。数组路径要真实一致或明确不支持；禁止生成字典后仍训练原始X。
- 允许合法identity状态，记录“无需变换”或“无候选”等原因；不要求一定提高公平指标才能算软件正确。变换条目数不是有效变换证据。检验实际语义值/类型/行身份变化及终局状态。
- 拆分p、q、yhat；确定性概率模型只在C按冻结统一规则选阈值；EG/TO保留随机策略q。BA/EO/DP共享决策口径；AUROC/AP/Brier/校准仅用事件p，分别显示不可用指标。
- 全组可估计性、正类、概率形状及范围、有限非负权重、溢出、缺组、单类、缺失、非法年度、重复/部分重叠等应有正反例。完整年度主PSU映射先建后筛，不能按臂重抽；缺稳定ID则拒绝无来源位置ID。
- 预处理先做来源定义的语义清理，再F拟合。缺失与UNKNOWN保留位、结构性缺失、数值全缺失规则、变换后类型、外推范围都不能靠astype(str)或填0糊过去。
- 修复TO统一fit的sample_weight关键字不匹配；生产两阶段接口明确F和C，禁止统一fit偷偷用同一集合校准；EG不支持的外部权重显式拒绝。真实ValueError不一律伪装成组不支持。

必要验收：真正BM被调用且输入仅F；禁止访问D6时主方法仍可运行；小例展示p=(.9,.1)与yhat=(1,0)的BA区别；7组缺1组主EO为NOT_ESTIMABLE；identity是合法状态但未应用变换不能假成功；修改S/T不会改变F/C状态；统一fit接口/两阶段接口均验证。使用独立进程guard且先于项目/pytest导入，源脚本与用例哈希入账；不在此阶段接触真实数据或安装依赖。

## R2：调查推断、选优与可审计runner（后续合成实现阶段）

- 以完整年度设计+domain mask实现域估计；singleton/df≤0按预登记策略处理，不给零方差或强行df=1。率/非光滑gap及配对差使用同一复制权重；有效复制门槛95%，保留B/B_eff/SE/df/status。不按T显著性选择区间。
- 正式B=2000，调试B小值必须在产物标记DEBUG而不能作为正式结果；预登记20个主对比，固定校正家族。零SE等退化状态显式处理，不给误导p=1；同时输出效应方向、未校正/调整区间。
- 选优按method×arm×backbone在S逐配置进行；不选择最优seed，失败种子使配置不完整。强预测参考按AP无公平约束选择。主操作点EO≤.10，.05/.20敏感性；外部预算与几何epsilon不同。
- 保存F模型、C阈值、S候选ledger/选优结果、来源与预处理hash、有效版本/参数/seed、资源/失败、设计诊断；T阶段无fit或选优。源码、状态与输入绑定才可追溯旧结果。
- 统一两个runner和demo的调用路线，防止旧数组路径继续冒充FairBias。标准JSON禁止NaN/Infinity；输出目录已存在即拒绝，模型与图表不可覆盖。分批生成复制权重，实际测量内存/墙钟；del+gc不是资源上限。

必要验收：手算率/设计例、独立参考方差容限、缺组复制、singleton、极端权重、零SE、20家族计数、失败种子、空候选、S只选各方法配置、T禁止fit、模型序列化、文件禁止覆盖、全进程访问审计。原19测试不能独自充当验收。

## R3：完整模型与数据适配冻结（合成矩阵通过后才计划真实开发）

- 主预测器LR，次预测器固定 `sklearn.ensemble.GradientBoostingClassifier`，按master网格；所有适配器用相同明确骨干接口。LR不能在文档写GBDT但实现固定LR。
- 主对照RW、LFR、EG-DP、EG-EO、TO-EO，加无干预预测参考。各适配器逐项核验fit/predict的y/A/权重/语义输入/输出类型以及具体库版本和原文来源。
- LFR恢复预登记预算及适当调参；2,000行短预算只允许单列的资源敏感性条件，记录表示学习和分类器各自训练样本量。LFR多组不支持是本实现边界。EG分开difference_bound、eps、max_iter、实际oracle_fit计数；TO只能在C拟合后处理。
- 恢复54核心+16AE+4调查权重训练+2Arm004几何路径，列出全部condition_id、config_id、expected_seed_ids和适配矩阵。AE和Joint使用同一配置的epsilon_candidate，不能退回未乘倍率epsilon_ref；strict slack=0与搜索接受不等式逐一明示。
- 小规模全矩阵与独立资源压力例先通过，再用2022/2023做支持度审计及开发。原始数据精确路径/hash/字段和分区规格由该阶段明确授权；当前Prompt不授权读取。

## R4：冻结后评价与论文（最后执行）

仅在前述阶段独立验收后，冻结配置、种子与模型，依协议评价2024。如实披露2024此前多次被看过和用于调试；不能重称未观察的独立测试集。新独立验证另行确定数据可比性和授权。

各臂同时给BA/EO/DP绝对差及区间、AP/AUROC/Brier/校准、组TPR/FPR/覆盖、失败率、耗时和训练种子差异。FairBias为主方法但允许RW或其他方法领先；不要更换主臂、主指标或删失败种子来制造优势。若主张效用保持，须在查看新评价结果前定义可接受损失及统计标准。

Arm004是去除6个残障分量的同人群特征消融。几何路径比较固定其他因素，保留首个分歧、候选、epsilon与状态证据。没有机制证据只描述相关差异。论文定位为医疗可及性障碍识别的应用研究，不把重复横断面调查写成临床纵向预警队列，也不从预测期A依赖推出法律结论或顶刊保证。

## 交付及停止点

本次RESULTS-R0交付：`RESULT_STATUS_ADDENDUM.md`、`REPAIR_IMPLEMENTATION_SPEC.md`、`FINDING_RESPONSE.md`、完整静态命令/自动hash清单、`WORKER_REPORT.md`。后续实现规格必须具体到文件、测试入口、解释器、guard、证据目录和退出条件；当前只规划后续，不写“已实施/已验收”。

最终报告精确使用以下标题：

```text
Gate:
Status:
Files changed:
Commands executed:
Permissions requested:
Tests executed:
Exact test results:
Input hashes:
Output hashes:
Row counts:
Assumptions:
Unresolved issues:
Git diff summary:
Proposed next step:
STOP — waiting for Codex review.
```

只执行本次RESULTS-R0并停止；不得直接启动R1或继续刷2024结果。普通文档和字段决定按master与监督报告处理，不需要反复询问用户；确实缺来源则明确标记待核验，完成其他已授权工作。
