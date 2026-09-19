# FairBias Benchmark B0 独立验收

日期：2026-09-14。监督结论：**REPAIR — B0 需要一次集中文档修订，尚未进入 B1。**

本次审查对象是 `docs/plans/fairbias_benchmark_b0_20260914/` 的六份规划文件，以及用户提交的 Gemini 执行记录。审查的是计划能否成为无歧义的实现依据，不是重新运行 FairBias 或检验实证优劣。没有确认 P0；下面按 **8 类 P1 规格/方法学问题、4 类 P2 执行与证据问题**列出有限返工清单。这些是规划缺口，不是声称已经发生十二个真实实验漏洞。

## 1. 已经独立确认的事项

- 当前分支 `research/nhis-fairbias`，HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`；无暂存文件。
- 六份 B0 文件真实存在，报告中可解析的 **28 条 SHA-256 声明全部匹配**。提交记录中的 worker report 散列也匹配：`270bc9640129bcd98ff33033b572d91633284da9e88f22057675a4f8e3225281`。
- 前次第四轮记录的 **15 个候选文件全部保持原 hash**。受保护文件相对 HEAD 无变化；`.gitignore` 相对基线标签的差异是已有历史差异，不是此次 B0 新修改。
- 四个被点名检查的解释器，其软件包 metadata 与 worker 清单一致；其中均未安装 fairlearn/aif360。不能把此结论扩大到整台机器所有可能的环境。
- 主方法定位、四臂及 003/004 同人群、F/C/S/T 年份角色、LFR 七群体不支持、p/q 区分、TO 的 C 拟合、2024 历史暴露等基本方向正确，应保留。

用户提供的记录与“只读检查并新增文档”相符。本次未发现源码变更、测试运行或真实数据训练证据；但文件散列及提交记录不能证明 worker 全进程绝无网络/数据访问，故不出具这种强于证据的认证。本次监督核验只读取文档、源码/AST、包 metadata 和 Git 状态，没有模型导入、测试执行或直接微观数据解析。

独立证据：[核验结果](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_B0_SUPERVISOR_REVIEW_20260914_evidence/verification.json)、[核验脚本](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_B0_SUPERVISOR_REVIEW_20260914_evidence/verify_documents.py)、[命令记录](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_B0_SUPERVISOR_REVIEW_20260914_evidence/commands.json)。六份被审文件已原样快照；这些文件和原始 master plan 均未改动。

## 2. P1：需要先修正的八类规格问题

### B0-R01：FairBias 的监督接口与保真声明不成立

**位置：** [能力表:174](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/DATA_AND_METHOD_CONTRACTS.md:174)、[保真表:191](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/EXPERIMENT_REGISTRY_DRAFT.md:191)、[worker 假设:78](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/B0_WORKER_REPORT.md:78)。

能力表把 FairBias 的 `Requires y at fit` 写为 No。当前 BM 的 [_nmi_gate_ok:212](/Users/lkc/Downloads/code_v_0_3/src/fairbias/mitigation.py:212) 在第 220 行将 Y 传给 NMI，[_make_candidate:271](/Users/lkc/Downloads/code_v_0_3/src/fairbias/mitigation.py:271) 使用这个检查；下游监督预测器也需要 Y。若按表适配，会漏传标签、删除 NMI gate 或暗中改变算法。

此外，文件一面标 `UNVERIFIED`，一面写 `No (verified mapping)` 和“algorithmic fidelity is maintained”；保真表直接宣称 HISP 与 Tang 对齐，却没有逐公式/作者代码/当前调用路径的证据。论文刊物填写 `Information Sciences` 也未给可核查出处，不能据这张表认证文献身份。

**修订：** 区分 BM 变换器与整条预测 pipeline 的能力，两者当前 fit 均须相应 Y；预测不能读取真值。逐项列出 H、stress-elbow、幂流、revisit、NMI gate、聚合、停止及预算的来源和真实入口，未核实写 `SOURCE_UNVERIFIED`。数值 epsilon 与外部 EO 不互换。保留“application-v1，保真待验证”，删除已完成保真的断言，不以删除现存 NMI gate 来迁就表格。

### B0-R02：GBDT 类与参数不匹配

**位置：** [registry:33](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/EXPERIMENT_REGISTRY_DRAFT.md:33)。

类名写 `HistGradientBoostingClassifier (or verified GradientBoostingClassifier)`，参数却是 `n_estimators` 和 `subsample`。独立读取当前 sklearn 1.7.1 的 AST：HistGB 的构造函数接受 `max_iter`，不接受这两个参数；GradientBoostingClassifier 接受。它们也不是仅改名就等价的模型。这会导致构造错误或 worker 自行换模型。

**修订：** 按 master plan 固定为 `sklearn.ensemble.GradientBoostingClassifier`，保留既定 4 点网格；在能力表/预算/各 Gate 同步。若将来研究 HistGB，另起方法/预测器 ID 和专门参数，不使用 `or` 给实施者临场选择。本结论来自已安装源码，不是运行模型得到的结果。

### B0-R03：分区、年份和记录身份规定互相冲突

**位置：** [过滤顺序:76](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/DATA_AND_METHOD_CONTRACTS.md:76)、[分区算法:108](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/DATA_AND_METHOD_CONTRACTS.md:108)、[记录身份:121](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/DATA_AND_METHOD_CONTRACTS.md:121)、[C03 验收:27](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/ISSUE_CLOSURE_MATRIX.md:27)、[B5a 步骤:184](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/GATE_SPECS_B1_B7.md:184)。

- 文档要求按臂筛除 Y/A 缺失后分区；master 要求先在完整年度设计主表建立 PSU 映射，再按臂形成域。B5a 的执行顺序也先过滤后建主映射。
- `Hash(PSTRAT,PPSU,seed) % 100` 没指定散列算法、字段编码、排序和版本；不能据此保证跨进程/环境稳定。若实施者选 Python 内建 hash，风险更明显。
- 数据契约拒绝 `2023.0`，C03 正例却要求它与 `2023` 等价。`id_norm` 仅写去空格，没有解决 1/1.0、带前导零 ID、分隔符碰撞或来源 ID 字段。

**修订：** 固定唯一顺序“年度设计主表 → 稳定 PSU 映射 → 四臂资格掩码”；固定一个带版本的分配算法及序列化规则。采用有限整数值规范：2023 和 2023.0 规范成同一年度，NaN/inf/非整数拒绝。ID 按来源 schema 规范，原始 HHX 等字段是否足以构成键需明确验证；绝不以重排后的 RangeIndex 冒充来源 ID。身份使用无歧义结构。列出跨进程、输入顺序变化和跨臂复用的未来正反例。

### B0-R04：未知类别、字段字典和全缺失处理没有形成可实施契约

**位置：** [UNKNOWN:80](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/DATA_AND_METHOD_CONTRACTS.md:80)、[编码器:162](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/DATA_AND_METHOD_CONTRACTS.md:162)、[字段表:39](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/DATA_AND_METHOD_CONTRACTS.md:39)。

文本承诺显式 UNKNOWN 独立位置，给出的实现却只是在 F 自动拟合 `OneHotEncoder(handle_unknown='ignore')`。若 F 没有 UNKNOWN，该配置不会凭空生成 UNKNOWN 列；未来未知值会成为全零编码。两种政策并不相同。

字段表主要复述名称，尚缺逐字段的实质值/特殊缺失/年度可用性/输出 dtype 引用，F 全缺失数值的政策也未写入。PCNT18UPTC/PCNTLT18TC 被描述为 family，注册表为 household，应按来源措辞校准。URBRRL 被列成设计变量并写死 2013 版本，却没有年份依据；本研究不纳入它可由特征策略规定，不能因排除就重新定义为抽样设计变量。

**修订：** 明确 UNKNOWN/MISSING 在冻结编码词表中的保留机制与类型统一规则；F 中学习的是实质经验支持，预留哨兵不是从 S/T 学习新类别。给每个变量可核对的 registry section、valid/missing codes、year_available、类型和转换政策，可用静态配置提取生成，无需读数据。补全 F 全缺失、超训练数值范围、合并后类型变更及未知值的未来验收案例。

### B0-R05：可估计性、概率语义和 Gate 验收标准打架

**位置：** [observed groups:80](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/EXPERIMENT_REGISTRY_DRAFT.md:80)、[风险/校准:91](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/EXPERIMENT_REGISTRY_DRAFT.md:91)、[B4:164](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/GATE_SPECS_B1_B7.md:164)、[B5a:195](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/GATE_SPECS_B1_B7.md:195)、[B6 fit 禁止:252](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/GATE_SPECS_B1_B7.md:252)。

- 公平指标遍历 observed groups，没有明确七组全覆盖门槛。缺失一整组可能被排除在 max/min 之外，而非返回主指标 NOT_ESTIMABLE。
- B4 要求主指标零缺失，只列 VALID/NOT_SUPPORTED/BUDGET_EXHAUSTED；但正确的单类/缺组负例应产生有原因的 null。B5a 又要求所有子组满足支持，而 master 允许不支持条件的显式状态。
- `true calibrated event probabilities` 不是 predict_proba 的自动性质；评价校准本来就应允许模型尚不校准。
- 新增的 `logit(y) ~ logit(p)` 不可照字面作用于 0/1 的 y；同时校准回归拟合与 B6 全面禁止 fit 冲突。

**修订：** 固定 expected_groups，主 gap 要求其分母全部有效，observed-only 仅作不同名称的描述。分清“合法不可估计状态”和“验证失败”，B4 必须有故意缺组/单类负例且预期 null+reason，禁止 NaN/Infinity 当有效 JSON 数值。B5a 能验收支持度审计，不强迫所有算法/组都可计算。p 定义为模型事件概率输出而非真实或已校准概率。当前主方案默认保留冻结概率的校准曲线，先移除本次新加且未定义的 slope/intercept 拟合；以后若增加，须明确它只是评价诊断、正确的二项模型及端点政策，不能反向校准或更新预测器。

### B0-R06：主比较家族算错，推断方法被省略成一句“bootstrap CI”

**位置：** [registry:147](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/EXPERIMENT_REGISTRY_DRAFT.md:147)、[B2 验收:94](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/GATE_SPECS_B1_B7.md:94)、[B6:245](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/GATE_SPECS_B1_B7.md:245)。

“5 External Methods”后实际列 Unmitigated、RW、LFR、EG-DP、EG-EO、TO-EO 六个。若全部进入，两轴×两指标应为 24 个，不能仍除以 20。master 的 20 个明确只包括五种外部公平性对照；Unmitigated 另作预测参考。

登记只保留复制权重公式和 99.75%，没有指定复制方差的中心/分母、t 自由度及区间构造，容易被实现为 2000 次样本的极端百分位区间。master [202 行](/Users/lkc/Downloads/code_v_0_3/docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md:202)与 [212 行](/Users/lkc/Downloads/code_v_0_3/docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md:212)有明确拟定方法及非光滑统计量验证前提，不能省略。B2 也没有落实独立参考实现、随机误差容限、coverage 场景及未通过时降级规则；一般非线性 bootstrap 方差不能要求与一阶线性化在每个有限样本精确相等。

**修订：** 列出 20 个唯一 contrast_id，排除 Unmitigated 于该主家族且保留其单列比较；锁定复制方差、df、同复制权重的配对算法、每次重算最差组、无效复制和种子均值的定义。按 master 区分描述性 95% 与拟定 Bonferroni t 区间。为线性率设可计算参考、为非线性指标设独立复制核对及合成覆盖场景，预先规定容限而非强行数值相等。验证未通过不宣称 FWER 已控制或设计一致；这也不是“confirmatory”一词足以克服既有 2024 暴露的问题。

### B0-R07：称“完整 70 条件”，却漏掉必要分支和关键配置

**位置：** [总矩阵:163](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/EXPERIMENT_REGISTRY_DRAFT.md:163)、[B3 任务:120](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/GATE_SPECS_B1_B7.md:120)、[B5b:212](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/GATE_SPECS_B1_B7.md:212)。

54 个核心条件加 16 个 AE 条件的算术是对的，但它不是 master 全部必需分支。调查加权下游训练的 baseline/FairBias 敏感性、Arm 004 控制因素的几何路径比较没有完整 ID/参数/实施 Gate；AE 也没有适配实施任务、C 上 BA 效用、候选族/顺序等规格，B4 却直接要求全部运行。

共同 C 阈值的候选、无权重 BA 目标和平局规则没有进入 registry；仅写“校准阈值”不足以确保所有方法一样处理。主 FairBias 几何/幂流/NMI/参考 epsilon 的实际配置表也不完整。

**修订：** 建立逐条件注册表，至少分开核心、两种 AE、加权训练、几何路径、评价视图（如 t=0.5）。为每个条件明确拟合任务、预测器、臂、算法版本、预算、种子与归属 Gate；重复利用核心产物的评价视图不虚增 fit。补齐共同阈值及 AE 内层效用、strict slack、冻结映射规则；将“70”限定为核心+AE 小计。计划中已经要求的这些分支不是此次另加研究目标。

### B0-R08：种子缺失和选择结果的失败政策未落入可执行登记

**位置：** [seed:51](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/EXPERIMENT_REGISTRY_DRAFT.md:51)、[按均值选优:111](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/EXPERIMENT_REGISTRY_DRAFT.md:111)。

登记要求按 seed mean 排序，却没有 master [162 行](/Users/lkc/Downloads/code_v_0_3/docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md:162)的“失败种子不能被删除后选优”。不同实现可能一个保留失败配置、另一个对成功种子平均，产生幸存者偏倚。预测参考平局规则、mean 可行与逐 seed 可行的区分、S 后不重新拟合等也应落表，不能依赖聊天补充。

**修订：** 明确 expected_seed_ids、observed_success_ids、failed_ids、配置完整性状态，缺少必需种子的配置不得参选；确定性重复只运行一次并有明确规则。基础设施恢复保持 config/seed，增加 attempt_id。保存每种子结果、失败率、聚合与平局政策。恢复 master 的不重拟合边界及强预测参考的平局规则，不以测试表现改变这些政策。

## 3. P2：四类实施与证据问题

### B0-R09：计算预算混淆迭代、拟合和硬上限

**位置：** [budget:173](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/EXPERIMENT_REGISTRY_DRAFT.md:173)。

把 EG max_iter=50 写成“Max 50 base estimator fits”，但没有推导每次迭代的 oracle 调用界；迭代数不能充当真实 fit 计数。LFR 少了 maxfun=5000，k 被称为维度而不是原型数量。各行给 5/10/30 分钟、2/4 GiB，正文却统一按 30 分钟/4 GiB，且未解释相对 master 的差异。

**修订：** 原生迭代上限、oracle/model-fit 计数、几何计数、walltime、RAM 独立字段；遵循 master 默认公共上限或明确登记无实证依据的拟定修订，不能两套规则并存。给各方法 grid×实际训练 seed 的上界、可复用层及不包含的内层成本。合成验收用小规模全链和独立压力案例，不必须在 B4 把未来所有真实规模完整调参再跑一遍才能证明接口正确。

### B0-R10：声称各 Gate 自包含，但执行入口、变更权限与不可覆盖策略不完整

**位置：** [B1:41](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/GATE_SPECS_B1_B7.md:41)、[B1 failure:63](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/GATE_SPECS_B1_B7.md:63)、[B5b 输出:207](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/GATE_SPECS_B1_B7.md:207)。

B1 仅列测试计数，没有 interpreter、guard 入口、精确测试/探针路径及 hash；修改 survey/temporal 却未将相应测试文件纳入改动许可。Issue 表把 E02/E03/E04 分配 B1，但 Gate 正文仅要求 C01–C06/E01，归属不一致。B4 需要持久化/失败测试却没列其文件权限。B5b/B6 使用固定 ledger/manifest 路径，没有唯一 run_id 与拒绝覆盖规则。失败时“immediately revert offending changes”也应限定为本 Gate 自己的局部改动，不能回滚现有用户/worker 修改。

**修订：** 每 Gate 列出可用解释器、只读依赖、输入/输出路径、允许文件、拟定精确命令和数据 guard。区分 B1 修复旧路径与 B2 新统一评价服务，避免同一功能分两次改语义。将不需要 B1 完成的 E 项移至明确后续 Gate，并保持全表一致。未来产物使用唯一 run_id，已有目录拒绝覆盖；冻结清单的 hash 与 Supervisor 决定不是未经定义的密码学数字签名。

### B0-R11：若干 CLOSED 证据指错测试，C05 根因公式写错

**位置：** [closed 表:52](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/ISSUE_CLOSURE_MATRIX.md:52)、[C05:29](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/ISSUE_CLOSURE_MATRIX.md:29)。

静态 AST 核对发现：所谓 scientific config 引用的 test:461 属于 bool 审计测试；single-group 引用 127 属于缺失组几何；nonfinite weights 引用 157 是合法零事件率；bool 引用 438 是平局选择；partition duplicate 引用 516 是来源 fingerprint。已修复事项可以保留，但这组位置并不支持对应声明。

C05 写成 `1/(K-1)`，实际 [bias_metric:275](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:275)是 `1/K`。修复应去掉未观测支持导致的 K 变化，不应修改成另一公式。

**修订：** 每条 CLOSED 用实际符号名、当前正确行号、对应独立结果文件和 hash。将“scientific config hashing 已关闭”限定为旧 seed/grid/smoke 反例，不遮蔽仍开放的实际 estimator 参数和无损缓存问题。更新公式和全部错误锚点，不重跑真实数据来补证据。

### B0-R12：报告措辞与证据范围需要收紧

**位置：** [preflight:15](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/B0_PREFLIGHT.md:15)、[环境范围:124](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/B0_PREFLIGHT.md:124)、[命令记录:23](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/B0_WORKER_REPORT.md:23)。

“up to date with origin”在无 fetch 条件下只能指本地 remote-tracking ref；“any local Python environment”超过四个已检查环境；`python -c "..."` 不能作为可复用精确命令。全部 file:// 链接虽指向存在文件，也不是本 app 推荐的本地行号链接格式。提出后续阶段所需权限不是已经获得这些权限。

**修订：** 明确现时核验、历史引用、worker 自述与未验证项；完整保存本轮执行命令，不补造没有运行过的命令；修正链接为绝对路径及行号。保持 report 自身 hash 在外部 manifest/最终回复，避免自引用 hash 问题。B0 已读源码不足之处可补读静态证据，不通过文字保证代替核验。

## 4. 集中返工范围及下一步

本轮决策是文档 **REPAIR**，不否定六份文件中已经做对的结构、hash 核对和研究方向。按照 [协议第 3 节](/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md:23)“Worker Change → Mandatory Report → Independent Codex Verification”的流程，B0 未 Accept 前不启动 B1。

下一轮仅在新目录 `docs/plans/fairbias_benchmark_b0_r1_20260914/` 新增修订版六份文档、逐 R01–R12 的响应表与证据 manifest；保留当前六份提交和本次审查原样。静态配置提取、文档链接/公式/计数核对允许，测试、模型训练、真实数据访问、联网安装、stage/commit 均不在这次文档返工范围内。

验收标准是：八类 P1 冲突消除，四类 P2 执行与证据问题可追溯；每条“已验证”有证据；未来实现者不需要替设计者猜数据/模型/统计选择。R01–R12 是本轮有限清单，不要求重新审查整个项目或增加新算法。涉及无法在 B0 核验的外部库能力，应保留明确的 B3 验证门槛，不能假装已经解决。

可直接转交的下一步指令见 [Gemini B0-R1 修订 Prompt](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_B0_R1_REPAIR_PROMPT_20260914.md)。本次监督者仅新增审查/证据/指令工件，没有修改 worker 提交、实现源码或原始研究方案。
