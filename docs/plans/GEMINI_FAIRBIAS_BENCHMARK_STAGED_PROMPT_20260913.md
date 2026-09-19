# 可直接交给 Gemini 的分阶段执行 Prompt

使用方式：首次把下方“主 Prompt”完整交给有本地仓库访问能力的 Gemini。首次只执行 B0；后续每次由 Codex 审查通过前置 Gate 后，使用文末相应阶段启动模板。完整路线不等于同时激活全部阶段。不要将“STOP”删除后让 worker 一路运行真实数据。

## 主 Prompt（首次完整复制）

你是本项目的 Gemini Implementation Worker。请为一篇以 **FairBias 为主方法** 的 NHIS 医疗公平性应用论文建设可复现的比较实验。Codex 是独立 Supervisor，你不能自我验收、越 Gate 执行或自行提交代码。

当前工作目录：`/Users/lkc/Downloads/code_v_0_3`。

**ACTIVE_GATE = FAIRBIAS-BENCHMARK-B0。此次只完成现状核验、详细设计和后续 Gate 规格；不修改实现代码、不运行测试或训练、不读取任何真实微观数据、不安装依赖、不 stage/commit/push。完成后按标准报告停止。**

### 一、先读治理与现有证据，再开始规划

按以下顺序读取，不能用摘要或历史聊天替代：

1. `/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md`
2. `/Users/lkc/Downloads/code_v_0_3/AGENTS.md`（若工作区无该文件，使用用户提供的 AGENTS 指令并如实记录）
3. `/Users/lkc/Downloads/code_v_0_3/docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md`
4. `/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_CONSOLIDATED_AUDIT_AND_APPLICATION_PLAN_20260913.md`
5. `/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_FOURTH_REVIEW_20260913.md`
6. `/Users/lkc/Downloads/code_v_0_3/docs/reports/NHIS_D8_R4B_SUPERVISOR_ERRATUM.md`
7. 与上述设计直接有关的 `configs/nhis/`、`src/fairbias/`、`src/nhis_fairbias/`、`scripts/` 和 `tests/` 源码。

`docs/GPT6_CODEBASE_AUDIT_AND_IMPROVEMENT_GUIDE.md` 仅作导航，科学定义以经过核对的源码、原始文献和审查证据为准。

先记录实际分支、完整 HEAD、staged/unstaged/untracked 状态、当前差异统计及相关源码 SHA-256。预期分支 `research/nhis-fairbias`，规划时 HEAD 为 `67e6659fa65249a8842e34af5d8969629efe4bca`，已有未验收 worker 修改；不能把现有差异当成你本次修改，也不能把 HEAD 当作包含这些修改的完整版本。

保护标签 `inherited-code-v0.3-baseline-20260828` 指向 `038897e9f751edac6e36445b7706eec5fdb15988`。14 个继承文件及 `.gitignore` 不可修改。区分已有历史差异与本次新变化，不擅自“恢复基线”或清理工作区。若发现身份/完整性不符合实际前提，保留证据并停止依赖这些前提的动作。

B0 不联网、不下载、不安装；可只读查询已安装包 metadata，不导入项目以触发运行。缺少依赖或文献来源时登记具体缺口和未来 Gate 的域名/版本验证方案，不伪称已验证，不在 B0 自动补装。所有真实数据、历史个体级预测以及可能含微观数据的 scratch/runs 产物均不得打开。源码、配置、文档与已知聚合审查报告可读。

### 二、研究目标与固定默认设计

研究问题：FairBias 在识别“过去 12 个月因费用延迟医疗”时，相比其他公平性方法，能否改善预测—公平性权衡，优势和代价在哪些群体、预测器和年份成立？FairBias 是主角，但不预设其获胜，不以获得显著优势为停止条件，不将他人算法改写成较弱版本。

主方法默认 `FAIRBIAS_BM application-v1`，`FairBias-BM→AE` 和 `FairBias-Joint-BM/AE` 为独立扩展/消融。必须登记相对 Tang 论文、作者实现、历史 D8 与本项目工程扩展的差异；不能仅凭 mode 名称宣称算法严格保真。任何将 Joint 改为主方法的决定须在真实开发前修订并冻结。

主预测器 LR，GBDT 为必做的预测器稳健性分析。公平性核心对照为：Reweighing、LFR reconstructed features + 共同预测器、EG-DP、EG-EO、TO-EO，另有 unmitigated 参考。核心每个预测器 27 个支持的方法-臂条件；不要把条件数说成总 fit 数。采用总方案中的参数网格、选择规则、种子与预算作为拟定默认值，B0 只提出有证据的必要修订。

四个主结局臂：SEX/21 特征、HISP 七群体/21 特征、DISAB include/21 特征、DISAB exclude/15 特征。Arm 003/004 必须相同行、同分区、同权重，仅差六个残疾构成特征；Arm 004 不是“只分析残疾人群”。LFR 七群体格子明确 NOT_SUPPORTED，不压成二群体。

新 benchmark 采用 2022 年内 PSU 级约 80% F / 20% C 划分，2023 完整 S，2024 完整 T。F 拟合所有基础变换/预测器；C 拟合共同全局阈值和 TO、提供 AE 内部开发反馈；S 仅按冻结规则选完整配置；T 仅最终冻结预测和评价。跨臂共享先于筛选的 PSU 主划分。不得将旧建议中“2023 拆 C/S”与本版混用。

2024 已被历史研究观察，不能声称从未接触的测试集。NHIS 是重复横断面；结局与特征不足以支持个体未来预测或因果改善的表述。

### 三、必须具体设计的数据接口

请设计而非泛泛写“统一 sklearn API”。给出每种方法的数据流和 fit/transform/predict 角色：

- 分区携带稳定 record_key、X_semantic、y、A、WTFA_A、PSTRAT、PPSU、year、role、eligibility 和 schema/source hash，各部分必须严格对齐。
- Y/A 缺失固定排除；X 缺失按 F 拟合的规则处理。设计变量与保护变量不进普通预测列。
- FairBias 几何使用语义特征；预测器用训练期 one-hot/数值缩放；每个 FairBias 候选在其变换后的 F 拟合编码器并冻结。合并或分箱后更新类型 schema。
- LFR 连续重构特征直接进入共同预测器。官方 transform 改写 labels/scores，必须保留原始 y，禁止伪标签作为真实结局；A 仅在 protected metadata，防止额外混入 features；推断用无真值占位标签并验证其不影响输出。
- RW 的公平训练权重与 WTFA_A 分离；sklearn AIF360 接口返回 `(X, weights)`，临时 protected index 不可破坏记录身份。多群体支持需真实对象验收。
- EG 的 reduction cost sample_weight 必须正确传到 oracle；DP/EO 分开；Moment difference_bound 与 EG eps 分开；混合决策 q 从基分类器硬决策按混合权重计算。
- TO 仅在 C 拟合，明确 prefit=True 与正类概率接口；推断需要 A；EO 不加入不支持的 tol 网格。
- p=事件概率、q=随机决策概率、ranking score、hard decision 必须分类型。EG/TO 的 q 不冒充事件风险概率，也不再用 0.5 阈值改造方法。

每个方法的能力表包括：是否改 X/权重、训练是否需要 y/A、预测是否需要 A、是否支持七群体、是否支持调查加权学习、是否随机训练/随机决策、输出类型、原生目标、官方实现出处、预计适配位置与测试。当前未验证的能力标 UNVERIFIED，不能把计划写成通过。

### 四、共同评价与优胜判据

主评价是 WTFA_A 加权 BA 和 EO gap 的二维权衡，同时报告概率模型的 AP/AUROC/Brier/校准及每组 TPR/FPR/覆盖率。DP gap、无权重评价、t=0.5 及训练权重敏感性必须保留。不能只挑 FairBias 擅长的 d_phi 指标。

使用总方案中的统一组间极差公式，不能混用 mean-pair 与 max-gap；零分母/缺失组返回 NOT_ESTIMABLE。随机政策用精确 q 计算期望混淆计数，说明比值指标的语义。

采用总方案 S 上的共同操作点：EO≤0.10 为主、0.05/0.20 为敏感性，在满足预算的有效配置中最大化加权 BA；无可行者完整报告。另保留 S 上 AP 最优的无公平干预预测参考。主操作点是研究刻度，不是临床阈值。所有模型/阈值/种子/操作点在 T 前冻结，禁止测试集前沿择优。

调查设计单独实现，WTFA_A+PSTRAT+嵌套 PPSU；域估计保留完整年度设计。使用相同分层 PSU 复制权重计算方法配对差值；调查不确定性与训练种子差异分别报告。主方案列出的 bootstrap、singleton/无效复制政策和多重比较规则必须落成可验证的规格，不能将 iid bootstrap 当作完成设计推断。

“优于”必须同时指明预测指标、公平指标、比较对象、人群、年份、训练制度与区间。保留不显著、代价、失败、不可行结果，不自动使用“显著”“临床可接受”或“全局最优”。

### 五、B0 必须交付的具体文档

仅在 `docs/plans/` 下新增唯一带日期或 run_id 的 B0 子目录，不覆盖现有 master plan、prompt、历史审查或 worker 文件。至少交付：

1. `B0_PREFLIGHT.md`：实际 Git/源码身份、保护对象、本次边界、已有修改、只读依赖现状。不要输出微观行。
2. `ISSUE_CLOSURE_MATRIX.md`：C01–C06、E01 与所有应用前置风险逐项映射至根因、文件、Gate、正反例、验收证据。已修复的条目保留 CLOSED 证据，不反复重做；尚未执行的检查不写 PASS。
3. `DATA_AND_METHOD_CONTRACTS.md`：四臂、逐年字段/缺失/编码、F/C/S/T、稳定身份、逐方法输入输出和能力矩阵，包括所有 NOT_SUPPORTED 条件。
4. `EXPERIMENT_REGISTRY_DRAFT.md`：方法/预测器/网格/种子、主次输出、选择/平局/失败规则、调查推断、完整矩阵计数与计算量估算；保留原论文/库/应用偏离映射。预算是上限估算，不捏造实测用时。
5. `GATE_SPECS_B1_B7.md`：B1、B2、B3、B4、B5a、B5b、B6、B7 每项列出依赖、允许变更、数据/网络权限、输入、实现任务、预期文件、有限验收矩阵、失败处理、退出条件；用户拿其中一项即可继续，不依赖你记住聊天。
6. `B0_WORKER_REPORT.md`：协议第 9 节完整报告，附本次文档 hash、链接检查与本次差异，不将文档核查叫模型测试。

当前问题至少包括：无损状态/实际 estimator 参数缓存、真实 registry 兼容、记录身份规范、正类与概率契约、空 category level 几何、通用/专用评价口径、有限权重合计溢出；后续必须用实际对象与完整调用路径验收。不要将新的应用编码/加权评价悄悄写回历史 D8 结果。

### 六、完整执行顺序与停止条件

完整路线为：B0 设计落表 → B1 正确性闭环 → B2 数据/共同评价/调查推断 → B3 方法适配 → B4 合成全链与冻结 → B5a 真实支持度 → B5b 真实开发与选优 → B6 冻结 2024 评价 → B7 论文/复现包。

**本轮只执行 B0。** 路线、下一步建议或本 Prompt 对后续阶段的描述都不是阶段授权。每个阶段完成后先停止，由 Codex 独立审查；修复、验证和 Git 提交授权分开。不要自行启动另一个 AI 代替独立验收。

普通工程选择采用主方案拟定默认值并说明理由，不把几十个细节逐个甩给用户。遇到影响研究对象/结局/数据用途、未授权真实数据或网络、保护对象完整性异常时，报告具体依赖和依据；完成不依赖该问题的当前阶段工作，不绕过边界。

结束时必须使用以下精确标题；未执行写 NOT RUN 与原因，不能空缺或编造：

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

现在开始 B0，完成上述文档及证据后停止。

## 后续 Gate 启动模板（仅在前置阶段得到独立 Accept 后使用）

下面的文字是未来启动模板，不是当前授权。发送前由 Codex 附上真实的 Accept 报告路径/散列和当前 Gate 规格；没有真实 Accept 证据就不能将模板中的条件当作已成立。

### B1：正确性修复

激活 FAIRBIAS-BENCHMARK-B1。先核验随附 B0 Accept 证据及 B1 Gate spec。仅执行获准范围内的 C01–C06/E01 问题族修复，保持已验收修复、继承基线和历史结果。验证只能使用有持久数据访问 guard 的合成数据；不能读 COMPAS/NHIS/MEPS，不能全库盲跑测试。每个 issue 交付根因、修复、真实对象正反例、精确结果及变更前后 hash。不得接着执行 B2，不得 stage/commit。按协议第 9 节报告并 STOP。

### B2：数据和评价基础

激活 FAIRBIAS-BENCHMARK-B2。核验 B1 Accept 和 B2 spec。实现新 benchmark 的语义/预测表示、四臂身份与 F/C/S/T 合约、统一输出/指标、调查推断和选择规则，仅用合成资料。完成独立调查基准、域估计、配对差值、类别/权重/输出边界及测试标签独立性验证。不得接真实数据或提前适配全部外部方法扩大范围。按协议报告并 STOP。

### B3：外部方法集成

激活 FAIRBIAS-BENCHMARK-B3。核验 B2 Accept 和当前 spec 明列的安装环境/HTTPS 域名权限。锁定专用依赖，实现七类核心方法的 LR/GBDT 适配和能力拒绝；逐方法用真实库对象验收。特别核验 LFR 原 y、RW index/weight、EG oracle/q、TO prefit/C/A。不能用自造替代实现冒充官方库，不能降低对照训练质量。所有验证为合成；报告未支持格，完成后 STOP。

### B4：完整合成彩排和冻结

激活 FAIRBIAS-BENCHMARK-B4。核验 B3 Accept。生成覆盖四臂、多组、多 PSU、三年角色的纯合成数据，运行完整 LR 链及 GBDT 适配链，纳入随机性、预算耗尽、不可估计、模型重载等情况。交付不含真实性能的配置/矩阵/选择/推断协议冻结包、资源预算及真实执行规格。测试 T 标签变化不影响拟合、阈值或选择。不能读取真实年度数据，报告后 STOP。

### B5a：真实支持度检查

激活 FAIRBIAS-BENCHMARK-B5a。只有随附 Gate spec 明确列出的本地 NHIS 2022/2023 文件可读，核验 B4 Accept、数据路径/来源/hash/权限后才读取。只做固定纳入、F/C PSU 划分、编码/缺失/组别/调查设计支持度检查，输出聚合，禁止模型性能、2024 读取及逐人输出。若核心条件不适用，报告设计问题，不换种子或事后合并组。报告后 STOP。

### B5b：真实开发和配置选择

激活 FAIRBIAS-BENCHMARK-B5b。核验 B5a Accept、冻结注册表和只允许 2022/2023 的数据清单。在 F/C/S 指定角色内运行获准矩阵，预算内完成候选并按预先规则选定模型；记录成功、失败、超时、无可行配置及完整实际计算量。不得读 2024，不加网格/选种子追求 FairBias 获胜。冻结全部预登记操作点、预测参考、消融和模型 hash；无法完成的条件如实报告。完成后 STOP，等待独立审查决定是否进入 B6。

### B6：冻结年度评价

激活 FAIRBIAS-BENCHMARK-B6。核验 B5b Accept、模型/配置签名及获准 2024 文件清单。只加载冻结模型与阈值做预测/统一评价，程序层禁止 fit 和选择入口；全部方法使用相同年度域及复制权重。2024 上的支持度问题只决定可估计状态，不触发训练补救。若评价代码有缺陷，隔离产物并报告，修复后按获准同一规则重算所有受影响方法，不能只重跑不利方法。输出完整聚合比较，按协议报告并 STOP。

### B7：论文与复现交付

激活 FAIRBIAS-BENCHMARK-B7。核验 B6 Accept，只使用获准聚合结果生成主表、图、补充材料和可复现说明。以 FairBias 应用为主线，公开预测/公平权衡、其他方法能力、计算成本、失败与局限；如实说明既往结局、重复横断面、2024 历史暴露、调查近似与非因果性质。论文中的所有数字与图必须追溯到冻结结果。不得自动上传微观数据、投稿、发布或提交 Git。交付审稿前材料及协议报告后 STOP。
