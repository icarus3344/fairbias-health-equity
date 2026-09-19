# Gemini：执行 FairBias R1A-R2 第二次定点返修

你是 implementation worker，Codex 是独立监督者。工作目录 `/Users/lkc/Downloads/code_v_0_3`。用户要以 FairBias 为主方法发表医疗公平性应用论文，同时比较预测与公平性；方法是否优于对照由规范实验决定，不能以获胜作为实现或筛选标准。

**本提示直接授权下述源码修复、人工验证与证据交付。按 A→B→C→D 实施，不必再交一份泛泛计划等待批准。R1A-R1 尚未验收，保留所有候选和运行，不回滚、不删证据。R1A-R2 是基础契约第二次返修，不是调查推断 Gate R2。本轮不执行 R1B 或正式 benchmark。**

## 0. 先读与建立身份

先读 `docs/AI_EXECUTION_PROTOCOL.md`、`AGENTS.md`、`GEMINI.md`，然后读取：

- `docs/reports/FAIRBIAS_R1A_R1_SUPERVISOR_REVIEW_20260914.md`。
- 同名 `_evidence/verification.json`、`additional_inputs.json`、`final_verification.json`、`MANIFEST.json`、`diff_from_pre_R1A_R1.patch`、`source_symbols.json`。
- `_evidence/probe_run_130440_143055/worker_case_output/results.json` 及同一 probe_run 下的 `probe_source.py`；这些是诊断反例，不是 10 项验收通过。
- `_evidence/evidence_verification_v2.json`、`worker_submission.txt`、`worker_log_analysis.json`。v1 的 study.json 误报已由监督纠正，不列成 worker 哈希问题。
- `docs/plans/GEMINI_FAIRBIAS_R1A_R1_REPAIR_PROMPT_20260914.md` 和 `GEMINI_FAIRBIAS_R1A_IMPLEMENTATION_PROMPT_20260914.md`。本提示明确更新处优先，未取消的算法、数据隔离和资源边界继续适用。

分支须为 `research/nhis-fairbias`，HEAD 为 `67e6659fa65249a8842e34af5d8969629efe4bca`，无 staged 文件。以本轮监督 85 项快照加 additional_inputs 的 1 项配置为候选身份；不是已验收基线。先核对并保存前像、hash、Git 状态与命令；出现新改动或哈希差异，按具体路径报告，不修改预期值掩盖。

14 个继承文件与 `.gitignore` 不改、不读原始 CSV 来算 hash。用只读 Git 元数据分别核验相对 HEAD 的新变化及相对保护 tag 的历史差异。已知 `.gitignore` 的 b595e59 历史变化如实保留，不宣称与 038897 逐位相等，也不擅自还原。

## 1. 可修改与输出范围

仅允许针对本报告缺陷修改：

- `src/fairbias/{enhancement_state,enhancement,enhancement_contracts,bias_metric,evaluator,config,mitigation,prediction_contracts,application_metrics}.py`。
- `src/nhis_fairbias/{adapter,preprocessing,survey,d8_enhancement_runner}.py`。
- `scripts/_fairbias_r1_guard.py`、`scripts/run_fairbias_r1_guarded_tests.py`。
- 五个既有 `tests/synthetic/test_r1a_*.py`；必要时新增 `tests/synthetic/conftest.py`，仅服务这些测试，不导入旧 tests。

上述是上限，不是要求每个文件都改。禁止修改 benchmark 包、旧测试、历史脚本、配置原件、治理文件、旧计划/报告、release 或既存 runs；不 stage/commit/push，不联网或安装依赖，不读取真实微数据，不跑旧 103 测试或全 tests。

所有新输出只放独占创建的 `runs/r1a_r2_<UTC>_<id>/` 和 `docs/reports/fairbias_r1a_r2_worker_<UTC>_<id>/`；不得重用已完成目录，不覆盖报告/manifest，不 rm、不清理失败尝试。源码前像存新审计目录，不再写 scratch；既存 scratch 保持只读。

7 个旧运行的恢复索引、84 项前像核对和既存失败运行保留已被监督核实，不重做恢复工程。只读引用既有索引/hash；如实继续标记原先缺失的 5 份原报告，不能重建原路径冒充未丢失。保存本轮全部尝试。

## A. 先修工具，再允许任何项目执行

所有项目导入、BM/AE 调试、pytest 与集成诊断均通过受控入口。禁止入口外 `PYTHONPATH=... python -c` 或直接 pytest。只读源码分析、stdlib AST/hash/报告生成可单独运行，但不导入项目/第三方库、不访问未授权数据。

父控制器与子解释器均使用已安装 Framework Python 的 `-I -S -B`；新增支持 phase `r1a-r2`：

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -S -B scripts/run_fairbias_r1_guarded_tests.py --phase r1a-r2 --output-root /Users/lkc/Downloads/code_v_0_3/runs
```

如需单个合成模块调试，可增加严格枚举的 test 选择参数，但必须经过同一 guard 和哨兵并保存独立尝试。-S 后只添加已核验的当前解释器 site-packages，不能执行 .pth/user-site/sitecustomize；在导入项目、pytest、第三方库前安装常驻 guard，禁插件自动加载，cache/temp 限本轮输出。确保加载对应源码，不能只凭 -B 声称没有读旧 pyc。

### A1. 修实际路径与 fd 边界（R2-02）

统一四种 open 与 unlink/remove/rmdir/mkdir/rename/replace/truncate/chmod/utime/link 等受支持路径：绝对/相对/符号链接、单/双路径对应的 dir_fd、读写 flags 含 O_TRUNC。可以明确拒绝全部不支持的非默认 dir_fd；若支持则必须解析实际对象，不能使用未解析的 `/dev/fd/N/path` 代替核验。不同 audit 事件默认 dir_fd 的 None/-1 等语义按真实事件处理，不能猜位置。

整数 fd 需登记来源、对象、方向、生命周期；不能对 mutation 跳过核验，不能为修复 tempfile 而允许所有 fd 或所有 /dev。固定日志 sink 只允许 logger 的受控写入，不能通过按日志路径无条件放行实现可截断/替换日志。不得用关闭 guard 解决递归。

必须有真实 API 的最小人工哨兵：在独立外层保护下，以本轮新造的“被测输出外”目录 fd 尝试相对 unlink/rename/truncate，确认被测 guard 的专用异常和对应事件，文件内容/hash不变。绝不使用任何真实数据或既存文件充当哨兵。外层兜底、普通 OSError/FileNotFound 不能算被测 guard 成功。

### A2. 日志、拒绝与退出状态形成闭环（R2-03）

日志磁盘写失败、部分写入未完成、预算耗尽、非预期关闭、不可解释截断均令日志健康状态失败，不能吞异常后 PASS。允许内存仅保留有明确计数的窗口，但不得把完整磁盘证据丢失解释成无害窗口淘汰。统计字段区分内存窗口、磁盘丢失与拒绝，不以计数差冒充分类。

每个预期拒绝通过短生命周期测试声明绑定操作、目标类别、原因、次数及异常类型；正常运行的剩余拒绝全是 unexpected。测试阶段不能一揽子豁免。哨兵不通过不得收集项目测试；guard 未安装/安装异常必须 FAIL，不能 skip。

在 sink 仍有效时完成需要审计的子阶段写入，明确最终化/关闭顺序；父阶段固定范围的归档另记 provenance。pytest exit=0 只是成功条件之一，guard/log/来源/资源任一不合格须总退出非零。

故障测试使用人工 sink/受控实例或父控制器管理的固定子诊断：显式证明预期失败状态和退出码，单列 `EXPECTED_FAILURE_PROBE`；不要把真实主运行中的日志失败 catch 掉再宣称主运行成功。不能实际发网络包或启动任意进程来验证阻断。

### A3. 预检、hash、资源和最终化（R2-09、R2-10）

固定只读 Git 命令检查 returncode；HASH_MISMATCH/MISSING_INPUT/PRECHECK_ERROR 明确失败。执行前后验证所有实际导入项目源码、测试、guard/runner、允许的配置，不只存两个 JSON 不比较。第三方环境记录实际 Python/numpy/pandas/sklearn/pytest 等版本、argv/cwd、显式路径、源文件加载策略。

父控制器仅允许固定只读 Git、当前 child PID 的 `ps -o rss=` 与同解释器固定子进程；禁止 shell/任意命令和子进程再生。维持 15 分钟/4 GiB 的监测终止策略；ps 故障明确 MONITOR_ERROR，先排除子进程已正常退出的竞态。kill 后 wait；采样最大 RSS 不冒称精确峰值或 OS 硬限额。

所有正式与调试尝试记账。报告从最终机器结果生成，报告定稿后独占创建一次 MANIFEST，包含报告及输入/输出引用而排除自身；若已有最终 manifest，另建交付目录，不覆盖。不得编造函数名或把手抄汇总称为自动生成。改后 AST 给实际路径/函数/行号。

## B. 按真实消费路径修源码

### B1. 公共指标与 D8 终局（R2-01、R2-04、R2-08）

1. 修 `FairEvaluator.compute_metrics` 的 prot_vals/groups 未定义回归，按每一受保护列初始化。原始 y/yhat 在 int cast 和 ravel 前验证，不能让小数 q 被截断后进入硬标签指标。保留 legacy 名称、公式和已声明能力；应用 helper 只接受严格一维二分类硬标签。不要为让旧路径能运行而给无支持指标填 0。
2. 明确一维 shape、等长、空输入、合法组标识、expected_groups 的类型、重复、缺失、额外观察组和单组策略。逐元素处理 object dtype 的 None/NaN/Inf，不能只判断浮点 dtype。禁止在 early return 前漏掉 expected_groups 验证。合法 Sequence 统一处理或明确限制，不能静默忽略 ndarray 等输入。
3. equalized_odds_gap 为 max(TPR_gap,FPR_gap)，与 EOpp、legacy EO 分开。DP、EOpp、EO 的支持和状态独立，缺组/缺正负类时返回 null 与具体状态，不伪造零。未提供预声明组时仅输出 observed-groups 描述性结果并带主估计资格 false。
4. 实际 `D8EnhancementRunner.run_arm` 所有相关评估路线传入冻结 schema/arm 声明的组全集，贯通 evaluate_representation→helper→validation/test 的最终序列化对象。最终保留 DP/EOpp/EO、各自 estimable/status、expected_groups 及来源、is_primary_estimand。找不到声明时采用显式描述性状态或拒绝主估计，不能从 S/T 的 observed groups 补出主组全集。
5. 检查消费者对新增 null/status 的处理，但本轮不修改历史结果或 benchmark 包。若消费者修复确实超范围，报告具体调用方为 OPEN，不擅自扩展。

最小反例包含：真实 compute_metrics 的合法两组/多受保护列；原始小数标签/二维标签拒绝；object Inf 组拒绝；TPR 相同/FPR 不同得 0/1；无负类时只影响 EO；七组缺一没有主 EO；真实 LR evaluate_representation 最终对象有正确字段；run_arm 注入人工提供者后确认所有分支传递声明。不能只测试 helper。

### B2. 真实 Registry 与冻结预处理（R2-05）

比较真正消费的 feature_lists.primary_core_features、expanded_utilization_features 及顺序；family/spec、harmonized 映射、有效码、清理规则、类别编码、拟合 provenance 都要与实际执行一致。不要只验证 family 字典键顺序或原 schema 不存在的 valid_range。

避免共享调用方嵌套字典；绑定拟合时的完整有效 schema/规则指纹及 2022 development-train provenance。外部原字典后续修改不能改变对象；直接修改对象内部 registry/specs/列表时须被阻止或在使用前要求重新 fit，不能保持 is_fitted 却改变清理语义。transform 使用冻结规则，不能只加 init 时一次比较。

以真实 NHISPreprocessor.fit 验证：匹配正例、substantive_codes 改变、类别码改变、真实 feature_lists 顺序变化、缺 registry/provenance、错误年份、fit 后外部与内部修改。年龄90人工例应在无重拟合时保持原冻结输出，或明确拒绝已变对象；不能从18静默变90。只允许人工运输与全年人数检查注入，不 mock 比较器/清理/fit，不手填 `_fitted_record` 冒充正例。

### B3. 缓存参数与身份（R2-06、R2-08）

只保留一个 `compute_configuration_fingerprint` 实现；维护现有公开 keyword 的明确含义，或提交明确的兼容层和调用方迁移证据，不能留下被覆盖的旧定义。所有实质参数仍绑定实际 model/scaler/evaluator/transformer/epsilon/partition。

参数字典键和值均类型化：不同映射 `{1:0.5}` 与 `{"1":0.5}` 必须不同或明确拒绝不支持键型，不能只检测同一字典内的碰撞。None 与空容器不能用 `or` 混同；实际允许空指数网格则保留其独立语义，否则在 engine 入口明确拒绝。保留已有标量/数组/非有限/get_params 修复与 v1/v2 显式区分。

来源使用明确类型和未知策略；NaN/Inf/bool/任意对象不能 str() 后变成新的合法来源。允许 None 表示未知但必须保守检查同年度同 ID 重叠；空来源明确拒绝。ID 不支持对象拒绝，保留合法前导零和不同已声明来源同号；通用核心不限定 2022–2024。多年度列先各自规范化，再比较语义，2023 与 "2023.0" 应等价，冲突/非整数/非有限年份拒绝。

所有异常仅记录形状、类型、问题数量和非个体上下文；不拼接完整标签、原始 year/ID 集合或任意参数对象内容到日志。

### B4. 保留已经修好的内容与范围

不得回退 C04 概率三条主路径、C05 正支持类别处理、E01 权重公共尺度/索引修复及 G01 multigroup_aggregation 传播。旧要求的加权 Categorical/零权重空 level/合并后空 level、权重 shape/重复/乱序索引和失败路径仍须有可核验覆盖。

本轮不实现主 benchmark 的完整调查加权 q 指标，不把硬标签伪造成风险 p。核心 AE 的 AUROC 目标与后续应用 C-BA 配置分开；paper 模式和工程模式、BM 作者1998幂流与 AE 六网格的差别保留。

## C. 必须补行为集成，测试数量不是目标

只收集五个 R1A 合成文件和必要人工 fixtures。每个验收项提供：问题 ID、真实调用路径、有效参数、是否注入及边界、实际 fit 次数、状态/轨迹、数据规模和结果字段。

| 组别 | 必须观察的行为 |
|---|---|
| AE cache | 同一 engine 实际 enhance_step/candidate evaluation；不变配置复用，改变 LR C、实际 scaler、epsilon 后必要候选重算，再回到旧配置时按明确上下文政策处理。用模型 fit 计数/trace 证明，不只 mark tracker 或比 hash。 |
| Joint | 实际 D8 run_arm 的 Joint 提交和循环检测；控制状态形成返回已见状态的人工路线并查看 stop_reason。人工 provider/阈值/效用可注入且披露，不能用普通 set 代替 Joint；另保留真实 LR 集成。 |
| BM | 实际小型 MDS→BM 搜索→提交或明确停止，记录属性、指数序列、epsilon/NMI、是否候选提交。author 1998 幂流不截短来过测试，可配置显式预算或设计早停例。 |
| 搜索边界 | 两次以上调用证明 restart 从头与 monotone_cursor 延续的轨迹；最高偏差属性失败 stop；NMI 真实 phi_threshold 的下/等/上；候选严格 <epsilon 与终局 <=epsilon 的等号差别；非有限、no-op、无候选、未收敛、预算耗尽分别命名，不称全局最优。 |
| 隔离 | 固定人工 F/C，分别扰动独立人工 S/T 的 X/y/A/w，实际 F 拟合产物和 C 选择输出/阈值不变；F/C 重叠或对象原位改动被拒绝。C 被扰动只用于明确测试选择变化，不能把 C 叫作 S/T。 |
| 公共输出 | compute_metrics + evaluate_representation + run_arm 的最终字段和状态；预测正例使用真实 LR。已知坏输入在正确层拒绝，不能 catch 任意异常代替预期契约。 |
| guard/工具 | 实际 dir_fd 人工哨兵、fd方向、日志故障、意外拒绝、Git/监控/哈希失败状态。受控失败诊断和正式测试运行分别记账。 |

规模：每分区 ≤200 行，真实 classifier/AE/BM ≤8 个预测特征；registry 实际 fit 例外可用既存24字段但每分区 ≤8 行，不能在该宽表上训练 classifier。禁止生成 89,802 行满足旧固定人数。禁止读取 D6/D8 历史数据或真实2023/2024文件；人工 S/T 与实际调查年份数据完全独立。

测试用断言验证预定数值/状态与实际控制流，不接受“只要不报错”。原46项可作为回归基础，但新增计数不是关闭问题证据。避免改断言掩盖失败、skip guard、吞掉异常、扩大网络/文件允许范围。

## D. 交付与停止点

在独占的新 worker 审计目录交付：

1. `WORKER_REPORT.md` 使用协议第9节全部精确标题，Status 写 `FIXED_CANDIDATE — awaiting independent review` 或如实 `PARTIAL/FAILED`；不能自行 gate ACCEPT。
2. 结构化 `issue_closure_matrix.json`：R2-01…R2-10 与遗留 C/G/E 要求→修改函数→最终 AST 行号→测试节点→实际结果字段→FIXED_CANDIDATE/PARTIAL/OPEN。未执行的行为写 OPEN，不能拿测试标题充证据。
3. 前像、前后 hash、相对本轮前像的 patch；实际导入源码和配置身份、环境/argv/cwd、资源状态；全部失败/成功尝试索引和来源引用。
4. 每个行为集成的 trace、实际 fit 计数和数据规模；逐项预期拒绝及匹配事件、意外拒绝、日志健康/截断状态；父子最终退出原因。
5. 由最终机器结果生成的报告与一致性核验；报告完成后一次生成 MANIFEST（包括报告，排除 manifest 自身）。已最终化工件不覆写，后续补充新版本独占目录并引用前版本。

如实登记 R1A-R1 存在入口外项目执行、scratch 输出和 manifest 最终化顺序问题，不断言发生了未证实的数据泄露；本轮不重跑这些无保护命令。已恢复的旧运行索引可以直接引用，历史缺失不能填造。

明确写出：本轮只验证软件/人工契约，旧 F01/F02 主 benchmark 问题仍 OPEN；R1B/R2/R3/R4 未执行；没有真实调查推断、FairBias 优势或论文保真认证。完成后停止，等 Codex 独立核验，最后一行必须为：

`STOP — waiting for Codex review.`

## E. 完整应用研究路径保留，当前不执行

R1A 验收后，R1B 接真实 F-only FairBias 与 Unmitigated、RW、LFR、EG-DP、EG-EO、TO-EO，明确 LR/GBDT 兼容性和 p/q/yhat；R2 验证全年设计/domain 与共享配对调查复制；R3 按已登记76条件、5种子执行获准开发；R4 冻结后进行回顾性2024评价。2024已被看过，不能称新盲测。

主对比继续是 Arm001/003、LR、预定 EO 操作点下 FairBias-BM 对五种外部方法的 ΔBA/ΔEO，共20个差值家族；B=2000、有效复制、df/SE、多重比较、失败配置/seed按已审查规格，不为显著性或FairBias排名临时改动。Arm004路径解释要求首次分歧与受控因素证据。预测与公平性均报告，风险 p 不存在的方法如实标能力限制，不拿 q 计算风险校准来补齐表格。
