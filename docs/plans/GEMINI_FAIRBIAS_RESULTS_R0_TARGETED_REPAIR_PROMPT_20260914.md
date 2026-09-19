# Gemini：RESULTS-R0.1 定点补正与可执行修复规格

你是 Gemini implementation worker，Codex 是 supervisor。工作目录 `/Users/lkc/Downloads/code_v_0_3`。FairBias 保持应用论文主方法；预测与公平性均比较，不预设胜出。

本次只执行 **RESULTS-R0.1 静态补正**。上一轮 verdict 为 REPAIR，未激活 R1。请完成下面明确的差额，不重写六份长篇说明、不重复全文承认 F01–F14、不跑真实实验。目标是交付一份下一轮可以直接实施和验收的规格。

## 一、输入、权限和保存边界

先读取：

- `/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md`、`AGENTS.md`、`GEMINI.md`。
- `/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_RESULTS_R0_SUPERVISOR_REVIEW_20260914.md`。
- `/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_RESULTS_R0_SUPERVISOR_REVIEW_20260914_evidence/verified/verification.json`、`corrected_aggregate.json`、`corrected_static_tables.md`，以及上一层 `verify_static.py`。
- `/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_BENCHMARK_RESULTS_REPAIR_PROMPT_20260914.md`、`FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md`。
- `/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_CONSOLIDATED_AUDIT_AND_APPLICATION_PLAN_20260913.md` 与 `FAIRBIAS_FOURTH_REVIEW_20260913.md`。
- 原 RESULTS-R0 六项交付、前次 `docs/plans/fairbias_benchmark_b0_r1_20260914/ISSUE_CLOSURE_MATRIX.md` 和 `EXPERIMENT_REGISTRY_DRAFT.md`。

只允许读取源码、测试文本、配置、文档和原交接已明确列出的聚合 JSON。只在新的唯一目录 `docs/plans/fairbias_benchmark_results_r0_1_<UTC>_<id>/` 新增本次文档及标准库核验材料。禁止改旧文档、源代码、测试实现或配置；禁止覆盖 runs、原 R0 交付和监督证据。禁止项目/第三方库导入、pytest、模型执行、个体数据读入、网络、安装依赖和 Git stage/commit/push。禁止为验明 C 人员交集而读取记录级数据。

可运行标准库 AST、JSON、hashlib、pathlib、subprocess 的静态检查；可读取指定解释器的 sys/sysconfig/importlib.metadata 信息，不导入 numpy/sklearn/fairlearn/aif360/项目以“验证环境”。不得推定不存在的 `.venv_benchmark` 已经可用。当前系统有 `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`，其依赖仍应按实际元数据核对。

## 二、自动修正事实清单，不重新计算模型指标

新增并保存可复现的标准库生成器 `generate_static_repair.py`，使用明确文件 allowlist。生成器输出唯一事实清单 `FACTS.json`，再由该清单渲染 `CORRECTIONS.md` 与报告的数值段，不手抄散列、人数或 AST 位置。

必须核验并列明：

1. 上轮监督报告实际哈希 `5ee400543c11ee5a7c559dc2e17ed82c8e5fa6c19cd6eca36b45843e4f96c8a2`；上轮 verification 实际哈希 `9a55c87665a19746c4917f39dadc3b4e2a38d4146cd2274aee3241978256e89d`。这些值是核验目标，不是免读取的替代数据。旧 41 项 manifest 全部仍匹配，错误发生在 worker 报告，不能写成输入已篡改。发现新的意外漂移按协议停止。
2. 原 run JSON 有 8 个非标准 NaN token，不是 15；通过 JSON 的 parse_constant 回调计数，同时保留类型和路径。不要双重 count 子串，不要改原文件。派生 JSON 用 null 与原因字段，`allow_nan=False`。
3. 原 R0 六项输出共 1708 行、110296 字节；新的交付另行动态统计。使用 `len(bytes.splitlines())` 并写明口径。
4. AST 有 19 个 test 函数；渲染 start/end 必须来自 lineno/end_lineno。无需执行测试。正文覆盖分析要明确是人工静态判断。
5. 使用实际 `arm -> methods/paired_contrasts/sample_sizes` schema；四臂原始数据、配对差值与监督提取交叉验证。方法表区分源码显式参数、依赖默认值、运行时未记录；没有版本证据就写 UNVERIFIED，不猜 installed defaults。历史 VALID 不改称方法有效。CI/p 仅转录旧实现输出，不补 df/B_eff、不重新计算。
6. 修正文案：RW 的点优势限定 BA/EO 两维；人数等式不能独立证明 C 个体交集；源码摘录必须逐字匹配，伪代码要标注；不以“全部测试为什么通过”的泛化叙述代替具体缺口。

`CORRECTIONS.md` 只需逐项链接旧错误和新证据，保留已正确的撤回声明，不重复整篇 addendum。保存完整生成命令、argv、cwd、解释器、退出码及输出；不能用 `...`。最终 `MANIFEST.json` 包含最终报告哈希但不包含自身，解决自引用问题；所有文件定稿后再生成它。

## 三、把 R1 拆成两个可执行子阶段，当前只写规格

新增 `R1_EXECUTION_SPEC.md`，明确 R1A/R1B 尚未启动。对每个子阶段给出：输入/输出、修改/新增文件 allowlist、真实函数/调用方、正反例、解释器及版本证据、完整命令、guard 入口、结果目录、失败状态和停止条件。不要写“覆盖全面”而省略实际用例。

### R1A：核心公共契约与几何配置贯通

建议的最小候选范围如下；它是交给 Codex 确认的规格，不是本次改代码许可。逐文件列明必要性；确实不需要的删除并说明调用路径证据：

| 缺陷/依赖 | 核心候选文件 | 不能被包装器替代的验收 |
|---|---|---|
| C01 | `src/fairbias/enhancement_state.py`、`enhancement.py`、`enhancement_contracts.py`；`src/nhis_fairbias/d8_enhancement_runner.py` | runtime 使用无损状态身份；3.000000001 与 3.000000002 不碰撞；真实 estimator C 改变使缓存失效；历史 v1 比对显式保留；真实 AE/Joint 调用路线验证，mock 仅隔离特定控制流并披露 |
| C02 | `src/nhis_fairbias/adapter.py`、`preprocessing.py` | 真实 `.registry` 与实际 schema 字段；匹配对象接受，语义字段变动拒绝，不能检查不存在的 `.feature_registry` |
| C03 | `src/fairbias/enhancement_contracts.py`；后续 benchmark `data_contracts.py` | 来源/年度/ID 的明确规范；bool、缺失、非整数年份拒绝；等价数值类型规范化、字符串前导零按来源保留；重复/部分交集/不同来源同号的正反例；不以普通特征相等代替同一记录 |
| C04 | `src/fairbias/enhancement_contracts.py`、`src/nhis_fairbias/d8_enhancement_runner.py`；公共输出校验器 | classes_=[1,0] 正类定位；非法范围/维度/行和/非有限值/单类/错标签拒绝；候选效用与终局评分均调用同一有效校验 |
| C05 | `src/fairbias/bias_metric.py` | 新增未观察 Categorical level 不改变散度；加权和无权重均验；零权重 level 不扩大支持 |
| C06 | `src/fairbias/evaluator.py`、新 benchmark metrics 的接口契约 | 应用 max−min 与历史 mean-pair 命名区分；缺组/单组不可估计；默认历史兼容不冒充应用指标 |
| E01 | `src/nhis_fairbias/survey.py` 及新 metrics 共用校验 | 两个 1e308 权重的比例应稳定给出0.5或明确拒绝，绝不能误报0；全局/组内分母和输出均验证 |
| BM配置依赖 | `src/fairbias/config.py`、`evaluator.py`、必要时 `mitigation.py` | 聚合字段确实传入几何引擎；effective params 与 hash 同步；现存所有兼容路线有显式策略 |

逐项明确“核心修复”还是“应用路径隔离”。后者不得将未修公共缺陷标成 CLOSED。保存已有通过修复，不回退 old dirty 文件。

算法规格必须把源码函数对应到规则：H=1 排除上下文；BM 交错幂流及 restart；highest-d_phi/失败停止；NMI 与 epsilon 来源；MDS stress/elbow；无效几何、有限预算和合法 no-op 的不同停止状态。多组 `abs(max(v1)-max(v2))` 与 `max(abs(v1-v2))` 要有区分反例，说明选择依据及实际 evaluator 路由。若文献来源未核验，应标 application-v1 的已声明实现，不以函数名称保证论文完全保真。

### R1B：FairBias 与所有对照的实际适配

范围仍为 `src/nhis_fairbias/benchmark/adapters/*.py`、`metrics.py`、`preprocessing.py`、`data_contracts.py` 及新增合成测试；具体列文件，不给无限制重构权限。先列统一接口的调用示例和状态机：

- 语义清理 → F拟合预处理/真实BM/基础预测器 → C冻结决策规则 → S选择完整配置 → 冻结 → T仅评价。原始语义清理与 F 拟合统计区分；原 D6 已拟合 parquet/字典不能当无泄漏原始输入。
- FairBias 主 fit 禁止 D6读盘、硬编码变换和均值差替代；实际变换作用于拟合及预测；数组无语义 schema 则明确拒绝。允许 no-op，拒绝“只生成字典但未作用X”的假成功。F指纹与映射/编码器/模型绑定。
- p、q、yhat 分开。确定性方法只在C按 master 的无权重BA、中点及边界候选、平局靠近0.5再选较大阈值规则冻结全局t。未校准不得悄悄用默认t充当主结果。EG/TO用原生q，不阈值化q、不把q作为事件风险p；TO base-score另列。
- TO unified fit 必须显式具备独立C，或在任何fit发生前拒绝；两阶段正例、F/C交集反例、未知组反例均列出。不能以“不再TypeError”作为全部验收。
- EG主条件拒绝不支持的外部调查权重；内部reduction_cost_weight必须实际到达oracle。仅给基础预测器乘外部权重不构成调查加权Moment。分开 difference_bound/eps/native objective。
- RW多组因子来自F；空单元拒绝；LFR学习预算恢复master，原y独立保留、A不混入共同预测特征、无真y推断不变性、记录重构和分类器各自样本量；Arm002为本实现NOT_SUPPORTED。
- full expected_groups、事件正/负支持、概率校验、有限非负权重/溢出、all-missing/UNKNOWN/变换后类型、冻结范围外外推、行身份及数组对齐均有有限明确反例集。
- 全年master PSU先建后按arm取domain；同cluster跨arm与输入行重排不变；缺稳定记录ID拒绝。改动S/T不能改变F/C拟合状态。

### 两个子阶段的命令与 guard 规格

为未来实现提出新入口 `scripts/run_fairbias_r1_guarded_tests.py`，支持 `--phase r1a|r1b --output-dir <new_absolute_path>`；R1A/R1B测试节点逐项命名。写出未来完整命令，明确脚本尚待获准新增；本轮不得执行它。R1A可指定上面的Framework解释器，R1B依赖必须以已安装环境的实证为前提；缺依赖报告阻塞，不能安装或创建虚拟环境冒充当前可用。

guard必须在pytest和项目导入之前安装，在测试收集、fixtures、模型fit/predict和序列化全程生效；进程内禁止网络和子进程绕过；路径resolve后按读写allowlist控制。禁止data目录、根目录COMPAS/Credit数据、真实parquet和D6历史结果。旧锚点若需要纯聚合常量，只放明确最小allowlist并在单独测试阶段处理；主方法集成必须在D6完全不可访问时通过。

先以新造的无敏感sentinel测试 guard 对 open/io.open/Path.open/os.open 的拒绝，以及 subprocess/socket 禁止；不得拿真实微数据做“拒绝测试”。说明其Python边界，不能声称覆盖未拦截的原生I/O；若引入其他进程或原生文件读取，需额外隔离设计再审查。

测试前后保存源码和用例hash、完整日志、退出码、guard拒绝事件、实际参数/调用账本和资源结果。每次输出新目录；失败保留，不覆盖重跑；禁止执行整个未审查的 `pytest tests`。小型真实BM合成集成与人工控制流mock分开计数，不能将spy调用本身当作算法保真证明。

## 四、R2–R4差额及完整矩阵

生成 `CONDITION_REGISTRY.json`：54 core+16 AE+4 weighted+2 path，共76个顶层条件；LFR Arm002的两个不支持单元另有排除记录，不冒充拟合成功。每项有 condition_id、arm、方法版本、backbone、配置网格引用、expected_seed_policy、训练权重目标、输出能力、所属阶段。配置展开另行计数；不把76称作训练次数。

保留前次草案四个weighted条件：Arm003/004 × LR × Unmitigated/FairBias-BM（只加权下游预测器，几何不偷换为调查加权）。两个path条件：Arm004/LR的BM和Joint fixed-2D；对应stress-elbow条件已在core/AE。若要改成Arm001双预测器，另列未激活的变更提案，当前不换。

参数与seed直接继承master，不凭运行结果缩预算：LR四档C；GBDT四配置；FairBias八epsilon倍率；LFR八公平配置×骨干；EG八difference_bound且eps=.01/max_iter=50；TO冻结参数。随机训练seed=[0,7,19,37,73]，确定性重复不伪造独立实验，不选最佳seed。强预测参考是无公平约束AP选择视图，单列但不重复制造模型fit。

在 `R1_EXECUTION_SPEC.md` 附录列 R2–R4 已定规则和明确未启动的执行入口：

- R2：全年度设计+domain、配对共享bootstrap、B=2000/95%有效复制/df/SE/退化状态；独立参考方差及非光滑gap覆盖验收；逐method×arm×backbone的S选优、完整seed门槛、空候选、失败状态；唯一目录、模型保存、strict JSON、分块内存、两个runner和demo统一调用路径。20主对比逐项列出：LR/主操作点、Arm001/003 × {RW,LFR,EG-DP,EG-EO,TO-EO} × {ΔBA,ΔEO}；分母不随失败缩小；同时保存描述性95%区间和调整区间。
- R3：所有adapter的LR/GBDT可达路径及版本来源；小规模76条件演练与独立资源压力例；epsilon_candidate贯穿BM/AE/Joint，slack=0、各接受不等式明确；paper模式不能直接开AE，应用变体单独命名并导出完整有效配置。资源预算采用master的4GiB/30分钟等基准，若更改，区分组件预算与全进程预算并提出证据。真实2022/2023仅在后续授权的精确路径/hash/字段下支持审计和开发。
- R4：冻结后2024回顾性评价，披露旧开发已看过T；禁止新调参/选seed/改主臂/挑显著区间。逐组率、覆盖、预测指标、失败与资源完整报告。Arm004为同人群删六个残障分量的特征消融；路径机制需首个分歧等证据。所有未完成阶段写 PLANNED/NOT_RUN，不写通过。

## 五、有限交付与验收

交付 `generate_static_repair.py`、`FACTS.json`、`CORRECTIONS.md`、`R1_EXECUTION_SPEC.md`、`CONDITION_REGISTRY.json`、完整命令/日志、`WORKER_REPORT.md`、最终 `MANIFEST.json`。允许合理拆分机器生成表；不新增图表或论文结果。

静态自检必须验证：旧文件不变；所有本次可解析JSON严格有效；报告数值与FACTS一致；19 AST位置全匹配；四臂提取全匹配；76唯一condition ID无重复、四weighted和两path准确；R01–R08逐项证据定位；R1A/B文件、函数、命令、guard与验收例明确。输入hash异常先查证，不能手工改报告值以蒙混。

最终报告精确使用协议第9节标题：Gate、Status、Files changed、Commands executed、Permissions requested、Tests executed、Exact test results、Input hashes、Output hashes、Row counts、Assumptions、Unresolved issues、Git diff summary、Proposed next step。Tests为 `NOT RUN — static audit response only`，静态自检另列真实命令及结果，不冒充pytest。

完成本次定点补正即停止。不得自批准RESULTS-R0.1或自动启动R1A。普通文档、字段和模板决定按本交接自主完成，不向用户反复索取已授权范围的许可。

STOP — waiting for Codex review.
