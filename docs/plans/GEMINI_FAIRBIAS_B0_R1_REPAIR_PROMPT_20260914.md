# Gemini：B0-R1 集中文档修订 Prompt

将以下正文完整交给能够访问本地仓库的 Gemini。此次仍是 B0 文档返工，不是 B1 实现启动。

---

你是 Gemini Implementation Worker，Codex 是独立 Supervisor。

**ACTIVE_GATE = FAIRBIAS-BENCHMARK-B0-R1**。

Codex 已核对你提交的六份 B0 文件：28 条已声明 hash 全匹配，15 个原有候选文件未变；基础研究方向保留。但 B0 的模型接口、数据身份、实验家族和实施规格存在冲突，监督结论为 **REPAIR**。本轮将这些问题一次集中修好，仅修订文档，不进入 B1，不运行任何实验。

## 1. 必读文件与权限

工作目录 `/Users/lkc/Downloads/code_v_0_3`。先读取：

1. `/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md`
2. `/Users/lkc/Downloads/code_v_0_3/AGENTS.md`、`/Users/lkc/Downloads/code_v_0_3/GEMINI.md`
3. `/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_B0_SUPERVISOR_REVIEW_20260914.md`
4. `/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_B0_SUPERVISOR_REVIEW_20260914_evidence/verification.json`
5. `/Users/lkc/Downloads/code_v_0_3/docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md`
6. 原始六份提交所在目录 `/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_20260914/`

随后按监督报告的 R01–R12 阅读必要源码、配置和既有独立证据。不要仅从旧报告复制行号。

允许：读取上述文档/源码/配置/已知聚合审查证据；查询包 metadata；用标准库解析 AST、静态 JSON 配置、核对文档链接/公式/计数/hash；只在新目录 `docs/plans/fairbias_benchmark_b0_r1_20260914/` 写入本轮修订文档及证据。

禁止：改动原始六份 B0、master plan、旧 prompt、审查报告或任何已有实现/测试；运行 pytest、模型、benchmark 或项目导入；打开真实 NHIS/COMPAS/MEPS 数据及个体预测；网络、依赖安装、stage/commit/push。修订目录已存在则使用新的唯一后缀，不覆盖完成产物。未验收的原有代码变动保持原样。

## 2. 必须完成的有限修订 R01–R12

### R01 — FairBias API 与算法身份

修正 FairBias 训练不需要 y 的错误。现有 `src/fairbias/mitigation.py` 的 `_nmi_gate_ok` 和 `_make_candidate` 明确使用 Y；下游分类器同样监督拟合。分别声明变换器/整条 pipeline 的 fit 与 predict 输入，不删除 NMI gate 来迁就旧表。

补充 H、几何维数、幂流/顺序、restart、NMI、组对聚合、epsilon reference、停止与预算的逐项来源→代码入口→未来验收映射。查不到原文的页码或来源就写 SOURCE_UNVERIFIED，登记后续核验 Gate；不得写已保证论文保真。核查论文的题名、刊物和 DOI，不保留没有来源的期刊名。移除 `verified mapping` 等与 UNVERIFIED 状态矛盾的断言。

### R02 — 固定一个可运行的 GBDT 类

主方案使用 `sklearn.ensemble.GradientBoostingClassifier`，配 `n_estimators=[100,200]`、`max_depth=[2,3]`、learning_rate=0.05、subsample=1.0。删除 HistGradientBoostingClassifier 的二选一写法；所有方法、预算和 Gate 使用同一骨干定义。可读已安装源码/metadata 核对，不在本轮实例化训练模型。

### R03 — 唯一的数据分区与身份政策

统一顺序：完整 2022 年度设计主表 → 一次 PSU F/C 映射 → 各臂资格掩码。不能先按每臂 Y/A 过滤再建 master design。

把不确定的 `Hash(...) % 100` 改成完全指定的算法。默认采用：按规范化 `(PSTRAT,PPSU)` 键稳定排序；NumPy Generator(PCG64(20260913)) 按该顺序生成一次 uniform 值，值 <0.2 的 PSU 入 C，其余入 F；记录 RNG 名称/版本/键编码及映射 hash。版本在实施环境锁中固定；跨臂共享，不换 seed，不使用 Python 内建 hash。此处只写规格，不读取真实数据生成映射。

2023 与 2023.0 是整数值年度的等价表示，规范后相同；NaN、inf、非整数值、bool 等非法输入拒绝。ID 按具体来源 schema 定义数字/字符串/前导零政策，不是简单 str/strip。使用无歧义结构，明确来源原始 ID 字段及其待验证唯一性；不以筛选后的 RangeIndex 冒充原始身份。给出跨进程、重排、跨臂、部分重叠的未来验收例。

### R04 — 编码与逐年字段规格

明确 UNKNOWN 和 MISSING 的冻结保留位；仅写 OneHotEncoder(handle_unknown='ignore') 不足以生成未在 F 出现的 UNKNOWN 列。说明如何保留哨兵并只从 F 学实质类别，禁止看 S/T 扩词表。

从现有 registry 静态提取完整字段表：source/harmonized 名、valid/missing codes、year_available、类型、角色、imputation、输出 schema。补齐全缺失数值、超 F 范围、类别合并后类型与未知输入政策。PCNT18UPTC/PCNTLT18TC 使用 household 的来源语义。URBRRL 如不纳入，只陈述特征政策，未核对年度资料前不写死版本或称为 PSU/分层设计变量。

### R05 — 可估计性和概率评价

所有主 gap 按 expected_groups 定义，HISP 七组缺一不可；observed-only 为不同名称的描述指标。单类/缺组/零分母是合法 NOT_ESTIMABLE/null+reason，不能为满足 B4“零缺失”而补零。B4 增加预期不可估计负例；B5a 验收支持度审计和状态，不要求每一条件都可估计。

p 表示模型事件概率输出，不能预先称 true/calibrated。主方案保留冻结概率的校准曲线，本轮移除新加且与禁止 fit 冲突的 slope/intercept 回归；不要对二元 y 直接取 logit。若以后增加校准诊断回归，另定义纯评价边界，不在此轮偷改 B6。

### R06 — 20 个对比及调查区间

主家族为 LR、主操作点、SEX 与 DISAB include 两臂，FairBias 对 RW/LFR/EG-DP/EG-EO/TO-EO 五种外部方法的 ΔBA/ΔEO，共 20 个唯一 contrast_id。Unmitigated 保留单列预测参考，不放入该 20 个家族。任何缺失方法都不缩小校正分母。

按 master plan 明确：有效复制统计量的样本方差、B_eff−1 分母、年度 PSU 数减分层数的 t 自由度、未调整与 Bonferroni t 区间、有效复制比例、每次重算 max/min 和同复制权重配对、固定模型与训练种子差异的区分。不能省略成“99.75% bootstrap CI”。补独立参考、有限容限、非光滑 gap 的合成覆盖场景、未通过时仅描述的降级政策。不要要求任意非线性 bootstrap 与 Taylor 线性化逐值完全相等，不预先断言已控制 FWER。

### R07 — 完整条件登记及阈值/AE 规格

54 核心 +16 AE 只能叫小计。补齐原 master 要求的 baseline/FairBias 调查加权下游训练敏感性、Arm 004 固定其他因素的几何路径比较；为这些分支给独立 ID、适用臂/预测器、完整参数、种子、实施 Gate 和预算。可以分批实施，不能凭一句“ablation”跳过实现工作。

建议最小明确安排：加权训练敏感性在 Arm 003/004、LR 上比较 Unmitigated 与 FairBias-BM（4 个新增条件）；路径比较在 Arm 004、LR 的 FairBias-BM 和 Joint 上增加 fixed=2 版本（2 个新增条件），与已有 stress-elbow 版本配对，保持从 F 得到的参照 epsilon 固定、幂流及其余参数相同。其他明确预登记的扩展可另列，但不得暗中扩大主对比家族。由静态注册表计算总数，不手写互相矛盾的数字；这些是拟定配置，本轮不运行。

把共同 C 阈值规则写入 registry：最大化无权重 BA；候选为不同概率相邻中点及边界；平局先离 0.5 最近，再取较大阈值。AE 在 C 上用同规则得到 BA 效用，明确 strict slack=0、候选族/顺序、停止与预算。两种 AE 的实现任务、合成验收安排在 B3/B4，不能等真实 B5b 才设计。t=0.5 等评价视图与新增拟合条件分开计数。

### R08 — 失败种子与选优

定义 expected_seed_ids、successful/failed seed IDs、配置完整性。缺少必需种子的配置不得通过丢弃失败种子计算均值参与选优。确定性条件不重复制造五次证据；基础设施恢复保持 config/seed 并记录 attempt_id。

明确均值可行不等于每个 seed 都可行，保留逐 seed 状态。补强预测参考的平局规则和 S 后不重新拟合 F+C+S 的约束；与主方案一致。

### R09 — 真实计算预算

不要把 EG 50 iterations 叫作最多 50 oracle fits。分字段列原生迭代、实际 model/oracle fit、几何计算、walltime 和 RAM；准确恢复 LFR maxiter/maxfun 及 k=原型数。消除各行 5/10/30 分钟与统一上限文字冲突，默认遵循 master 的公共 30 分钟/4 GiB 边界。

给 grid×seed 的外层拟合上界、可复用层和内层成本的计数责任。B4 采用小规模完整矩阵+独立压力案例；无需为了证明接口正确先做一次未来真实规模全调参。

### R10 — 能直接实施的 Gate 规格

每个 Gate 给出明确 interpreter/环境前提、输入与证据路径/hash、允许变更文件、拟定命令、guard 机制、验证对象、退出条件。当前命令与未来计划命令分开，不能声称已执行后者。

修正 B1 的源文件/测试允许名单与问题分配：survey/temporal 测试需要可修改范围；B1 修复当前公共契约，B2 建新 benchmark 统一服务，保留历史输出定义。E02/E03/E04 等问题归属与 Gate 正文一致，不无意扩大第一阶段。

每次运行输出唯一 run_id，已有目录拒绝覆盖；删除固定路径覆盖风险。失败只修复/撤销本 Gate 自己可辨认的局部改动，不回滚现有 worker/user 变更。模型 hash 与 Supervisor Accept 记录不冒称未实现的密码学签名。B1 至少明确使用已核对的 Framework Python 及已有纯合成 guard/核心/13 probe/第四轮/扩展检查的具体路径，逐项静态核实可用性。

### R11 — 修复错误的证据锚点与公式

按真实函数名和独立结果修正 CLOSED 表。原 test:461 是 bool 审计，127 是缺失组几何，157 是零事件率，438 是平局，516 是来源 fingerprint；它们不能支持旧表所配的不同结论。保留已通过的有限修复，不把 seed/grid/smoke 已闭环写成整个配置缓存已闭环。

C05 的源码系数为 1/K，不是 1/(K−1)；只修复文档错误和有效支持定义，不改任何实现。所有引用先验证文件存在、行号落在实际符号/结果位置。

### R12 — 范围与报告准确性

环境结论限定为四个已检查解释器；没有 fetch 的 origin 状态只称本地 tracking ref。区分 worker 记录、当前监督核验、历史资料及未验证项。使用绝对本地文件链接及正确行号。

完整记录本轮真实执行命令；禁止 `python -c "..."` 省略式证据，不补造历史命令。报告自身 hash 放外部 manifest/最终回复。不要因为写了未来 Gate 规格就认为获得实施或网络权限。

## 3. 交付与本轮完成条件

在新的 B0-R1 目录内交付：

- 修订版 `B0_PREFLIGHT.md`、`ISSUE_CLOSURE_MATRIX.md`、`DATA_AND_METHOD_CONTRACTS.md`、`EXPERIMENT_REGISTRY_DRAFT.md`、`GATE_SPECS_B1_B7.md`、`B0_WORKER_REPORT.md`。
- `REPAIR_RESPONSE.md`：R01–R12 每条列原问题、修订位置、静态证据、仍属未来验收的内容，不把文档修订称作代码缺陷已修复。
- 完整命令与静态检查证据、精确 SHA-256 manifest；只有本轮真实执行的文档/配置/AST 检查结果。

文件之间必须一致。一个实现者按新文档就能确定类/参数、数据角色、组别支持、训练标签、阈值、种子、比较家族、推断与失败政策。B0 不能验证的外部算法能力保留 UNVERIFIED，并指定 B3 正反例；不要为了清零状态提前安装或运行模型。

完成前复核原始六份 B0、master plan、已有 15 个候选文件 hash 未改变。测试栏写 `NOT RUN — B0-R1 permits static document/source checks only`。使用协议第 9 节全部精确标题：

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

此次不再询问普通模型/编码选择；采用以上与 master 一致的默认值。若静态来源确实不足，列明确缺口及后续验证门槛，完成其余文档工作。现在只执行 B0-R1，完成后停止，不能自行进入 B1。
