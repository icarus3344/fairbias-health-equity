# Gemini：执行 FairBias R1A-R1 定点返修

你是 implementation worker；Codex是独立监督者。工作目录 `/Users/lkc/Downloads/code_v_0_3`。用户希望以FairBias为主方法发表应用论文，比较预测表现与其他论文的公平性方法；不能把“希望FairBias最好”作为筛选结果的规则。

**本提示已经授权下述R1A-R1实际修复与人工验证，无需再提交泛泛implementation plan等待许可。上轮PASS申请被拒绝；保留所有候选代码和运行作为证据，不做Git回滚、不删除。按A→B→C执行，完成后提交协议报告并停止。R1B、真实数据、下载安装和Git提交均未授权。**

## 0. 必读、输入与修改边界

先读 `docs/AI_EXECUTION_PROTOCOL.md`、`AGENTS.md`、`GEMINI.md`；再读：

- `docs/reports/FAIRBIAS_R1A_SUPERVISOR_REVIEW_20260914.md`
- 同名 `_evidence/verification.json`、`diff_from_pre_R1A.patch`、`source_symbols.json`
- `_evidence/probe_run_123437_880986/results.json` 与 `probe_source.py`，了解真实反例和注入限制，不把诊断计数当接受结果
- `_evidence/worker_log_analysis.json`
- `docs/plans/GEMINI_FAIRBIAS_R1A_IMPLEMENTATION_PROMPT_20260914.md`（算法规则与未取消要求仍有效）

分支须为`research/nhis-fairbias`，HEAD须为`67e6659fa65249a8842e34af5d8969629efe4bca`，无stage。当前源码须与本次verification的候选输入哈希一致。该哈希是“被审查的未验收候选身份”，不是已通过基线。发现额外变化应报告具体文件，不能改预期值蒙混。

先在新目录保存本轮源码前像、hash、Git状态与命令。14个继承文件和.gitignore不动，不读取原始CSV来计算hash。只用Git元数据区分：对当前HEAD的新改动、对保护tag的历史差异。已知`.gitignore`历史差异来自b595e59，记账保留，不宣称逐位等于038897，也不擅自恢复。

允许修改的源码范围沿用R1A：`src/fairbias/{enhancement_state,enhancement,enhancement_contracts,bias_metric,evaluator,config,mitigation,prediction_contracts,application_metrics}.py`；`src/nhis_fairbias/{adapter,preprocessing,survey,d8_enhancement_runner}.py`。确有对应缺陷才改。

允许修改本轮两个工具 `scripts/_fairbias_r1_guard.py`、`scripts/run_fairbias_r1_guarded_tests.py` 和五个现有 `tests/synthetic/test_r1a_*.py`。允许新增一个`tests/synthetic/conftest.py`用于人工fixtures/计数，但不得导入旧tests、放宽guard或读取历史数据。禁止修改benchmark包、旧tests、历史脚本、治理文件、旧报告、旧计划、release及既存runs。

新增输出只能进入独占创建的 `runs/r1a_r1_<UTC>_<id>/` 与 `docs/reports/fairbias_r1a_r1_worker_<UTC>_<id>/`。所有失败尝试保留；不覆盖、不rm、不清理“过期”证据。不得stage/commit/push、访问网络、安装包或读取微数据。

## A. 先补证据保全并修执行工具

### A1. 如实登记上轮违规和证据缺失

基于监督verification的attempt_retention，盘点7个上轮R1A run目录；只读这些运行中的stdout/stderr、preflight、telemetry、sentinel/test摘要、guard日志，不访问其他历史runs或release。五个已删除report目录对应的原run仍在；在**新的恢复清单**登记每个原run路径、文件大小/hash、存在/缺失项，标记`RECONSTRUCTED_INDEX_FROM_RETAINED_RUN`与生成时间。无需再次复制数百MB日志，可逐文件哈希引用。

不要重新创建原目录冒充未曾删除，不声称恢复了不存在的原报告或原manifest。将额外103测试记为上轮超范围运行；旧测试有D6聚合工件依赖，本轮不能重跑它们。可保留上轮pass计数为worker陈述，不能升级为本轮合成验收。

### A2. 修guard，先保证启动与日志路径

仅用已安装Framework Python。新正式入口必须以以下方式启动父控制器，子解释器同样使用`-I -S -B`：

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -S -B scripts/run_fairbias_r1_guarded_tests.py --phase r1a-r1 --output-root /Users/lkc/Downloads/code_v_0_3/runs
```

你需先实现此phase。-S后仅显式加入当前解释器已核验的site-packages绝对路径，不执行用户site、sitecustomize、usercustomize或.pth。安装常驻guard后才导入项目、pytest和第三方包。禁用插件自动加载、pyc写入，临时文件/cache只写唯一run。父控制器允许固定参数的只读Git检查、`ps -o rss= -p <本次child_pid>`和同一解释器子进程；禁止shell、任意外部命令和子进程再生。

必须落实：

1. 日志sink先建立，日志写入不再调用触发自身记录的open；使用固定已核验fd或等效方案。不得关闭全局guard规避递归。限定事件内存和磁盘预算，摘要含事件总数、拒绝数、截断/丢弃数；日志损坏或预算耗尽不能伪PASS。关键拒绝必须可追溯。
2. 源码只能是明确允许的src源码和本轮工具/测试；不能读src/scripts中的任意CSV/JSON。配置只允许features.json、study.json及manifest明确列出的无个体信息配置。Python stdlib/site-packages用具体解释器路径，不允许任意sys.path、用户site、整个项目、/etc或/Library/Preferences。若需要系统时区，只列实际必要的明确文件。
3. 四种open路线统一解析绝对/相对路径、符号链接、flags和dir_fd，覆盖O_TRUNC。未知fd拒绝；确需标准输出/受控日志/已验证fd须登记能力和方向，不能统一放行整数或/dev目录。
4. 删除、改名、替换、截断、链接与目录写操作也受边界约束；常驻期间不能变更run外文件。禁止所有未声明网络活动、DNS和UDP发送，以及fork/exec/spawn/system/Popen等进程出口。检测到安装audit失败即失败，不能吞异常后宣称installed。
5. 先用**本轮新造的人工文件和假repo目录**验证绝对路径、软链接、相对路径前缀、四种open和fd/flags。禁止以真实data_COMPAS/data_Credit_Card或真实parquet作哨兵。
6. 网络/进程验证只能把明确的GuardViolationError及对应事件当成功；ConnectionRefused、FileNotFound、普通OSError等均不是阻断证据。可在独立防护之内做回调/拦截器验证，区分“外层兜底阻止”与“被测guard生效”，不实际发网络包或启动任意命令。缺guard必须FAIL，不能skip。
7. 未预期拒绝即标该运行失败，保留完整原因并以新run重试；不能因为pytest exit=0便忽略收集/导入期间异常。哨兵失败不得进入项目测试。
8. 保存源码与测试执行前/后的hash；确保导入代码对应源码（避免旧pyc混淆）。解释器/包版本、运行argv/cwd、输出目录、guard参数在机器摘要记录。
9. 维持15分钟/4GiB控制；监测失败时不声称资源限制成立；区分时间、内存、监测故障，kill后wait回收。0.5秒ps最大值只能叫采样最大RSS，不能称精确峰值或硬内存上限。

guard只提供明确Python路线的防护，不称OS沙箱；本轮不允许原生额外文件访问、ctypes调用系统IO或外部进程来绕过它。A阶段完成并通过自检后可在同一授权内继续B/C；这不是worker自批准项目gate，最终仍由Codex独立验收。

## B. 保留已有效修复，补齐具体缺陷

| ID | 必须完成 | 最小行为证据 |
|---|---|---|
| C01 | 配置参数采用带类型的稳定无损序列化，包含模块+类型身份、真实get_params(deep=True)与实际scaler构造参数。禁止str(v)代替结构、禁止get_params失败降为None。遇到不支持的对象明确拒绝；不能把非有限值转字符串。运行时状态/缓存/循环明确记录v2，历史明确v1；版本拼写错误和歧义键拒绝。 | LR C变化真实使同一engine缓存失效、不变复用；数组中部变化不得碰撞或须明确拒绝数组；int1与str1区分；NaN/坏get_params拒绝；scaler指纹与实际执行对象一致；v1历史标识仍可显式复现。 |
| C02 | 按真实schema消费关系验证registry。至少substantive_codes、semantic_type、missing规则、feature_lists及顺序、harmonized映射、实际拟合规则身份；要求必需元数据，不以不存在的valid_range字段代替真实范围。避免外部修改共享嵌套字典悄悄改变拟合后语义。 | 真实NHISPreprocessor.fit的匹配正例；改变AGEP_A substantive_codes、类别有效码、列表顺序的拒绝/明确重拟合要求；年龄90等人工输入能显示前后处理差异。缺registry/provenance单独测试，不仅错误年份。 |
| C03 | 身份改为无歧义结构编码，明确来源注册/未知来源策略。混用未知与已知来源不能据此认定同年度同ID独立。拒绝None/空/不支持的ID和来源；多个年度字段冲突必须明确处理。保留前导零和合法不同来源同号；通用核心不限制2022–2024。 | source=`a:b`,id=`c`与source=`a`,id=`b:c`区分；已知/未知source交叠例拒绝；同source部分重叠/重复拒绝；2023/2023.0等价；bool/非整数/非有限year拒绝；同特征不同ID接受。 |
| C04 | 保留三条已接入概率主路径；验证row_sum_tol有限非负及合理界限、期望二分类标签和正类声明、数值dtype/空样本策略。明确概率验证异常，不伪造硬标签风险。 | NaN/Inf/负tol拒绝；坏行和不因tol异常通过；classes反序、错长度/shape、单类、未知类、无predict_proba分别走候选与终局；含真实LR正例及声明过的人工classifier。 |
| C05 | 保留正经验支持过滤，并补加权Categorical/合并后空level真实路线。 | 二类散度1；加空level不变；加零权重level不变；合并后空level不扩大K。 |
| C06 | 新增明确命名equalized_odds_gap=max(TPR_gap,FPR_gap)，保留equal_opportunity/TPR_gap及legacy兼容；不要改主研究估计目标。严格硬标签y/yhat=0或1，shape/长度/组标识有限且合法，拒绝q的小数输入。预声明expected_groups贯通FairEvaluator和D8终局；若未声明只能标observed-groups描述性，不给主指标资格。逐指标独立状态。 | TPR相同/FPR不同：EOpp=0、equalized_odds=1；某组无负类：DP和有支持的EOpp可估计，但equalized_odds不可估计；七组缺一不可给主EO；非法yhat=2和小数拒绝；终局调用而非仅helper通过；legacy SP=2/3与应用DP=1命名不同。 |
| E01 | 保留最大正权重缩放与validator溢出拒绝，比例直接入口增加长度、唯一索引、顺序、形状验证；原raw缺失代码/正权重mask口径明确保留，不暗改分母含义。 | 两个1e308给0.5、validator拒绝总和溢出；[a,b]与[b,c]索引直接入口拒绝；重复/乱序索引和二维权重拒绝；全零/组内无支持/公共尺度不变性。 |
| G01 | 保留实际multigroup_aggregation传播；补有效配置与指纹一致性、实际几何与搜索行为证据。修Sequence导入等同范围小错。 | spy实际捕获author_max_pair；两上下文向量反例；其余GEOM集成见C。 |

legacy输出可继续兼容，但必须从名称/metadata看出公式与可估计口径，不能把旧EO名直接替换成另一个定义。新增公共接口可用兼容的keyword参数；不得为修复C06随意更新旧结果或旧tests的期望值。

R1A仍不实现主benchmark完整加权q指标；硬标签helper应拒绝q，R1B/R2再接独立的决策概率期望计数与风险p能力契约。保留核心AUROC效用，应用C-BA目标按后续独立配置实现。

## C. 用真实调用路径验证，禁止“标题覆盖”

只收集五个R1A合成test文件，必要的新conftest只服务这五个文件。不得`pytest tests`、不得重跑那103个旧测试。每个test报告问题ID、调用路径、实际模型fit次数、是否mock/注入、数据规模、有效参数与状态。

必须补的集成：

- **AE**：实际调用enhance_step；在同一engine重复不变配置、改变C、改变有效scaler、微小阈值/状态差异，观测真实候选缓存命中/失效及状态记录。只有hash比较不算集成。
- **Joint**：实际D8 run_arm的Joint提交/循环路线。历史adapter/provider/阈值数据依赖只能注入人工对象/常量；不读D6。控制流反例可注入人工效用，但另有真实LR集成；不能用普通set代替Joint。
- **BM**：真实小型MDS后调用BM候选搜索/提交，记录选择属性、指数尝试、epsilon/NMI、stop原因；只调用calculate_epsilon不算BM。人工例要使其早停或显式预算停止，不缩短作者幂流。
- **边界**：H=1近全上下文；BM `[3,1/3,5,1/5,...,1999,1/1999]`共1998项，AE六网格不变；restart每次从0、monotone_cursor另列；最高偏差属性失败的stop行为；NMI公式与实际phi_threshold；candidate严格<epsilon和终局<=分开；非有限/无候选/no-op/未收敛/预算耗尽分开，不写全局最优。paper模式不能因测试需要放开AE。
- **隔离**：同一人工F/C固定，任意扰动独立人工S/T的特征、标签、组/权重不得影响F拟合和C选择结果。另测F/C误复用记录被拒绝、对象后续原位修改被发现。只改fit_X然后验hash不能代替S/T不变性。
- **规模与拟合真实性**：每分区≤200行；真实classifier/AE/BM例≤8预测特征。registry契约因既存schema要求，可以例外使用完整24字段、每分区≤8行来真实fit预处理，不在该宽表上训练classifier；在报告单独列明。不得生成89,802行来满足固定人数检查，不得手填_fitted_record作为真实fit正例。允许仅绕过旧全年固定人数检查和文件运输层，比较器与实际清理/fit不mock。

参数、源码、错误输出应保存在最终机器结果中，不能抄旧行号；重新用AST导出修改后位置。每个合成例的S/T与历史2023/2024文件无关。正常预期拒绝与非预期失败分开计数，后者不可被catch-all、skip或改断言掩盖。

## D. 交付、最终验收停止点

每次运行都产生唯一目录、test节点和退出码、trace、实际fit/行数、guard事件与摘要、argv/cwd/解释器、环境版本、资源监控状态、全部源码/配置/test前后hash、所有尝试索引。测试数量是描述信息，不是验收目标。

最终审计包包含：

1. `WORKER_REPORT.md`：严格协议第9节标题。
2. 问题→修改函数→测试节点→机器输出字段矩阵，每项标FIXED_CANDIDATE/PARTIAL/OPEN；worker不标gate ACCEPT。
3. 源码patch和前后hash，注明与本轮preimage比较，而非误把全部HEAD dirty变化算作新改动。
4. 原失败run恢复索引、本轮所有成功/失败尝试索引；既存缺失证据如实列OPEN。
5. 机器结果与报告一致性检查。正式报告由最终结构化结果生成；不要把手抄表称机器自动生成。
6. 报告定稿后生成MANIFEST，包含报告、源码/test/config身份、运行元数据和证据引用，排除manifest自身；不可覆盖旧manifest。哈希表只证明绑定/不变性，不升级科学结论。

明确写出：R1A-R1已做与未做；旧F01/F02主benchmark问题仍OPEN；R1B/R2/R3/R4未执行；没有真实调查推断、FairBias优势或论文保真认证。结尾必须为：

`STOP — waiting for Codex review.`

## E. 完整研究路径仍保留，当前不执行

验收R1A后，R1B真实F-only FairBias与外部RW/LFR/EG-DP/EG-EO/TO-EO、Unmitigated对接，统一LR/GBDT和p/q/yhat能力；R2验证全年设计+domain、配对复制、非光滑gap覆盖；R3按已审查76条件、5种子在合成演练后做获准开发；R4冻结后做回顾性2024评价。

主对比仍为Arm001/003、LR、预定EO操作点下FairBias-BM相对五个外部方法的ΔBA/ΔEO，共20个差值家族；B=2000、有效复制要求、df与SE、失败配置/seed均按既定规格处理，不为结果“变好”改变家族。2024此前已被看过，不能恢复成盲测措辞。Arm004路径解释需第一处分歧和受控因素证据。FairBias是研究主角，结果必须允许它在某些维度不优。
