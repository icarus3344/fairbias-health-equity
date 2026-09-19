# FairBias RESULTS-R0.1 审查与监督补正

2026-09-14；branch=`research/nhis-fairbias`；HEAD=`67e6659fa65249a8842e34af5d8969629efe4bca`。对象为 `docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/` 七项交付。

**原交付判定：REPAIR，不能照其 R1 规格实施。监督补正后的静态交接包：ACCEPT，仅作为进入 R1A 的规格，不是算法、测试或实验结果通过。** 本报告、附属机器核验、修正版注册表和新的 R1A prompt 共同替代原稿错误内容；不覆盖原稿，也不再要求 worker 重写 R0 文档。仅激活 R1A 核心合成代码修复，R1B–R4 继续等待对应阶段验收。

本轮只做标准库静态检查和已安装依赖的元数据读取；未导入项目、未运行 pytest/模型、未读取个体数据、未访问网络、未修改实现、未 stage/commit。证据在 [verification.json](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_RESULTS_R0_1_SUPERVISOR_REVIEW_20260914_evidence/verification.json)。

## 已通过的静态部分

Worker FACTS 的 66 项文件指纹全部匹配；其中四臂人数、方法数据、配对差值、8 个 NaN 和 19 个 AST 测试位置均与源文件一致。Worker MANIFEST 的六项被散列输出全部匹配；七项交付总计 **141502 字节、3234 行**。上一轮监督证据 **137/137** 项仍匹配，未发现被检查旧输入遭到修改。

76 个顶层条件及两个排除单元存在，ID 无重复；四个加权训练条件恢复为 Arm003/004 × LR × {Unmitigated,FairBias-BM}，两条路径恢复为 Arm004/LR 的 BM 与 Joint fixed-2D。分开 R1A/R1B、拒绝 EG 不支持的调查权重、要求 TO 独立 C 的方向正确。上述通过范围不包括模型执行真实性或推断有效性。

## 必须纠正的发现

### S01 / P1：所谓“Exact Class / Function Scope”包含 14 个不存在的符号

位置：[R1_EXECUTION_SPEC:36](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/R1_EXECUTION_SPEC.md:36)。例如 `EnhancementState.compute_state_id`、`FairnessEnhancementEngine._evaluate_candidate`、`ScientificConfig.fingerprint`、`RecordKeyValidator.validate`、`SurveyProportionValidator.validate` 都不存在。不是命名偏好，而是未定位到实际修改路线。

实际路径包括 `hash_transform_state` / `StatefulCandidateTracker`、`FairAccuracyEnhancement.configuration_fingerprint/enhance_step`、`compute_configuration_fingerprint`、`EvaluationPartition._extract_record_ids`、`NHISStudyAdapter.__init__`、`weighted_binary_proportion` / `validate_survey_weights`。已从源码 AST 导出全部真实符号与位置，见 [source_symbols.json](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_RESULTS_R0_1_SUPERVISOR_REVIEW_20260914_evidence/source_symbols.json)。修复不得通过新造同名空壳来使错误规划“看似正确”。

### S02 / P1：算法公式与控制流解释错误，照做会改成另一种算法

位置：[规则:45](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/R1_EXECUTION_SPEC.md:45)、[CORRECTIONS:39](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/CORRECTIONS.md:39)。

- **H=1** 是从 available 特征集合最多排除一个属性，保留近完整上下文。available={a,b,c} 时是 {a,b}、{a,c}、{b,c}、{a,b,c}，不是“holding at most one attribute constant”。[实现:99](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:99)
- **restart** 指每次重新搜索该属性时从幂流头开始；不是停滞后遍历未访问特征。失败后换其他特征属于独立的 `failed_attribute_mode='next'` 工程行为；主路线为 highest-d_phi、失败停止。[幂搜索:376](/Users/lkc/Downloads/code_v_0_3/src/fairbias/mitigation.py:376)、[失败停止:580](/Users/lkc/Downloads/code_v_0_3/src/fairbias/mitigation.py:580)
- 新规格比较的 `max_{i,j}|v_i-v_j|` 与 `|max(v)-min(v)|` 对同一有限实向量本来就相等；真正需要区分的是两个上下文向量的 `abs(max(v1)-max(v2))` 与 `max(abs(v1-v2))`。例如 v1=(.9,.1)、v2=(.1,.9)，分别是 **0 和 .8**。v 的分量是 group-pair 的子集值，不是外部组 TPR。保留现存 `author_max_pair` 分支的含义，显式贯通配置；不把预测 EO 极差替换进几何。[实现:413](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:413)
- AE 实际 `DEFAULT_POLY_GRID=(1/7,1/5,1/3,3,5,7)`，不是 CORRECTIONS 写的 `(0.5,2,0.33,3,0.25,4)`。[实现:32](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:32)
- NMI 原判据是按候选属性计算 `(nmi_before-nmi_after)/(nmi_before+1e-10) <= phi_threshold`。不能改为一个未定义的全矩阵 NMI 比例阈值。[实现:212](/Users/lkc/Downloads/code_v_0_3/src/fairbias/mitigation.py:212)
- 达到 epsilon 是可行性，不是全局最优性证明；`OPTIMAL_CONVERGENCE` 应改为有界、明确的可行停止状态。候选处现有严格 `< epsilon` 与终局 `<= epsilon` 应分别保留并记录，不能写成所有位置统一 `<=`。

本审查确认的是当前实现含义；没有因为函数名含 author/paper 就重新认证其与论文完全等价。

### S03 / P1：主对比家族改变，且配置引用尚未可解析

位置：[主对比:187](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/R1_EXECUTION_SPEC.md:187)、[registry:32](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/CONDITION_REGISTRY.json:32)。

五个外部主对照应为 **RW、LFR、EG-DP、EG-EO、TO-EO**。本轮把 EG-DP 换成 Unmitigated，虽仍有 20 个对比，却改变了多重比较家族。Unmitigated 保留预测参考及描述性对照，不替换 EG-DP。[master:202](/Users/lkc/Downloads/code_v_0_3/docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md:202)

原 registry 的14种 `config_grid_ref` 只是字符串，没有具体可解析定义；不能直接驱动完整实验。部分 GBDT 条件仅配 seed=0，而已安装 sklearn 的源码说明分裂时存在特征随机排列及同收益分裂的随机性；不能仅据 subsample=1 判定无需训练种子敏感性。[本地 sklearn 源码:1425](/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages/sklearn/ensemble/_gb.py:1425)

已生成 [监督修正版注册表](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_RESULTS_R0_1_SUPERVISOR_REVIEW_20260914_evidence/REVIEWED_CONDITION_REGISTRY.json)：保留76条件，补16个具体网格定义（TO参数独立），逐项列20个主对比，GBDT统一使用预定五个训练种子。配置、seed与统计复制是不同层级；剩余估计器默认值须在 R3 从实际环境完整导出。此注册表状态为计划，不授权训练或数据访问。

### S04 / P1：C03/C06 的验收仍在改变公共接口或混淆统计对象

位置：[核心表:38](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/R1_EXECUTION_SPEC.md:38)。

通用 `EvaluationPartition` 不应硬编码只能处理2022–2024，也不能把拒绝所有 `2023.0` 当作解决等价记录身份的办法。有限整数值年份应规范化，非整数/布尔/缺失拒绝；年度研究范围由 NHIS 应用层限制，ID字符串前导零按来源保留。

C06 是公共预测指标的命名、聚合、缺组口径问题；用“单群体 d_phi=None”替代预测 DP/EO 的正反例，无法关闭它。内部几何、历史 mean-pair 指标和新应用 max−min 指标须明确分开。候选与终局正类校验也不能在另一节又退回无条件 `predict_proba[:,1]`。

### S05 / P1：访问 guard 还缺真实可执行边界

位置：[guard:163](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/R1_EXECUTION_SPEC.md:163)。

`socket.connect` 不是模块级函数；文件部分仅写允许目录，没有规定拦截 builtins.open、io.open、Path.open、os.open/审计事件，也没说明允许的 Python 标准库与依赖路径。单个 `/tmp` sentinel 不能证明这些路线和子进程都被覆盖。R1A先实现并验证 guard，失败就停，不得删掉拦截来换取测试通过。新 prompt 已给真实入口、分阶段 preflight、明确测试节点、库路径与覆盖边界。此阶段不宣称 Python guard 是操作系统级隔离。

### S06 / P2：机器证据正确，叙述却再次手工改错

位置：[WORKER_REPORT:28](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/WORKER_REPORT.md:28)、[自动生成声明:101](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/CORRECTIONS.md:101)。

四项监督输入的文本哈希错误，而 FACTS 的66项全部正确；报告的总行数3235应为3234。`generate_static_repair.py` 只生成 FACTS，没有渲染 CORRECTIONS 或 WORKER_REPORT，因此“全部由 FACTS 自动生成”与源码不符。它仍用 `open(...,'w')`、`exist_ok=True`，缺文件只加入不输出的 missing_files，哈希不匹配也仅输出布尔值而不退出。[生成器:115](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/generate_static_repair.py:115)、[输出:257](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/generate_static_repair.py:257)

本轮没有再次执行该生成器；用独立脚本检查所有事实，并将其保留为历史工具，不让它作为 R1A 验收入口。错误哈希及真实值均在 verification 中逐项保存，不推断动机，不称作旧文件篡改。

### S07 / P2：补正文案仍有伪源码和过强推论

位置：[CORRECTIONS:132](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/CORRECTIONS.md:132)。标作 verbatim 的 LFR 片段仍使用不存在于该段的 X_F/df_F/dataset_sub；真实实现用 df、bld_sub。[实际源码:108](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_lfr.py:108) “人数等式证明整个2022含C”仍越过集合证据；只可说旧状态无法证明为新F专属，不能用于主方法。未核验的完整年度人数约32600应删除；R2以后从获准数据产出完整设计诊断。

另外，阈值候选0和1配合 `p>=t`，在p含1时不能表示全阴性策略。未来统一阈值器须显式表示边界策略，输出完整规则；本轮不据此声称当前BA最优值一定改变。

## 本轮补正完成后的执行安排

**下一步只做 R1A。** [监督定稿的 R1A 实施 prompt](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_IMPLEMENTATION_PROMPT_20260914.md) 给出真实源码入口、最小文件范围、算法规则、合成验收例和停止点。Worker直接按此实现，不再重写R0方案。原先错误规格不再是实施依据。

Framework Python 3.13.2 的元数据复核为 numpy2.2.6、pandas2.3.1、scipy1.16.1、sklearn1.7.1、pytest8.4.1；这不代替运行时兼容性测试。该环境无 fairlearn/aif360，R1A不需要这两包，R1B依赖处理仍未授权。

R1A完成必须提交源码差异、guard日志、真实对象正反例、失败和旧兼容证据，再独立审查。原两项P0（benchmark主FairBias没有真实BM、p/q混用）在R1B验收前仍为OPEN。R2的设计方差/非光滑gap覆盖、S选优及失败种子；R3的完整矩阵与数据适配；R4的冻结后回顾性2024评价均保留，不因本次静态补正而获得通过资格。FairBias继续是主研究对象，比较结果仍允许其他方法更优。
