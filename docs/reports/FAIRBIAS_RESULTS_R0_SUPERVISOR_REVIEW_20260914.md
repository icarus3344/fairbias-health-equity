# FairBias RESULTS-R0 独立监督审查

审查日期：2026-09-14。对象：`docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/` 六项交付及本轮提交记录。branch=`research/nhis-fairbias`，HEAD=`67e6659fa65249a8842e34af5d8969629efe4bca`。

**Gate verdict：REPAIR。撤回过度结论的方向接受；静态证据和实现规格尚未完整通过，不激活 R1，不重跑真实数据。** 这轮交付是审计响应，没有修复算法。上一轮 F01/F02 两项 P0 仍然开放。本轮没有发现新的 P0 实现变更，也没有证据证明旧输入遭到篡改。

审查只读取源码、文档、配置和明确列出的历史聚合 JSON，运行标准库静态核验；未运行 pytest、未导入项目、未拟合模型、未读取个体微数据、未访问网络。保留全部旧文件，仅新增本报告、证据和下一轮交接。没有 stage/commit。

## 已经确认的进展

- 撤回“全局 Pareto 最优”“高保真 BM 学习”“2024 独立盲测”“临床/法律合规保证”“全量 LFR”等表述是正确的；历史结果应继续标为探索性记录。
- 最终 manifest 的四臂方法数据及配对差值与原始 JSON 一致；四臂样本数正确；确有 27 个历史 `VALID` 状态和 1 个 `NOT_SUPPORTED`，这些是执行状态，不是方法学验收。
- RW 在这份历史汇总的 **BA/EO 二维点估计** 上四臂均优于 FairBias。这不能证明修复后的方法优劣，也不能推广到包含 DP 的三维 Pareto 判断。
- 旧 15 项候选、当前 23 项 benchmark 源码/测试、上一轮 34 项输入均与记录的哈希一致；上一轮证据 manifest 的 41/41 项匹配。原始运行 JSON 未变。14 个继承文件相对保护 tag 无变化；`.gitignore` 相对 HEAD 无变化，其相对 tag 的历史差异不是本轮产生。

完整机器核验见 [verification.json](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_RESULTS_R0_SUPERVISOR_REVIEW_20260914_evidence/verified/verification.json)，正确的聚合提取见 [corrected_aggregate.json](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_RESULTS_R0_SUPERVISOR_REVIEW_20260914_evidence/verified/corrected_aggregate.json)。以上哈希检查范围不是整个磁盘的历史写入审计。

## P1：必须在进入实现前解决

### R01：底层缺陷没有对应的实施范围，包装器无法替代核心修复

位置：[新规格:40](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/REPAIR_IMPLEMENTATION_SPEC.md:40)、[C01–C06/E01 对照:53](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/REPAIR_IMPLEMENTATION_SPEC.md:53)。

规格主要允许改 `benchmark/`，却声称关联旧 C01–C06/E01。实际 C01 的八位小数状态身份和缓存调用在 `src/fairbias/enhancement_state.py`、`enhancement.py`、`enhancement_contracts.py` 及 D8 runner；C02 在 `src/nhis_fairbias/adapter.py` 与真实预处理对象；C05 在 `src/fairbias/bias_metric.py`；E01 在 `src/nhis_fairbias/survey.py`。只给 FairBiasAdapter 增加指纹或在新 metrics 中防溢出，不能宣布这些公共路径已修复。[旧缺陷定位:46](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_CONSOLIDATED_AUDIT_AND_APPLICATION_PLAN_20260913.md:46)

本轮这些核心文件的哈希仍与上一轮相同，所以不存在“写了新规划，旧问题自然消失”的依据。应把 R1 拆为 **R1A 核心契约修复 → R1B 主方法与对照适配**；逐项列出实际函数、最小文件范围、正反例、调用方与独立验收。不走旧路径可以是明确设计，但不能把“绕开”写成“关闭”。

### R02：真实 BM 的配置贯通与 AE/Joint 执行方式仍未落实

位置：[BM 规格:40](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/REPAIR_IMPLEMENTATION_SPEC.md:40)、[AE 矩阵:116](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/REPAIR_IMPLEMENTATION_SPEC.md:116)。

“调用真实引擎、作者幂序列、restart”还不足以实施。当前 [evaluator:294](/Users/lkc/Downloads/code_v_0_3/src/fairbias/evaluator.py:294) 没有向 `compute_dphi_matrix` 传递多组聚合模式，不能因底层存在 `author_max_pair` 分支就认定它生效。该分支实际计算 `abs(max(v1)-max(v2))`，再在上下文间平均，不是 `max(abs(v1-v2))`。[bias_metric:413](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:413)

需要冻结并测试 H=1 的“最多排除一个属性”上下文、BM 的 `[3,1/3,5,1/5,...,1999,1/1999]` 交错幂流、restart、NMI gate、F 内 epsilon reference 及倍率、MDS 维数和停止预算。六值 `DEFAULT_POLY_GRID` 属于 AE，不能替代 BM 幂流。[上下文:99](/Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py:99)、[幂流:79](/Users/lkc/Downloads/code_v_0_3/src/fairbias/config.py:79)

另外，当前 paper 模式显式拒绝 `use_accuracy_enhancement=True`，所以“16 个 AE 条件”必须有命名清楚的应用执行路径，不能直接打开该开关；AE/Joint 使用同一配置的 `epsilon_candidate`、slack=0，接受不等式与终止可行性需要分别声明。[config:240](/Users/lkc/Downloads/code_v_0_3/src/fairbias/config.py:240)、[原交接要求:55](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_BENCHMARK_RESULTS_REPAIR_PROMPT_20260914.md:55)

### R03：接口修复的验收过弱，可能引入新的训练语义错误

位置：[EG/TO 规格:45](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/REPAIR_IMPLEMENTATION_SPEC.md:45)、[TO 验收:70](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/REPAIR_IMPLEMENTATION_SPEC.md:70)。

- EG 写成“把外部 sample_weight 路由给基础预测器，或报不支持”。外部调查权重与 reduction 内部成本权重是不同对象；只路由到基础模型不等于实现调查加权 Moment/目标。当前主实验无调查权重，最明确的规则是拒绝不支持的外部调查权重，同时用 spy 验证 reduction 自己生成的成本权重确实到达 oracle。不得据此宣称 EG 已支持总体加权公平约束。[master:118](/Users/lkc/Downloads/code_v_0_3/docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md:118)
- TO 的“调用 unified fit 不再 TypeError”不足以验收。只修关键字，会继续执行 [同一 X/y/A 同时拟合和校准:91](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_threshold_optimizer.py:91)。必须在任何拟合之前拒绝缺 C 的统一入口，或让统一入口显式接受互不重叠的 F/C；合法两阶段调用须另有正例。
- 确定性方法将 q 标为 NOT_APPLICABLE 可以成立，但公共 runner 必须显式分派到阈值化的 yhat；不能继续所有方法统一调用旧 `predict_decision_proba`。需要完整调用例与“未在 C 冻结阈值则拒绝”的状态契约。阈值规则直接继承 master，不能只写“例如最大 BA”留待运行时自由选择。[master:150](/Users/lkc/Downloads/code_v_0_3/docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md:150)

### R04：76 条件的总数没变，但科学设计发生未解释的漂移

位置：[新矩阵:121](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/REPAIR_IMPLEMENTATION_SPEC.md:121)、[前次 registry:212](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_b0_r1_20260914/EXPERIMENT_REGISTRY_DRAFT.md:212)。

前次草案的四个加权训练条件是 **Arm 003/004 × LR × {Unmitigated,FairBias-BM}**；本轮改成 **Arm 001 × {LR,GBDT} × {Unmitigated,FairBias}**。两者都等于四，但回答不同问题；新规格没有列为变更提案或解释原因。前次两条几何路径是 Arm004/LR 的 BM 与 Joint fixed-2D；本轮只写“维度策略比较”，没有锁定两条是什么。

应保留原草案范围或显式提出版本化变更，不得冒充已批准的设计。本轮仍缺逐行 `condition_id/config_id/expected_seed_ids`、具体参数和适配能力映射。76 是顶层条件数，不是拟合次数。R2 的 2 GB、R3 的 3 GiB、master 的 4 GiB 也需区分组件预算与进程总预算，说明修订依据，不把口头目标当实测证明。

### R05：两项输入哈希错误，证据报告不能直接作为 gate 输入

位置：[WORKER_REPORT:25](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/WORKER_REPORT.md:25)。

| 输入 | Worker 声称 SHA-256 | 独立重算 SHA-256 |
|---|---|---|
| 上一轮监督报告 | `2b143b8118029ff6d2146c245c7395e804f58c4228fe00b213b2c9399b1e9447` | `5ee400543c11ee5a7c559dc2e17ed82c8e5fa6c19cd6eca36b45843e4f96c8a2` |
| 上一轮 verification.json | `6955ee388a108b68aa4a2f8c5c7d81a97e68c8577eb372a1fa06700c25a0728c` | `9a55c87665a19746c4917f39dadc3b4e2a38d4146cd2274aee3241978256e89d` |

两份当前文件与上一轮保存的 manifest 完全一致。结论是 **worker 报告中的哈希不正确**；本轮没有证据判定伪造动机，也不能写成文件被改动。新交付应自动从同一事实清单生成哈希、字节数和行数，完整保留命令及生成脚本。

### R06：R1–R4 是路线图，尚不是满足本 gate 要求的可执行规格

位置：[新规格:26](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/REPAIR_IMPLEMENTATION_SPEC.md:26)、[原交接:23](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_BENCHMARK_RESULTS_REPAIR_PROMPT_20260914.md:23)。

文件没有实际测试解释器、命令、访问 guard 入口和可审计输出路径；也缺在 pytest/项目导入前安装 guard 的流程。仅写“strictly synthetic”和“Comprehensive synthetic test suite”不能约束旧测试导入 D6 历史工件，更不能确保状态、源文件和模型绑定。

R2 还缺独立方差参考、非光滑 gap 的覆盖验证、完整种子配置判定及 20 个对比的逐项清单；R3/R4 不能仅凭测试数或 B=2000 获得推断资格。这些要求原交接已经列出，本轮不是新增研究任务。下一轮只需定点补齐，不必重写 F01–F14 全文。

## P2：自动证据与叙述还需修正

### R07：机器表、Markdown 表和报告互相矛盾

- [manifest:12](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/AUTOMATED_STATIC_MANIFEST.json:12) 记 15 个 NaN，实际 JSON parser 回调记录 **8 个**。原因在 [提交记录:319](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_RESULTS_R0_SUPERVISOR_REVIEW_20260914_evidence/verified/worker_submission.txt:319)：`count(': NaN') + count(': NaN,')` 将带逗号的七项重复计数。使用解析器识别非标准常量，不用重叠子串计数。
- [测试表:83](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/AUTOMATED_STATIC_TABLES.md:83) 的 **19/19 个函数结束行**不等于 AST 的 `end_lineno`，而 manifest 的 AST 清单正确。大多多记一行，Arm003/004 parity 测试又少记一行。报告可以描述人工覆盖判断，但不得把人工位置标成 AST 自动导出。
- [磁盘报告:53](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/WORKER_REPORT.md:53) 写总计 1,775 行；实际六项 **1,708 行、110,296 字节**，最后聊天回复已改对但磁盘报告未同步。
- [命令:15](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/WORKER_REPORT.md:15) 仍用 `...`，六项交付没有保存生成器。原附件中有较完整的两次提取代码；第二次确实修复了取错 JSON key 的问题，因此不把第一版空提取误报为当前 bug，但 NaN 计数错误仍保留。

这些问题不意味着全部数字错误：当前 manifest 的 BA/EO/DP 和配对 CI/p 值与原始汇总一致。它们仍是存在方法学缺陷的历史结果，原始零 p 值和不完整设计诊断应原样标为旧实现输出，不当成有效显著性证据。

### R08：事实和推论仍有几处混写

位置：[F03 回应:79](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/FINDING_RESPONSE.md:79)、[addendum:43](/Users/lkc/Downloads/code_v_0_3/docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/RESULT_STATUS_ADDENDUM.md:43)。

`27451=21871+5580` 与旧 D6 provenance 支持“不能证明该字典只在新 F 拟合、不得用于主实验”，但仅凭人数相等不能独立证明具体 C 记录的集合交集。本轮未读取记录身份，应保留这个证据边界；无需为修正文案而打开微数据。

F05 中以源码块呈现的 `np.random.choice(len(X_F)) / self.lfr.fit(dataset_sub)` 不是实际代码；实际采用 `np.random.default_rng` 创建生成器后对 `df` 抽样，并显式传入 maxiter/maxfun。可标为伪代码，或逐字摘录真实有限行段。[实际实现:108](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_lfr.py:108) “全部 19 测试通过的原因”也应改为具体已观察的覆盖缺口，静态审查不能对每项历史通过作唯一因果归因。

## 下一步与项目判断

FairBias 继续作为论文主研究对象，但目前应优先证明“主方法确实运行、评价对象一致、比较预算及分区一致”。这些条件满足后，结果支持优势、条件性优势或无优势，都可以形成诚实的应用研究；当前无法承诺发表层级或方法必然胜出。

完整顺序维持：**定点修复本轮静态证据/规格 → R1A 核心契约 → R1B FairBias 与对照适配 → R2 调查推断、选优和 runner → R3 完整矩阵、合成演练及获准的 2022/2023 开发 → R4 冻结后 2024 回顾性评价与论文**。R1A/R1B 是对原 R1 的实施拆分，不删任何模型和稳健性条件。

已新增 [Gemini 定点修复交接](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_RESULTS_R0_TARGETED_REPAIR_PROMPT_20260914.md)。该交接只激活 RESULTS-R0.1 静态补正，不授权后续实现或真实数据运行。无需用户重复确认已授权的静态工作。

审计自身的首次核验脚本因旧清单实际为 list 而非 dict 退出；失败及已复制快照保留，修正后在新的 `verified/` 目录成功完成。详见 [证据说明](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_RESULTS_R0_SUPERVISOR_REVIEW_20260914_evidence/README.md)。未将失败尝试或任何静态计数记作测试通过。
