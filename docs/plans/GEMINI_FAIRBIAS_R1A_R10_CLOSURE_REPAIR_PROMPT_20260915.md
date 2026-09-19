# 交给Gemini：FairBias R1A-R10收尾修复（2026-09-15）

你是Gemini implementation worker。本提示是当前R1A修复的生效任务。Codex已完成R9独立复核：**REJECT交付；R7-03指定内部拟合契约CLOSED，loaded来源缺口CLOSED，剩余三类P1收尾。** 保留当前有效代码，不整体回滚、不删旧包、不自批ACCEPT。

先读[AI执行协议](/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md)、[AGENTS.md](/Users/lkc/Downloads/code_v_0_3/AGENTS.md)、[GEMINI.md](/Users/lkc/Downloads/code_v_0_3/GEMINI.md)、[R9监督报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R9_SUPERVISOR_REVIEW_20260915.md)及[前序R9提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R9_TARGETED_REPAIR_PROMPT_20260915.md)。本提示收敛范围，前序未收敛边界继续生效。

读取R9监督_evidence中的verification.json、evidence_verification.json、metadata_verification_v2.json、delivery_verification.json、probe_run_101411_521520/results.json、final_verification.json和MANIFEST.json。注意：4组完成不是4组契约合格；生成器诊断针对当前d57bcbc版本，该版本与worker清单承诺f89c215版本不同，不能补证已删除的历史7项诊断。

## 范围、起点和中断续作

- 分支`research/nhis-fairbias`、HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`，无暂存；核对R9监督117项输入及快照，记录本轮前像。外部变化先明确归属，不覆盖。
- 可改：`scripts/run_fairbias_r1_guarded_tests.py`、必要时`scripts/_fairbias_r1_guard.py`、`tests/synthetic/test_r1a_guard.py`中的日志/证据验证节点及必要人工fixture。本轮新报告目录可新增纯stdlib生成器/校验器/诊断脚本。
- **已关闭的test_isolation_f_c_and_s_t_perturbation函数保持只读**；不要再修改其模型/scaler spy、S/T扰动。其现有回归由完整受保护suite保留。全部生产模型、预处理、指标、benchmark代码、配置、JSON/其他旧测试、治理文件只读。
- 14继承文件、`.gitignore`及全部R9和更早工件只读。不修改/移动/删除scratch/test_generator_isolated_probes.py，不重建已删除probe_work来假装恢复历史。
- 每次运行新建`runs/r1a_r10_<UTC>_<id>/`；本轮生成器、静态校验和交付写到`docs/reports/fairbias_r1a_r10_worker_<UTC>_<id>/`。所有人工交付故障在新的独占子目录，每次attempt不能复用之前的目录。
- **所有项目导入、模型/scaler调试、pytest和单节点必须从常驻guard入口执行。** 仅sklearn、人工数据、-I -S -B、import失败均不豁免。不要再直接python -c运行MinMaxScaler.fit或导入FairBias。当前任务无需任何新的模型调试。
- stdlib静态AST/hash/文本读取和纯stdlib交付生成器/其人工故障可直接执行，必须遵守新目录和留存规则；不得借此import/exec模型项目源码。需要实际调用runner项目校验函数时走guard入口，不把它包装成“纯静态读取”。
- 不联网、不安装、不读真实微数据、不跑旧全量tests或正式实验、不暂存/提交/push。保留15分钟/4GiB采样约束；模型人工分区≤200行、≤8特征，现有隔离fixture8行/2特征即可；完整schema24字段、fit分区≤8行。
- 保持H=1、完整1998项BM幂流、AE六网格、几何/revisit和冻结研究计划。不得为了通过交付校验改变结果或成功阈值。

遇到流中断后：重新读本提示与本轮最后一条持久attempt记录，核对源码身份和目录状态；未完成attempt保留为INTERRUPTED/UNKNOWN。若目录已有MANIFEST，整包只读；继续需要新attempt目录。**禁止先rmtree清理probe目录，禁止删除MANIFEST重建，禁止通过新的运行补填旧退出码。** 本轮保存每条实际命令、起止、退出、输出路径和源码身份。全部批次已授权顺序完成，不需反复询问用户。

先静态增加phase和必要白名单，模型/项目执行统一：

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -S -B scripts/run_fairbias_r1_guarded_tests.py --phase r1a-r10 --output-root /Users/lkc/Downloads/code_v_0_3/runs
```

## 批次一：保留事实并建立新交付身份（R7-01）

1. 新补充记录R9附件149、164行两次直接模型尝试；后者若声称ModuleNotFoundError/exit1，区分worker声称与可恢复的真实输出。缺证据写UNKNOWN；模块找不到不是guard阻止IO。
2. 记录334、336、339、345、350、356行六次scratch诊断调用及357行删除命令。当前probe_work不存在，结果无法从现存源码恢复。不要把六次合并为“7负例全过，exit0”。
3. R8两次生成器入口的逐条退出仍无持久依据，保持UNKNOWN，引用旧worker的0/1声明但不当作核验事实；不要再次填两个0。所有历史修正写本轮补充表，旧报告不改。
4. 保留R9清单f89c215/29746与当前生成器d57bcbc/30206的矛盾。若没有原始源码版本，准确声明未能证明生成时身份；不能用当前文件重新计算R9清单。下一轮只建立新的可审计身份。
5. 新生成器版本与诊断脚本先保存到本轮目录、记录hash，随后诊断；改动后再测对应受影响用例，每次新目录。最终冻结源码后生成成功包，清单包含生成器/校验器/诊断来源/完整case结果/退出记录。最终包之后不再编辑任何包内文件。

## 批次二：精确绑定闭合事件路径（R7-02）

1. 用真实父控制器已知的run目录/summary路径验证尾事件，规范化后完整比较。不要用endswith、basename或reason中的文字推断它属于当前run。人工副本应显式携带被审计run的预期身份；不要因为复制后所在目录不同误拒正例。
2. 保留现有严格计数、登记操作/目标/原因/模式和loaded/compiled类型/集合/hash校验。为同一真实裁决路径验证：合法2120→2122记录通过；两条尾目标换成`/unrelated/run/test_summary.json`或`/unrelated/run/not_test_summary.json`均拒绝。删除1/2条、替换action和已修复登记反例继续拒绝。
3. 所有坏材料只用新人工副本，失败明确传播到非零退出/判决，不能只给helper正常结果。测试因权限/导入错误失败不等于目标契约生效。
4. 准确描述编译覆盖：pytest同buffer rewrite/hash/compile已实现并保留；剩余PARTIAL是runner首次入口编译未被绑定，函数开始后重读不能替代它。本轮不必扩大成OS沙箱工程，不假称“pytest架构无法捕获”或“补偿等价完整”。

## 批次三：让完整校验决定最终化（R7-04）

保持小而明确的validator，修复现有要求，不再写一版仅拼Markdown的新模板。

1. 在任何成功报告/最终MANIFEST写入前，验证必需preflight、summary、execution_verdict、telemetry、pre/post/loaded/compiled来源和完整guard日志。文件缺失、类型/字段错误、hash不一致、子项FAIL与总PASS矛盾，都拒绝成功最终化。来源数据与当前run/测试/脚本身份绑定，不能只信外层PASS。
2. PASSED记录必须对应真实且唯一的完整node集合与要求的测试，核对源码/实际收集执行信息；不能仅比较≥88和去重。未来测试数由实际结果产生，不以固定数量替代覆盖。
3. 7个trace按真实契约核对完整必需字段、类型、值及对应测试/候选关系；执行矩阵已有trace_assertions，不能只要求每个trace存在两三个键。fit_all_from_f=false、fit_source=T_test、内部fit数0等不能标通过。
4. **只有同一份整体校验结果为真才能返回成功、写FIXED_CANDIDATE及成功MANIFEST。** 为false必须非零退出，只在新独占失败目录留下FAILED/INCOMPLETE诊断。不要先写成功报告再计算裁决。
5. 预检、总裁决、最终验证使用一致schema和哨兵集合规则；当前“入口≥13、最终==13”会产生整体false后仍成功。可以固定当前必需13个哨兵，也可明确支持扩展，但未知/多余条目必须有一致处理，不能为通过故障测试随意放宽。
6. 保留现有7种已有效行为，再逐项复现下表新增拒绝。**执行实际main**，保留人工输入、case结果、命令/退出、所测生成器hash。外部diff等副作用可以受控替代，校验及最终化逻辑不替代，不改真实worker材料。

| 用例 | 必需结果 |
|---|---|
| 合法材料 | 一次最终化成功，机器裁决与报告一致 |
| 相同最终包再次生成 | 拒绝，全包原字节不变 |
| 明确FAIL；无关非空trace；同一node重复88次；telemetry FAIL；preflight FAIL | 原有拒绝保持 |
| isolation关键值false/T_test/0且键都存在 | 因trace契约拒绝 |
| 必需isolation节点换成不存在的唯一节点，仍88条 | 因真实/必需node不满足拒绝 |
| compiled来源文件缺失 | 因必需来源缺失拒绝 |
| guard_events日志缺失 | 因必需日志缺失拒绝 |
| integrity_passed=false而总PASS/overall_success=true | 因判决矛盾拒绝 |
| 任一输入能令最终overall_verification_passed=false（包括第14哨兵的原反例） | 绝不生成成功包；退出/状态/清单一致，若明确支持额外哨兵则仍须独立证明false分支拒绝 |

7. 当前F01/F02机器原义已恢复，保持OPEN/R1B；自然语言同步，F02是BA/EO/DP评估对象区别，不只是变量重命名。原47ID保留，worker新增R9-01仅作为历史汇总引用，不以其FIXED覆盖原问题。R7-03及已关闭问题保留具体范围和证据。
8. 所有AST位置从当前源码重新计算；46个带位置条目目前11项漂移，不能仅给新ID计算。嵌套check_dir_fd存在，使用正确限定/递归查找，不误写不存在。
9. 报告测试总数及逐文件数、资源、来源、hash均从最终机器事实生成。R9真实14/24/23/18/9不能改写为凑总数的14/19/17/16/22。最终回复hash直接取已冻结文件；当前旧history/matrix回复hash错误仅做新补充，不改旧包。
10. candidate.patch针对R9监督117项前像；cumulative git diff明确tracked-only，另列新增未跟踪内容身份。最终校验白名单、117输入预期差异、分支/HEAD/无暂存、保护文件、R9监督清单及旧worker/run指纹。生成器和人工诊断本身均计入改动/交付清单。

完成必要的单节点/故障校验后，运行一次本轮受保护完整synthetic suite。失败修复后按依赖范围再验证，每次保留新attempt；源码不再变化且全部要求通过后不反复重跑。已关闭内部模型/scaler测试无需新变异实验。若仍未满足要求，交付PARTIAL/OPEN及实际证据，不能用更改期望或生成器文字补齐。

## 项目路线和标准交付

本提示不授权R1B、真实数据或提交。R1A接受后另授权：R1B从F实际学习FairBias-BM/AE/Joint，适配Unmitigated、RW、LFR、EG-DP、EG-EO、TO-EO与LR/GBDT，统一F/C/S/T、权重/组类型能力、p/q/yhat、调参预算和阈值冻结；R2全年survey domain/共享配对复制及既定20个主差值家族、B=2000；R3获准执行冻结76条件×5种子，具体master/registry不换条件；R4对已查看2024做冻结后的回顾性评价。unsupported组合给原因。

FairBias仍是应用论文主方法，比较预测、公平性及权衡，不预设一定优于竞争方法，不用旧失效排名倒推调参。

按协议第9节交付：Gate:、Status:、Files changed:、Commands executed:、Permissions requested:、Tests executed:、Exact test results:、Input hashes:、Output hashes:、Row counts:、Assumptions:、Unresolved issues:、Git diff summary:、Proposed next step:。

提交报告/清单/每次attempt及固定用例真实结果，明确已关闭和仍未满足事项。不能自批ACCEPT、不能暂存/提交。最后一行：

STOP — waiting for Codex review.
