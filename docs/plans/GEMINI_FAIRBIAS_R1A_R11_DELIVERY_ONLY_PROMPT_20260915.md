# 交给Gemini：FairBias R1A-R11仅修交付层（2026-09-15）

你是Gemini implementation worker。Codex对R10给出**REPAIR**：已有模型契约、来源和日志路径修复保留；剩余R7-01诊断留存及R7-04完整交付校验。**本轮只修改新交付目录中的纯stdlib工具，复用已核实R10运行，不改runner/test/model、不新增phase、不重新跑模型或88项suite。** 本提示明确收敛前序“改源码后重跑”的范围。

先读[执行协议](/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md)、[AGENTS.md](/Users/lkc/Downloads/code_v_0_3/AGENTS.md)、[GEMINI.md](/Users/lkc/Downloads/code_v_0_3/GEMINI.md)、[R10监督报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R10_SUPERVISOR_REVIEW_20260915.md)及[前序R10提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R10_CLOSURE_REPAIR_PROMPT_20260915.md)。读取R10监督_evidence中的verification.json、evidence_verification.json、delivery_details.json、metadata_verification.json、probe_run_122524_176197/results.json、final_verification.json和MANIFEST.json。

注意：原探针fixed_candidate_written只匹配单行状态文本，R10换行模板使它为false；delivery_details已按实际字段确认6种坏材料仍为FIXED_CANDIDATE/整体true。不要用这个格式标记反驳实际生成结果。

## 范围与固定输入

- 分支`research/nhis-fairbias`，HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`，无暂存。核对R10监督120项输入及快照；预期本轮全部保持原值。
- 新文件仅放`docs/reports/fairbias_r1a_r11_worker_<UTC>_<id>/`，包括生成器、纯stdlib校验模块、诊断器、attempt记录、人工副本及报告。目录排他创建，不新增scratch文件。
- 全部已有src/scripts/tests/configs、治理文件、14继承文件和`.gitignore`只读；所有R10及更早报告/run/诊断/监督证据只读，不修补旧MANIFEST。
- 固定执行来源为`runs/r1a_r10_20260915_105822Z_d582f745/`。开始时验证pre/post/current37项、loaded/compiled34项、来源run身份及R10监督指纹仍一致。若真实来源变化，停止依赖该输入的工作并报告，不能改副本数字让它匹配；可继续其他无依赖静态工作。
- 新交付明确`delivery_stage=R1A-R11`、`source_run_id=r1a_r10_20260915_105822Z_d582f745`、`source_phase=r1a-r10`。诊断副本所在目录名不改变被审计原run身份；不要编造R11训练run或把工作目录basename当来源ID。
- 允许纯stdlib读取源码文本/AST、hash、JSON/日志，运行本轮纯stdlib生成器及其人工诊断。不导入或执行模型项目源码、不运行pytest、不做fit、不注入sys.path/PYTHONPATH、不联网、不安装、不读真实微数据、不暂存/提交/push。
- 可复用现有已验证的**纯stdlib证据校验逻辑**，放入本轮目录中的独立校验模块；避免给生成器再写一个较弱的简化版。不能借复用导入整个训练工程。若必须改白名单外文件或执行项目模型，先报告具体依赖，不擅自扩展；完成可独立完成的工作。

## 批次一：完整验证来源、日志和trace（R7-04）

建立单一总校验入口，使其结果共同控制退出码、报告状态及成功MANIFEST。在任何成功报告/清单写入前完成验证。原R10的≥15条目/≥100行不是充分验证，必须替换。

1. 来源材料：pre、post、loaded、compiled必须存在且合法。严格验证根/条目schema、必需字段、非bool整数bytes/count、非空集合、声明数=实际数、每项hash/bytes、执行前后/可信源码身份、loaded/compiled集合关系、实际runner与已执行测试来源。不要要求所有未执行pre配置都出现在loaded。复用已验证的集合规则，不能凭长度或外层PASS判断。
2. 绑定当前来源run与实际执行材料：child summary、telemetry、execution_verdict及preflight的必需字段/类型/子项和总判决保持一致；引用可信R10上下文校验，不能由被验证材料自己随意声明预期来源。
3. 日志：逐行JSON解析，验证事件schema/类型、decision、计数/summary截止关系、拒绝登记/消费、意外拒绝、精确summary闭合路径。使用可信原run路径作为参数；人工副本仍按原run身份审计。不能仅判断非空行≥100，不能依赖日志reason文字自证路径。
4. 测试node：保留当前88个必需真实node的完整集合校验；同时绑定实际源码和执行证据。当前列表可作为冻结契约，新增测试是另一项未来工作，不在本轮扩张；不能只检查数量/去重。
5. Trace：将现有issue_closure_matrix的trace_assertions与前序已关闭契约整理成明确schema/值/关系，**对7个trace都执行**。缺字段、错误类型、false值、事件结构或计数相互矛盾必须拒绝，不只校验isolation几个布尔。
6. 至少覆盖BM人工流标识/实际推进字段、正式1998流及策略、cache真实复用/fit次数、公共指标入口与键、F/C来源/冻结阈值/内部fit摘要一致性、joint事件结构与cycle_detected、NMI候选/终局边界等既有声明。按当前源测试定义解释，不能凭名称猜新期望；允许引用明确验证过的原证据，不重新定义算法。
7. 整体false或任何校验错误只留下新独占FAILED/INCOMPLETE诊断，非零退出；禁止生成成功报告或成功清单后才计算裁决。写一次机制保留。

固定验收用例均执行本轮**实际生成器main及总校验路径**：

| 材料 | 结果 |
|---|---|
| 可信R10正常材料 | 验证通过，可一次最终化；报告明确来源R10 |
| 同一成功包重复生成 | 拒绝，原字节不变 |
| 原13种normal/repeat/负例行为 | 保留R10已有效结果 |
| loaded/compiled hash、bytes、声明数损坏 | 拒绝；各字段单独变动也必须触发对应检查 |
| loaded含enhancement.py，compiled删它 | 拒绝 |
| pre或post缺失/不一致 | 拒绝 |
| 日志100行not-json | 拒绝，而不是满足行数就通过 |
| 合法JSON日志尾目标换成其他目录 | 拒绝，与真实runner一致 |
| 其余trace保留键但值置坏 | 拒绝；按7个trace分别验证，不靠组合故障中的第一项掩盖其他未检查字段 |

最后一行使用监督已有反例：人工流类型invalid、正式流长度0、cache复用false、公共入口invalid/空keys、selection_source=T_test/阈值999、joint events为字符串、terminal.equal_feasible=false。该表只是现有要求的具体反例，不可把通过表内少数常量变异当成完整schema验证。

## 批次二：诊断独占留存和清单闭合（R7-01）

1. R10有203、230行两次诊断脚本调用，中间源码编辑，204行另有生成器复查；当前只有一个固定结果集。新补充表逐项记录可见事实及UNKNOWN，不断言首轮成功，也不通过本轮重跑补填历史身份/退出。
2. 每轮诊断使用唯一UTC/id目录并排他创建；每case再有独占路径。运行前保存所测工具源码或不可变版本引用和hash；改动后必须新attempt，不能覆盖之前输入/结果。中断保留INTERRUPTED/UNKNOWN及已有输出。
3. 持久保存每个子进程实际command、cwd、来源run身份、起止、returncode、完整stdout/stderr、输入hash、输出前后hash及case结果。原pytest的child_stdout副本不能充当诊断生成器日志。负例必须由目标检查拒绝，权限/语法/导入错误不能算契约通过。
4. 当前R10的diagnostics278个文件不在顶层MANIFEST。新包必须递归覆盖所有诊断文件，或以可核验hash链引用独占子MANIFEST及汇总。需要明确验证每个case及失败/中断attempt，不能只纳入脚本文件或“all_passed”字段。
5. 最终MANIFEST排他写一次；成功包封存后，**生成器与诊断器均先拒绝在该包下写入**。验证二次生成和二次诊断不会改变任一原文件。若需要修订，在新的包目录执行，不删除旧清单。
6. 从已封存诊断事实计算报告中的用例数/通过数，不预写“13全过”。确需外部diff等操作时用只读命令并保存真实执行状态；不执行模型。

## 批次三：由机器事实生成交付报告

- 当前源码、31项清单及32项回复hash已匹配，保留该历史事实；下一轮从新最终包直接读取hash，不手抄旧值。
- 明确复用了R10的88 passed/0failed，而非声称执行了R11 pytest。按来源日志取14/24/23/18/9、pytest3.60秒、run6.11秒及numpy overflow RuntimeWarning；预算是4GiB采样约束，不写2048MB。指标和警告必须从真实记录生成，缺信息就标未核验。
- 当前46项AST位置都已修复，原48个ID含义保持；R7-03、R7-05/06、R8-01维持关闭范围；R7-02具体来源/日志反例已闭合，runner首次编译捕获保留准确PARTIAL。R7-01/04依据本轮证据列候选或未完成，不能自批ACCEPT。
- F01＝主benchmark尚未从F学习FairBias BM，OPEN/R1B；F02＝p/q/yhat混用改变BA/EO/DP评估对象，OPEN/R1B。当前不做接线或变量重命名。
- 本轮没有源模型改动，candidate diff必须准确表示新交付工具；cumulative git diff仍仅tracked文件，不能借旧累计差异暗示本轮改了模型。新增工具本身纳入文件列表、源码身份和清单。
- 最后核对120项监督输入、R10及历史worker/run/监督清单、分支/HEAD/无暂存与保护文件不变；诊断已冻结后再封存完整新包。全部检查通过且代码未再变化后不重复执行。

批次已授权顺序完成，不需逐批询问。若仍有内容校验或留存要求未实现，准确报告PARTIAL/OPEN及触发，不写“全部完成”。不要为收尾扩大成新的安全平台或再次改模型测试。

本轮不授权R1B、真实数据、提交或论文性能结论。R1A接受后另授权R1B真实BM/AE/Joint与竞争方法（Unmitigated/RW/LFR/EG-DP/EG-EO/TO-EO）及LR/GBDT适配，统一F/C/S/T、组类型/权重能力、p/q/yhat和调参/阈值冻结；R2既定survey domain、共享配对复制、20个主差值家族/B=2000；R3冻结76条件×5种子；已查看2024按回顾性评价。FairBias为应用论文主方法，不预设胜出。

按协议第9节交付：Gate:、Status:、Files changed:、Commands executed:、Permissions requested:、Tests executed:、Exact test results:、Input hashes:、Output hashes:、Row counts:、Assumptions:、Unresolved issues:、Git diff summary:、Proposed next step:。提交真实工具/运行来源/诊断/清单路径，不能自批ACCEPT或暂存提交。最后一行：

STOP — waiting for Codex review.
