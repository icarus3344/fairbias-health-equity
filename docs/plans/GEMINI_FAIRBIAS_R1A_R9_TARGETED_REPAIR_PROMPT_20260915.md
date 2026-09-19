# 交给Gemini：FairBias R1A-R9定向修复提示（2026-09-15）

你是Gemini implementation worker。当前任务是R1A-R8监督REJECT之后的有界修复，**不是正式benchmark，也不是R1B**。Codex保留当前有效源码修复，只撤回交付验收资格；不要执行整体回滚、删除记录或重写旧交付。以下批次已授权顺序完成，无须每批重复询问用户。

先读[执行协议](/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md)、[AGENTS.md](/Users/lkc/Downloads/code_v_0_3/AGENTS.md)、[GEMINI.md](/Users/lkc/Downloads/code_v_0_3/GEMINI.md)、[R8监督报告](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R8_SUPERVISOR_REVIEW_20260915.md)及[生效前序R8提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R8_TARGETED_REPAIR_PROMPT_20260915.md)。旧提示边界在本提示未明确收敛处继续有效。

读取R8监督_evidence中的verification.json、evidence_verification.json、delivery_verification.json、两个probe_run的results.json、diagnostic_attempt_notes.json、final_verification.json和MANIFEST.json。第一轮5组完成不是5组契约通过，且有5次被阻止的旧生成器只读请求；生成器组以第二次无拒绝重检为准。不要将监督fixture漏配当作worker缺陷，也不要忽略重检复现的4种失败材料成功最终化。

## 起点、修改白名单与执行边界

- 分支`research/nhis-fairbias`，HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`，无暂存。核对R8监督114项输入/快照，记录本轮前像；如有外部变化，先明确归属，不静默覆盖。
- 可修改`scripts/_fairbias_r1_guard.py`、`scripts/run_fairbias_r1_guarded_tests.py`、`tests/synthetic/test_r1a_guard.py`及必要的纯人工fixture支持。可在本轮新报告目录建立stdlib交付生成器/验证器/负例fixture。优先复用当前已经有效的逻辑，避免重新写一版只拼报告的生成器。
- `src/fairbias/application_metrics.py`及identity测试的JSON修复本轮只读；其他生产源码、配置、benchmark、旧测试、治理文件和scratch只读。需要新的诊断node时先在白名单测试文件静态添加，然后走入口。发现白名单外生产bug则报告证据，不擅自修；完成无依赖工作。
- 14继承根文件与`.gitignore`保持不变；不处理既有历史差异。所有R8及更早run/报告/监督证据只读，不移动、删除或补写旧清单。
- 每次执行新建`runs/r1a_r9_<UTC>_<id>/`；新交付/生成器/静态验证材料放`docs/reports/fairbias_r1a_r9_worker_<UTC>_<id>/`。MANIFEST最终化后整包只读，修订另开目录，不能删MANIFEST后重建。
- **所有项目导入、AE/LR/scaler调试、pytest及单节点一律经过常驻guard入口。人工数据、仅sklearn、-I -S -B均无豁免。** 不直接python -c导入/拟合、不注入sys.path/PYTHONPATH绕过入口。此前3次直接尝试必须补记，本轮不得重复。
- stdlib纯静态AST/hash/日志读取及纯stdlib交付生成器可直接执行；不得借生成器动态import/exec模型源码。生成器故障只操作新的人工证据副本，不改正式run。
- 模型每分区≤200人工行、≤8预测特征；本次隔离诊断继续8行/2特征即可。完整preprocessing/adapter schema仍24字段、每fit分区≤8人工行，不新增预处理研究。保留15分钟/4GiB采样监测并准确说明保护局限。
- 不联网、不安装、不读取真实微数据、不跑旧全量tests或正式模型实验、不暂存/提交/push。保持H=1、完整1998项BM幂流、AE六网格、几何/revisit策略及既定调查和实验计划。

先静态加入phase及必要node白名单，然后所有模型/项目执行使用：

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -S -B scripts/run_fairbias_r1_guarded_tests.py --phase r1a-r9 --output-root /Users/lkc/Downloads/code_v_0_3/runs
```

单节点也使用同一入口传完整node ID，每次保留独占run。不要为了少一次guard运行而直接调试模型。

## 批次一：来源补记及真正拒绝缺失证据（R7-01、R7-02）

1. 新补充表记录R8附件69、102、116行的3次直接模型调试尝试；区分尝试与完成、项目导入与纯第三方拟合。无法恢复的退出码写UNKNOWN，不从最后成功推断。记录377、378行两次生成器入口；区分worker声称的0/1与持久可验证退出证据。
2. 保留当前6个R8 run（5FAIL/1PASS）、旧R7/R6历史及其不确定范围。本轮从开始保存每次run/单节点/故障/生成器/静态校验的命令、起止、实际退出、源码身份、输出目录和父子关系。不再用只有最终命令的列表声称全部命令受保护。
3. 父控制器必须要求loaded和compiled都存在且有效，loaded解析失败或None立即拒绝。验证根对象、sources非空字典、必需sha256/bytes、严格类型、声明模块数及实际数量，并绑定pre/loaded/compiled/入口/选中测试的集合关系。未执行配置/源码不要硬塞入loaded。
4. 使用实际父裁决路径测试正常和故障材料。helper测试可辅助，但必须证明失败传播到非零退出及execution_verdict。可用独占人工材料和安全的受保护故障入口，不要篡改已完成run。
5. 日志保持现有严格整数/bool/list与计数关系检查，补齐登记字段、操作/目标/原因匹配、已知match_mode、消费次数与身份/阶段的关联。原因字段允许None的语义须明确；非None必须按guard本身实际规则核验。不要以消费表抄下的事实字段代替原登记约束。
6. 闭合检查根据当前真实summary写入/日志关闭机制定义应有事件，核对数量与action/规范化target/reason及顺序，不能再仅要求差值在1至2或decision为ALLOWED。不发明新的通用安全框架。
7. pytest同buffer AST/compile保持。入口首次编译捕获若可在当前白名单内用有界启动绑定实现则完成；若只在函数执行后重读，准确保留PARTIAL及覆盖限制，不声称完整执行字节证明。不得通过改标签或扩大为OS沙箱工程解决。

固定验收表：

| 检查 | 正常材料 | 必须拒绝的人工故障 |
|---|---|---|
| loaded父路径 | 当前完整来源一致 | 缺文件、坏JSON、空根对象且删compiled enhancement.py；空sources；声明数错误；单项缺hash/bytes；错误hash/bytes/类型 |
| compiled集合 | 当前实际执行集合 | loaded完整但缺enhancement.py；单节点缺对应测试；保留原未登记源码/错误hash回归 |
| 日志事实 | 当前合法summary→闭合事件 | 缺必需字段、错误计数、字符串健康值；删1或2条真实尾事件；替换任意ALLOWED尾事件 |
| 登记关联 | 正常消费记录与事件相符 | 缺登记操作/目标；无关操作/目标；非匹配原因；未知match_mode配无关目标 |

不要只新增helper参数测试却漏过parent初始化None路径。每个负例应由其目标校验拒绝，不能因fixture无法读取/导入而算通过。

## 批次二：直接验证内部拟合内容（R7-03）

1. 现有模型概率和条件可以保留为补充诊断，但不得再作为F来源证明。直接核对真正scaler.fit和model.fit的输入内容；在AE之前安装，覆盖baseline及全部候选。
2. 用固定F和当前候选changed_dict建立预期变换参照，核对scaler拟合输入及学到的data_min/data_max/scale等适用状态；用F拟合的缩放状态核对实际模型fit特征、y及适用索引/权重。合法候选变换后的F与原始F可不同；不要把正常幂变换当错误。
3. 不只在fit_transform入口看raw_X或看shape。监督反例正是：fit_transform入口X正确，底层MinMaxScaler.fit实际收到X+0.125；26次学到的min全部偏移，原测试仍通过。
4. 正常/故障使用**同一个身份测试**。正常必须通过；仅内部模型fit输入+0.125仍失败（已有效，保留）；仅内部scaler.fit改为同形状X+0.125也必须失败。实际spy/断言保持启用，不靠专门读取“当前处于故障模式”的标记制造预期失败。
5. 在trace写出可核验的实际调用、候选关联及内容校验摘要；fit_all_from_f必须从全部适用断言得出，不能只由26次计数、形状或单标量概率条件得出。禁止写个人数据；fixture全部人工，可记录紧凑hash/差异/状态。
6. S和T逐项扰动已经完成，保留冻结模型/变换/阈值及现有回归，不重写。终局<=、忽略权重故障敏感性、JSON、预处理、BM/NMI/cache/公共指标等已关闭子项不重新实现。

## 批次三：让生成器真正裁决，修复原问题身份（R7-04）

1. 保留现有写一次保护；入口先验证已有最终包不可变及全部必需输入，任何生成失败另保留FAILED/INCOMPLETE诊断，不能写成功最终包。新生成器和真实负例证据纳入清单。
2. 实际main逐项读取并校验preflight、execution_verdict、summary、来源/日志与资源核验。严格检查类型/必需字段/一致性；整体裁决从各项结果合成，不能固定overall_verification_passed=True。现有true/false矛盾必须被拒绝。
3. PASSED日志绑定实际完整唯一node集合、收集/执行结果及必须node。一个node重复88行不能代表88测试。基于真实结果计算数量，不通过固定“>=88”决定契约完整性。
4. 7个trace必须核对各自必需结构、字段/类型/实测值和对应执行；非空字典不是充分条件。矩阵trace_assertions/JSON指针需实际运行验证；unknown符号、不存在/未执行node、缺字段/值不符不得标FIXED_CANDIDATE。输入中的true也不能替代生产行为证据。
5. 为实际main保留独占人工副本测试：合法一次生成成功；重复最终化拒绝且全包不变；显式FAIL拒绝；全部trace变成非空无关字典拒绝；仅单node重复88行拒绝；telemetry监测FAIL但execution_verdict PASS的矛盾拒绝；preflight FAIL拒绝。各故障单独注入，保存实际退出及结果；禁止改最终worker材料模拟负例。
6. 额外检查当前要求中尚未形成生产证据的源符号、输入身份/前像、历史不可变、日志和清单；无法证明保持PARTIAL。不要重新造一个更大验证框架，针对已有契约使用小的明确validator。
7. 固定原ID及含义，按监督纠正保留47个现有ID及明确范围。R6-03仍数值fit/transform清理，R8-01 CLOSED；R7-05/06原范围CLOSED；R7-01至04依据本轮证据保持候选或PARTIAL，不自行宣布ACCEPT。
8. **F01＝主benchmark未从F学习FairBias BM，OPEN/R1B**，不得复制R6 worker矩阵的“多组公平性扩展/R2”。**F02＝风险p、随机决策q及硬标签yhat混用导致BA/EO/DP比较对象不同，OPEN/R1B**，不是全仓变量重命名。自然语言与机器矩阵一致；不要改问题后关闭。
9. candidate.patch针对R8监督114项前像；cumulative git diff标明tracked-only，另列新增未跟踪文件身份和内容差异。新增生成器是改动，纳入文件列表和清单。
10. 最后复核白名单、114项输入预期变化、分支/HEAD/无暂存、保护文件、R8监督清单及历史worker/run指纹。所有hash和结果从最终机器事实读取，避免再手抄错误MANIFEST hash。

## 完成与阶段路线

完成必要的单节点/故障测试后，运行一次本轮受保护完整synthetic suite。失败先定位和修复，每次保留新的run；通过且源码未再改时不反复重跑。逐项提交固定验收表的正/负例结果、实际命令/退出和证据路径。不得为了过表而忽略unexpected denial、清空日志、弱化断言或改成功门槛。未完成事项准确写PARTIAL/OPEN，不能通过换phase、加测试数或手写成功报告替代。

R1A尚未接受，本提示不授权R1B/真实数据。接受后由Codex另授权：

1. R1B：从F实际学习FairBias-BM/AE/Joint；适配Unmitigated、RW、LFR、EG-DP、EG-EO、TO-EO及LR/GBDT，统一F/C/S/T、组类型/权重能力、p/q/yhat、选择预算和冻结阈值；unsupported组合明确原因。
2. R2：全年survey domain、共享配对复制及既定20个主差值家族、B=2000等推断规则验证。
3. R3：获准后执行既定76条件×5种子，master/registry具体条件保持，不用总数掩盖换条件。
4. R4：2024已被查看，冻结后按回顾性评价披露，不包装成从未触及的前瞻验证。

FairBias是应用论文主方法，同时比较预测、公平性及权衡；不预设一定优于竞争方法，不用旧失效排名倒推调参。

严格按协议第9节交付：Gate:、Status:、Files changed:、Commands executed:、Permissions requested:、Tests executed:、Exact test results:、Input hashes:、Output hashes:、Row counts:、Assumptions:、Unresolved issues:、Git diff summary:、Proposed next step:。

提交报告/清单/attempt索引与真实正负例证据；不能自批ACCEPT、不能暂存/提交。最后一行：

STOP — waiting for Codex review.
