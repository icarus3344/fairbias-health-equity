# FairBias R1A-R1 独立复审：暂不验收

日期：2026-09-14。分支 `research/nhis-fairbias`；HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`。审查对象是此 HEAD 之上的未提交候选工作树，并非只审查该 commit。

**监督结论：R1A-R1 的“全部修复、无遗留问题”申请不予验收，进入 R1A-R2 定点返修。现在不能启动正式 benchmark，也不能据此宣称 FairBias 优于对照方法。** 保留当前候选代码、失败尝试和历史证据，不执行破坏性回滚。这里的 R1A-R2 指第二次基础契约返修，不是后续调查推断 Gate R2。

## 1. 本次核验范围与证据

- 独立审查源码、相对上轮候选的 diff、worker 报告与用户粘贴的执行记录；核验 85 项输入快照，另补登记已授权的 `configs/nhis/study.json`，共 86 项。worker 相对上轮候选修改 16 个源码/工具/合成测试文件；`preprocessing.py` 相对上轮候选没有变化。
- worker 成功运行的 22 项执行前后文件哈希相同且与当前文件一致；84 项前像匹配上轮监督快照；46 个 PASSED 节点确实存在。**这是日志和源码身份核验，没有独立重跑整套 46 项。**
- 独立执行 10 组小型合成诊断，全部完成，诊断程序错误为 0，监督外层 guard 非预期拒绝为 0；其中多组成功复现缺陷，不能写成“10 项验收通过”。真实 LR 拟合最多 6 行、1 个特征；预处理真实拟合使用 8 行、24 个既存 schema 字段，仅替换文件运输与历史全年人数检查。
- 7 个旧运行的恢复索引及引用哈希成立；上轮监督 manifest 的 110 项保持不变；本轮 3 次运行均保留。恢复索引已完成，原先丢失的 5 个旧报告目录仍属于历史缺失，索引不等于恢复原件。
- 本监督轮只新增报告、提示词和诊断证据；没有修改生产源码，没有读取真实微数据、联网、安装依赖、stage 或 commit。

主要证据：[诊断结果](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R1_SUPERVISOR_REVIEW_20260914_evidence/probe_run_130440_143055/worker_case_output/results.json)、[实际诊断源码](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R1_SUPERVISOR_REVIEW_20260914_evidence/probe_run_130440_143055/probe_source.py)、[源码与历史完整性核验 v2](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R1_SUPERVISOR_REVIEW_20260914_evidence/evidence_verification_v2.json)、[候选 diff](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R1_SUPERVISOR_REVIEW_20260914_evidence/diff_from_pre_R1A_R1.patch)。v1 核验曾因监督快照未含 study.json 误报不匹配，v2 明确纠正；这是监督查找遗漏，不是 worker 源码不一致，两个版本均保留。

诊断源码顶部沿用了旧说明中的“未安装 worker guard”，该句不适用于本轮：最后一组确实安装并调用它。执行源码和结果原样保留，不事后改写；日志预算/关闭故障是该组刻意注入的诊断条件，不把这份诊断日志当成合格正式运行日志。

## 2. 已确认的实质进展

| 项目 | 本轮确认 | 尚不能据此推导 |
|---|---|---|
| 应用公平性 helper | TPR 相同/FPR 不同的人工例得到 EOpp=0、equalized odds=1；某组缺负类时 DP/EOpp 可估计而 EO 不可估计 | 所有公共入口与最终报告已正确接通 |
| 参数/状态编码 | 标量 int 与 str、数组中部差异可区分；NaN 参数、坏 get_params、非法状态版本被拒绝；实际 scaler 工厂已接入指纹 | 所有嵌套参数、空网格和真实缓存行为均可靠 |
| 身份、概率与权重 | 分隔符碰撞修复；未知来源与已知来源交叠例被拒绝；None ID 被拒绝；反序 classes 正确；NaN 概率容差被拒绝；两个 1e308 权重的比例为 0.5，错位索引被拒绝 | 零泄漏、全部来源类型和调查设计推断已验证 |
| Registry | 实际修改 AGEP_A substantive_codes 的预拟合对象被 adapter 拒绝 | 列表顺序与拟合后不可变性已实现 |
| 几何/AE 测试 | 已出现真实 enhance_step、真实小型 MDS/BM 调用，以及配置传播 spy | 真正 Joint 循环、重复搜索游标、NMI/epsilon 边界已覆盖 |
| 执行证据 | 日志递归问题缓解：成功运行日志 1,444 条，18 次拒绝，文件约 446 KiB；源码绑定与失败运行保留改善 | 18 次拒绝都按规则匹配为预期，或 guard 已无逃逸路径 |

## 3. P0：阻断当前验收

### R2-01 公共 compute_metrics 对合法输入直接崩溃

位置：[evaluator.py:149](/Users/lkc/Downloads/code_v_0_3/src/fairbias/evaluator.py:149)、[159](/Users/lkc/Downloads/code_v_0_3/src/fairbias/evaluator.py:159)、[168](/Users/lkc/Downloads/code_v_0_3/src/fairbias/evaluator.py:168)。本轮删掉了循环中的 `prot_series/prot_vals/groups` 初始化，却继续引用 `prot_vals` 与 `groups`。

**已复现：4 行合法二分类输入、一个两组受保护变量，调用真实 `FairEvaluator.compute_metrics`，立即抛出 `NameError: name 'prot_vals' is not defined`。** 46 项通过的测试没有捕捉这个公共 API 回归。影响包含受保护列的指标计算，不是仅在极端输入触发。

修复必须恢复每列对应的实际变量，并覆盖多个受保护列。此外 [129](/Users/lkc/Downloads/code_v_0_3/src/fairbias/evaluator.py:129) 在验证前强制转 int，会把小数决策概率截断成硬标签；仅修 NameError 仍不满足 C06。应先验证原始输入，再做必要转换；legacy 和 application 指标名称/公式保持区分。

### R2-02 文件变更 guard 忽略 dir_fd，可越过声明的输出边界

位置：[_fairbias_r1_guard.py:327](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:327)。mutation audit 按当前目录解析相对路径，忽略 remove/rename 等事件携带的目录 fd；部分整数 fd 分支直接跳过。open 的 [168](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:168) 使用 `/dev/fd/N/path` 拼接，也不是统一的真实目标解析。

**已通过真实 API 复现：在 worker 输出目录中执行相对路径 `os.unlink(..., dir_fd=outside_fd)`，成功删除位于 worker 输出边界外的新造人工哨兵。** 该人工文件仍在监督外层允许的独占诊断目录内；没有触碰任何既存文件或数据。因而这是被测 guard 的实际失效，不是外层保护代替它成功，也不是仅手动调用 audit 回调。

须统一处理单/双路径及其 dir_fd，无法安全解析的非默认 dir_fd 明确拒绝；fd 操作必须核验方向和所属对象。P0 指本项目执行边界失效，不宣称操作系统或远程服务存在漏洞。

## 4. P1：方法学与证据可靠性问题

### R2-03 审计日志故障不使运行失败，拒绝次数被当成预期分类

位置：[_fairbias_r1_guard.py:119](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:119)、[139](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:139)；[runner:206](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:206)。写失败被吞掉；预算耗尽只增加计数；关闭 sink 后仍可调用 logger 而不失败。独立故障注入确认：磁盘预算耗尽与关闭 sink 后的 DENIED 事件调用均正常返回。

runner 先关闭日志再写结果；把 pytest 阶段所有拒绝的计数差直接命名为 `denials_in_tests`，没有逐项核对预期事件，也没有要求 `truncated_events==0` 或日志健康才成功。故“0 unexpected denials”不能从该实现推出。修复为显式日志健康状态、精确的预期拒绝匹配及非零失败退出；保护固定日志 sink，最终化顺序可追溯。Python audit hook 仍不是对原生扩展的完整 OS 沙箱，不应承诺该能力。

### R2-04 D8 最终指标丢字段，预声明组集合没有从 run_arm 传入

位置：[d8_enhancement_runner.py:325](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:325)、[369](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:369)、[392](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:392)。helper 已计算 EO/status，但返回字典仅导出 DP/EOpp。`run_arm` 的四个调用点 [753](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:753)、[792](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:792)、[932](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:932)、[1195](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:1195) 均未提供 expected_groups。

真实 LR 终局人工例显式声明一个缺失组：DP 为 null，但 validation 没有 equalized_odds_gap、逐指标状态或主估计资格。不能区分“完整组比较不可估计”与“观察到的部分组看起来公平”。必须贯通配置/入口→评估→序列化结果，并保留原因与组定义来源；不能用观察到的组集合冒充预声明组全集。

### R2-05 Registry 比较对象不完整；拟合后清理规则可被外部修改

位置：[adapter.py:141](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:141) 比较 family 字典键顺序，而 [preprocessing.py:175](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:175) 实际消费 feature_lists 的列顺序。独立反例反转 primary_core_features 后，真实拟合的对象仍被 adapter 接受。

[preprocessing.py:165](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:165) 仅浅复制 registry，[183](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:183) 继续共享嵌套规则。**真实 fit 和 adapter 接受后，只改调用方原字典，年龄 90 的同一人工输入从输出 18 变成 90，is_fitted 仍为 True。** 这破坏冻结预处理语义；不是已经证实真实数据泄漏，但使零泄漏契约不足以成立。

需要按真正消费的字段、顺序和 harmonized 映射比较；深复制/不可变快照并绑定拟合规则身份，transform 时不允许内部或外部修改悄悄生效。现有 [registry 测试:84](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_registry_and_weights.py:84) 仍主要增加原 schema 不存在的 valid_range，未覆盖上述实际路径。

### R2-06 配置指纹仍有碰撞，公共函数被重复定义覆盖

位置：[enhancement_contracts.py:383](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:383) 把字典键转成字符串；`{1:0.5}` 与 `{"1":0.5}` 仍碰撞。[445](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:445) 使用 truthy 默认值，导致实际空指数网格与六指数网格得到相同配置指纹；已在真实 engine 属性与指纹上复现。若这些不同配置参与缓存复用，缓存上下文不能区分它们；本轮未把此反例夸大为已观察到真实训练结果误复用。

[314](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:314) 与 [411](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:411) 两个同名公开函数并存，前一个残留且无返回，后一个覆盖它并丢失原 keyword。旧 `algorithm_mode=` 调用已复现 TypeError。应保留单一明确实现，typed key/value 编码，None 与空值区分或明确拒绝非法空网格；兼容已存在的参数或提供有证据的迁移，不静默改变 API。

### R2-07 测试标题仍超过真实覆盖，不能证明搜索与零泄漏

- [CACHE-1:98](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_state_and_config.py:98) 仍主要人工 mark cache 后比较上下文，没有实际候选模型 fit 次数证明同配置复用、变配置重算。
- [CACHE-3:147](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_state_and_config.py:147) 增加了真实 AE 调用，这是进步；其 Joint 部分仍是普通 Python set，没有执行 D8 Joint 提交/循环链。
- [GEOM-5:253](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:253) 检查幂流长度、属性及单次搜索，未证明重复搜索时 restart/monotone_cursor 的指数轨迹不同。[GEOM-6:298](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:298) 检查 NMI 范围/配置异常，不是 NMI 接受边界与严格 epsilon 等号边界。
- [ISOLATION-2:34](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:34) 扰动的是 EvaluationPartition.selection（此处为候选选择 C），然后比较 F 的数据指纹；没有独立 S/T，没有真实拟合对象或 C 选择输出比较，也没有组/权重扰动。

需要真实调用链+可解释人工 oracle。可注入运输层/人工效用，必须说明注入位置，另保留真实 LR/MDS 例；不能把普通 set、数据 hash 或函数名字当成完整行为证据。

### R2-08 输入规范化与错误输出仍有死角

[application_metrics.py:10](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:10) 把二维标签自动展平；[74](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:74) 仅对浮点 dtype 检查组的非有限值。独立例中二维标签被接受，object dtype 的组 `[0,0,inf,inf]` 也被标成合法主估计。expected_groups 应先严格验证，不能在单组 early return 时绕过；空/重复/未知组政策与输出资格须一致。FairEvaluator 对非 list/tuple 的 Sequence 还会静默忽略，应统一规范化或明确拒绝。

[enhancement_contracts.py:238](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:238) 把非字符串 source 任意转字符串，NaN 成为合法 `nan` 来源；[127](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:127) 未把它识别为未知来源。已复现 identity 编码接受 NaN；跨 partition 的潜在绕过机制由源码推导，本轮未另跑该完整反例。[246](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:246) 在年度规范化前比较原值，合法等价的 2023 与 "2023.0" 被误拒绝。

错误文本 [141](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:141) 包含整组重叠 year/ID，[application_metrics.py:21](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:21) 和 [30](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:30) 可打印标签数组。真实数据运行若记录异常会有行级信息进入日志的风险；本轮仅产生人工值，不是已发生真实数据泄露。改成类型、形状、失败数量与非个体上下文。

## 5. P2：执行与交付可靠性

### R2-09 预检、监测和最终化尚未形成失败闭环

位置：[runner:94](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:94)、[305](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:305)、[332](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:332)。Git 查询未逐项检查返回码；前后 hash 被保存但未作为一致性门禁；ps 异常被忽略。成功运行的 hash 本轮确实相等，问题是遇到失败/改变时工具没有可靠拒绝，而不是本轮已出现不一致。

补充明确 PRECHECK/HASH/MONITOR_ERROR；区分子进程已正常退出与监控故障；保留 kill 后 wait 的改进。环境文件应有实际 numpy/pandas/sklearn 等版本、argv/cwd 与已加载源码身份。`-B` 只禁止写 pyc，不能单凭它断言旧 pyc 不可能被读取。0.5 秒 RSS 采样只能报告“采样最大值”，不是精确峰值或硬内存上限。

### R2-10 Worker 的全部受控/全部修复叙述超过提交证据

[提交记录:4](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R1_SUPERVISOR_REVIEW_20260914_evidence/worker_submission.txt:4) 起包含入口外 `python -c` 项目导入和 BM 执行，及三次入口外 geometry pytest；不满足上轮要求的所有项目执行均经常驻 guard。可见代码使用人工数据，没有证据证明这些命令读取了真实数据；也不能反过来把它们写成全程受控。只列最后一次正式命令遗漏了这些尝试。

记录还显示在已有 MANIFEST 生成后再写报告并覆盖 MANIFEST；这破坏“报告定稿后一次最终化”的交付规则，但当前最终哈希已能对上，不据此指控恶意篡改。`scratch/r1a_r1_preimage` 在上轮授权输出目录外，后续保持只读登记，不再扩散或删除。报告提到的 `_validate_preprocessor` 在当前代码不是对应函数名，实际比较在 adapter 构造器中，须从最终 AST/源码生成位置。

## 6. P3：下一步研究与应用论文

先完成本轮基础契约，再按既定路线推进；无需推倒已完成的条件登记或重复整理旧恢复索引：

1. **R1A-R2**：修 R2-01 至 R2-10；以行为反例关闭问题，提交给 Codex 独立复核。
2. **R1B**：真实 F-only FairBias-BM/AE/Joint adapter，接入 Unmitigated、RW、LFR、EG-DP、EG-EO、TO-EO，与 LR/GBDT 的能力匹配；统一风险 p、决策概率 q、硬标签 yhat。旧 F01/F02（主 benchmark 实际 FairBias 执行及 p/q 混用）仍 OPEN，R1A 修复不能替它们结案。
3. **R2**：统一复杂调查权重/分层/PSU、全年设计上的 domain 估计、配对复制与非光滑 gap 的验证。保留预定 B=2000、有效复制、df/SE 与多重比较规则；不能把本轮比例函数测试称成调查推断验证。
4. **R3**：按已登记的 76 条件、5 种子及计算预算做演练和获准开发。主对比仍是 Arm001/003、LR、EO 操作点下 FairBias-BM 相对五种外部方法的 ΔBA/ΔEO，共 20 差值家族；GBDT及其他比较按既定主次层级报告，不事后挑选胜者。
5. **R4**：冻结后回顾性评价 2024。2024 已被看过，不能重新称盲测。Arm004 的“路径敏感性”只能在控制配置、定位首次分歧和完整搜索轨迹后作为受限解释，不能替代因果证据或宣称全面复现作者几何。

研究定位保持“FairBias 为主方法的医疗公平性应用研究”；结果允许它在某些方法、群体或预测维度不占优。论文价值可以来自可靠的权衡和外部年份稳定性，不能以获胜为验收条件。完整研究边界继续参照[主计划](/Users/lkc/Downloads/code_v_0_3/docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md)及[R1A-R1 提示词](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R1_REPAIR_PROMPT_20260914.md)。

本报告列出本轮已证实及明确标注的推断问题，不构成“所有潜在漏洞已穷尽”保证。下一轮可执行交付见 [Gemini R1A-R2 提示词](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R2_REPAIR_PROMPT_20260914.md)。最终只新增文档/证据的核验与文件绑定见同名 evidence 目录的 final_verification.json 和 MANIFEST.json。
