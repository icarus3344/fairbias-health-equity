# FairBias R1A-R3 独立监督审查

审查日期：2026-09-14。目录：`/Users/lkc/Downloads/code_v_0_3`。分支：`research/nhis-fairbias`。HEAD：`67e6659fa65249a8842e34af5d8969629efe4bca`。

**裁决：REPAIR / NOT_ACCEPTED。不能接受“18项全部解决、无遗留问题”，暂不授权 stage/commit 或进入 R1B。** 本轮已有实质修复；保留候选和全部历史记录，集中完成下面剩余契约，不重做已验证模块。R1A-R4 是基础契约的第四次返修，与正式实验 R4 无关。

## 核验范围与已确认的进展

审查对象是 worker 最终交付 `fairbias_r1a_r3_worker_20260914_141512Z_495bd5d4`、同名运行工件、用户提交的操作记录以及当前候选源码。源文件未提交，因此 HEAD 不能代替候选文件身份。

- 独立核验 worker 日志有 **68 个 PASSED 节点**，AST 有68个测试；矩阵引用的39个不同测试节点均存在。没有再次运行整套 worker 测试，不能称“监督独立复跑68/68”。13个哨兵及受控故障成功来自已核验的 worker 工件。
- 22个预执行/执行后文件指纹相同，均匹配当前文件；29个执行模块清单的哈希也匹配当前源码。最终 worker MANIFEST 的16项全部匹配，前轮监督 MANIFEST 的130项未变。三个 worker 尝试均保留。
- 上轮 **D8 KeyError 的具体反例已关闭**：真实 LR、6行训练、1预测特征的终局调用可以完成；缺组返回 null、状态和 `primary_result_eligible=false`。组全集之外的观察组、字符串组序列和 Decimal Infinity 均正确拒绝。相关入口：[D8 evaluation](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:245)、[组校验](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:56)。这不等于四条 run_arm 路线都已集成验证。
- **AE 缓存有真实改善**：同一 engine 的首次/重复/修改 C/更换实际 scaler，独立观测到 fit 增量 **13、0、13、13**，每次训练仅8行、2特征。基线效用缓存见[enhancement.py](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:530)。这四个数是本人工例结果，不是普遍拟合预算，也不是 Joint 证据。
- 直接跨 registry 加载被拒绝，合法工件往返一致；直接修改 `fitted_record` 返回副本、反转公开列清单，不再改变 preprocessor.transform 自身的结果。但工件生命周期和适配器仍有缺口。
- 日志整数 fd 的旧别名反例已被实际 `GuardViolationError` 阻断；未发生的 expected denial 会抛 `AssertionError`。集中 verdict 函数正例为 PASS，11类单因素故障均变为 FAIL。实际源码加载器已创建安装，worker日志未发现项目 pyc 读取。不能再沿用“仅在文档中声称安装加载器”的旧结论。

上述结果只支持软件契约判断，不支持 FairBias 优于其他方法或调查总体推断。

## P0：本轮未确认新的严重崩溃或越界漏洞

本轮没有复现上轮 D8 的正常输入崩溃。此前修复的公共指标、dir_fd具体反例保持候选状态；本轮未对所有旧攻击路线重新测试。下面的 P1 问题仍足以阻止主实验验收。“没有新增P0”不代表所有潜在漏洞已穷尽。

## P1：仍阻碍可信实验的缺陷

### R4-01：导出／重载会给未验证统计量重新赋予可信身份

位置：[preprocessing.py:520](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:520)、[load_fit_json:537](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:537)、[adapter legacy检查:136](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:136)。延续原 R2-05 / R3-02。

`export_fit_json` 用当前可变的 `self.specs` 计算新身份，未先验证它仍等于拟合时的规则，也未保留旧工件的未验证状态。`load_fit_json` 看到匹配的 rule_identity 就把 legacy 标记设为 false。

独立反例使用8行人工数据，年龄全部90，真实按年龄0–100规则 fit，没有手填拟合统计：

1. 直接把该工件载入官方年龄上限85的对象会被正确拒绝。
2. 把这份人工工件转换为无身份的旧格式，load 后 adapter 正确拒绝；但再经公开 export→load，`is_legacy_unverified` 变成 false，adapter 接受，年龄90仍被用作插补值。
3. 同一个真实拟合对象，把公开 specs 改成官方规则后，transform 正确拒绝；export→load 却再次绕过拒绝，adapter 接受同一组年龄90统计。

因此本轮解决的是“直接加载错配”，还没有解决“身份沿工件生命周期不变”。这会让不同清理规则或来源的统计量混入模型比较，不能凭当前 schema hash 证明原先拟合合格。

同一入口还接受未知 `format_version`，忽略工件记录的列顺序。反向问题是 legacy 对象进行新的合法真实 fit 后，legacy 标记仍为 true，误拒绝有效对象。修复应使用拟合时完整身份、显式版本和状态转换；旧格式不能仅因再保存获得主分析资格，新 fit 成功才能建立新的可信身份。不要迁移真实历史工件来“修复”旧结果。

### R4-02：适配器重新读取可变列清单，绕过预处理冻结

位置：[preprocessing.get_feature_family_lists](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:505)、[adapter.get_feature_names](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:246)、[adapter.get_cohort](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:308)。延续原 R2-05 / R3-02。

transform 使用冻结列序，但 adapter 从公开 `primary_core_features` 取清单，再对 transform 输出二次切列。独立人工例：合法 fit 并构造 adapter 后，移除公开清单中的 agep_a；再次实际 get_cohort，输出从 **21列变20列**，没有重新拟合；同时 transform 本身仍输出年龄。另一例中 transform 不变，但特征类别列表顺序改变。

只证明 transform 不变，不能证明最终预测矩阵不变。需要让 transform、feature-family接口、adapter特征名及最终cohort使用同一冻结定义，或统一在使用前拒绝变更。验收必须经过 get_cohort，不仅调用 transform。

### R4-03：Joint、BM游标和S/T隔离仍缺真正的集成证据

延续原 R2-07 / R3-04。以下是明确的测试覆盖缺口，不据此断言生产 Joint 一定错误或真实数据已经泄漏。

| 契约 | 当前实际执行 | 缺失的验收证据 |
|---|---|---|
| Joint 循环 | [state测试:192](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_state_and_config.py:192)仍只是普通 set 的插入/查询 | 五个测试文件没有实际 run_arm 调用；须观察生产 Joint 的候选提交、committed_state、A→B→A停止及最终状态 |
| D8四路线 | [终局测试:351](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_identity_and_probabilities.py:351)真实调用 evaluate_representation，值得保留 | run_arm四个调用点的来源、预期组、所有终局分支仍未被真实串联 |
| BM游标 | [GEOM-7:332](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:332)分别对两个新engine调用一次内部数值搜索 | 同时改 revisit 与 sequence 两个参数，不能归因于 revisit；须同一官方序列下至少两次调用、记录实际指数轨迹 |
| epsilon/NMI | [GEOM-8:374](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:374)确实经过 `_make_candidate` 测试严格阈值 | 终局 `<=` 未执行；NMI gate 被直接固定为 true，未检验实际 phi_threshold 下/等/上边界 |
| F/C隔离 | [isolation测试:116](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:116)对相同F独立fit两个LR并预测相同C；partition的selection却放入S | 没有执行FairBias候选选择、变换或阈值选择；没有独立S和T及A/w扰动，不能证明完整F/C路径不依赖S/T |

GEOM-9 已真实调用 mitigate_step 并检查失败属性，这部分有进步；AE同配置/C/scaler缓存行为也已独立证实，不应继续笼统称“所有搜索测试都是假测试”。下一轮只补具体缺项，以及已有要求中的epsilon缓存失效、F/C数据变化和返回旧配置策略。

### R4-04：关闭矩阵的编号、函数引用和“全部完成”结论不可靠

位置：[matrix:19](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r3_worker_20260914_141512Z_495bd5d4/issue_closure_matrix.json:19)、[worker报告:153](/Users/lkc/Downloads/code_v_0_3/docs/reports/fairbias_r1a_r3_worker_20260914_141512Z_495bd5d4/WORKER_REPORT.md:153)、[生成器:77](/Users/lkc/Downloads/code_v_0_3/scratch/build_r1a_r3_delivery.py:77)。延续原 R2-10 / R3-05。

39个不同测试引用都存在是改善，但“节点存在”不等于“该节点验证该问题”。例如 R2-01 写 `FairBiasEvaluator.compute_metrics`，真实类是 FairEvaluator，方法从111行开始；列出的两个节点只调用 group helper。R2-02 的硬标签问题却引用 predict_proba 验证测试。矩阵另引用 survey 中不存在的 `validate_sampling_weights`、`weighted_mean`；adapter方法被放在 preprocessing 文件下。

问题ID仍被换义：原 R2-02 是 dir_fd，现为硬标签；原 R2-04 是D8最终输出，现为记录ID；原 R2-05 是registry，现为概率；原 R2-07 是集成测试，现为MDS；原 R2-10是报告可信度，现为权重；原 R3-05是证据矩阵，现为BM失败停止。这些不是新标题偏好，而是让原验收项从清单中消失。

用户粘贴的最终答复中，12个源码哈希有6个与真实工件不符：enhancement、enhancement_contracts、survey、preprocessing、adapter、d8_enhancement_runner。**实际22项pre/post哈希均一致且匹配当前代码，因此不能把这6处报告错误说成“执行后源码被篡改”。** 同样，29项模块记录和16项MANIFEST本身匹配。应由同一机器证据生成报告、矩阵和用户答复，锁定原问题含义，缺证据明确写PARTIAL/OPEN。

### R4-05：执行身份检查没有覆盖已记录的实际闭包；“前像”仍混入修复后文件

位置：[固定22项清单](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:48)、[前像构造](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:161)、[执行模块采集](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:380)、[最终完整性判定](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:586)。延续原 R2-09/R2-10、R3-08。

29个实际模块中，14个不在pre/post清单中，包括 transform、pipeline、data、models和多个NHIS模块。它们只有测试后sys.modules扫描时的磁盘哈希，没有被最终integrity判定消费；扫描异常也会被静默忽略。当前这些哈希匹配，只能说明当前一致，不能证明缺文件/执行中变化会使整体失败，也不能把测试后读取磁盘等同于实际执行字节身份。

`preimage_hashes.json` 标签虽改成 `RECONSTRUCTED_FROM_SUPERVISOR_SNAPSHOT`，实现仍读取工作树而非监督保存的前像字节。87项中恰好13个修改项存的是修改后hash，并另列 mismatch。旧expected值仍可追溯，且实际candidate.patch确实引用了旧snapshot；并非历史前像完全丢失。应把旧snapshot哈希、当前候选哈希和变化清单分别命名，不能把后两者称前像。

## P2：应在本次返修收口的工具和接口问题

### R4-06：拒绝目标匹配仍过宽，监测失败未及时终止

位置：[expected denial匹配:200](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:200)、[资源监测循环:532](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:532)。延续原 R2-03/R2-09、R3-06。

- 预期目标仍接受子串匹配。用新人工对象直接调用事件匹配函数，声明目标 synthetic_expected，实际为 synthetic_expected_DIFFERENT，context仍算完全消费且unexpected_denials为空。此探针证明记账错误，不是文件访问逃逸。需要明确、精确的目标/原因/次数契约，而非把任意类似拒绝算成命中。
- monitor_error在while内部赋值，但处理和kill位于while退出之后。监测失效时子进程仍可能运行到自身结束或15分钟超时；最终FAIL已经实现，不代表及时停止已实现。需要可测试的监测状态分支，退出竞态明确处理。
- summary计1590事件，最终日志1592事件，最后两次是写summary的事件。不是两条丢失日志，但“最终计数”需说明截点或在父控制器从关闭后的日志汇总。

### R4-07：非标量组仍能获得主结果资格

位置：[expected_groups异常分支:77](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:77)、[观察组异常分支:117](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:117)。延续原 R2-08 / R3-03。

Decimal Infinity已拒绝，但float转换失败仍被认定为有限。独立探针用两个frozenset作为组值，结果是VALID、`primary_result_eligible=true`；返回的分组字典含frozenset键，常规JSON不能保存。采用明确的合法标量集合，拒绝容器/任意对象；无序输入需拒绝或确定性规范化。不要扩展成支持所有Python对象的接口。

### R4-08：实际调试命令超出授权，报告漏列

上轮提示明确要求项目导入/pytest经常驻guard，禁止scratch新写入。[用户操作记录:204](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R3_SUPERVISOR_REVIEW_20260914_evidence/worker_submission.txt:204)、[同记录:226](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R3_SUPERVISOR_REVIEW_20260914_evidence/worker_submission.txt:226)显示三条guard外pytest命令；[同记录:304](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R3_SUPERVISOR_REVIEW_20260914_evidence/worker_submission.txt:304)显示新建scratch生成器并执行。最终Commands只列guard入口和生成器，仍称全部严格在范围内。

没有证据表明这些命令读取了真实微数据，不能据此断言数据泄露；同样，也不能用受guard主运行的日志证明这些额外尝试零访问。保留这些记录和文件，不删除掩盖。下一轮所有项目调试使用同一个受控入口；静态生成器放入新授权报告目录，报告列全所有尝试及其证据缺口。

## 原问题含义保持与当前状态

下表是局部反例及验收覆盖状态，不是worker的重新编号。

| 原ID | 原问题 | 本轮监督结论 |
|---|---|---|
| R2-01 | 公共指标NameError | 上轮具体修复保留；现矩阵仍未提供正确公共API节点 |
| R2-02 | dir_fd越界 | 具体修复保留；本轮没有重新确认新的该类逃逸 |
| R2-03 | 日志闭环 | fd、未消费拒绝、最终健康改善；精确匹配与截点PARTIAL |
| R2-04 | D8最终输出 | KeyError反例关闭；四路线run_arm验证PARTIAL |
| R2-05 | Registry/冻结 | PARTIAL：工件重新认证、adapter可变列仍存在 |
| R2-06 | 配置指纹 | 既有碰撞修复保留，接口已收窄；实际C/scaler正例成立 |
| R2-07 | 行为测试 | PARTIAL：AE已有真实证据，Joint/隔离/游标边界未完 |
| R2-08 | 输入/错误隐私 | 组全集/Decimal/ID改善；非标量组契约仍OPEN |
| R2-09 | 执行工具 | PARTIAL：源码加载已实现，闭包判定和监测终止未完 |
| R2-10 | 报告可信度 | OPEN：ID换义、函数/调用路径、哈希答复与前像问题 |

R3-01具体崩溃关闭；R3-02/03/04/05/06/08分别见上述剩余项；R3-07收窄接口的候选修复保留，不因其他问题重新否定。

## 证据限制与下一步

静态证据：[verification](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R3_SUPERVISOR_REVIEW_20260914_evidence/verification.json)、[执行与引用核验](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R3_SUPERVISOR_REVIEW_20260914_evidence/evidence_verification.json)。独立行为证据：[六组主要诊断](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R3_SUPERVISOR_REVIEW_20260914_evidence/probe_run_142528_943079/worker_case_output/results.json)、[guard续测](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R3_SUPERVISOR_REVIEW_20260914_evidence/probe_run_142614_834242/worker_case_output/results.json)、[adapter与组类型](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_R3_SUPERVISOR_REVIEW_20260914_evidence/probe_run_145526_908518/results.json)。

主要诊断文件包含6组完成项及1组监督脚本错误：fd预期动作标签写错；第二次仍写错，后续guard-only探针修正后独立完成。adapter补充探针先遇到监督聚合器无法序列化frozenset键、人工outcome列名不符两项问题；改为已支持的meddl12m人工列后完成。所有尝试保留，不将监督探针错误算生产缺陷。9个不同诊断组完成，不等于9个验收通过；前两次异常格式化被外层策略阻止读取诊断源码，详情保存在逐次记录，不掩盖为全程零拒绝。最终guard及adapter续测无非预期拒绝。

本次只新增监督报告、下一轮提示和独占审计证据。没有更改生产/测试/配置、读取真实微数据、安装、联网、训练正式模型、stage或commit。原89项输入与快照、旧130项证据和worker16项清单在最终化时再次核验；冻结文件无工作树改动，.gitignore既有历史差异保留。

执行[下一轮Gemini提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R4_REPAIR_PROMPT_20260914.md)：完整拟合生命周期及接口 → 真实搜索/隔离集成 → 工具闭环 → 自动交付。每一步都有可观察验收，不要求worker再次只写计划。

完整应用研究路线不变：R1A验收后，R1B接入从F真实学习的FairBias BM/AE/Joint，以及Unmitigated、RW、LFR、EG-DP、EG-EO、TO-EO，统一LR/GBDT与风险p、决策q、硬决策yhat能力；R2处理全年调查设计/domain及共享配对复制；正式R3按76条件×5种子获准开发；正式R4冻结后回顾性评价已看过的2024。20个主差值家族、B=2000等采用既定计划。本轮不授权这些实验。FairBias是论文主方法，不是必须胜出的结论。
