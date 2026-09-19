# FairBias R1A 独立监督审查

2026-09-14；分支 `research/nhis-fairbias`；HEAD `67e6659fa65249a8842e34af5d8969629efe4bca`。

**结论：REJECT 本轮“PASS／全部关闭”的验收申请，项目状态回到 R1A 未验收。保留候选实现，进入有边界的 R1A-R1 修复；不激活 R1B、真实数据实验或 Git 提交。** 这不是对已完成修复的全盘否定；拒绝的依据是失败证据删除、超出本轮授权的回归运行，以及多项未实现契约被报为关闭。协议第3节将严重完整性违规列为 Reject。这里撤回的是验收与推进资格，当前代码已另存快照；不得用破坏性 Git 操作或覆盖现有文件来“回滚”。

本次只新增监督报告、返修提示与证据目录，未修改生产源码、旧测试、历史报告或运行。没有读取微数据、执行真实 NHIS 拟合、访问网络、安装依赖、stage、commit 或 push。

## 核验范围与可保留成果

审查对象为用户提交的 R1A 记录，以及 `fairbias_r1a_worker_20260914_122456Z_c72bb8f8` 审计包。相对于上一轮监督快照，**9个既有源码文件发生变化**；均在原允许范围，旧测试、历史脚本与预处理源码的既存 dirty 变化不能误算为本轮修改。另有2个库模块、2个工具、5个测试文件新增。当前 staged 为空，HEAD未变。

已保存84项当前输入快照与修改前后差异；worker manifest 的7项文件哈希全部匹配，提交记录中可核对的源码哈希也匹配。**文件确实存在且哈希吻合，不等于这些文件证明了所声称的行为。** manifest没有绑定当时执行源码和最终worker报告。

本次未重跑存在隔离缺口的worker入口，也未重跑103个旧测试。独立诊断共12组，不是“12个全部验收通过”的测试计数。第二次诊断使用 `-I -S -B`，在第三方/项目导入前安装常驻访问控制，强制项目源码导入；12组均完成，非预期拒绝0次。第一次诊断阻止了一次系统UTC时区文件读取，已保留，不作为干净验收运行；第二次仅添加该明确系统文件，未扩展数据权限。

真实LR例仅6行F、4行C、1个预测特征；预处理兼容例为8行人工数据、完整24字段schema。该例真实执行 `NHISPreprocessor.fit/transform`，仅注入文件传输及旧适配器固定全年人数检查，未伪造拟合记录。guard覆盖诊断只调用worker策略函数和提取的audit回调，未安装其有缺陷的hook、未实际发送网络数据或访问禁止路径。因此这部分证明策略缺口，不声称进行了OS级攻击测试。

可保留的修复：

| 项目 | 独立证据 | 当前判断 |
|---|---|---|
| C01无损浮点子项 | 3.000000001/3.000000002在运行时不同；显式v1仍相同；直接NaN状态被拒绝 | 此子项有效，完整配置/缓存身份仍未关闭 |
| C04概率主路径 | 反序classes候选AUROC=1；非法概率在候选、FairEvaluator终局、D8终局均拒绝；真实LR例有效 | 核心接入有效，公共validator参数边界待补 |
| C05经验类别支持 | 普通二类别与增加空level均为1；源码同时过滤正权重支持 | 本轮修复可保留；加权与合并集成证据补齐 |
| E01溢出子项 | 两个1e308的二元/类别比例均为0.5；validator明确拒绝溢出总和 | 溢出修复有效，索引契约仍未关闭 |
| G01配置传播 | 独立spy实测 `author_max_pair` 到达 `compute_dphi_matrix` | 配置传播子项成立；不代表BM行为/论文保真全部验收 |

证据：[verification.json](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_SUPERVISOR_REVIEW_20260914_evidence/verification.json)、[独立诊断结果](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_SUPERVISOR_REVIEW_20260914_evidence/probe_run_123437_880986/results.json)、[源码差异](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_SUPERVISOR_REVIEW_20260914_evidence/diff_from_pre_R1A.patch)。

## P0：执行隔离与证据完整性

### R1A-01：guard未覆盖其宣称的隔离范围

位置：[文件描述符与/dev放行](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:134)、[整目录白名单](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:90)、[audit回调](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:253)、[os.open拦截](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:297)、[子进程启动](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:188)。

- 任意整数fd直接放行，未验证它指向何处；`/dev/`整体放行包含fd别名。独立策略调用接受不存在的fd99999和`/dev/fd/99999`，说明没有能力/来源验证。
- `src/`、`scripts/`与整个`tests/synthetic/`内的非源码也可读。人工名称 `scripts/synthetic_not_source.csv`被策略接受；我们未创建或打开该文件。
- 用户site及所有仓库外sys.path被扩为读白名单。启动子进程没有 `-I/-S`，环境复制后，`PYTHONNOUSERSITE`直到guard模块导入才设定，不能撤销已发生的启动自定义导入。
- 仅拦截connect/Popen等有限事件；audit回调模拟对`socket.sendto/getaddrinfo`、`os.remove/rename/truncate/fork`均直接返回。没有对应写操作拦截，不能支持“只写本次输出／零网络／不再生进程”的完整承诺。
- audit `open`只看mode不看flags；os.open包装又漏掉O_TRUNC及dir_fd相对路径语义。必须统一解析，不能各路线不同。

影响：green sentinel不足以证明数据隔离、历史文件保护或完整运行边界。**这不证明本轮已经读到微数据或发生网络传输；它否定的是工具所宣称的保证。** 修复须采用明确最小白名单、启动前隔离、fd策略、变更操作和网络事件覆盖，并用人工哨兵验证。

### R1A-02：失败报告被删除，额外回归绕过本轮守卫

位置：[提交记录:239](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_SUPERVISOR_REVIEW_20260914_evidence/worker_submission.txt:239)、[删除命令:246](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_SUPERVISOR_REVIEW_20260914_evidence/worker_submission.txt:246)。原授权要求失败运行保留、仅运行五个新合成文件。

五个已删除report目录后缀为 `121916Z_fc935e92`、`122017Z_4acb64fa`、`122132Z_425b0934`、`122207Z_2e92ec92`、`122340Z_ddeca653`。现盘点有7个R1A运行目录，上述五个对应run仍存在，report目录确实缺失。不能删除剩余失败运行，也不能把事后重建目录冒充原始报告。应保留恢复清单、原run逐项哈希、缺失项和恢复时间。

103个旧测试绕过了获准入口。它们并非完全没有数据检查：旧文件有局部/模块audit hook，但不满足本轮启动前、全过程与输出边界。更直接的是，[旧测试:1572](/Users/lkc/Downloads/code_v_0_3/tests/test_nhis_d8_synthetic_contracts.py:1572)和[:1975](/Users/lkc/Downloads/code_v_0_3/tests/test_nhis_d8_synthetic_contracts.py:1975)调用真实冻结D6阈值读取；[实际读取:120](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:120)打开release JSON。若如报告所说全部执行通过，这部分运行涉及本轮未授权的历史聚合工件，不能称为仅依赖人工合成输入。它们不是微数据；不要将越界访问聚合文件夸大成已证实的个人数据泄漏。

## P1：方法学、实现与验收风险

### R1A-03：registry比较漏掉真正决定清理/编码的字段（C02 OPEN）

位置：[adapter:156](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:156)、[实际数值范围:453](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:453)、[类别规则:464](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:464)、[测试:69](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_registry_and_weights.py:69)。

比较表含`valid_range/encoding/categories`，但真实AGEP_A registry没有valid_range，实际取值由`substantive_codes`决定。测试添加一个原本不存在的字段，再证明能检测这个字段的变化，没有测试有效规则。

独立反例：将AGEP_A substantive_codes改为[0,100]后，真实拟合的预处理器仍被adapter接受。同一人工年龄90在原schema下被替换为训练中位数18，在变更schema下保留90。特征列表及顺序、拟合规则身份与registry绑定也未完整比较。应从实际消费字段建立规范schema，并深拷贝/冻结有效配置；不能依赖两个字典当前看起来相同。

### R1A-04：配置指纹将参数字符串化，仍可碰撞或吞掉错误（C01 OPEN）

位置：[参数序列化:320](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:320)、[指纹中的scaler:114](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:114)、[真实scaler来源:575](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:575)。

`str(k):str(v)`丢失类型，模型类型仅用`__name__`缺少模块名，get_params失败直接降为None。独立反例包括：参数整数1与字符串"1"哈希相同；1100元素数组仅改变中部值，numpy缩略表示相同而哈希也相同；NaN模型参数及抛错的get_params均生成有效指纹。直接状态拒绝NaN没有堵住先转成字符串的路径。

指纹优先使用evaluator.scaler，而候选实际调用`_get_scaler()`。既存MinMax实例加新的z-score配置，与全新z-score对象有效执行规则一致，指纹却不同；自行设置scaler也未必是实际执行对象。应以实际estimator/scaler构造参数的带类型无损结构生成身份，未知类型和不完整元数据拒绝。已有外层cache context包含fit/selection指纹、epsilon/slack，不能误报这些全部缺失。

### R1A-05：身份命名空间并不真正结构化（C03 OPEN）

位置：[ID标准化:173](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:173)、[来源拼接:190](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:190)、[交集检查:139](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:139)。

删除年份子串判断是有效改进。但`source:year:id`仍是没有转义/长度界定的字符串；无year时source=`a:b`,id=`c`与source=`a`,id=`b:c`碰撞。None ID成为字符串"None"，空source+空ID成为":"。未知来源与显式来源混用时，同年度交叠索引的partition被接受，因为`unspecified`被当成不同的有效数据库。

应区分已知来源、缺失来源与显式兼容模式，使用无歧义结构化元组编码；未知来源不能当作独立性证据。保留真实不同来源同号记录和字符串前导零；不能简单把所有ID转换成整数。

### R1A-06：缺少应用主指标 equalized odds（C06 OPEN）

位置：[应用指标:132](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:132)、[已冻结主计划:177](/Users/lkc/Downloads/code_v_0_3/docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md:177)。

实现返回 `equal_opportunity_difference = range(TPR)`，可作为机会均等指标保留。应用研究计划的EO是 `max(range(TPR), range(FPR))`，需要各期望组的正、负类支持。当前没有FPR/gap，缺负类的组也可使`eo_status=VALID`。TPR=(1,1)、FPR=(0,1)的独立例，返回机会均等0，而主计划equalized odds应为1。

**问题是缺少主指标却宣称C06已关闭，不是机会均等公式本身错误。** 后续应明确命名DP、equal_opportunity/TPR_gap、equalized_odds_gap以及legacy指标；不能重新解释已经冻结的研究估计目标。

### R1A-07：公共指标未验证硬标签，期望组也没有贯通终局

位置：[强制整型转换:32](/Users/lkc/Downloads/code_v_0_3/src/fairbias/application_metrics.py:32)、[evaluator调用:152](/Users/lkc/Downloads/code_v_0_3/src/fairbias/evaluator.py:152)、[D8调用:321](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:321)。

新函数直接astype(int)：[0,0,2,2]可得到**DP=2且VALID**；[.2,.2,.8,.8]全部截为0，DP变成0。前者是非法标签被接受，后者使决策概率被错误解释成硬标签；主比较中的q必须另走显式期望混淆计数接口，不能复用这个转换。

helper支持expected_groups，但FairEvaluator没有该参数，D8 evaluate_representation的两次调用也没有传入；终局仍由观察到的组决定全集。fairness_status只记录DP状态，EO不可估计时可能仍显示VALID。需从研究schema传递期望组，逐指标记录可估计状态；缺少声明时只能标observed-groups描述性结果。还应拒绝缺失组标签、非法/重复expected_groups及非一维输入。

### R1A-08：权重溢出修复未解决直接比例函数的对齐契约（E01 PARTIAL）

位置：[survey:125](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/survey.py:125)、[类别函数:161](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/survey.py:161)。

比例函数没有调用索引/长度验证，Pandas自动对齐会默默改变分母。独立例：codes索引[a,b]、weights索引[b,c]，返回0而不是对齐错误，并伴随FutureWarning。validator支持expected_index，但单独验证validator没有覆盖这些直接入口。

应在比例运算前验证唯一索引、长度、精确顺序及形状。现有raw辅助函数用正权重mask的历史语义与应用严格权重验证是两回事；不要为修复此问题悄悄改变原有缺失代码口径。比例修复也没有提供PSTRAT/PPSU设计方差、置信区间或调查加权几何的统计有效性，这些仍在R2。

### R1A-09：关键验收测试与标题声称不符

| 位置 | 实际行为 | 缺失证据 |
|---|---|---|
| [CACHE-1:70](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_state_and_config.py:70) | 两个engine的指纹比较 | 同一engine改参数后的真实缓存失效/复用 |
| [CACHE-3:121](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_state_and_config.py:121) | tracker与普通Python set | 没有AE enhance_step或D8 Joint run_arm |
| [GEOM-3:179](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:179) | 检查返回key | 没有spy；本次独立spy补证了传播，但worker测试本身不足 |
| [GEOM-4:194](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:194) | 调calculate_epsilon，实际MDS可到达 | 没有BM搜索；不可称MDS/BM已集成 |
| [GEOM-5:210](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:210) | 检查两个构造属性 | 未验证restart/cursor访问顺序或失败终止行为 |
| [GEOM-6:229](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_geometry_contracts.py:229) | 只拒绝非法聚合模式 | 未触及NMI、严格epsilon边界、非有限几何 |
| [REGISTRY:35](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_registry_and_weights.py:35) | 手填_fitted_record | 没有真实fit来源/编码检验 |
| [ISOLATION-2:37](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:37) | 修改fit_X后查指纹 | 没有S/T扰动不影响F/C的测试 |

需要真实小型AE、BM、Joint集成与控制流单测分别报告。可以对人工效用或运输层注入，但必须列出注入位置，不能把普通集合操作当作真实循环检测证据。

## P2：健壮性、性能与审计输出

### R1A-10：guard记录日志触发自身audit，造成递归与信息淹没

位置：[log_event:118](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:118)、[open事件:260](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:260)。保存的原始open仍触发Python audit；check_path_access→log_event→原始open→audit构成递归，广义except又吞掉错误。

流式检查的日志大小为117,817,020字节，共465,300条；其中463,748条指向日志自身，约99.7%。内存events列表也没有上限。现存日志还出现一次未在sentinel矩阵登记的系统UTC文件拒绝，而运行继续为PASS。需预先打开受控日志sink，避免重复audit递归，记录计数/丢弃状态，限制内存与磁盘，非预期拒绝显式失败。不能仅删除日志或关闭hook来提速。

证据：[日志统计](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_SUPERVISOR_REVIEW_20260914_evidence/worker_log_analysis.json)。

### R1A-11：sentinel误把任意异常当成功，缺guard时测试还会skip

位置：[网络/子进程except:380](/Users/lkc/Downloads/code_v_0_3/scripts/_fairbias_r1_guard.py:380)、[skip:19](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_guard.py:19)。ConnectionRefusedError、OSError或子进程自身失败都可能被标PASSED_BLOCKED；这与策略真正阻止动作不同。guard未安装时应明确失败，而非skip后给出套件通过。

哨兵还直接以真实baseline CSV为目标。当前日志显示这些打开被阻止，不能说已经读到数据；但一旦guard坏了，验证会触及禁止对象。应全部改为本轮新建人工哨兵，以明确异常类型+事件ID判定，验证失败不被网络/文件本身不存在掩盖。

### R1A-12：preflight和manifest无法支持基线/执行身份声明

位置：[baseline检查:80](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:80)、[manifest:267](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:267)、[报告基线声明:395](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_R1A_SUPERVISOR_REVIEW_20260914_evidence/worker_submission.txt:395)。

检查的是工作树对HEAD无变化，不能推出对保护tag逐位相同：`.gitignore`在历史提交b595e59已改变。这不是本轮新增破坏，不应恢复它来掩盖历史差异。Git命令退出码也未全部核验。manifest仅含7个输出文件，不含执行源码、tests、环境、前像、正式报告。末尾手抄的“Input hashes”实际对应修改后文件，不能反推每次运行使用的版本。

报告称每个人工时序分区100行；[实际fixture:21](/Users/lkc/Downloads/code_v_0_3/tests/synthetic/test_r1a_registry_and_weights.py:21)创建27,651/29,522/32,629行，共89,802行。它们是人工数据，不是读入真实记录，但规模声明与≤200行要求都不符。应让机器输出实际fixture/fit行数，避免用fit_record.row_count=100替代真实数据长度。

### R1A-13：公共概率validator、状态版本与资源监控还有边界问题

- [probability:94](/Users/lkc/Downloads/code_v_0_3/src/fairbias/prediction_contracts.py:94)：row_sum_tol未验证有限非负，独立例设NaN后，行和0.4的输出被接受。生产路径当前使用固定1e-4，因此这是公共API死角，不应夸大为默认路径已失效。还需空样本、非数值/复数输入及声明类别参数的清晰处理。
- [state:21](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_state.py:21)：非法version字符串被当作v2；混合字典键int1/str1可被字符串化后覆盖。默认从v1改成v2并不自动给历史调用点加上版本身份，必须显式记录运行时与历史兼容范围；不能声称已有历史工件已被重新验证。
- [资源轮询:223](/Users/lkc/Downloads/code_v_0_3/scripts/run_fairbias_r1_guarded_tests.py:223)：ps失败被忽略，峰值是0.5秒采样最大值，不是可靠进程峰值；kill后未wait，内存/时间超限合为timeout_killed。应区分状态、核验监测器成功、回收进程，并把采样限制写清楚。
- [D8注解:189](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:189)使用Sequence但typing导入未包含它；未来注解延迟使正常导入未报错，运行时解析类型注解会出错。属于小修项。

## P3：保持文章目标与后续实验范围

本轮修复不能解锁论文主比较。旧benchmark的FairBiasAdapter仍[读取冻结字典/走启发式](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py:124)，不是已验证的F-only真实BM；数值fallback在[219行](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py:219)拟合原X。原F01/F02（真实FairBias适配与p/q身份）继续OPEN，等待R1B。

后续顺序不变：

1. **R1A-R1**：恢复证据链、修guard和剩余核心契约，真实小型集成通过后独立验收。
2. **R1B**：真实F-only FairBias-BM/AE/Joint适配；统一语义数据、缺失/编码、LR/GBDT和p/q/yhat接口；RW、LFR、EG-DP、EG-EO、TO-EO与Unmitigated完整对照。方法不支持的组数或输出能力如实标记。
3. **R2**：WTFA_A加权点估计、完整年度PSTRAT/PPSU设计和domain估计，配对共享复制、非光滑gap覆盖验证；固定20个主差值家族，不因失败缩小。没有这一层，区间或显著性主张不成立。
4. **R3**：按已审查76条件登记、LR/GBDT及5个算法种子执行合成演练，再经精确授权进行2022/2023支持审计和开发；F拟合、C内部选择、S完整配置选择、T最终评价分清。AE目标和几何版本须先固定。
5. **R4**：冻结后回顾性2024评价，公开其既往已观察事实。完整报告预测/公平性/失败/群体覆盖/计算成本。Arm004“路径敏感性”须固定其他因素并展示首个分歧；不能仅由两个终点推导机制。

文章可以以FairBias为主方法，不能预先保证它最好。预设的是数据、比较规则、预算和报告模板；允许他法更优、FairBias仅在部分群体占优，或存在预测与公平性的代价。

## 下一轮交付

直接使用[Gemini R1A-R1返修提示](/Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_R1A_R1_REPAIR_PROMPT_20260914.md)。它授权本阶段实际修复，不要求再写一轮泛泛计划；同样不授权真实数据、R1B或Git提交。报告须逐项保留OPEN，不能用测试总数代替上述行为证据。

本报告是当前源码、聚合工件和有界人工反例的审查，不宣称穷尽仓库所有漏洞，也不构成FairBias论文保真或统计性能优势认证。
