# FairBias 最新返工候选的独立验收

日期：2026-09-13。审查对象：工作者最新提交审查的 15 个未提交候选文件及其验证材料。

**决定：REPAIR。** 上轮列出的 11 个反例已经通过，但新增配置导出入口存在确定的运行异常，命名空间隔离仍未满足约定，状态哈希变更也缺少历史兼容处理。本轮没有确认 P0，不作 REJECT；依据 [执行协议第 3 节](/Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md:24)，不批准提交或真实数据重跑。

本次只新增监督者报告和证据，没有修改工作者实现、测试、历史 run 或冻结文件。名义编码、总体调查推断、stress-elbow 稳定性、全特征全局排序继续作为已披露的后续方法学任务，不要求本轮实现。

## 1. 验证结论

分支为 `research/nhis-fairbias`，HEAD 为 `67e6659fa65249a8842e34af5d8969629efe4bca`。

| 独立检查 | 结果 |
|---|---|
| 当前 15 个候选文件与工作者散列清单 | 15/15 一致 |
| 六文件核心套件 + D6 指定 test_03 | **156 passed in 64.30s** |
| 复制到新临时目录复跑工作者 11 项探针 | **11 PASS / 0 FAIL** |
| 补充边界与集成检查 | **13 项：4 PASS / 9 FAIL**；含同一问题的多个观察、两项绕过首个构造错误后的条件性检查及一项既有 schema 问题，不等于九个新增独立 Bug |
| 工作者证据 manifest | 10/10 一致 |
| 上轮监督者证据 manifest | 33/33 一致；更早一轮为 15/15 一致 |
| 三个历史 R4 主 JSON | 与上轮散列相同 |
| 14 个继承文件和 `.gitignore` 对 HEAD | 无本轮未提交差异 |

解释器与依赖为 Python 3.13.2、scikit-learn 1.7.1、NumPy 2.2.6、pandas 2.3.1、pytest 8.4.1。核心测试在独立持续数据/网络访问 guard 下运行；唯一记录的打开尝试为 [D8 预期 sentinel](/Users/lkc/Downloads/code_v_0_3/tests/test_nhis_d8_synthetic_contracts.py:1143)，在打开 parquet 前被阻断。两组探针的访问尝试日志均为空。未执行真实 NHIS/CSV 模型训练。

证据在 [本轮证据目录说明](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_THIRD_REVIEW_20260913_evidence/README.md)。补充检查使用合成数据、历史聚合变换字典和脚本的精确输出代码片段；没有运行完整 R4 主流程。首次检查遇到的异常也保留在日志中，不把未完成的探针运行记为通过。

## 2. P1：新增配置导出入口直接抛出 TypeError（原 R3）

**位置：** [D8EnhancementRunner.get_active_parameters](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:569)、[AE 构造函数](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:39)、[R4 新调用点](/Users/lkc/Downloads/code_v_0_3/scripts/run_nhis_d8_r4_substantive.py:316)。

新方法构造 `FairAccuracyEnhancement` 时漏传两个必需参数 `cate_attrs`、`num_attrs`。无需模型拟合或数据读取，直接调用 `D8EnhancementRunner(allow_real_data=False).get_active_parameters()` 即得到：

```text
TypeError: FairAccuracyEnhancement.__init__() missing 2 required positional arguments: 'cate_attrs' and 'num_attrs'
```

R4 主脚本现已接入这个方法，因此在前置步骤成功的情况下会在 Step 3 配置导出处中断。156 项测试与 11 个旧探针没有覆盖这个新公共入口；它们测试的是 AE 自己的参数方法及默认值比较。

**进一步的配置一致性问题：** 为检查构造错误之后的代码，监督者仅在临时 mock 中补齐 dummy AE 的空特征列表，没有编辑源码。随后执行主脚本原样提取的 Step 3 输出片段，发现：

- seed=37 时，`active_parameters.random_seed=37`，但 `frozen_config.random_seed` 仍为 0；seed=0 和 seed=37 的 `r4_config_sha256` 相同。该散列只覆盖部分更新后的字典，没有绑定实际随机种子；字典内 classifier 的 random_state 也仍是固定值。详见 [配置复制/散列](/Users/lkc/Downloads/code_v_0_3/scripts/run_nhis_d8_r4_substantive.py:325)和 [固定 seed](/Users/lkc/Downloads/code_v_0_3/scripts/run_nhis_d8_r4_substantive.py:99)。
- `smoke_test=True` 的参数方法仍声明 C3/C4 预算为 5/10；实际 [C3](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:904)、[C4](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:1051)预算表达式为 2/3。此检查直接求值实际预算表达式，不声称执行了完整 smoke 实验。

这两项是**临时补齐构造参数后**可见的后续问题，不是当前未修改入口已经成功导出的证据。当前默认 R4 的 smoke=False、seed=0 不触发这两个值差异，但其公共参数接口宣称的“实际生效配置”仍不成立。

**验收条件：** 公共 getter 无需数据即可调用；参数输出与实际 engine 构造使用同一份有效配置，避免 dummy 对象与另写预算分支漂移。对默认值、非默认 seed 和 smoke 模式做合成输出测试。若冻结主流程仅允许 seed=0，应明确拒绝其他 seed；若允许，应更新完整有效配置及其散列。只补上两个缺失参数尚不足以关闭 R3。

## 3. P1：记录命名空间只完成部分检查（原 R5）

**位置：** [全值相同判断](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:117)、[首行年份推断](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:128)、[集合构造](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:139)、[指纹](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:153)。

上轮的 RangeIndex 部分交集反例已修复，默认索引类型不再决定是否检查。但仍有以下边界：

| 独立输入 | 当前结果 | 问题 |
|---|---|---|
| 不同且明确的年度来源，恰好相同的 X/y 值和本地索引 | 拒绝 | 先比较值、后处理来源，仍把相同观测值当成同一记录。 |
| fit 年份为 `[2022,2023]`、IDs 为 `[0,1]`；selection 年份为 `[2023,2024]`、IDs 为 `[1,2]`，共享记录为 `2023:1` | 接纳 | 将每个分区首行年份用于整个分区，掩盖实际共享记录。 |
| fit 内 IDs 为 `[0,0,1,2]`，selection IDs 不重叠 | 接纳 | 转成 set 前没有拒绝分区内部重复记录 ID。 |
| 数据及索引不变，仅显式来源由 A/B 改成 C/D | fit/selection 指纹都不变 | 新增来源没有进入审计身份指纹。数值计算未必改变，但来源追溯和上下文身份不完整。 |

混合年份例子验证通用 EvaluationPartition 自己新增的自动年份推断分支，不代表已经在历史 NHIS 年度 adapter 中观察到泄漏。该例应按逐行来源判断交集，或在只支持单一来源的契约下明确拒绝混合来源。

新 [adapter 检查](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:122)已正确拒绝 `unspecified` 的预拟合对象，这个具体反例可以关闭。它仍只校验年份/角色，没有对注入预处理器与 adapter 的特征注册配置做一致性核验；应补充兼容性检查或明确其输入约束，不将其表述为完整来源认证。

**验收条件：** 以明确的来源与稳定记录 ID 作为身份，不用 X/y 数值相等替代身份判定；检查分区内重复及跨分区交集。混合来源逐行处理或明确拒绝，不能只读取第一行。来源纳入指纹。保留同来源交集必须拒绝、独立来源同号及相同观测值应可用的正反例。

## 4. P1：无损状态序列化未与历史哈希约定隔离（原 R1 的兼容性）

**位置：** [移除舍入的序列化](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_state.py:30)、[公用 changed_dict_hash](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_state.py:41)、[冻结预期值](/Users/lkc/Downloads/code_v_0_3/scripts/run_nhis_d8_r4_substantive.py:163)、[历史状态比较](/Users/lkc/Downloads/code_v_0_3/scripts/run_nhis_d8_r4_substantive.py:649)。

取消舍入、无损保存候选参数是正确方向。但该公用函数同时承担运行时缓存与历史状态识别，两个用途没有版本隔离。

用同一份冻结 `D6_ARM_003` 变换字典，分别调用上轮快照函数与当前函数，得到：

| 同一输入 | 状态哈希 |
|---|---|
| 上轮八位小数序列化 | `38fa54a06a9c6427` |
| 当前无损序列化 | `95ce9da442e66adb` |
| 历史 R2/R4 锚点 | `38fa54a06a9c6427` |

差异来自 `agep_a.power=0.14285714285714285`。其他三臂没有这一非整数指数，当前比较仍一致。**字典内容没有变，哈希规则变了。** 当前无损结果也与 D6 包装文件内较长的逻辑 SHA 前缀一致，说明不同历史工件原本就存在不同序列化约定，不能仅凭统一的 `hash` 名称直接互比。

此外，现有 R4 [第 653 行](/Users/lkc/Downloads/code_v_0_3/scripts/run_nhis_d8_r4_substantive.py:653)请求历史记录的 `canonical_hash`，实际 R2 文件使用 `d6_hash` / `d8_hash`，会先遇到 KeyError。这是本轮核查发现的**既有 schema 不匹配**，不是归因于此次序列化修改；修正字段后仍须处理上述不同哈希规则的比较。

**验收条件：** 保留运行时无损参数，给状态哈希增加明确方案/版本。历史对照使用对应历史方案或经校验的规范化内容比较，显式记录新旧规则；检查四个冻结字典和实际历史 schema。不能重新引入有损缓存键，也不能覆盖历史 JSON 或直接改旧 expected hash 来让测试通过。

## 5. 已关闭事项及 P2 证据修订

本轮确认了实质修复：几何配置变化、近邻候选指数和跨上下文 cycle 的旧 engine 反例均通过；RangeIndex 部分交集与未知来源预处理器的旧反例通过；实际平分行为与新描述一致；布尔值序列化已修复；空样本/权重溢出仍正确拒绝。

R4 新增 [candidate_audit_events.json writer](/Users/lkc/Downloads/code_v_0_3/scripts/run_nhis_d8_r4_substantive.py:576)已经存在。监督者将真实 AE 合成事件送入原样提取的 writer，写出一条事件，保持 `strictly_feasible=false`、`relaxed_feasible=true` 及 JSON 布尔类型。因此“没有候选 writer”的上轮问题关闭；完整主流程可达性仍受第 2 节运行错误限制。

[Erratum](/Users/lkc/Downloads/code_v_0_3/docs/reports/NHIS_D8_R4B_SUPERVISOR_ERRATUM.md:110)现在准确区分历史字段、后验诊断和未来 schema；Arm 4 已限定为共同配置变更下的路径敏感性，R4 stress-elbow 与探索固定 2D 的关系也已澄清。上述 R7 返工要求可以关闭。[survey 注释](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/survey.py:62)已明确资格标志仅为基础数据检查，不代表设计方差可估计。

工作者材料另有两处应自动更正的 P2 证据错误：

- 本次引用的上轮监督者 manifest 包含 **33** 项，不是报告中的 15 项；33 项实际均匹配，没有发现篡改。
- 报告给出的 parquet sentinel SHA 为 `9bc0be2f...`，与 [冻结常量](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d6_temporal_runner.py:124)和历史 reference manifest 的 `49f415132ff0be0228f8533f9f74c48cd79ff6fa8be66db085f7329d9b083383` 不符。本轮没有读取原始 parquet 来计算其当前散列；这里指出的是所声称来源与已核实参考值不一致，不能把未读取的文件写成已重新验哈希。

P3 方法学任务保持延期，不因测试通过而宣称完成，也不要求本轮扩展实现。

## 6. 下一次验收的有限范围

1. 修复 getter 的运行错误，让默认与非默认参数输出、预算、科学配置散列一致；提供无需真实数据的 writer 集成验证。
2. 完成记录身份的来源/重复/交集/指纹边界，并明确预拟合配置兼容要求。
3. 隔离无损运行时哈希与历史哈希协议，修正读取实际历史 schema 的比较路径；保持历史工件原样。
4. 自动生成准确证据引用，复跑现有核心套件、已经通过的 11 项探针及本轮新增反例。使用新目录和持续 guard，报告区分无 mock 的失败与隔离首个错误后的条件性结果。

工作者按协议第 9 节提交材料，以 `STOP — waiting for Codex review.` 结尾。已通过的修复保留，不要求重新实现已关闭项目。
