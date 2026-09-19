# FairBias 第四次候选验收

日期：2026-09-13。对象：工作者最新返工报告中的 15 个未提交候选文件。

**决定：REPAIR。** 上轮 13 个具体反例已通过，配置导出和历史状态比对已取得实质进展。但“运行时使用无损哈希”和“预拟合配置兼容检查”未接入正确调用路径，记录身份还存在数值类型绕过。当前不能确认有限返工范围已完成，因此不批准本轮提交或真实数据重跑。未确认 P0，也未发现支持 REJECT 的证据。

本轮只新增报告与证据；未修改实现、测试或历史工件，未执行 Git staging/commit。继续保留已披露的四项方法学延期任务，不要求本轮实现。

## 1. 独立验证

分支：`research/nhis-fairbias`。HEAD：`67e6659fa65249a8842e34af5d8969629efe4bca`。

| 检查 | 结果 |
|---|---|
| 15 个候选文件的大小/SHA-256 | 全部匹配工作者清单 |
| 六文件核心套件 + D6 指定 test_03 | **163 passed in 63.68s** |
| 工作者 13 项探针复制到新目录复跑 | **13 PASS / 0 FAIL** |
| 追加 7 项调用路径/集成检查 | **3 PASS / 4 FAIL**，四个失败观察分属下述三个问题 |
| 最新工作者证据 manifest | 9/9 匹配 |
| 上轮监督者证据 manifest | 60/60 匹配；更早一轮为 33/33 匹配 |
| 三个历史 R4 主 JSON | 与前次审查散列相同 |
| 14 个继承文件和 `.gitignore` 对 HEAD | 无本轮未提交差异 |

使用 Python 3.13.2 / scikit-learn 1.7.1 / NumPy 2.2.6 / pandas 2.3.1 / pytest 8.4.1。所有测试和探针在新的临时目录运行，持续 guard 阻止真实数据打开及网络请求。核心套件仅记录一次 [预期 D8 sentinel](/Users/lkc/Downloads/code_v_0_3/tests/test_nhis_d8_synthetic_contracts.py:1143)，在打开 parquet 前阻断；两组探针的访问尝试日志均为空。未执行真实 NHIS/继承 CSV 模型拟合。

证据与复现脚本见 [本轮证据说明](/Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_FOURTH_REVIEW_20260913_evidence/README.md)。AE 探针使用人工效用/几何值隔离控制流，分数不是真实性能估计；其余使用合成数据和历史聚合变换字典，不运行完整 R4 main。

## 2. P1：无损版本存在，但运行时仍调用有损默认值

**位置：** [默认版本](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_state.py:21)、[hash 默认版本及别名](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_state.py:53)、[AE 当前状态](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:528)、[AE 候选状态](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py:635)、[配置指纹](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:289)。

新代码提供了 `v1_legacy_8dec` 和 `v2_lossless`，但 `canonical_json_dump()` 与 `hash_transform_state()` 默认都设为 v1。AE 各调用点没有传入版本，配置指纹也继续调用默认序列化。新增 `hash_transform_state_lossless()` 没有被这些运行时路径使用。D8 joint 的 [当前状态](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:1045)、[BM 候选](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:1071)、[AE 候选](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py:1137)同样使用默认 v1。

**真实 AE 调用链中的两个反例：**

- 当前幂为 `3.000000001`，候选为 `3.000000002`。默认状态哈希相同，显式 v2 哈希不同。未修改实现的 AE 只评价基准，随后把候选拒绝为 `CYCLE_DETECTED`。仅在诊断对照中将 AE 的哈希依赖临时替换为无损函数，同一搜索正确选择 `x`。这证明错误在调用路由；没有把对照结果当成当前实现通过。
- 同一 engine 的 `min_utility_gain` 从 `2e-9` 改为 `1e-9`，配置指纹仍相同。人工设置增益 `1.5e-9` 后，复用 engine 跳过候选，新 engine 对照则接受。仅修变换状态哈希，仍不能消除配置缓存中的舍入碰撞。

这两个例子针对公开参数接口的精确性，不证明冻结默认网格的历史结果已经受影响。工作者现有测试用近邻指数 `3.00001/3.00002`，它们不会在八位小数处碰撞，因而仍可全部通过。

**验收条件：** 历史比对显式使用对应 v1；AE、D8 joint、排名/候选/配置缓存等运行时身份显式使用无损方案。不要靠同一个共享默认值同时满足两种语义。补充真实 engine 的小于八位小数差异反例，并同时验证四臂历史参考仍通过；不改历史 JSON，不恢复有损候选签名。

## 3. P1：预处理器兼容检查读取错误属性与错误 schema，实际被跳过

**位置：** [新增兼容检查](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py:131)、[真实预处理器属性](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:164)、[真实注册表结构](/Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py:175)。

adapter 读取 `preprocessor.feature_registry`，真实 `NHISPreprocessor` 保存的属性却是 `registry`。因此正常对象的 `prep_reg` 为 None，新增检查直接跳过。即使只修属性名，下面的 `get("features", {})` 仍与真实 schema 不符：真实配置使用 `primary_core`、`expanded_utilization`、`feature_lists` 等键，空集合条件又会绕过比较。

**已复现：** 使用真实 NHISPreprocessor 对合成 2022/development_train 数据拟合，分别注入匹配注册表和修改过 `AGEP_A.substantive_codes` 的注册表。保留 adapter 原始年份/样本量/角色检查，仅替换文件读取为生成的 89,802 行年份/角色元数据。两个对象都被接纳；其中配置不一致者应被拒绝。诊断明确确认真实对象没有 `feature_registry` 属性，其 `registry` 也没有顶层 `features`。

这不是已经观察到历史 D6/D8 的错误拟合，而是新防御性检查没有发挥作用。年份/角色对 `unspecified` 的拒绝仍然有效，应保留。

**验收条件：** 对真实对象的 `registry` 建立兼容约定，比较实际特征列表与影响预处理的语义配置，如类型、有效编码、缺失处理规则；不能只检查不存在的键或特征数量。必要配置缺失应拒绝。用真实已拟合对象做匹配/不匹配双向测试，不用人为添加 `.feature_registry` 属性的假对象证明通过。

## 4. P1：年份的整数/浮点表示可绕过共享记录检查

**位置：** [逐行 ID 构造](/Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py:146)。

从首行年份改为逐行来源解决了原反例，但当前直接用字符串格式化年份。整数 `2023` 与浮点 `2023.0` 会生成不同记录身份。

**已复现：** fit 年份 `[2022,2023]`、IDs `[0,1]`；selection 年份 `[2023.0,2024.0]`、IDs `[1,2]`。共享记录仍是同一 `2023:1`，相同特征/标签，但两侧生成 `2023:1` 与 `2023.0:1`，分区被接纳。仅改变年份列的数值 dtype 就改变隔离结论。

**验收条件：** 在生成记录身份前规范化年份及 ID 类型，明确拒绝无效/缺失年份，使用无歧义的结构化身份。覆盖整数/浮点年度的等价表示和原有同来源交集、独立年度同号、分区内重复的正反例。此项仍属于已约定的记录身份边界，不涉及改变研究分区或模型。

## 5. 可关闭事项和证据表述

以下项目本轮可以关闭：

- 新 getter 已可直接调用，无需上轮补齐参数的 shim；默认与非默认 seed 的科学配置及散列、smoke 预算的旧反例通过。
- 不同明确来源的相同观测值、整数年份混合来源交集、分区内重复、来源指纹的原反例通过；第 4 节是同一身份规则尚未覆盖的数值类型边界。
- 历史四臂兼容比较已经通过。监督者额外执行了当前 Step 11 的原样状态比较代码块，四臂均 PASS；移除一个参考臂时正确返回 FAIL，没有 KeyError。Arm 3 同时记录 v1=`38fa54a06a9c6427`、v2=`95ce9da442e66adb`。**历史兼容分支已修复；运行时仍用 v1 是第 2 节的另一半问题。**
- 候选审计输出、布尔类型、勘误历史字段归属、复合路径敏感性和调查推断范围说明保持前次已通过状态。

P2 证据修订：本次报告引用的 `FAIRBIAS_THIRD_REVIEW_20260913_evidence/manifest.json` 实际有 **60** 项，不是工作者写的 33 项。60 项全部匹配，没有发现篡改。parquet 散列现在已正确标明为冻结参考常量，未声称本轮重新读取验证，这项更正接受。报告所称“13 项全部未打 patch/shim”应限定为不再需要 getter 构造 shim；其候选 writer 探针仍使用效用/几何 mock，这一点应如实披露。

P3 方法学任务仍按既定范围延期：名义编码、总体调查推断、stress-elbow 稳定性及全特征排序。它们不构成本轮新增的返工要求。

## 6. 下一次验收的有限清单

1. 明确接通所有运行时无损哈希与配置指纹调用，保留历史兼容分支。
2. 修正真实注册表属性/schema 的兼容检查，提供真实预拟合对象的正反例。
3. 规范化记录身份中的年份类型，保留已通过的分区用例。
4. 自动生成准确证据清单，在新目录及持续 guard 下复跑核心套件、现有 13 项探针与本轮调用路径反例。

不需要重跑真实 NHIS 来完成上述验收，也不要求回滚已通过的修改。工作者按协议第 9 节报告，结尾 `STOP — waiting for Codex review.`。
