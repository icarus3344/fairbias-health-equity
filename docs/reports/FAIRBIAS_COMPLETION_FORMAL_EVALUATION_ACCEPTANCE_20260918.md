# 完整 FairBias：正式评价主管验收

2026-09-18。结论：**ACCEPT — 本批80个固定BM+AE/Joint模型的正式评价与合并结果已验收。** 此结论不等于算法普遍优越、完成扩展参数网格、关闭所有历史BM研究问题或论文已可投稿。

## 本次范围和执行

用户授权完整推进正式评价、预测与公平性分析及论文整理。全程本地执行；未访问或重启服务器、未训练/调参、未修改原925模型研究、未Git暂存/提交。只新增评价/统计/导出入口与测试、图表及报告。

原80训练产物、99训练来源文件及准备数据完成哈希核验。新分析分支`fairbias_adaptive_completion_v1`，95个分析入口/来源文件在S前绑定。政策见`docs/plans/FAIRBIAS_COMPLETION_EVALUATION_POLICY_20260918.md`，在本批S/T产生之前固定。历史2024 T已知，所有新产物均保留`known_T=true`。

流程：元数据准入 → 本机C重载决策校验及S预测 → 独立S选择检查 → study冻结 → 主管独立T release → 四臂并行本地T评价 → 独立统计审计 → 单独版本的合并导出。未根据S/T结果更换候选、阈值、种子、参数、比较家族或统计定义。

## 实现和发现的问题

- 新增`src/nhis_fairbias/benchmark/completion_evaluation.py`：精确80任务覆盖、来源/运行环境/模型/准备数据绑定、原对照登记映射、S冻结、study和release校验、T预测及原预测复用。
- 新增`src/nhis_fairbias/benchmark/completion_statistics.py`：固定主要20/次要316/合并336端点，公共年度设计、均值种子指标、共同PSU重抽样、成对BA线性化和EO保守投影。
- 测试阶段修复训练快照源码二次验哈缺口、重新hash后对照/比较映射错误、选择映射未重建、study B/seed/arm字段未强制检查等问题。修复均早于真实S/T。
- 原登记中24个LR对照槽位只用seed0，四个Arm002 LFR/FRAPPÉ组合为NOT_SUPPORTED。入口保留原种子规则与不支持槽位，不强制生成五seed或误称运行失败。
- 第一版合并因观测provenance额外携带年度人数`rows=32629`而拒绝`SHARD_IDENTITY_CHANGED`。四臂预测/统计已成功；未重跑。新增`scripts/merge_nhis_completion_archive.py`版本`completion_archive_merge_v2`，复用原已验证的语义provenance校验，并额外校验人数与各表一致。原冻结源、四臂产物、失败日志保持原样。**复现合并请使用该新脚本，冻结CLI中的旧merge子命令仍保留失败实现。**
- S中缺保护组支持会在非有限JSON输出时fail closed；本批实际S支持完整且全部成功。此泛化限制另记，不用默默填值修复本批结果。

## 独立验证

主管独立执行：新增完整评价契约**95 passed**（包含生成数据S→study→release→T）；已有调查/风险/选择/catalog相关契约**52 passed**；新归档导出正向及错误来源/人数/哈希/重复臂/缺配对回归**7 passed**。这是相关集合，不声称全仓库测试均运行。

80个C决策数组与Linux审计哈希逐字节一致；风险概率数组字节均不一致，已明确记录，未声称跨平台风险位等。numpy/scipy/sklearn/pandas/joblib核心版本匹配；Python补丁版本、架构与threadpoolctl有差异。新论文风险概率采用本机绑定的环境结果，不把跨平台差异解释为已量化的机器精度误差。

独立子代理审查全部源码/产物哈希、349模型身份、96条目/168配对、均值与seed SD、共同重抽样数组、成对BA区间和EO投影，17084条算术/身份断言通过，最大数值差6.66e−16。审查边界见`FAIRBIAS_COMPLETION_REAL_EVALUATION_REVIEW_20260918.md`：未重新从个体记录计算所有PSU方差，也未重载269个旧模型；原预测哈希与其已验收study绑定。

主管额外核验：原study→merged覆盖映射、四个summary/manifest哈希、新导出脚本哈希、全部新方法点估计、主要及合并家族结论方向、40个Joint/BM+AE匹配seed对的保存预测数组。6/8条件的五seed风险及决策数组完全相同；其余两格的差异完整保留。所有80结果中的几何seed与预测器seed均与登记seed匹配。

## 结果范围

| 项目 | 数量 |
|---|---:|
| 新BM+AE / Joint模型 | 40 / 40 |
| 复用原冻结模型预测 | 269 |
| 唯一模型总计 | 349 |
| 条件条目 | 96（90可评价，6缺模型） |
| 配对比较 | 168（156可评价，12缺模型） |
| 新方法在S满足EO≤0.10的配置 | 4/16 |
| 2024年度设计 | 32629人、52层、662 PSU、df610 |

6个缺模型条目均在Arm002：4个不支持方法/骨干槽位、2个旧固定BM缺完整种子槽位。新80个模型均已评价；没有因超时留下Joint NA。家族分母仍为20/316/336，未删不可估槽位。

结果解读和论文准备度见[完整交付与论文主线](../paper/FAIRBIAS_COMPLETION_MAIN_FINDINGS_20260918.md)。不能将新完成预算结果回填成原预算成功，也不能把当前单一固定配置视为每种方法最佳可能表现。

## 关键证据

工作目录：`artifacts/nhis/completion_evaluation_20260918/`。

| 文件 | SHA256 |
|---|---|
| control/admission_v1.json | 2a83612b23048a54ce09196ec553b9881ee87bf5b5930f36fc53bc0e52d1a628 |
| control/selection_v1.json | 46e58b19a02ea5144b9c8fe11459c38b25db2fe9e503129125114a7e7167e69d |
| control/study_v1.json | 3c4572ec327d8d97a2f108fe3c82a8619370a78981c8fe1abdb496677b2933fb |
| control/t_release_v1.json | 64d34e601996ddcda1bf687b33fd9db2f6e259f94a29ae0ada41b08bef9964ff |
| merged_summary_v1.json（导出v2） | 641fe2ec523c8998cfe4c1f48ae2d5fb1e6ecbe9a5eb9e20b16af33d47c6b406 |

最终机器可读证据：`control/supervisor_final_acceptance_v1.json`。失败的合并日志为`control/merge_v1.log`，成功导出为`control/merge_v2.log`。每臂目录内含独立manifest、summary、聚合统计、复制的新模型预测及共同PSU重抽样指标；这些内部个体预测不进入公开论文表图。

服务器可以保持关闭；无在途训练或本轮评价任务。自动任务保持暂停。论文图表/人群描述和文稿继续在本地完成，不需要重新租卡。
