# Gemini 补修指令：NHIS D8-R1B

制定者：Codex Supervisor。日期：2026-09-08。

**D8-R1 审查结论为 REPAIR，当前仅放行 R1B 补修，不得进入 R2。**

先阅读：

1. [执行协议](../AI_EXECUTION_PROTOCOL.md)。
2. [原 D8-R1 完整规范](NHIS_D8_ENHANCEMENT_REPAIR_PLAN_20260908.md)。
3. [Codex 独立审查及证据](../reports/NHIS_D8_R1_CODEX_REVIEW_20260908T062327Z_ef10b683.md)。

本文件补充原规范，不扩大真实数据、共享模块或 Git 权限。原计划的已知 `.gitignore` 差异处置继续有效。

## 1. 启动与证据留存

- 不再对旧 precheck 要求所有待修复源码与第一次版本一致。R1B 起点是 worker 的 `artifacts/nhis_d8_repair/20260908T061131Z_fcf9dc7a/post_repair_manifest.json`，以及 Codex `artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/integrity.json` 中的同一组源码指纹。
- 原 precheck 继续用于保护文件、6 个冻结共享模块和旧结果。出现新差异先上报，不擅自恢复。
- 新建唯一 `artifacts/nhis_d8_repair/<UTC>_<suffix>/` 保存 R1B 前后源码快照、增量 diff、完整命令、日志、分区／配置合同样例和 manifest。
- 不覆盖 R1 工人报告和 Codex 证据。若要更正旧声明，用新的 addendum 链接原报告。

## 2. 优先修复数据隔离和报告

1. 删除/替换 `test_backward_compatibility_ae_disabled` 对默认真实 COMPAS 路径的依赖。使用内存 fake loader 或在临时目录显式生成的合成 CSV，并把输出指向新的临时／artifact 目录。
2. 测试开始前启用真实数据读取拦截，覆盖 `data/`、旧 COMPAS/Credit 路径和 parquet；允许的合成文件必须有明确生成证据与精确临时路径。校验 guard 在实际读文件前触发；整个测试过程记录尝试次数。
3. 对旧 NHIS smoke 的启动/终止事件编写 incident addendum。只读已有执行输出，说明已知命令、版本、时间／任务标识（可获得时）、可能访问的分区、确证到达阶段及未知范围。没有证据时填 UNKNOWN，不填 0，也不为调查事件重启旧程序。
4. 更正 R1 的“全为合成”和“所有真实数据文件零打开”的无依据声明。必须区分 COMPAS 测试读取路径已被证实，与旧 NHIS 进程的具体访问量未确认。

## 3. 逐项落实 R1B-02 至 R1B-08

- 公平门禁必须显式启停，启用时完整检查所有保护维度与合法 active feature 集合，逐值检查有限且非负，并验证 epsilon/slack/min_gain 等参数；关闭时使用未评估状态，禁止假 0。
- 结构化失败从 evaluator 传递到 engine 和 runner；基准失败必须停止 arm。非预期异常不能变成无候选、循环或普通公平拒绝。
- 同一 estimator/scaler/encoding 合同用于 baseline、候选和最终评估；配置必须真正生效。可通过 `sklearn.base.clone`/显式 factory 实现，不修改被冻结的共享模块。缺所需概率接口明确失败。
- 所有标签相关候选规则只用 fit；验证 protected 索引、参数与显式 partition 一致性，指纹绑定实际数据、schema、角色与完整配置。防止可变 DataFrame 内容悄悄变化。
- 每步只有最终提交的候选标 accepted；其他有增益候选标 eligible-but-not-selected。状态链、预算计数和 BM/AE 事件可独立重放核对。
- CLI 只允许独占创建目录；初始化前写启动 manifest，任何失败都保留有边界的状态／原因；完成时记录输出哈希。不得将原有失败目录当成空目录重用。
- 删除所有基于 Mock 类型识别的生产分支。测试替身通过正常依赖注入，执行同一合同。
- 补全真实联合循环检测、预算与停止类型验证，不用 tracker 单元测试替代 runner 行为验证。

仍然只修改原 D8-R1 白名单。不得修改 `mitigation.py`、`bias_metric.py`、`transform.py`、`evaluator.py`、`models.py`、`config.py`；如确实需要共享模块改变，提交最小必要提案而不自行修改。不要引入新的优化策略、参数调优或真实数据运行。

## 4. 测试要求

把 Codex 独立 suite 的 13 个未通过反例纳入正式行为测试，保留已有通过案例，并补齐审查指出的覆盖空白。独立脚本位于：

`artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/adversarial_review.py`

这份脚本是行为证据，不冻结具体修复 API。若改为结构化失败而非异常，可调整断言接入处，但必须验证同一语义，不能弱化为“非空返回值”或“存在某个字段”。同时明确列出每项原失败与新测试的映射。

验收重点：

- 数据守卫下所有正式合成测试通过，真实数据读取尝试为 0；另有专门 guard 自检，其预期拦截次数单独报告。
- 缺保护维度、缺几何特征、任意位置 NaN、无限预算均不能放行。
- 基准失败和候选耗尽可区分；基准失败不会继续最终正常评估。
- DT 与 LR 等配置分别与独立 oracle 一致；scaler 的 fit 统计和预测前处理也实际检查。
- 改 selection 标签不会影响提案的 NMI／类别阳性率；它可以影响候选选择效用，这是既定合同。
- 两个候选都改善时，恰好一个 committed；拒绝、异常、映射冲突都有可解释事件。
- CLI 测试必须调用真正入口配 fake runner；覆盖初始化失败、运行失败、部分产物、已有空目录、已有失败 manifest 和正常唯一目录。
- 全局循环、状态缓存、指纹变化、JSON round-trip 和 AE=False 都有实际路径验证。

禁止将 2024 表格或“Joint 优于基线”设为通过条件。保存完整测试输出，不以口头摘要代替。

## 5. 交付与停止

交付新报告 `docs/reports/NHIS_D8_R1B_REPAIR_REPORT_<run_id>.md`，使用协议 Section 9 完整标题，逐项链接整改证据、incident addendum、合成测试日志、前后 manifest 和增量 diff。

状态使用 `IMPLEMENTED_SYNTHETIC_VERIFIED_PENDING_CODEX_REVIEW`，或准确的未完成／阻塞状态。旧 NHIS 读取范围若无法确认，应保留未决事实，而不是声称补修使历史暴露归零。Codex 将分别审查实现正确性与数据边界记录。

完成后停止，最后一行：

`STOP — waiting for Codex review.`
