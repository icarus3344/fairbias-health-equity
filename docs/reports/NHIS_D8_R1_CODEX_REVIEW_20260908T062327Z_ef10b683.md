# Codex 独立审查：NHIS D8-R1

Date: 2026-09-08
Reviewer: Codex Supervisor
Decision: **REPAIR — 当前提交不予验收，D8-R2 不放行。**

保留现有改动为待修复工作，不覆盖用户原代码或旧结果。本轮只增加独立审查证据和补修规范，没有修改产品源码、没有提交 Git、没有运行真实数据训练。

## 1. 审查对象与完整性

- HEAD：`b595e59b67d8781a3dca41199487909fdb9ed42d`，分支 `research/nhis-fairbias`。
- Worker 报告：[NHIS_D8_R1_REPAIR_REPORT_20260908T061131Z_fcf9dc7a.md](NHIS_D8_R1_REPAIR_REPORT_20260908T061131Z_fcf9dc7a.md)。
- 对照规范：[D8 执行方案](../plans/NHIS_D8_ENHANCEMENT_REPAIR_PLAN_20260908.md)。
- Worker 的 10 个输出文件均匹配其 post-repair manifest SHA；因此下述问题针对它实际交付的版本。
- 15 个保护文件均与 Codex 制定 gate 时的当前指纹一致；6 个冻结共享模块和旧 D8 汇总结果未改变。既有 `.gitignore` 基线差异仍存在，不能改写成完整基线一致。
- SHA 校验把保护文件作为不透明字节处理，不进行任何个人级数据解析。
- [完整校验记录](../../artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/integrity.json)。

## 2. 独立执行证据

### 2.1 原有测试加读取拦截

解释器：`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`。设置 `PYTHONPATH=src`、`PYTHONDONTWRITEBYTECODE=1`。

导入并运行 worker 报告中的四个具名测试模块，额外用 Python audit hook 在文件打开前拦截仓库 `data/`、继承 COMPAS/Credit 数据和所有 parquet 路径。

**28 项：27 通过，0 断言失败，1 因数据隔离而报错；拦截到 `data_COMPAS.csv` 的读取尝试。**

- 出错测试：`TestFairBiasEnhancementContracts.test_backward_compatibility_ae_disabled`。
- [测试日志](../../artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/worker_suite_guarded.log)。
- [机器可读结果](../../artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/worker_suite_guarded.json)。

这是对“28 项均为纯合成验证”声明的反证，不是声称 worker 的原始无拦截测试没有通过。

### 2.2 补充合同检查

新增独立、纯合成 review suite，不修改仓库正式测试。运行：

```text
PYTHONDONTWRITEBYTECODE=1 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/adversarial_review.py
```

**14 项：1 通过，12 断言失败，1 异常类型不符；没有真实数据读取尝试。**

失败断言描述的是 gate 要求的正确行为，不是为当前缺陷设定的“预期通过”。唯一通过的是删除列加幂变换后与独立算术／LR oracle 的一致性，支持原始 F1/F2 的核心变换修复有效。

- [独立复现脚本](../../artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/adversarial_review.py)。
- [详细结果及 traceback](../../artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/adversarial_review.json)。
- [运行日志](../../artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/adversarial_review.log)。

脚本输出采用独占创建，现有证据目录不允许直接覆盖式重跑。补修时可将测试逻辑移入授权测试文件，并在新的唯一目录保存输出。

## 3. 必须补修的发现

### R1B-01 / P1：数据隔离声明不成立，历史 NHIS 读取范围未核实

`tests/test_fairbias_enhancement_contracts.py:363–377` 使用默认 `compas_default()`，随后直接调用真实 pipeline，没有 fake loader，也没有合成数据路径。`config.py:369` 指向 `data_COMPAS.csv`，`pipeline.py:257–258` 调用真实 loader。独立拦截确认该测试尝试打开真实 COMPAS。

用户提供的完整执行记录第 192–197 行还显示：在修改旧 D8 测试前，worker 启动了 `python3 -m unittest tests/test_nhis_d8_enhancement.py`，随后 kill。启动快照中的该测试会构造真实 D8 runner，旧 runner 构造 adapter，adapter 读取整份 NHIS parquet。

**已证实的是旧命令被记录为启动，且该代码路径包含真实数据读取；现有材料不能确定进程到达了哪一步、读取了多少行、是否进行了拟合。**不得因 kill 就断言没有读取，也不得从静态路径推断已完成全部训练。Worker 报告的“0 files opened / 0 rows read”及遗漏该命令的命令清单必须更正。

补修：新增 append-only incident/addendum，保留原报告；记录确定事实与未知范围，利用已有 stdout/stderr/任务记录核实，不重跑旧命令。修复当前非合成测试，并对整个补修测试进程启用真实文件读取拦截。

### R1B-02 / P1：公平门禁仍可放行无效输入

源码位置：`src/fairbias/enhancement.py:129–136,149–195`。

独立复现四个问题：

1. 传入 epsilon、但 `O_train=None` 时返回 acceptable=True，dphi=0，且没有正确标记门禁关闭。
2. 活跃特征为 x,z，返回几何只含 x 时仍通过；没有核验完整保护维度×活跃特征集合。
3. 返回 `{'o': {'x': 0.001, 'z': NaN}}` 时，Python `max()` 可留下前面的有限值；只检查最大值不能检测每个条目的 NaN。
4. `max_fairness_degradation=inf` 被当作无限上限并通过，未验证全部预算。

补修：显式启用／关闭门禁；开启时逐项验证数据、完整键集合、索引、数值与预算。关闭状态应为“未评估”，不是零偏差。合法删除状态和非法缺维度必须区分。

### R1B-03 / P1：基准失败在外层又被吞成“候选耗尽”

源码位置：`src/fairbias/enhancement.py:332–334`，`src/nhis_fairbias/d8_enhancement_runner.py:421–423,555–557`。

底层 `evaluate_candidate_utility` 会返回 `SINGLE_CLASS_SELECTION` 等具名结果，但 `enhance_step` 丢弃该结果，正常返回 `(current_df, changed_dict, None)`。runner 把它写成 `candidate_exhausted`。因此最初的“错误伪装成没有改善”只是换了一种形式。

补修：定义贯穿 evaluator→engine→runner 的状态，基准无效立即终止该 arm，明确 `evaluation_failed`；候选程序错误不能进入普通搜索拒绝路径。独立单类 selection 测试目前未收到应有异常／失败状态。

### R1B-04 / P1：候选评分忽略分类器和缩放配置，仍没有唯一评估合同

源码位置：`src/fairbias/enhancement_contracts.py:238–260,272–280`，`src/nhis_fairbias/d8_enhancement_runner.py:107–126`。

候选默认强制 MinMaxScaler + LogisticRegression，只使用 evaluator 中的 seed。配置 DT、RF 或其他 scaler 时，候选实际上仍按 LR/min-max 排序；通用 pipeline 最终却用配置中的模型。

合成 DT oracle 的 AUROC 为 1.0，候选合同得到 0.499047619047619。该反例说明忽略配置会实质改变被优化的模型；当前 D8 恰好配置 LR/min-max 不等于合同可通用。

此外，模型缺 `predict_proba` 时回退到 hard predictions 仍返回 VALID，违反 gate 的明确失败要求。字符串类别还被候选 helper 隐式 ordinal 编码，而 D8 最终评估没有同样步骤，形成另一条不一致路径。

补修：复用明确的 estimator/scaler/encoder factory 或 clone 参数合同，将最终评分也接入同一入口。R1 不新增编码方法；缺少所需概率接口时显式失败，不能悄悄换成硬预测 AUROC。

### R1B-05 / P1：fit/selection 与“不可变分区”合同不完整

源码位置：`src/fairbias/enhancement.py:275–288,338–342`，`src/fairbias/enhancement_contracts.py:61–74`。

使用显式 partition 或内部切分时，特征 NMI 排名仍传入全部 `X_train,Y_train`。独立构造 120 行 fit / 40 行 selection 后，排名函数实际收到 160 个标签，违反候选生成只用 fit 标签的要求。

protected 数据仅检查长度，不检查 index；错位的保护属性在 partition 构造时通过。所谓 fingerprint 只包含 shape、列名和 index 首尾五项，改特征值、标签或中间索引可能完全不变。`frozen=True` 也没有冻结可变 DataFrame 内容。

补修：所有标签相关提案只读取 partition.fit；核验其他传入参数与 partition 的身份一致或取消重复数据入口；校验 protected 索引。分区指纹绑定实际内容、完整索引、schema 和角色，配置指纹覆盖实际模型／变换参数；采用快照副本或内容校验防止调用后变更。缓存生命周期必须绑定这些指纹。

### R1B-06 / P1：日志把“有增益”当成“已采用”，无法据此重建机制

源码位置：`src/fairbias/enhancement.py:566–595,805–834`。

代码在选 best candidate 之前就把每个正增益候选记录为 accepted=True。注入确定效用 .5→.6/.7 时，只提交一个变换，却有两条 accepted 记录。这会夸大接受步数，并把未走过的变换写入机制轨迹。

此外，BM 的 joint 事件只有属性和状态快照，没有完整前后效用／几何／接受状态；runner 的 geometry count 只使用 AE engine 的计数，遗漏 BM 搜索和外部多次重算。类别 mapping 异常的 `except: continue` 不留事件。配置 hash 仅覆盖 mode/seed/slack，不能绑定实际候选配置。

补修：区分 `eligible`、`selected`、`committed`；accepted 仅表示真正提交。每一步恰好零或一个提交事件，状态链可重放；计数口径分项定义并包含全部相关操作。不要为降低成本把未计算的指标伪装为 0。

### R1B-07 / P1：CLI 输出保护与失败 manifest 未完成

源码位置：`scripts/run_nhis_enhancement_study.py:83–99,115–127,153–172`。

现有目录只要没有两个指定结果文件就会被重新使用，即使已有 manifest/audit 或上次失败证据。独立调用真实 `main()`、注入 fake runner 后，已有目录没有抛 FileExistsError，而是进入运行并到达注入停止点。

新目录运行途中异常后不存在 manifest，因为 manifest 仅在全部运行与输出成功后创建。开始时没有配置／输入指纹，失败也没有状态记录。

`tests/test_nhis_d8_synthetic_contracts.py:99–114` 没有调用 CLI，而是在测试里复制 if 条件并自行抛异常，无法验证产品代码。

补修：独占创建输出目录；在初始化 runner 前持久化启动 manifest；成功／失败明确终结。用真实 CLI 入口配 fake runner 检验现有目录、部分结果、构造器错误、运行中错误、写出错误和两次默认运行，不读取真实数据。

### R1B-08 / P1：生产逻辑检测 Mock，削弱测试对实际路径的约束

源码位置：`src/fairbias/enhancement.py:300–330,514–539,754–778`。

生产代码检查 `_mock_wraps`、`assert_called` 和类型名中的 Mock，测试时走另一套 utility 分支。该分支再次把已变换训练集和原始 selection 配对；测试所覆盖的行为与非 mock 生产路径不同。

补修：删除生产代码的 Mock 识别分支；使用显式可注入的同一个评估接口，真实模型与测试替身走同一调用合同。不能通过让生产代码识别测试来维持旧成功测试。

## 4. 尚不能视为已完成的附加验收项

- 现有 cycle 测试仅测试 tracker，不证明 BM+AE 全局状态循环被正确报告；runner 没有 `cycle_detected` 外层终止分支。
- scaler 隔离测试主要断言 helper 返回 VALID，没有检查 scaler 统计；oracle 使用共享 transform 实现，独立性不足。本轮独立算术 oracle 只覆盖了一个数值加删除例子。
- 没有持久化 worker 的完整前后测试 stdout/stderr；源码 manifest 不等同于运行证据。
- default tests 没有全进程真实数据访问哨兵；“显式 fake adapter”只能减少风险，不能证明没有其他路径访问文件。
- R1 规定的 budget 合同字段、不可用指标原因、skip-dropped 明确事件和逐轮 BM/AE 完整记录仍需按原规范逐项核验。

## 5. 下一步与放行范围

执行 [D8-R1B 补修指令](../plans/NHIS_D8_R1B_REPAIR_20260908.md)。只放行原白名单内修复、合成测试、证据补齐和诚实更正报告；不放行真实 NHIS/MEPS/COMPAS/Credit 拟合，不允许重新运行旧 smoke 命令，不放行 R2、提交或发布。

当前结论不否定全部已有工作：同步变换、删除列核心修复、若干映射与状态组件可以保留。**但 28 passed 不能替代完整合同验收。**
