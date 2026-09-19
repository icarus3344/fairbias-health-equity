# A0–A2 gate report v3

Gate：`A0-A2_NHIS_BIAS_ATTRIBUTE_AUDIT_V3`

日期：2026-09-19（Asia/Shanghai）

总判定：**A0 PASS WITH LIMITS；A1 STRUCTURE ACCEPTED / SCIENTIFIC FREEZE REPAIR REQUIRED；A2 ACCEPTED FOR SYNTHETIC CONTRACTS；A3 NOT AUTHORIZED；2025 LOCKED**。

该判定只验收计划、静态绑定、接口合同与合成测试，不声称发现任何新的 NHIS 公平差异，也不将历史 2024 结果重新包装为未见验证。

## Gate

`A0-A2_NHIS_BIAS_ATTRIBUTE_AUDIT_V3`

## Status

- A0 资产定位、哈希和 2024 官方元数据：`PASS_WITH_LIMITS`。
- A1 anchor/expanded、逐 O 域、固定 panel 与统计 family 接口：`STRUCTURE_ACCEPTED_SCIENTIFIC_CONTENT_NOT_FROZEN`。
- A2 隔离合同与合成验证：`ACCEPTED_FOR_SYNTHETIC_CONTRACTS`。
- A3 真实性能扫描：`BLOCKED_NOT_AUTHORIZED`。
- 2025 微观结局/性能：`LOCKED_NOT_READ`。

A1 未完成科学冻结的原因不是软件失败，而是 delta、正式候选 O/分组/参照、少量确认性方法对及 simultaneous CI 生产方案仍缺少规定的领域/统计审批。按预注册纪律，这些缺口必须在结果前解决。

## Files changed

全部为 additive v3 文件；v2 和历史资产未覆盖：

- `RESEARCH_PLAN_V3.md`
- `IMPLEMENTATION_HANDOFF_V3.md`
- `A0_ASSET_INVENTORY_V3.md`
- `VARIABLE_ATLAS_V3_DRAFT.csv`
- `PRE_FREEZE_REGISTRY_V3_DRAFT.json`
- `DECISION_LOG_V3.md`
- `A1_INTERFACE_AND_FAMILY_SPEC_V3.md`
- `A0_A2_GATE_REPORT_V3.md`
- `planning_verification_v3.json`
- `src/nhis_fairbias/bias_attribute_audit_v3.py`
- `tests/synthetic/test_bias_attribute_audit_v3.py`
- `tests/synthetic/test_bias_attribute_audit_v3_registry.py`

当前分支为 `research/nhis-fairbias`，检查时 HEAD 为 `67e6659fa65249a8842e34af5d8969629efe4bca`。工作树在本 gate 前已很脏；未清理、reset、提交或修改用户既有变更。受保护根文件检查未发现本 gate diff。

## Commands executed

关键验收命令如下；其他命令仅为只读定位、页面渲染和哈希检查：

```text
python3 -m py_compile src/nhis_fairbias/bias_attribute_audit_v3.py tests/synthetic/test_bias_attribute_audit_v3.py tests/synthetic/test_bias_attribute_audit_v3_registry.py
jq empty docs/plans/nhis_bias_attribute_discovery_20260918/PRE_FREEZE_REGISTRY_V3_DRAFT.json
PYTHONPATH=src pytest -q tests/synthetic/test_bias_attribute_audit_v3.py tests/synthetic/test_bias_attribute_audit_v3_registry.py tests/benchmark/test_survey_domain_contract.py tests/benchmark/test_survey_linearization.py tests/benchmark/test_survey_sufficient_totals.py tests/test_nhis_survey.py
PYTHONPATH=src pytest -q tests/benchmark/test_catalog_inference.py tests/benchmark/test_catalog_inference_supervisor.py
git diff --check -- <v3 additive paths>
```

有一次组合测试命令误列了不存在的 `tests/benchmark/test_record_alignment.py`，pytest 因路径不存在而未收集测试；随后使用上面的实际测试集合完整重跑，不能把该次调用计为通过。

## Permissions requested

无。全程本地；未联网下载、未启动服务器、未使用收费服务、未上传资料。

## Tests executed

A2 覆盖以下合同：

- 严格 SHA-256 与 exact-order 对齐；重排/重复/缺失 fail closed；
- 每个 O 自己的 panel 共同域；完整调查设计保留零贡献 PSU；
- 主 panel hash 冻结以及 supplemental panel parent 约束；
- Q1 `d=0` 与 Q3/Q4 `d_method-d_baseline=0` 的 family/contrast 分离；
- Holm p 值与 ordinary/simultaneous CI 分离，NA slot 保留在分母；
- 最大 gap 从完整预声明组件 simultaneous intervals 投影；
- delta 三态、precision/detectable effect 与四类绝对性能权衡；
- 2025 全条件 fail-closed access lock；
- registry/atlas/source hash/main-panel 静态绑定。

## Exact test results

- 主验收集合：**40 passed, 3 warnings in 1.70s**。
- 3 个 warning 均来自既有 `src/nhis_fairbias/benchmark/survey_batch.py:60` 的矩阵乘法（divide by zero / overflow / invalid）；对应测试通过，本 gate 未把它们静默标成新代码无警告。
- optional catalog probe：**collection failed**，因为当前解释器未安装 `aif360`；失败发生在旧 adapter import，未执行 v3 代码。另一次更早 probe 的 `test_catalog_domain_types.py` 和 `test_frozen_evaluation_contract.py` 也因同一可选依赖在 collection 阶段停止。A0–A2 没有授权安装或改写旧 adapter，因此没有扩大环境。
- Python compilation：通过。
- registry JSON：通过严格解析。
- atlas：31 行、31 个唯一变量；正式候选种子 24，atlas-only 扩展 7。

## Input hashes

完整输入表见 `A0_ASSET_INVENTORY_V3.md`。关键绑定包括：

- protocol：`1a3d38929d65493f8352d6b46d32f65fa836502d08a3c2d2704dd713e5c3beac`
- v2 research plan：`1a90183fe1b3248be86bdbce3b4461443c77c6f13815282033e5de66e9f31152`
- v2 handoff：`bb6aeac0d81419fdf58a5be716589959de5a9d95b1a8cd00f4c6446239ae3600`
- 2024 official codebook：`04ef4aa4b86c1b341cfdbae4389d9d7726584c92e579e4acac98b7c240fda3c9`
- completion admission：`2a83612b23048a54ce09196ec553b9881ee87bf5b5930f36fc53bc0e52d1a628`
- main panel canonical hash：`19ae854b2dc986aa42dca555814a5aae4ff8fb063a18ab9f65bae66876773c7f`

## Output hashes

机器可读清单见 `planning_verification_v3.json`。该清单不自哈希，避免自引用。

## Row counts

- A0–A2 处理的真实微观数据行：**0**。
- 新真实性能记录：**0**。
- 2025 读取记录：**0**。
- atlas：31 条（24 正式种子 + 7 atlas-only）；正式 registry 保持 24，未超过 30 的上限。
- main risk panel：5 个方法成员，每个 5 个已绑定模型 ID；尚未生成新的 2023 S predictions。

## Assumptions

- 2024 是已知结果的回顾性年度，不是未见验证。
- `p_event` 风险输出与固定阈值策略构成主 panel 可比部署合同；不同输出或需要 O-at-inference 的方法进入 supplemental panel。
- A2 max-standardized joint-replicate simultaneous CI 仅为候选实现；未完成参考/覆盖验证前不用于真实确认性结论。
- 现有最小 n/事件/ESS/PSU 规则只表示可估性，不表示 precision 足够。

## Unresolved issues

需要负责人集中给出、但未阻止已完成的软件工作：

1. 应用场景、FNR/FPR 等确认性端点的实际代价、delta、单位、领域依据、批准者和批准时间；
2. 允许进入正式候选的 O、分组/参照、嵌套/相关处理，以及 clinical/context/process 切片能支持何种主张；
3. 少量确认性 method-baseline 对，其余方法迁移矩阵保持描述性；
4. simultaneous CI 的参考实现/模拟覆盖接受标准；
5. 2025 权限、一次性 release 责任人，以及伦理、作者、AI 披露、署名/资助/冲突等人类事实。

技术缺口：completion 备份中没有可复用的 2023 `*_predictions_S.npz`。在 A1 科学冻结后，A3 还需要单独版本化的 2023 replay manifest、模型/源码/预测/order hash 和对齐验收。

## Git diff summary

本 gate 只添加上列 v3 文档、隔离模块与合成测试；未修改 v2、旧 runner、原始/处理数据、模型、预测或历史结果。output-hash 清单创建后，对全部 12 个未跟踪 v3 文件逐一执行等价的 `git diff --no-index --check /dev/null <file>`；修复报告中的 Markdown 行尾空格和两个 Python 文件的 EOF 空行后通过。

## Proposed next step

先补齐 `DECISION_LOG_V3.md` 中集中的人类科学决定，同时完成 2022/2023 候选元数据 harmonization、明确 family manifest 和 simultaneous CI 参考/覆盖 QA。只有新的 gate receipt 显示 A1 科学冻结且 A0–A2 总体接受，才可生成并执行 A3 的 2023 replay/真实性能扫描。A4 仍需等 A3.5 的 Methods 草稿、O 流程图、第一版结果图和 claim-evidence 表后另行决定；2025 继续锁定。

A3 NOT AUTHORIZED；2025 LOCKED。

STOP — waiting for Codex review.
