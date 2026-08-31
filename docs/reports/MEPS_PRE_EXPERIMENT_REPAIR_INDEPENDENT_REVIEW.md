# MEPS Pre-Experiment Repair — Independent Read-Only Review (Decision 0006 Preparation Gate)

**Review date:** 2026-08-30
**Reviewer role:** Independent read-only review agent (pre-Codex evidence preparation; this report is not an ACCEPT and grants no commit authorization)
**Reviewed HEAD:** `e5e11f5e2621253bebe8309038a57c6c13cdc9aa`
**Branch:** `research/meps-hc252-longitudinal`
**Revision:** 2026-08-30，依据 Codex REPAIR 判定的最小修复指令仅修改本报告文件本身：状态计数修正为 38 个未跟踪条目 / 展开 44 个实际文件（新增项为本报告自身）；Unresolved issues 首位加入提交依赖闭包阻塞项；Proposed next step 改为依赖闭包审计；并澄清 `READY_FOR_CODEX_VERIFICATION` 的语义。未修改任何代码或配置。

---

Gate: MEPS Decision 0006 只读复核准备（独立复核，非实现、非提交授权）

Status: `READY_FOR_CODEX_VERIFICATION` — 该状态仅表示代码语义复核通过，不表示当前提交集合可接受。这不等于 ACCEPT，也不构成提交授权。全部 9 项验证内容与 preflight 一致，未发现语义性阻塞差异；但提交依赖闭包尚未审计（见 Unresolved issues 第 1 项），在确定可独立检出的最小完整提交集合之前，不得授权任何提交。

Files changed: None — read-only review. 未修改、未暂存、未提交、未 push 任何仓库文件。未读取任何 MEPS 微数据、未运行训练或 pipeline、未访问 Panel 27 任何结果。（注：本报告文件系复核完成后经项目负责人明确授权写入的唯一新增文件，写入时刻晚于复核判定，不影响复核期间的只读状态；该新增项即当前相对复核期间多出的唯一未跟踪条目，使未跟踪状态条目由 37 增至 38、展开后的实际文件数由 43 增至 44。）

Commands executed:
1. `pwd && git branch --show-current && git rev-parse HEAD && git status --porcelain | wc -l`
2. `git status --porcelain`（全部 44 条目逐条核对）
3. `git diff --cached --stat` / `git diff --cached --name-only | wc -l`（暂存区核验）
4. `git status --porcelain -uall | grep '^??'`（未跟踪展开计数）
5. `git diff inherited-code-v0.3-baseline-20260828 HEAD --stat --`（14 冻结根文件 + `.gitignore` 漂移核验）
6. Grep `136` 于 `src/meps_fairness/` 与 `tests/`
7. 只读检查全部 11 个 gate 文件及 `configs/cohort_and_variables.json`
8. `.venv/bin/python -c`（变量字典结构交叉核验：组名、连续/类别计数、年龄与月度变量）
9. `git diff --stat/-- <file>`（三个非本 gate 文件改动性质核验）
10. `.venv/bin/python -m unittest tests.test_meps_pre_experiment_repair -v`
11. `.venv/bin/python -m unittest tests.test_gate7_harmonization.TestGate7Harmonization.<5 个非数据测试> -v`
12. `.venv/bin/python -m json.tool configs/study.json >/dev/null` 与 `configs/cohort_and_variables.json >/dev/null`
13. `.venv/bin/python -m py_compile`（5 个指定文件）
14. `git diff --check`
15. `shasum -a 256`（12 个复核文件）
16. 复核后 `git status --porcelain | wc -l`、`git diff --cached --name-only | wc -l`、`git log -1 --format='%H'`（状态未变确认）

未执行任何被禁命令：未运行完整测试套件、未运行 `scripts/run_meps_pipeline.py` 或 `run_pipeline`、无训练、无 bootstrap、无真实数据测试、无下载/解压/数据扫描。

Permissions requested: 无。全部命令为只读或仓库 `.venv` 内本地测试；未使用网络（本轮未访问 AHRQ 或任何外部页面）。

Tests executed:
- `tests.test_meps_pre_experiment_repair`（10 项非数据测试）
- 5 项 Gate 7 非数据/合成测试（JSON 有效性、共享模式成员、合成队列与目标推导、非法随访码失败关闭、时间契约与泄漏拒绝）
- 两个配置文件的 `json.tool` 校验、5 个文件的 `py_compile`、`git diff --check`

Exact test results:
- Pre-experiment repair 套件：`Ran 10 tests in 0.001s` / `OK`（全部 10 项逐一 `ok`）
- Gate 7 选定套件：`Ran 5 tests in 0.016s` / `OK`（全部 5 项逐一 `ok`）
- `configs/study.json`、`configs/cohort_and_variables.json`：JSON 有效（exit 0）
- `py_compile`：5 个文件全部编译通过
- `git diff --check`：无空白错误（exit 0）

Input hashes:
- 起始 HEAD：`e5e11f5e2621253bebe8309038a57c6c13cdc9aa`（与要求一致）
- 基线标签：`inherited-code-v0.3-baseline-20260828`（冻结文件对比输出为空，零漂移）
- `configs/cohort_and_variables.json`：`53e7c22c45a02b8d6787eb680a295d95283cd50a8479f7ca4bef5719e7754c65`（与修复报告声称值一致）
- 未打开、未哈希任何 MEPS 微数据。

Output hashes（复核文件当前 SHA-256，独立计算；标 ★ 者与修复报告声称值逐一复现一致）:
- `configs/study.json` ★ `e0201919abc52eb884aa5c39dc6fc1edd73fd8ce4145f9396c35add266865797`
- `docs/reports/GATE_3_PROTOCOL.md` `047c6339da8929fad668c8db406bc1845f4a387556d0449b95b78eaff3fb2d3d`
- `docs/research/RESEARCH_PROTOCOL.md` ★ `39d0a979263543f70e14199cd6f4dd1d13f93a570b5e734fb3e031ef8d03e398`
- `docs/research/STATISTICAL_ANALYSIS_PLAN.md` ★ `afbfeac1ee0a701c8d8aa4d9d0556d75cafbbddb0e2ec82c52916754f1602259`
- `docs/decisions/0006-meps-pre-experiment-repair-gate.md` `3580baa854a0aed6254bcb9ee73b8933fb5766c39efa6b91ae100dcf86f4588c`
- `docs/reports/MEPS_PRE_EXPERIMENT_REPAIR.md` `6a3f2eafc39382bfeee918a18a1c3dda15920dee9e11de16d997c74e69d6adeb`
- `src/meps_fairness/pipeline.py` ★ `b9d9c782506bcdf670841dd4db793585400239ab2de9d5d6cbca56c225724239`
- `tests/test_gate10_models.py` `8e8d80de51e44b1edc08f82e83623e2b8e5de402fbc228b81142e0dadb97e337`
- `tests/test_gate11_smoke.py` `18e4a5189eaf198058396e2636ba62d78d3951f703990662d234355f51ebc39e`
- `tests/test_gate12_evaluation.py` `d6a4955e65ab6409e6496136f1342aa990e9e1a7718a0bae724ebccd30cb15ba`
- `tests/test_meps_pre_experiment_repair.py` ★ `fcda07c7d73c8cbbf39e7edc1c1ea710b06f6180e2030aa427aef74f52738b2c`
- `configs/cohort_and_variables.json` ★ `53e7c22c45a02b8d6787eb680a295d95283cd50a8479f7ca4bef5719e7754c65`（作为输入字典）

（修复报告因自指而未嵌入自身哈希的 Decision 0006 与报告本身，此处独立记录如上。）

Row counts: Not applicable — 未读取、未生成任何 MEPS 数据行、结局值、分布或指标。

Assumptions:
- Preflight 实测：目录 `/Users/lkc/Downloads/code_v_0_3`；分支 `research/meps-hc252-longitudinal`；HEAD `e5e11f5e2621253bebe8309038a57c6c13cdc9aa`；暂存区 0 项；复核期间 `git status --porcelain` 共 44 条目 = 7 个已修改跟踪文件 + 37 个未跟踪状态条目，`-uall` 展开后恰为 43 个实际未跟踪文件（与要求口径一致）；14 冻结根文件与 `.gitignore` 相对基线标签零漂移，且工作树修改清单不含任何冻结文件。**当前计数修正**：本报告文件写入后，未跟踪状态条目为 **38**，展开后实际文件为 **44**；唯一新增项即本报告文件自身（`docs/reports/MEPS_PRE_EXPERIMENT_REPAIR_INDEPENDENT_REVIEW.md`），故 `git status --porcelain` 总条目现为 45。
- 九项验证内容逐一确认：(1) `study.json` 年龄（AGEY1X 18–64）、12 个基线月份、12 个随访月份、74 个预测变量（19 连续 + 55 类别）与保护属性映射均非空，且与 `cohort_and_variables.json` 及 `cohort.py` 常量机械一致（手动交叉核验 + 测试机械断言）；(2) 方法臂仅表述为 `exploratory_survey_weighted_group_aware_centering_requires_protected_attribute_at_inference`，`scientific_role` 明示 `not_fairbias_or_tang`，推理期保护属性要求显式（`study.json`、SAP §4.2、RESEARCH_PROTOCOL §5、pipeline.py 模型配置）；(3) `src/meps_fairness/` 全文无 `136`，本 gate 测试亦不以其为功效逻辑或运行时预期状态，且有守护测试；(4) 功效门控由运行时事件数 + 命名阈值 `MIN_DEVELOPMENT_POSITIVE_EVENTS = 200` 生成；(5) 合成输入低于（7<11）、等于（200）、高于（999）阈值时 `locked_panel27_status` 均保持锁定，无 `UNLOCKED` 路径，且清单 `prerequisites_satisfied` 恒为 `False`；(6) 同一分区 Platt 拟合后的诊断在 `study.json`、pipeline.py（`APPARENT_CALIBRATION_EVIDENCE_NATURE`）、SAP §2.1、RESEARCH_PROTOCOL §4.1 均标为 apparent calibration-fit，非独立验证；(7) HC-217/225/234 为 `locked_candidate_development_panel_metadata_only`，`outcome_access_in_this_gate: prohibited`，无下载/合并授权，HC-244 为既有观察开发面板；(8) Panel 25 疫情期可比性风险保留于 `study.json` comparability_note、SAP §1.1、RESEARCH_PROTOCOL §4.1.1 与 Decision 0006；多面板 pooling、权重归一、年份重叠均需另开统计设计门；(9) 修复报告 Section 9 十五个标题完整并以 `STOP — waiting for Codex review.` 结束。
- 三个非本 gate 改动（`data/__init__.py`、`data/download.py`、`test_download_meps.py`）经 diff 核验为归档成员扩展名白名单/下载安全的既有工作，未混入功效、命名、校准或面板政策内容，修复报告亦声明保留不改。
- 复核后工作树状态复验不变（复核时点 44 条目、暂存区 0、HEAD 不变，当前计数修正见第一条）；`py_compile` 产生的 `__pycache__` 被 `.gitignore` 忽略，未新增状态条目。

Unresolved issues（第 1 项为阻塞项；其余为观察项，供 Codex 裁量）:
1. **[阻塞] 提交依赖闭包尚未审计。** 报告原建议仅提交已复核的修复路径，但 `src/meps_fairness/pipeline.py` 与新增测试依赖至少 12 个尚未进入 HEAD、也未纳入本轮复核范围的文件：`configs/cohort_and_variables.json`；`src/meps_fairness/data/` 下 4 个（`cohort.py`、`prepare.py`、`preprocess.py`、`split.py`）；`src/meps_fairness/evaluation/` 下 4 个（`__init__.py`、`calibration.py`、`inference.py`、`metrics.py`）；`src/meps_fairness/models/` 下 3 个（`__init__.py`、`baseline.py`、`mitigation.py`）。此外，选定的 Gate 7 测试文件 `tests/test_gate7_harmonization.py` 与 schema snapshot 产物也尚未进入 HEAD。若按原建议精确提交，干净检出将无法导入 pipeline、无法运行新增测试，不能形成完整、可复现的提交。必须先完成依赖闭包审计，在确定可独立检出的最小完整提交集合之前，不授权任何提交。
2. `tests/test_gate7_harmonization.py:336-343`（`test_real_panel26_cohort_extraction_and_power`）将历史观察值 `136` 硬编码为真实数据集成测试的期望值。该文件属既有 Gate 7 产物，不在本 gate 变更或复核清单内，本轮未执行；其性质为对已提取数据的历史观察完整性断言，非管线功效逻辑或运行时状态文本，不触及 Decision 0006 对 in-scope 源码/测试的验收口径。是否在后续 gate 中改写为运行时计数比较，由 Codex 决定。
3. `tests/test_repairs_regression.py:66,77` 断言手稿与主张矩阵文档包含 `136`，属文档完整性守护（确保历史观察被如实记录），非功效逻辑；同为既有产物。
4. “拆分结构（split structure）需另开统计设计 gate”明示于修复报告 Unresolved issues（第 68 行），而 SAP §1.1 与 RESEARCH_PROTOCOL §4.1.1 的表述为变量协调、权重归一、年份重叠与疫情期可比性；实质覆盖，仅位置不同。
5. 本轮未使用网络，未重新核验 AHRQ 官方元数据页面；候选面板序列的官方出处记录于 Decision 0006 与 `study.json` 的 URL 字段，其可信度沿用修复时的证据。

Git diff summary: 只读复核，复核期间无新增改动；复核后经授权仅新增本报告文件一个未跟踪条目。现存差异为 4 个本 gate 修改的跟踪文件（`configs/study.json`、`docs/reports/GATE_3_PROTOCOL.md`、`docs/research/RESEARCH_PROTOCOL.md`、`docs/research/STATISTICAL_ANALYSIS_PLAN.md`）加 3 个既有非本 gate 修改（下载/架构安全相关），以及当前 38 个未跟踪状态条目（展开 44 个实际文件；复核期间为 37/43，唯一新增项为本报告文件自身）；暂存区为空；`git diff --check` 无空白错误；冻结基线零漂移。

Proposed next step: 由 Codex 下发 Gates 6–13 依赖闭包只读审计 gate，对 `pipeline.py` 与新增测试在干净检出下可导入、可运行所需的最小完整文件集合进行审计并圈定提交边界；在确定可独立检出的最小完整提交集合之前，不授权任何提交。下一步是依赖闭包审计，不是多面板数据访问，也不是模型训练。

STOP — waiting for Codex review.
