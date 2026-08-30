# FairBias Round 4.1 Repair：official 模式终止保证、模式更名与测试计数表述修复报告

> **Round 4.1 Repair-2 修订（2026-08-30 Codex 对本报告的 REPAIR 判定）**：本报告的 P1 更名（`official_code_unweighted_reimplementation`，"官方代码兼容模式"）经复审仍属过度表述——单调游标是对官方实现（每次重访从头重搜）的显式偏离，两份数据轨迹不变不能证明一般行为等价；模式已在 Repair-2 再次更名为 `official_code_derived_monotone_cursor_unweighted`（官方代码衍生·带终止安全扩展的变体，非官方代码行为等价声明）。同时修复：(P1) 类别属性失败被误记为幂流穷尽（已按数值/类别分别记录 scope 并补回归测试）；(P2) 本报告漏记了 14:45 执行生成的两份 engineering `max_iterations=3` 产物（已在下方 Output hashes 完整披露六份）。详见 [REWORK_FAIRBIAS_ROUND4_1_REPAIR_2.md](REWORK_FAIRBIAS_ROUND4_1_REPAIR_2.md)。本报告正文保留 Repair-1 时点的历史描述（含当时的模式名）。

Gate: REWORK round 4.1 REPAIR — 响应 Codex 监督者 2026-08-30 对提交 `b9cc9ca` 的 REPAIR 判定（P0/P1/P2 三项；不回滚 `b9cc9ca`，通过本修复轮解决；修复范围严格限定在 FairBias 模式、对应测试与两份报告，未触碰未提交 MEPS 文件）
Status: COMPLETE — P0 official 模式幂流单调游标已实现并钉上 3 项回归测试；P1 模式更名为 `official_code_unweighted_reimplementation`（官方代码兼容模式），全部"论文方法严格复现"表述已改写；P2 237 项测试已改称混合工作区回归套件，b9cc9ca 自身证据钉为干净导出 155 passed / 3 skipped。修复后双模式实证重跑：四个运行数字与 b9cc9ca 报告逐一一致（无变化）。

---

## 1. 监督者 REPAIR 判定逐项修复

### P0：official 模式不存在报告所称的终止保证

**诊断确认**：原 [mitigation.py](../../src/fairbias/mitigation.py) `_search_numerical` 每次重访数值属性时从幂流头部重新搜索，仅跳过当前幂次（`abs(power - current_power) < 1e-12`）；不记录已消费位置。监督者诊断复现的往返 `3 → 1/3 → 3` 成立：第三次重访时 current=1/3，从头搜索会再次接受 3。叠加 [pipeline.py](../../src/fairbias/pipeline.py) official 模式的无预算 `while True`，终止无保证。

**修复（单调游标方案）**：
- `FairBiasMitigation.__init__` 新增 `self._exponent_stream_cursors: Dict[str, int]`（每数值属性的幂流游标），并显式保存 `self.preserve_exponent_order`。
- `_search_numerical` 在 official 模式（`preserve_exponent_order=True`）下从持久化游标处开始搜索；**每个被搜索到的幂次位置（跳过/拒绝/接受均计入）立即消费、游标只向后推进**，任何位置对同一属性不会被第二次搜索。工程模式保持原 restart-from-head 行为（由 `max_iterations` 预算兜底），游标字典保持为空。
- **fail closed 记录**：official 模式下搜索失败时 `non_convergence` 记为 `search_scope="official_power_stream"`、含 `stream_positions_consumed`，reason 明示"幂流在单调游标下穷尽、无位置可重试、official 循环无迭代预算"；不施加任何变换。
- **终止论证**（替换原不成立的"幂流有限 ⇒ 终止"）：每个数值属性至多消费 1998 个幂流位置；类别合并链每次接受严格合成映射（类别数单调下降，受类别数上界约束）；"dropped" 属性离开排名（d_phi=0 且不在列内）；ε 球内不接受任何变换。故被接受的变换总数有界，无预算 `while True` 必然终止（ε 球、逐属性幂流穷尽或类别合并链穷尽）。

**回归测试**（[tests/test_fairbias_algorithm_modes.py](../../tests/test_fairbias_algorithm_modes.py) 新增 `TestOfficialModeMonotonePowerStreamCursor`，3 项）：
1. `test_revisited_attribute_never_reuses_or_oscillates_powers`：以短交错流 (3, 1/3, 5, 1/5) + 全接受 stub 直接构造监督者诊断情形，断言重访消费序列严格为 [3, 1/3, 5, 1/5]（修复前第三访会再次接受 3）、无幂次复用、第五访 fail closed（`search_scope="official_power_stream"`、`stream_positions_consumed=4`、游标越过流尾）。该测试在修复前代码上必失败。
2. `test_cursor_survives_rejected_candidates`：被拒绝的幂次位置同样被消费，后续重访不得重试。
3. `test_engineering_mode_keeps_legacy_restart_and_no_cursor`：工程模式保持升序 + restart-from-head（重访可复用幂次，正是官方无预算循环必须游标的原因），且不创建游标状态。

### P1：将官方代码行为过度表述为"论文方法严格复现"

**修复**：模式标识符整体更名 `official_unweighted_reproduction` → `official_code_unweighted_reimplementation`（官方代码兼容模式 / official-code compatibility mode），全部措辞改写为"官方代码行为的未加权重实现"，并显式声明：论文正文（第 5 页）规定以 elbow plot 确定 MDS 最优维数，固定 dim=2 是官方代码实现行为；**论文正文模式（elbow 维数选择）未在本代码库实现**（如需"论文正文模式"须另设独立模式，本轮未实现、未声称）。改动覆盖：
- [src/fairbias/config.py](../../src/fairbias/config.py)：`ALGORITHM_MODE_OFFICIAL` 常量值、模块级模式契约、`resolved()` 文档与全部错误消息（含"NOT the paper text's elbow-plot selection"）。
- [src/fairbias/pipeline.py](../../src/fairbias/pipeline.py)：模块/类文档字符串、`algorithm_mode_definition` payload 字符串、单终态键 `final_results_official_code_unweighted_reimplementation`、`final_states`、`best_selection_reason`、termination note（含游标语义）。
- [src/fairbias/bias_metric.py](../../src/fairbias/bias_metric.py)、[src/fairbias/mitigation.py](../../src/fairbias/mitigation.py)：固定维数短路与幂流搜索的注释/文档统一为官方代码兼容措辞。
- [scripts/run_fairbias_benchmark.py](../../scripts/run_fairbias_benchmark.py)、[scripts/empirical_rework_validation.py](../../scripts/empirical_rework_validation.py)：模式比较改用 `ALGORITHM_MODE_OFFICIAL` 常量（消除硬编码漂移），帮助文本与输出标签同步更名。
- 测试：模式契约断言全部更新，并新增负向断言——旧键 `final_results_official_unweighted_reproduction` 不得在 payload 复现。
- 两份报告（MD + Canvas）：§1/§3/表述边界/Assumptions 的"论文方法严格复现"全部改写（见 §3）。

### P2：237 项测试不能作为该提交的独立测试数

**修复**：[REWORK_FAIRBIAS_ROUND4_1_MODE_SEPARATION.md](REWORK_FAIRBIAS_ROUND4_1_MODE_SEPARATION.md) 与 Canvas 报告中，237 项计数一律改称**混合工作区回归套件**（工作区混有未提交 MEPS Gates 6–13 测试文件），并显式标注"不得作为 b9cc9ca 提交自身的全量测试证据"；b9cc9ca 自身的独立测试证据钉为干净导出 `Ran 155 tests / OK (skipped=3)`（监督者已独立复现）。涉及 Status 行、§4 模板（Tests executed / Exact test results / Commands executed）与 Canvas 统计卡及证据链表。

## 2. 修复后双模式实证重跑（seed=0；描述性观测）

14:45 的重跑共生成**六份**产物：脚本一次执行生成 4 份（official × 2 无预算 + engineering × 2，脚本默认 `max_iterations=3`），另以 `max_iterations=10` 单独补跑 engineering × 2；其中预算=10 的两份 engineering 与 official × 2 共四份构成与 round-4.1 报告可比的集合（下表），另两份 `max_iterations=3` 的 engineering 产物当时未列入本报告（Repair-2 已补录至 Output hashes）。四个可比运行的终止状态、轮数、ACC/EO 与 b9cc9ca 报告**逐一一致**——幂流单调游标未改变既有实证轨迹（COMPAS/Credit 中数值属性的重访本就顺流推进，游标只是把该性质由偶然变为保证）：

| 运行 | 模式 | 终止 | test ACC | test EO |
|---|---|---|---|---|
| COMPAS | official | converged=true · epsilon_reached · 4 轮（terminal_max_dphi=0.0009444） | 0.6397 | 0.0207 |
| Credit | official | converged=true · epsilon_reached · 11 轮（terminal_max_dphi=0.0001659） | 0.8153 | 0.0142 |
| COMPAS | engineering | converged=true · epsilon_reached · 4 轮 | 0.6397 | 0.0207 |
| Credit | engineering | converged=false · iteration_budget_exhausted · 10 轮（terminal_max_dphi=0.0002168） | 0.8133 | 0.0238 |

## 3. 报告修订（两份）

- **Markdown**（[REWORK_FAIRBIAS_ROUND4_1_MODE_SEPARATION.md](REWORK_FAIRBIAS_ROUND4_1_MODE_SEPARATION.md)）：顶部新增 Repair 修订说明块；§1 P1-2 行的"官方幂流有限且每属性幂次严格推进 ⇒ 终止有保证"改写为单调游标机制（并注明原说法不成立）；§3 模式名更名 + "表述边界"改写（复现的是官方代码行为，非论文正文方法）；§4 模板 Status/Commands/Tests/Exact results 按 P2 改口径；Assumptions 1/3 改写。
- **Canvas**（`~/.qoder/projects/-Users-lkc-Downloads-code-v-0-3/canvases/round41-mode-separation-completion-report.canvas.tsx`，仓库外文件，不入提交）：标题区新增 Repair 提示；237 统计卡改称混合工作区回归套件（tone=warning）；§二 模式卡片更名并注明 dim=2 为官方代码行为；§三 增加重跑不变说明；§五 证据链表区分混合套件与干净导出证据；§六 结论附 Repair 修订说明。

## 4. 标准报告模板

Gate: REWORK round 4.1 REPAIR（如上）

Status: COMPLETE（P0/P1/P2 全部落地；3 项新回归测试；双模式实证重跑数字与 b9cc9ca 一致；未提交任何修复，等待 Codex 重新审阅）

Files changed（相对 HEAD=`b9cc9ca`，全部在授权修复范围内）:
- `src/fairbias/mitigation.py` — P0 本体：`preserve_exponent_order` 持久化 + `_exponent_stream_cursors` 单调游标 + `_search_numerical` 游标式搜索 + official 失败语义（`search_scope="official_power_stream"`、`stream_positions_consumed`）
- `src/fairbias/config.py` — P1 本体：`ALGORITHM_MODE_OFFICIAL = "official_code_unweighted_reimplementation"`、模式契约/`resolved()`/错误消息改写（显式声明非论文正文方法、论文正文模式未实现）
- `src/fairbias/pipeline.py` — P1：文档/`algorithm_mode_definition`/单终态键/`final_states`/`best_selection_reason`/termination note 更名改写；P0：无预算循环注释改为游标终止论证
- `src/fairbias/bias_metric.py` — P1：固定维数短路注释改为官方代码兼容措辞
- `scripts/run_fairbias_benchmark.py`、`scripts/empirical_rework_validation.py` — P1：模式比较改用常量、帮助文本/输出标签更名
- `tests/test_fairbias_algorithm_modes.py` — P0 回归测试 3 项 + P1 更名断言 + 旧键负向断言
- `docs/reports/REWORK_FAIRBIAS_ROUND4_1_MODE_SEPARATION.md` — P0/P1/P2 报告修订
- `docs/reports/REWORK_FAIRBIAS_ROUND4_1_REPAIR.md` — 本报告（新增）
- （仓库外）Canvas 报告同步修订，不入提交

Commands executed:
- `PYTHONPATH=src .venv311/bin/python -m unittest tests.test_fairbias_algorithm_modes tests.test_fairbias_mitigation`（`Ran 30 tests` / `OK`）
- `PYTHONPATH=src .venv311/bin/python -m unittest tests.test_fairbias_pipeline tests.test_fairbias_mds_dimension_golden tests.test_fairbias_clean_checkout`（`Ran 14 tests` / `OK`）
- `PYTHONPATH=src .venv311/bin/python -m unittest discover -s tests -p "test_*.py"`（混合工作区回归套件：`Ran 240 tests in 65.032s` / `OK`，EXIT=0；含未提交 MEPS Gates 6–13 测试文件与 3 项新游标回归测试，非本修复提交自身的独立证据口径）
- 干净树验证：`git archive HEAD`（b9cc9ca）解压至 `runs/_repair_check/` → 叠加 7 个修复文件 → 干净树内 `PYTHONPATH=src … unittest discover -s tests -p "test_fairbias_*.py"`（`Ran 85 tests` / `OK (skipped=3)`；3 项 skip 为 Git checkout 专属测试在无 .git 目录下的预期跳过）
- 双模式实证重跑：`PYTHONPATH=src:scripts .venv311/bin/python scripts/empirical_rework_validation.py`（一次执行生成 official × 2 + engineering × 2 默认预算 3，共 4 份）+ engineering × 2（`max_iterations=10`）——合计六份产物（见 Output hashes；Repair-2 补录完整清单，Repair-2 已将脚本默认预算改为 10 使一次执行只生成预声明的四份）
- `shasum -a 256`（输入/输出取证）；`git diff --numstat` / 冻结文件零改动核验

Permissions requested: 无（无网络访问意图；注：混合工作区套件中既有未提交 MEPS 下载测试按其自身设计写入 stub 文件，非本轮行为）

Tests executed: 混合工作区回归套件（240 项）+ fairbias 定向（30 + 14 项）+ 干净树 fairbias 套件（85 项，b9cc9ca + 修复叠加）。

Exact test results:
- 混合工作区回归套件：`Ran 240 tests in 65.032s` / `OK`（b9cc9ca 时点为 237 项；净增 3 = 本轮游标回归测试）
- 定向（algorithm_modes + mitigation）：`Ran 30 tests` / `OK`；定向（pipeline + golden + clean_checkout）：`Ran 14 tests` / `OK`
- 干净树（b9cc9ca + 修复叠加）：`Ran 85 tests` / `OK (skipped=3)`

Input hashes:
- `data_COMPAS.csv` — `4b2bc1d55553f1c2061768bd245b725ee57b3f7e438f612b78c60f308848cb27`（与 round-4/4.1 一致）
- `data_Credit_Card.csv` — `303cf916663273a345671688c96e9be7a83c76e39f0a2727f8d0ed33b8d6df1e`（与 round-4/4.1 一致）
- fixture `tests/fixtures/official_uciadult/distance_matrix_step_0.csv` — `5f9fd4f81766ce724bd9866bd139503d252f68c9bc7e5065ff34ace0f57228e1`（与 PROVENANCE.json 一致）
- 代码状态：HEAD=`b9cc9ca`（未提交任何修复，等待重新审阅）

Output hashes（runs/rework_empirical/，唯一 run_id 不覆盖；Repair-2 补录完整六份——原报告仅列其中四份，漏记两份 `max_iterations=3` 的 engineering 产物）:
- official COMPAS `fairbias_compas_seed0_20260830_144501/results.json` — `d0870d2417c1425c88a694db6289dc62f17ec7e6c3317312e2d0a5acb7d7932c`
- official Credit `fairbias_credit_seed0_20260830_144509/results.json` — `ea6af7237480637fe18456e344b60fce5f5f2cfe8a87ee05723a07bbbdbde315`
- engineering COMPAS（预算 3，**Repair-2 补录**，非可比集合成员）`fairbias_compas_seed0_20260830_144511/results.json` — `740854e87c499ff9afefe1f2310b85e31f42655e414734035839ccca64201885`
- engineering Credit（预算 3，**Repair-2 补录**，非可比集合成员）`fairbias_credit_seed0_20260830_144517/results.json` — `c47b5b5370c3440dadfa555b3562409494f21eb4ba75fbc43eef29f6ea14ab42`
- engineering COMPAS（预算 10，可比集合）`fairbias_compas_seed0_20260830_144539/results.json` — `f0ad7f162ab4ccdef6fb30f53568d53e90fd14b639f0e82344e352d920f2a7b4`
- engineering Credit（预算 10，可比集合）`fairbias_credit_seed0_20260830_144555/results.json` — `da7a1908eb6183e8f29080b7aa67477317b4d347cf7bf655db990206463e704c`

（注：以上六份为 Repair-1 时点旧模式名 `official_code_unweighted_reimplementation` 的产物；Repair-2 更名后已在 REWORK_FAIRBIAS_ROUND4_1_REPAIR_2.md 重新生成并另行列哈希。）

Row counts（observed，与 round-4/4.1 一致）:
- COMPAS：train=3950 (0.6400) / validation=987 (0.1599) / test=1235 (0.2001)，总 6172
- Credit：train=19200 (0.6400) / validation=4800 (0.1600) / test=6000 (0.2000)，总 30000

Assumptions:
1. 单调游标语义：被搜索即消费（跳过/拒绝/接受均计入），保证"幂次不会复用"；这是对官方实现（每次从头重搜）的一处**显式偏离**，目的是为无预算循环提供终止保证——已在 config.py 模式契约与 mitigation.py 文档中如实记录。若监督者裁定该偏离不可接受，备选方案为状态循环检测 + fail closed。
2. official-code 兼容模式在 COMPAS/Credit 上的既有轨迹不受游标影响（重跑数字逐一不变）；该不变性是经验观测，非保证。
3. 论文正文模式（elbow plot 选 MDS 维数）本轮未实现、未声称；如需须另设独立模式。
4. engineering 模式行为与 b9cc9ca 完全一致（数字逐一复现，仅措辞/键名不变——本轮 engineering 路径代码零改动）。

Unresolved issues:
1. 数值编码类别（承 round-3/4.1）：MEPS 接入时必须由变量字典显式指定 numeric/categorical（归入 MEPS 接入前前置 gate）。
2. MEPS 侧 P1/P2（study.json arm 过度宣称、meps pipeline.py 硬编码 136）：归入 MEPS Gates 6–13 整理 gate，本轮未触碰。
3. MEPS Gates 6–13 未提交文件：保持原状，未纳入本轮。
4. survey-weighted 实现（Gate D）：仍为预声明。
5. 多 seed 稳健性运行仍属论文实验协议差距（如实列为未做）。

Git diff summary:
- 本轮修复（相对 HEAD=`b9cc9ca`，8 个已跟踪文件 + 1 个新报告）：`src/fairbias/{config,mitigation,pipeline,bias_metric}.py`（+186/−90）、`scripts/{run_fairbias_benchmark,empirical_rework_validation}.py`（+19/−14）、`tests/test_fairbias_algorithm_modes.py`（+224/−20）、`docs/reports/REWORK_FAIRBIAS_ROUND4_1_MODE_SEPARATION.md`（+14/−11）；新增 `docs/reports/REWORK_FAIRBIAS_ROUND4_1_REPAIR.md`。
- 14 个冻结基线根文件与 `.gitignore` 零改动（`git diff` 为空）。
- 未提交 MEPS 工作区文件（`src/meps_fairness/`、`tests/test_gate*.py` 等）未触碰；工作区中原有的 `src/meps_fairness/data/{__init__,download}.py`、`tests/test_download_meps.py` 修改为修复开始前已存在的未提交状态，不属于本轮 diff。
- 未执行任何 git stage/commit（等待 Codex 重新审阅通过）。

Proposed next step:
1. 监督者重新审阅：(a) P0 游标实现与 3 项回归测试（含监督者诊断 3→1/3→3 的直接复现）；(b) P1 更名与措辞（含旧键负向断言）；(c) P2 报告口径；(d) 干净树 85 项通过 + 双模式数字不变。
2. 审阅通过后授权提交本轮修复。
3. 之后再进入 MEPS Gates 6–13 整理 gate。

STOP — waiting for Codex review.
