# MEPS Pre-Experiment Repair Gate Report

Gate: MEPS pre-experiment repair — Decision 0006; reconcile verified variable mappings, honest method-arm naming, data-derived power gating, candidate development-panel metadata, and calibration evidence boundaries without running an experiment.

Status: IMPLEMENTED_PENDING_INDEPENDENT_REVIEW — no training, no pipeline execution, no MEPS microdata read/download, no Panel 27 outcome access, no staging, no commit, and no push. This report is implementation/self-verification evidence only and is not an independent Codex acceptance.

Files changed:
- `configs/study.json` — Gate 7 mappings filled; honest implemented arm; apparent-fit calibration contract; dynamic power contract; fixed metadata-only candidate development sequence HC-217/225/234/244.
- `docs/research/RESEARCH_PROTOCOL.md` — verified mapping status, development-panel boundary, honest method and calibration labels.
- `docs/research/STATISTICAL_ANALYSIS_PLAN.md` — same methodological reconciliation plus explicit no-pooling boundary.
- `docs/reports/GATE_3_PROTOCOL.md` — historical arm labels marked superseded; no retroactive experiment authority.
- `docs/decisions/0006-meps-pre-experiment-repair-gate.md` — gate specification, AHRQ-only network scope, acceptance criteria and official metadata evidence.
- `src/meps_fairness/pipeline.py` — runtime power record and lock-status helpers; literal observed event count removed; calibration and model configuration labels made explicit.
- `tests/test_gate10_models.py` — comment uses honest method name.
- `tests/test_gate11_smoke.py` — runtime power assertions no longer expect a literal event count or a threshold-specific field name.
- `tests/test_gate12_evaluation.py` — same dynamic power assertion repair.
- `tests/test_meps_pre_experiment_repair.py` — new non-data configuration/power/lock regression suite.
- Outside repository: `~/.qoder-cn/projects/-Users-lkc-Downloads-code-v-0-3/memory/development-practice-specification-63ee77fc.md` — Qoder rule: private memory cannot replace gate, authorization, Decision/Section 9, or Git evidence.

Commands executed:
- Read-only Git/status/diff inspection and exact source/config/report inspection.
- Read-only official AHRQ metadata search restricted to `meps.ahrq.gov`; no archive or microdata download.
- JSON parsing/validation for `configs/study.json` and `configs/cohort_and_variables.json`.
- Python bytecode compilation for the changed Python source/tests.
- Targeted non-data unit tests listed below.
- One initial invocation with the system `python3` stopped during test import because that interpreter lacks `numpy`; no test body ran. The same scope was then rerun with the repository `.venv/bin/python` interpreter.
- `git diff --check`, frozen baseline comparison, SHA-256 hashing of configuration/code/report inputs and outputs.

Permissions requested: No additional system permission. Project owner explicitly authorized continuing implementation in the supervising conversation. Decision 0006 limited network use to official AHRQ metadata pages and prohibited data downloads.

Tests executed:
- `tests.test_meps_pre_experiment_repair` — 10 pure configuration/power/lock tests; no data or model fitting.
- Five selected Gate 7 tests: JSON structure, shared schema membership, synthetic cohort derivation, invalid-code fail-closed behavior, and temporal leakage rejection.
- JSON syntax validation and Python compilation checks.

Exact test results:
- Pre-experiment repair suite: `Ran 10 tests in 0.003s` / `OK`.
- Selected Gate 7 non-data/synthetic suite: `Ran 5 tests in 0.015s` / `OK`.
- Initial system-interpreter attempt: import error `ModuleNotFoundError: No module named 'numpy'`; superseded by the successful repository-environment runs above.
- `configs/study.json`: valid JSON.
- Changed Python files: compile successfully.
- `git diff --check`: no whitespace errors.
- Full MEPS suite deliberately not executed because real-data integration tests call `run_pipeline` and train models, which this gate prohibits.

Input hashes:
- `configs/cohort_and_variables.json` — `53e7c22c45a02b8d6787eb680a295d95283cd50a8479f7ca4bef5719e7754c65`.
- Starting repository HEAD — `e5e11f5e2621253bebe8309038a57c6c13cdc9aa`.
- No MEPS microdata input was opened or hashed in this gate.

Output hashes:
- `configs/study.json` — `e0201919abc52eb884aa5c39dc6fc1edd73fd8ce4145f9396c35add266865797`.
- `docs/research/RESEARCH_PROTOCOL.md` — `39d0a979263543f70e14199cd6f4dd1d13f93a570b5e734fb3e031ef8d03e398`.
- `docs/research/STATISTICAL_ANALYSIS_PLAN.md` — `afbfeac1ee0a701c8d8aa4d9d0556d75cafbbddb0e2ec82c52916754f1602259`.
- `src/meps_fairness/pipeline.py` — `b9d9c782506bcdf670841dd4db793585400239ab2de9d5d6cbca56c225724239`.
- `tests/test_meps_pre_experiment_repair.py` — `fcda07c7d73c8cbbf39e7edc1c1ea710b06f6180e2030aa427aef74f52738b2c`.
- Qoder private rule — `13dc0750f120905fd7d328d08b4b72f466270111e95dd36cbf0325a3619f58b3`.
- Decision 0006 and this report are self-referential gate artifacts; their hashes are intentionally not embedded in themselves.

Row counts: Not applicable. No MEPS data rows, outcome values, distributions, or metrics were read or generated. Previously reported Panel 26 counts remain historical observations, not outputs of this gate. Panel 27 remains locked.

Assumptions:
- HC-217/225/234/244 are frozen only as the official two-year longitudinal candidate-development sequence preceding HC-252; this does not authorize download, outcome inspection, pooling, model fitting, or estimand change.
- AHRQ's Panel 25 pandemic-era response/comparability warning must be addressed before any combined-panel analysis.
- The implemented group-aware centering heuristic remains exploratory and requires protected-group information at inference; it is not FairBias or Tang et al.
- Same-partition post-Platt metrics are apparent calibration-fit diagnostics only.

Unresolved issues:
- A separate statistical-design gate must decide whether and how to use multiple development panels, including common-variable harmonization, survey-weight normalization, overlapping years, pandemic heterogeneity, and split structure. No naive pooling is allowed.
- The current executable pipeline remains operationally Panel-26-specific (`split_panel26_duid_grouped` and Panel-26 result field names). Generalization is deferred until the multi-panel design is approved.
- Gates 6–13 and their existing untracked code/reports remain unaccepted as a batch; this repair does not validate their empirical results or scientific claims.
- No independent reviewer has yet reproduced this gate's diff and tests. No commit is authorized.

Git diff summary: Four previously tracked files changed in this gate (`configs/study.json`, Gate 3 report, protocol, SAP): +141/−58. Two new gate artifacts were added (Decision 0006 and this report), one new 166-line regression test was added, and four pre-existing untracked MEPS implementation/test files were edited in place. The current untracked state is 37 status entries expanding to 43 actual files; three entries/files were added by this gate and the remainder pre-existed it. Pre-existing unrelated tracked modifications in `src/meps_fairness/data/{__init__,download}.py` and `tests/test_download_meps.py` were preserved. The 14 frozen root files and `.gitignore` have zero drift. Staging remains empty.

Proposed next step: Independent read-only review of Decision 0006 scope, the exact gate-touched paths, the 15 non-data tests, and the candidate-panel official metadata. If accepted, authorize a separate commit containing only the explicitly reviewed MEPS pre-experiment repair paths; do not include unrelated download/schema work via broad staging. Then issue a distinct multi-panel statistical-design gate before any additional data access or training.

STOP — waiting for Codex review.
