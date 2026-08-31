# MEPS Gates 6–13 Dependency Closure Audit — Repair-1

Gate: Gates 6–13 and Decision 0004/0006 — final non-data dependency-closure audit after `REPAIR`
Status: `READY_FOR_CODEX_REVIEW_AFTER_REPAIR` — evidence below is independent closure evidence only; it is not `ACCEPT`, does not authorize staging or committing, and does not authorize downloads, pipeline execution, training, bootstrap, or Panel 27 access.

Files changed:

- `src/meps_fairness/data/__init__.py` — bound `get_peak_rss_gb` and `guard_against_prohibited_inspections`, both previously advertised by `__all__` but unbound.
- `tests/test_download_meps.py` — added a package-export regression assertion; retained the existing offline `.dta` manifest and ZIP-safety coverage.
- `docs/decisions/0006-meps-pre-experiment-repair-gate.md` — recorded the Decision 0004 governance predecessor and the narrower current repair boundary.
- `docs/research/STATISTICAL_ANALYSIS_PLAN.md` — removed the paper-informed description of the simple helper and marked future endpoints/mappings with their current locked status.
- `docs/reports/GATE_13_FINAL_SYNTHESIS.md` — replaced the unqualified `155/155` and `Unresolved issues: None` claims with an evidence-bounded historical-status report.
- `docs/reports/MEPS_GATES_6_13_DEPENDENCY_CLOSURE_AUDIT.md` — regenerated as this final audit report.

The final non-archive candidate closure contains exactly 50 paths: all 7 modified tracked paths plus all 43 untracked non-archive paths. The three Gate 6/7 download-related paths and Decision 0004 are included below rather than classified as separate or optional work.

FINAL_NONARCHIVE_CLOSURE (50):

REQUIRED_CONFIG (4):

- `configs/cohort_and_variables.json`
- `configs/meps_schema_expectations.json`
- `configs/meps_stata_artifacts.json`
- `configs/study.json`

REQUIRED_RUNTIME (16):

- `scripts/prepare_meps.py`
- `scripts/run_meps_pipeline.py`
- `src/meps_fairness/data/__init__.py`
- `src/meps_fairness/data/cohort.py`
- `src/meps_fairness/data/download.py`
- `src/meps_fairness/data/prepare.py`
- `src/meps_fairness/data/preprocess.py`
- `src/meps_fairness/data/split.py`
- `src/meps_fairness/evaluation/__init__.py`
- `src/meps_fairness/evaluation/calibration.py`
- `src/meps_fairness/evaluation/inference.py`
- `src/meps_fairness/evaluation/metrics.py`
- `src/meps_fairness/models/__init__.py`
- `src/meps_fairness/models/baseline.py`
- `src/meps_fairness/models/mitigation.py`
- `src/meps_fairness/pipeline.py`

REQUIRED_TEST (10):

- `tests/test_download_meps.py`
- `tests/test_gate10_models.py`
- `tests/test_gate11_smoke.py`
- `tests/test_gate12_evaluation.py`
- `tests/test_gate7_harmonization.py`
- `tests/test_gate8_split.py`
- `tests/test_gate9_preprocessing.py`
- `tests/test_meps_pre_experiment_repair.py`
- `tests/test_meps_prepare.py`
- `tests/test_repairs_regression.py`

REQUIRED_REPORT_OR_GOVERNANCE (20):

- `docs/data/MEPS_SCHEMA_SNAPSHOT.json`
- `docs/decisions/0004-continuous-gemini-batch-implementation.md`
- `docs/decisions/0006-meps-pre-experiment-repair-gate.md`
- `docs/reports/GATE_3_PROTOCOL.md`
- `docs/reports/GATE_6_SCHEMA.md`
- `docs/reports/GATE_7_HARMONIZATION.md`
- `docs/reports/GATE_8_COHORT_SPLIT.md`
- `docs/reports/GATE_9_PREPROCESSING.md`
- `docs/reports/GATE_10_MODELS.md`
- `docs/reports/GATE_11_SMOKE.md`
- `docs/reports/GATE_12_EVALUATION.md`
- `docs/reports/GATE_13_FINAL_SYNTHESIS.md`
- `docs/reports/MEPS_GATES_6_13_DEPENDENCY_CLOSURE_AUDIT.md`
- `docs/reports/MEPS_PRE_EXPERIMENT_REPAIR.md`
- `docs/reports/MEPS_PRE_EXPERIMENT_REPAIR_INDEPENDENT_REVIEW.md`
- `docs/research/CLAIMS_MATRIX.md`
- `docs/research/CONFERENCE_READINESS_CHECKLIST.md`
- `docs/research/MANUSCRIPT_SKELETON.md`
- `docs/research/RESEARCH_PROTOCOL.md`
- `docs/research/STATISTICAL_ANALYSIS_PLAN.md`

EXCLUDED_FROM_CLOSURE:

- `archive/baseline_v0.3/PROVENANCE.json` — excluded by explicit path; not opened, copied, hashed, staged, or submitted.
- `archive/baseline_v0.3/all_results_baseline_v0.3.json` — excluded by explicit path; not opened, copied, hashed, staged, or submitted.
- Ignored runtime directories and MEPS data artifacts (`data/raw/`, `data/interim/`, `runs/`, `outputs/`, `artifacts/`) contributed only HEAD `.gitkeep` placeholders where present; no data artifact was copied. The temporary tree contained no `.dta`, `.ssp`, or `.xpt` files and no `archive/baseline_v0.3/*` file.

DEPENDENCY_AND_DOWNLOAD_REVIEW:

- The fresh candidate was built from HEAD `e5e11f5e2621253bebe8309038a57c6c13cdc9aa`, then overlaid with the exact 50-path closure above. The two archive files were excluded by path.
- `Artifact` now carries `archive_allowed_member_extensions`; omitted manifests retain the secure legacy default `(.ssp, .xpt)`, while `configs/meps_stata_artifacts.json` parses `.dta` for both Stata archives.
- Both the first-download and idempotent-skip paths pass the Artifact-specific extension tuple into ZIP validation. ZIP validation remains single-member, rejects traversal/backslash/absolute paths, directories, symlinks, encryption, invalid extension, size/ratio violations, and CRC failures; `.part` creation and no-clobber promotion remain covered.
- `meps_fairness.data.__all__` was checked after import: every advertised name is bound, including the two repaired helpers.
- The 87-test download suite is entirely offline: temporary directories, in-memory ZIPs, and mocked HTTP responses only. No official endpoint or MEPS file was read.
- Decision 0004 is present in the closure and is explicitly named as the governance predecessor of Decision 0006. Decision 0006 remains the narrower repair boundary and does not authorize a commit or experiment.

Dependency graph:

- 15 local `meps_fairness` modules were imported from the temporary tree under an ambient-`PYTHONPATH`-cleared environment; all module `__file__` paths resolved beneath the temporary tree's `src/`.
- AST scan: `AST_LOCAL_MODULES=15`, `AST_LOCAL_EDGES=23`, `AST_LOCAL_CYCLES=[]`.
- The configuration check confirmed two `.dta` Artifact records, and the package-export check confirmed no missing `__all__` names.

Commands executed:

- Read `AGENTS.md` and `docs/AI_EXECUTION_PROTOCOL.md` before acting; inspected the current Git state, Gate 6–13 reports, Decision 0004/0006, source, and test bodies.
- Built a new temporary tree from `git archive HEAD` and overlaid the exact 50 non-archive paths; no repository file was staged, committed, moved, or deleted.
- Validated the five JSON inputs with the repository virtual environment's `json.tool`.
- Compiled 40 Python files under `src/meps_fairness/`, `scripts/`, and `tests/` with the repository virtual environment.
- Imported the 15 local MEPS package modules with `PYTHONPATH` set only to the temporary tree's `src/` and the ambient `PYTHONPATH` removed.
- Ran the local AST import-edge/cycle scan and package-export/`.dta` manifest assertions.
- Ran the four approved non-data/synthetic test scopes listed below from the temporary tree.
- Computed SHA-256 for every non-self-referential closure path on both the original worktree and the temporary tree; all 49 pairs matched.
- Rechecked the original worktree's branch, HEAD, staging count, expanded status, `git diff --check`, and protected-baseline file set after the audit.
- No download, archive extraction from repository data, MEPS microdata read, pipeline invocation, model training, bootstrap, Panel 27 operation, staging, commit, or push was executed.

Permissions requested: None. No network, package installation, elevated permission, data-use expansion, staging, or commit authorization was requested.

Tests executed:

- `tests.test_download_meps` — complete offline download/archive/provenance/security suite.
- `tests.test_meps_prepare` — complete synthetic preparation/extraction/schema-scan/security suite; all files were temporary test fixtures.
- `tests.test_meps_pre_experiment_repair` — Decision 0006 configuration, naming, runtime-power, calibration, and lock tests.
- Five explicitly approved Gate 7 tests: JSON validity; shared-schema membership; synthetic cohort/target derivation; invalid follow-up-code fail-closed behavior; and temporal leakage rejection.
- No full Gate 7–12 modules, `tests/test_repairs_regression.py`, real Panel 26 integration, Panel 27 operation, pipeline, training, or bootstrap was run.

Exact test results:

- `tests.test_download_meps`: `Ran 87 tests in 0.503s` / `OK`.
- `tests.test_meps_prepare`: `Ran 30 tests in 0.398s` / `OK`.
- `tests.test_meps_pre_experiment_repair`: `Ran 10 tests in 0.003s` / `OK`.
- Approved Gate 7 subset: `Ran 5 tests in 0.014s` / `OK`.
- Combined approved scope: `132/132` tests passed in the temporary candidate.
- JSON validation: 5/5 files valid; Python compilation: 40/40 files passed.
- Required import check: 15/15 modules resolved from the temporary tree; package export check passed; `.dta` manifest check passed.
- AST cycle check: passed with no cycles.
- Hash comparison: 49/49 original-worktree vs temporary-tree non-self pairs equal.
- The historical `155/155` statement in the former Gate 13 report was not used as current evidence; it remains described as historical/mixed-worktree context only.

Input hashes:

- Starting HEAD: `e5e11f5e2621253bebe8309038a57c6c13cdc9aa`.
- Starting HEAD tree: `8c372a1a887c1f995e346ed7432fee84fde9a204`.
- Protected baseline tag: `inherited-code-v0.3-baseline-20260828`.
- Protected baseline commit: `038897e9f751edac6e36445b7706eec5fdb15988`.
- No MEPS microdata input was opened, read, generated, or hashed.

Output hashes:

The following SHA-256 is identical on the original worktree and the temporary candidate for each listed path. The audit report itself is included in the 50-path closure but is self-referential, so its hash is intentionally not embedded.

- `configs/cohort_and_variables.json` — `53e7c22c45a02b8d6787eb680a295d95283cd50a8479f7ca4bef5719e7754c65`
- `configs/meps_schema_expectations.json` — `be9ff7df9394224e6e317ffd77eaf281c0bc49573073ab6d9596572794d8ada5`
- `configs/meps_stata_artifacts.json` — `d21770f35936f5c90ed643f62f8f14760c5c10b0f5d94cf2fd66af273c4fb476`
- `configs/study.json` — `e0201919abc52eb884aa5c39dc6fc1edd73fd8ce4145f9396c35add266865797`
- `docs/data/MEPS_SCHEMA_SNAPSHOT.json` — `64570813a2c8f73f9684d63c47f48f7efc9f384d11e2a9818ccd7003f25fdd09`
- `docs/decisions/0004-continuous-gemini-batch-implementation.md` — `f8ede3e23f352ca0a52167a834d5ac71c4d56f86f1da82ccfb51c37c6a2b3deb`
- `docs/decisions/0006-meps-pre-experiment-repair-gate.md` — `00d1b6a8f94b022eb913c1db53d773210f7f627de749517a9300237d7da41de4`
- `docs/reports/GATE_10_MODELS.md` — `192eb4153fff37f08c25a0307d0d6381b1fb51c0df07385d518c68592fc605ed`
- `docs/reports/GATE_11_SMOKE.md` — `e37e7b76f34de7ca6a59b4abad169a6c155c5fc44ea4a4279b69b6ec208d6a89`
- `docs/reports/GATE_12_EVALUATION.md` — `3c4a7509c6002f17403b41b277e11a8680dab1b645384027cfaa278fbab0286e`
- `docs/reports/GATE_13_FINAL_SYNTHESIS.md` — `a30f5277f432be3c60797d1b367239cd78aee7e63c76e0410fa904ecebf3163c`
- `docs/reports/GATE_3_PROTOCOL.md` — `047c6339da8929fad668c8db406bc1845f4a387556d0449b95b78eaff3fb2d3d`
- `docs/reports/GATE_6_SCHEMA.md` — `1b1aeb8debd7d2d4a977c16cca0d59e16817ab1e594e53132fbb39b901d0ff8a`
- `docs/reports/GATE_7_HARMONIZATION.md` — `5b814286f6c40c091b84a23d3e04799c9db55b289a338f39d3b00e196850592c`
- `docs/reports/GATE_8_COHORT_SPLIT.md` — `43f69d7db7b5ffcd711dc64d47620eb36158bd21dbb1f043fb18c2b991d5c5c0`
- `docs/reports/GATE_9_PREPROCESSING.md` — `ee21760c22c567482d0cdc7b741a28e8dc7830528dbe0cf9683f8c047c10bb73`
- `docs/reports/MEPS_PRE_EXPERIMENT_REPAIR.md` — `6a3f2eafc39382bfeee918a18a1c3dda15920dee9e11de16d997c74e69d6adeb`
- `docs/reports/MEPS_PRE_EXPERIMENT_REPAIR_INDEPENDENT_REVIEW.md` — `0656db3577a5b2e4d6a6a0838cc38cb35f818c577377a6318ded08e78f6313f7`
- `docs/research/CLAIMS_MATRIX.md` — `cdddda8178a153b9e5fde9e823e33cedf0467edd936773042ac6882735167d27`
- `docs/research/CONFERENCE_READINESS_CHECKLIST.md` — `4dae5c0557aad8c3dc147dde107837b88b62e271137f747ee04b0b06f2669e6c`
- `docs/research/MANUSCRIPT_SKELETON.md` — `28ae76d6440749e72b23592416d0a0178028684f828ffc4680080bb7d17e07d5`
- `docs/research/RESEARCH_PROTOCOL.md` — `39d0a979263543f70e14199cd6f4dd1d13f93a570b5e734fb3e031ef8d03e398`
- `docs/research/STATISTICAL_ANALYSIS_PLAN.md` — `8863da460561e0d01545b3950c1016d2ae6d0bfcf63ca393a85f21259277c926`
- `scripts/prepare_meps.py` — `b91a0baee692fef82fbf4d0d38a4cd2e082d917679734c47bb83530569ad8662`
- `scripts/run_meps_pipeline.py` — `b6bbfb64997a9b769dce379f9f296735f06f084ec07d4057c56aaae2d0784ad1`
- `src/meps_fairness/data/__init__.py` — `1818539eed29ab927cc6fd77cd9a43f0533af065be416123efec6cbbe90f44cc`
- `src/meps_fairness/data/cohort.py` — `d6f31bc3d57d8454262e8efaecd879075c685e7936558e0e960bf0408822b5f3`
- `src/meps_fairness/data/download.py` — `6142d2555c3a76c349470b6942cfba7b2fe9b47035063a55540ef9852e7c347b`
- `src/meps_fairness/data/prepare.py` — `f5594fc1209590798340e44ab5881b8535bc0812d78ef63ed62674d2d2d28149`
- `src/meps_fairness/data/preprocess.py` — `a5a64caed10b2341e43aa53b5fd2acc7b703c902083216407ebc6e7abea712b2`
- `src/meps_fairness/data/split.py` — `7fc8e98e5f223871e0930977877c44b78d32dbeefa2a2b8bcb9d88e4445162f1`
- `src/meps_fairness/evaluation/__init__.py` — `3fabc42674935a6d71e2cc2ae09fa1ff47bba9f2eca66fbb1c6f028d23e1d927`
- `src/meps_fairness/evaluation/calibration.py` — `995107d4d4855050333c70c238773615fd31a22de7b58a3ddaabd13739defdc2`
- `src/meps_fairness/evaluation/inference.py` — `88dad232aec336eff6388aa8ea38cd0f5bf4211bc05ead2e5b164b5d1c6317f8`
- `src/meps_fairness/evaluation/metrics.py` — `cc639d44d06852fe8d3d5c8cfae9e95bd26d9a3c96350c0ddc2a8ac7fd7276af`
- `src/meps_fairness/models/__init__.py` — `b66ae065b7b1de67fca38b7fd6455d96b921f97d6308089d1065609c49c11291`
- `src/meps_fairness/models/baseline.py` — `8d06c8c7424e45f6d7b2486e273c98f73896dc9486ca58b46f91fd3ee1986281`
- `src/meps_fairness/models/mitigation.py` — `34571f0fda3f21b465aa422906e7e54d608290f8638616489a519c3f64a18ab7`
- `src/meps_fairness/pipeline.py` — `b9d9c782506bcdf670841dd4db793585400239ab2de9d5d6cbca56c225724239`
- `tests/test_download_meps.py` — `4b047c709326cda834fb89f9950ace711656abde42c66779944ed29241137687`
- `tests/test_gate10_models.py` — `8e8d80de51e44b1edc08f82e83623e2b8e5de402fbc228b81142e0dadb97e337`
- `tests/test_gate11_smoke.py` — `18e4a5189eaf198058396e2636ba62d78d3951f703990662d234355f51ebc39e`
- `tests/test_gate12_evaluation.py` — `d6a4955e65ab6409e6496136f1342aa990e9e1a7718a0bae724ebccd30cb15ba`
- `tests/test_gate7_harmonization.py` — `d190f2c2d0d6ef4c5af8e72bd0c6ed3d7acedcdbc19f38386414dd82b64469ec`
- `tests/test_gate8_split.py` — `d5e38acc93539d3a2c17d236ceba74d282b47ac5eb06bd9cf088363618bf4321`
- `tests/test_gate9_preprocessing.py` — `a067e54e141eb249b32158397d8a9d767ba25d37149b04e3babfae7520ff68b1`
- `tests/test_meps_pre_experiment_repair.py` — `fcda07c7d73c8cbbf39e7edc1c1ea710b06f6180e2030aa427aef74f52738b2c`
- `tests/test_meps_prepare.py` — `f66af3a6f5e2e919ff7fc6651883df481801b96ffcf38b2b409fa52675a7b2c5`
- `tests/test_repairs_regression.py` — `39dae5514345ab6d3d9ec87393044f2b48de7b6c96c4a00805264a6fb3ac63fa`

Row counts: Not applicable. No MEPS data rows, outcome values, protected-group distributions, predictor distributions, model metrics, or bootstrap outputs were read or generated. Synthetic test fixtures were temporary and not retained.

Assumptions:

- The 50-path closure is the exact current non-archive overlay from the original worktree: 7 modified tracked files plus 43 untracked non-archive files. No unexpected non-archive status path was omitted.
- Decision 0004 is included as governance history and is not silently replaced; Decision 0006 narrows its operational scope for this repair.
- Gate 6/7 historical worker reports may contain their own prior run records, but their old test/empirical statements are not promoted to current acceptance evidence by this audit. Gate 13 now states that boundary explicitly.
- The simple MEPS group-mean-difference helper is an exploratory engineering diagnostic, not a Tang/FairBias metric; the SAP no longer calls it paper-informed.
- Panel 27 remains locked, and the current status is still a review boundary rather than a commit or experiment authorization.

Unresolved issues:

- Codex acceptance and any staging/commit authorization remain pending; this report deliberately does not self-accept.
- Historical Gate 6–13 empirical claims were not revalidated in this non-data closure audit, so they remain historical provenance rather than current scientific findings.
- No later multi-panel statistical-design gate has authorized additional development-panel outcome access, pooling, download, training, bootstrap, or Panel 27 evaluation.

Git diff summary: The original worktree remains on `research/meps-hc252-longitudinal` at HEAD `e5e11f5e2621253bebe8309038a57c6c13cdc9aa`, with 7 modified tracked files, 45 expanded untracked files (including this report and the two explicitly excluded archive files), and 0 staged files. `git diff --check` passed. The 14 inherited root files and `.gitignore` have zero drift against `inherited-code-v0.3-baseline-20260828`. No archive file was included in the candidate closure.

Proposed next step: Codex reviews this regenerated 50-path closure, the separate download-safety evidence, the Decision 0004/0006 lineage, and the corrected Gate 13/SAP wording. Until an explicit supervisor decision, keep the closure unstaged and uncommitted and keep data, pipeline, training, bootstrap, and Panel 27 operations prohibited.

STOP — waiting for Codex review.
