# Gate Report: NHIS D8-R1B (Remediation of R1 Findings, Strict Data Isolation, and Pure Synthetic Verification)

Gate: NHIS-D8-R1B
Status: REMEDIATED_SYNTHETIC_VERIFIED_PENDING_CODEX_REVIEW
Files changed:
- Modified: `src/fairbias/enhancement.py`
- Modified: `src/fairbias/enhancement_contracts.py`
- Modified: `src/nhis_fairbias/d8_enhancement_runner.py`
- Modified: `scripts/run_nhis_enhancement_study.py`
- Modified: `tests/test_fairbias_enhancement.py`
- Modified: `tests/test_fairbias_enhancement_contracts.py`
- Modified: `tests/test_nhis_d8_enhancement.py`
- Modified: `tests/test_nhis_d8_synthetic_contracts.py`
- Tracked (prior R1 modification, unmodified in R1B): `src/fairbias/pipeline.py`
- Tracked (prior R1 addition, unmodified in R1B): `src/fairbias/enhancement_state.py`
- Added report: `docs/reports/NHIS_D8_R1B_REPAIR_REPORT_20260908T063242Z_5ac3c57a.md`

Commands executed:
1. Git baseline and status verification:
   `git status -s`
   `git diff inherited-code-v0.3-baseline-20260828 -- app.py classifiers.py config.py data_COMPAS.csv data_Credit_Card.csv eval.py main.py module_AE.py module_BM.py module_load.py module_transform.py requirements.txt results/all_results.json start.sh .gitignore`
   `git diff HEAD -- src/fairbias/mitigation.py src/fairbias/bias_metric.py src/fairbias/transform.py src/fairbias/evaluator.py src/fairbias/models.py src/fairbias/config.py`
2. Pre-R1B snapshot and manifest creation:
   Archived starting file copies, diffs, untracked file list, and pre-repair manifest to `artifacts/nhis_d8_repair/20260908T063242Z_5ac3c57a/`. Verified SHA-256 hashes against Codex `artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/integrity.json`.
3. Test suite execution under process-level `sys.addaudithook` data guard:
   `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m unittest -v artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/adversarial_review.py` (14 tests)
   `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m unittest -v tests/test_fairbias_enhancement.py tests/test_nhis_d8_enhancement.py tests/test_fairbias_enhancement_contracts.py tests/test_nhis_d8_synthetic_contracts.py` (41 tests)
   `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m unittest -v tests/test_fairbias_evaluator.py tests/test_fairbias_golden_formulas.py tests/test_fairbias_transform.py` (30 tests)
4. Artifact archival and manifest generation:
   Created `post_r1b_snapshot/`, `post_r1b_manifest.json`, `post_r1b_tracked_diff_vs_head.patch`, `incremental_r1b_diff.patch`, `worker_suite_guarded.log`, `worker_suite_guarded.json`, `adversarial_review.log`, `adversarial_review.json` in `artifacts/nhis_d8_repair/20260908T063242Z_5ac3c57a/`.

Permissions requested:
None.

Tests executed:
1. `artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/adversarial_review.py` (14 independent adversarial tests):
   - `test_configured_classifier_matches_utility_oracle`: DecisionTree classifier configuration produces matching AUROC with independent oracle.
   - `test_existing_directory_rejected_by_real_cli`: Real CLI rejects existing non-empty directory with `FileExistsError`.
   - `test_feature_ranking_uses_only_fit_partition`: Feature NMI correlation ranking accesses only fit partition labels (80/80 rows), never selection labels.
   - `test_infinite_slack_is_invalid`: Fairness evaluation rejects infinite `max_fairness_degradation`.
   - `test_invalid_baseline_must_reach_caller_as_failure`: Invalid baseline utility evaluation propagates as `RuntimeError`/`ValueError` to caller.
   - `test_missing_probabilities_not_replaced_with_hard_predictions`: Estimators lacking probability predictions are rejected (`is_valid=False`).
   - `test_missing_protected_data_cannot_disable_enabled_guard`: Missing protected data with enabled guard is explicitly rejected (`is_acceptable=False`).
   - `test_nan_after_finite_entry_is_invalid`: Non-finite/NaN geometry metrics are rejected even if finite entries precede them.
   - `test_only_committed_candidate_logged_accepted`: Only the single best candidate committed is logged with `accepted=True`; others are logged as `ELIGIBLE_NOT_COMMITTED`.
   - `test_original_drop_bug_fixed_with_arithmetic_oracle`: Arithmetic oracle verification for dropped column + power transform state.
   - `test_partial_geometry_is_invalid`: Missing active features in geometry output rejected as `PARTIAL_GEOMETRY_RESULTS`.
   - `test_partition_fingerprint_binds_values_and_labels`: Partition fingerprints change when underlying values or labels change.
   - `test_protected_index_alignment_is_validated`: Index mismatch between feature partition and protected attributes raises `ValueError`.
   - `test_real_cli_records_midrun_failure_manifest`: Real CLI writes `execution_manifest.json` with status `FAILED` upon mid-run exception.

2. `tests/test_fairbias_enhancement_contracts.py` (29 comprehensive contract tests):
   - 17 original D8-R1 contracts: partition identity/power/rebin/drop transformation, selection score oracle, dropped column regression handling, feature collapse handling, single class selection error, non-finite output rejection, index contract validation, category mapping composition/conflicts/JSON round-trip/string preservation, state cache re-evaluation, cycle detection, transactional safety on rejection, fairness guard boundary check, audit event logging, and backward compatibility with synthetic pipeline.
   - 12 R1B remediation contracts: missing protected data rejection, partial geometry rejection, NaN geometry entry rejection, infinite slack rejection, invalid baseline propagation, DT utility oracle matching, missing probability rejection, protected index validation, data-bound fingerprint validation, fit-only NMI ranking, single committed candidate logging, and arithmetic oracle match.

3. `tests/test_nhis_d8_synthetic_contracts.py` (5 synthetic runner and CLI contracts):
   - Sentinel error on real parquet access without permission.
   - Synthetic runner execution across all 4 conditions (baseline, canonical, posthoc, joint) verifying dual BM+AE events and AUPRC metrics.
   - Real CLI non-empty directory collision rejection (`FileExistsError`).
   - Real CLI mid-run failure manifest generation (`status: FAILED`).
   - Candidate audit events logging in runner.

4. `tests/test_fairbias_enhancement.py` (5 behavioral tests):
   - Ranking identifies top active feature without touching dropped features.
   - Monotonic exponent tracking without oscillation or infinite loops.
   - Fairness degradation bound rejection.
   - Utility rejection when candidate score drops.
   - Successful enhancement acceptance when score strictly improves.

5. `tests/test_nhis_d8_enhancement.py` (2 tests):
   - Group fairness gaps calculation (DP and EO differences).
   - Synthetic smoke test runner execution on arm 1.

Exact test results:
- Guarded Worker Suite (`tests/test_fairbias_enhancement.py`, `tests/test_nhis_d8_enhancement.py`, `tests/test_fairbias_enhancement_contracts.py`, `tests/test_nhis_d8_synthetic_contracts.py`):
  Ran 41 tests in 3.333s. Status: OK (41 passed, 0 failures, 0 errors, 0 blocked read attempts).
- Codex Adversarial Review Suite (`artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/adversarial_review.py`):
  Ran 14 tests in 0.058s. Status: OK (14 passed, 0 failures, 0 errors, 0 blocked read attempts).
- Core FairBias Suite (`tests/test_fairbias_evaluator.py`, `tests/test_fairbias_golden_formulas.py`, `tests/test_fairbias_transform.py`):
  Ran 30 tests in 0.156s. Status: OK (30 passed, 0 failures, 0 errors).
- Overall D8 Verification: 55/55 passed, 0 failures, 0 errors, 0 blocked read attempts.

Incident Addendum (R1B-01 Verification & Audit Scope):
- Confirmed Historical Fact 1: In Gate D8-R1, test `test_backward_compatibility_ae_disabled` in `tests/test_fairbias_enhancement_contracts.py` utilized `FairBiasConfig.compas_default()`, which invoked `load_raw()` on `data_COMPAS.csv`. In R1B, this test was completely refactored to use in-memory synthetic DataFrames with mocked loader and hash functions. Zero disk files were read.
- Confirmed Historical Fact 2: Early in Gate D8-R1 execution, the command `python3 -m unittest tests/test_nhis_d8_enhancement.py` was started prior to test modification and was subsequently terminated via kill signal after approximately 12 seconds. That un-refactored test constructed a `D8EnhancementRunner` with default adapter settings which pointed to `data/processed/nhis/nhis_2022_2024_core.parquet`.
- Scope of Unknowns: The exact byte offset or row count ingested by pyarrow before the kill signal took effect cannot be definitively determined from process logs. No model training reached completion. In accordance with supervisor instruction, this legacy command was not re-run.
- Preventive Audit Hook Enforcement: In Gate D8-R1B, every active test suite installed a process-level `sys.addaudithook` data guard intercepting `open` calls targeting `data/`, `data_COMPAS.csv`, `data_Credit_Card.csv`, and `*.parquet`. All 41 tests in the worker suite and all 14 tests in the adversarial review suite completed with exactly 0 blocked read attempts (`blocked_read_attempts: []`).

Remediation Item Breakdown:
1. R1B-01 (Data Isolation & Incident Addendum): Completely eliminated all disk reads in test suites; installed global audithooks in all 4 test modules; confirmed 0 blocked reads.
2. R1B-02 (Fairness Guard Strictness):
   - Rejects `O_train=None` when guard is enabled (`is_acceptable=False`, `rejection_reason="MISSING_PROTECTED_DATA_WITH_ENABLED_GUARD"`).
   - Validates active feature completeness across all protected dimensions against geometry outputs (`PARTIAL_GEOMETRY_RESULTS`).
   - Checks every individual geometry entry for NaN, Inf, or negative values (`NON_FINITE_CANDIDATE_DPHI`).
   - Rejects non-finite or negative `max_fairness_degradation` and reference epsilons.
3. R1B-03 (Baseline Utility Failure Propagation):
   - When baseline utility evaluation returns `is_valid=False`, `enhance_step` raises `RuntimeError(f"Baseline utility evaluation failed ({base_res.validity_status}): {base_res.error_message}")`.
   - `D8EnhancementRunner` catches `(RuntimeError, ValueError)` and records `termination_reason="evaluation_failed"` and `fairness_feasible=False`.
4. R1B-04 (Configured Classifier & Scaler Contract):
   - `evaluate_candidate_utility` uses `clone(evaluator.model)` and `evaluator._get_scaler()` to strictly honor configured estimators and scalers (supporting DT, RF, LR).
   - Explicitly rejects models lacking `predict_proba` or `decision_function` with `is_valid=False` and status `MISSING_PROBABILITY_PREDICTOR`.
5. R1B-05 (Partition Immutability & Fit-Only NMI):
   - Feature correlation ranking in `enhance_step` receives strictly `partition.fit_X` and `partition.fit_y`.
   - `EvaluationPartition` validates exact index alignment between feature data and protected attributes (`protected_fit.index.equals(fit_X.index)`).
   - Partition fingerprints are bound to actual data content, labels, and index hashes via `pd.util.hash_pandas_object`.
6. R1B-06 (Candidate Audit Events & Committed Semantics):
   - Only the single best candidate committed in each step is logged with `accepted=True`.
   - All other evaluated candidates with positive utility gain (`gain > min_gain`) are logged with `accepted=False` and `rejection_reason="ELIGIBLE_NOT_COMMITTED"`.
   - Candidates with gain <= min_gain are logged with `rejection_reason="INSUFFICIENT_GAIN: ..."`.
7. R1B-07 (CLI Output Protection & Manifest Lifecycle):
   - CLI checks `if out_dir.exists() and any(out_dir.iterdir()): raise FileExistsError(...)`.
   - Initial `execution_manifest.json` with status `"RUNNING"` is written immediately upon directory creation.
   - On unhandled exception during execution, CLI writes `execution_manifest.json` with status `"FAILED"`, error type, and traceback before re-raising.
   - On completion, CLI writes `execution_manifest.json` with status `"COMPLETED"` and file SHA-256 digests.
8. R1B-08 (Removal of Production Mock Detection):
   - Completely deleted all inspection of `Mock`, `_mock_wraps`, and `assert_called` from `src/fairbias/enhancement.py`. All paths execute unified production contracts.

Input hashes (Pre-R1B Snapshot):
- `src/fairbias/enhancement.py`: `6efd3ec9303b9620b41de350ad50998058db68ed4f35fb28d6953d443ff81b77` (36897 bytes)
- `src/fairbias/pipeline.py`: `71082b87094d4b633044b597c361731f36e0b53b871cc4f712f39efed675a240` (36364 bytes)
- `src/fairbias/enhancement_contracts.py`: `e24a2eba67102557805ca109351d02772d9495a71e40718aa4b97d23f35fa5d7` (12339 bytes)
- `src/fairbias/enhancement_state.py`: `9d00d4e240763a436f844257a6f298b3af5b6f7521bf1791304928bf3744a151` (7348 bytes)
- `src/nhis_fairbias/d8_enhancement_runner.py`: `bb4a410ba35f18debad6db0a6040c2bfa6f3379ecc0a6cf93f7f0529bd7e2f27` (24944 bytes)
- `scripts/run_nhis_enhancement_study.py`: `af82631e4a7f6a4ac6a58e21405c9894134ad563a7bf242fbe7cc56b63bfbd1a` (7226 bytes)
- `tests/test_fairbias_enhancement.py`: `d33ee31734d6242c6eea79ddb96af105773a6975d0a0fae9ff3f0efa90dac9fc` (5758 bytes)
- `tests/test_nhis_d8_enhancement.py`: `02dd848f04b141f54853165bd689747e3dcc589ccd363b3d9715d57e05707bc2` (2917 bytes)
- `tests/test_fairbias_enhancement_contracts.py`: `a19e90490a2c39871c6ee0df76f7da0a41b8aad6a4ccf1a7a50ddd79c0e5649f` (17010 bytes)
- `tests/test_nhis_d8_synthetic_contracts.py`: `8a4fa53d6ff9ed2cc08a88a4028791e2247d36d5732f1f404dc29f448c75aa2f` (5549 bytes)

Output hashes (Post-R1B Remediation):
- `src/fairbias/enhancement.py`: `f533dba7810d24196a856e8ba3f33ce6f3282d0d2b6e70ee5872b83e24a7ec38` (39036 bytes) [CHANGED]
- `src/fairbias/pipeline.py`: `71082b87094d4b633044b597c361731f36e0b53b871cc4f712f39efed675a240` (36364 bytes) [UNCHANGED]
- `src/fairbias/enhancement_contracts.py`: `19c8bc7eea218cbe76d30eeb972f7d317e2c44413a2e94499da7f64b0a9886e2` (14102 bytes) [CHANGED]
- `src/fairbias/enhancement_state.py`: `9d00d4e240763a436f844257a6f298b3af5b6f7521bf1791304928bf3744a151` (7348 bytes) [UNCHANGED]
- `src/nhis_fairbias/d8_enhancement_runner.py`: `2891972d7c53c27a8ffc89f2ab8f3efd786d693c97ccee3b829274690b68b616` (25416 bytes) [CHANGED]
- `scripts/run_nhis_enhancement_study.py`: `30b107cfdb4cbfd1b5a16f7f2cb01554783b8f4f681705af25bc92ffcf40623b` (8507 bytes) [CHANGED]
- `tests/test_fairbias_enhancement.py`: `5fbe497b35d80e1db70a00a34936a2b50f3c5b9f380931e24215549886ca7c02` (6974 bytes) [CHANGED]
- `tests/test_nhis_d8_enhancement.py`: `4f6d44f06b5a00574c677e0794aae42297f055ef4df1248347c2021335de91d9` (3448 bytes) [CHANGED]
- `tests/test_fairbias_enhancement_contracts.py`: `8443e2639e629cf4cb93ad383d9a94d907613b6ecf2d8921883fffdacf502f74` (24770 bytes) [CHANGED]
- `tests/test_nhis_d8_synthetic_contracts.py`: `d2e541ca3abedb023b10acd7b295614b2183aa57b37601f8f1325b5336048489` (6947 bytes) [CHANGED]

Protected Files Integrity:
- 14 baseline root files: Zero diff vs protected tag `inherited-code-v0.3-baseline-20260828`.
- `.gitignore`: Matches precheck recorded state (4 presentation/build lines added prior to gate).
- 6 frozen shared modules: Zero diff vs HEAD (`src/fairbias/mitigation.py`, `src/fairbias/bias_metric.py`, `src/fairbias/transform.py`, `src/fairbias/evaluator.py`, `src/fairbias/models.py`, `src/fairbias/config.py`).

Row counts:
- Real NHIS microdata: 0 rows read, 0 records sampled, 0 microdata files opened (verified by `sys.addaudithook` data guard blocking all parquet/CSV access).
- Synthetic test data: 120 train rows, 60 selection rows per contract test iteration; 100/150 synthetic rows per mock cohort test.

Assumptions:
1. Strict Synthetic Verification: Empirical claims and testing in R1B are strictly synthetic. No real microdata was accessed.
2. Committed Candidate Semantics: Audit trail events differentiate between candidates that met minimal gain (`gain > min_gain`) but were superseded, versus the single winning candidate committed to state.
3. Legacy Fairness Bound Policy: Continues to follow `legacy_epsilon_plus_absolute_slack` (threshold + 0.02) with strict input validation.

Unresolved issues:
1. Real Data Execution: Real NHIS cohorts for 2022, 2023, and 2024 were not opened or evaluated; empirical study on NHIS arms remains to be scheduled in subsequent authorized gates.
2. Pre-existing Gitignore Drift: The pre-existing baseline drift in `.gitignore` (`PRE_EXISTING_BASELINE_DRIFT_RECORDED_NOT_RESOLVED`) remains recorded and untouched.
3. Complex Survey Weighting: D8 evaluates unweighted sample-level representations; survey design weights, strata, and PSU clustering remain deferred to Gate D8-R5.
4. Gate Gating: Gate D8-R2 through D8-R5 are not authorized and have not been executed.

Git diff summary:
- Tracked files modified: `src/fairbias/enhancement.py`, `src/fairbias/pipeline.py`.
- Untracked new files: `src/fairbias/enhancement_contracts.py`, `src/fairbias/enhancement_state.py`, `src/nhis_fairbias/d8_enhancement_runner.py`, `scripts/run_nhis_enhancement_study.py`, `tests/test_fairbias_enhancement.py`, `tests/test_nhis_d8_enhancement.py`, `tests/test_fairbias_enhancement_contracts.py`, `tests/test_nhis_d8_synthetic_contracts.py`, `docs/reports/NHIS_D8_R1B_REPAIR_REPORT_20260908T063242Z_5ac3c57a.md`.
- No protected files were modified. No commits were created or staged.

Proposed next step:
Codex review of the D8-R1B remediation, test evidence, and audit logs. Awaiting supervisor review and decision.

STOP — waiting for Codex review.
