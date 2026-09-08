# NHIS D8-R1B Supervisor Handoff & Context Guide

**Repository**: `https://github.com/icarus3344/fairbias-health-equity`  
**Active Branch**: `research/nhis-fairbias`  
**Current Gate**: `NHIS-D8-R1B` (Remediation of R1 Findings, Strict Data Isolation, and Pure Synthetic Verification)  
**Current Gate Status**: `REMEDIATED_SYNTHETIC_VERIFIED_PENDING_CODEX_REVIEW`  
**Date**: 2026-09-08  

---

## 1. Supervisor Role & Governance Model

This repository operates under a strict gated AI supervision model governed by [`docs/AI_EXECUTION_PROTOCOL.md`](AI_EXECUTION_PROTOCOL.md) and [`AGENTS.md`](../AGENTS.md):

- **Supervisor Role (Web-based GPT / Codex)**:
  - Holds supervisory authority over task gating, requirements validation, code review, test result verification, and Git commit authorization.
  - Reviews implementation against gate specifications and verification logs.
  - Issues formal review decisions (`ACCEPT`, `REPAIR`, or `REJECT`).
  - Formulates the exact, bounded specification for the next gate before any worker action.
- **Worker Role (Local Gemini Agent)**:
  - Subordinate implementation worker.
  - Executes strictly the assigned tasks within the active gate specification.
  - Must never self-approve gates or execute out-of-scope refactoring, training, or git actions without explicit supervisor instruction.
  - Concludes every gate report with the mandatory marker: `STOP — waiting for Codex review.`

---

## 2. Protected Baseline & Safety Rules

1. **14 Inherited Baseline Files & `.gitignore`**:
   - Protected tag: `inherited-code-v0.3-baseline-20260828` (`038897e9f751edac6e36445b7706eec5fdb15988`).
   - The 14 root files (`app.py`, `classifiers.py`, `config.py`, `data_COMPAS.csv`, `data_Credit_Card.csv`, `eval.py`, `main.py`, `module_AE.py`, `module_BM.py`, `module_load.py`, `module_transform.py`, `requirements.txt`, `results/all_results.json`, `start.sh`) and `.gitignore` are permanent historical baselines. They must never be altered, moved, renamed, reformatted, or deleted. All new work is strictly additive.
2. **6 Frozen Shared Modules in `src/fairbias/`**:
   - `mitigation.py`, `bias_metric.py`, `transform.py`, `evaluator.py`, `models.py`, `config.py` are frozen core modules. Zero modifications allowed.
3. **Data Isolation Mandate**:
   - Real NHIS/MEPS microdata (`data/`, `data_COMPAS.csv`, `data_Credit_Card.csv`, `*.parquet`) must not be accessed during synthetic contract and unit tests.
   - All tests run under a process-level Python audit hook (`sys.addaudithook`) that intercepts and aborts any attempted `open` calls targeting protected microdata paths.

---

## 3. Milestone Context: Milestone D8 (Transform Enhancement)

### 3.1 Prior Milestone History
- Milestones D1 through D7 established the baseline pipeline, NHIS cohort standardization, empirical evaluation, and stepwise replay harnesses (archived in `archive/` and git history).

### 3.2 Gate D8-R1 & Codex Review
- In Gate D8-R1, Gemini implemented the transform enhancement algorithm (`src/fairbias/enhancement.py`), contracts (`src/fairbias/enhancement_contracts.py`), and runner (`src/nhis_fairbias/d8_enhancement_runner.py`).
- Codex Supervisor conducted an independent adversarial audit (`docs/reports/NHIS_D8_R1_CODEX_REVIEW_20260908T062327Z_ef10b683.md`) and issued a **REPAIR** decision, identifying 8 specific findings labeled **R1B-01 through R1B-08**.

### 3.3 Gate D8-R1B Remediation Summary
Gemini Worker completed all 8 remediations specified in [`docs/plans/NHIS_D8_R1B_REPAIR_20260908.md`](plans/NHIS_D8_R1B_REPAIR_20260908.md):

| Finding | Topic | Remediation Implemented | Key Verification |
| :--- | :--- | :--- | :--- |
| **R1B-01** | Data Isolation & Incident Audit | Refactored `test_backward_compatibility_ae_disabled` to use synthetic DataFrames and mocked loaders. Documented historical kill incident in Incident Addendum. Added `sys.addaudithook` to all test suites. | 0 blocked read attempts across 55 test executions. |
| **R1B-02** | Fairness Guard Strictness | Rejects `O_train=None` when guard enabled (`MISSING_PROTECTED_DATA_WITH_ENABLED_GUARD`). Validates full protected-dim $\times$ active-feature geometry completeness (`PARTIAL_GEOMETRY_RESULTS`). Checks all individual geometry entries for NaN/Inf/negative. Rejects non-finite/negative slack. | `test_missing_protected_data_cannot_disable_enabled_guard`, `test_partial_geometry_is_invalid`, `test_nan_after_finite_entry_is_invalid`, `test_infinite_slack_is_invalid`. |
| **R1B-03** | Baseline Utility Propagation | Baseline utility failure raises `RuntimeError` in `enhance_step` instead of silently collapsing to "candidates exhausted". Runner traps this and sets `termination_reason="evaluation_failed"` and `fairness_feasible=False`. | `test_invalid_baseline_must_reach_caller_as_failure`. |
| **R1B-04** | Configured Classifier & Scaler | `evaluate_candidate_utility` clones `evaluator.model` and uses `evaluator._get_scaler()`, honoring configured DT, RF, LR estimators. Models lacking probability predictors are explicitly rejected (`is_valid=False`, status `MISSING_PROBABILITY_PREDICTOR`). | `test_configured_classifier_matches_utility_oracle`, `test_missing_probabilities_not_replaced_with_hard_predictions`. |
| **R1B-05** | Partition Immutability & Fit NMI | `enhance_step` restricts feature correlation ranking strictly to `partition.fit_X` and `partition.fit_y`. `EvaluationPartition` validates exact index alignment between feature data and protected attributes. Partition fingerprints bind data values, labels, and indices via `hash_pandas_object`. | `test_feature_ranking_uses_only_fit_partition`, `test_protected_index_alignment_is_validated`, `test_partition_fingerprint_binds_values_and_labels`. |
| **R1B-06** | Candidate Audit Event Semantics | Only the single best winning candidate committed to state is logged with `accepted=True`. Other eligible candidates (`gain > min_gain`) not committed are logged with `accepted=False` and `rejection_reason="ELIGIBLE_NOT_COMMITTED"`. | `test_only_committed_candidate_logged_accepted`. |
| **R1B-07** | CLI Collision Guard & Manifest | `scripts/run_nhis_enhancement_study.py` rejects existing non-empty output directories (`FileExistsError`). Writes `execution_manifest.json` with status `"RUNNING"` on start, traps exceptions to record status `"FAILED"` with traceback, and records `"COMPLETED"` on success. | `test_existing_directory_rejected_by_real_cli`, `test_real_cli_records_midrun_failure_manifest`. |
| **R1B-08** | Purge Mock Inspection | Purged all `Mock`, `_mock_wraps`, and `assert_called` inspection from production code (`src/fairbias/enhancement.py`). Tests use standard patching on `evaluate_candidate_utility`. | Unified production execution verified across all suites. |

---

## 4. Test Results & Verification Evidence

All tests pass 100% with zero blocked microdata reads:

1. **Codex Adversarial Review Suite** (`artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/adversarial_review.py`):
   - **14 / 14 passed**, 0 failures, 0 errors, 0 blocked read attempts (0.058s).
2. **Guarded Worker Suite** (`tests/test_fairbias_enhancement.py`, `tests/test_nhis_d8_enhancement.py`, `tests/test_fairbias_enhancement_contracts.py`, `tests/test_nhis_d8_synthetic_contracts.py`):
   - **41 / 41 passed**, 0 failures, 0 errors, 0 blocked read attempts (3.333s).
3. **Core FairBias Suite** (`tests/test_fairbias_evaluator.py`, `tests/test_fairbias_golden_formulas.py`, `tests/test_fairbias_transform.py`):
   - **30 / 30 passed**, 0 failures, 0 errors (0.156s).
4. **Total D8 Verification**: **55 / 55 passed**, 0 failures, 0 errors.

---

## 5. Key File Index for Supervisor Review

### Reports & Plans
- Worker Gate Report: [`docs/reports/NHIS_D8_R1B_REPAIR_REPORT_20260908T063242Z_5ac3c57a.md`](reports/NHIS_D8_R1B_REPAIR_REPORT_20260908T063242Z_5ac3c57a.md)
- Codex Previous Review: [`docs/reports/NHIS_D8_R1_CODEX_REVIEW_20260908T062327Z_ef10b683.md`](reports/NHIS_D8_R1_CODEX_REVIEW_20260908T062327Z_ef10b683.md)
- R1B Remediation Plan: [`docs/plans/NHIS_D8_R1B_REPAIR_20260908.md`](plans/NHIS_D8_R1B_REPAIR_20260908.md)
- Original D8 Plan: [`docs/plans/NHIS_D8_ENHANCEMENT_REPAIR_PLAN_20260908.md`](plans/NHIS_D8_ENHANCEMENT_REPAIR_PLAN_20260908.md)

### Source Code
- Enhancement Algorithm: [`src/fairbias/enhancement.py`](../src/fairbias/enhancement.py)
- Enhancement Contracts & Oracles: [`src/fairbias/enhancement_contracts.py`](../src/fairbias/enhancement_contracts.py)
- Enhancement State Representation: [`src/fairbias/enhancement_state.py`](../src/fairbias/enhancement_state.py)
- NHIS D8 Enhancement Runner: [`src/nhis_fairbias/d8_enhancement_runner.py`](../src/nhis_fairbias/d8_enhancement_runner.py)
- Study CLI Entrypoint: [`scripts/run_nhis_enhancement_study.py`](../scripts/run_nhis_enhancement_study.py)
- Pipeline Integration: [`src/fairbias/pipeline.py`](../src/fairbias/pipeline.py)

### Test Suites
- Comprehensive Contracts: [`tests/test_fairbias_enhancement_contracts.py`](../tests/test_fairbias_enhancement_contracts.py)
- Synthetic Runner Contracts: [`tests/test_nhis_d8_synthetic_contracts.py`](../tests/test_nhis_d8_synthetic_contracts.py)
- Core Algorithm Unit Tests: [`tests/test_fairbias_enhancement.py`](../tests/test_fairbias_enhancement.py)
- Runner Unit Tests: [`tests/test_nhis_d8_enhancement.py`](../tests/test_nhis_d8_enhancement.py)
- Codex Adversarial Review Suite: [`artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/adversarial_review.py`](../artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/adversarial_review.py)

### Audit Artifacts
- R1B Repair Manifest & Logs: [`artifacts/nhis_d8_repair/20260908T063242Z_5ac3c57a/`](../artifacts/nhis_d8_repair/20260908T063242Z_5ac3c57a/)
- Codex Review Manifest & Logs: [`artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/`](../artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/)

---

## 6. Supervisor Decision Protocol (For Web-based GPT)

When reviewing this repository on GitHub, please follow these steps:

1. **Verify Baseline & Scope**:
   - Confirm zero changes to 14 inherited files (`app.py`, etc.) and 6 frozen shared modules (`mitigation.py`, etc.).
   - Confirm all test suites use purely synthetic data under strict audit isolation.
2. **Evaluate Remediations (R1B-01 through R1B-08)**:
   - Check code in `src/fairbias/enhancement.py` and `src/fairbias/enhancement_contracts.py` against the findings in `docs/reports/NHIS_D8_R1_CODEX_REVIEW_20260908T062327Z_ef10b683.md`.
   - Verify that all 14 adversarial checks in `artifacts/nhis_d8_codex_review/20260908T062327Z_ef10b683/adversarial_review.py` are addressed and pass.
3. **Issue Supervisor Review Verdict**:
   - If satisfied:
     - Issue decision: `ACCEPT NHIS-D8-R1B`.
     - Authorize the next gate (e.g. `NHIS-D8-R2`: Baseline Replication & Synthetic Microdata Evaluation, or whatever gate specification you define).
     - Specify exact requirements, files permitted to touch, prohibited actions, and verification criteria.
   - If further remediation is required:
     - Issue decision: `REPAIR NHIS-D8-R1C`.
     - Detail the remaining defects, file paths, line numbers, and required contract behaviors.
