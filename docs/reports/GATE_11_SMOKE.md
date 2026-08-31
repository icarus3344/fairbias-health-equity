# Gate 11: Smoke Workflow, Platt Calibration, Capacity Freezing & Pre-Unlock Manifest Report

## 1. Executive Summary

Gate 11 executes the end-to-end smoke verification pipeline, survey-weighted Platt probability calibration, 10% operational capacity threshold freezing, design-aware stratified-PSU bootstrap inference, and immutable pre-unlock manifest compilation on branch `research/meps-hc252-longitudinal`. Under authorized execution:
- **Smoke Workflow & Non-Evidentiary Labeling**: The smoke execution was executed to verify pipeline integrity, numerical stability, and reproducibility. In accordance with Section 8 of the AI Execution Protocol, smoke results are explicitly designated **non-evidentiary software verification records**.
- **Survey-Weighted Platt Probability Calibration & Apparent Fit Boundary**: Logistic scaling fit on the untouched Calibration partition (20%) using survey analysis weights `LONGWT` and light L2 regularization to prevent separation overflow. Metrics computed on this calibration partition are strictly designated **apparent calibration-fit diagnostics**, not out-of-sample or holdout performance.
- **Operational Decision Threshold Freezing**: The 10% weighted population capacity decision threshold was frozen on the calibrated Panel 26 calibration set.
- **Subgroup Fairness Audit & Strict Suppression**: Evaluated subgroup TPR, FPR, PPV, and selection rates across `RACETHX` (OMB 5 categories) and `SEX` (Male/Female). Subgroups with $n < 100 \lor \text{pos} < 20 \lor \text{neg} < 20 \lor n_{\text{eff}} < 50$ are strictly marked `"Suppressed (Insufficient Sample/Power)"` without ad hoc merging. 100% of evaluation subgroup cells are suppressed; consequently, pairwise max TPR gaps are `None` (Not Estimable) and propagate through `primary_fairness_endpoint` and bootstrap inference without fabricating a numeric zero or fairness confidence interval. Utility bootstrap inference remains valid and estimable.
- **Design-Aware Stratified-PSU Bootstrap**: Executed paired difference bootstrap resampling within `VARSTR` strata across `VARPSU` clusters.
- **Computational Audit & Resource Bounds**: Smoke runtime < 10 seconds, peak RSS < 0.35 GB.
- **Pre-Unlock Manifest & Holdout Gating Audit**:
  - Pre-unlock manifest generated strictly as an immutable file in the unique run directory `runs/<run_id>/pre_unlock_manifest.json`, with a mutable pointer recorded at `runs/latest_run_manifest_pointer.json` and execution logged to `runs/runs_history.jsonl`.
  - Prerequisite Audit:
    1. Tests passed: 100% passing.
    2. Panel 26 eligible positives: **136 cases** (under the prespecified 200-positive statistical power threshold).
    3. Frozen pipeline hashes: Recorded and verified.
    4. Leakage checks: Passed.
    5. Survey variance structure: Valid (Kish $n_{\text{eff}} = 1,745.2$).
    6. Smoke runtime & RSS: Passed (< 30 min, < 16 GB).
  - **Pre-Unlock Verdict**: Because Panel 26 contains 136 positive cases ($< 200$), the study is underpowered under the prespecified primary continuous-coverage estimand. Under canonical stop conditions and hard safety boundaries, **Panel 27 (HC-252) remains strictly LOCKED**. The study stopped before Panel 27 evaluation; no conference paper result package is complete.


---

## 2. Section 9 Standard Worker Report

Gate: Gate 11 — Smoke Workflow, Platt Calibration, Capacity Freezing & Pre-Unlock Manifest
Status: COMPLETED_BY_GEMINI_BATCH_PENDING_FINAL_CODEX_REVIEW
Files changed:
- docs/reports/GATE_11_SMOKE.md
- scripts/run_meps_pipeline.py
- src/meps_fairness/evaluation/__init__.py
- src/meps_fairness/evaluation/calibration.py
- src/meps_fairness/evaluation/inference.py
- src/meps_fairness/evaluation/metrics.py
- src/meps_fairness/pipeline.py
- tests/test_gate11_smoke.py
Commands executed:
- /Users/lkc/Downloads/code_v_0_3/.venv311/bin/python -m py_compile scripts/run_meps_pipeline.py src/meps_fairness/pipeline.py src/meps_fairness/evaluation/calibration.py src/meps_fairness/evaluation/metrics.py src/meps_fairness/evaluation/inference.py tests/test_gate11_smoke.py
- PYTHONPATH=src /Users/lkc/Downloads/code_v_0_3/.venv311/bin/python -m unittest tests/test_gate11_smoke.py -v
- /Users/lkc/Downloads/code_v_0_3/.venv311/bin/python scripts/run_meps_pipeline.py --help (from /tmp)
- git diff --check
Permissions requested: None
Tests executed:
- `test_weighted_metrics_synthetic`: Verified weighted AUROC, AUPRC, Brier, calibration statistics, and capacity-aware metrics.
- `test_subgroup_suppression_logic`: Verified suppression of small subgroups ($n < 100 \lor \text{pos} < 20$).
- `test_platt_calibrator_and_threshold_freezing`: Verified survey-weighted Platt probability calibration and 10% capacity threshold freezing.
- `test_smoke_pipeline_end_to_end_integration`: Verified full end-to-end smoke run, runtime/RSS bounds, pre-unlock manifest compilation, apparent-fit labeling, and holdout locking.
- `test_trapezoid_integration_backward_compatibility`: Verified backward-compatible trapezoid integration with fallback support.
- `test_platt_calibrator_numerical_stability_on_extreme_probs`: Verified numerical stability and absence of separation overflow on extreme logits.
- `test_cli_subprocess_alternate_cwd`: Verified CLI `--help` invocation from alternate cwd.
Exact test results:
- 7/7 tests passed cleanly.
Input hashes:
- Inherited baseline tag: inherited-code-v0.3-baseline-20260828 (commit 038897e9f751edac6e36445b7706eec5fdb15988)
- data/interim/meps/h244/h244.dta: 5cf983c94fd9ed8d8377c9ad27bebd905c545327eca823a8c8412bf4e66eaa70
- src/meps_fairness/data/cohort.py: 8d1f2e82110c71a3e9c5eb79e2c608f62f83141f2ff2ad07b57bfb7b34b15096
- src/meps_fairness/data/split.py: e6fe753907ec347895bc6e3ef54ca854cf60aa4f7a55ae5c2c5c93c31878b19a
- src/meps_fairness/data/preprocess.py: c0d5885c48ec9a9beea4fbcfca16f393855ff4bbff1b83149c4fbf5a0494cf8e
- src/meps_fairness/models/baseline.py: f55bbffc304d9c490a6e0e64c39f1c7d23d8c1c4f74d084d5df68b209e9e1c25
- src/meps_fairness/models/mitigation.py: 8a7098c199587426162354c0e64e525143a53caab6db5a4fead4bcadcd8254c2
Output hashes:
- scripts/run_meps_pipeline.py: 2dc063546747b0a701d017a02294101c70e28e4695bfe9873d6e5a6aa6ca8488
- src/meps_fairness/evaluation/calibration.py: ecf9ecf4a7c06eb6f6587fa9f3ce098555e5138127387fc15ae4ae0c00d41eb3
- src/meps_fairness/evaluation/metrics.py: b253f93796d11342b47e5b6028a6fcf746b1fe751be67c714ec50ebcb141ee55
- src/meps_fairness/evaluation/inference.py: 7a82fc8db463e26c6d05be19694e9f73360ebc6a1e3590059bda20320a7b4588
- src/meps_fairness/pipeline.py: f7e06180327ca3ce45df9e0f6b3e34b9cf4fba8bfa04fb46294d1f274cb3b7d7
- tests/test_gate11_smoke.py: 41819d9b62f1cf0fb982181519001b972e2dc629a43a088bc015f624d7cb27a1
- docs/reports/GATE_11_SMOKE.md: self-referential report artifact
Row counts:
- Panel 26 Cohort: 2,882 eligible records -> 1,732 Train, 569 Validation, 581 Calibration.
Assumptions:
- Smoke run is labeled as non-evidentiary software verification.
- Metrics evaluated on the calibration partition are apparent fit diagnostics.
- Decision threshold is frozen at 10% weighted population capacity on the Calibration partition.
Unresolved issues:
- Statistical Power Finding: Panel 26 analytic cohort exhibits 136 positive cases ($< 200$), triggering canonical stop condition for temporal holdout unlock. Study stopped before Panel 27 evaluation.
Git diff summary: 7 additive untracked files (`scripts/run_meps_pipeline.py`, `src/meps_fairness/evaluation/__init__.py`, `src/meps_fairness/evaluation/calibration.py`, `src/meps_fairness/evaluation/inference.py`, `src/meps_fairness/evaluation/metrics.py`, `src/meps_fairness/pipeline.py`, `tests/test_gate11_smoke.py`, `docs/reports/GATE_11_SMOKE.md`). Zero inherited baseline files modified.
Proposed next step: Proceed to Gate 12 (Fail-Closed Locked-Holdout Report & Stop Condition Audit).
STOP — waiting for Codex review.

