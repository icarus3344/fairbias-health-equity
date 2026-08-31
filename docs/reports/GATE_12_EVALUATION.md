# Gate 12: Fail-Closed Locked-Holdout Report & Stop Condition Audit

## 1. Executive Summary

Gate 12 documents the fail-closed locked-holdout evaluation workflow across the five prespecified random development seeds (`20260828` through `20260832`), evaluates the conditional holdout unlock token (`CODEX_BATCH_20260829`), and enforces canonical stop conditions on branch `research/meps-hc252-longitudinal`. Under authorized execution:
- **Fail-Closed Locked-Holdout Status**: This report is a **fail-closed locked-holdout report, NOT a completed temporal evaluation**. The study stopped before Panel 27 evaluation; no conference paper result package is complete.
- **Pre-Unlock Prerequisite Verification & Fail-Closed Gating**:
  - The conditional unlock token `CODEX_BATCH_20260829` required that all Gate 7–11 tests pass, Panel 26 target has $\ge 200$ positive events, all model and configuration hashes are frozen and recorded, leakage tests pass, and survey design structures are valid.
  - **Empirical Power Finding**: On development Panel 26 (HC-244), the eligible continuous-baseline cohort contains exactly 2,882 individuals with exactly **136 positive coverage interruption events** ($Y=1$, event rate 4.72%).
  - **Fail-Closed Stop Condition Enforcement**: Because $136 < 200$, the statistical power threshold was not satisfied. In strict compliance with Section 10 of the Statistical Analysis Plan and the project governance mandate, **Panel 27 (HC-252) remains strictly LOCKED under `LOCKED_UNDERPOWERED_STOP_CONDITION`**. Zero Panel 27 outcome variables, distributions, protected subgroup event distributions, prevalence, or evaluation metrics were accessed, inspected, or tuned.
- **Exploratory Panel 26 Development Evaluation (5-Seed Summary with 74-Predictor Specification)**:
  - Executed formal survey-weighted Platt scaling calibration, 10% capacity threshold freezing, and paired design-aware stratified-PSU bootstrap inference across all 5 seeds (`20260828`..`20260832`).
  - Evaluated as apparent calibration-fit diagnostics on the 20% calibration partition ($n=581$). Results recorded in immutable run manifests and `runs/formal_panel26_summary.json`.
  - **Primary Discrimination**:
    - Unmitigated Baseline: Weighted AUROC $0.5564 \pm 0.0519$, Weighted AUPRC $0.0638 \pm 0.0245$.
    - Exploratory Survey-Weighted Centering Heuristic: Weighted AUROC $0.5580 \pm 0.0503$, Weighted AUPRC $0.0750 \pm 0.0409$.
    - Paired Differences: $\Delta \text{AUROC} = +0.0016 \pm 0.0084$, $\Delta \text{AUPRC} = +0.0112 \pm 0.0177$.
  - **Primary Fairness Endpoint & Cell Suppression**:
    - Evaluated under strict suppression rules ($n < 100 \lor \text{pos} < 20 \lor \text{neg} < 20 \lor n_{\text{eff}} < 50$).
    - **100% of race/ethnicity and sex subgroup cells on the evaluation partition were suppressed**. All subgroup max TPR disparities and primary fairness endpoints are strictly `None` (Not Estimable). Stratified PSU bootstrap propagates non-estimability without fabricating a scalar or confidence interval (`fairness_max_tpr_gap` status `NOT_ESTIMABLE_SUPPRESSED`), while retaining valid utility bootstrap inference.
- **Zero-Clobber & Execution Provenance**: All runs generated unique, collision-resistant directories under `runs/` with immutable pre-unlock manifests, mutable latest pointers, and append-only run logging.


---

## 2. Section 9 Standard Worker Report

Gate: Gate 12 — Fail-Closed Locked-Holdout Report & Stop Condition Audit
Status: COMPLETED_BY_GEMINI_BATCH_PENDING_FINAL_CODEX_REVIEW
Files changed:
- docs/reports/GATE_12_EVALUATION.md
- tests/test_gate12_evaluation.py
Commands executed:
- /Users/lkc/Downloads/code_v_0_3/.venv311/bin/python -m py_compile tests/test_gate12_evaluation.py
- PYTHONPATH=src /Users/lkc/Downloads/code_v_0_3/.venv311/bin/python -m unittest tests/test_gate12_evaluation.py -v
- git diff --check
Permissions requested: None
Tests executed:
- `test_holdout_remains_locked_under_underpowering_stop_condition`: Verified that Panel 27 holdout remains strictly locked because Panel 26 positive cases (136) $< 200$.
- `test_multi_seed_reproducibility_panel26`: Verified multi-seed execution determinism and hash invariance across runs.
- `test_subgroup_suppression_truth_on_panel26`: Verified that 100% of race/ethnicity and sex subgroups on the evaluation partition are suppressed.
- `test_no_clobber_manifest_and_pointer_behavior`: Verified immutable unique run directories and mutable root pointer behavior.
Exact test results:
- 4/4 tests passed cleanly in 20.702s.
Input hashes:
- Inherited baseline tag: inherited-code-v0.3-baseline-20260828 (commit 038897e9f751edac6e36445b7706eec5fdb15988)
- data/interim/meps/h244/h244.dta: 5cf983c94fd9ed8d8377c9ad27bebd905c545327eca823a8c8412bf4e66eaa70
Output hashes:
- tests/test_gate12_evaluation.py: updated test suite for Gate 12
- docs/reports/GATE_12_EVALUATION.md: self-referential report artifact
- runs/formal_panel26_summary.json: updated 5-seed formal execution summary
Row counts:
- Panel 26 Eligible Cohort: 2,882 individuals (136 positive events, 2,746 negative events).
- Panel 27 (HC-252): 8,292 structural rows (LOCKED under stop condition, zero outcome rows evaluated).
Assumptions:
- Prerequisite check failed closed upon detecting 136 positive cases (< 200), enforcing strict holdout protection.
Unresolved issues: None. Stop condition appropriately enforced without fabricating holdout results.
Git diff summary: 2 additive untracked files (`tests/test_gate12_evaluation.py`, `docs/reports/GATE_12_EVALUATION.md`). Zero inherited baseline files modified.
Proposed next step: Proceed to Gate 13 (Final Synthesis, Research Package & Conference Readiness Checklist).
STOP — waiting for Codex review.
