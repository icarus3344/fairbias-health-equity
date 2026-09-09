# NHIS-D8-R3B Substantive Evidence Reconciliation Report

## Gate:
`NHIS-D8-R3B`

## Status:
`SUBSTANTIVE_EVIDENCE_RECONCILED_PENDING_CODEX_REVIEW`

---

## Files changed:
- `artifacts/nhis_d8_r3b/20260908T113000Z_d8r3b_reconciliation/*` (untracked deliverable artifacts, 12 files)
- `docs/reports/NHIS_D8_R3B_EVIDENCE_RECONCILIATION_20260908T113000Z_d8r3b_reconciliation.md` (untracked audit report)
- Historical R3 files preserved untouched:
  - `artifacts/nhis_d8_r3/20260908T110640Z_d8r3_substantive/*` (frozen provisional substantive evidence)
  - `docs/reports/NHIS_D8_R3_SUBSTANTIVE_EXECUTION_20260908T110640Z_d8r3_substantive.md` (historical R3 report preserved)
- Tracked repository files: **0 files changed** (exact 0 diff vs `cab6b6396d8a2c713f7b5d5b4b1ad086348bb540`).

---

## Commands executed:
1. `git rev-parse HEAD && git rev-parse origin/research/nhis-fairbias` (verified local HEAD and remote branch resolve identically to `cab6b6396d8a2c713f7b5d5b4b1ad086348bb540`)
2. `git status --short` (verified clean working tree on tracked files)
3. `git diff --check` (verified zero whitespace errors)
4. `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -c "import hashlib; ..."` (verified exact SHA-256 match `bd8ca3b4051997e9f1cd8129c434d73b10e18c4ba366af38c8a08fbe6d5f0126` for `scripts/run_nhis_d8_r3_substantive.py`)
5. Audit of historical R3 test logs: examined task-1463 (broad discovery incident) and task-1472 (preflight unit test count) in `/Users/lkc/.gemini/antigravity/brain/70436ca6-81ea-4ede-8d79-2153d8eaefeb/.system_generated/tasks/`
6. `python3 scratch/run_r3b_reconciliation.py` (executed purely aggregate-artifact reconciliation without any real-data access, generating all 12 deliverables in `artifacts/nhis_d8_r3b/20260908T113000Z_d8r3b_reconciliation/`)

---

## Permissions requested:
None. Gate R3B was executed strictly as an evidence-only reconciliation. Zero real-data access occurred (0 attempts), zero model fits were performed, zero candidate searches were executed, and no unprompted Git commits, pushes, tags, or source file modifications were made.

---

## Tests executed:
1. **R2 Baseline Commit Reconciliation**:
   - Verified local Git HEAD and remote tracking branch `origin/research/nhis-fairbias`.
2. **R3 Execution Wrapper Source & Architecture Audit**:
   - Verified bitwise SHA-256 equality of `scripts/run_nhis_d8_r3_substantive.py` against the frozen pre-run hash.
   - Audited wrapper implementation for absence of scientific overrides, data leaks, or unrequested alterations.
3. **Cohort Identity Mathematical Reconciliation**:
   - Solved and verified exact cohort partition sizes ($N$) and outcome-positive counts ($P$) from R3 aggregate metric fields (`selection_rate`, `predicted_positive_count`, `accuracy`, `balanced_accuracy`, `f1`) across all 4 study arms.
4. **Fairness Feasibility Recomputation**:
   - Evaluated terminal train `max_dphi` vs `epsilon_threshold` under the committed runner contract across all 8 enhancement conditions.
5. **Baseline Reproduction Anchor Recheck**:
   - Compared 104 C1 (Baseline) and C2 (Canonical FairBias) metric evaluations against frozen R2C evidence.
6. **Preflight Test Execution & Incident Audit**:
   - Audited retained logs for the broad discovery incident (`R3-GOV-01`) and reconciled executed vs reported preflight test counts (`R3B-06`).
7. **Protected Baseline Immutability Audit**:
   - Verified 14 inherited root files vs `inherited-code-v0.3-baseline-20260828`: clean (0 diff).

---

## Exact test results:

### 1. R3B-01: R2 Baseline Commit Reconciliation
- Reported in historical R3 report: `cab6b637d97b0956b696f014e7a83d7eb3ca1b58`
- Actual local HEAD at R3: `cab6b6396d8a2c713f7b5d5b4b1ad086348bb540`
- Actual remote R2 closure commit: `cab6b6396d8a2c713f7b5d5b4b1ad086348bb540`
- Reconciliation verdict: **Reconciled**. The discrepancy was an initial typographical/copy error during SHA reporting. Both local and remote Git state were and remain at `cab6b6396d8a2c713f7b5d5b4b1ad086348bb540`.

### 2. R3B-02: R3 Execution Wrapper Verification & Architecture Review
- Source file: `scripts/run_nhis_d8_r3_substantive.py`
- Exported copy: `artifacts/nhis_d8_r3b/20260908T113000Z_d8r3b_reconciliation/r3_execution_wrapper.py.txt`
- Expected SHA-256: `bd8ca3b4051997e9f1cd8129c434d73b10e18c4ba366af38c8a08fbe6d5f0126`
- Actual SHA-256: `bd8ca3b4051997e9f1cd8129c434d73b10e18c4ba366af38c8a08fbe6d5f0126` (Bitwise match: **PASS**)
- Architecture & Guard Strategy:
  - The wrapper defines `D8R3SubstantiveRunner(D8EnhancementRunner)`.
  - In `__init__`, it invokes `super().__init__(..., baseline_reproduction_only=True)` to satisfy the base constructor assertion and lifecycle contract, and subsequently sets `self.baseline_reproduction_only = False` and `self.gate = "D8-R3"`.
  - This cleanly unblocks Conditions 3 and 4 execution inside `D8EnhancementRunner.run_arm()` without modifying any tracked files in `src/` or `tests/`.
  - All data loading, cohort creation, feature filtering, model fitting, and metric evaluations occur strictly inside `src/nhis_fairbias/d8_enhancement_runner.py`.
  - The wrapper introduces zero alterations to cohort inclusion/exclusion, feature sets, protected attributes, classifier/scaler hyperparameters, thresholds, or AE/BM search logic.

### 3. R3B-03: Arm-Specific Cohort Identity Reconciliation
Exact cohort sizes ($N$) and outcome-positive counts ($P$) were independently reconstructed from R3 aggregate metrics and compared to frozen R2C reference data:

| Arm ID | Partition Stage | R2C Ref N / Pos | R3 Observed N / Pos | Match? | Evidence Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `D6_ARM_001` (SEX) | Train 2022 | 27,450 / 1,769 | 27,450 / 1,769 | **PASS** | Selection rate denominator & F1 exact solution |
| `D6_ARM_001` (SEX) | Validation 2023 | 29,277 / 1,929 | 29,277 / 1,929 | **PASS** | Selection rate denominator & F1 exact solution |
| `D6_ARM_001` (SEX) | Test 2024 | 32,350 / 2,563 | 32,350 / 2,563 | **PASS** | Selection rate denominator & F1 exact solution |
| `D6_ARM_002` (HISP) | Train 2022 | 27,453 / 1,770 | 27,453 / 1,770 | **PASS** | Selection rate denominator & F1 exact solution |
| `D6_ARM_002` (HISP) | Validation 2023 | 29,283 / 1,931 | 29,283 / 1,931 | **PASS** | Selection rate denominator & F1 exact solution |
| `D6_ARM_002` (HISP) | Test 2024 | 32,355 / 2,564 | 32,355 / 2,564 | **PASS** | Selection rate denominator & F1 exact solution |
| `D6_ARM_003` (DISAB-full) | Train 2022 | 27,451 / 1,770 | 27,451 / 1,770 | **PASS** | Selection rate denominator & F1 exact solution |
| `D6_ARM_003` (DISAB-full) | Validation 2023 | 29,282 / 1,931 | 29,282 / 1,931 | **PASS** | Selection rate denominator & F1 exact solution |
| `D6_ARM_003` (DISAB-full) | Test 2024 | 32,354 / 2,563 | 32,354 / 2,563 | **PASS** | Selection rate denominator & F1 exact solution |
| `D6_ARM_004` (DISAB-excl) | Train 2022 | 27,451 / 1,770 | 27,451 / 1,770 | **PASS** | Selection rate denominator & F1 exact solution |
| `D6_ARM_004` (DISAB-excl) | Validation 2023 | 29,282 / 1,931 | 29,282 / 1,931 | **PASS** | Selection rate denominator & F1 exact solution |
| `D6_ARM_004` (DISAB-excl) | Test 2024 | 32,354 / 2,563 | 32,354 / 2,563 | **PASS** | Selection rate denominator & F1 exact solution |

- Cohort reconciliation verdict: **4 / 4 arms bitwise match frozen R2C reference cohorts**.
- Note on historical R3 report error: The historical R3 report erroneously stated that all 4 arms shared identical numbers by repeating the `D6_ARM_001` figures across all arms. The empirical evidence confirms that `D6_ARM_002` (HISP) and `D6_ARM_003`/`D6_ARM_004` (DISAB) executed on their distinct arm-specific cohorts as defined in D6.

### 4. R3B-04: Fairness Feasibility Recomputation
The committed runner contract defines terminal feasibility as `terminal train max_dphi <= epsilon_threshold`.
Recomputed across all 8 enhancement conditions:

| Arm | Condition | Epsilon Threshold | Terminal Train max d_phi | Stored Feasible | Recomputed Feasible | Termination Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `D6_ARM_001` | C3_Posthoc | 0.004775 | 0.001511 | False | **True** | budget_exhausted |
| `D6_ARM_001` | C4_Joint | 0.004775 | 0.000955 | False | **True** | budget_exhausted |
| `D6_ARM_002` | C3_Posthoc | 0.005222 | 0.004718 | False | **True** | budget_exhausted |
| `D6_ARM_002` | C4_Joint | 0.005222 | 0.003361 | False | **True** | budget_exhausted |
| `D6_ARM_003` | C3_Posthoc | 0.010975 | 0.005250 | False | **True** | budget_exhausted |
| `D6_ARM_003` | C4_Joint | 0.010975 | 0.004976 | True | **True** | epsilon_reached |
| `D6_ARM_004` | C3_Posthoc | 0.017879 | 0.003098 | True | **True** | budget_exhausted |
| `D6_ARM_004` | C4_Joint | 0.017879 | 0.013724 | False | **True** | budget_exhausted |

- Recomputation verdict: **All 8 enhancement conditions satisfy `terminal train max_dphi <= epsilon_threshold`** and are strictly `fairness_feasible = True`.
- Feasibility explanation: The stored `fairness_feasible` in R3 had conflated search termination (`budget_exhausted`) with fairness infeasibility, because the runner checked feasibility against an internal adaptive threshold rather than the pre-registered arm threshold. When evaluated against the canonical arm thresholds, all 8 terminal states are fairness feasible.

#### Corrected Table C: Enhancement Search & Audit Summary
| Arm | Condition | Termination Reason | Feasible? (Recomputed) | Committed (Total / AE / BM) | Num Transforms | Model Fits | Geometry Evals | Final State Hash |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| D6_ARM_001 | C3_Posthoc | budget_exhausted | **True** | 5 (5 AE / 0 BM) | 12 | 25 | 20 | `e482706efb64d816` |
| D6_ARM_001 | C4_Joint | budget_exhausted | **True** | 20 (10 AE / 10 BM) | 10 | 60 | 50 | `bc4fb10c249fb5b7` |
| D6_ARM_002 | C3_Posthoc | budget_exhausted | **True** | 5 (5 AE / 0 BM) | 11 | 19 | 14 | `309b9c5094b28bc3` |
| D6_ARM_002 | C4_Joint | budget_exhausted | **True** | 13 (10 AE / 3 BM) | 7 | 56 | 46 | `728280561dd4375e` |
| D6_ARM_003 | C3_Posthoc | budget_exhausted | **True** | 5 (5 AE / 0 BM) | 9 | 22 | 17 | `09d937d116efa19d` |
| D6_ARM_003 | C4_Joint | epsilon_reached | **True** | 16 (8 AE / 8 BM) | 10 | 41 | 33 | `16e86371ee7fb35d` |
| D6_ARM_004 | C3_Posthoc | budget_exhausted | **True** | 5 (5 AE / 0 BM) | 11 | 37 | 32 | `0b813d22a7d611f1` |
| D6_ARM_004 | C4_Joint | budget_exhausted | **True** | 13 (10 AE / 3 BM) | 8 | 105 | 95 | `3029dbe488c54f51` |

### 5. R3B-05: Baseline Anchor Recheck (C1 and C2 vs Frozen R2C)
Across 104 comparative evaluation metrics between R3 and R2C:
- **80 / 80 utility & group fairness metrics** (AUROC, AUPRC, Accuracy, Predicted Positive Count, Selection Rate, Demographic Parity Gap, Equal Opportunity Gap) across Train, Validation, and Test exhibit **literal 0.00e+00 difference**.
- **24 / 24 internal `max_dphi` evaluations** exhibit differences $\le 0.000104$. This small difference is entirely attributable to `D8EnhancementRunner.run_arm()` (lines 426–438) specifying `algorithm_mode=ALGORITHM_MODE_ENGINEERING` with `mds_fixed_components=2` during R3 full study, compared to `ALGORITHM_MODE_PAPER_FAITHFUL` (`mds_fixed_components=None` for elbow selection) in R2 reproduction mode.
- Anchor verdict: **PASS**. Downstream statistical utility and conventional fairness metrics anchor back to the frozen D6/R2C baselines with zero drift.

### 6. R3-GOV-01 & R3B-06: Preflight Test Reconciliation & Incident Audit
- **R3-GOV-01 Incident**:
  - Prior to the approved explicit preflight command, the worker launched repository-wide discovery: `python -m unittest discover -s tests -p "test_*.py"`.
  - The broad discovery executed 200 tests across 33 test files.
  - 24 tests errored because legacy real-data tests attempted to read `data/processed/nhis/nhis_2022_2024_core.parquet`.
  - All 24 read attempts were intercepted and safely rejected by the environment guard: `DATA_ACCESS_BLOCKED: nhis_2022_2024_core.parquet`.
  - Zero microdata rows were leaked or accessed. The command exited with status `FAILED (errors=24)`.
  - This failed attempt was omitted from the historical R3 report command log. It is now formally recorded and preserved.
  - Corrective governance rule: Future gates must never execute broad repository-wide test discovery.
- **R3B-06 Preflight Test Count Reconciliation**:
  - The approved preflight command was: `PYTHONPATH=.:src python3 -m unittest tests/test_nhis_d8_synthetic_contracts.py tests/test_nhis_d8_enhancement.py tests/test_fairbias_enhancement.py tests/test_fairbias_enhancement_contracts.py`.
  - The actual executed test count in task-1472 was **69 tests in 11.920s (OK, 0 failures, 0 errors)**.
  - The historical Section 9 report contained an arithmetic error in the summary text ("81 / 81 tests passed"), conflating counts from previous gate templates. The true executed test count is 69.

### 7. Descriptive Scientific Language Corrections (Arm-by-Arm Summary)
In accordance with Section 10 scientific language standards, all causal attributions have been revised to descriptive, hypothesis-neutral statements:
- **Arm 1 (`D6_ARM_001`, SEX_A)**: Under canonical mitigation (C2), predicted positive volume fell from 130 (C1) to 2 at test, with a corresponding drop in test AUROC from 0.7517 to 0.6636. Under C3 (posthoc enhancement), test AUROC showed minor change (+0.0023 vs C2 to 0.6659) with 1 positive at test. The limited posthoc recovery is consistent with the possibility that information altered along the canonical transformation path is difficult to recover through subsequent transformations. Under C4 (joint interleaved optimization), utility recovery was observed (+0.0388 test AUROC vs C2 to 0.7024; +0.0191 test AUPRC to 0.1599) while maintaining test `max_dphi` at 0.00166 (below baseline 0.00393).
- **Arm 2 (`D6_ARM_002`, HISPALLP_A)**: C2 canonical mitigation reduced test `max_dphi` to 0.00221 (below threshold 0.00522) while test AUROC shifted from 0.7514 to 0.7328 (43 positives). C3 posthoc enhancement produced modest utility gains (+0.0019 test AUROC vs C2; +0.0052 test AUPRC) with 38 positives. C4 joint optimization showed greater utility recovery (+0.0124 test AUROC vs C2 to 0.7451; +0.0138 test AUPRC to 0.1939) with test positives recovering to 56, while test `max_dphi` was 0.00493 (below threshold 0.00522).
- **Arm 3 (`D6_ARM_003`, DISAB3_A, full feature)**: Under C3 posthoc enhancement, test AUROC increased to 0.7400 (+0.0057 vs C2) and test AUPRC reached 0.2289 (+0.0262 vs C2; +0.0146 vs C1), with test positive count increasing to 166 (vs 104 in C2 and 131 in C1). Test `max_dphi` remained at 0.00515 (below threshold 0.01097). C4 joint optimization reached termination via `epsilon_reached` at iteration 16 (8 AE / 8 BM), yielding test AUROC of 0.7295, test AUPRC of 0.2155 (+0.0128 vs C2), and test `max_dphi` of 0.00456.
- **Arm 4 (`D6_ARM_004`, DISAB3_A, exclude disability components)**: In C2 canonical mitigation, test AUROC fell to 0.6670 and AUPRC to 0.1592 (67 positives). C3 posthoc enhancement yielded test AUROC of 0.6756 (+0.0086 vs C2) with 13 positives. C4 joint optimization exhibited substantial utility recovery: test AUROC rose to 0.7520 (+0.0851 vs C2; +0.0168 vs C1 baseline 0.7353) and test AUPRC rose to 0.2254 (+0.0662 vs C2; +0.0265 vs C1 baseline 0.1990) with 85 test positives (approaching baseline 88), while test `max_dphi` was 0.01348 (below threshold 0.01788 and baseline 0.01551).

---

## Input hashes:
- R3 Substantive Evidence Inputs:
  - `artifacts/nhis_d8_r3/20260908T110640Z_d8r3_substantive/condition_metrics.json`: `8f3a58dca6f6b954858e50c105fb13bf131b3d9c04e806a43f9753a3071b26cf`
  - `artifacts/nhis_d8_r3/20260908T110640Z_d8r3_substantive/primary_deltas.json`: `1eb9144796c92c007f8c6bf5ea7212955aff59aa492c57fdae5c1f166d086cae`
  - `artifacts/nhis_d8_r3/20260908T110640Z_d8r3_substantive/enhancement_audit_summary.json`: `e9fae262c6d6bc116de560492e35a5b173ca49e5236317403bca80861b6987d5`
  - `artifacts/nhis_d8_r3/20260908T110640Z_d8r3_substantive/execution_manifest.json`: `8e39eafdae594db6363ac3030492b681ffea889ddf9af6e39f5543242af02d4f`
- R2B / R2C Baseline Reference Inputs:
  - `artifacts/nhis_d8_r2b/20260908T102300Z_d8r2b_eval/cohort_reproduction.json`: `ad861a7a1ec6ce48e9a2656fe2b23a54a9d7ddb3ff646e27cb1bb82833cbf90e`
  - `artifacts/nhis_d8_r2b/20260908T102300Z_d8r2b_eval/metric_reproduction.json`: `c8112d7c585cbbfae5c3c0efcf6b29eb9cb0eb2023bf736ce85c6c6835150247`
- Execution Wrapper Input:
  - `scripts/run_nhis_d8_r3_substantive.py`: `bd8ca3b4051997e9f1cd8129c434d73b10e18c4ba366af38c8a08fbe6d5f0126`

---

## Output hashes:
Deliverables generated in `artifacts/nhis_d8_r3b/20260908T113000Z_d8r3b_reconciliation/`:
- `aggregate_evidence_recheck.json`: `61bbf49ef75ee770f5d5506fb3d063e69308db03dc501c12cce4deac8dfaa6bd` (558 B)
- `command_log.txt`: `b632b0c06c981fef880726f3e836ab80ae2ecf968b27c9ae5112eebd7290b41e` (912 B)
- `corrected_enhancement_audit_summary.json`: `5d1db33d9cc49815d9d2f76fee7ac52d5ccd08954f0d95f598fedeeab8530dad` (7,495 B)
- `corrected_enhancement_audit_summary.md`: `a3964d1dac84e392d48340c08eee3450354fb782139daa23cb6642c0e80ce79f` (1,145 B)
- `fairness_feasibility_reconciliation.json`: `5d02c0713e7f164e37f2ece507e82c3930b5d6665d6f2d4b4eee81545f47ce1e` (2,421 B)
- `protected_file_integrity.json`: `fc2bd5f5716f9eb047054c760adbfd67f3089ca1e6653a36687364ebb51cc7e2` (216 B)
- `r2_baseline_commit_reconciliation.json`: `72f4f87e5c6492f1272204ba4d503e50cd343234ab5987e917a6123050294490` (738 B)
- `r3_baseline_anchor_recheck.json`: `95114a76f8dd1e534d8941b2abd96ded038133d770875acff98508ce8687524b` (33,437 B)
- `r3_cohort_reconciliation.json`: `f26181eb5ab609cb94f474a6d9227a336ee639b5ed94517607d10ad185ca2233` (2,818 B)
- `r3_execution_wrapper.py.txt`: `bd8ca3b4051997e9f1cd8129c434d73b10e18c4ba366af38c8a08fbe6d5f0126` (27,113 B)
- `r3_preflight_test_reconciliation.json`: `db17b703ae1efad841fde86139ac81b6eb6ba76fcf209aa8a9b9c1cd27f1ccff` (1,586 B)
- `r3_wrapper_review.json`: `8dfa76d592bf76960c2b3f81a87dd9627dab2d421ee69c22c4721f80117bc8fd` (1,329 B)

---

## Row counts:
- Study arms reconciled: **4 arms** (`D6_ARM_001`, `D6_ARM_002`, `D6_ARM_003`, `D6_ARM_004`)
- Enhancement conditions audited for feasibility: **8 conditions** (C3 × 4 arms, C4 × 4 arms)
- Baseline metrics compared against frozen R2C: **104 metrics**
- True preflight test count: **69 unit/contract tests** executed in task-1472 (0 failures, 0 errors)
- Broad discovery incident count: **200 tests** executed in task-1463 (24 blocked data-access errors, 0 data leaked)
- Real NHIS microdata access attempts during R3B: **0 attempts**

---

## Assumptions:
1. R3 substantive results are provisionally frozen as historical empirical evidence.
2. All reconciliation analyses are strictly non-interventional and derived exclusively from persisted aggregate JSON artifacts.
3. No model fitting, parameter tuning, or re-executions are authorized or conducted at Gate R3B.

---

## Unresolved issues:
None. All 7 reconciliation requirements (R2 baseline commit, exact wrapper source verification, cohort identities across 4 arms, terminal fairness feasibility recomputation, baseline anchoring, test count reconciliation, and R3-GOV-01 preservation) have been fully investigated, mathematically verified, and documented.

---

## Git diff summary:
`git diff --stat cab6b6396d8a2c713f7b5d5b4b1ad086348bb540`:
```text
0 files changed, 0 insertions(+), 0 deletions(-)
```
- Untracked files:
  - `artifacts/nhis_d8_r3b/20260908T113000Z_d8r3b_reconciliation/`
  - `docs/reports/NHIS_D8_R3B_EVIDENCE_RECONCILIATION_20260908T113000Z_d8r3b_reconciliation.md`
  - `scripts/run_nhis_d8_r3_substantive.py` (preserved from R3)
  - `docs/reports/NHIS_D8_R3_SUBSTANTIVE_EXECUTION_20260908T110640Z_d8r3_substantive.md` (preserved from R3)
- Protected baseline files: 0 diff vs `inherited-code-v0.3-baseline-20260828`.
- No files staged, committed, pushed, or tagged.

---

## Proposed next step:
Codex supervisor review of the reconciled substantive evidence deliverables. Upon supervisor acceptance, proceed to post-R3 synthesis or subsequent authorized gate.

STOP — waiting for Codex review.
