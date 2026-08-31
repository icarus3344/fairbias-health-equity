# Gate 9: Train-Only Preprocessing, Missing-Value Contracts & Design Quality Checks Report

## 1. Executive Summary

Gate 9 implements and verifies the train-only feature preprocessing pipeline, per-variable missing value contracts, and complex survey design quality checks on branch `research/meps-hc252-longitudinal`. Under authorized execution:
- **Strict Split Before Fitting**: All imputations (training medians), standardizations (training means and standard deviations), and categorical one-hot encodings are fit exclusively on the Training partition (or Train+Validation for frozen refitting).
- **Per-Variable Missing Contracts (Elimination of Global Negative Rule)**:
  1. *Legitimate Negative Values Preserved*: Continuous income and poverty variables (`TTLPY1X`, `FAMINCY1`, `POVLEVY1`) preserve legitimate negative values (e.g., -$300, -2.13% FPL) as valid observed data; only exact codebook sentinel `-1` / `-1.0` (and item non-response `-7, -8, -15`) is treated as missing.
  2. *Structural Zero Recoding*: For employment characteristics (`HOUR1`, `HOUR2`, `NUMEMP1`, `NUMEMP2`, `WAGEPY1X`), codebook-defined inapplicable unemployment (`-1`) is recoded to structural zero (`0.0`), not imputed with median worker values.
  3. *Prior-Round Inheritance*: Variables coded `-2` (DETERMINED IN PREVIOUS ROUND) for `HOUR2`, `NUMEMP2`, `CHOIC2`, `SELFCM2`, `UNION2` inherit their respective Round 1 values before fitting/transforming.
  4. *Categorical Inapplicable Dummies*: Categorical variables preserve codebook-defined inapplicable categories (`-1`) as valid distinct structural category dummies (`<col>__-1`), while true item non-response codes (`-7, -8, -15`) are excluded.
- **Unseen Level Handling**: Unseen categorical levels occurring out-of-sample in validation, calibration, or holdout partitions map cleanly to all-zero indicator representations without throwing runtime exceptions or mutating fitted dimensions.
- **Complex Survey Design Structure Verification**:
  - `LONGWT`: 100% positive, finite, non-zero across all eligible records (minimum: 2,405.0, maximum: 367,953.5, mean: 53,300.9).
  - `VARSTR`: 100% valid design strata across 105 unique strata.
  - `VARPSU`: 100% valid primary sampling units across 357 distinct clusters.
  - Kish Effective Sample Size: $n_{\text{eff}} = 1,785.7$ (far exceeding the minimum power threshold of 50.0).

---

## 2. Section 9 Standard Worker Report

Gate: Gate 9 — Train-Only Preprocessing, Missing-Value Contracts & Design Quality Checks
Status: COMPLETED_BY_GEMINI_BATCH_PENDING_FINAL_CODEX_REVIEW
Files changed:
- docs/reports/GATE_9_PREPROCESSING.md
- src/meps_fairness/data/preprocess.py
- tests/test_gate9_preprocessing.py
Commands executed:
- /Users/lkc/Downloads/code_v_0_3/.venv311/bin/python -m py_compile src/meps_fairness/data/preprocess.py tests/test_gate9_preprocessing.py
- PYTHONPATH=src /Users/lkc/Downloads/code_v_0_3/.venv311/bin/python -m unittest tests/test_gate9_preprocessing.py -v
- git diff --check
Permissions requested: None
Tests executed:
- `test_train_only_fit_and_zero_test_leakage`: Verified preprocessor fits only on train data and applies out-of-sample transforms without leaking statistics or failing on unseen categories.
- `test_missing_code_imputation_and_indicator`: Verified training median imputation and missing indicator generation.
- `test_legitimate_negative_income_preservation`: Verified TTLPY1X, FAMINCY1, POVLEVY1 preserve legitimate negative losses and only impute sentinel -1.
- `test_structural_zero_employment_handling`: Verified -1 inapplicable unemployment is recoded to structural zero (0.0).
- `test_prior_round_inheritance`: Verified prior-round inheritance for Round 2 variables coded -2.
- `test_categorical_structural_inapplicable_preservation`: Verified categorical -1 is preserved as distinct category dummy and -7/-8/-15 are dropped.
- `test_survey_design_quality_checks_on_panel26`: Verified positive weights, non-missing strata/PSU, and Kish effective sample size ($n_{\text{eff}} = 1,785.7$) on Panel 26.
Exact test results:
- 7/7 tests passed cleanly in 0.293s.
Input hashes:
- Inherited baseline tag: inherited-code-v0.3-baseline-20260828 (commit 038897e9f751edac6e36445b7706eec5fdb15988)
- src/meps_fairness/data/cohort.py: updated cohort module
- src/meps_fairness/data/split.py: updated split module
Output hashes:
- src/meps_fairness/data/preprocess.py: updated preprocessor with variable-specific missing contracts
- tests/test_gate9_preprocessing.py: updated test suite with comprehensive regression tests
- docs/reports/GATE_9_PREPROCESSING.md: self-referential report artifact
Row counts:
- Panel 26 Preprocessed Train Matrix: 1,732 rows x feature columns.
- Panel 26 Preprocessed Validation Matrix: 569 rows x feature columns.
- Panel 26 Preprocessed Calibration Matrix: 581 rows x feature columns.
Assumptions:
- Missing indicator columns are generated dynamically for continuous features exhibiting non-response codes on the training partition.
- Preprocessing transformation pipelines are refit on the combined Train+Validation (80%) partition during model refitting.
Unresolved issues: None.
Git diff summary: 2 additive untracked files (`src/meps_fairness/data/preprocess.py`, `tests/test_gate9_preprocessing.py`, `docs/reports/GATE_9_PREPROCESSING.md`). Zero inherited baseline files modified.
Proposed next step: Proceed to Gate 10 (Predictive Modeling, Exploratory Group-Aware Centering & Survey-Weighted Extension).
STOP — waiting for Codex review.
