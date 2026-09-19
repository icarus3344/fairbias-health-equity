# AUTOMATED STATIC AUDIT TABLES: Benchmark Method Matrix, Cohort Counts, Directional Discrepancies, and Test Inventory

- **Date**: 2026-09-14
- **Manifest Reference**: `docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/AUTOMATED_STATIC_MANIFEST.json`
- **Data Source**: Direct static AST parsing of `src/nhis_fairbias/benchmark/` and JSON parsing of `runs/sequential_full_benchmark_20260914_103657Z/integrated_full_benchmark_summary.json`.
- **Constraint**: Zero project imports, zero pytest execution, zero model execution, zero raw individual microdata reading.

---

## 1. Method Capabilities, Hyperparameters, Training Policies, and Output Contracts

| Method ID | Upstream / Literature Reference | Underlying Model Backbone | Training Data & Component Policy | Hyperparameter Configuration (Actual Run) | Output Type Contract | Observed Execution Status | Known Protocol Defects |
|---|---|---|---|---|---|---|---|
| **UNMITIGATED** | Standard Predictive Baseline | `LogisticRegression(C=1.0, max_iter=1000)` | Partition F ($N_F$) | $C=1.0$, `solver=lbfgs`, unweighted | `predict_proba[:, 1]` treated as decision probability $q$ | `VALID` across all 4 arms | Conflates risk probability $p$ with decision probability $q$; no Set C thresholding |
| **FAIRBIAS_BM** | Tang et al. (2024) / Application Adapter | `LogisticRegression(C=1.0, max_iter=1000)` | Partition F ($N_F$) + pre-computed D6 dictionaries | Static D6 release rules (`frozen_changed_dict.json`) | `predict_proba[:, 1]` treated as decision probability $q$ | `VALID` across all 4 arms | Bypasses BM learning on F; loads 2022 full-year D6 rules; array fallback fits raw $X$; conflates $p$ and $q$ |
| **REWEIGHING** | Kamiran & Calders (2012) / AIF360 | `LogisticRegression(C=1.0, max_iter=1000)` with fairness weights | Partition F ($N_F$) | Unit base weights, multigroup probability ratio weighting | `predict_proba[:, 1]` treated as decision probability $q$ | `VALID` across all 4 arms | Conflates $p$ and $q$; no Set C thresholding |
| **LFR_RECONSTRUCTED** | Zemel et al. (2013) / AIF360 | Latent representation + downstream `LogisticRegression` | Subsample of 2,000 rows on F for representation; full F for LR | $k=5, A_z=50, A_x=0.01, A_y=1.0$, `maxiter=50`, `maxfun=100` | `predict_proba[:, 1]` treated as decision probability $q$ | `VALID` on Arms 001, 003, 004; `NOT_SUPPORTED` on Arm 002 | Truncated representation learning budget; multigroup not supported on Arm 002 |
| **EG_DP** | Agarwal et al. (ICML 2018) / Fairlearn | Sequence of `LogisticRegression(C=1.0)` | Partition F ($N_F$) | `DemographicParity`, $eps=0.05, max\_iter=15, C=1.0$ | Randomized policy positive probability $q \in [0, 1]$ | `VALID` across all 4 arms | `sample_weight` accepted in signature but silently dropped; truncated iteration budget |
| **EG_EO** | Agarwal et al. (ICML 2018) / Fairlearn | Sequence of `LogisticRegression(C=1.0)` | Partition F ($N_F$) | `EqualizedOdds`, $eps=0.05, max\_iter=15, C=1.0$ | Randomized policy positive probability $q \in [0, 1]$ | `VALID` across all 4 arms | `sample_weight` accepted in signature but silently dropped; truncated iteration budget |
| **TO_EO** | Hardt et al. (NeurIPS 2016) / Fairlearn | Base `LogisticRegression` on F + Postprocessor on C | Base on F ($N_F$); Threshold calibration on C ($N_C$) | `EqualizedOdds`, objective=`balanced_accuracy_score`, `prefit=True` | Randomized policy positive probability $q \in [0, 1]$ | `VALID` across all 4 arms | `fit()` signature mismatch (`sample_weight` vs `sample_weight_F`) raises TypeError; unified fit reuses same set |

---

## 2. Partition Sample Sizes ($N_F, N_C, N_S, N_T$) Across Experimental Arms

| Arm Identifier | Protected Attribute | Partition F (2022 Fitting) | Partition C (2022 Calibration) | Partition S (2023 Selection) | Partition T (2024 Evaluation) | Reported 3-Year Pooled Total | Test Set Evaluation Size ($N_T$) |
|---|---|---|---|---|---|---|---|
| **Arm 001** | `SEX_A` (2 groups: Male, Female) | 21,869 | 5,581 | 29,277 | 32,350 | 89,077 | **32,350** |
| **Arm 002** | `HISPALLP_A` (7 race/ethnicity groups) | 21,872 | 5,581 | 29,283 | 32,355 | 89,091 | **32,355** |
| **Arm 003** | `DISAB3_A` (2 groups; includes 6 impairment features) | 21,871 | 5,580 | 29,282 | 32,354 | 89,087 | **32,354** |
| **Arm 004** | `DISAB3_A` (2 groups; excludes 6 impairment features) | 21,871 | 5,580 | 29,282 | 32,354 | 89,087 | **32,354** |

> **Audit Note**: The 89,091 count previously presented in summary headings is the aggregate across all three years (2022, 2023, 2024). The true out-of-year evaluation cohort evaluated in Set T consists of 32,350 to 32,355 individuals.

---

## 3. Directional Metric Discrepancies and Point-Dominance Audit

### 3.1. Performance and Fairness Comparison: FairBias vs. Baseline and Reweighing

| Arm | Method | Balanced Accuracy ($BA$) | Equalized Odds Gap ($EO$) | Demographic Parity Gap ($DP$) | $\Delta BA$ vs. Baseline | $\Delta EO$ vs. Baseline | $\Delta BA$ vs. Reweighing | $\Delta EO$ vs. Reweighing | Reweighing Dominates FairBias? |
|---|---|---|---|---|---|---|---|---|---|
| **Arm 001** | UNMITIGATED | 0.54807 | 0.00322 | 0.00117 | Ref | Ref | -0.00013 | -0.00151 | — |
| | **FAIRBIAS_BM** | 0.52335 | 0.01530 | 0.00541 | **-0.02472** | **+0.01208** (+375.0%) | **-0.02486** | **+0.01057** | **YES (RW higher BA, lower EO)** |
| | REWEIGHING | 0.54820 | 0.00473 | 0.00195 | +0.00013 | +0.00151 | Ref | Ref | — |
| **Arm 002** | UNMITIGATED | 0.54803 | 0.10851 | 0.05779 | Ref | Ref | +0.00198 | +0.01406 | — |
| | **FAIRBIAS_BM** | 0.53208 | 0.09498 | 0.04864 | **-0.01595** | **-0.01354** (-12.5%) | **-0.01397** | **+0.00053** | **YES (RW higher BA, lower EO)** |
| | REWEIGHING | 0.54605 | 0.09445 | 0.04960 | -0.00198 | -0.01406 | Ref | Ref | — |
| **Arm 003** | UNMITIGATED | 0.54816 | 0.06202 | 0.04872 | Ref | Ref | +0.00198 | +0.05548 | — |
| | **FAIRBIAS_BM** | 0.54386 | 0.02666 | 0.03180 | **-0.00429** | **-0.03536** (-57.0%) | **-0.00231** | **+0.02012** | **YES (RW higher BA, lower EO)** |
| | REWEIGHING | 0.54618 | 0.00654 | 0.01168 | -0.00198 | -0.05548 | Ref | Ref | — |
| **Arm 004** | UNMITIGATED | 0.54010 | 0.01349 | 0.01722 | Ref | Ref | +0.00071 | +0.00913 | — |
| | **FAIRBIAS_BM** | 0.52954 | 0.02425 | 0.00373 | **-0.01055** | **+0.01076** (+79.8%) | **-0.00984** | **+0.01988** | **YES (RW higher BA, lower EO)** |
| | REWEIGHING | 0.53939 | 0.00436 | 0.00880 | -0.00071 | -0.00913 | Ref | Ref | — |

> **Key Takeaway**: Reweighing strictly point-dominates FairBias on both balanced accuracy and equalized odds gap across all four experimental arms. The claim that FairBias is "globally Pareto optimal" is empirically false on this benchmark run.

### 3.2. Paired Contrast Confidence Intervals and P-Values from Existing Summary JSON

| Arm | Contrast Pair | Metric | Point Difference | 95% Confidence Interval | Two-Sided p-value | Degrees of Freedom ($df$) |
|---|---|---|---|---|---|---|
| **Arm 001** | FairBias vs. Baseline | Balanced Accuracy | -0.02472 | [-0.02791, -0.02153] | 0.0000 | Omitted in JSON ($B=30$) |
| | FairBias vs. Baseline | Equalized Odds | +0.01208 | [-0.00072, +0.02488] | 0.0644 | Omitted in JSON ($B=30$) |
| | FairBias vs. Reweighing | Balanced Accuracy | -0.02486 | [-0.02810, -0.02161] | 0.0000 | Omitted in JSON ($B=30$) |
| | FairBias vs. Reweighing | Equalized Odds | +0.01057 | [-0.00365, +0.02479] | 0.1449 | Omitted in JSON ($B=30$) |
| **Arm 002** | FairBias vs. Baseline | Balanced Accuracy | -0.01595 | [-0.01858, -0.01332] | 0.0000 | Omitted in JSON ($B=30$) |
| | FairBias vs. Baseline | Equalized Odds | -0.01354 | [-0.08469, +0.05762] | 0.7088 | Omitted in JSON ($B=30$) |
| | FairBias vs. Reweighing | Balanced Accuracy | -0.01397 | [-0.01637, -0.01157] | 0.0000 | Omitted in JSON ($B=30$) |
| | FairBias vs. Reweighing | Equalized Odds | +0.00053 | [-0.07027, +0.07132] | 0.9883 | Omitted in JSON ($B=30$) |
| **Arm 003** | FairBias vs. Baseline | Balanced Accuracy | -0.00429 | [-0.00516, -0.00342] | 0.0000 | Omitted in JSON ($B=30$) |
| | FairBias vs. Baseline | Equalized Odds | -0.03536 | [-0.05415, -0.01657] | 0.0002 | Omitted in JSON ($B=30$) |
| | FairBias vs. Reweighing | Balanced Accuracy | -0.00231 | [-0.00299, -0.00164] | 3.38e-11 | Omitted in JSON ($B=30$) |
| | FairBias vs. Reweighing | Equalized Odds | +0.02012 | [+0.00805, +0.03220] | 0.0011 | Omitted in JSON ($B=30$) |
| **Arm 004** | FairBias vs. Baseline | Balanced Accuracy | -0.01055 | [-0.01224, -0.00886] | 0.0000 | Omitted in JSON ($B=30$) |
| | FairBias vs. Baseline | Equalized Odds | +0.01076 | [-0.01078, +0.03229] | 0.3270 | Omitted in JSON ($B=30$) |
| | FairBias vs. Reweighing | Balanced Accuracy | -0.00984 | [-0.01140, -0.00829] | 0.0000 | Omitted in JSON ($B=30$) |
| | FairBias vs. Reweighing | Equalized Odds | +0.01988 | [+0.00544, +0.03433] | 0.0070 | Omitted in JSON ($B=30$) |

---

## 4. AST Test Function Inventory (19 Test Functions)

| Test File Path | Function Name | Start Line | End Line | Purpose / Coverage Target | Identified Gap or Defect |
|---|---|---|---|---|---|
| `tests/benchmark/test_adapters.py` | `test_unmitigated_adapter` | 35 | 43 | Validates `UnmitigatedAdapter` fit/predict | Only checks shape and binary outputs; does not verify probability semantics |
| `tests/benchmark/test_adapters.py` | `test_reweighing_adapter` | 45 | 60 | Validates `ReweighingAdapter` fit/predict | Checks non-negative weights; does not verify multi-group weight derivation |
| `tests/benchmark/test_adapters.py` | `test_lfr_adapter_binary` | 62 | 69 | Validates binary `LFRAdapter` | Uses synthetic array; does not check 2,000-row subsampling truncation |
| `tests/benchmark/test_adapters.py` | `test_lfr_adapter_multigroup_not_supported` | 71 | 76 | Checks `NotSupportedError` on multi-group | Confirms Arm 002 failure; does not test contract enforcement |
| `tests/benchmark/test_adapters.py` | `test_fairlearn_eg_dp_adapter` | 78 | 85 | Validates Fairlearn `EG_DP` adapter | Does not test `sample_weight` handling (silent drop undetected) |
| `tests/benchmark/test_adapters.py` | `test_fairlearn_eg_eo_adapter` | 87 | 94 | Validates Fairlearn `EG_EO` adapter | Does not test `sample_weight` handling (silent drop undetected) |
| `tests/benchmark/test_adapters.py` | `test_fairlearn_threshold_optimizer` | 96 | 104 | Validates `ThresholdOptimizerAdapter` | Directly calls `fit_base()` and `calibrate()`; bypasses unified `fit()` with signature bug |
| `tests/benchmark/test_adapters.py` | `test_fairbias_adapter` | 106 | 119 | Validates generic `FairBiasAdapter` array path | Tests synthetic array fallback which fits raw $X$ without applying transformations |
| `tests/benchmark/test_adapters.py` | `test_fairbias_adapter_semantic_arms` | 121 | 145 | Validates `FairBiasAdapter` on Arm 003 | Asserts non-empty dictionary, which passed by reading pre-computed D6 disk release |
| `tests/benchmark/test_data_contracts.py` | `test_make_record_key` | 15 | 21 | Validates `make_record_key()` format | Does not test non-integer year or float conversion bugs |
| `tests/benchmark/test_data_contracts.py` | `test_master_psu_partitioning_atomic` | 23 | 32 | Validates atomic PSU cluster allocation | Tests single call; does not test RNG draw sequence shift across arms |
| `tests/benchmark/test_data_contracts.py` | `test_generate_synthetic_cohort_invariants` | 34 | 47 | Validates synthetic cohort generation | Verifies basic column presence and row count invariants |
| `tests/benchmark/test_data_contracts.py` | `test_arm3_arm4_cohort_parity` | 49 | 81 | Validates Arm 003 vs Arm 004 population parity | Verifies identical respondents between 003 and 004 on synthetic data |
| `tests/benchmark/test_metrics.py` | `test_hand_calculated_survey_metrics` | 16 | 35 | Verifies hand-calculated weighted confusion metrics | Validates weighted formulas with clean inputs |
| `tests/benchmark/test_metrics.py` | `test_survey_fairness_gaps` | 37 | 57 | Verifies weighted DP and EO gap formulas | Only tests 2-group cases; does not test missing groups in multi-group expected groups |
| `tests/benchmark/test_metrics.py` | `test_rescaled_psu_bootstrap_engine` | 59 | 74 | Tests rescaled bootstrap weight matrix shape | Verifies shape $(B, N)$ and non-negativity; does not test singleton strata |
| `tests/benchmark/test_metrics.py` | `test_evaluate_with_survey_bootstrap` | 76 | 96 | Tests end-to-end bootstrap inference | Runs small test with $B=10$; does not verify $df$ or replicate proportion guards |
| `tests/benchmark/test_selection.py` | `test_selection_feasible_winner` | 9 | 29 | Verifies selection of winner within budget | Tests candidate filtering on Set S |
| `tests/benchmark/test_selection.py` | `test_selection_budget_exhausted_fallback` | 31 | 46 | Verifies fallback when no candidate feasible | Tests boundary fallback behavior |

> **Audit Finding**: All 19 test functions passed in the earlier run because they either tested synthetic fallback branches, called fragmented methods directly (bypassing unified interfaces), or asserted superficial properties (e.g. non-empty dictionary loaded from disk). They did not validate true BM execution, probability calibration, survey degrees of freedom, or zero-leakage contracts.
