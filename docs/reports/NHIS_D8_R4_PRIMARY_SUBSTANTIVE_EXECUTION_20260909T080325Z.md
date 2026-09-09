# NHIS-D8-R4 Primary Frozen D6-Geometry Substantive Execution Report

## Gate:
`NHIS-D8-R4`

## Status:
`PRIMARY_D6_GEOMETRY_RESULTS_FROZEN_PENDING_CODEX_REVIEW`

---

## Files changed:
- `scripts/run_nhis_d8_r4_substantive.py` (untracked standalone execution runner for Gate D8-R4)
- `artifacts/nhis_d8_r4/20260909T080325Z_d8r4_substantive/*` (untracked deliverable artifacts, 18 files)
- `docs/reports/NHIS_D8_R4_PRIMARY_SUBSTANTIVE_EXECUTION_20260909T080325Z.md` (untracked audit report)
- `scripts/run_nhis_enhancement_study.py` (added `--r4-primary-authorized` CLI gate unlock flag)
- `src/nhis_fairbias/d8_enhancement_runner.py` (added `r4_primary_authorized` immutable contract property and fail-closed checks)
- `tests/test_nhis_d8_synthetic_contracts.py` (added `TestNHISD8R4GateContracts` covering R4-01 through R4-12)
- Inherited root files: **0 files changed** (exact 0 diff vs `inherited-code-v0.3-baseline-20260828`)
- Core `src/fairbias/` files: **0 files changed** (exact 0 diff vs `3bc3c40b7edabdbbd0cb443fd7bfca2d4d1e6909`)

---

## Commands executed:
1. `git log -n 1 --oneline` (confirmed `R4_BASELINE_COMMIT=3bc3c40b7edabdbbd0cb443fd7bfca2d4d1e6909`)
2. `git status --short` & `git diff --check` (verified clean working tree vs baseline and zero whitespace errors)
3. `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m py_compile scripts/run_nhis_d8_r4_substantive.py` (syntax and compilation check)
4. `PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m unittest tests/test_nhis_d8_synthetic_contracts.py` (ran 60 tests in 27.96s, OK)
5. `PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 scripts/run_nhis_d8_r4_substantive.py` (single authorized real-data substantive execution across all 4 arms and 4 conditions, duration: 544.86s)
6. Post-run verification and manifest finalization via `scratch/complete_r4_post_run_audit.py` (producing `c1_c2_anchor_recheck.json`, `engineering_sensitivity_comparison.json`, `post_run_source_verification.json`, `environment_manifest.json`, and `execution_manifest.json`)

---

## Permissions requested:
None. Substantive execution was performed strictly within authorized R4 boundaries on real NHIS microdata under a frozen configuration. Exactly ONE execution pass was conducted. No secondary runs, parameter tuning, model selection, or unprompted Git commits, pushes, or tags were performed.

---

## Tests executed:
1. **R4 Gate Synthetic Contract Suite** (`tests/test_nhis_d8_synthetic_contracts.py`):
   - All 12 R4-specific synthetic tests (`TestNHISD8R4GateContracts`) passed (R4-01 through R4-12).
   - Full test module (60 tests) passed in 27.96s with 0 failures, 0 errors.
2. **Protected Baseline Immutability Audit**:
   - 14 inherited root files vs `inherited-code-v0.3-baseline-20260828`: verified 0 diff, clean.
   - Core `src/fairbias/` files vs `3bc3c40b7edabdbbd0cb443fd7bfca2d4d1e6909`: verified 0 diff, clean.
3. **Pre-Run vs Post-Run Scientific Source Hash Verification**:
   - Verified that all 12 scientific source files and 8 reference artifact files remained bitwise identical before and after the real-data run (`all_scientific_source_hashes_match = true`).
4. **In-Run C1/C2 Anchor Barrier Check**:
   - Compared observed C1 Baseline and C2 Canonical FairBias outputs against frozen R2C results (`artifacts/nhis_d8_r2b/20260908T102300Z_d8r2b_eval/`):
     * Cohort sizes and positive counts: 4 / 4 arms PASS.
     * Canonical state hashes: 4 / 4 arms PASS (`40511e6c0d55b0ff`, `4c0bbba5d40022d6`, `38fa54a06a9c6427`, `ff0a2fb81596b598`).
     * Metrics across Train 2022, Validation 2023, and Test 2024: 136 / 136 metric comparisons PASS.
     * `all_barriers_passed = true`.

---

## Exact test results:

### 1. Preflight and Provenance Verification:
- `all_scientific_source_hashes_match`: **True** (20 / 20 files verified)
- `root_14_files_clean`: **True** (0 diff vs `inherited-code-v0.3-baseline-20260828`)
- `core_fairbias_clean`: **True** (0 diff vs `3bc3c40b7edabdbbd0cb443fd7bfca2d4d1e6909`)
- `r4_baseline_commit`: `3bc3c40b7edabdbbd0cb443fd7bfca2d4d1e6909`
- `r4_config_sha256`: `5dd77c584c1bfc4a6949aa5de5ed03666e818ee0ea2c981532d4052bb475c4d2`
- Execution duration: **544.86s**

---

### 2. Table A: Full Temporal Condition Matrix

| Arm | Condition | Stage | AUROC | AUPRC | Accuracy | Bal. Acc. | F1 | Brier | Pos. Count | Selection Rate | Max d_phi | DP Gap | EO Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| D6_ARM_001 | C1_Baseline | train_2022 | 0.7801 | 0.2080 | 0.9345 | 0.5092 | 0.0395 | 0.0559 | 104 | 0.00379 | 0.00477 | N/A | N/A |
| D6_ARM_001 | C1_Baseline | validation_2023 | 0.7791 | 0.2008 | 0.9325 | 0.5069 | 0.0314 | 0.0574 | 111 | 0.00379 | 0.00420 | 0.00102 | 0.00049 |
| D6_ARM_001 | C1_Baseline | test_2024 | 0.7517 | 0.2153 | 0.9197 | 0.5078 | 0.0349 | 0.0683 | 130 | 0.00402 | 0.00393 | 0.00058 | 0.00029 |
| D6_ARM_001 | C2_Canonical | train_2022 | 0.7053 | 0.1351 | 0.9355 | 0.5000 | 0.0000 | 0.0584 | 1 | 0.00004 | 0.00049 | N/A | N/A |
| D6_ARM_001 | C2_Canonical | validation_2023 | 0.7050 | 0.1439 | 0.9341 | 0.5003 | 0.0010 | 0.0594 | 1 | 0.00003 | 0.00083 | 0.00007 | 0.00130 |
| D6_ARM_001 | C2_Canonical | test_2024 | 0.6636 | 0.1409 | 0.9207 | 0.5000 | 0.0000 | 0.0714 | 2 | 0.00006 | 0.00140 | 0.00001 | 0.00000 |
| D6_ARM_001 | C3_Posthoc | train_2022 | 0.7070 | 0.1345 | 0.9356 | 0.5000 | 0.0000 | 0.0583 | 0 | 0.00000 | 0.00151 | N/A | N/A |
| D6_ARM_001 | C3_Posthoc | validation_2023 | 0.7070 | 0.1471 | 0.9341 | 0.5000 | 0.0000 | 0.0593 | 0 | 0.00000 | 0.00139 | 0.00000 | 0.00000 |
| D6_ARM_001 | C3_Posthoc | test_2024 | 0.6659 | 0.1424 | 0.9207 | 0.5000 | 0.0000 | 0.0713 | 1 | 0.00003 | 0.00180 | 0.00006 | 0.00000 |
| D6_ARM_001 | C4_Joint | train_2022 | 0.7303 | 0.1420 | 0.9355 | 0.5000 | 0.0000 | 0.0580 | 1 | 0.00004 | 0.00095 | N/A | N/A |
| D6_ARM_001 | C4_Joint | validation_2023 | 0.7302 | 0.1474 | 0.9340 | 0.5000 | 0.0000 | 0.0591 | 2 | 0.00007 | 0.00098 | 0.00001 | 0.00000 |
| D6_ARM_001 | C4_Joint | test_2024 | 0.7024 | 0.1599 | 0.9209 | 0.5006 | 0.0023 | 0.0705 | 3 | 0.00009 | 0.00166 | 0.00005 | 0.00036 |
| D6_ARM_002 | C1_Baseline | train_2022 | 0.7803 | 0.2079 | 0.9344 | 0.5094 | 0.0405 | 0.0559 | 106 | 0.00386 | 0.00522 | N/A | N/A |
| D6_ARM_002 | C1_Baseline | validation_2023 | 0.7791 | 0.2008 | 0.9325 | 0.5066 | 0.0304 | 0.0574 | 109 | 0.00372 | 0.00565 | 0.01172 | 0.11111 |
| D6_ARM_002 | C1_Baseline | test_2024 | 0.7514 | 0.2148 | 0.9197 | 0.5076 | 0.0342 | 0.0683 | 127 | 0.00393 | 0.00528 | 0.00872 | 0.03814 |
| D6_ARM_002 | C2_Canonical | train_2022 | 0.7601 | 0.1735 | 0.9351 | 0.5016 | 0.0078 | 0.0569 | 25 | 0.00091 | 0.00189 | N/A | N/A |
| D6_ARM_002 | C2_Canonical | validation_2023 | 0.7618 | 0.1792 | 0.9339 | 0.5021 | 0.0092 | 0.0580 | 23 | 0.00079 | 0.00274 | 0.00518 | 0.11111 |
| D6_ARM_002 | C2_Canonical | test_2024 | 0.7328 | 0.1801 | 0.9205 | 0.5029 | 0.0130 | 0.0697 | 43 | 0.00133 | 0.00211 | 0.00376 | 0.01370 |
| D6_ARM_002 | C3_Posthoc | train_2022 | 0.7609 | 0.1809 | 0.9353 | 0.5017 | 0.0078 | 0.0567 | 21 | 0.00076 | 0.00469 | N/A | N/A |
| D6_ARM_002 | C3_Posthoc | validation_2023 | 0.7645 | 0.1900 | 0.9341 | 0.5036 | 0.0153 | 0.0576 | 29 | 0.00099 | 0.00268 | 0.00518 | 0.11111 |
| D6_ARM_002 | C3_Posthoc | test_2024 | 0.7347 | 0.1853 | 0.9206 | 0.5030 | 0.0131 | 0.0694 | 38 | 0.00117 | 0.00344 | 0.00376 | 0.01370 |
| D6_ARM_002 | C4_Joint | train_2022 | 0.7580 | 0.1796 | 0.9352 | 0.5004 | 0.0022 | 0.0566 | 12 | 0.00044 | 0.00267 | N/A | N/A |
| D6_ARM_002 | C4_Joint | validation_2023 | 0.7573 | 0.1822 | 0.9341 | 0.5034 | 0.0143 | 0.0579 | 26 | 0.00089 | 0.00256 | 0.00518 | 0.11111 |
| D6_ARM_002 | C4_Joint | test_2024 | 0.7302 | 0.1810 | 0.9208 | 0.5025 | 0.0108 | 0.0696 | 27 | 0.00083 | 0.00248 | 0.00794 | 0.03448 |
| D6_ARM_003 | C1_Baseline | train_2022 | 0.7803 | 0.2079 | 0.9344 | 0.5094 | 0.0405 | 0.0559 | 106 | 0.00386 | 0.01097 | N/A | N/A |
| D6_ARM_003 | C1_Baseline | validation_2023 | 0.7789 | 0.2007 | 0.9324 | 0.5066 | 0.0304 | 0.0575 | 110 | 0.00376 | 0.01042 | 0.00870 | 0.03609 |
| D6_ARM_003 | C1_Baseline | test_2024 | 0.7513 | 0.2143 | 0.9196 | 0.5078 | 0.0349 | 0.0683 | 131 | 0.00405 | 0.00926 | 0.01311 | 0.05366 |
| D6_ARM_003 | C2_Canonical | train_2022 | 0.7709 | 0.1986 | 0.9340 | 0.5047 | 0.0227 | 0.0563 | 83 | 0.00302 | 0.00497 | N/A | N/A |
| D6_ARM_003 | C2_Canonical | validation_2023 | 0.7683 | 0.1908 | 0.9323 | 0.5039 | 0.0198 | 0.0580 | 92 | 0.00314 | 0.00493 | 0.00258 | 0.01172 |
| D6_ARM_003 | C2_Canonical | test_2024 | 0.7344 | 0.2027 | 0.9196 | 0.5052 | 0.0247 | 0.0690 | 104 | 0.00321 | 0.00458 | 0.00490 | 0.02776 |
| D6_ARM_003 | C3_Posthoc | train_2022 | 0.7748 | 0.2334 | 0.9357 | 0.5175 | 0.0696 | 0.0549 | 127 | 0.00463 | 0.00525 | N/A | N/A |
| D6_ARM_003 | C3_Posthoc | validation_2023 | 0.7733 | 0.2233 | 0.9340 | 0.5135 | 0.0548 | 0.0566 | 113 | 0.00386 | 0.00448 | 0.00285 | 0.01330 |
| D6_ARM_003 | C3_Posthoc | test_2024 | 0.7400 | 0.2289 | 0.9211 | 0.5159 | 0.0645 | 0.0678 | 166 | 0.00513 | 0.00515 | 0.00579 | 0.02616 |
| D6_ARM_003 | C4_Joint | train_2022 | 0.7692 | 0.2248 | 0.9356 | 0.5137 | 0.0556 | 0.0553 | 101 | 0.00368 | 0.00497 | N/A | N/A |
| D6_ARM_003 | C4_Joint | validation_2023 | 0.7607 | 0.2130 | 0.9339 | 0.5100 | 0.0416 | 0.0571 | 88 | 0.00301 | 0.00491 | 0.00488 | 0.02913 |
| D6_ARM_003 | C4_Joint | test_2024 | 0.7298 | 0.2162 | 0.9210 | 0.5105 | 0.0434 | 0.0684 | 109 | 0.00337 | 0.00455 | 0.00717 | 0.03762 |
| D6_ARM_004 | C1_Baseline | train_2022 | 0.7668 | 0.1929 | 0.9338 | 0.5033 | 0.0173 | 0.0565 | 78 | 0.00284 | 0.01788 | N/A | N/A |
| D6_ARM_004 | C1_Baseline | validation_2023 | 0.7626 | 0.1884 | 0.9321 | 0.5035 | 0.0188 | 0.0581 | 95 | 0.00324 | 0.01714 | 0.00210 | 0.00538 |
| D6_ARM_004 | C1_Baseline | test_2024 | 0.7353 | 0.1990 | 0.9195 | 0.5034 | 0.0174 | 0.0691 | 88 | 0.00272 | 0.01551 | 0.00211 | 0.01352 |
| D6_ARM_004 | C2_Canonical | train_2022 | 0.7237 | 0.1676 | 0.9336 | 0.4998 | 0.0033 | 0.0577 | 59 | 0.00215 | 0.00363 | N/A | N/A |
| D6_ARM_004 | C2_Canonical | validation_2023 | 0.7106 | 0.1610 | 0.9320 | 0.4996 | 0.0030 | 0.0594 | 67 | 0.00229 | 0.00341 | 0.00031 | 0.00158 |
| D6_ARM_004 | C2_Canonical | test_2024 | 0.6670 | 0.1592 | 0.9193 | 0.5008 | 0.0068 | 0.0712 | 67 | 0.00207 | 0.00459 | 0.00010 | 0.00388 |
| D6_ARM_004 | C3_Posthoc | train_2022 | 0.7287 | 0.1860 | 0.9352 | 0.5009 | 0.0045 | 0.0565 | 16 | 0.00058 | 0.00310 | N/A | N/A |
| D6_ARM_004 | C3_Posthoc | validation_2023 | 0.7158 | 0.1820 | 0.9341 | 0.5019 | 0.0082 | 0.0581 | 16 | 0.00055 | 0.00275 | 0.00025 | 0.00158 |
| D6_ARM_004 | C3_Posthoc | test_2024 | 0.6756 | 0.1787 | 0.9209 | 0.5015 | 0.0062 | 0.0700 | 13 | 0.00040 | 0.00406 | 0.00016 | 0.00107 |
| D6_ARM_004 | C4_Joint | train_2022 | 0.7352 | 0.2009 | 0.9356 | 0.5040 | 0.0167 | 0.0561 | 28 | 0.00102 | 0.00475 | N/A | N/A |
| D6_ARM_004 | C4_Joint | validation_2023 | 0.7272 | 0.1971 | 0.9342 | 0.5044 | 0.0183 | 0.0577 | 31 | 0.00106 | 0.00522 | 0.00097 | 0.00950 |
| D6_ARM_004 | C4_Joint | test_2024 | 0.6866 | 0.1874 | 0.9209 | 0.5027 | 0.0116 | 0.0696 | 27 | 0.00083 | 0.00386 | 0.00150 | 0.00646 |

---

### 3. Table B: Primary Deltas vs Canonical FairBias (C2)

| Arm | Stage | Enhancement | Delta AUROC vs C2 | Delta AUPRC vs C2 | Delta Pos Count vs C2 | Delta Selection Rate vs C2 | Delta Max d_phi vs C2 | Delta DP Gap vs C2 | Delta EO Gap vs C2 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| D6_ARM_001 | train_2022 | C3_Posthoc | +0.0017 | -0.0006 | -1 | -0.00004 | +0.00102 | N/A | N/A |
| D6_ARM_001 | train_2022 | C4_Joint | +0.0250 | +0.0069 | +0 | +0.00000 | +0.00047 | N/A | N/A |
| D6_ARM_001 | validation_2023 | C3_Posthoc | +0.0019 | +0.0032 | -1 | -0.00003 | +0.00057 | -0.00007 | -0.00130 |
| D6_ARM_001 | validation_2023 | C4_Joint | +0.0252 | +0.0035 | +1 | +0.00003 | +0.00015 | -0.00006 | -0.00130 |
| D6_ARM_001 | test_2024 | C3_Posthoc | +0.0023 | +0.0015 | -1 | -0.00003 | +0.00040 | +0.00005 | +0.00000 |
| D6_ARM_001 | test_2024 | C4_Joint | +0.0388 | +0.0191 | +1 | +0.00003 | +0.00026 | +0.00004 | +0.00036 |
| D6_ARM_002 | train_2022 | C3_Posthoc | +0.0009 | +0.0075 | -4 | -0.00015 | +0.00280 | N/A | N/A |
| D6_ARM_002 | train_2022 | C4_Joint | -0.0021 | +0.0061 | -13 | -0.00047 | +0.00079 | N/A | N/A |
| D6_ARM_002 | validation_2023 | C3_Posthoc | +0.0027 | +0.0108 | +6 | +0.00020 | -0.00006 | +0.00000 | +0.00000 |
| D6_ARM_002 | validation_2023 | C4_Joint | -0.0045 | +0.0031 | +3 | +0.00010 | -0.00019 | +0.00000 | +0.00000 |
| D6_ARM_002 | test_2024 | C3_Posthoc | +0.0019 | +0.0052 | -5 | -0.00015 | +0.00133 | +0.00000 | +0.00000 |
| D6_ARM_002 | test_2024 | C4_Joint | -0.0026 | +0.0009 | -16 | -0.00049 | +0.00037 | +0.00418 | +0.02078 |
| D6_ARM_003 | train_2022 | C3_Posthoc | +0.0039 | +0.0349 | +44 | +0.00160 | +0.00028 | N/A | N/A |
| D6_ARM_003 | train_2022 | C4_Joint | -0.0017 | +0.0262 | +18 | +0.00066 | -0.00000 | N/A | N/A |
| D6_ARM_003 | validation_2023 | C3_Posthoc | +0.0050 | +0.0324 | +21 | +0.00072 | -0.00046 | +0.00027 | +0.00159 |
| D6_ARM_003 | validation_2023 | C4_Joint | -0.0076 | +0.0221 | -4 | -0.00014 | -0.00002 | +0.00230 | +0.01741 |
| D6_ARM_003 | test_2024 | C3_Posthoc | +0.0057 | +0.0262 | +62 | +0.00192 | +0.00058 | +0.00088 | -0.00160 |
| D6_ARM_003 | test_2024 | C4_Joint | -0.0046 | +0.0134 | +5 | +0.00015 | -0.00003 | +0.00226 | +0.00986 |
| D6_ARM_004 | train_2022 | C3_Posthoc | +0.0050 | +0.0184 | -43 | -0.00157 | -0.00053 | N/A | N/A |
| D6_ARM_004 | train_2022 | C4_Joint | +0.0115 | +0.0332 | -31 | -0.00113 | +0.00112 | N/A | N/A |
| D6_ARM_004 | validation_2023 | C3_Posthoc | +0.0052 | +0.0210 | -51 | -0.00174 | -0.00066 | -0.00005 | -0.00000 |
| D6_ARM_004 | validation_2023 | C4_Joint | +0.0166 | +0.0361 | -36 | -0.00123 | +0.00181 | +0.00066 | +0.00792 |
| D6_ARM_004 | test_2024 | C3_Posthoc | +0.0086 | +0.0195 | -54 | -0.00167 | -0.00052 | +0.00006 | -0.00280 |
| D6_ARM_004 | test_2024 | C4_Joint | +0.0196 | +0.0282 | -40 | -0.00124 | -0.00072 | +0.00140 | +0.00258 |

---

### 4. Table C: Enhancement Search & Audit Summary

| Arm | Condition | Termination Reason | Feasible? | Committed (Total / AE / BM) | Num Transforms | Candidates | Model Fits | Geometry Evals | Final State Hash |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| D6_ARM_001 | C3_Posthoc | budget_exhausted | False | 5 (5 AE / 0 BM) | 12 | 0 | 25 | 20 | `e482706efb64d816` |
| D6_ARM_001 | C4_Joint | budget_exhausted | False | 20 (10 AE / 10 BM) | 10 | 0 | 60 | 50 | `bc4fb10c249fb5b7` |
| D6_ARM_002 | C3_Posthoc | budget_exhausted | False | 5 (5 AE / 0 BM) | 11 | 0 | 19 | 14 | `309b9c5094b28bc3` |
| D6_ARM_002 | C4_Joint | budget_exhausted | False | 20 (10 AE / 10 BM) | 11 | 0 | 54 | 44 | `2fa5f58ebee88072` |
| D6_ARM_003 | C3_Posthoc | budget_exhausted | False | 5 (5 AE / 0 BM) | 9 | 0 | 22 | 17 | `09d937d116efa19d` |
| D6_ARM_003 | C4_Joint | epsilon_reached | True | 18 (9 AE / 9 BM) | 10 | 0 | 57 | 48 | `322e6506ac0d5fc3` |
| D6_ARM_004 | C3_Posthoc | budget_exhausted | True | 5 (5 AE / 0 BM) | 11 | 0 | 37 | 32 | `0b813d22a7d611f1` |
| D6_ARM_004 | C4_Joint | epsilon_reached | True | 16 (8 AE / 8 BM) | 11 | 0 | 70 | 62 | `cfe12b363df949cd` |

---

### 5. Historical Engineering Sensitivity Comparison (vs R3 Exploratory Run)

1. **Arm 1 (SEX_A)**:
   - C3: `state_hash_match = true` (`e482706efb64d816`), $\Delta \text{AUROC} = 0.0000$, $\text{max } d_\phi = 0.001802$ vs $0.001802$.
   - C4: `state_hash_match = true` (`bc4fb10c249fb5b7`), $\Delta \text{AUROC} = 0.0000$, $\text{max } d_\phi = 0.001659$ vs $0.001658$.
   - **Classification**: `qualitatively robust across geometry` (identical discrete state trajectories).
2. **Arm 2 (HISPALLP_A)**:
   - C3: `state_hash_match = true` (`309b9c5094b28bc3`), $\Delta \text{AUROC} = 0.0000$.
   - C4: `state_hash_match = false` (`2fa5f58ebee88072` vs `728280561dd4375e`), 20 vs 13 steps, $\Delta \text{AUROC} = -0.0149$, but improved fairness constraint satisfaction ($\text{max } d_\phi = 0.00248$ in R4 vs $0.00493$ in R3).
   - **Classification**: `magnitude-sensitive`.
3. **Arm 3 (DISAB3_A)**:
   - C3: `state_hash_match = true` (`09d937d116efa19d`), $\Delta \text{AUROC} = 0.0000$.
   - C4: `state_hash_match = false` (`322e6506ac0d5fc3` vs `16e86371ee7fb35d`), 18 vs 16 steps, but **both** reached `epsilon_reached` feasible fairness ($\text{max } d_\phi \le 0.00500$), test AUROC $0.7298$ vs $0.7295$ ($\Delta \text{AUROC} = +0.0003$).
   - **Classification**: `magnitude-sensitive`.
4. **Arm 4 (DISAB3_A)**:
   - C3: `state_hash_match = true` (`0b813d22a7d611f1`), $\Delta \text{AUROC} = 0.0000$.
   - C4: `state_hash_match = false` (`cfe12b363df949cd` vs `3029dbe488c54f51`), 16 vs 13 steps. Crucially, under the primary frozen D6 geometry, R4 achieved **`epsilon_reached` feasible fairness** ($\text{max } d_\phi = 0.00386 \le 0.00500$), whereas R3 exploratory geometry failed feasibility and exhausted budget ($\text{max } d_\phi = 0.01348 > 0.00500$).
   - **Classification**: `magnitude-sensitive`.

---

## Input hashes:
- `R4_BASELINE_COMMIT`: `3bc3c40b7edabdbbd0cb443fd7bfca2d4d1e6909`
- `R4_CONFIG_SHA256`: `5dd77c584c1bfc4a6949aa5de5ed03666e818ee0ea2c981532d4052bb475c4d2`
- `data/processed/nhis/nhis_2022_2024_features.parquet`: `49f415132ff0be0228f8533f9f74c48cd79ff6fa8be66db085f7329d9b083383`
- `docs/releases/NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7/D6_ARM_001/frozen_changed_dict.json`: `a296b1b59419cf9fb084bbaf00e2b4f6e4a2c262f7d5c7f8976b3f7f89b9148d`
- `docs/releases/NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7/D6_ARM_002/frozen_changed_dict.json`: `ee6cba8bf2c286e1074aeeea046522c7104e17efd04be2da83187c3a0bcf5f76`
- `docs/releases/NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7/D6_ARM_003/frozen_changed_dict.json`: `56195b058c42289650d3a5ca1207e99ca19655f4fa6e7b165507e155c65f9733`
- `docs/releases/NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7/D6_ARM_004/frozen_changed_dict.json`: `b9d21c326d91da3c0b0292723c3188ee279d464db31d1b951c8901eb25aa7a8d`
- `docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_001/train_dphi_before_after.json`: `bf9a6fa5899933ee66ef11d5952d7e5dca52d431d8e1216508933e144365774a`
- `docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_002/train_dphi_before_after.json`: `4fc2194a2b27cc362d2a450535e5a26a310c85e25287f3b23267597bceae0e22`
- `docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_003/train_dphi_before_after.json`: `287661cf6654f595f9d6eb44fe115e5899ea2ea7730e70a75f80bcf2e2ffb2ae`
- `docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_004/train_dphi_before_after.json`: `7b243be44443ff45cffea61a2936a28292c97486e9273c5df3ec183569738f76`

---

## Output hashes:
- `c1_c2_anchor_recheck.json`: `c434cef539f798559587e30b4e8e9df0fb1b1521a1f7354d7c98b648473f7bfc` (42,673 bytes)
- `command_log.txt`: `4170309811492040b575b7f5febdd5565ed5f23aa6b9dfbf346ded12dcdbbbeb` (923 bytes)
- `condition_metrics.json`: `98e03a69c92483ffc298ad4b4817bad0dc2ce8171a9170a3d77462f554eb0b0b` (30,399 bytes)
- `condition_metrics.md`: `56ac95f36743cecb90efe8173b8dfc48953df27d8a1f2616ee6f1e45a8bb633c` (6,957 bytes)
- `engineering_sensitivity_comparison.json`: `04668f38f9c417554407f730740021fd1f45770ff7686ce5c0e605b5aa5d565b` (5,831 bytes)
- `enhancement_audit_summary.json`: `cb3fa81dcbfb9d9144eb181d6635e27b0a817d710fc13754496f581de6015215` (7,609 bytes)
- `enhancement_audit_summary.md`: `11015af3f12e017749fbf51df6cd0e054a046123e57a6cafe88b5d34288086bf` (1,179 bytes)
- `environment_manifest.json`: `ed06517c1687eb36466e4c99de532eddd94e4c85b9be01c31dd0c9dcf8ed2af0` (386 bytes)
- `frozen_d6_reference_manifest.json`: `d9ec5e2cc094e926079d71fbcd449ff767a43ab91611a6a8b9d93d5ad4111110` (2,091 bytes)
- `joint_trajectory_events.json`: `9d8df1137543a90c187479aaaafbf713e6a41a0bf853252ef9d9f37641bc6e3c` (52,249 bytes)
- `post_run_source_verification.json`: `e0ff23c53402796c4a366b71a96557a9e41e002645549689489cad1f9c4e17bf` (6,101 bytes)
- `pre_run_git_diff.patch`: `7d27ac8bbe38b1bf16659290e079d4dbdaa6c063fb1b2b5ffef58f48edd53cc0` (18,988 bytes)
- `pre_run_source_manifest.json`: `ab5b8f99a108e9c7f66288aebd702e3a5ad46dfabc14cdc929b28822187b3d4f` (3,327 bytes)
- `primary_deltas.json`: `82276540bfc57de6c0f8b1fe9029a29a927007d6f5ec31d8210d25182b9c025b` (18,716 bytes)
- `primary_deltas.md`: `7d1b5cf5efa614b5340b01f10d1511deecba590d17f56db0e1239bbe6b3ce25f` (5,805 bytes)
- `protected_file_integrity.json`: `e758fea6567f917c387b7b95480eb2eee2a08155d2911780dd77739309163719` (431 bytes)
- `r4_config_manifest.json`: `b9e3990401d769b1e32535e75c32b9eebf0a2e5aedc5ad5c5eb12ba4b114ed0c` (3,991 bytes)
- `execution_manifest.json`: `adfd1dfeb7cf7ef89dc721e25e173e67ae7daec96614a95574cb11ef841cf5e1` (5,646 bytes)

---

## Row counts:
- `condition_metrics.json`: **48 rows** (4 arms $\times$ 4 conditions $\times$ 3 temporal stages)
- `primary_deltas.json`: **24 rows** (4 arms $\times$ 3 temporal stages $\times$ 2 enhancement conditions)
- `enhancement_audit_summary.json`: **8 rows** (4 arms $\times$ 2 enhancement conditions)
- `c1_c2_anchor_recheck.json`: **136 metric rows** (all PASS) + 4 cohort records (all PASS) + 4 state records (all PASS)
- `post_run_source_verification.json`: **20 verified files** (all PASS)
- `engineering_sensitivity_comparison.json`: **4 arms** compared across C3 and C4 against R3

---

## Assumptions:
- `SUBSTANTIVE_D6_GEOMETRY` is the authoritative primary research mode for Gate D8.
- The historical R3 execution is preserved as `EXPLORATORY_ENGINEERING_GEOMETRY_RUN` and serves strictly as sensitivity analysis.
- The training epsilon thresholds loaded from `docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/<ARM>/train_dphi_before_after.json` are immutable authoritative baselines.

---

## Unresolved issues:
None. All 18 deliverable artifacts are serialized, verified, and anchored to prior accepted gates. Exactly one execution pass was conducted.

---

## Git diff summary:
```text
 M scripts/run_nhis_enhancement_study.py
 M src/nhis_fairbias/d8_enhancement_runner.py
 M tests/test_nhis_d8_synthetic_contracts.py
?? scripts/run_nhis_d8_r4_substantive.py
?? artifacts/nhis_d8_r4/20260909T080325Z_d8r4_substantive/
?? docs/reports/NHIS_D8_R4_PRIMARY_SUBSTANTIVE_EXECUTION_20260909T080325Z.md
```
- Protected baseline files: **0 diff** (clean vs `inherited-code-v0.3-baseline-20260828`)
- Core `src/fairbias/` files: **0 diff** (clean vs `3bc3c40b7edabdbbd0cb443fd7bfca2d4d1e6909`)
- No Git commit, push, or tag was created during R4.

---

## Proposed next step:
Submit report and substantive evidence bundle to Codex supervisor for formal review and determination of Gate `NHIS-D8-R4` acceptance.

---

STOP — waiting for Codex review.
