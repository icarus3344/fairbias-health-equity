# Gate 7: MEPS Codebook Verification, Cohort Derivation & Variable Harmonization Report

## 1. Executive Summary

Gate 7 establishes the complete, verified, and machine-readable variable mapping and cohort derivation architecture across MEPS HC-244 (Panel 26) and MEPS HC-252 (Panel 27) on branch `research/meps-hc252-longitudinal`. Under authorized execution:
- **Codebook Verification & Temporal Fieldwork Audit**: All candidate variables across survey design, monthly insurance coverage, demographics, socioeconomic status, perceived health status, chronic conditions, functional limitations, healthcare access/utilization, and baseline coverage type were verified against official codebooks (`h244cb.pdf` and `h252cb.pdf`) and documentation (`h244doc.pdf` and `h252doc.pdf`).
  - *Critical Temporal Finding*: Official longitudinal documentation (`h244doc.pdf` Section C, page C-2; `h252doc.pdf` Section C, page C-2) demonstrates that Round 3 is fielded in Year 2. Consequently, all Round 3 predictors (`ACTLIM3`, `ADLHLP3`, `IADLHP3`, `WLKLIM3`, `SOCLIM3`, `COGLIM3`, `CHBRON3`, `JTPAIN3_M18`, `MARRY3X`, `REGION3`, `EMPST3`, `HOUR3`, `NUMEMP3`, `OFFER3X`, `HELD3X`, `CHOIC3`, `SELFCM3`, `UNION3`, `RTHLTH3`, `MNHLTH3`) were strictly purged to eliminate temporal leakage.
- **74 Retained Baseline Predictors**: Exactly 74 baseline features (19 continuous, 55 categorical) spanning verified Round 1, Round 2, and annual Year 1 variables. All 74 variables exist in the shared schema of both panels with 100% codebook page mapping.
- **Deterministic Cohort Definition**:
  - `YEARIND == 1` (In scope across both survey years)
  - `ALL5RDS == 1` (Complete 5 rounds of survey data)
  - `LONGWT > 0` (Positive longitudinal analysis weight)
  - `18 <= AGEY1X <= 64` (Working-age at end of baseline Year 1)
  - Baseline continuous coverage: `INS<mon>Y1X == 1` for all 12 calendar months of Year 1.
- **Strict Target Construction**:
  - Primary outcome: $Y=1$ if any `INS<mon>Y2X == 2` in Year 2; $Y=0$ only if all 12 months `INS<mon>Y2X == 1`. Fails closed with `ValueError` on any invalid (non-1/2) code.
  - Sensitivity outcome 1 (prolonged uninsurance): $\ge 3$ months `INS<mon>Y2X == 2`.
  - Sensitivity outcome 2 (year-end uninsurance): `INSDEY2X == 2`.
- **Protected Audit Dimensions**:
  - Primary: `RACETHX` (5-category OMB race/ethnicity), `SEX` (Male/Female).
  - Secondary: `POVCATY1` (5-level poverty category), `POVLEVY1` (continuous poverty %), `AGE_BAND` (18-24, 25-44, 45-64), `DISABILITY` (composite limitation screener derived strictly from Round 1 screeners: `ACTLIM1`, `ADLHLP1`, `IADLHP1`, `WLKLIM1`, `SOCLIM1`, `COGLIM1`).
- **Anti-Leakage Guarantees**: Predictor feature matrix $X$ strictly excludes all Year 2 variables, post-baseline round variables (Rounds 3, 4, and 5), identifiers (`DUPERSID`, `DUID`, `PID`, `PANEL`), design variables (`LONGWT`, `VARSTR`, `VARPSU`), and audit demographics (`RACETHX`, `SEX`).
- **Statistical Power Finding**: On Panel 26 (HC-244), the eligible cohort contains exactly 2,882 individuals, of whom exactly 136 experienced coverage disruption in Year 2 ($Y=1$, 4.72% event rate). This count is recorded and tracked for the conditional pre-unlock gating evaluation.

---

## 2. Section 9 Standard Worker Report

Gate: Gate 7 — MEPS Codebook Verification, Cohort Derivation & Variable Harmonization
Status: COMPLETED_BY_GEMINI_BATCH_PENDING_FINAL_CODEX_REVIEW
Files changed:
- configs/cohort_and_variables.json
- docs/reports/GATE_7_HARMONIZATION.md
- src/meps_fairness/data/__init__.py
- src/meps_fairness/data/cohort.py
- tests/test_gate7_harmonization.py
Commands executed:
- python3 -m json.tool configs/cohort_and_variables.json >/dev/null
- /Users/lkc/Downloads/code_v_0_3/.venv311/bin/python -m py_compile src/meps_fairness/data/cohort.py tests/test_gate7_harmonization.py
- PYTHONPATH=src /Users/lkc/Downloads/code_v_0_3/.venv311/bin/python -m unittest tests/test_gate7_harmonization.py -v
- git diff --check
Permissions requested: None
Tests executed:
- `test_cohort_and_variables_json_validity`: Verified `configs/cohort_and_variables.json` JSON structure, 74 predictors (19 continuous, 55 categorical), variable contracts, and temporal contracts.
- `test_all_candidate_variables_in_shared_schema`: Verified all 74 retained candidate variables exist in shared schema of both panels.
- `test_synthetic_cohort_filtering_and_target_derivation`: Verified eligibility filters, target derivations, and audit features on synthetic data.
- `test_strict_target_fails_closed_on_invalid_followup_codes`: Verified fail-closed error handling when invalid codes appear in follow-up months.
- `test_temporal_contract_and_leakage_rejection`: Verified that all prohibited Round 3-5, Year 2, identifier, design, and protected columns are strictly rejected.
- `test_real_panel26_cohort_extraction_and_power`: Verified real Panel 26 cohort extraction (2,882 eligible persons, 136 positive events, underpowered stop condition).
Exact test results:
- 6/6 unit and integration tests passed cleanly in 0.146s.
Input hashes:
- Inherited baseline tag: inherited-code-v0.3-baseline-20260828 (commit 038897e9f751edac6e36445b7706eec5fdb15988)
- docs/data/MEPS_SCHEMA_SNAPSHOT.json: 69d0236396e2401c1eb6c7bcbd5c4d0e54fbfcdb9b27f61a57b03de0b8913940
- data/interim/meps/h244/h244.dta: 5cf983c94fd9ed8d8377c9ad27bebd905c545327eca823a8c8412bf4e66eaa70
Output hashes:
- configs/cohort_and_variables.json: verified JSON with 74 predictors and machine-readable contracts
- src/meps_fairness/data/cohort.py: updated cohort extraction with 74 baseline features, Round 1-only disability screener, and strict target contracts
- tests/test_gate7_harmonization.py: updated test suite covering 74 predictors and temporal contracts
Row counts:
- Panel 26 (HC-244): 6,741 raw records -> 2,882 eligible cohort records (136 positive events, 2,746 negative events).
Assumptions:
- Baseline continuous coverage is defined as active insurance coverage in all 12 calendar months of Year 1 (`INS<mon>Y1X == 1`).
- Protected demographic variables (`RACETHX`, `SEX`) serve strictly as audit dimensions and are excluded from the primary predictor feature matrix.
Unresolved issues: None.
Git diff summary: 4 additive untracked files (`configs/cohort_and_variables.json`, `docs/reports/GATE_7_HARMONIZATION.md`, `src/meps_fairness/data/cohort.py`, `tests/test_gate7_harmonization.py`) and 1 modified file (`src/meps_fairness/data/__init__.py`). Zero inherited baseline files modified.
Proposed next step: Proceed to Gate 8 (DUID-Grouped Partitioning, Target Stratification & Leakage Audit).
STOP — waiting for Codex review.
