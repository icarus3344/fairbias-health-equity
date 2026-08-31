# Gate 8: DUID-Grouped Partitioning, Target Stratification & Leakage Audit Report

## 1. Executive Summary

Gate 8 implements and verifies the household-grouped (`DUID`) 60/20/20 partitioning architecture and anti-leakage audit on MEPS HC-244 (Panel 26) on branch `research/meps-hc252-longitudinal`. Under authorized execution:
- **Household-Level Grouping**: Splitting is executed strictly at the Dwelling Unit ID (`DUID`) level. All individuals residing within the same dwelling unit are assigned to the identical partition.
- **Stratified Partitioning**: Partition allocation uses grouped stratification balancing positive outcome presence and aggregate sampling weight deciles across folds:
  - **Training Partition (60%)**: Used for fitting preprocessing pipelines and training candidate predictive models and fairness mitigation parameters.
  - **Validation Partition (20%)**: Reserved exclusively for hyperparameter, feature policy, and mitigation tuning.
  - **Calibration Partition (20%)**: Isolated during model selection; reserved for survey-weighted Platt scaling and freezing the 10% operational capacity threshold.
  - **Refit Partition (80%)**: Exact union of Training and Validation partitions for frozen model refitting.
- **Zero Household & Index Leakage**: Programmatically verified that $\text{DUID}_{\text{train}} \cap \text{DUID}_{\text{val}} = \emptyset$, $\text{DUID}_{\text{train}} \cap \text{DUID}_{\text{cal}} = \emptyset$, and $\text{DUID}_{\text{val}} \cap \text{DUID}_{\text{cal}} = \emptyset$.
- **Deterministic Assignment Hashes**: The partition assignment is cryptographically hashed (SHA-256) per random seed (`20260828`..`20260832`) for immutable reproducibility.
- **Empirical Partition Sizes (Panel 26)**:
  - Total eligible cohort: 2,882 records across 1,840 dwelling units.
  - Train: 1,732 records (1,104 DUIDs, ~60.1%)
  - Validation: 569 records (368 DUIDs, ~19.7%)
  - Calibration: 581 records (368 DUIDs, ~20.2%)
  - Train + Validation (Refit): 2,301 records (1,472 DUIDs, ~79.8%)

---

## 2. Section 9 Standard Worker Report

Gate: Gate 8 — DUID-Grouped Partitioning, Target Stratification & Leakage Audit
Status: COMPLETED_BY_GEMINI_BATCH_PENDING_FINAL_CODEX_REVIEW
Files changed:
- docs/reports/GATE_8_COHORT_SPLIT.md
- src/meps_fairness/data/split.py
- tests/test_gate8_split.py
Commands executed:
- /Users/lkc/Downloads/code_v_0_3/.venv311/bin/python -m py_compile src/meps_fairness/data/split.py tests/test_gate8_split.py
- PYTHONPATH=src /Users/lkc/Downloads/code_v_0_3/.venv311/bin/python -m unittest tests/test_gate8_split.py -v
- git diff --check
Permissions requested: None
Tests executed:
- `test_synthetic_duid_grouped_split_properties`: Verified DUID grouping, zero household overlap, refit partition union, and assignment hash reproducibility on synthetic households.
- `test_real_panel26_split_integration`: Verified 60/20/20 partition proportions, zero leakage, and presence of positive/negative events across all partitions on real Panel 26 data.
Exact test results:
- 2/2 tests passed cleanly in 0.324s.
Input hashes:
- Inherited baseline tag: inherited-code-v0.3-baseline-20260828 (commit 038897e9f751edac6e36445b7706eec5fdb15988)
- src/meps_fairness/data/cohort.py: 8d1f2e82110c71a3e9c5eb79e2c608f62f83141f2ff2ad07b57bfb7b34b15096
- data/interim/meps/h244/h244.dta: 5cf983c94fd9ed8d8377c9ad27bebd905c545327eca823a8c8412bf4e66eaa70
Output hashes:
- src/meps_fairness/data/split.py: e6fe753907ec347895bc6e3ef54ca854cf60aa4f7a55ae5c2c5c93c31878b19a
- tests/test_gate8_split.py: cf9e9f653457db2adff2b5fe1f78eaecb40a33c2aebc911cf339e0839f88d227
- docs/reports/GATE_8_COHORT_SPLIT.md: self-referential report artifact
Row counts:
- Panel 26 Cohort: 2,882 records -> Train: 1,732 (60.1%), Validation: 569 (19.7%), Calibration: 581 (20.2%), Refit: 2,301 (79.8%).
Assumptions:
- All individuals sharing a DUID are partitioned together to prevent intra-household information leakage.
- Seed 20260828 is the primary development split seed; seeds 20260829–20260832 are preserved for five-seed reproducibility.
Unresolved issues: None.
Git diff summary: 2 additive untracked files (`src/meps_fairness/data/split.py`, `tests/test_gate8_split.py`, `docs/reports/GATE_8_COHORT_SPLIT.md`). Zero inherited baseline files modified.
Proposed next step: Proceed to Gate 9 (Train-Only Preprocessing, Missing-Value Contracts & Design Quality Checks).
STOP — waiting for Codex review.
