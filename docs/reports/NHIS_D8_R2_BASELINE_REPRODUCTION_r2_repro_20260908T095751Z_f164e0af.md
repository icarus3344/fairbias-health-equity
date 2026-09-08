# NHIS-D8-R2 Baseline Reproduction and Controlled Evaluation Report

## Gate:
`NHIS-D8-R2`

## Status:
`REPRODUCED_PENDING_CODEX_REVIEW`

## Files changed:
- `src/nhis_fairbias/d8_enhancement_runner.py` (working-tree modification)
- `scripts/run_nhis_enhancement_study.py` (working-tree modification)
- `tests/test_nhis_d8_synthetic_contracts.py` (working-tree modification)
- `scripts/reproduce_d6_baselines.py` (untracked standalone verification script)

## Commands executed:
1. `git rev-parse HEAD` (verified supervisor-authorized R1 closure commit: `d29f7e8fa2e0ab56f21de7407b7d7d921b7b9955`)
2. `git diff --stat inherited-code-v0.3-baseline-20260828 -- app.py classifiers.py config.py data_COMPAS.csv data_Credit_Card.csv eval.py main.py module_AE.py module_BM.py module_load.py module_transform.py requirements.txt results/all_results.json start.sh` (verified 0 diff)
3. `git diff --stat d29f7e8fa2e0ab56f21de7407b7d7d921b7b9955 -- src/fairbias/mitigation.py src/fairbias/bias_metric.py src/fairbias/transform.py src/fairbias/evaluator.py src/fairbias/models.py src/fairbias/config.py` (verified 0 diff)
4. `PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 scratch/run_r2_phase_a_preflight.py` (guarded execution under `sys.addaudithook`; 58/58 tests passed, 0 real data reads)
5. `PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 scripts/reproduce_d6_baselines.py --output-dir artifacts/nhis_d8_r2/20260908T095751Z_f164e0af --random-seed 0` (controlled execution across 4 arms, verification against frozen D6 release references)
6. `git diff --check` (clean, 0 whitespace errors)

## Permissions requested:
None. Controlled baseline reproduction was executed strictly within authorized boundaries (`--baseline-reproduction-only --allow-real-data`) evaluating Condition 1 (Baseline) and Condition 2 (Canonical FairBias) exclusively. No unprompted Git commits, pushes, tags, or substantive enhancement experiments were performed.

## Tests executed:
1. **Phase A: Process-Level Guarded Synthetic Preflight Suite** (58 unit and contract tests under `sys.addaudithook` real-data blocking):
   - Audit hook intercept verification: verified attempts to access real microdata (`data/processed/nhis/...`) raise `RuntimeError: DATA_ACCESS_BLOCKED`.
   - `tests/test_nhis_d8_synthetic_contracts.py` (17 tests, including `TestD8R2BaselineReproductionContracts` verifying structural bypass of Conditions 3/4, 0 candidate model fits, CLI exclusivity, and real-data guards).
   - `tests/test_fairbias_enhancement.py` (4 tests).
   - `tests/test_fairbias_enhancement_contracts.py` (37 tests).
2. **Phase B: Protected File and Baseline Integrity Audit**:
   - 14 inherited root files vs `inherited-code-v0.3-baseline-20260828`: verified 0 diff, clean.
   - `.gitignore` baseline drift vs `inherited-code-v0.3-baseline-20260828`: verified pre-existing drift recorded and untouched (`PRE_EXISTING_BASELINE_DRIFT_RECORDED_NOT_RESOLVED`).
   - 6 frozen core modules in `src/fairbias/` vs `d29f7e8fa2e0ab56f21de7407b7d7d921b7b9955`: verified 0 diff, clean.
3. **Phase C & D: Controlled Real-Data Reproduction Suite** across all 4 target arms:
   - `D6_ARM_001` (SEX, full-feature)
   - `D6_ARM_002` (HISP, full-feature)
   - `D6_ARM_003` (DISAB, full-feature)
   - `D6_ARM_004` (DISAB, exclude-disability-components)
   - Validation against frozen release baselines: `docs/releases/NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7` and `docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609`.

## Exact test results:
- **Phase A Preflight**: **58 / 58 passed**, 0 failures, 0 errors, **0 blocked real-data read attempts** under process-level audit hook.
- **Phase B Integrity**: **PASS** (14 inherited baseline files: clean, 0 diff; 6 frozen core modules: clean, 0 diff).
- **Phase C & D Reproduction Verification**:
  - `ALL_REPRODUCTION_BARRIERS_PASS: True`.
  - **Cohort Validation**: 4 / 4 arms exact match on split sizes ($N$) and positive counts ($Y=1$) across Train ($N=24,365$), Validation ($N=29,277$ to $29,283$), and Test ($N=32,350$ to $32,355$).
  - **Schema Validation**: 4 / 4 arms exact match on categorical (18 or 12) and numerical (3) feature specifications and protected attributes.
  - **Canonical State Hash Validation**: 4 / 4 arms exact match on accepted transform sequence length and SHA-256 state hashes:
    - `D6_ARM_001`: 10 transforms, hash `40511e6c0d55b0ff` (exact match)
    - `D6_ARM_002`: 9 transforms, hash `4c0bbba5d40022d6` (exact match)
    - `D6_ARM_003`: 6 transforms, hash `38fa54a06a9c6427` (exact match)
    - `D6_ARM_004`: 8 transforms, hash `ff0a2fb81596b598` (exact match)
  - **Metric Reproduction Comparison**:
    - **64 / 64 metrics** across all 8 condition rows achieved literal **`0.00e+00` absolute difference** vs frozen D6 references (well within tolerance $\le 1.00 \times 10^{-10}$ for floating-point metrics and $0$ for integer counts).

### Complete Reproduction Table:
| Condition Row | Metric | D6 Reference | D8 Reproduced | Absolute Difference | Tolerance | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| SEX baseline | `auroc` | 0.75168139 | 0.75168139 | 0.00e+00 | 1.00e-10 | **PASS** |
| SEX baseline | `auprc` | 0.21530154 | 0.21530154 | 0.00e+00 | 1.00e-10 | **PASS** |
| SEX baseline | `accuracy` | 0.91965997 | 0.91965997 | 0.00e+00 | 1.00e-10 | **PASS** |
| SEX baseline | `predicted_positive_count` | 130 | 130 | 0.00e+00 | 0 | **PASS** |
| SEX baseline | `selection_rate` | 0.00401855 | 0.00401855 | 0.00e+00 | 1.00e-10 | **PASS** |
| SEX baseline | `max_dphi` | 0.00392525 | 0.00392525 | 0.00e+00 | 1.00e-10 | **PASS** |
| SEX baseline | `demographic_parity_gap` | 0.00058300 | 0.00058300 | 0.00e+00 | 1.00e-10 | **PASS** |
| SEX baseline | `equal_opportunity_gap` | 0.00029305 | 0.00029305 | 0.00e+00 | 1.00e-10 | **PASS** |
| SEX canonical | `auroc` | 0.66361631 | 0.66361631 | 0.00e+00 | 1.00e-10 | **PASS** |
| SEX canonical | `auprc` | 0.14086677 | 0.14086677 | 0.00e+00 | 1.00e-10 | **PASS** |
| SEX canonical | `accuracy` | 0.92071097 | 0.92071097 | 0.00e+00 | 1.00e-10 | **PASS** |
| SEX canonical | `predicted_positive_count` | 2 | 2 | 0.00e+00 | 0 | **PASS** |
| SEX canonical | `selection_rate` | 0.00006182 | 0.00006182 | 0.00e+00 | 1.00e-10 | **PASS** |
| SEX canonical | `max_dphi` | 0.00140299 | 0.00140299 | 0.00e+00 | 1.00e-10 | **PASS** |
| SEX canonical | `demographic_parity_gap` | 0.00001018 | 0.00001018 | 0.00e+00 | 1.00e-10 | **PASS** |
| SEX canonical | `equal_opportunity_gap` | 0.00000000 | 0.00000000 | 0.00e+00 | 1.00e-10 | **PASS** |
| HISP baseline | `auroc` | 0.75136925 | 0.75136925 | 0.00e+00 | 1.00e-10 | **PASS** |
| HISP baseline | `auprc` | 0.21480999 | 0.21480999 | 0.00e+00 | 1.00e-10 | **PASS** |
| HISP baseline | `accuracy` | 0.91967238 | 0.91967238 | 0.00e+00 | 1.00e-10 | **PASS** |
| HISP baseline | `predicted_positive_count` | 127 | 127 | 0.00e+00 | 0 | **PASS** |
| HISP baseline | `selection_rate` | 0.00392520 | 0.00392520 | 0.00e+00 | 1.00e-10 | **PASS** |
| HISP baseline | `max_dphi` | 0.00528135 | 0.00528135 | 0.00e+00 | 1.00e-10 | **PASS** |
| HISP baseline | `demographic_parity_gap` | 0.00872401 | 0.00872401 | 0.00e+00 | 1.00e-10 | **PASS** |
| HISP baseline | `equal_opportunity_gap` | 0.03813559 | 0.03813559 | 0.00e+00 | 1.00e-10 | **PASS** |
| HISP canonical | `auroc` | 0.73276792 | 0.73276792 | 0.00e+00 | 1.00e-10 | **PASS** |
| HISP canonical | `auprc` | 0.18007305 | 0.18007305 | 0.00e+00 | 1.00e-10 | **PASS** |
| HISP canonical | `accuracy` | 0.92047597 | 0.92047597 | 0.00e+00 | 1.00e-10 | **PASS** |
| HISP canonical | `predicted_positive_count` | 43 | 43 | 0.00e+00 | 0 | **PASS** |
| HISP canonical | `selection_rate` | 0.00132901 | 0.00132901 | 0.00e+00 | 1.00e-10 | **PASS** |
| HISP canonical | `max_dphi` | 0.00210791 | 0.00210791 | 0.00e+00 | 1.00e-10 | **PASS** |
| HISP canonical | `demographic_parity_gap` | 0.00375940 | 0.00375940 | 0.00e+00 | 1.00e-10 | **PASS** |
| HISP canonical | `equal_opportunity_gap` | 0.01369863 | 0.01369863 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-full baseline | `auroc` | 0.75127253 | 0.75127253 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-full baseline | `auprc` | 0.21433313 | 0.21433313 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-full baseline | `accuracy` | 0.91963899 | 0.91963899 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-full baseline | `predicted_positive_count` | 131 | 131 | 0.00e+00 | 0 | **PASS** |
| DISAB-full baseline | `selection_rate` | 0.00404896 | 0.00404896 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-full baseline | `max_dphi` | 0.00925842 | 0.00925842 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-full baseline | `demographic_parity_gap` | 0.01310644 | 0.01310644 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-full baseline | `equal_opportunity_gap` | 0.05366061 | 0.05366061 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-full canonical | `auroc` | 0.73438279 | 0.73438279 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-full canonical | `auprc` | 0.20274015 | 0.20274015 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-full canonical | `accuracy` | 0.91960809 | 0.91960809 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-full canonical | `predicted_positive_count` | 104 | 104 | 0.00e+00 | 0 | **PASS** |
| DISAB-full canonical | `selection_rate` | 0.00321444 | 0.00321444 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-full canonical | `max_dphi` | 0.00457553 | 0.00457553 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-full canonical | `demographic_parity_gap` | 0.00490456 | 0.00490456 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-full canonical | `equal_opportunity_gap` | 0.02776016 | 0.02776016 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-exclude baseline | `auroc` | 0.73525571 | 0.73525571 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-exclude baseline | `auprc` | 0.19896138 | 0.19896138 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-exclude baseline | `accuracy` | 0.91948445 | 0.91948445 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-exclude baseline | `predicted_positive_count` | 88 | 88 | 0.00e+00 | 0 | **PASS** |
| DISAB-exclude baseline | `selection_rate` | 0.00271991 | 0.00271991 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-exclude baseline | `max_dphi` | 0.01550818 | 0.01550818 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-exclude baseline | `demographic_parity_gap` | 0.00211012 | 0.00211012 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-exclude baseline | `equal_opportunity_gap` | 0.01351797 | 0.01351797 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-exclude canonical | `auroc` | 0.66697342 | 0.66697342 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-exclude canonical | `auprc` | 0.15924733 | 0.15924733 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-exclude canonical | `accuracy` | 0.91926810 | 0.91926810 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-exclude canonical | `predicted_positive_count` | 67 | 67 | 0.00e+00 | 0 | **PASS** |
| DISAB-exclude canonical | `selection_rate` | 0.00207084 | 0.00207084 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-exclude canonical | `max_dphi` | 0.00458622 | 0.00458622 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-exclude canonical | `demographic_parity_gap` | 0.00009984 | 0.00009984 | 0.00e+00 | 1.00e-10 | **PASS** |
| DISAB-exclude canonical | `equal_opportunity_gap` | 0.00387587 | 0.00387587 | 0.00e+00 | 1.00e-10 | **PASS** |

## Input hashes:
- Real microdata input: `data/processed/nhis/nhis_2022_2024_features.parquet` (SHA-256: `49f415132ff0be0228f8533f9f74c48cd79ff6fa8be66db085f7329d9b083383`, size: 2,740,478 bytes)
- Frozen reference directory (test): `docs/releases/NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7/`
- Frozen reference directory (train/val): `docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/`
- Protected baseline tag: `inherited-code-v0.3-baseline-20260828` (`038897e9f751edac6e36445b7706eec5fdb15988`)
- R1 closure commit: `d29f7e8fa2e0ab56f21de7407b7d7d921b7b9955`

## Output hashes:
Deliverable bundle directory: `artifacts/nhis_d8_r2/20260908T095751Z_f164e0af/`
- `execution_manifest.json`: `089418c331a05ff1d6174e77a0157bdd2841fdeefa0d3c91969749e68e1225db` (3,058 bytes)
- `environment_manifest.json`: `c055a71d1b83130d3245333c04f420ca34e952f1feee394cd228ec87a9cd29e4` (711 bytes)
- `frozen_reference_manifest.json`: `8a42789cc7f71c55f775c6b7aa4b11a0c7cf9ba5e938946ad3bc8f493f211b5a` (715 bytes)
- `cohort_reproduction.json`: `0adc0a9ff273327df115033f2fc6af537ee4b08199d8aca1888fd97f03254499` (1,250 bytes)
- `schema_reproduction.json`: `6b57d9eb3dc11bc3e083658cf30edeee0b6266c5ee71e1fa6ba84472b26294a5` (2,373 bytes)
- `canonical_state_reproduction.json`: `e72de624e8ba726331fdd0bdf3a85026f868b4547ad09978b2df9137b34bf2dc` (848 bytes)
- `metric_reproduction.json`: `eefc3b9ac4c4e8e0c9326e832426a016acd4a92a97ff0669351a109402c9ed6a` (18,340 bytes)
- `metric_reproduction.md`: `a89624dc1a343e7f130335eaba88443cfbcbc601edc13bb3965c7dc07e0445dc` (6,540 bytes)
- `protected_file_integrity.json`: `7635cf04fcea6e44338e2229d6ed05a116b9714aa8d53afefdcbb0dc7c285970` (4,437 bytes)
- `git_diff.patch`: `6a129215fbc672aea3e79d5c2cb273712ab04b47816ce6e780a396733f75cd36` (20,875 bytes)

## Row counts:
- Total rows in `nhis_2022_2024_features.parquet`: 85,992 rows.
- Partition sizes:
  - Temporal Train (2022): 24,365 rows.
  - Temporal Validation (2023): 29,277 rows (`D6_ARM_001`), 29,283 rows (`D6_ARM_002`), 29,282 rows (`D6_ARM_003` & `D6_ARM_004`).
  - Temporal Test (2024): 32,350 rows (`D6_ARM_001`), 32,355 rows (`D6_ARM_002`), 32,354 rows (`D6_ARM_003` & `D6_ARM_004`).
- Candidate enhancement models trained: **0** (Conditions 3 & 4 structurally bypassed in baseline reproduction mode).
- Enhancement steps executed: **0**.

## Assumptions:
1. **Scientific Reproduction Integrity**: D6 baseline and canonical FairBias representations were constructed using `ALGORITHM_MODE_PAPER_FAITHFUL` with stress-elbow MDS selection (`mds_fixed_components=None`) and `PRIMARY_D6_RANDOM_SEED = 0`. Setting runner default `random_seed=0` and faithful mode under `baseline_reproduction_only=True` faithfully reproduced all canonical state transformations and metric values bitwise.
2. **Metric Convention Alignment**: D6 canonical release evaluated `auprc` as Average Precision via `average_precision_score(y_true, y_score)`. The D8 runner computes this convention under `"auprc"` while preserving `"auprc_trapezoidal"` for trapezoidal area.
3. **Data Scope Discipline**: Real NHIS microdata access was strictly restricted to Conditions 1 and 2 for baseline reproduction. Conditions 3 and 4 were completely bypassed. No enhancement claims or results were evaluated on real data.
4. **Governance Incident Carry-Forward**: `R1D-GOV-01: CLOSED`. Commit `d29f7e8fa2e0ab56f21de7407b7d7d921b7b9955` was authorized by Codex supervisor and cleanly pushed.

## Unresolved issues:
1. Historical pre-existing `.gitignore` drift (`PRE_EXISTING_BASELINE_DRIFT_RECORDED_NOT_RESOLVED`) preserved untouched.
2. Substantive enhancement execution (Conditions 3 & 4) on real data remains unauthorized pending Codex review and formal approval to proceed to D8-R3.

## Git diff summary:
All working tree modifications relative to commit `d29f7e8fa2e0ab56f21de7407b7d7d921b7b9955`:
- `src/nhis_fairbias/d8_enhancement_runner.py`:
  - Added `baseline_reproduction_only: bool = False` argument and defaulted `random_seed: int = 0`.
  - Added real-data access guard: raises `PermissionError` if `allow_real_data=True` and `baseline_reproduction_only=False`.
  - Added structural short-circuit: when `baseline_reproduction_only=True`, the runner sets algorithm mode to `ALGORITHM_MODE_PAPER_FAITHFUL`, fits Condition 1 and Condition 2, and immediately returns before Condition 3 or Condition 4 are initialized (0 candidate models fit, 0 enhancement steps executed).
  - Aligned evaluation metric outputs: `"auprc"` computed via `average_precision_score` (exact D6 convention), added `"balanced_accuracy"`, `"f1"`, `"demographic_parity_gap"`, `"equal_opportunity_gap"`.
- `scripts/run_nhis_enhancement_study.py`:
  - Added `--baseline-reproduction-only` flag and `--random-seed` (default 0).
  - Added CLI safety check: `--allow-real-data` requires `--baseline-reproduction-only`.
- `tests/test_nhis_d8_synthetic_contracts.py`:
  - Added `TestD8R2BaselineReproductionContracts` test suite verifying structural bypass of Conditions 3/4, 0 candidate model fits, real-data safety guards, and CLI exclusivity.
- `scripts/reproduce_d6_baselines.py` (untracked):
  - Standalone verification script executing the 4 target arms, verifying cohorts and schemas, comparing against frozen D6 release references, asserting reproduction barriers, and exporting the complete audit bundle.

## Proposed next step:
Submit reproduction evidence bundle in `artifacts/nhis_d8_r2/20260908T095751Z_f164e0af/` and this report to Codex Supervisor for review. Await Codex gate verdict (`ACCEPT`, `REPAIR`) and formal authorization prior to initiating substantive enhancement evaluations in D8-R3.

STOP — waiting for Codex review.
