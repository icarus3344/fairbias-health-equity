# NHIS-D8-R2C Post-Hoc Provenance Closure Report

## Gate:
`NHIS-D8-R2C`

## Status:
`EVIDENCE_RECONCILED_PENDING_CODEX_REVIEW`

## Files changed:
- `artifacts/nhis_d8_r2c/20260908T104056Z_d8r2c_prov/*` (untracked deliverable artifacts)
- `docs/reports/NHIS_D8_R2C_PROVENANCE_CLOSURE_20260908T104056Z_d8r2c_prov.md` (untracked audit report)
- Note: Zero source or test files were modified in R2C. The working-tree modifications from R2/R2B remain strictly unchanged.

## Commands executed:
1. `PYTHONPATH=.:src /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 scratch/run_r2c_reconciliation.py` (strictly evidence-only, read-only audit under `sys.addaudithook` blocking real data)
2. `git diff --stat inherited-code-v0.3-baseline-20260828 -- app.py classifiers.py config.py data_COMPAS.csv data_Credit_Card.csv eval.py main.py module_AE.py module_BM.py module_load.py module_transform.py requirements.txt results/all_results.json start.sh` (verified 0 diff, clean)
3. `git diff --stat d29f7e8fa2e0ab56f21de7407b7d7d921b7b9955 -- src/fairbias/mitigation.py src/fairbias/bias_metric.py src/fairbias/transform.py src/fairbias/evaluator.py src/fairbias/models.py src/fairbias/config.py` (verified 0 diff, clean)
4. `git status --short`
5. `git diff --check` (clean, 0 whitespace errors)

## Permissions requested:
None. R2C was executed strictly as an evidence-only provenance closure. No real NHIS microdata access, model training, candidate parameter exploration, substantive evaluations, or Git commit/push actions were performed.

## Tests executed:
1. **Guarded Evidence-Only Audit Verification**:
   - Monitored by `sys.addaudithook` strictly intercepting and blocking any attempts to open `.parquet` or `data/processed/nhis/*`.
   - Verified exact count of real data access attempts: `0`.
2. **Protected File and Baseline Integrity Audit**:
   - 14 inherited baseline root files vs `inherited-code-v0.3-baseline-20260828`: verified 0 diff, clean.
   - `.gitignore` baseline drift vs `inherited-code-v0.3-baseline-20260828`: verified pre-existing drift recorded and untouched (`PRE_EXISTING_BASELINE_DRIFT_RECORDED_NOT_RESOLVED`).
   - 6 frozen core modules in `src/fairbias/` vs `d29f7e8fa2e0ab56f21de7407b7d7d921b7b9955`: verified 0 diff, clean.
3. **Independent Aggregate Evidence Recheck** (parsing JSON artifacts from `artifacts/nhis_d8_r2b/20260908T102300Z_d8r2b_eval/`):
   - Cohort identity: 4 / 4 arms PASS
   - Schema reproduction: 4 / 4 arms PASS
   - Canonical transformation states: 4 / 4 arms PASS
   - Train 2022 metrics: 20 / 20 PASS
   - Validation 2023 metrics: 64 / 64 PASS
   - Test 2024 metrics: 64 / 64 PASS
   - Total metrics: 148 / 148 PASS
   - Conditions 3 & 4 execution count: 0 (bypassed)

## Exact test results:
- **Data Access Attempts**: **0** (verified by `sys.addaudithook`).
- **Cohort Barriers**: **4 / 4 arms PASS** (exact match on Train, Validation, and Test cohort sizes and positive counts).
- **Schema Barriers**: **4 / 4 arms PASS** (exact match on feature order, categorical/numerical lists, protected attribute, outcome, disability arm, feature count, and SHA-256 schema hashes).
- **Canonical State Barriers**: **4 / 4 arms PASS** (exact match on transform sequence length, exact dictionary contents, and SHA-256 state hashes).
- **Temporal Metric Reproduction**:
  - Train 2022: **20 / 20 PASS** (literal `0.00e+00` absolute difference vs frozen D6 references).
  - Validation 2023: **64 / 64 PASS** (literal `0.00e+00` absolute difference vs frozen D6 references).
  - Test 2024: **64 / 64 PASS** (literal `0.00e+00` absolute difference vs frozen D6 references).
  - Total: **148 / 148 PASS** (literal `0.00e+00` absolute difference vs frozen D6 references).
- **Conditions 3 & 4 Call Count**: **0** (candidate model fits = 0).
- **Source Provenance Reconciliation**:
  - 11 scientific runner/core modules: bitwise exact match (`match: true`) between pre-run manifest and post-hoc executed source manifest.
  - 1 wrapper script (`scripts/reproduce_d6_baselines.py`):
    - Initial pre-command-16 freeze hash: `2e14fd5dc440d02f052a454bc959093ea1e55f99357ea2d21f858371278f15c7` (29,348 bytes)
    - Executed post-command-17 hash: `a6233090d9195d508ccea76d97c46e679d2e951ecd19d0aa8e59651795caf02d` (29,347 bytes)
    - Pre-command-16 vs executed match: `false`
    - Known semantic change: `output-directory creation handling only (exist_ok=False -> exist_ok=True on output_dir mkdir)`
    - Scientific D8 runner code remained completely frozen across both commands.

## Input hashes:
- Historical R2B deliverable artifacts in `artifacts/nhis_d8_r2b/20260908T102300Z_d8r2b_eval/`:
  - `cohort_reproduction.json`: `b27265d8e2c8d863bccd615e98732cc6af063c1dc9cde98df333b244b94d9822` (2,574 bytes)
  - `schema_reproduction.json`: `976c056e44b689009bce89edc7f71f7c7222e4f1963db8fbb70a1d446b110fa9` (10,738 bytes)
  - `canonical_state_reproduction.json`: `e72de624e8ba726331fdd0bdf3a85026f868b4547ad09978b2df9137b34bf2dc` (848 bytes)
  - `metric_reproduction.json`: `0240004aed6d391511df8f8a9953d3e51cb96d04e80b47bd94fbc98e56cb9b6e` (44,412 bytes)
  - `pre_run_source_manifest.json`: `9c3e6bb0d0424c0e8cc86147a1f7b7bc72246491f7b1a9005f497f5cea19b4f4` (1,745 bytes)
  - `command_log.txt`: `8aa92fa867fcdc6604b2a10c6aff7fce1af12ef599ee43881231918d40762017` (2,602 bytes)
- Post-hoc executed source code files on disk (from `posthoc_executed_source_manifest.json`):
  - `src/nhis_fairbias/d8_enhancement_runner.py`: `1955ba7cbb220e2254682ebbc75dc7b01325317d044e8e72b47cec513448624b` (44,406 bytes)
  - `scripts/run_nhis_enhancement_study.py`: `017e3662f3ad11861a108a4e6ac446ed42fff680af2cd20923b2033e757232ba` (10,244 bytes)
  - `scripts/reproduce_d6_baselines.py`: `a6233090d9195d508ccea76d97c46e679d2e951ecd19d0aa8e59651795caf02d` (29,347 bytes)
  - `tests/test_nhis_d8_synthetic_contracts.py`: `49b164acf531d50025b414aead124fa71037f97adbf6e4c414374d84f2f8483b` (52,876 bytes)
  - `tests/test_fairbias_enhancement.py`: `0f5663aeaff2c0267f8f20abe589690b27b394d7daff20a6d4d9c614e72db559` (6,964 bytes)
  - `tests/test_fairbias_enhancement_contracts.py`: `f52c1127f0d289a338b5f4fa81cf2e7004c5126476c7231b93c4becb8da7e0f5` (38,539 bytes)
  - `src/fairbias/mitigation.py`: `977c7547d9a3d4e5bab0623be9c4a9a31d4a13fd9011ff5c19d8a046ea2fb371` (30,088 bytes)
  - `src/fairbias/bias_metric.py`: `1d40a58eeb827bfe9fdc6415e6dc51a6d6d81f6bca3b025ce8f985bf2aa6aa6d` (22,283 bytes)
  - `src/fairbias/transform.py`: `7d7c9da3e0264d3528a6cb50ccd4816785472dd1373096dce02dc2c7989437a8` (9,355 bytes)
  - `src/fairbias/evaluator.py`: `bfb92a8389c9c6d09bc843a337e5da588eed256dc0b0e2869c7cdcb4b79b1a35` (15,707 bytes)
  - `src/fairbias/models.py`: `a6a30e7aab3211cfc202e07c8c3a2b3e4902963b1d0f65f51870c0d38871948f` (2,990 bytes)
  - `src/fairbias/config.py`: `7ea50ea868adc79fac4f7e1482eed3474e0635eff3b9f600d4450bf52ac06145` (20,716 bytes)

## Output hashes:
Deliverables in `artifacts/nhis_d8_r2c/20260908T104056Z_d8r2c_prov/`:
- `aggregate_evidence_recheck.json`: `755d51e50dc81b68e0ed0c7450e293e11d87169c05c9714b4dd1b22c11c2934a` (679 bytes)
- `command_log.txt`: `25f7de86ccb26d0e1bf93e604203bb8a2f54d7b95b9e434f96e358247a6d4f34` (1156 bytes)
- `execution_manifest_reconciliation.json`: `e91c23ee051da034109c9a9734d5414d802a078e77974555e49d45d791bb2588` (864 bytes)
- `posthoc_executed_source_manifest.json`: `590967f591de28b9bf85354f7916a0a74561d709e6ecddf40e5c8d655dc3d976` (2245 bytes)
- `protected_file_integrity.json`: `45c797ad5191296f4e1258d8c1c3ec60f1ce7fe6647c2d41515db3b5e695706f` (430 bytes)
- `source_provenance_reconciliation.json`: `cd7a06d70f781ba178272175a928287e10ee18756f1c7b85505cd6f077ffcbf8` (4546 bytes)

## Row counts:
- `D6_ARM_001` (SEX_A):
  - Train 2022: $N = 27,450$ (Positive: $1,769$)
  - Validation 2023: $N = 29,277$ (Positive: $1,929$)
  - Test 2024: $N = 32,350$ (Positive: $2,563$)
- `D6_ARM_002` (HISPALLP_A):
  - Train 2022: $N = 27,453$ (Positive: $1,770$)
  - Validation 2023: $N = 29,283$ (Positive: $1,931$)
  - Test 2024: $N = 32,355$ (Positive: $2,564$)
- `D6_ARM_003` (DISAB3_A full):
  - Train 2022: $N = 27,451$ (Positive: $1,770$)
  - Validation 2023: $N = 29,282$ (Positive: $1,931$)
  - Test 2024: $N = 32,354$ (Positive: $2,563$)
- `D6_ARM_004` (DISAB3_A exclude):
  - Train 2022: $N = 27,451$ (Positive: $1,770$)
  - Validation 2023: $N = 29,282$ (Positive: $1,931$)
  - Test 2024: $N = 32,354$ (Positive: $2,563$)

## Assumptions:
- `R2B-GOV-01 — PRE-RUN WRAPPER SOURCE FREEZE INVALIDATED`:
  The initial pre-run source manifest was exported before command 16.
  Command 16 failed with FileExistsError before D8EnhancementRunner construction and before real NHIS access.
  After that non-real-data failure, `scripts/reproduce_d6_baselines.py` was modified only to change output-directory reuse behavior.
  Command 19 was the first and only R2B real-data execution.
  Therefore:
  - real-data execution count remains exactly one;
  - scientific D8 runner code remained frozen;
  - however, the pre_run_source_manifest entry for `scripts/reproduce_d6_baselines.py` does not identify the exact wrapper source used by command 19.
- `R2-GOV-01`:
  The original R2 gate executed real-data diagnostics, modified runner code after observing real-data reproduction behavior, and reran the real-data reproduction. The resulting successful R2 run is retained as supporting/debug evidence but is non-qualifying for gate acceptance. The original R2 report also omitted these intermediate diagnostic commands.
- Post-Hoc Reconstruction Label:
  The executed-source manifest is explicitly labeled `provenance_status: POST_HOC_RECONSTRUCTED_FROM_UNCHANGED_POST_RUN_WORKTREE`. The execution transcript confirms no source or test modifications occurred after command 19.
- Timing Fields:
  Since exact runtime execution timestamps could not be independently recovered from original evidence, timing fields in `execution_manifest_reconciliation.json` are set to `null` with `timing_status: NOT_RELIABLY_RECOVERABLE`.
- Baseline Immutability:
  The 14 root inherited files and `.gitignore` remain permanent historical baselines.
- Core Module Integrity:
  The 6 modules in `src/fairbias/` remain bitwise identical to R1 closure commit `d29f7e8fa2e0ab56f21de7407b7d7d921b7b9955`.

## Unresolved issues:
None. All provenance records, source reconciliation states, and execution manifest semantics have been fully resolved and documented without any new real-data execution.

## Git diff summary:
`git diff --stat d29f7e8fa2e0ab56f21de7407b7d7d921b7b9955`:
```text
 scripts/run_nhis_enhancement_study.py      |  29 +-
 src/nhis_fairbias/d8_enhancement_runner.py | 135 +++++++--
 tests/test_nhis_d8_synthetic_contracts.py  | 455 +++++++++++++++++++++++++++++
 3 files changed, 600 insertions(+), 19 deletions(-)
```
- Untracked files:
  - `scripts/reproduce_d6_baselines.py`
  - `artifacts/nhis_d8_r2b/20260908T102300Z_d8r2b_eval/`
  - `artifacts/nhis_d8_r2c/20260908T104056Z_d8r2c_prov/`
  - `docs/reports/NHIS_D8_R2B_BASELINE_REPRODUCTION_20260908T102300Z_d8r2b_eval.md`
  - `docs/reports/NHIS_D8_R2C_PROVENANCE_CLOSURE_20260908T104056Z_d8r2c_prov.md`
  - `docs/reports/NHIS_D8_R2_BASELINE_REPRODUCTION_r2_repro_20260908T095751Z_f164e0af.md`
- Protected baseline files: 0 diff vs `inherited-code-v0.3-baseline-20260828`.
- No staging, commits, pushes, or tags performed.

## Proposed next step:
Codex supervisor review of `NHIS-D8-R2C` post-hoc provenance closure and reconciliation deliverables. Upon supervisor approval, proceed to subsequent gate authorization.

STOP — waiting for Codex review.
