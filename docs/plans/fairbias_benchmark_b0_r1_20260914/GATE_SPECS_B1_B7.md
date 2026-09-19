# Standalone Specifications for Future Benchmark Gates B1–B7 (B0-R1 Revision)

**Document ID**: `docs/plans/fairbias_benchmark_b0_r1_20260914/GATE_SPECS_B1_B7.md`  
**Execution Timestamp**: `2026-09-14T15:55:00+08:00`  
**Active Gate**: `FAIRBIAS-BENCHMARK-B0-R1`  
**Governance Authority**: [AI Execution Protocol](file:///Users/lkc/Downloads/code_v_0_3/docs/AI_EXECUTION_PROTOCOL.md) | [AGENTS.md](file:///Users/lkc/Downloads/code_v_0_3/AGENTS.md) | [GEMINI.md](file:///Users/lkc/Downloads/code_v_0_3/GEMINI.md)  
**Supervisor**: Codex Supervisor  
**Implementation Worker**: Gemini Implementation Worker  

---

## 1. Operating Instructions & Governance Invariants

Each gate specification in this document is **completely self-contained**. Any authorized agent or collaborator can execute a gate independently without relying on conversational context.

**Mandatory Governance Invariants**:
1. No gate may be activated until the preceding gate has received formal, independent written acceptance (`ACCEPT`) from the Codex Supervisor.
2. The 14 inherited baseline root files and `.gitignore` are permanent and immutable.
3. Microdata privacy is strictly enforced: zero raw records in logs, stdout, or reports.
4. Unique Run Directories: All pipeline executions must write to unique, timestamped directories (e.g. `runs/benchmark_v1_<timestamp>_<run_id>/`). If a target directory already exists, execution must fail closed. Overwriting is strictly prohibited.
5. Localized Failure Rollback: In the event of a gate failure or reject verdict, rollback is restricted strictly to the changes introduced during that active gate; pre-existing worker candidate files and baseline tags must never be disturbed.
6. Every gate concludes with the standardized Section 9 report terminating with:  
   `STOP — waiting for Codex review.`

---

## 2. Gate B1: Correctness & Interface Remediation

- **Gate ID**: `FAIRBIAS-BENCHMARK-B1`
- **Objective**: Remediate the core correctness issues (C01–C06, E01) identified during supervisor audits, establishing a sound code foundation before building the application benchmark.
- **Prerequisites & Dependencies**: Formal `ACCEPT` of Gate B0-R1 by Codex Supervisor.
- **Environment Precondition**: Framework Python (`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`) possessing `scikit-learn 1.7.1`, `numpy 2.2.6`, `pandas 2.3.1`, `scipy 1.16.1`, `pytest 8.4.1`.
- **Permitted File Modifications**:
  - Source files:
    - [`src/fairbias/enhancement_state.py`](file:///Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_state.py)
    - [`src/fairbias/enhancement.py`](file:///Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement.py)
    - [`src/fairbias/enhancement_contracts.py`](file:///Users/lkc/Downloads/code_v_0_3/src/fairbias/enhancement_contracts.py)
    - [`src/fairbias/bias_metric.py`](file:///Users/lkc/Downloads/code_v_0_3/src/fairbias/bias_metric.py)
    - [`src/fairbias/evaluator.py`](file:///Users/lkc/Downloads/code_v_0_3/src/fairbias/evaluator.py)
    - [`src/nhis_fairbias/adapter.py`](file:///Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/adapter.py)
    - [`src/nhis_fairbias/preprocessing.py`](file:///Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/preprocessing.py)
    - [`src/nhis_fairbias/survey.py`](file:///Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/survey.py)
    - [`src/nhis_fairbias/d8_enhancement_runner.py`](file:///Users/lkc/Downloads/code_v_0_3/src/nhis_fairbias/d8_enhancement_runner.py)
  - Test files:
    - [`tests/test_audit_remediation_probes.py`](file:///Users/lkc/Downloads/code_v_0_3/tests/test_audit_remediation_probes.py)
    - [`tests/test_fairbias_enhancement_contracts.py`](file:///Users/lkc/Downloads/code_v_0_3/tests/test_fairbias_enhancement_contracts.py)
    - [`tests/test_nhis_d8_synthetic_contracts.py`](file:///Users/lkc/Downloads/code_v_0_3/tests/test_nhis_d8_synthetic_contracts.py)
    - [`tests/test_nhis_survey.py`](file:///Users/lkc/Downloads/code_v_0_3/tests/test_nhis_survey.py)
    - [`tests/test_nhis_d6_temporal.py`](file:///Users/lkc/Downloads/code_v_0_3/tests/test_nhis_d6_temporal.py)
- **Data & Network Permissions**: Default Deny. Zero network access. Zero reading of real NHIS microdata (parquet/csv). Execution is strictly 100% synthetic under persistent file-access guards.
- **Specific Implementation Tasks**:
  1. **C01 (State & Config Cache)**: Route runtime AE, Joint BM/AE, and candidate trackers explicitly to `v2_lossless` state hashing. Bind classifier `get_params(deep=True)` and preprocessing transformers to configuration fingerprints.
  2. **C02 (Registry Compatibility)**: Inspect `preprocessor.registry` and semantic sections (`primary_core`, substantive codes, missing rules). Fail closed on altered substantive code dictionaries.
  3. **C03 (Record Identity)**: Enforce strict integer normalization for survey years; construct structured record keys `(source, int(year), str(id_norm))` to prevent float representation leaks (`2023` vs `2023.0`).
  4. **C04 (Probability Contracts)**: Validate classifier probability outputs (shape, finite values in $[0, 1]$, row sum $= 1.0$, binding to positive class via `classes_`). Distinguish event risk $p$ from randomized decision policy $q$.
  5. **C05 (Categorical Geometry)**: In divergence calculation, ensure the denominator $k = \max(1, \text{len}(\text{union\_index}))$ is computed strictly over non-empty observed categories, maintaining invariance when unused levels exist in categorical series.
  6. **C06 (Unified Metrics)**: Establish unified metric calculations using extreme range $\max - \min$ and explicit `NOT_ESTIMABLE` handling for single-group slices.
  7. **E01 (Weight Sum Overflow)**: Add sum-finiteness validation and stable normalization for survey weights.
- **Planned Execution Commands**:
  ```bash
  /Library/Frameworks/Python.framework/Versions/3.13/bin/pytest tests/test_audit_remediation_probes.py -v
  /Library/Frameworks/Python.framework/Versions/3.13/bin/pytest tests/test_fairbias_enhancement_contracts.py -v
  /Library/Frameworks/Python.framework/Versions/3.13/bin/pytest tests/test_nhis_d8_synthetic_contracts.py -v
  ```
- **Acceptance Verification Matrix**:
  - Core test suite under persistent data guard: 100% passing tests.
  - Previous 13 remediation probes: 13/13 PASS.
  - Fourth Review calling-path probes (lossless routing, real registry check, integer year): 7/7 PASS.
  - Extended boundary probes (probability bounds, categorical level invariance, weight overflow): 12/12 PASS.
  - Historical 4-arm regression test against `v1` legacy hashes: PASS.
- **Failure Protocol**: If any regression or legacy test fails, revert ONLY changes made in Gate B1.
- **Exit Condition**: Complete pass of all synthetic regression suites; submit Section 9 audit report; `STOP — waiting for Codex review.`.

---

## 3. Gate B2: Benchmark Data Contracts, Semantic Representation & Survey Inference

- **Gate ID**: `FAIRBIAS-BENCHMARK-B2`
- **Objective**: Implement the Benchmark V1 architecture: F/C/S/T data loaders, 3-tier semantic/one-hot feature encoders, unified survey-weighted evaluation metrics, and the rescaled PSU bootstrap inference engine.
- **Prerequisites & Dependencies**: Formal `ACCEPT` of Gate B1 by Codex Supervisor.
- **Environment Precondition**: Framework Python (`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`).
- **Permitted File Modifications**:
  - Additive benchmark package directory: `src/nhis_fairbias/benchmark/`
    - `__init__.py`
    - `data_contracts.py`
    - `preprocessing.py`
    - `metrics.py`
    - `survey_inference.py`
    - `selection.py`
  - Benchmark configuration directory: `configs/nhis/benchmark_v1/`
  - Benchmark unit tests: `tests/benchmark/`
- **Data & Network Permissions**: Default Deny. Zero network access. Zero real NHIS microdata reads. All data contracts and inference routines verified on synthetic data with known ground-truth statistical properties.
- **Specific Implementation Tasks**:
  1. Implement `data_contracts.py` generating partition dictionaries with stable record keys, eligibility masks, and PSU-level F/C splitting via `np.random.Generator(np.random.PCG64(20260913))`.
  2. Implement `preprocessing.py` enforcing the 3-tier representation (semantic layer, geometric layer, and one-hot prediction layer fit on F with reserved `'MISSING'` and `'UNKNOWN'` tokens).
  3. Implement `metrics.py` calculating survey-weighted confusion matrices, BA, selection rates, DP gap, and EO gap over `expected_groups`, returning `NOT_ESTIMABLE` on missing group support.
  4. Implement `survey_inference.py` executing rescaled PSU bootstrap ($B=2000$, seed `20260914`), sample variance calculation ($B_{\text{eff}}-1$ denominator), $df = \sum_h m_h - H$, paired contrast calculations ($\Delta\text{BA}, \Delta\text{EO}$), singleton stratum detection, and Bonferroni adjustments.
  5. Implement `selection.py` executing pre-registered selection rules on Set S (2023) across operating points (0.10, 0.05, 0.20) with deterministic tie-breaking and seed completeness verification.
- **Planned Execution Commands**:
  ```bash
  /Library/Frameworks/Python.framework/Versions/3.13/bin/pytest tests/benchmark/test_data_contracts.py -v
  /Library/Frameworks/Python.framework/Versions/3.13/bin/pytest tests/benchmark/test_metrics.py -v
  /Library/Frameworks/Python.framework/Versions/3.13/bin/pytest tests/benchmark/test_survey_inference.py -v
  /Library/Frameworks/Python.framework/Versions/3.13/bin/pytest tests/benchmark/test_selection.py -v
  ```
- **Acceptance Verification Matrix**:
  - Hand-calculated survey-weighted examples match analytical expectations to $10^{-6}$.
  - Rescaled bootstrap variance matches standard survey linearization benchmarks on synthetic test designs within stochastic tolerance.
  - Test label permutation test: permuting outcome labels on Set T produces zero change in F-fitted transformers, C-calibrated thresholds, or S-selected configurations.
  - Category permutation test: permuting category labels produces identical one-hot predictions.
- **Exit Condition**: Full test pass on synthetic benchmark suite; submit Section 9 audit report; `STOP — waiting for Codex review.`.

---

## 4. Gate B3: External Fairness Method Adapters & Dependency Management

- **Gate ID**: `FAIRBIAS-BENCHMARK-B3`
- **Objective**: Create an isolated benchmark virtual environment, install pinned wheels for `fairlearn` and `aif360`, and implement standardized method adapters for all benchmark algorithms across LR and `GradientBoostingClassifier` backbones.
- **Prerequisites & Dependencies**: Formal `ACCEPT` of Gate B2 by Codex Supervisor; explicit Supervisor authorization for network access to PyPI.
- **Environment Precondition**: Dedicated virtual environment (e.g. `./.venv_benchmark/`).
- **Permitted File Modifications**:
  - Dependency lockfile: `configs/nhis/benchmark_v1/requirements_benchmark.lock`
  - Adapter directory: `src/nhis_fairbias/benchmark/adapters/`
    - `base.py`
    - `adapter_unmitigated.py`
    - `adapter_fairbias.py`
    - `adapter_reweighing.py`
    - `adapter_lfr.py`
    - `adapter_reductions.py`
    - `adapter_threshold_optimizer.py`
  - Adapter test suite: `tests/benchmark/test_adapters.py`
- **Data & Network Permissions**: Network access allowed **ONLY** for HTTPS downloads of pinned packages from `https://pypi.org` and `https://files.pythonhosted.org`. Downloads must use atomic `.part` verification. Zero real microdata access.
- **Specific Implementation Tasks**:
  1. Build isolated virtual environment; install exact pinned releases of `fairlearn`, `aif360`, and underlying dependencies; generate immutable provenance manifest with SHA-256 hashes.
  2. Implement `base.py` defining standard `MethodAdapter` interface with explicit capability declarations.
  3. Implement `adapter_unmitigated.py` for standard LR and `GradientBoostingClassifier` backbones.
  4. Implement `adapter_fairbias.py` encapsulating `FairBias-BM application-v1` with supervised fit (NMI gate) and strict feasibility ($\text{slack}=0$).
  5. Implement `adapter_reweighing.py` wrapping AIF360 multi-group Reweighing, isolating sample weights from survey weights.
  6. Implement `adapter_lfr.py` wrapping AIF360 LFR, isolating continuous reconstructed features, preserving original labels $y$, and enforcing `NOT_SUPPORTED` for Arm 002.
  7. Implement `adapter_reductions.py` wrapping Fairlearn Exponentiated Gradient (DP and EO), routing oracle cost weights, and computing decision probability $q$.
  8. Implement `adapter_threshold_optimizer.py` wrapping Fairlearn ThresholdOptimizer on Set C with `prefit=True`, requiring $A$ during inference.
- **Planned Execution Commands**:
  ```bash
  ./.venv_benchmark/bin/pytest tests/benchmark/test_adapters.py -v
  ```
- **Acceptance Verification Matrix**:
  - Each adapter executes `fit`, `predict_proba` (or `predict_decision_proba`), and `predict` on synthetic cohorts.
  - LFR on Arm 002 (7-group) raises explicit `NotSupportedError`.
  - EG reduction oracle cost weights verified to alter base estimator loss function.
  - Inference methods proven to have zero dependence on ground-truth outcome labels $y$.
- **Exit Condition**: All adapters pass synthetic integration suite; submit Section 9 audit report; `STOP — waiting for Codex review.`.

---

## 5. Gate B4: Full Synthetic Dress Rehearsal & Protocol Freeze

- **Gate ID**: `FAIRBIAS-BENCHMARK-B4`
- **Objective**: Execute a complete dress rehearsal of the benchmark matrix on synthetic data (using downscaled cohorts for combinatorial paths plus isolated stress cases), stress-testing computation, memory, serialization, and error recovery, and permanently freezing the experimental protocol.
- **Prerequisites & Dependencies**: Formal `ACCEPT` of Gate B3 by Codex Supervisor.
- **Environment Precondition**: Benchmark virtual environment (`./.venv_benchmark/`).
- **Permitted File Modifications**:
  - Pipeline runner: `src/nhis_fairbias/benchmark/runner.py`
  - Synthetic rehearsal script: `scripts/run_benchmark_synthetic_rehearsal.py`
  - Frozen protocol manifest: `configs/nhis/benchmark_v1/frozen_protocol_manifest.json`
  - Rehearsal tests: `tests/benchmark/test_pipeline_rehearsal.py`
- **Data & Network Permissions**: Default Deny. Zero network access. Zero real NHIS microdata access. Full synthetic rehearsal.
- **Specific Implementation Tasks**:
  1. Generate synthetic 3-year multi-arm dataset mimicking the structure, sample sizes, and PSU clustering of NHIS 2022, 2023, and 2024.
  2. Execute the 76-condition benchmark matrix across LR and `GradientBoostingClassifier` backbones.
  3. Validate timeout handling (30 min wall-clock per fit) and memory tracking (4 GiB limit).
  4. Inject negative test cases (missing demographic groups, single-class labels) and verify that pipeline produces structured null returns (`NOT_ESTIMABLE`) without crashing.
  5. Verify checkpointing and resume capability: simulate interrupted execution and confirm recovery with incremented `attempt_id`.
  6. Test serialization: export all fitted models, reload from disk, and assert bitwise identical predictions.
  7. Generate and record the `frozen_protocol_manifest.json` containing exact Git commit, parameter grids, seed schedules, operating points, and source code SHA-256 hashes.
- **Planned Execution Commands**:
  ```bash
  ./.venv_benchmark/bin/python scripts/run_benchmark_synthetic_rehearsal.py --output-dir runs/benchmark_v1_synthetic_rehearsal/
  ./.venv_benchmark/bin/pytest tests/benchmark/test_pipeline_rehearsal.py -v
  ```
- **Acceptance Verification Matrix**:
  - All conditions complete successfully or terminate with documented expected statuses (`VALID`, `NOT_SUPPORTED`, `BUDGET_EXHAUSTED`, `NOT_ESTIMABLE`).
  - Model reload test: 100% bitwise parity on predictions.
  - Frozen protocol manifest verified by Codex Supervisor.
- **Exit Condition**: Complete synthetic rehearsal execution; immutable frozen manifest generated; submit Section 9 audit report; `STOP — waiting for Codex review.`.

---

## 6. Gate B5a: Real NHIS Support & Distribution Audit (2022 & 2023 Only)

- **Gate ID**: `FAIRBIAS-BENCHMARK-B5a`
- **Objective**: Audit the actual distribution, missingness rates, subgroup sample sizes, and survey design characteristics in authorized NHIS 2022 and 2023 data, establishing empirical support without fitting models or calculating predictive performance.
- **Prerequisites & Dependencies**: Formal `ACCEPT` of Gate B4 by Codex Supervisor; explicit Supervisor authorization specifying exact local NHIS 2022 and 2023 file paths and checksums.
- **Permitted File Modifications**:
  - Audit script: `scripts/audit_nhis_support.py`
  - Output report: `docs/reports/NHIS_REAL_DATA_SUPPORT_AUDIT_202609XX.md`
- **Data & Network Permissions**: Read-only access to authorized local NHIS 2022 and 2023 data files. **READING NHIS 2024 IS STRICTLY FORBIDDEN**. Zero network access.
- **Specific Implementation Tasks**:
  1. Verify SHA-256 hashes of authorized local NHIS 2022 and 2023 raw data files against pre-registered provenance records.
  2. Execute master PSU-level F/C partitioning on the 2022 master design table using `np.random.Generator(np.random.PCG64(20260913))`.
  3. Apply eligibility filtering for `MEDDL12M_A` and demographic arms (Arms 001–004).
  4. Audit subgroup cell counts, positive/negative event counts, Kish effective sample sizes, and item non-response rates.
  5. Audit and record estimability support; if certain minority cells in Arm 002 lack events, document the expected `NOT_ESTIMABLE` outcomes without altering seeds or models.
  6. Confirm that zero model training, threshold calibration, or performance evaluations are executed.
- **Planned Execution Commands**:
  ```bash
  ./.venv_benchmark/bin/python scripts/audit_nhis_support.py --years 2022,2023 --output-report docs/reports/NHIS_REAL_DATA_SUPPORT_AUDIT_202609XX.md
  ```
- **Acceptance Verification Matrix**:
  - Zero microdata records output to stdout, logs, or reports.
  - File access audit confirms zero access attempts to 2024 data files.
  - Comprehensive report on subgroup support across all 4 arms.
- **Exit Condition**: Support audit approved by Codex Supervisor; submit Section 9 audit report; `STOP — waiting for Codex review.`.

---

## 7. Gate B5b: Real Data Model Fitting on F/C & Configuration Selection on S

- **Gate ID**: `FAIRBIAS-BENCHMARK-B5b`
- **Objective**: Execute model fitting on Partition F, calibrate decision thresholds and postprocessors on Partition C, evaluate all candidates on Set S (2023), and freeze winning configurations for all operating points prior to opening 2024 data.
- **Prerequisites & Dependencies**: Formal `ACCEPT` of Gate B5a by Codex Supervisor.
- **Permitted File Modifications**:
  - Batch runner script: `scripts/run_benchmark_dev_execution.py`
  - Output directory: `runs/benchmark_v1_dev_<timestamp>_<run_id>/`
  - Frozen selection bundle: `artifacts/benchmark_v1/frozen_selected_models_manifest.json`
- **Data & Network Permissions**: Read-only access to authorized NHIS 2022 and 2023 files. **READING NHIS 2024 IS STRICTLY FORBIDDEN**. Zero network access.
- **Specific Implementation Tasks**:
  1. Execute Phase 1 (LR Core, 27 conditions): Fit models on F, calibrate on C, evaluate candidates on S.
  2. Execute Phase 2 (GBDT Core, 27 conditions): Fit `GradientBoostingClassifier` on F, calibrate on C, evaluate on S.
  3. Execute Phase 3 (FairBias AE and Ablations, 22 conditions): Fit on F, calibrate on C, evaluate on S.
  4. Enforce seed completeness: Disqualify configurations with partial seed failures from winning selection.
  5. Apply frozen selection rules on Set S to identify winning configurations for operating points (0.10, 0.05, 0.20) and unmitigated predictive reference.
  6. Serialize winning model artifacts and thresholds, computing SHA-256 hashes into `frozen_selected_models_manifest.json`.
- **Planned Execution Commands**:
  ```bash
  ./.venv_benchmark/bin/python scripts/run_benchmark_dev_execution.py --config configs/nhis/benchmark_v1/frozen_protocol_manifest.json --output-dir runs/benchmark_v1_dev_202609XX/
  ```
- **Acceptance Verification Matrix**:
  - Zero read operations on 2024 data.
  - All 76 conditions executed or accounted for with verified failure reasons.
  - Winning model hashes verified and reproducible.
- **Exit Condition**: Winning configurations locked and verified by Codex Supervisor; submit Section 9 audit report; `STOP — waiting for Codex review.`.

---

## 8. Gate B6: Frozen Cross-Year Evaluation on Set T (2024 Only)

- **Gate ID**: `FAIRBIAS-BENCHMARK-B6`
- **Objective**: Conduct out-of-year evaluation on NHIS 2024 by applying the frozen models and thresholds selected in Gate B5b, computing complex survey-weighted metrics, paired contrasts, and rescaled bootstrap confidence intervals.
- **Prerequisites & Dependencies**: Formal `ACCEPT` of Gate B5b by Codex Supervisor; explicit Supervisor authorization to access NHIS 2024 file.
- **Permitted File Modifications**:
  - Evaluation runner script: `scripts/run_benchmark_eval_2024.py`
  - Output directory: `runs/benchmark_v1_eval_2024_<timestamp>_<run_id>/`
  - Summary metrics export: `artifacts/benchmark_v1/final_evaluation_metrics_2024.json`
- **Data & Network Permissions**: Read-only access to authorized local NHIS 2024 data file. Zero network access.
- **Specific Implementation Tasks**:
  1. Load frozen model bundle and verify all SHA-256 hashes against `frozen_selected_models_manifest.json`.
  2. Load and filter NHIS 2024 cohort for Arms 001–004.
  3. Execute forward inference (`predict` / `predict_proba`) using frozen models and calibrated thresholds. Enforce programmatic block on any `.fit()` method call.
  4. Compute survey-weighted evaluation metrics (BA, EO gap, AP, AUROC, Brier, per-group TPR/FPR) on Set T.
  5. Compute paired contrasts ($\Delta\text{BA}, \Delta\text{EO}$) across the 20 primary confirmatory comparisons.
  6. Execute $B=2000$ rescaled PSU bootstrap replicates using identical replicate weights across all models, deriving Bonferroni-adjusted 99.75% confidence intervals and unadjusted 95% intervals.
- **Planned Execution Commands**:
  ```bash
  ./.venv_benchmark/bin/python scripts/run_benchmark_eval_2024.py --models artifacts/benchmark_v1/frozen_selected_models_manifest.json --output-dir runs/benchmark_v1_eval_2024_202609XX/
  ```
- **Acceptance Verification Matrix**:
  - Zero `.fit()` calls during execution.
  - Paired contrasts computed on identical respondent rows and identical bootstrap replicate weights.
  - Model weights verified unchanged before and after inference.
- **Exit Condition**: Final aggregate evaluation metrics approved by Codex Supervisor; submit Section 9 audit report; `STOP — waiting for Codex review.`.

---

## 9. Gate B7: Scientific Manuscript, Figures, Tables & Reproduction Package

- **Gate ID**: `FAIRBIAS-BENCHMARK-B7`
- **Objective**: Synthesize the verified empirical findings into publication-ready scientific artifacts, including figures, tables, TRIPOD+AI transparency disclosures, and an automated reproduction package.
- **Prerequisites & Dependencies**: Formal `ACCEPT` of Gate B6 by Codex Supervisor.
- **Permitted File Modifications**:
  - Manuscript source files: `docs/manuscript/`
  - Visualization artifacts: `docs/figures/`
  - Reproduction orchestration script: `scripts/reproduce_benchmark_results.py`
  - Public archive documentation: `docs/REPRODUCTION_GUIDE.md`
- **Data & Network Permissions**: Zero network access. Zero export or publication of individual microdata records. Only aggregate statistics and source code may be packaged.
- **Specific Implementation Tasks**:
  1. Generate Figure 1: NHIS cohort flow diagram and temporal F/C/S/T design architecture.
  2. Generate Figure 2: Survey-weighted 2D Pareto trade-off curves (Balanced Accuracy vs Equalized Odds Gap) on Set S and frozen points on Set T.
  3. Generate Figure 3: Subgroup-level identification rates ($TPR_g$) and false alarm rates ($FPR_g$) across demographic arms.
  4. Generate Table 1: Demographic and survey design characteristics across 2022, 2023, and 2024 cohorts.
  5. Generate Table 2: Primary confirmatory comparison table showing $\Delta\text{BA}$ and $\Delta\text{EO}$ with Bonferroni-adjusted bootstrap confidence intervals across the 20 contrasts.
  6. Generate Supplementary Tables: Full 76-condition results, sensitivity analyses (unweighted metrics, $t=0.5$ threshold, weighted training), and computational resource consumption.
  7. Draft method section and discussion adhering to TRIPOD+AI, highlighting both the strengths and empirical costs of FairBias-BM.
  8. Build automated reproduction script verifying that all reported figures and tables regenerate deterministically from aggregate JSON artifacts.
- **Planned Execution Commands**:
  ```bash
  ./.venv_benchmark/bin/python scripts/reproduce_benchmark_results.py --input-dir artifacts/benchmark_v1/ --output-dir docs/figures/
  ```
- **Acceptance Verification Matrix**:
  - 100% of reported numerical values in text and tables match frozen Gate B6 JSON artifacts bit-for-bit.
  - Zero individual microdata rows present in any public file or commit.
  - TRIPOD+AI checklist completed and verified.
- **Exit Condition**: Manuscript and reproduction package formally approved by Codex Supervisor and project owner; submit Section 9 audit report; `STOP — waiting for Codex review.`.
