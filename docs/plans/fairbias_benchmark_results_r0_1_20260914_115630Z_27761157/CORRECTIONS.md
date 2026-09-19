# CORRECTIONS & AUDIT REMEDIATION: Point-by-Point Resolution of Supervisory Findings R01–R08

- **Date**: 2026-09-14
- **Gate**: **RESULTS-R0.1 (Targeted Remediation & Executable Specification)**
- **Supervisory Audit Reference**: `docs/reports/FAIRBIAS_RESULTS_R0_SUPERVISOR_REVIEW_20260914.md`
- **Single Source of Truth**: `docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/FACTS.json`
- **Condition Registry**: `docs/plans/fairbias_benchmark_results_r0_1_20260914_115630Z_27761157/CONDITION_REGISTRY.json`
- **Execution Constraint**: Static remediation only. No code implementation, no model runs, no pytest execution, no microdata access.

---

## 1. Remediation of Finding R01: Core Defects vs. Adapter Scope (R1 Splitting)

### 1.1 Concession & Analysis
In the previous R0 specification, implementation scope was restricted to `src/nhis_fairbias/benchmark/`, while claiming alignment with historical defects C01–C06 and E01. As the supervisor correctly noted, modifying adapter wrappers alone cannot remediate defects residing in core legacy files:
- **C01 (Lossless State Identity & Caching)**: Located in `src/fairbias/enhancement_state.py`, `enhancement.py`, `enhancement_contracts.py`, and `src/nhis_fairbias/d8_enhancement_runner.py`.
- **C02 (Registry Compatibility)**: Located in `src/nhis_fairbias/adapter.py` and `src/nhis_fairbias/preprocessing.py`.
- **C03 (Record Key Specification)**: Located in `src/fairbias/enhancement_contracts.py` (and downstream `data_contracts.py`).
- **C04 (Probability Output Contract)**: Located in `src/fairbias/enhancement_contracts.py` and runner evaluators.
- **C05 (Categorical Support Sets)**: Located in `src/fairbias/bias_metric.py`.
- **C06 (Extreme Gap Metrics)**: Located in `src/fairbias/evaluator.py` and benchmark metrics.
- **E01 (Weight Sum Overflow)**: Located in `src/nhis_fairbias/survey.py`.

Leaving these files untouched while claiming defects are addressed would constitute application-level bypassing rather than architectural closure.

### 1.2 Action Taken
Phase R1 is formally bifurcated into two sequential, independently verified sub-phases:
1. **Phase R1A (Core Public Contracts & Geometry Configuration Propagation)**:
   - File scope: `src/fairbias/enhancement_state.py`, `src/fairbias/enhancement.py`, `src/fairbias/enhancement_contracts.py`, `src/fairbias/bias_metric.py`, `src/fairbias/evaluator.py`, `src/fairbias/config.py`, `src/nhis_fairbias/adapter.py`, `src/nhis_fairbias/survey.py`.
   - Remediates C01–C06 and E01 at their original sources and verifies configuration propagation.
2. **Phase R1B (FairBias & Comparator Adapter Public Implementation)**:
   - File scope: `src/nhis_fairbias/benchmark/adapters/*.py`, `metrics.py`, `preprocessing.py`, `data_contracts.py`.
   - Implements strict F/C/S/T lifecycle, $p/q/\hat{y}$ separation, master PSU design, and comparator contracts.

---

## 2. Remediation of Finding R02: True BM Execution & AE/Joint Configuration

### 2.1 Concession & Analysis
The previous R0 specification mandated "invoking the true BM engine" but lacked concrete technical specifications for how FairBias-BM, AE, and Joint variants execute:
1. **Multi-group Aggregation Discrepancy**: In `src/fairbias/bias_metric.py` (line 413), the `author_max_pair` branch computes $\text{mean}_{c} |\max(v_1) - \max(v_2)|$, whereas true maximal pairwise disparity across groups is $\max_{g_1, g_2} |v_{g_1} - v_{g_2}|$. Furthermore, `src/fairbias/evaluator.py` (line 294) hardcoded default calls to `compute_dphi_matrix` without propagating the aggregation parameter.
2. **Author BM Power Sequence vs. AE Poly Grid**: BM in Tang et al. (2024) utilizes the interleaved sequence $[3, 1/3, 5, 1/5, \dots, 1999, 1/1999]$ (`src/fairbias/config.py:79`), whereas the 6-value `DEFAULT_POLY_GRID = (0.5, 2.0, 0.33, 3.0, 0.25, 4.0)` belongs to the AE heuristic search. Mixing them compromises algorithm fidelity.
3. **Paper Mode Configuration Incompatibility**: In `src/fairbias/config.py` (line 240), `tang2024_paper_faithful` explicitly asserts `use_accuracy_enhancement is False`. Therefore, evaluating 16 AE conditions cannot simply toggle this flag in paper mode.

### 2.2 Action Taken
In `R1_EXECUTION_SPEC.md`:
- **Algorithmic Rules Frozen**:
  - $H=1$ exclusion context (holding at most one attribute constant, `bias_metric.py:99`).
  - True BM interleaved power stream $[3, 1/3, 5, 1/5, \dots, 1999, 1/1999]$ with restart revisit on stagnation.
  - Multi-group aggregation explicitly parameterized and routed through `evaluator.py` into `compute_dphi_matrix`.
  - NMI utility gate evaluated on Partition F.
- **Application Profile Isolation**:
  - Primary BM is registered as `FairBias-BM application-v1` (paper-faithful BM geometry on Partition F).
  - AE and Joint variants are registered under explicit application profiles (`application-v1-ae` and `application-v1-joint`), holding `epsilon_candidate` identical to the corresponding BM condition, with strict geometric feasibility ($\text{slack} = 0$).

---

## 3. Remediation of Finding R03: Robust Interface Acceptance & Training Semantics

### 3.1 Concession & Analysis
1. **EG Weight Handling**: Routing external weights to the base estimator does not implement a survey-weighted reduction Moment. Fairlearn's `ExponentiatedGradient` algorithm optimizes an unweighted error objective against empirical constraint violations.
2. **ThresholdOptimizer Unified Fit**: Merely fixing the keyword argument (`sample_weight` vs `sample_weight_F`) leaves the unified `fit()` method calibrating thresholds on the same training set used to fit the base estimator (`adapter_threshold_optimizer.py:91-92`), causing severe overfitting.
3. **Deterministic Output Semantics**: Returning `NOT_APPLICABLE` for decision probability $q$ requires that the public runner dispatches to thresholded decisions $\hat{y} = \mathbb{I}(p \ge t^*)$ based on Set C calibration.

### 3.2 Action Taken
In `R1_EXECUTION_SPEC.md`:
- **EG Protocol**: In primary unweighted conditions, external survey weights are strictly rejected with an informative error. Reduction cost weights generated internally by EG are verified via test spies to reach the base estimator.
- **TO Protocol**: Unified `fit()` must either require explicit, disjoint `(X_F, y_F)` and `(X_C, y_C, A_C)` arguments or fail closed before initiating any fitting. Calibrating on Set F is prohibited.
- **Threshold Freezing on C**: For all probabilistic models (Unmitigated, FairBias, RW, LFR), a global threshold $t^*$ is calibrated strictly on Partition C maximizing unweighted Balanced Accuracy across midpoints and boundary candidates (ties broken by distance to 0.5, then larger threshold), per Master Plan Section 5.1. $\hat{y}$ is frozen for Set S and Set T evaluation.

---

## 4. Remediation of Finding R04: Condition Registry Alignment (76 Conditions)

### 4.1 Concession & Analysis
In the R0 specification, the 4 survey-weighted training sensitivity conditions and 2 geometric path conditions drifted from the previously drafted experiment plan (`EXPERIMENT_REGISTRY_DRAFT.md`). Specifically, weighted training was altered to Arm 001 with dual backbones, rather than Arm 003 and Arm 004 with LR, and the path conditions were unspecified.

### 4.2 Action Taken
The canonical 76-condition master matrix from `EXPERIMENT_REGISTRY_DRAFT.md` is restored in full in `CONDITION_REGISTRY.json`:
1. **Core Conditions (54)**: 7 methods $\times$ 4 arms $- 1$ (LFR Arm 002) = 27 conditions $\times$ 2 backbones (LR, GBDT).
2. **AE Ablation Conditions (16)**: 2 variants (BM$\to$AE, Joint) $\times$ 4 arms $\times$ 2 backbones.
3. **Survey-Weighted Downstream Training (4)**: Arm 003 and Arm 004 $\times$ LR backbone $\times$ {Unmitigated, FairBias-BM} (weights applied only to downstream predictor).
4. **Geometric Path Sensitivity (2)**: Arm 004 $\times$ LR backbone $\times$ {FairBias-BM fixed-2D, FairBias-Joint fixed-2D}, directly paired with primary stress-elbow versions.
5. **Excluded Cells (2)**: LFR on Arm 002 for LR and GBDT explicitly recorded with status `NOT_SUPPORTED`.

---

## 5. Remediation of Finding R05: Input Hash Inconsistencies in Worker Report

### 5.1 Concession & Analysis
The previous `WORKER_REPORT.md` reported incorrect SHA256 hashes for two key supervisory inputs:
- `docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914.md` (reported `2b143b81...`, actual `5ee400543c11ee5a7c559dc2e17ed82c8e5fa6c19cd6eca36b45843e4f96c8a2`).
- `docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914_evidence/verification.json` (reported `6955ee38...`, actual `9a55c87665a19746c4917f39dadc3b4e2a38d4146cd2274aee3241978256e89d`).

Independent verification by the supervisor confirmed that the physical files were completely unmodified and matched the historical manifest. The error resulted from manual/intermediate copy-paste in the worker report text.

### 5.2 Action Taken
- The factual SHA256 hashes are verified in `FACTS.json`:
  - `docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914.md`: `5ee400543c11ee5a7c559dc2e17ed82c8e5fa6c19cd6eca36b45843e4f96c8a2`.
  - `docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914_evidence/verification.json`: `9a55c87665a19746c4917f39dadc3b4e2a38d4146cd2274aee3241978256e89d`.
- All output tables and reports in RESULTS-R0.1 are generated directly from `FACTS.json`, preventing transcription discrepancies.

---

## 6. Remediation of Finding R06: Actionable Executable Specification

### 6.1 Concession & Analysis
The previous specification provided high-level descriptions but omitted concrete Python interpreters, full shell commands, guard execution architectures, and exact evidence directory specifications.

### 6.2 Action Taken
In `R1_EXECUTION_SPEC.md`:
- Specific interpreter selected: `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3` (verified with `numpy 2.2.6`, `scipy 1.16.1`, `scikit-learn 1.7.1`, `pandas 2.3.1`, `pytest 8.4.1`).
- Exact commands, argument lists, and guard installation procedures prior to pytest/project import are fully specified.
- Guard failure sentinel probes (testing simulated unauthorized access to prohibited directories) are defined.

---

## 7. Remediation of Finding R07: Discrepancies Across Tables, Manifest, and Reports

### 7.1 Concession & Analysis
1. **NaN Token Count**: Previous manifest counted 15 tokens by summing `: NaN` and `: NaN,`, double-counting trailing commas. Standard JSON `parse_constant` callback verifies exactly **8 non-standard NaN tokens**.
2. **AST Line Numbers**: Previous Markdown table had slight manual offsets on end lines. AST parser `lineno` and `end_lineno` must be rendered verbatim.
3. **Line and Byte Counts**: Synchronized to exact disk measurements: previous R0 outputs consist of **1,708 lines and 110,296 bytes**.

### 7.2 Action Taken
`generate_static_repair.py` parses AST and JSON using standard library callbacks, recording exact values into `FACTS.json` and rendering them into the final report.

---

## 8. Remediation of Finding R08: Fact vs. Inference Precision

### 8.1 Concession & Analysis
1. **D6 Population Equality ($27,451 = 21,871 + 5,580$)**: Proves that D6 transformation rules were trained on the entire 2022 cohort (including the partition now allocated to Set C). However, without microdata inspection, it should not be stated as proof of specific record-level intersection.
2. **LFR Code Citation**: Previous text presented an idealized pseudocode block rather than quoting the actual source implementation.
3. **Test Suite Analysis**: Test passes should be attributed to specific observed coverage gaps (synthetic array fallbacks, disk release loading, shallow shape checks) rather than speculative causal statements.

### 8.2 Action Taken
- Narrative precision updated in all documentation.
- The actual code from `src/nhis_fairbias/benchmark/adapters/adapter_lfr.py` (lines 108–112) is cited verbatim:
  ```python
  if len(X_F) > 2000:
      rng = np.random.default_rng(self.random_state)
      sub_idx = rng.choice(len(df_F), size=2000, replace=False)
      df_sub = df_F.iloc[sub_idx].reset_index(drop=True)
      ...
      self.lfr.fit(dataset_sub, maxiter=50, maxfun=100)
  ```
