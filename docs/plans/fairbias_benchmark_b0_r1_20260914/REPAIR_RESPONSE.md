# Comprehensive Response to Supervisor Review: Remediations R01–R12

**Document ID**: `docs/plans/fairbias_benchmark_b0_r1_20260914/REPAIR_RESPONSE.md`  
**Execution Timestamp**: `2026-09-14T15:55:00+08:00`  
**Active Gate**: `FAIRBIAS-BENCHMARK-B0-R1`  
**Authority Reference**: [Supervisor Review of B0](file:///Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_B0_SUPERVISOR_REVIEW_20260914.md) | [Supervisor Review Verification JSON](file:///Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_B0_SUPERVISOR_REVIEW_20260914_evidence/verification.json) | [B0-R1 Repair Prompt](file:///Users/lkc/Downloads/code_v_0_3/docs/plans/GEMINI_FAIRBIAS_B0_R1_REPAIR_PROMPT_20260914.md)  
**Supervisor**: Codex Supervisor  
**Implementation Worker**: Gemini Implementation Worker  

---

## 1. Overview of B0-R1 Document Repair

Following the `REPAIR` determination in the independent Codex Supervisor review of Gate B0, this document provides an item-by-item accounting of all modifications made across the revised planning documents in `docs/plans/fairbias_benchmark_b0_r1_20260914/`.

**Operational Boundary Reaffirmation**:
- These revisions represent **purely additive planning and specification corrections**.
- No implementation code in `src/`, `configs/`, `scripts/`, or `tests/` was modified.
- No model training, benchmark runs, or test suites were executed.
- No real microdata was accessed.
- Document revisions do not declare code issues "fixed"; open code issues remain classified as `OPEN` or `UNVERIFIED` and are systematically mapped to subsequent execution gates.

---

## 2. Item-by-Item Response: R01 through R12

### R01 — FairBias API & Algorithm Identity
- **Identified Defect**: In the original B0 capability table, FairBias was marked as `Requires y at fit: No`. In reality, `src/fairbias/mitigation.py` explicitly uses $Y$ inside `_nmi_gate_ok` (lines 212–224) and `_make_candidate` (line 271). In addition, the draft asserted "fidelity is maintained", `No (verified mapping)`, and cited `Information Sciences` without verified volume/page/DOI evidence.
- **Specific Remediation**:
  - In `DATA_AND_METHOD_CONTRACTS.md` (Table 5 & Section 5.1): Explicitly updated FairBias capability to `Requires y at fit: Yes`. Distinguished the BM transformer fit (requires $X, y, A$ to evaluate NMI gate) from downstream classifier fit (requires $X_{\text{trans}}, y$) and inference (`predict` requires $X$ only, strictly prohibited from accessing $y$).
  - Preserved the NMI gate in documentation without modifying source code.
  - Added an itemized algorithmic mapping table for FairBias:
    1. Cardinality $H$: Fixed at $H=1$ mapped to `src/fairbias/mitigation.py:175`.
    2. MDS Dimensionality: Stress-elbow heuristic mapped to `src/fairbias/bias_metric.py:431`.
    3. Power Grid: Interleaved sequence `[1, 2, 0.5, 3, 0.33, 4, 0.25, 5, 0.2]` mapped to `src/fairbias/enhancement.py:DEFAULT_POLY_GRID`.
    4. Restart/Revisit: Mapped to `src/fairbias/mitigation.py:320`.
    5. Information Gate: NMI gate mapped to `src/fairbias/mitigation.py:212, 271`.
    6. Multi-group Aggregation: Max-pair divergence mapped to `src/fairbias/bias_metric.py:165`.
    7. Stopping & Budget: Mapped to `src/fairbias/mitigation.py:360`.
  - Formally designated algorithm as `FairBias-BM application-v1 (fidelity UNVERIFIED, pending Gate B3)`. Marked publication venue as `SOURCE_UNVERIFIED`. Removed contradictory assertions.
- **Static Evidence Verified**: Inspected `src/fairbias/mitigation.py:212-224` (`normalized_mutual_info_score(y, col_vals)`) and line 271 (`self._nmi_gate_ok(candidate_df, Y, nmi_org, attr)`).
- **Future Verification Content**: Gate B3 unit tests verifying transformer and pipeline fit behavior with and without $y$, and verifying literature citations against authoritative publisher text.

---

### R02 — Fix GBDT Class to `sklearn.ensemble.GradientBoostingClassifier`
- **Identified Defect**: `EXPERIMENT_REGISTRY_DRAFT.md` stated `HistGradientBoostingClassifier (or verified GradientBoostingClassifier)` while specifying parameters `n_estimators` and `subsample`. In scikit-learn 1.7.1, `HistGradientBoostingClassifier` accepts `max_iter` and rejects `n_estimators` and `subsample`, creating a constructor error.
- **Specific Remediation**:
  - In `EXPERIMENT_REGISTRY_DRAFT.md` (Section 2.1 & Section 6) and `GATE_SPECS_B1_B7.md`: Removed `HistGradientBoostingClassifier` and all alternative phrasing.
  - Fixed the GBDT backbone strictly to `sklearn.ensemble.GradientBoostingClassifier` with parameters:
    $$\text{n\_estimators} \in \{100, 200\} \times \text{max\_depth} \in \{2, 3\}, \quad \text{learning\_rate} = 0.05, \quad \text{subsample} = 1.0$$
  - Synchronized this class across all condition tables, budgets, and gate specifications.
- **Static Evidence Verified**: AST parameter inspection from supervisor `verification.json:332-391` confirming `GradientBoostingClassifier` in `sklearn/ensemble/_gb.py:1466` accepts `n_estimators` and `subsample`.
- **Future Verification Content**: Gate B3 adapter integration tests instantiating and training `GradientBoostingClassifier` on synthetic data.

---

### R03 — Unique Data Partitioning & Identity Policy
- **Identified Defect**: Partitioning sequence was inverted (filtering by arm $Y/A$ prior to master partitioning); PSU allocation was specified ambiguously as `Hash(...) % 100`; survey year formatting had conflicting rules for `2023.0` vs `2023`.
- **Specific Remediation**:
  - In `DATA_AND_METHOD_CONTRACTS.md` (Section 3 & Section 4) and `GATE_SPECS_B1_B7.md`: Fixed the execution sequence:
    1. Construct 2022 Master Survey Design Table from all records with valid design variables (`WTFA_A > 0`, `PSTRAT`, `PPSU`).
    2. Generate Master PSU Allocation Mapping on sorted unique `(int(PSTRAT), int(PPSU))` pairs using `np.random.Generator(np.random.PCG64(20260913))`. Allocate cluster to Calibration Set C if uniform variate $u < 0.20$; else Fitting Set F.
    3. Apply arm-specific eligibility masks ($Y \in \{1, 2\}$ and valid protected attribute $A$) within partitions.
  - Canonicalized survey year to finite integer: `2023` and `2023.0` are equivalent and normalize to integer `2023`. Non-integers, `NaN`, `inf`, and booleans are strictly rejected.
  - Defined record key as `(source, int(year), str(id_norm))` based on official `HHX` ID; explicitly prohibited RangeIndex as record identity.
- **Static Evidence Verified**: NumPy PCG64 random generator specification; C03 unit test logic in `tests/test_audit_remediation_probes.py`.
- **Future Verification Content**: Gate B1/B2 unit tests evaluating partition stability across processes, data row order permutations, and cross-arm parity.

---

### R04 — Categorical Encoding, UNKNOWN Tokens & Static Feature Extraction
- **Identified Defect**: Relying solely on `OneHotEncoder(handle_unknown='ignore')` drops unobserved categories to all-zeros vectors rather than an explicit unknown indicator column. Predictor table lacked full field-level specifications. `PCNT18UPTC` was mislabeled as family count. `URBRRL` was incorrectly called a survey design variable.
- **Specific Remediation**:
  - In `DATA_AND_METHOD_CONTRACTS.md` (Section 2.2 & 2.3):
    - Statically extracted and tabulated all 21 primary core features directly from `configs/nhis/features.json`, specifying Official Name, Harmonized Name, Domain, Semantic Type, Substantive Codes, Missing Codes, Year Availability, Study Role, Imputation on F, and Output Representation.
    - Calibrated `PCNT18UPTC` and `PCNTLT18TC` descriptions to "household" (matching CDC codebook).
    - Clarified that `URBRRL` is excluded by feature selection policy; it is an administrative covariate, not a complex survey design variable.
    - Specified encoding contract: An explicit `'UNKNOWN'` category token is reserved in the vocabulary fitted on Partition F (alongside `'MISSING'`). Unseen category levels in S and T are mapped to `'UNKNOWN'`, ensuring an active indicator column.
    - Documented edge-case policies: F-all-missing continuous features trigger `FeatureImputationError`; out-of-range values extrapolate linearly without unannounced clipping; consolidated levels produce updated schemas.
- **Static Evidence Verified**: Direct parsing of `configs/nhis/features.json` metadata for all 21 primary core features, outcomes, and protected attributes.
- **Future Verification Content**: Gate B2 unit tests verifying one-hot column activation when synthetic unobserved categories are introduced in test data.

---

### R05 — Estimability, Probability Semantics, and Evaluation Boundaries
- **Identified Defect**: Metric formulas evaluated over observed groups without enforcing full `expected_groups` support. Zero-denominators were not cleanly separated from validation failures. Model output $p$ was called "true calibrated". An invalid `logit(y) ~ logit(p)` regression was added.
- **Specific Remediation**:
  - In `EXPERIMENT_REGISTRY_DRAFT.md` (Section 3.2 & 3.3) and `GATE_SPECS_B1_B7.md`:
    - Explicitly defined `expected_groups` for each arm (Arm 001: 2 groups; Arm 002: 7 groups; Arms 003/004: 2 groups).
    - Established that primary gap metrics (`EO_gap`, `DP_gap`, `EOpp_gap`) require valid denominators across ALL expected groups; if any group lacks support, the metric returns `status = 'NOT_ESTIMABLE'` (or null with reason).
    - Gaps computed on observed subsets are labeled `observed_EO_gap` and restricted to descriptive reporting.
    - Defined $p_i = \hat{P}(Y=1 \mid X_i)$ as the model-predicted probability output, removing claims of "true/calibrated".
    - Removed the spurious `logit(y) ~ logit(p)` regression. Retained decile calibration curves on Set T.
    - Updated Gate B4 specifications to include negative test fixtures that intentionally lack group support to verify structured null returns.
    - Updated Gate B5a to audit support without failing if empirical data lacks minority support in certain cells.
- **Static Evidence Verified**: Mathematical formulation in Master Plan Section 6.1 and 6.2.
- **Future Verification Content**: Gate B2 and Gate B4 synthetic unit tests evaluating intentional single-group and zero-cell fixtures.

---

### R06 — 20 Primary Contrasts & Survey Variance Estimation
- **Identified Defect**: Listed 6 comparators (yielding 24 contrasts) while dividing by 20. Omitted concrete bootstrap sample variance formulas, degrees of freedom, and coverage verification criteria.
- **Specific Remediation**:
  - In `EXPERIMENT_REGISTRY_DRAFT.md` (Section 5.3):
    - Explicitly enumerated the **20 unique primary contrast IDs**:
      - Primary Backbone: Logistic Regression (LR).
      - Primary Operating Point: $\text{EO\_gap} \le 0.10$.
      - Demographic Arms: Arm 001 (`SEX_A`) and Arm 003 (`DISAB3_A` full).
      - Comparators: Exactly 5 external methods (Reweighing, LFR, EG-DP, EG-EO, TO-EO).
      - Metrics: $\Delta\text{BA}$ and $\Delta\text{EO}$.
      - Total: $5 \text{ methods} \times 2 \text{ metrics} \times 2 \text{ arms} = \mathbf{20}$ unique contrast IDs.
    - Unmitigated is formally separated as a standalone predictive baseline reference outside the 20-contrast family.
    - Enforced fixed Bonferroni denominator ($M=20$) regardless of external method completion status.
    - Specified rescaled PSU bootstrap ($B=2000$, seed `20260914`), sample variance formula with $B_{\text{eff}}-1$ denominator, $df = \sum_h m_h - H$, and Bonferroni adjusted $99.75\%$ intervals ($\alpha = 0.05/20 = 0.0025$).
    - Noted empirical coverage verification in Gate B2; if non-smooth gap coverage falls short of nominal, intervals are downgraded to descriptive.
- **Static Evidence Verified**: Master Plan Section 6.3 and Section 7.
- **Future Verification Content**: Gate B2 synthetic complex survey coverage experiments.

---

### R07 — Complete 76-Condition Registry, Set C Thresholding, and AE Inner Loop
- **Identified Defect**: The original B0 registry labeled 70 conditions as "total", omitting survey-weighted training sensitivity and geometric path sensitivity. Set C threshold calibration and AE inner loop utility rules were not fully specified.
- **Specific Remediation**:
  - In `EXPERIMENT_REGISTRY_DRAFT.md` (Section 2.1, 4.1, 4.2, 6.1):
    - Clarified that 70 is the Core (54) + AE (16) subtotal.
    - Formally registered the remaining 6 model fitting conditions:
      - 4 Survey-Weighted Training conditions: Unmitigated and FairBias-BM on Arms 003 and 004, LR backbone.
      - 2 Geometric Path Sensitivity conditions: Arm 004, LR backbone, FairBias-BM and Joint with fixed 2D MDS, paired with primary stress-elbow versions.
      - Grand Total = **76 unique model fitting conditions**.
    - Registered evaluation views (0 additional model fits): $t=0.5$ threshold sensitivity, unweighted metrics.
    - Fully specified Set C threshold calibration: maximize unweighted BA on C across midpoints + bounds, tie-breaking by min $|t-0.5|$, then max $t$.
    - Fully specified AE inner loop: C thresholding evaluating BA on C, strict $\text{slack}=0.0$ ($d_\phi \le \epsilon_{\text{ref}}$), candidate family and stopping criteria.
- **Static Evidence Verified**: Master Plan Section 5.1, 8, and D8 runner architecture.
- **Future Verification Content**: Gate B3/B4 implementation and synthetic execution of all 76 conditions.

---

### R08 — Seed Completeness & Selection Protocol
- **Identified Defect**: Averaged metrics across seeds without defining a policy for failed seeds (introducing survivor bias risk). Missing non-refitting invariant.
- **Specific Remediation**:
  - In `EXPERIMENT_REGISTRY_DRAFT.md` (Section 4.4):
    - Formally defined configuration completeness states: `COMPLETE` (all 5 seeds succeeded), `PARTIAL_FAILURE` ($\ge 1$ seed failed), `TOTAL_FAILURE`.
    - Enforced survivor bias elimination: any configuration with status `PARTIAL_FAILURE` is disqualified from participating in winning configuration selection. Discarding failed seeds and averaging surviving seeds is strictly prohibited.
    - Documented infrastructure recovery: failed attempts resume with identical `(config_id, seed)` and an incremented `attempt_id`.
    - Enforced non-refitting invariant: models selected on Set S are **NEVER re-fitted on F+C+S**.
    - Formulated tie-breaking hierarchy for unmitigated predictive reference on Set S.
- **Static Evidence Verified**: Master Plan Section 5.2.
- **Future Verification Content**: Gate B2 and Gate B4 synthetic selection engine testing.

---

### R09 — Real Computational Budget Limits
- **Identified Defect**: Conflated Exponentiated Gradient iterations with base estimator fits. Omitted LFR optimization parameters `maxiter` and `maxfun`. Table contained conflicting 5/10/30 minute limits.
- **Specific Remediation**:
  - In `EXPERIMENT_REGISTRY_DRAFT.md` (Section 2.2 & 6.2):
    - Disentangled iterations from oracle fits: EG $\text{max\_iter}=50$ represents reduction iterations, which upper-bound the number of base estimator oracle calls during LP solving.
    - Clarified LFR parameter $k \in \{5, 10\}$ represents prototype latent vectors, and recorded optimization limits $\text{maxiter}=5000, \text{maxfun}=5000$.
    - Harmonized all conditions under the Master Plan authoritative boundary: **30 minutes wall-clock timeout** and **4 GiB peak resident memory** per condition fit.
    - Specified downscaled synthetic cohorts for Gate B4 rehearsal plus isolated stress cases.
- **Static Evidence Verified**: Master Plan Section 5.2 and Section 9.
- **Future Verification Content**: Gate B4 synthetic rehearsal memory/time profiling.

---

### R10 — Executable Gate Specifications (B1 through B7)
- **Identified Defect**: Missing Python interpreter preconditions; incomplete file modification whitelists; fixed output directories risking overwrites; non-localized failure rollbacks; inaccurate "cryptographic digital signature" wording.
- **Specific Remediation**:
  - In `GATE_SPECS_B1_B7.md`:
    - Specified Framework Python (`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`) for Gates B1 and B2; dedicated benchmark virtualenv for Gate B3 onwards.
    - Expanded Gate B1 file whitelist to include all necessary test files (`tests/test_nhis_survey.py`, `tests/test_nhis_d6_temporal.py`, etc.).
    - Strictly allocated C01–C06 and E01 to Gate B1; moved E02/E03/E04/E06 to appropriate subsequent gates.
    - Enforced unique timestamped run directories (`runs/benchmark_v1_<timestamp>_<run_id>/`) with fail-closed checks if target directory exists.
    - Confined failure rollback strictly to active gate changes, protecting baseline and pre-existing candidate files.
    - Replaced "digital signature" terminology with verified SHA-256 checksums and Supervisor Accept manifests.
- **Static Evidence Verified**: Repository file structure and Git status inspection.
- **Future Verification Content**: Standalone execution of each gate in sequence.

---

### R11 — Corrected Closed Evidence Anchors & C05 Formula
- **Identified Defect**: C05 formula was written as $1/(K-1)$ instead of $1/K$. Closed table cited line numbers that did not match test function headers in `tests/test_audit_remediation_probes.py`.
- **Specific Remediation**:
  - In `ISSUE_CLOSURE_MATRIX.md` (C05 row & Section 3):
    - Corrected C05 divergence formula to:
      $$\text{cat\_diff}[col] = \frac{1}{k} \sum_{k=1}^K \left| \frac{p_k}{\sum p} - \frac{n_k}{\sum n} \right|$$
      where $k = \max(1, \text{len}(\text{union\_index}))$, matching `src/fairbias/bias_metric.py:275` exactly. Clarified that the defect is unobserved category levels inflating $k$.
    - Corrected all test method anchor line references in `tests/test_audit_remediation_probes.py`:
      - `test_runner_active_parameter_getter_callable_without_data`: lines 525–534.
      - `test_active_parameters_match_smoke_budgets`: lines 535–542.
      - `test_independent_namespaces_identical_values_allowed`: lines 485–491.
      - `test_duplicate_ids_within_partition_rejected`: lines 499–513.
      - `test_audit_partition_fingerprints_bind_source`: lines 514–524.
      - `test_p1_02_ae_rejects_missing_group_geometry`: lines 123–131.
      - `test_p1_05_single_group_dp_not_estimable`: lines 132–137.
      - `test_p1_05_undefined_tpr_returns_null`: lines 138–143.
      - `test_p1_03_infinite_weights_not_inference_eligible`: lines 144–153.
      - `test_p2_02_valid_zero_event_is_zero`: lines 154–158.
      - `test_probe_tiebreak_retains_first_grid_occurrence`: lines 409–442.
      - `test_probe_candidate_audit_preserves_boolean_json_types`: lines 443–462.
      - `test_historical_state_barrier_all_arms`: lines 543–554.
    - Strictly qualified closed items to their verified probe scopes without claiming overall configuration caching is closed.
- **Static Evidence Verified**: Direct AST inspection of `tests/test_audit_remediation_probes.py` and `src/fairbias/bias_metric.py:275`.
- **Future Verification Content**: Gate B1 execution of updated unit test suite.

---

### R12 — Scope, Report Accuracy, and Verification Manifest
- **Identified Defect**: Overly broad environment claims; git tracking status unverified via fetch; ellipsis in command logs; self-referential report hash.
- **Specific Remediation**:
  - In `B0_PREFLIGHT.md` and `B0_WORKER_REPORT.md`:
    - Restricted environment scope strictly to the 4 inspected interpreters.
    - Clarified tracking branch is based on local tracking ref without network fetch.
    - Recorded full, exact execution commands without ellipsis.
    - Computed report hash externally in manifest and final summary.
    - Created `verification_manifest.json` containing complete machine-readable verification data.
- **Static Evidence Verified**: Supervisor `verification.json` and local command execution history.
- **Future Verification Content**: Independent audit by Codex Supervisor.
