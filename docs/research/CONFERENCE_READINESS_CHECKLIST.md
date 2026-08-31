# Conference Readiness & Reproducibility Checklist

This checklist documents the audit status and scientific integrity bounds of the MEPS Longitudinal Fairness research development package. Target venues (MLHC, CHIL, FAccT) and submission deadlines remain TBD pending resolution of statistical power and supervisory authorization.

---

## 1. Ethical & Legal Compliance

- [x] **AHRQ Data Use Agreement (DUA)**: Research uses public-use files (PUF) exclusively for aggregate statistical reporting; zero individual re-identification or linkage attempts.
- [x] **Microdata Privacy Enforcement**: Zero individual-level microdata rows or unsuppressed sensitive cells in repository logs, code comments, or manuscript drafts.
- [x] **Cell Suppression Rule**: 100% of subgroup cells on the evaluation partition were suppressed under $n < 100 \lor \text{pos} < 20 \lor \text{neg} < 20 \lor n_{\text{eff}} < 50$. All suppressed subgroup cells yield fairness Not Estimable (`None`) without fabricating a numeric zero or fairness bootstrap inference. Zero unsuppressed disparity claims made.
- [x] **Explicit Beneficial Purpose**: Intended application is bounded strictly to supportive retention outreach (navigators, reminder assistance). Punitive, underwriting, or coverage exclusion uses are explicitly forbidden.


---

## 2. Methodological Rigor & Survey Design

- [x] **Complex Survey Analysis Weights**: Primary utility (Weighted AUPRC, Weighted AUROC) and fairness endpoints utilize `LONGWT`.
- [x] **Survey Design Variance Estimation**: Statistical inference employs design-aware stratified-PSU bootstrap resampling within `VARSTR` strata across `VARPSU` clusters.
- [x] **Zero Intra-Household Leakage**: Split partitions enforce household-level grouping strictly on `DUID`.
- [x] **Train-Only Preprocessing**: Missing value imputations and standardizations are fit strictly on the Training partition without test-set distribution leakage.
- [x] **Apparent Fit Labeling**: Metrics evaluated on the calibration partition are strictly designated as apparent calibration-fit diagnostics, not out-of-sample holdout performance.
- [x] **Exploratory Mitigation Labeling**: The mitigation method is explicitly documented as an exploratory group-aware centering heuristic requiring protected attributes at inference time, not a faithful Tang et al. (2024) reconstruction.

---

## 3. Computational Scaling & Code Architecture

- [x] **Linear Computational Complexity**: Disparity quantification operates in $O(N \cdot D)$ time using column attribute summaries. Full $O(N^2)$ sample-by-sample pairwise distance matrices are completely eliminated.
- [x] **Resource Consumption**: Pipeline executes in $< 10$ seconds with peak RSS $< 0.35$ GB (far below the 30-minute and 16-GB protocol limits).
- [x] **Deterministic Reproducibility**: Execution is parameterized across 5 frozen random seeds (`20260828`..`20260832`), with cryptographic SHA-256 hashes generated for split assignments, model configs, and run manifests.
- [x] **No-Clobber Artifacts**: Immutable manifests stored in unique timestamped run directories; root pointers explicitly labeled as mutable convenience pointers.
- [x] **Comprehensive Test Suite**: Unit and integration test discovery passing cleanly across all test modules.

---

## 4. Scientific Honesty & Stop Condition Integrity

- [x] **Power Reporting**: Accurately reported that Panel 26 cohort contains 136 positive events ($< 200$), documenting the underpowering finding honestly.
- [x] **Fail-Closed Stop Enforcement**: Maintained 100% lock on Panel 27 (HC-252) under `LOCKED_UNDERPOWERED_STOP_CONDITION` without accessing test outcomes.
- [x] **Completion Status**: Acknowledged that the study stopped before Panel 27 evaluation, no conference paper result package is complete, and venue/deadlines remain TBD.
- [x] **Claims Boundary**: Manuscript and claims matrix strictly enforce permitted vs. prohibited claims boundaries with zero fabricated numbers.

