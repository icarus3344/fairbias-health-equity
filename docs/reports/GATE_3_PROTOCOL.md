# Gate 3: Pre-Data Scientific Protocol & Study Specification Report

> **Superseded method-arm note (2026-08-30):** Decision 0006 invalidates this historical report's `paper-informed reconstruction` and `survey-weighted mitigation extension` arm names. The implemented method is an exploratory survey-weighted group-aware centering heuristic requiring protected-group information at prediction time; it is not FairBias or Tang et al. This historical Gate 3 report is not current authorization to run an experiment.

## 1. Executive Summary & Audit Verdict

Gate 3 establishes the formal pre-data scientific research protocol, statistical analysis plan, official MEPS source registry, architectural decision record for two-panel temporal validation, machine-readable study configuration, and governance milestone reports for the MEPS longitudinal fairness investigation on branch `research/meps-hc252-longitudinal`.

- **Codex Supervisor Verdict**: `ACCEPTED_AFTER_INDEPENDENT_AUDIT`
- **Independent Peer Review Status**: Two independent Gemini read-only reviews returned `NO_BLOCKING_FINDINGS` and one P2 report-format issue (remediated).
- **Active Branch**: `research/meps-hc252-longitudinal`
- **Inherited File Diff**: 0 lines modified across all 14 inherited root files and `.gitignore`.
- **Created Pre-Data Artifacts (7 Total)**:
  1. `docs/research/RESEARCH_PROTOCOL.md`
  2. `docs/research/STATISTICAL_ANALYSIS_PLAN.md`
  3. `docs/data/MEPS_SOURCE_REGISTRY.md`
  4. `docs/decisions/0002-two-panel-temporal-validation.md`
  5. `configs/study.json`
  6. `docs/reports/GATE_2_GOVERNANCE.md`
  7. `docs/reports/GATE_3_PROTOCOL.md`

---

## 2. Official Supervisor-Verified Evidence Cited (Access Date: 2026-08-28)

1. **AHRQ MEPS HC-252 Panel 27 Longitudinal Data Public Use File** (2022–2023; September 2025 release; 8,292 person records; 2,648 variables; 7,812 with `ALL5RDS=1`; `YEARIND`, `LONGWT`, `VARSTR`, `VARPSU`):
   - Details: `https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-252`
   - Documentation: `https://meps.ahrq.gov/data_stats/download_data/pufs/h252/h252doc.pdf`
   - Codebook: `https://meps.ahrq.gov/data_stats/download_data/pufs/h252/h252cb.pdf`
2. **AHRQ MEPS HC-244 Panel 26 Longitudinal Data Public Use File** (2021–2022; September 2024 release; 6,741 person records; 2,737 variables; 6,295 with `ALL5RDS=1`; `YEARIND`, `LONGWT`, `VARSTR`, `VARPSU`):
   - Details: `https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-244`
   - Documentation: `https://meps.ahrq.gov/data_stats/download_data/pufs/h244/h244doc.pdf`
   - Codebook: `https://meps.ahrq.gov/data_stats/download_data/pufs/h244/h244cb.pdf`
3. **General Longitudinal Survey Guidance**:
   - `https://meps.ahrq.gov/mepsweb/data_stats/more_info_download_data_files.jsp`
4. **AHRQ MEPS Data Use Agreement**:
   - `https://meps.ahrq.gov/data_stats/data_use.jsp` (Statistical reporting and analysis only; absolute prohibition against re-identification; no attempts to link MEPS data with other individually identifiable records; HC-252 documentation restricts MEPS-NHIS linkage to the AHRQ Data Center, NCHS Research Data Center, or U.S. Census Research Data Center network; formal citation of AHRQ/MEPS required).
5. **Methodological Paper**:
   - Tang, Z., Lu, T., & Li, T. (2024). *Metric-Independent Mitigation of Unpredefined Bias in Machine Classification*. Intelligent Computing, 3, Article 0083. DOI: `https://doi.org/10.34133/icomputing.0083`. Erratum: Intelligent Computing, Article 0125. DOI: `https://doi.org/10.34133/icomputing.0125` (Funding omission; does not invalidate methodology).

---

## 3. Summary of Frozen vs. Deferred Decisions

### 3.1 Decisions Frozen in Gate 3
- **Study Purpose & Prohibitions**: Beneficial retention/outreach research only; strict prohibition against underwriting, pricing, coverage denial, eligibility determination, benefit reduction, or punitive actions.
- **Target Population**: Working-age adults aged 18–64 at baseline year end, both-years eligibility rule (`YEARIND == 1`), complete five rounds (`ALL5RDS == 1`), positive weight (`LONGWT > 0`), continuously insured in Year 1.
- **Two-Panel Architecture & Development Flow**:
  - Development cohort: HC-244 Panel 26 partitioned by `DUID` into 60% Train, 20% Validation, 20% Calibration.
  - Preprocessing fit on train only; validation used solely to select model hyperparameters, feature policies, and mitigation settings.
  - Selected pipeline refit on combined Train + Validation (80%).
  - Survey-weighted Platt logistic calibrator fit on untouched 20% calibration partition.
  - Fixed 10% weighted population selection probability threshold frozen from calibrated calibration set and applied unchanged to Panel 27.
- **Holdout Lock Policy (HC-252 Panel 27)**:
  - Allowed pre-lock checks: schema column names/types, documented-code compatibility, file dimensions/counts/hashes, and non-outcome structural integrity checks.
  - Prohibited: inspecting outcome prevalence, protected-group rates/distributions, feature distributions, or computing/tuning performance metrics.
  - Temporal holdout provides stronger temporal transport assessment within MEPS; not external validation or deployment simulation.
- **Primary Comparison & Endpoints**:
  - Primary comparison: Survey-weighted mitigation extension versus unmitigated survey-weighted logistic regression (identical predictor policy and split).
  - Primary utility endpoint: Survey-weighted AUPRC on Panel 27.
  - Primary fairness endpoint: within each primary audit dimension, maximum absolute pairwise group TPR gap at 10% weighted capacity, then maximum across the race/ethnicity and sex dimensions (category mappings deferred to Gate 7).
  - Inference: Paired differences with design-aware 95% confidence intervals. No single scalar proof of fairness; no noninferiority margin invented.
- **Survey Design & Uncertainty**: `LONGWT` for weighted domain estimates representing the frozen eligible cohort (not all US adults); stratified PSU bootstrap resampling using `VARSTR` and `VARPSU` with stop-and-escalate policy if unsupported.
- **Capacity Metric Basis**: Fixed capacity fractions at 5%, 10%, 20% defined by cumulative `LONGWT` and risk rank; unweighted top-K as sensitivity.
- **Predictor Timing & Missing Values**: Information strictly restricted to baseline Year 1; per-variable codebook decoding; no global negative-value imputation; train-only imputer/scaler fit.
- **Model Arms**: Primary survey-weighted logistic regression; secondary random forest and histogram gradient boosting (subject to native sample-weight and seed support).
- **Mitigation Arms**: Unmitigated baseline, paper-informed reconstruction, survey-weighted extension.
- **Fairness & Suppression**: Group-specific calibration and error-rate gaps (TPR, FPR, PPV, selection rate), bias concentration quantity. Suppression threshold: unweighted $n < 100$, positive $< 20$, negative $< 20$, or Kish $n_{\text{eff}} < 50$ (marked unavailable, no ad-hoc merging).
- **Seeds & Resource Limits**: Frozen seeds `20260828`–`20260832`; no $O(N^2)$ sample-pair matrices; local peak RSS $< 16\text{ GB}$; smoke run $< 30\text{ minutes}$.
- **Stop Conditions**: Unharmonizable targets, invalid design variables, positive cases $< 200$, data leakage failure (nonzero MI between legitimate baseline predictors and Y2 outcome is expected signal; leakage includes Y2 columns in predictors, cross-partition overlap, preprocessing beyond authorized partitions, or Panel 27 tuning), unsupported survey resampling, manifest irreproducibility.

### 3.2 Decisions Deferred to Gate 7 (Codebook Verification)
- Exact monthly health insurance status variable names and numeric code categories.
- Exact baseline year-end age variable name and decoding.
- Exact sociodemographic predictor feature list and cross-panel variable mapping.
- Protected dimension category mappings and codes (race/ethnicity, poverty category, age band, disability/limitation status).

---

## 4. Section 9 Standard Worker Report

Gate: Gate 3 — Pre-Data Scientific Protocol & Study Specification (Repair)
Status: Repaired; supervisor audit verdict ACCEPTED_AFTER_INDEPENDENT_AUDIT; two independent Gemini read-only reviews returned NO_BLOCKING_FINDINGS and one P2 report-format issue (remediated).
Files changed:
- docs/reports/GATE_3_PROTOCOL.md
Commands executed:
- python3 -m json.tool configs/study.json >/dev/null
- git diff --check
- git status --short --untracked-files=all
Permissions requested: None (no network access, no package installations, no staging or committing without Codex supervisor instruction).
Tests executed:
- python3 -m json.tool configs/study.json >/dev/null
- git diff --check
- git status --short --untracked-files=all
Exact test results:
- python3 -m json.tool configs/study.json >/dev/null: exit code 0 (valid JSON)
- git diff --check: exit code 0 (no whitespace errors or conflict markers)
- git status --short --untracked-files=all: exit code 0 (0 baseline files modified; only Gate 3 additive files present)
Input hashes:
- Inherited baseline tag: inherited-code-v0.3-baseline-20260828 (commit 038897e9f751edac6e36445b7706eec5fdb15988, tree 9e43047f69326a844cec1e7acdb6726af555dff3)
- No microdata hashes (none downloaded or processed)
Output hashes: None (no microdata or generated data artifacts; pre-data specification files only)
Row counts: Not applicable: no data downloaded or inspected.
Assumptions:
- Study scope, definitions, and boundaries are frozen pre-data prior to any microdata ingestion.
- Primary fairness endpoint is defined exactly as: within each primary audit dimension, maximum absolute pairwise group TPR gap at 10% weighted capacity, then maximum across the race/ethnicity and sex dimensions.
- Protected dimension category mappings and monthly insurance variable names remain deferred to Gate 7 codebook verification.
- Two-panel temporal validation assesses temporal transport within MEPS rather than external deployment simulation.
Unresolved issues: None.
Git diff summary: docs/reports/GATE_3_PROTOCOL.md updated to record supervisor verdict ACCEPTED_AFTER_INDEPENDENT_AUDIT, independent reviews (NO_BLOCKING_FINDINGS, one P2 report-format issue), exact primary fairness endpoint definition, and Section 9 report headings. Zero baseline files modified.
Proposed next step: Await Codex supervisor commit authorization of Gate 3 additive artifacts and progression to next scheduled gate.
STOP — waiting for Codex review.
