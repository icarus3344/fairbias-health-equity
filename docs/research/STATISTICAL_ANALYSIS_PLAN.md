# Statistical Analysis Plan (SAP)

**Study Title**: Longitudinal Prediction and Fair Allocation of Health Insurance Retention Outreach in MEPS
**Date**: 2026-08-28
**SAP Version**: 1.1.0-pre-experiment-repair
**Status**: Partially frozen — Gate 7 mappings verified; no new experiment authorized

---

## 1. Study Design & Cohort Flow

### 1.1 Source Datasets
The study keeps Panel 27 as the sole locked temporal holdout. The fixed candidate development sequence is the four consecutive two-year longitudinal PUFs immediately preceding it:
- **Candidate Development Panels (metadata frozen; additional outcome access not authorized here)**: HC-217 Panel 23 (2018–2019), HC-225 Panel 24 (2019–2020), HC-234 Panel 25 (2020–2021), and HC-244 Panel 26 (2021–2022).
- **Currently observed development panel**: HC-244 Panel 26; its prior development audit found 136 eligible positive outcomes, below the frozen threshold of 200.
- **Temporal Holdout Panel**: MEPS HC-252 Panel 27 Longitudinal Data Public Use File (covering 2022 [Year 1] to 2023 [Year 2]; released September 2025; 8,292 total person records; 7,812 with `ALL5RDS = 1`).

The four candidate development PUFs must not be naively concatenated. Common-variable harmonization, across-panel survey-weight normalization, overlapping calendar years, and AHRQ's documented pandemic-era comparability concerns for Panel 25 require a separate statistical-design gate before any pooled analysis.

### 1.2 Inclusion and Exclusion Criteria

```mermaid
flowchart TD
    A["MEPS Longitudinal PUF Records (HC-244 / HC-252)"] --> B{"YEARIND == 1?"}
    B -- No --> Excl1["Exclude: Not In-Scope Both Years"]
    B -- Yes --> C{"ALL5RDS == 1?"}
    C -- No --> Excl2["Exclude: Incomplete 5 Rounds"]
    C -- Yes --> D{"Age 18–64 at Y1 End?"}
    D -- No --> Excl3["Exclude: Age <18 or >=65"]
    D -- Yes --> E{"LONGWT > 0?"}
    E -- No --> Excl4["Exclude: Zero Survey Weight"]
    E -- Yes --> F{"Continuously Insured in All 12 Months of Y1?"}
    F -- No --> Excl5["Exclude: Baseline Uninsured or Interrupted"]
    F -- Yes --> G["Eligible Analytic Longitudinal Cohort"]
```

1. **Both-Years Eligibility**: Must be eligible / in scope across both survey years (`YEARIND == 1`).
2. **Complete Survey Rounds**: Must have completed all five survey rounds over the two-year panel (`ALL5RDS == 1`).
3. **Target Age Range**: Must be aged 18 to 64 inclusive at the end of baseline Year 1 (Y1 age variable).
4. **Positive Longitudinal Weight**: Must have a positive longitudinal analysis weight (`LONGWT > 0`).
5. **Continuous Baseline Insurance**: Must have active health insurance coverage in all 12 calendar months of baseline Year 1. Individuals with any uninsurance month in Year 1 are excluded to isolate the transition from continuous coverage to coverage disruption.
6. **Exact Variable Mapping Status**: Gate 7 verified `AGEY1X`, the 12 `INS...Y1X` baseline months, the 12 `INS...Y2X` follow-up months, 74 baseline predictors, survey-design variables, and audit-variable derivations. The canonical machine-readable mapping is `configs/cohort_and_variables.json`; `configs/study.json` must remain mechanically consistent with it.

---

## 2. Dataset Partitioning, Development Flow & Test Isolation Policy

### 2.1 Development Cohort Partitioning & Refit Flow (HC-244 Panel 26)
To prevent intra-household information leakage while preserving population balance, partitioning is performed strictly at the household/dwelling unit level:
- **Unit of Grouping**: Dwelling Unit ID (`DUID`). All individuals sharing a `DUID` are assigned to the same partition. Never split by row.
- **Initial Partition Ratios**:
  - **Training Set (60%)**: Used for feature transformation fitting, imputer fitting, predictive model training, and fairness mitigation parameter exploration.
  - **Validation Set (20%)**: Used solely to select model hyperparameters, feature selection policies, and fairness mitigation trade-off settings.
  - **Calibration Set (20%)**: Untouched during feature and model selection.
- **Refit Protocol**:
  - Once model hyperparameters, feature selection policies, and mitigation settings are frozen, the selected preprocessing pipeline, mitigation arm, and base model are **refit on the combined Train + Validation partition (80%)**.
- **Probability Calibration**:
  - A survey-weighted Platt logistic calibrator is fit on the untouched Calibration Set (20%). Platt scaling is prespecified as the primary calibration method (not left data-dependent).
  - Metrics computed on that same Calibration Set after fitting the calibrator are **apparent calibration-fit diagnostics**, not out-of-sample validation. Independent performance claims remain unavailable while Panel 27 is locked.
- **Threshold Freezing**:
  - One conventional probability threshold is frozen from the calibrated Panel 26 calibration set at 10% weighted population selection. This numeric threshold is applied unchanged to Panel 27, distinguishing fixed-threshold evaluation from rank-based top-K capacity evaluation within each panel.
- **Grouping Algorithm**: Grouped stratified split approximately balancing the target prevalence and survey weight deciles across folds. The grouping seed and assignment hashes are permanently recorded in the run manifest.

### 2.2 Locked Temporal Holdout Policy (HC-252 Panel 27)
- **Strict Isolation**: HC-252 Panel 27 is fully locked during pipeline development. No outcome prevalences, protected-group event rates, feature summaries/distributions, or evaluation metrics may be viewed or utilized to adjust any pipeline parameter.
- **Allowed Pre-Lock Checks**: Checking schema column names and types, documented-code compatibility, file dimensions and record counts, publisher/file hashes, and non-outcome structural integrity checks (e.g., verifying presence of `ALL5RDS`, `YEARIND`, `DUID`, `LONGWT`, `VARSTR`, `VARPSU`).
- **Prohibited Pre-Lock Checks**: Inspecting outcome prevalence, protected-group rates or distributions, predictor feature distributions, and calculating or tuning performance metrics.
- **Single Evaluation Run**: Formal evaluation on Panel 27 occurs exactly once after all models, transformers, hyperparameters, calibration maps, and decision thresholds are frozen.
- **Rerun Policy**: Re-evaluation is permitted only in the event of a documented, fatal code error, accompanied by an explicit incident report.
- **Claim Limitation**: Panel 27 represents a temporal holdout providing stronger temporal transport assessment within MEPS, **not** an independent external validation dataset or deployment simulation.

---

## 3. Predictor Timing & Missing Value Strategy

### 3.1 Strict Predictor Temporal Cutoff
- Predictor features must reflect information available on or before the conclusion of baseline Year 1 (Round 3 / end of Year 1).
- **Prohibited Predictors**:
  - Any Year 2 (Y2) variables (e.g., Y2 coverage, Y2 health status, Y2 healthcare utilization, Y2 employment transitions).
  - Post-baseline events or future-derived variables.
  - Record and person identifiers (`DUPERSID`, `DUID`, `PID`, `PANEL`, family IDs), which serve exclusively for grouping, survey design, and provenance.

### 3.2 Missing Value Handling
- **No Global Rules**: Global negative-value imputation (e.g., treating all negative numbers as -1 or NaN) is strictly prohibited.
- **Codebook-Specific Decoding**: Missing values must be decoded variable by variable from the official codebooks, distinguishing structural/inapplicable states from nonresponse or cannot-compute states only when the documented value labels support that distinction. No numeric missing-code list is assumed globally.
- **Imputation & Encoding Protocol**:
  - All categorical encoders, numerical scalers, missing-indicator flags, and imputation models must be fit strictly on authorized training data (Training partition during development; Train+Validation during refit) and applied out-of-sample to Validation, Calibration, and Temporal Holdout sets.

---

## 4. Model Architectures & Mitigation Arms

### 4.1 Predictive Model Arms
1. **Primary Predictive Baseline**: Survey-weighted Logistic Regression with elastic-net regularization.
2. **Secondary Machine Learning Models**:
   - Random Forest Classifier (with sample weights).
   - Histogram Gradient Boosting Classifier (with sample weights).
   - *Constraint*: Secondary models will only be executed if installed library versions natively support survey sample weights and deterministic reproducibility; unsupported features will not be promised.

### 4.2 Fairness Mitigation Arms
Only two implemented arms are currently named:
1. **Unmitigated Survey-Weighted Logistic Regression**: Group-agnostic at inference.
2. **Exploratory Survey-Weighted Group-Aware Centering Heuristic**: Learns survey-weighted group offsets and requires the protected group at prediction time. It is an engineering heuristic, not FairBias, not Tang et al. (2024), and not a paper reconstruction. Its results are exploratory and cannot support a deployable group-blind intervention claim.

---

## 5. Implemented Comparison Boundary, Future Endpoints & Capacity-Aware Metrics

### 5.1 Implemented comparison and prespecified future locked-holdout endpoints
- **Exploratory Comparison**: The implemented group-aware centering heuristic versus unmitigated survey-weighted logistic regression, using identical baseline predictors and split structure. This is not yet an accepted primary scientific comparison.
- **Prespecified Future Utility Endpoint**: Survey-weighted Area Under the Precision-Recall Curve (weighted AUPRC) on Panel 27, if a later supervisor-authorized holdout evaluation becomes permissible.
- **Prespecified Future Fairness Endpoint**: Within each primary audit dimension, take the maximum absolute pairwise group difference in TPR at 10% weighted capacity; then take the maximum across primary dimensions on Panel 27. Gate 7 verified the source-variable and category mappings, but the endpoint remains unavailable while Panel 27 is locked.
- **Primary Inference Standard**: Paired difference estimation with design-aware 95% confidence intervals. No single scalar will be defined as proof of fairness, and no arbitrary noninferiority margin will be invented. All other models, metrics, groups, and intersections are designated secondary or exploratory.

### 5.2 Discrimination & Probabilistic Accuracy Metrics
- **Weighted AUROC**: Area under the survey-weighted Receiver Operating Characteristic curve.
- **Weighted AUPRC**: Area under the survey-weighted Precision-Recall curve (primary utility endpoint).
- **Weighted Brier Score**: Mean squared error of predicted probabilities:
  $$\text{Brier} = \frac{\sum_{i} w_i (\hat{p}_i - y_i)^2}{\sum_{i} w_i}$$

### 5.3 Calibration Metrics
- **Calibration Intercept & Slope**: From survey-weighted logistic calibration regression: $\text{logit}(y) = a + b \cdot \text{logit}(\hat{p})$.
- **Expected Calibration Error (ECE)**: Weighted average difference between predicted risk and empirical event rate across risk deciles.

### 5.4 Capacity-Aware Operational Metrics
Reflecting practical resource constraints in public health retention outreach:
- **Prespecified Capacities**: Fixed capacity fractions at 5%, 10%, and 20%.
- **Capacity Definition**: Primary capacity is defined as the top $K\%$ of the represented population ranked by predicted risk and accumulated by longitudinal weight `LONGWT`. Unweighted top $K\%$ of sample is computed as secondary sensitivity only.
- **Recall at Capacity (5%, 10%, 20%)**: Proportion of all coverage-interrupted individuals captured within the top $K\%$ weighted population.
- **Precision (PPV) at Capacity (5%, 10%, 20%)**: Proportion of individuals in the top $K\%$ weighted population who actually experience coverage interruption.

### 5.5 Fixed-Threshold Metrics
- Conventional binary classification metrics (Sensitivity, Specificity, PPV, NPV, F1) evaluated at the single probability threshold frozen from the calibrated Panel 26 calibration set at 10% weighted selection and applied unchanged to Panel 27.

---

## 6. Fairness Reporting & Subgroup Suppression Policy

### 6.1 Protected Demographic Dimensions
- **Primary Audit Dimensions**:
  - Race / Ethnicity
  - Sex
- **Secondary Audit Dimensions**:
  - Baseline Family Poverty Category
  - Baseline Age Band
  - Baseline Disability / Functional Limitation Status
- **Category Mapping Status**: Gate 7 verified the source variables, coding definitions, and group derivations. No Panel 27 subgroup result is available while the holdout remains locked.
- **Role**: Protected attributes serve exclusively as audit and evaluation variables; they are not primary predictors.

### 6.2 Primary Fairness Metrics
- **Weighted Subgroup Calibration & Brier Score**: Evaluated independently within each demographic subgroup.
- **Subgroup Error-Rate Gaps**:
  - True Positive Rate (TPR) Gap / Equal Opportunity Disparity (primary fairness endpoint at 10% weighted capacity).
  - False Positive Rate (FPR) Gap / Predictive Equality Disparity.
  - Positive Predictive Value (PPV) Gap / Predictive Parity Disparity.
  - Selection Rate Gap / Demographic Parity Disparity.
- **Bias Concentration Quantity**: Not an active MEPS endpoint. Any simple group-mean-difference helper in the exploratory implementation is an engineering diagnostic only; it is not a Tang et al. (2024) or FairBias metric and must not be presented as reproducing the published method.
- **Reporting Standard**: Disaggregate group-specific values and inter-group gaps with design-aware 95% confidence intervals. No single fairness scalar will be used to assert that a model is "fair."

### 6.3 Subgroup Suppression & Sample Size Rules
To prevent unreliable statistical estimation and disclosure risks in small subgroups:
- An audit subgroup or intersectional cell is **suppressed** from reporting if:
  1. Unweighted sample size $n < 100$, OR
  2. Total positive outcomes (coverage interruptions) $< 20$, OR
  3. Total negative outcomes (retained coverage) $< 20$, OR
  4. Kish effective sample size $n_{\text{eff}} = \frac{(\sum w_i)^2}{\sum w_i^2} < 50$.
- **Suppression Protocol**: Suppressed cells are explicitly labeled `"Suppressed (Insufficient Sample/Power)"`. Under no circumstances will groups be merged *ad hoc* after viewing empirical results.

---

## 7. Complex Survey Weighting & Statistical Uncertainty

### 7.1 Analysis Weights & Domain Representation
- **Analysis Weight**: `LONGWT` is applied to all primary point estimates to produce survey-weighted domain estimates representing the civilian noninstitutionalized US population satisfying the frozen eligible cohort criteria, not an unrestricted claim about all US adults.
- **Design Variables**: `VARSTR` (strata) and `VARPSU` (primary sampling unit) represent the complex sampling structure. They are design variables and **must never be used as predictive features**.

### 7.2 Variance Estimation & Confidence Intervals
- **Primary Methodology**: 95% confidence intervals and paired difference tests are computed using design-aware stratified PSU bootstrap/resampling using `VARSTR` and `VARPSU`, subject to implementation validation.
- **Strict Substitution Policy**: BRR and generic menus of alternative variance estimators are removed. If the survey design or data structure cannot support the validated stratified PSU resampling method, execution must stop for escalation rather than silently substituting an unvalidated method.
- **Multiplicity Control**: No confirmatory mitigation comparison is currently accepted. The implemented centering comparison and all subgroup/intersectional analyses remain exploratory unless a later gate freezes an estimable primary arm.

---

## 8. Sensitivity Analyses

1. **Outcome Definition Sensitivity**:
   - *Sensitivity Outcome 1*: Prolonged uninsurance ($\ge 3$ uninsured months in Year 2).
   - *Sensitivity Outcome 2*: End-of-year uninsurance (uninsured in December / Month 12 of Year 2).
   - *Requirement*: Evaluated only if official codebooks unambiguously support construction.
2. **Survey Weighting Sensitivity**:
   - Unweighted model training and evaluation versus survey-weighted estimation.
3. **Capacity Basis Sensitivity**:
   - Unweighted top $K\%$ sample capacity versus primary survey-weighted population capacity.

---

## 9. Seeds, Resource Limits & Execution Guardrails

### 9.1 Frozen Random Seeds
To guarantee exact reproducibility across all stochastic operations (splitting, initialization, bootstrap resampling), five fixed seeds are frozen:
- Seeds: `20260828`, `20260829`, `20260830`, `20260831`, `20260832`.

### 9.2 Computational Scaling Guardrails
- **Prohibited Operations**: Full sample-by-sample pairwise distance or difference matrices of $O(N^2)$ complexity (e.g., legacy `eval.py` `num-c` / `num-d` operations) are strictly prohibited on full MEPS cohorts.
- **Local Target Hardware**: Apple Silicon Mac (24 GB unified memory).
  - Peak RSS memory limit: $< 16\text{ GB}$.
  - Smoke test execution time: $< 30\text{ minutes}$.
- **Server Escalation**: If computational demands exceed local thresholds, execution may be escalated to a compute server as a resource management decision without altering the research estimand.

---

## 10. Study Stop & Escalation Conditions

Execution must immediately pause and escalate to the supervisor under any of the following conditions:
1. **Target Variable Harmonization Failure**: Monthly insurance status variables cannot be consistently harmonized between HC-244 and HC-252.
2. **Design Structure Invalidation**: Survey design variables (`LONGWT`, `VARSTR`, `VARPSU`) fail structural validity checks (e.g., negative weights, unlinked PSUs).
3. **Statistical Underpowering**: Fewer than 200 eligible positive outcomes across an authorized, frozen development design. The observed HC-244 result fails this threshold. Panel 27 outcomes must not be inspected to evaluate power; Panel 27 remains locked until development power and all independent-review prerequisites are satisfied.
4. **Data Leakage Failure**: Any automated leakage test fails. Note: Nonzero mutual information between legitimate baseline predictors and the Year 2 outcome is expected predictive signal and is not leakage. Leakage failures include:
   - Inclusion of Year 2 or post-baseline columns in predictor feature sets.
   - Person-level or household (`DUID`) overlap across train/validation/calibration partitions.
   - Preprocessing transformers (scalers, encoders, imputers) fitted beyond authorized training partitions.
   - Use of Panel 27 outcomes, subgroup distributions, or performance metrics for pipeline tuning or selection.
5. **Variance Estimation Failure**: If stratified PSU bootstrap resampling cannot be validated or supported on the design structure.
6. **Manifest Irreproducibility**: A completed pipeline run cannot be reproduced identically from its execution manifest and random seed.

---

## 11. Decisions Frozen Now vs. Deferred to Gate 7

| Analysis Decision | Gate 3 Status | Gate 7 Action |
|---|---|---|
| Primary Estimand & Target Population Definition | **FROZEN** | Confirmed against decoded codebook |
| Both-Years Eligibility Rule (`YEARIND == 1`) | **FROZEN** | Verified in data extraction filter |
| Candidate development sequence (HC-217/225/234/244) / HC-252 holdout | **FROZEN AS METADATA ONLY** | Pooling and outcome access require a separate gate |
| DUID-Grouped Partitioning (60/20/20) & Refit Flow | **FROZEN** | Split assigned and hashed |
| Survey-Weighted Platt Calibration on Calibration Set | **FROZEN** | Implemented in calibrator module |
| Fixed 10% Weighted Calibration Threshold Policy | **FROZEN** | Frozen and transferred to holdout |
| Implemented method-arm naming | **FROZEN** | Exploratory group-aware centering; not FairBias/Tang |
| Subgroup Suppression Rules ($n<100$, pos/neg $<20$, $n_{\text{eff}}<50$) | **FROZEN** | Implemented in reporting tables |
| Reproducibility Seeds (`20260828`–`20260832`) | **FROZEN** | Bound in configuration files |
| Exact Monthly Insurance Variable Names & Codes | **VERIFIED** | Canonical mapping in `configs/cohort_and_variables.json` |
| Exact 74-variable baseline predictor list | **VERIFIED** | 19 continuous + 55 categorical |
| Protected-dimension sources and derivations | **VERIFIED** | `RACETHX`, `SEX`, poverty, age-band and disability contracts |
| Baseline Year-End Age Variable Name | **VERIFIED** | `AGEY1X` |
