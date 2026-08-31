# Fair Allocation of Health Insurance Retention Outreach in Complex Longitudinal Surveys: An Empirical Investigation of Algorithmic Disparities in MEPS

**Working Title**: Fair Allocation of Health Insurance Retention Outreach in Complex Longitudinal Surveys  
**Target Submission**: TBD (Target venues under consideration: Machine Learning for Healthcare [MLHC] / ACM CHIL / FAccT; deadlines TBD)  
**Status**: Exploratory Internal Development Report (Study Stopped Fail-Closed Before Panel 27 Evaluation; No Conference Paper Result Package Complete)  
**Date**: 2026-08-29  

---

## Abstract

Health insurance coverage continuity is a critical determinant of healthcare access and financial protection in the United States. While proactive outreach programs (such as navigators and renewal assistance) can mitigate administrative disenrollment, outreach budgets are heavily capacity-constrained. Machine learning models trained on longitudinal survey data offer potential for risk stratification, but predictive models risk exacerbating systemic disparities across protected demographic groups. In this study, we formulate a survey-weighted risk prediction and fair-allocation framework developed on longitudinal cohorts from the Medical Expenditure Panel Survey (MEPS HC-244 Panel 26, 2021–2022) using a verified zero-leakage 74-predictor baseline feature set (19 continuous, 55 categorical). We evaluate a group-agnostic survey-weighted logistic regression baseline alongside an exploratory group-aware centering heuristic and its survey-weighted extension. Under a prespecified 10% weighted operational capacity constraint, we evaluate apparent calibration-fit diagnostics across five random development seeds (`20260828`–`20260832`): the unmitigated baseline achieved mean Weighted AUROC $0.5564 \pm 0.0519$ and Weighted AUPRC $0.0638 \pm 0.0245$, while the exploratory survey-weighted centering heuristic achieved Weighted AUROC $0.5580 \pm 0.0503$ and Weighted AUPRC $0.0750 \pm 0.0409$. Crucially, under our strict cell suppression protocol ($n < 100 \lor \text{pos} < 20 \lor \text{neg} < 20 \lor n_{\text{eff}} < 50$), 100% of race/ethnicity and sex subgroup cells on the evaluation partition were suppressed; thus, no formal disparity comparison is estimable. Furthermore, because Panel 26 yielded only 136 eligible positive coverage interruption events (falling short of the prespecified 200-event threshold), the study executed a mandatory fail-closed stop condition: Panel 27 (HC-252) remained completely locked, zero temporal holdout evaluation was conducted, and no conference-ready validation package is complete. All findings reported herein represent exploratory internal development evidence.

---

## 1. Introduction & Ethical Scope

### 1.1 Policy Context & Motivation
Under the Affordable Care Act and post-pandemic Medicaid renewals, administrative coverage interruptions represent a substantial public health challenge. Individuals frequently lose coverage not due to true eligibility changes, but because of burdensome re-enrollment procedures, lack of navigation assistance, and documentation barriers. Proactive outreach (e.g., multilingual navigation assistance, targeted renewal notices) is effective but resource-constrained. Public health agencies must allocate these supportive interventions efficiently and equitably.

### 1.2 Explicit Beneficial Scope & Prohibited Applications
In accordance with ethical AI governance and the study protocol:
- **Authorized Beneficial Purpose**: Allocating supportive retention outreach, multilingual assistance, and proactive renewal navigation to individuals at elevated risk of administrative coverage interruption.
- **Strictly Prohibited Applications**: Underwriting, actuarial pricing, coverage denial or rescission, benefit reduction, Medicaid eligibility exclusion, or any punitive decision-making.

---

## 2. Study Population & Cohort Flow

### 2.1 Data Source
We utilize public-use longitudinal data from the Medical Expenditure Panel Survey (MEPS), sponsored by the Agency for Healthcare Research and Quality (AHRQ) and the National Center for Health Statistics (NCHS).
- **Development Panel**: MEPS HC-244 Panel 26 Longitudinal Data PUF (2021–2022, 6,741 total person records).
- **Temporal Holdout Panel**: MEPS HC-252 Panel 27 Longitudinal Data PUF (2022–2023, 8,292 total person records; maintained strictly LOCKED under underpowered stop condition).

### 2.2 Inclusion Criteria & Estimand
1. **Age**: 18–64 years old at the end of baseline Year 1 (`18 <= AGEY1X <= 64`).
2. **Survey Continuity**: In scope in both survey years (`YEARIND == 1`) and complete data collection across all 5 rounds (`ALL5RDS == 1`).
3. **Survey Weighting**: Positive longitudinal analysis weight (`LONGWT > 0`).
4. **Baseline Continuous Coverage**: Covered by private or public health insurance across all 12 calendar months of Year 1 (`INS<mon>Y1X == 1`).

**Analytic Cohort Size (Panel 26)**: Exactly 2,882 individuals across 1,840 dwelling units (`DUID`).

### 2.3 Target Outcome & Power Finding
- **Primary Outcome ($Y \in \{0, 1\}$)**: Experiencing at least one month of uninsurance during follow-up Year 2 ($\exists m: \text{INS}m\text{Y2X} == 2$).
- In Panel 26, exactly 136 individuals experienced coverage interruption ($Y=1$, unweighted event rate 4.72%).
- **Underpowering Finding**: The observed 136 positive events fell short of the prespecified statistical power requirement of $\ge 200$ positive cases, triggering the fail-closed stop condition before Panel 27 holdout evaluation.
- **Sensitivity Outcomes**: Prolonged uninsurance ($\ge 3$ months) and year-end uninsurance (`INSDEY2X == 2`).

---

## 3. Methodology & Gated Development Flow

### 3.1 Household-Grouped Partitioning & Zero-Leakage Architecture
To prevent intra-household leakage, partitioning is performed strictly at the Dwelling Unit (`DUID`) level:
- **Training Partition (60%, $n=1,732$)**: Used for fitting feature transformations and initial model training.
- **Validation Partition (20%, $n=569$)**: Used for hyperparameter tuning.
- **Calibration Partition (20%, $n=581$)**: Isolated from model training; used for survey-weighted Platt calibration and decision threshold freezing.
- **Refit Partition (80%, $n=2,301$)**: Models are refit on combined Train+Validation data before calibration.

### 3.2 Train-Only Preprocessing & Missing Value Contracts
- Conservative zero-leakage predictor set: exactly 74 baseline features (19 continuous, 55 categorical) spanning verified Round 1, Round 2, and annual Year 1 variables. All Round 3 variables are strictly excluded because longitudinal documentation demonstrates Round 3 is fielded in Year 2.
- Continuous income/poverty variables (`TTLPY1X`, `FAMINCY1`, `POVLEVY1`) preserve legitimate negative values; only exact codebook sentinels (`-1`, `-7`, `-8`, `-15`) are treated as missing.
- Inapplicable employment codes (`HOUR1`, `HOUR2`, `NUMEMP1`, `NUMEMP2`, `WAGEPY1X == -1`) are recoded to structural zero ($0.0$).
- Prior-round inheritance (`-2: DETERMINED IN PREVIOUS ROUND`) for `HOUR2`, `NUMEMP2`, `CHOIC2`, `SELFCM2`, `UNION2` is resolved to Round 1 values.
- Categorical variables preserve codebook-defined inapplicable categories (`-1`) as distinct structural dummies, while item non-response codes (`-7`, `-8`, `-15`) are treated as missing.
- Demographic protected attributes (`RACETHX`, `SEX`) are strictly excluded from the baseline feature matrix $X$.

### 3.3 Model Formulations & Mitigation Status
1. **Primary Baseline (Group-Agnostic)**: Survey-weighted Logistic Regression with L2 regularization (`WeightedLogisticClassifier`). Operates strictly on $X$ without requiring protected attributes at inference time.
2. **Exploratory Group-Aware Centering Heuristic**: Computes group-conditional offsets and applies disparity shrinkage. *Methodological notice*: The intended paper arm (Tang et al., 2024) was not implemented; current method is an exploratory group-aware heuristic at inference and not the primary paper result. Protected attributes are never silently injected into the predictor feature matrix $X$.
3. **Exploratory Survey-Weighted Centering Extension**: Incorporates survey weights (`LONGWT`) into group conditional means $\mathbb{E}_w[X_j \mid O=g] = \frac{\sum_{i \in g} w_i X_{i,j}}{\sum_{i \in g} w_i}$.

### 3.4 Capacity-Constrained Decision Threshold Freezing & Calibration Diagnostics Boundary
- Under a fixed 10% operational capacity budget, we sort calibrated probabilities $\hat{p}_i$ and identify the threshold $\tau_{10\%}$ capturing the top 10% of cumulative represented population weight on the Calibration partition.
- *Methodological Boundary*: Metrics computed on the calibration partition where Platt scaling was fit are **apparent calibration-fit diagnostics**, not out-of-sample or holdout validation performance.

### 3.5 Subgroup Fairness Audit & Strict Suppression
We audit subgroup performance across OMB 5-category race/ethnicity (`RACETHX`) and sex (`SEX`). In compliance with DUA disclosure prevention rules, cells with $n < 100 \lor \text{pos} < 20 \lor \text{neg} < 20 \lor n_{\text{eff}} < 50$ are strictly marked `"Suppressed (Insufficient Sample/Power)"` without ad hoc group collapsing.

---

## 4. Exploratory Internal Development Evidence (Panel 26 5-Seed Summary)

*Provenance Note*: All metrics below are exploratory internal development evidence computed from 5-seed Panel 26 executions (`runs/formal_panel26_summary.json`, seeds `20260828`–`20260832`), evaluated as apparent fit diagnostics on the 20% calibration partition ($n=581$). They do NOT represent conference-ready validation or temporal holdout evaluation.

### 4.1 Discrimination & Apparent Calibration Diagnostics
Across 5 development random seeds (`20260828`–`20260832`):
- **Weighted AUROC**:
  - Unmitigated Baseline: $0.5564 \pm 0.0519$ (individual runs: $0.5626, 0.4992, 0.6149, 0.5973, 0.5079$)
  - Exploratory Survey-Weighted Mitigation: $0.5580 \pm 0.0503$ (individual runs: $0.5700, 0.5108, 0.6164, 0.5914, 0.5013$)
  - Paired Difference $\Delta \text{AUROC}$: $+0.0016 \pm 0.0084$
- **Weighted AUPRC**:
  - Unmitigated Baseline: $0.0638 \pm 0.0245$ (individual runs: $0.0557, 0.0452, 0.0834, 0.0700, 0.0645$)
  - Exploratory Survey-Weighted Mitigation: $0.0750 \pm 0.0409$ (individual runs: $0.0565, 0.0515, 0.1474, 0.0636, 0.0562$)
  - Paired Difference $\Delta \text{AUPRC}$: $+0.0112 \pm 0.0177$
- **Weighted Brier Score**:
  - Unmitigated Baseline: $0.0458 \pm 0.0041$
  - Exploratory Survey-Weighted Mitigation: $0.0456 \pm 0.0042$
- **Capacity Diagnostics at 10% Operational Budget**:
  - Unmitigated: Recall@10% $0.1225 \pm 0.1144$, Precision@10% $0.0596 \pm 0.0553$
  - Exploratory Mitigation: Recall@10% $0.1801 \pm 0.1451$, Precision@10% $0.0854 \pm 0.0689$

### 4.2 Subgroup Disparity Audit & Cell Suppression Reality
- **Subgroup Cell Suppression Rate**: **100% of race/ethnicity and sex subgroup cells on the evaluation partition were suppressed**.
- Across all 5 seeds, every demographic group in the 20% calibration partition had fewer than 20 positive cases (e.g., Hispanic: 4 positives, NH White: 15 positives, NH Black: 3 positives, Asian: 1 positive, Other: 1 positive; Male: ~12 positives, Female: ~12 positives).
- **Primary Fairness Endpoint**: Because zero subgroup cells met the minimum sample size ($n \ge 100$) and event count ($\text{pos} \ge 20$) thresholds, **all subgroup max TPR gaps and primary fairness endpoints are Not Estimable (`None`) on the evaluation partition**. All suppressed cells yield Not Estimable without fabricating a numeric zero, and design-aware bootstrap fairness inference propagates non-estimability (`fairness_max_tpr_gap` status `NOT_ESTIMABLE_SUPPRESSED`). Utility bootstrap inference remains estimable and valid. Any prior claim of numerical disparity reduction is entirely unsupported and withdrawn; all race/ethnicity and sex subgroup cells on the evaluation partition were suppressed.


---

## 5. Statistical Power Finding & Fail-Closed Holdout Lock

In accordance with Section 10 of the Statistical Analysis Plan:
- Panel 26 eligible positive outcome cases totaled **136 cases**, falling below the prespecified 200-positive statistical power threshold.
- The pipeline executed a strict fail-closed stop condition: **Panel 27 (HC-252) remains 100% LOCKED under `LOCKED_UNDERPOWERED_STOP_CONDITION`**.
- Zero Panel 27 outcome data, distributions, protected-group summaries, prevalence, or metrics were accessed, preserving complete holdout integrity for future prospective evaluation when sufficient statistical power is available.
- **Scientific Status**: The research project is halted at the internal development stage. No conference paper result package is complete.

---

## 6. Formal AHRQ MEPS Citation Guidance

All research utilizing these materials must include the following formal citations:
1. **MEPS HC-244**: Agency for Healthcare Research and Quality. *Medical Expenditure Panel Survey HC-244: 2021–2022 Panel 26 Longitudinal Data File*. Rockville, MD: AHRQ, September 2024.
2. **MEPS HC-252**: Agency for Healthcare Research and Quality. *Medical Expenditure Panel Survey HC-252: 2022–2023 Panel 27 Longitudinal Data File*. Rockville, MD: AHRQ, September 2025.
3. **DUA Compliance**: This research was conducted in compliance with the AHRQ Data Use Agreement for Public Use Files, solely for statistical analysis and reporting, with zero re-identification or external identifiable record linkage.

