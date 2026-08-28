# 2. Two-Panel Temporal Validation Strategy for MEPS Longitudinal Evaluation

Date: 2026-08-28

## Context

Evaluating machine learning models and algorithmic fairness interventions on longitudinal survey data requires validation strategies that evaluate temporal transport across survey panels. In health insurance retention programs, models trained on historical baseline survey data are evaluated for their ability to predict coverage interruption in subsequent cohorts.

A conventional evaluation design would utilize a single survey panel (e.g., MEPS HC-252 Panel 27) and perform a random row-level or household-level split (e.g., 80% train / 20% test). However, this within-panel approach has critical limitations:
1. **Concurrent vs. Temporal Evaluation**: A random split within a single panel evaluates whether a model generalizes to unobserved households in the same survey years, masking temporal drift and cross-year population shifts across consecutive panels.
2. **Test Isolation Vulnerabilities**: Iterative model development and hyperparameter exploration on a single panel create risks of informational leakage and parameter overfitting across validation folds.

To establish a rigorous temporal evaluation framework within MEPS, we adopt a two-panel design leveraging MEPS HC-244 Panel 26 Longitudinal Data Public Use File (2021–2022) for model development and MEPS HC-252 Panel 27 Longitudinal Data Public Use File (2022–2023) as a locked temporal holdout.

---

## Decision

1. **Adopt Two-Panel Temporal Validation Architecture**:
   - **Development Cohort**: MEPS HC-244 Panel 26 Longitudinal Data Public Use File (2021–2022) is designated exclusively for all development activities. Within Panel 26, individuals are partitioned strictly by Dwelling Unit ID (`DUID`) into 60% Training, 20% Model & Mitigation Validation, and 20% Calibration sets.
   - **Frozen Development Flow**: Preprocessing is fit on the 60% training partition; validation is used solely to select hyperparameters, feature policies, and mitigation settings; frozen choices are refit on Train + Validation (80%); a survey-weighted Platt logistic calibrator is fit on the untouched 20% calibration partition; and one conventional probability threshold is frozen at 10% weighted population selection.
   - **Temporal Holdout Cohort**: MEPS HC-252 Panel 27 Longitudinal Data Public Use File (2022–2023) is designated exclusively as a locked temporal holdout.
2. **Strict Test Isolation Rules**:
   - Panel 27 target values, feature distributions, subgroup prevalences, and metrics are completely inaccessible for any tuning, model selection, transformation fitting, threshold optimization, or protected group definition.
   - Pre-lock checks are strictly limited to schema column names and types, documented-code compatibility, file dimensions/hashes, and non-outcome structural integrity checks.
   - Formal evaluation on Panel 27 is executed exactly once after freezing the entire pipeline. Re-runs are permitted only for verified code-error correction accompanied by an incident log.
3. **Precise Claim Boundaries & Terminology**:
   - Panel 27 must be described strictly as a **temporal holdout** providing a stronger temporal transport assessment within MEPS than single-panel cross-validation.
   - Prohibited Terminology: Panel 27 must **never** be characterized as an "independent external validation cohort" or "deployment simulation", because both panels share MEPS sampling frames, survey instruments, and weighting methodologies. Temporal drift is measured descriptively without causal attribution.

---

## Consequences & Trade-offs

- **Positive (Temporal Transport Assessment)**: Provides a stronger temporal transport assessment within MEPS by testing whether models trained on 2021 baseline features (predicting 2022 interruptions) maintain discrimination, calibration, and subgroup error-rate parity when applied to 2022 baseline features (predicting 2023 interruptions).
- **Positive (Strict Test Isolation)**: Complete separation between the development panel (HC-244) and the holdout panel (HC-252) prevents subtle data snooping or over-optimistic parameter tuning.
- **Positive (Calibration Drift Assessment)**: Enables explicit measurement of calibration slope and intercept shifts across consecutive survey panels, assessed descriptively without causal attribution.
- **Negative (Sample Size Allocation)**: Limits the development sample size to HC-244 rather than pooling both panels. However, pooling would destroy temporal holdout validity.
- **Negative (Harmonization Burden)**: Requires rigorous variable-level harmonization between Panel 26 and Panel 27 codebooks in Gate 7.

