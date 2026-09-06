# Source Architecture & Module Guide

The `src/` directory contains the core implementation of the **FairBias** algorithmic framework and its empirical application to the **CDC NHIS 2022–2024** public health survey benchmarks.

```text
src/
├── fairbias/                # Core Algorithm Engine (Tang et al., 2024 & Survey Extensions)
│   ├── bias_metric.py       # Demographic Parity, Equalized Odds, Accuracy, and Metric Registry
│   ├── config.py            # Hyperparameters, optimization settings, and runtime configs
│   ├── data.py              # Data loaders, one-hot encoding, and feature transformers
│   ├── enhancement.py       # Accuracy enhancement routines (post-processing / calibration)
│   ├── evaluator.py         # Multi-metric evaluation suites and threshold tuning
│   ├── mitigation.py        # Geometric manifold optimization & bias mitigation algorithms
│   ├── models.py            # Classifier interfaces (Logistic Regression, LightGBM, MLP)
│   ├── pipeline.py          # Unified FairBias end-to-end execution pipeline
│   ├── transform.py         # Leakage-free feature transformation routines
│   └── transform_trace.py   # High-resolution transformation tracking and step logging
│
└── nhis_fairbias/           # CDC NHIS 2022-2024 Healthcare Fairness Benchmark
    ├── schema.py            # Survey variable schema and validation rules
    ├── harmonize.py         # Multi-year survey variable harmonization across 2022-2024
    ├── survey.py            # Complex survey design: sampling weights, strata, and PSU bindings
    ├── features.py          # Harmonized feature space extraction and categorical encoding
    ├── pooled.py            # Pooled multi-year cohort construction and master split
    ├── preprocessing.py     # Leakage-free preprocessors fit strictly on training partitions
    ├── adapter.py           # Interface adapter linking NHIS cohorts to the FairBias engine
    ├── evaluation.py        # Survey-weighted evaluation metrics and bootstrap inference
    ├── audit.py             # Subgroup fairness audits and distribution checks
    ├── d4_runner.py         # D4 Master Split baseline runner
    ├── d5_weighted_runner.py# D5 Complex survey-weighted manifold debiasing runner
    ├── d6_temporal_runner.py# D6 Longitudinal temporal robustness (2022 -> 2023 -> 2024)
    └── d7_stepwise_replay.py# D7 High-resolution stepwise feature attribution & mechanism replay
```

---

## 1. `fairbias/`: Core Algorithmic Engine

The `fairbias` package is a faithful, modular implementation of the FairBias debiasing framework:
- **`mitigation.py`**: Optimizes the geometric manifold projection $\mathbf{Z} = \mathbf{X} \mathbf{W}$ by jointly minimizing cross-entropy classification loss and geometric distance disparity between protected demographic groups.
- **`bias_metric.py`**: Computes standard algorithmic fairness metrics including:
  - **Equalized Odds (EO)**: Disparity in True Positive Rate (TPR) and False Positive Rate (FPR) across groups.
  - **Demographic Parity (DP / SP)**: Difference in positive prediction rates across groups.
  - **Survey-Weighted Extensions**: Adapts fairness metrics to incorporate complex survey sampling weights ($w_i$).
- **`pipeline.py`**: Coordinates the entire lifecycle: feature scaling $\to$ manifold optimization $\to$ classifier training $\to$ evaluation.

---

## 2. `nhis_fairbias/`: CDC NHIS Healthcare Equity Benchmark

The `nhis_fairbias` package scales the algorithmic framework to real-world national health survey microdata:

### Data Ingestion & Harmonization (`harmonize.py`, `survey.py`)
- Standardizes CDC NHIS survey variables across the 2022, 2023, and 2024 releases.
- Preserves survey design variables (`WTFA_A` sampling weights, `STRAT_P` strata, `PSU_P` primary sampling units) for survey-weighted estimation.

### Research Milestones & Experiment Modules:
1. **D4 Master Split (`d4_runner.py`, `d4_test_release.py`)**:
   - Master stratified split isolating train ($64\%$), validation ($16\%$), and test ($20\%$) cohorts.
   - Evaluates baseline models against standard unweighted FairBias.
2. **D5 Survey-Weighted Geometry (`d5_weighted_runner.py`, `d5_weighted_release.py`)**:
   - Integrates survey sampling weights into the pairwise distance matrices and loss gradients of the manifold transformation.
   - Guarantees debiasing reflects the true target population rather than sample unbalances.
3. **D6 Temporal Out-of-Time Robustness (`d6_temporal_runner.py`, `d6_temporal_test_release.py`)**:
   - Tests longitudinal stability by training on historical survey cohorts (2022) and evaluating out-of-time on subsequent survey years ($2023 \to 2024$).
4. **D7 Stepwise Replay & Terminal Mechanism Audit (`d7_stepwise_replay.py`, `d7_terminal_mechanism.py`)**:
   - Step-by-step replay through the 5 iteration checkpoints of the mitigation algorithm.
   - Quantifies the logit contributions and individual feature attribution shifts driving the fairness-accuracy Pareto frontier.
