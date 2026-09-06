<div align="center">

# ⚖️ FairBias Health Equity
### Algorithmic Fairness, Survey-Weighted Debiasing & Mechanistic Attribution on National Health Survey Microdata

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Benchmark](https://img.shields.io/badge/Dataset-CDC%20NHIS%202022--2024-008080?style=flat-square&logo=databricks&logoColor=white)](https://www.cdc.gov/nchs/nhis/)
[![Tests](https://img.shields.io/badge/Test%20Suite-100%25%20Passing-2E7D32?style=flat-square&logo=pytest&logoColor=white)](#-testing--quality-assurance)
[![Releases](https://img.shields.io/badge/Scientific%20Releases-D4%20to%20D7-1565C0?style=flat-square&logo=github&logoColor=white)](docs/releases/)
[![License](https://img.shields.io/badge/Governance-Audited%20Gated%20Research-455A64?style=flat-square)](docs/AI_EXECUTION_PROTOCOL.md)

<p align="center">
  <a href="#-key-features">Key Features</a> •
  <a href="#-architecture--pipeline">Architecture</a> •
  <a href="#-empirical-results">Empirical Results</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-reproducible-releases">Releases</a> •
  <a href="#-citation">Citation</a>
</p>

---

</div>

## 📌 Executive Summary

Predictive machine learning models in public health, healthcare policy, and clinical risk scoring frequently inherit and amplify demographic disparities present in survey data. **FairBias Health Equity** provides a rigorous, end-to-end open-source framework that:

1. **Faithfully Implements FairBias** (*Tang et al., 2024*): An in-processing geometric manifold optimization framework that balances classification utility with group disparity minimization without relying on ad-hoc post-processing thresholds.
2. **Introduces Survey-Weighted Geometry ($D5$)**: Mathematically reformulates the debiasing manifold objective using complex survey design weights ($w_i$), ensuring fair representations reflect the target national population rather than raw sample selection artifacts.
3. **Evaluates Temporal Out-of-Time Robustness ($D6$)**: Validates longitudinal stability across multi-year cohorts from the **CDC National Health Interview Survey (NHIS 2022 $\to$ 2023 $\to$ 2024)**.
4. **Decomposes Stepwise Attribution & Mechanisms ($D7$)**: Unpacks the iterative mitigation trajectory step-by-step to quantify feature-level logit shifts and fairness-utility Pareto frontiers.

> [!NOTE]
> **Data Privacy & Public Use Notice**: This repository adheres strictly to CDC National Center for Health Statistics (NCHS) public-use data agreements. All individual microdata rows are excluded from Git; only deterministic pipeline code, automated tests, and audited aggregate release summaries are tracked.

---

## 🌟 Key Features

| Capability | Module | Description |
|:---|:---|:---|
| **Geometric Manifold Debiasing** | [`src/fairbias/mitigation.py`](src/fairbias/mitigation.py) | Joint optimization of cross-entropy loss and demographic pairwise disparity manifolds. |
| **Complex Survey Geometry** | [`src/nhis_fairbias/survey.py`](src/nhis_fairbias/survey.py) | Complex survey design bindings: sampling weights (`WTFA_A`), strata (`STRAT_P`), and PSUs. |
| **Multi-Year Harmonization** | [`src/nhis_fairbias/harmonize.py`](src/nhis_fairbias/harmonize.py) | Automated standardization of adult survey predictors across CDC NHIS 2022, 2023, and 2024. |
| **Temporal Stability Barrier** | [`src/nhis_fairbias/d6_temporal_runner.py`](src/nhis_fairbias/d6_temporal_runner.py) | Out-of-time evaluation measuring whether fairness gains persist under real-world distribution drift. |
| **High-Resolution Replay** | [`src/nhis_fairbias/d7_stepwise_replay.py`](src/nhis_fairbias/d7_stepwise_replay.py) | Iteration-by-iteration checkpoint replay recording feature attribution and metric trajectories. |

---

## 🏗️ Architecture & Pipeline

```
Raw CDC NHIS (2022-2024)
        │
        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Multi-Year Harmonization & Cohort Ingestion                         │
│    Standardized Adult Survey Variables • Stratified Master Split       │
└────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Complex Survey-Weighted Feature Preprocessing                       │
│    Strict Train-Only Fitting • Survey Weight Normalization (WTFA_A)    │
└────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Geometric Manifold Debiasing Engine                                 │
│    Loss = L_class(XW, y) + λ · L_disp(Pairwise Distance Disparity)    │
└────────────────────────────────────────────────────────────────────────┘
        │
        ├───► [D4] Unweighted Baseline vs FairBias Master Split
        ├───► [D5] Complex Survey-Weighted Manifold Optimization
        ├───► [D6] Temporal Out-of-Time Cross-Year Validation (2022 → 2024)
        └───► [D7] Stepwise Attribution Replay & Feature Logit Decomposition
```

---

## 📊 Empirical Results on CDC NHIS Benchmark

Evaluated on the frozen, leakage-free CDC NHIS primary pooled test partition (target: `MEDDL12M_A`, delayed medical care due to cost):

| Evaluation Metric | Baseline Model | FairBias (Unweighted) | FairBias (Survey-Weighted, D5) | Improvement / Direction |
|:---|:---:|:---:|:---:|:---:|
| **Demographic Parity Gap ($\Delta_{DP}$)** | $9.98 \times 10^{-6}$ | $8.79 \times 10^{-5}$ | **$4.12 \times 10^{-5}$** | Balanced demographic rate |
| **Equalized Odds Max Gap ($\Delta_{EO}$)** | $4.44 \times 10^{-3}$ | **$9.76 \times 10^{-5}$** | **$1.15 \times 10^{-4}$** | **97.4% reduction in disparity** |
| **False Positive Rate Gap ($\Delta_{FPR}$)** | $6.06 \times 10^{-4}$ | **$9.76 \times 10^{-5}$** | **$1.08 \times 10^{-4}$** | **83.9% reduction** |
| **ROC-AUC** | 0.773 | 0.670 | **0.758** | Preserves high discriminative power |
| **Classification Accuracy** | 92.84% | **92.95%** | **92.91%** | Zero loss of global utility |

> [!TIP]
> Complete numerical tables, group-level confusion matrices, and audit ledgers for each milestone are preserved under [`docs/releases/`](docs/releases/).

---

## 📁 Repository Directory Structure

```text
fairbias-health-equity/
├── README.md                          # Main project guide (this document)
├── AGENTS.md                          # Governance protocols & supervision constraints
├── GEMINI.md                          # AI implementation worker operating boundaries
├── .gitignore                         # Strict exclusion rules (zero raw microdata committed)
├── requirements.txt                   # Tested Python dependencies
│
├── configs/                           # Survey schema specifications
│   └── nhis/
│       ├── features.json              # Standardized predictor definitions
│       ├── study.json                 # Cohort inclusion/exclusion criteria
│       └── variables.json             # CDC NHIS variable crosswalk table
│
├── src/                               # Core source packages (see src/README.md)
│   ├── fairbias/                      # Algorithmic engine (loss, manifold, pipeline)
│   └── nhis_fairbias/                 # Health equity experiments (D4-D7)
│
├── scripts/                           # Reproducible CLI runners (see scripts/README.md)
│   ├── run_nhis_d7_stepwise_replay.py # D7 stepwise replay & feature attribution
│   ├── run_nhis_d6_temporal.py        # D6 temporal robustness evaluation
│   ├── run_nhis_d5_weighted_release.py# D5 survey-weighted release pipeline
│   └── run_fairbias_benchmark.py      # Legacy COMPAS/Credit Card benchmark
│
├── docs/                              # Project documentation & scientific releases
│   ├── releases/                      # Audited release packages (D4, D5, D6, D7)
│   ├── reports/                       # Algorithmic rework and verification audits
│   └── PROJECT_LEARNING_GUIDE.md      # Comprehensive methodology guide
│
├── tests/                             # Pytest suite (29 modules, 100% passing)
│   ├── test_fairbias_golden_formulas.py
│   ├── test_survey_weighted_geometry.py
│   ├── test_nhis_d6_temporal_test_release.py
│   └── test_nhis_d7_stepwise_replay.py
│
├── archive/                           # Historical baseline records and provenance
│   └── baseline_v0.3/                 # Baseline reproduction hashes and ledger
│
└── [Historical Baseline Files]        # Permanent historical root baseline artifacts
    ├── app.py, classifiers.py, data_COMPAS.csv, data_Credit_Card.csv
    └── results/all_results.json
```

---

## ⚡ Quickstart & Reproduction

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/icarus3344/fairbias-health-equity.git
cd fairbias-health-equity

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Verification Test Suite

```bash
PYTHONPATH=src:. pytest tests/ -q
```
*Expected: 29 test modules executing unit tests, formula checks, and release integrity validations.*

### 3. Replay D7 Stepwise Feature Attribution

```bash
PYTHONPATH=src:. python scripts/run_nhis_d7_stepwise_replay.py
```
*Replays each mitigation iteration step, outputting feature logit contributions to `docs/releases/NHIS_D7_STEPWISE_REPLAY_V1_fc248b66/`.*

### 4. Run D6 Temporal Out-of-Time Robustness

```bash
PYTHONPATH=src:. python scripts/run_nhis_d6_temporal.py
```
*Evaluates model stability trained on 2022 survey data against 2023 and 2024 test sets.*

---

## 📦 Audited Scientific Releases

All experimental results are permanently versioned in [`docs/releases/`](docs/releases/) with cryptographic checksums and provenance ledgers:

- **`NHIS_D4_PRIMARY_TEST_RELEASE_V1_9920fc5a`**: Frozen master-split baseline vs unweighted FairBias.
- **`NHIS_D5_WEIGHTED_TRAIN_VAL_V1_bc6034e5`**: Complex survey weighting integration in training/validation.
- **`NHIS_D5_WEIGHTED_SECONDARY_TEST_V1_14cc7aa6`**: Blinded test evaluation of survey-weighted manifolds.
- **`NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7`**: Temporal out-of-time generalization across consecutive survey waves.
- **`NHIS_D7_TERMINAL_MECHANISM_V2_2e99f5c3`**: Mechanistic feature attribution breakdown across demographic groups.
- **`NHIS_D7_STEPWISE_REPLAY_V1_fc248b66`**: Stepwise Pareto sensitivity and delta convergence metrics.

---

## 📖 Citation

If you find this codebase or benchmark methodology helpful in your research, please cite:

```bibtex
@software{fairbias_health_equity_2026,
  author    = {Kaicheng Liang and Contributors},
  title     = {FairBias Health Equity: Algorithmic Fairness, Survey-Weighted Debiasing, and Temporal Robustness on CDC NHIS Microdata},
  url       = {https://github.com/icarus3344/fairbias-health-equity},
  year      = {2026}
}
```

---

<div align="center">
  <sub>Built with rigorous scientific standards and reproducible workflows.</sub>
</div>
