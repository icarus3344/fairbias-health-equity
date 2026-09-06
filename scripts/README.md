# CLI Scripts & Reproduction Guide

This directory provides command-line execution harnesses for data ingestion, pipeline training, and model evaluation.

```text
scripts/
├── download_nhis.py                  # Downloads CDC NHIS (2022-2024) microdata with checksum verification
├── prepare_nhis.py                   # Harmonizes raw survey files into interim cohort Parquet tables
├── prepare_nhis_features.py          # Extracts feature matrices and categorical encoders
├── run_fairbias_benchmark.py         # Runs FairBias benchmark on legacy datasets (COMPAS, Credit Card)
├── run_nhis_d4_preflight.py          # Preflight checks for D4 master split experiment
├── run_nhis_d4_test_release.py       # Executes D4 test set evaluation
├── run_nhis_d5_weighted_preflight.py # Preflight checks for D5 survey-weighted experiment
├── run_nhis_d5_weighted_release.py   # Executes D5 survey-weighted train/val release
├── run_nhis_d5_weighted_test_release.py # Executes D5 survey-weighted test release
├── run_nhis_d6_temporal.py           # Executes D6 longitudinal temporal robustness experiment
├── run_nhis_d6_temporal_test.py      # Executes D6 out-of-time test barrier evaluation
├── run_nhis_d7_stepwise_replay.py    # Replays 5-step mitigation trajectory and records feature attribution
└── run_nhis_d7_terminal_mechanism.py # Terminal mechanism decomposition and feature logit audit
```

---

## 🛠️ Reproduction Workflows

### 1. D7 Stepwise Replay & Mechanism Audit (Primary Entrypoint)
Deconstructs the feature-level logit shifts and fairness metric trajectories across all optimization steps:
```bash
PYTHONPATH=src:. python scripts/run_nhis_d7_stepwise_replay.py
```
- **Inputs**: Harmonized NHIS 2022–2024 pooled cohort.
- **Outputs**: Stepwise CSVs and metrics recorded in `docs/releases/NHIS_D7_STEPWISE_REPLAY_V1_fc248b66/`.
- **Runtime**: ~3–5 minutes.

### 2. D6 Temporal Out-of-Time Robustness
Evaluates whether fairness gains achieved on historical cohorts (2022) persist across subsequent survey years (2023–2024):
```bash
PYTHONPATH=src:. python scripts/run_nhis_d6_temporal.py
```
- **Outputs**: Summary metrics and cross-year comparison tables in `docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/`.

### 3. D5 Complex Survey-Weighted Geometry Release
Executes the survey-weighted manifold debiasing pipeline:
```bash
PYTHONPATH=src:. python scripts/run_nhis_d5_weighted_release.py
```
