# Algorithmic Fairness Research: MEPS HC-252 Longitudinal Study

## Overview

This repository contains two distinct tracks of work:
1. **Inherited Baseline Track (`code_v_0_3`)**: An immutable historical codebase evaluating algorithmic fairness methods on legacy benchmark datasets (COMPAS, Credit Card). This legacy code is preserved in place solely for provenance, auditability, and comparative baselining. It contains known methodological limitations and does not represent scientifically validated findings.
2. **MEPS Research Track (`research/meps-hc252-longitudinal`)**: A separate, rigorously governed research track investigating longitudinal prediction of next-year health-insurance interruption using the Medical Expenditure Panel Survey (MEPS HC-252 Panel 27). This study is designed strictly for beneficial retention and outreach interventions—never for underwriting, pricing, coverage denial, or eligibility determinations.

> **Status Notice**: The proposed MEPS longitudinal study is currently in the infrastructure and governance setup stage. As of Gate 2, no empirical MEPS data has been downloaded, no models have been trained or evaluated, and no scientific claims or paper results exist.

---

## Project Status

| Milestone / Gate | Description | Status |
|---|---|---|
| **Gate 0** | Read-Only Audit & Flaw Characterization | Completed (Pre/post hash verified, zero writes) |
| **Gate 1** | Baseline Freezing, Git Initialization & Tagging | Completed (Tag `inherited-code-v0.3-baseline-20260828` at `038897e9f751edac6e36445b7706eec5fdb15988`) |
| **Gate 2** | Governance Documentation & Additive Directory Skeleton | Completed (Current gate) |
| **Gate 3+** | MEPS Data Ingestion, Leakage-Free Pipeline & Empirical Evaluation | Pending Gated Execution |

---

## Repository Structure

```text
code_v_0_3/
├── README.md                          # Repository overview and guide (this file)
├── AGENTS.md                          # Supervisor & agent routing instructions
├── GEMINI.md                          # Worker-specific execution guidelines
├── .gitignore                         # Git exclusion rules
│
├── docs/                              # Project governance and reports
│   ├── AI_EXECUTION_PROTOCOL.md       # Canonical AI execution & safety protocol
│   ├── decisions/                     # Architecture & Governance Decision Records
│   │   └── 0001-inherited-baseline.md # Decision record freezing inherited baseline
│   └── reports/                       # Gate completion audit reports
│       ├── GATE_0_READ_ONLY_AUDIT.md  # Gate 0 read-only audit report
│       └── GATE_1_BASELINE.md         # Gate 1 baseline freezing report
│
├── configs/                           # Experiment and pipeline configuration files
├── src/                               # Modular research source code
│   └── fairbias/                      # Core fairness and modeling library
├── tests/                             # Unit, integration, and leakage test suites
├── scripts/                           # Reproducible pipeline execution scripts
│
├── data/                              # Data directories (ignored by git except placeholders)
│   ├── raw/                           # Raw survey microdata (read-only, checksum-verified)
│   ├── interim/                       # Intermediate transformed survey cohorts
│   └── processed/                     # Leakage-free train/validation/test feature matrices
│
├── runs/                              # Execution manifests and run logs (git-ignored)
├── outputs/                           # Summary tables, metrics, and visualization artifacts
├── artifacts/                         # Serialized models and evaluation bundles
├── archive/                           # Archived legacy materials and auxiliary scripts
│
└── [Inherited Baseline Files]        # Immutable historical root files (Commit 038897e)
    ├── app.py                         # Historical Flask UI
    ├── classifiers.py                 # Historical baseline classifiers
    ├── config.py                      # Historical configuration
    ├── data_COMPAS.csv                # Historical COMPAS benchmark data
    ├── data_Credit_Card.csv           # Historical Credit Card benchmark data
    ├── eval.py                        # Historical evaluation routines
    ├── main.py                        # Historical entry point
    ├── module_AE.py                   # Historical Accuracy Enhancement module
    ├── module_BM.py                   # Historical Bias Mitigation module
    ├── module_load.py                 # Historical data loader
    ├── module_transform.py            # Historical data transformations
    ├── requirements.txt               # Historical dependencies
    ├── start.sh                       # Historical start script
    └── results/                       # Historical run outputs
        └── all_results.json           # Historical execution record
```

---

## Safe Start & AI Protocol

All development in this repository is governed by the single canonical policy:
👉 [`docs/AI_EXECUTION_PROTOCOL.md`](docs/AI_EXECUTION_PROTOCOL.md)

Key principles:
1. **Inherited Material is Immutable**: Do not alter, rename, reformat, or delete any of the 14 inherited root files or `.gitignore`.
2. **Additive Development**: All new implementation code and data reside in additive subdirectories (`src/`, `configs/`, `docs/`, `scripts/`, `tests/`) on the `research/meps-hc252-longitudinal` branch (root-level build/governance metadata such as `pyproject.toml` is permitted only when explicitly authorized by a gate).
3. **No Unsupervised Actions**: The worker agent operates strictly within the active gate specification and submits structured reports for Codex supervisor review. Commits are made only upon supervisor authorization.
4. **Data Integrity & Privacy**: Microdata from MEPS must never be committed, exposed via raw row prints, or linked to external identifiable sources.
5. **Empirical Claim Boundaries**: No MEPS empirical claims are allowed until produced by the validated, leakage-free, survey-aware pipeline across subsequent gates. Such verified empirical outputs are distinguished from subsequent external peer review.
