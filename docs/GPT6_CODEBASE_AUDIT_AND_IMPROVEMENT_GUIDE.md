# GPT-6 Codebase Audit & Improvement Guide: FairBias Healthcare Equity Benchmark

## Executive Overview

This document serves as a structured audit framework and strategic guide for **GPT-6** (and senior academic/algorithmic reviewers) to evaluate the codebase in the `research/nhis-fairbias` branch of [`fairbias-health-equity`](https://github.com/icarus3344/fairbias-health-equity/tree/research/nhis-fairbias).

The repository implements the **FairBias** algorithmic framework (Tang et al., 2024) and extends it to national healthcare survey data from the **CDC National Health Interview Survey (NHIS 2022–2024)**, establishing rigorous complex survey weighting, longitudinal temporal robustness, and accuracy-enhancement optimization.

---

## 1. Repository Architecture & File Roadmap

The repository separates the frozen historical baseline from modern, modular research pipelines:

```text
fairbias-health-equity/ (branch: research/nhis-fairbias)
├── src/
│   ├── fairbias/                     # Core Algorithmic Engine (Domain-Agnostic)
│   │   ├── mitigation.py             # Geometric manifold optimization & debiasing
│   │   ├── bias_metric.py            # Demographic Parity, Equalized Odds & Survey Weights
│   │   ├── enhancement.py            # Accuracy Enhancement (AE) candidate search & state
│   │   ├── enhancement_contracts.py  # Strict invariant contracts for AE & BM steps
│   │   ├── enhancement_state.py      # Immutable snapshotting of optimization states
│   │   ├── models.py                 # Estimator interfaces (LogisticRegression, LightGBM, MLP)
│   │   ├── pipeline.py               # End-to-end FairBias pipeline orchestrator
│   │   └── transform.py              # Feature transformation & scaling
│   │
│   └── nhis_fairbias/                # CDC NHIS Healthcare Survey Benchmark
│       ├── survey.py                 # Complex survey design (WTFA_A, STRAT_P, PSU_P)
│       ├── schema.py                 # Harmonized survey variable definitions
│       ├── harmonize.py              # Multi-year variable alignment (2022-2024)
│       ├── features.py               # Categorical encoding & pre-split pipelines
│       ├── adapter.py                # Bridge between NHIS DataFrames & FairBias engine
│       ├── evaluation.py             # Survey-weighted AUROC, AUPRC, and calibration
│       ├── d6_temporal_runner.py     # Out-of-time longitudinal evaluation (2022 -> 2023/2024)
│       ├── d7_stepwise_replay.py     # Attribution tracking across debiasing iterations
│       └── d8_enhancement_runner.py  # Accuracy Enhancement runner (C1-C4 conditions)
│
├── scripts/
│   ├── run_nhis_d6_temporal.py       # D6 temporal robustness entrypoint
│   ├── run_nhis_d7_stepwise_replay.py# D7 mechanism replay entrypoint
│   ├── run_nhis_enhancement_study.py # D8 enhancement runner CLI
│   └── run_nhis_d8_r4_substantive.py # Gate D8-R4 primary substantive runner
│
├── tests/
│   ├── test_nhis_d8_synthetic_contracts.py # 60 rigorous contract & invariant tests
│   ├── test_fairbias_enhancement.py        # Core AE unit & integration tests
│   ├── test_fairbias_enhancement_contracts.py
│   └── test_survey_weighted_geometry.py    # Weighted distance & loss tests
│
└── docs/
    ├── AI_EXECUTION_PROTOCOL.md      # Gated execution & supervisory governance rules
    └── reports/
        ├── NHIS_D8_R4_PRIMARY_SUBSTANTIVE_EXECUTION_20260909T080325Z.md
        ├── NHIS_D8_R4B_FINAL_EVIDENCE_RECONCILIATION_20260909T092500Z_d8r4b_reconciliation.md
        └── NHIS_D8_R4B_SUPERVISOR_ERRATUM.md
```

---

## 2. Core Scientific & Algorithmic Principles

### 2.1 The Four Benchmark Conditions (Gate D8)
1. **C1 (Baseline)**: Unmitigated classifier trained on original feature space $\mathbf{X}$.
2. **C2 (Canonical FairBias)**: Frozen FairBias manifold projection $\mathbf{Z} = \mathbf{X}\mathbf{W}$ optimized for demographic parity ($\Delta\Phi \le \epsilon$).
3. **C3 (Posthoc Enhancement)**: Accuracy Enhancement (AE) candidate search applied strictly *after* reaching the C2 terminal state, constrained never to violate $\epsilon$.
4. **C4 (Joint Enhancement)**: Interleaved Bias Mitigation (BM) and Accuracy Enhancement (AE) searching the joint Pareto frontier.

### 2.2 Four Protected Attribute Arms
- `D6_ARM_001`: **SEX** (`SEX_A`, Male vs. Female)
- `D6_ARM_002`: **ETHNICITY** (`HISP_A`, Hispanic vs. Non-Hispanic)
- `D6_ARM_003`: **DISABILITY-FULL** (`DISAB_full`, Adults with disabilities across full population)
- `D6_ARM_004`: **DISABILITY-EXCLUDE** (`DISAB_exclude`, Adults with disabilities in restricted domain)

### 2.3 The Critical Scientific Discovery
- **C3 Posthoc is Path-Robust**: Across both paper-faithful and engineering geometries, C3 converges to identical terminal states and identical test AUROC across all 4 arms ($\Delta\text{AUROC} = 0.0000$).
- **C4 Joint is Strongly Geometry/Path-Sensitive**: Interleaving BM and AE causes significant trajectory divergence between the paper-faithful stress-elbow geometry and exploratory engineering step rules (especially on `DISAB_exclude` where engineering yielded AUROC 0.7520 in an infeasible regime, whereas paper-faithful achieved AUROC 0.6866 while strictly satisfying the frozen D6 fairness criterion).

---

## 3. What GPT-6 Should Inspect (Audit Checklist)

### 3.1 Algorithmic & Mathematical Correctness
- [ ] **Loss Formulations & Manifold Projection** (`src/fairbias/mitigation.py`):
  - Check whether the geometric distance minimization and cross-entropy classification objectives are properly combined.
  - Verify that the gradient computations or optimization routines adhere to Tang et al. (2024).
- [ ] **Interleaved BM + AE Search** (`src/fairbias/enhancement.py`, `enhancement_contracts.py`):
  - Verify state immutability, backtrack behavior on constraint violation, and termination conditions.
  - Verify that $\epsilon$-fairness checks use the exact frozen D6 thresholds.
- [ ] **Stress-Elbow vs. Engineering Step Rules**:
  - Inspect the stopping criterion: does the stress-elbow criterion prevent over-smoothing of group manifolds?

### 3.2 Complex Survey Statistics & Epidemiological Rigor
- [ ] **Survey Weights Integration** (`src/nhis_fairbias/survey.py`, `src/fairbias/bias_metric.py`):
  - Are sampling weights (`WTFA_A`) correctly applied to weighted confusion matrices, weighted demographic parity ($\Delta\Phi$), and weighted AUROC/AUPRC?
  - Are unweighted calculations kept separate as sensitivity checks?
- [ ] **Variance Estimation & Clustering**:
  - Does the evaluation report point estimates with proper acknowledgment of stratum (`PSTRAT`) and primary sampling unit (`PPSU`) design?
  - Are domain estimates (subgroup analysis like `DISAB_exclude`) free from unconditional filtering bias?

### 3.3 Data Hygiene & Leakage Prevention
- [ ] **Split Isolation**:
  - Confirm that preprocessors, scalers, imputers, and manifold transformations are fit strictly on training partitions (`train_idx`).
  - Verify that the test split is evaluated only once at the terminal state and never used for candidate selection, hyperparameter tuning, or early stopping.
- [ ] **Temporal Separation (D6 Out-of-Time Protocol)**:
  - Check temporal splits: 2022 (train/val) $\to$ 2023/2024 (test). Is there any forward-looking leakage across survey years?

### 3.4 Software Architecture & Defensive Engineering
- [ ] **Modularity & Decoupling**:
  - Is `fairbias/` strictly domain-agnostic?
  - Does `nhis_fairbias/adapter.py` cleanly map tabular survey structures into algorithm tensors without leaking survey logic into the optimizer?
- [ ] **Error Handling & Contract Testing**:
  - Review `tests/test_nhis_d8_synthetic_contracts.py`: are the 60 contract tests comprehensive, covering edge cases, empty candidates, extreme weights, and constraint boundaries?
  - Are post-constructor mutations properly blocked (e.g., `r4_primary_authorized` property setter raises `AttributeError`)?

### 3.5 Provenance, Governance & Errata Reconciliation
- [ ] **Review Governance Errata** (`docs/reports/NHIS_D8_R4B_SUPERVISOR_ERRATUM.md`):
  - Verify the reclassification of `d8_enhancement_study_results.json` as a legacy pre-existing file.
  - Verify that the narrative for Arm 4 reflects descriptive path/geometry sensitivity rather than unsupported causal claims.

---

## 4. Potential Directions for Improvement

### 4.1 Methodological & Scientific Extensions
1. **Mechanistic Ablation of C4 Path Sensitivity**:
   - Conduct a systematic parametric interpolation between the exploratory engineering step sizes and the paper-faithful stress-elbow geometry.
   - Map the continuous Pareto surface to mathematically explain why interleaved BM+AE trajectories diverge while Posthoc AE remains stationary.
2. **Multi-Metric Fairness Constraints**:
   - Extend the mitigation engine from Demographic Parity ($\Delta\Phi$) to Equalized Odds (simultaneous TPR/FPR parity) under complex survey weights.
3. **Design-Adjusted Inference & Confidence Bands**:
   - Incorporate Balanced Repeated Replication (BRR) or Jackknife repeated weights to compute 95% confidence intervals on the test $\Delta\text{AUROC}$ and $\Delta\Phi$.

### 4.2 Algorithmic Optimization & Computational Scaling
1. **Vectorized Candidate Batching**:
   - Currently, candidate transformations in AE are evaluated sequentially. Vectorizing or parallelizing candidate score evaluations would accelerate the search by an order of magnitude.
2. **Continuous Relaxation of Feature Transforms**:
   - Replace greedy discrete candidate search with a differentiable, gradient-guided feature transformation parameterization.

### 4.3 Engineering & Usability
1. **Modern Packaging (`pyproject.toml`)**:
   - Standardize project packaging with `pyproject.toml` and clean CLI entry points via `typer` or `click`.
2. **Interactive Visualization Artifacts**:
   - Provide an automated interactive dashboard (e.g., Plotly / HTML) visualizing the 2D optimization paths (Fairness vs. Accuracy) for each arm.

---

## 5. Ready-to-Use Prompts for Guiding GPT-6

The following prompts can be copied directly into conversations with GPT-6.

### Prompt A: Master Comprehensive Audit (Code & Methodology)
```markdown
You are a Principal AI Research Scientist and Senior Software Architect specializing in Algorithmic Fairness, Machine Learning in Healthcare, and Complex Survey Statistics.

I have open-sourced our research codebase on GitHub:
Repository: https://github.com/icarus3344/fairbias-health-equity
Branch: research/nhis-fairbias
Commit: 93fb10c15f29e523171ea89ae33a67d6d147bfea

Please perform a rigorous, end-to-end peer review of this repository, covering:
1. **Algorithmic Fidelity**: Audit `src/fairbias/mitigation.py` and `src/fairbias/enhancement.py` against the Tang et al. (2024) FairBias specification. Are the geometric manifold projections, stress-elbow criteria, and BM/AE steps mathematically sound?
2. **Survey Methodology**: Audit `src/nhis_fairbias/survey.py` and evaluation routines. Are CDC NHIS sampling weights (WTFA_A), strata, and clusters correctly incorporated into fairness disparities and utility metrics without domain-estimation bias?
3. **Data Leakage & Split Isolation**: Audit preprocessing and evaluation pipelines. Is the zero-leakage contract strictly enforced across train/val/test and longitudinal temporal splits (D6)?
4. **Code Architecture & Test Depth**: Audit the 60 synthetic contract tests in `tests/test_nhis_d8_synthetic_contracts.py`. What edge cases, numerical instabilities, or failure modes are not yet covered?
5. **Scientific Narrative**: Review `docs/reports/NHIS_D8_R4B_SUPERVISOR_ERRATUM.md`. Is the distinction between C3 (path-robust) and C4 (geometry-sensitive) scientifically defensible?

Provide your audit with clear severity ratings (P0: Critical Flaw, P1: Methodological Risk, P2: Code Quality / Optimization, P3: Enhancement Suggestion), cite specific file paths and line numbers, and provide actionable recommendations.
```

### Prompt B: Algorithmic & Mathematical Deep-Dive
```markdown
Focus specifically on the mathematical and optimization mechanics of our FairBias accuracy enhancement framework in `src/fairbias/` and `src/nhis_fairbias/d8_enhancement_runner.py` on the `research/nhis-fairbias` branch:
1. **Interleaved BM+AE vs. Sequential Posthoc**: Analyze why Posthoc Enhancement (C3) is mathematically path-robust across geometries (yielding identical test AUROC across all 4 NHIS arms), whereas Joint Enhancement (C4) exhibits extreme geometry/path sensitivity. Is this an inherent mathematical property of the projection manifold, or an artifact of candidate ordering?
2. **Constraint Enforcement**: Analyze how the frozen epsilon threshold ($\Delta\Phi \le \epsilon$) is enforced during candidate evaluation. Is there any risk of gradient vanishing, local entrapment, or metric gaming?
3. **Survey-Weighted Distance**: Review how sample weights are weighted in pairwise distance calculations. Are there numerical stability concerns with extreme survey weights?

Provide theoretical analysis, mathematical formulation checks, and suggestions for proving or ablaing the path-sensitivity phenomenon.
```

### Prompt C: Academic Paper & Methodological Roadmap
```markdown
We are preparing an academic manuscript for submission to a top-tier ML/Health Informatics venue (e.g., NeurIPS, ICML, FAccT, JAMIA) based on the findings in `docs/reports/NHIS_D8_R4_PRIMARY_SUBSTANTIVE_EXECUTION_20260909T080325Z.md` and `docs/reports/NHIS_D8_R4B_SUPERVISOR_ERRATUM.md`.

Our primary empirical findings on CDC NHIS (2022–2024):
- C1 Baseline $\to$ C2 Canonical FairBias reduces demographic disparity significantly, with expected utility tradeoff.
- C3 Posthoc Enhancement achieves modest accuracy recovery while remaining 100% path-robust across optimization geometries ($\Delta\text{AUROC} = 0.0000$).
- C4 Joint Interleaving demonstrates high geometry sensitivity, recovering utility on some arms while diverging on others.

Please advise on:
1. **Framing & Narrative**: How should we position the "Path-Robustness of Posthoc vs. Path-Sensitivity of Joint" to make a compelling theoretical and empirical contribution?
2. **Anticipating Reviewer Criticisms**: What potential weaknesses or rebuttal attacks will reviewers raise regarding:
   - The divergence of Arm 4 (DISAB-exclude)?
   - Evaluation on tabular survey data vs. benchmark datasets (COMPAS/Adult)?
   - Fairness definition selection ($\Delta\Phi$ vs. Equalized Odds)?
3. **Key Ablation Experiments**: What minimal, high-impact synthetic or offline ablation experiments should we run to make the submission watertight?
```
