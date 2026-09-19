# RESULT STATUS ADDENDUM: Reclassification and Claim Retractions for Sequential Full Benchmark 20260914_103657Z

- **Date**: 2026-09-14
- **Branch**: `research/nhis-fairbias`
- **Target Run ID**: `sequential_full_benchmark_20260914_103657Z`
- **Target Artifact**: `runs/sequential_full_benchmark_20260914_103657Z/integrated_full_benchmark_summary.json`
- **Status Classification**: **EXPLORATORY EXECUTION RECORD** (`REJECT_FOR_CONFIRMATORY_USE`)
- **Governing Protocol**: `docs/AI_EXECUTION_PROTOCOL.md` (Sections 1, 3, 7, 8)
- **Supervisor Audit**: `docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914.md`

---

## 1. Purpose and Formal Reclassification

This document provides a formal, binding addendum to the benchmark execution record `sequential_full_benchmark_20260914_103657Z`.

Following the independent Codex supervisory audit (`FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914.md`), the execution outputs and summary recorded under `runs/sequential_full_benchmark_20260914_103657Z/` are formally **reclassified as an exploratory execution record**. They **cannot and do not constitute confirmatory statistical evidence** of algorithm performance, fairness improvement, or methodological superiority of FairBias over baseline or comparator algorithms.

In accordance with Section 4 and Section 6 of `docs/AI_EXECUTION_PROTOCOL.md`, raw artifacts in `runs/` are permanent historical records and are **not deleted or overwritten**. All corrections, revocations, and boundary delimitations are established strictly through this addendum.

---

## 2. Explicit Claim Retractions

The following specific claims generated during or following the execution of `sequential_full_benchmark_20260914_103657Z` are hereby **fully retracted**:

### 2.1. Retraction of "Global Pareto Optimal" and "Pareto Dominance"

- **Retracted Statement**: Statements asserting that FairBias achieves "global Pareto optimality" across accuracy and fairness, or dominates comparator methods on the balanced accuracy (BA) vs. equalized odds (EO) frontier.
- **Empirical Fact from Audit**: In the existing execution summary (`integrated_full_benchmark_summary.json`), **Reweighing (RW) point-dominates FairBias on both BA and EO in all four experimental arms**:
  - **Arm 001 (SEX)**: RW achieves higher BA (0.54820 vs. 0.52335) and lower EO gap (0.00473 vs. 0.01530).
  - **Arm 002 (HISP 7 groups)**: RW achieves higher BA (0.54605 vs. 0.53208) and lower EO gap (0.09445 vs. 0.09498).
  - **Arm 003 (DISAB include)**: RW achieves higher BA (0.54618 vs. 0.54386) and lower EO gap (0.00654 vs. 0.02666).
  - **Arm 004 (DISAB exclude)**: RW achieves higher BA (0.53939 vs. 0.52954) and lower EO gap (0.00436 vs. 0.02425).
- **Directional Changes vs. Baseline**:
  - FairBias balanced accuracy decreased relative to Unmitigated in all 4 arms: Arm 001 ($\Delta\text{BA} = -0.0247$), Arm 002 ($\Delta\text{BA} = -0.0159$), Arm 003 ($\Delta\text{BA} = -0.0043$), Arm 004 ($\Delta\text{BA} = -0.0106$).
  - FairBias equalized odds gap worsened (increased) relative to Unmitigated in Arm 001 (+375.0%) and Arm 004 (+79.8%).
- **Rectification**: All claims of Pareto dominance or unmitigated superiority are retracted. Reporting must present paired differences with explicit effect directions and two-sided uncertainty intervals.

### 2.2. Retraction of "High-Fidelity Tang et al. (2024) BM Learning"

- **Retracted Statement**: Claims that the benchmark executed the faithful FairBias-BM manifold geometry optimization on Partition F.
- **Empirical Fact from Audit**: `src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py` (lines 124–149) resolved transformation rules by reading pre-computed D6 release dictionaries (`frozen_changed_dict.json`) or hardcoded source constants. These historical dictionaries were trained on full 2022 cohorts ($N=27,451$), which conflated Partition F ($N=21,871$) and Partition C ($N=5,580$). The dynamic fallback (lines 150–179) was an unverified heuristic of group mean differences on continuous columns, omitting MDS, stress, manifold geometry, and author power sequences. The synthetic array branch (lines 214–220, 240) fit and predicted directly on untransformed raw $X$.
- **Rectification**: The benchmark run did not execute FairBias-BM learning. The rule-loading branch is reclassified as a legacy transfer artifact (`FROZEN_D6_TRANSFER`), not freshly learned FairBias.

### 2.3. Retraction of "Independent Blind Test" on 2024 Data

- **Retracted Statement**: Claims that the evaluation on 2024 NHIS data (Set T) was an independent, blind lockbox evaluation.
- **Empirical Fact from Audit**:
  1. The 2024 dataset had been previously inspected and processed across earlier research milestones (D4, D5, D8).
  2. The worker execution transcript (`worker_submission.txt`, lines 50–539) documents iterative pipeline modifications and smoke tests directly observing Set T metrics before final runs.
  3. `scripts/run_sequential_full_benchmark.py` (lines 230–249) generated Set T predictions inside the training loop prior to Set S candidate selection (line 271).
- **Rectification**: Set T evaluation is designated strictly as **retrospective cross-year exploration**. No claims of prospective blind validation are permitted.

### 2.4. Retraction of Legal and Clinical Compliance Assertions

- **Retracted Statement**: Statements asserting that Fairlearn's `ThresholdOptimizer` is "legally non-compliant / illegal disparate treatment" or has "zero clinical compliance", and claims that FairBias provides guaranteed legal or clinical compliance.
- **Empirical Fact from Audit**: `ThresholdOptimizer` requires the protected attribute $A$ at inference time (`sensitive_features`), which is an architectural dependency characteristic. No legal jurisdiction evidence, statutory analysis, or healthcare regulatory review was conducted.
- **Rectification**: The requirement of $A$ at deployment is described neutrally as an operational and deployment constraint. All legal and regulatory conclusions are retracted.

### 2.5. Retraction of "Full LFR Representation Learning"

- **Retracted Statement**: Statements asserting that Zemel et al. (2013) LFR failed algorithmically or that all methods were trained under equal budgets.
- **Empirical Fact from Audit**: `src/nhis_fairbias/benchmark/adapters/adapter_lfr.py` (lines 108–112) subsampled only 2,000 rows when $N_F > 2,000$ to learn its latent representation, with truncated optimizer budgets (`maxiter=50`, `maxfun=100`), while downstream LR used all 21,869 rows.
- **Rectification**: LFR results in this run reflect a truncated sub-sample configuration. Retract claims of general algorithmic inferiority.

### 2.6. Retraction of NMI Gate and Utility Protection Claims

- **Retracted Statement**: Claims that an information-theoretic NMI gate protected predictive utility during the benchmark execution.
- **Empirical Fact from Audit**: The main benchmark path loaded static dictionaries and did not execute the NMI gating mechanism during model fitting.
- **Rectification**: All claims regarding active NMI gate utility preservation in this run are retracted.

### 2.7. Clarification of Total Population vs. Test Set Sample Sizes

- **Retracted Statement**: Reporting and table headers presenting $N=89,091$ as the benchmark evaluation test cohort.
- **Empirical Fact from Audit**: 89,091 (or 89,077 / 89,087 across arms) is the three-year pooled respondent count across Partition F (2022, ~21,871), Partition C (2022, ~5,580), Partition S (2023, ~29,280), and Partition T (2024, ~32,350). The actual Set T test evaluation sample size is **32,350 to 32,355**.
- **Rectification**: All future reporting must report $N_F, N_C, N_S, N_T$ separately.

---

## 3. Preservation and Scope of Historical Run

The JSON file `runs/sequential_full_benchmark_20260914_103657Z/integrated_full_benchmark_summary.json` remains in the repository as an immutable record of historical pipeline execution. It serves as:
1. Proof of execution for adapter interface mechanics;
2. Empirical evidence demonstrating the necessity of the architectural repairs mandated in `REPAIR_IMPLEMENTATION_SPEC.md`;
3. A baseline for verifying that future confirmatory benchmark runs do not repeat these methodological and statistical shortcomings.
