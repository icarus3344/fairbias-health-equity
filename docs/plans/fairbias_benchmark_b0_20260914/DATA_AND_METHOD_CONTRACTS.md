# Data and Method Contracts Specification (Benchmark V1)

**Document ID**: `docs/plans/fairbias_benchmark_b0_20260914/DATA_AND_METHOD_CONTRACTS.md`  
**Execution Timestamp**: `2026-09-14T13:36:15+08:00`  
**Active Gate**: `FAIRBIAS-BENCHMARK-B0`  
**Authority Reference**: [Master Plan](file:///Users/lkc/Downloads/code_v_0_3/docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md) | [Consolidated Audit](file:///Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_CONSOLIDATED_AUDIT_AND_APPLICATION_PLAN_20260913.md) | [features.json](file:///Users/lkc/Downloads/code_v_0_3/configs/nhis/features.json)  
**Supervisor**: Codex Supervisor  
**Implementation Worker**: Gemini Implementation Worker  

---

## 1. Experimental Arms Specification

The benchmark evaluates healthcare barrier identification models across four explicitly defined experimental arms:

| Arm ID | Protected Dimension ($A$) | Number of Groups & Categories | Feature Set Name | Feature Count | Scientific Purpose & Boundary |
|---|---|---|---|---|---|
| **Arm 001** | `SEX_A` | 2 groups:<br>• `1` = Male<br>• `2` = Female | `primary_core` | 21 semantic features | Primary binary demographic equity benchmark. Baseline axis for comparison against historical literature. |
| **Arm 002** | `HISPALLP_A` | 7 groups:<br>• `01` = Hispanic<br>• `02` = NH White only<br>• `03` = NH Black only<br>• `04` = NH Asian only<br>• `05` = NH AIAN only<br>• `06` = NH AIAN and any other<br>• `07` = NH Other and multiple | `primary_core` | 21 semantic features | Multi-group fairness evaluation. Tests algorithmic capability under severe sample imbalance and small-cell minority estimation. |
| **Arm 003** | `DISAB3_A` | 2 groups:<br>• `1` = With disability<br>• `2` = Without disability | `primary_core` (full) | 21 semantic features | Health disparity benchmark including 6 explicit disability component proxy variables. |
| **Arm 004** | `DISAB3_A` | 2 groups:<br>• `1` = With disability<br>• `2` = Without disability | `primary_core` (exclude components) | 15 semantic features | Feature-ablation sensitivity arm. Evaluates performance when 6 disability component proxies are removed from feature set. |

### Crucial Methodological Constraints for Arms 003 & 004
1. **Identical Population & Partitions**: Arm 004 is **NOT** a restricted subpopulation analysis of persons with disabilities. 
2. **Strict Cohort Parity**: Arm 003 and Arm 004 must have identical respondent rows, identical temporal partitions (F, C, S, T), identical survey weights (`WTFA_A`), and identical design strata (`PSTRAT`, `PPSU`).
3. **Exact Feature Difference**: The only difference between Arm 003 and Arm 004 is the exclusion of the 6 constituent disability component features (`visiondf_a`, `hearingdf_a`, `diff_a`, `comdiff_a`, `uppslfcr_a`, `cogmemdff_a`).

---

## 2. Variables, Missingness & Encoding Contracts

### 2.1 Outcomes
- **Primary Outcome**: `MEDDL12M_A` ("During the past 12 months, delayed medical care because of cost").
  - Substantive Coding: `1` (Yes) $\to 1$; `2` (No) $\to 0$.
  - Invalidation: Non-substantive codes (`7` Refused, `8` Not Ascertained, `9` Don't Know) are excluded from the analysis cohort during initial eligibility filtering.
- **Secondary Outcome (Reserved Extension)**: `MEDNG12M_A` ("During the past 12 months, needed medical care but did not get it due to cost").
  - Substantive Coding: `1` $\to 1$; `2` $\to 0$.

### 2.2 Predictor Semantic Features (21 Primary Core)
From [`configs/nhis/features.json`](file:///Users/lkc/Downloads/code_v_0_3/configs/nhis/features.json):
1. **Numerical Features (3 variables)**:
   - `agep_a`: Age of sample adult (18–85; top-coded at 85).
   - `pcnt18uptc`: Number of persons aged 18+ in family (top-coded at 3).
   - `pcntlt18tc`: Number of children aged <18 in family (top-coded at 3).
2. **Categorical Features (18 variables)**:
   - `educp_a`: Educational attainment (ordinal/nominal).
   - `region`: Census region (nominal: 1 Northeast, 2 Midwest, 3 South, 4 West).
   - `ratcat_a`: Ratio of family income to poverty threshold (ordinal).
   - `empwrklsw1_a`: Worked for pay last week (binary).
   - `empwrkft1_a`: Full-time work status (binary).
   - `notcov_a`: Health insurance coverage status (binary).
   - `phstat_a`: General health status (ordinal: 1 Excellent to 5 Poor).
   - `hypev_a`: Ever told had hypertension (binary).
   - `chlev_a`: Ever told had high cholesterol (binary).
   - `dibev_a`: Ever told had diabetes (binary).
   - `asev_a`: Ever told had asthma (binary).
   - `visiondf_a`: Difficulty seeing even with glasses (ordinal, disability component).
   - `hearingdf_a`: Difficulty hearing even with hearing aid (ordinal, disability component).
   - `diff_a`: Difficulty walking or climbing steps (ordinal, disability component).
   - `comdiff_a`: Difficulty communicating (ordinal, disability component).
   - `uppslfcr_a`: Difficulty with self-care (ordinal, disability component).
   - `cogmemdff_a`: Difficulty remembering or concentrating (ordinal, disability component).
   - `usualpl_a`: Usual place for healthcare (nominal).

### 2.3 Complex Survey & Administrative Design Variables
The following variables are survey design and record-tracking metadata. Under no circumstances may they enter feature matrices $X$:
- `WTFA_A`: Final annual sample adult sampling weight.
- `PSTRAT`: Variance estimation stratum identifier.
- `PPSU`: Variance estimation pseudo-primary sampling unit identifier.
- `URBRRL`: 2013 NCHS urban-rural classification code.
- `SRVY_YR` / `YEAR`: Calendar survey year.
- `HHX`: Household identifier.
- `RECTYPE`: Record type indicator.

### 2.4 Missingness & Leakage Prevention Rules
1. **Outcome ($Y$) & Protected Attribute ($A$) Missingness**: Any respondent record where $Y$ or the designated arm's $A$ is missing, non-substantive, or invalid is excluded from that arm's cohort prior to partitioning.
2. **Feature ($X$) Missingness**:
   - Continuous numerical features: Imputed using median values computed **strictly on Partition F**. Imputation parameters are frozen and applied inductively to C, S, and T.
   - Categorical features: Missing substantive values are mapped to an explicit `'MISSING'` category level.
   - Unseen categories in S or T: Mapped to an explicit `'UNKNOWN'` category token. They must never expand the one-hot vocabulary or trigger vector shape mismatches.

---

## 3. Temporal Partitioning: F / C / S / T Architecture

The benchmark institutes a strict 4-partition lifecycle designed to eliminate cross-round and cross-temporal leakage:

```
[ 2022 Primary Cohort (100% Eligible) ]
   ├── Fitting Set (F): ~80% PSUs  ──> Fit Semantics, Scalers, FairBias BM, RW/LFR/EG models
   └── Calibration Set (C): ~20% PSUs ──> Calibrate Thresholds, Fit TO Postprocessor, AE Inner Utility

[ 2023 Full Cohort (100% Eligible) ]
   └── Selection Set (S) ─────────────> Evaluate Pre-registered Grids, Select Frozen Configurations

[ 2024 Full Cohort (100% Eligible) ]
   └── Frozen Evaluation Set (T) ─────> Final Retrospective Frozen Inference (Read-only, Locked until B6)
```

### 3.1 Partition Definitions & Boundaries
| Partition | Data Year & Cohort Scope | Permitted Operational Roles | Prohibited Usages |
|---|---|---|---|
| **Set F** (Fitting) | Year 2022, ~80% PSUs | • Fit semantic transformers, scalers, one-hot encoders.<br>• Fit FairBias BM greedy transformations.<br>• Fit base classifiers (LR, GBDT), RW weights, LFR representations, EG reductions.<br>• Compute reference $\epsilon_{\text{ref}}$. | • Never access C, S, or T data.<br>• Never use test labels. |
| **Set C** (Calibration) | Year 2022, ~20% PSUs | • Fit global decision thresholds $t$ for probability models.<br>• Fit ThresholdOptimizer (TO) postprocessors with `prefit=True`.<br>• Provide internal validation feedback for AE candidate acceptance. | • Never fit basic scalers or encoders.<br>• Never claim Set C performance as unbiased generalization. |
| **Set S** (Selection) | Year 2023, 100% Eligible | • Evaluate all pre-registered hyperparameter configurations across seeds.<br>• Execute frozen selection criteria to pick winning configurations per operating point. | • Never fit model parameters, encoders, or postprocessors.<br>• Never provide feedback to restart iterative searches. |
| **Set T** (Evaluation) | Year 2024, 100% Eligible | • Retrospective out-of-year frozen inference and complex survey-weighted evaluation.<br>• Compute paired differences ($\Delta\text{BA}, \Delta\text{EO}$) and design-based bootstrap CIs. | • Completely locked during Gates B0–B5b.<br>• Zero fitting, threshold adjustment, or hyperparameter selection allowed. |

### 3.2 PSU-Level Partitioning Algorithm
1. The 2022 master design table contains sorted, unique `(PSTRAT, PPSU)` clusters.
2. Partition allocation uses pseudo-random assignment:
   $$\text{PSU} \to C \quad \text{if } \text{Hash}(\text{PSTRAT}, \text{PPSU}, \text{seed}=20260913) \pmod{100} < 20 \quad \text{else } F$$
3. Entire PSUs are assigned atomically: all respondents sharing the same `(PSTRAT, PPSU)` reside entirely in F or entirely in C. No PSU is ever split across partitions.
4. The master F/C partition mapping is generated once and shared across all 4 arms.

---

## 4. Stable Record Identity Contract

To guarantee cross-partition isolation and prevent entity leakage, every respondent record possesses a strictly normalized, immutable identity key:

### 4.1 Key Specification
$$\text{record\_key} = \text{source} + \text{":"} + \text{str}(\text{int}(\text{year})) + \text{":"} + \text{id\_norm}$$
- `source`: String identifier of the dataset (e.g. `'nhis'`).
- `year`: Finite integer calendar survey year (`2022`, `2023`, `2024`). Floating-point representations (`2023.0`) or missing values (`NaN`) are strictly rejected.
- `id_norm`: Canonical respondent ID string without trailing or leading whitespace.

### 4.2 Partition Data Structure Schema
Every evaluation partition object returned by benchmark data loaders conforms to the following dictionary contract:
```python
{
    "record_key": np.ndarray[str],        # Unique normalized record identifiers, shape (N,)
    "X_semantic": pd.DataFrame,           # 21 (or 15) raw semantic features, shape (N, P)
    "y": np.ndarray[int],                 # Binary outcome labels in {0, 1}, shape (N,)
    "A": np.ndarray[int],                 # Protected attribute categories, shape (N,)
    "WTFA_A": np.ndarray[float],          # Finite, non-negative sampling weights, shape (N,)
    "PSTRAT": np.ndarray[int],            # Complex survey variance stratum, shape (N,)
    "PPSU": np.ndarray[int],              # Complex survey variance PSU, shape (N,)
    "year": int,                          # Survey year (2022, 2023, or 2024)
    "role": str,                          # 'fitting_F', 'calibration_C', 'selection_S', 'evaluation_T'
    "eligibility_mask": np.ndarray[bool], # Boolean array indicating cohort inclusion, shape (N,)
    "schema_hash": str,                   # SHA-256 hash of feature schema configuration
    "source_hash": str,                   # SHA-256 hash of raw data source
}
```

---

## 5. Three-Tier Feature Representation

To resolve the integer coding defect (M01), Benchmark V1 enforces a three-tier feature architecture:

1. **Tier 1: Semantic Layer**:
   - Preserves original domain variables as defined in `features.json`.
   - Missing continuous values are imputed with medians fit on F.
   - Missing categorical values are assigned the explicit level `'MISSING'`.
2. **Tier 2: FairBias Geometric Layer**:
   - Operates on semantic feature entities rather than one-hot expanded matrices.
   - Computes pairwise divergence, MDS coordinate embeddings, and greedy power/grouping transformations per feature entity.
   - Outputs transformed semantic features alongside an updated feature schema.
3. **Tier 3: Downstream Classifier Prediction Layer**:
   - Continuous features: Scaled via `MinMaxScaler` fitted strictly on F (or transformed F).
   - Categorical features: One-hot encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` fitted strictly on F (or transformed F).
   - LFR continuous representations: Passed directly to downstream classifiers without one-hot expansion.

---

## 6. Method Input/Output & Capability Matrix

The benchmark benchmarks 7 core methods across LR and GBDT backbones. Every adapter must declare its capabilities explicitly:

| Method ID | Literature & Source | Modifies $X$? | Modifies Weights? | Requires $y$ at fit? | Requires $A$ at fit? | Requires $A$ at predict? | Supports Arm 002 (HISP 7-group)? | Supports Survey Training Weights? | Stochastic Fit / Decision? | Output Type Contract | Native Objective | Verification Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **UNMITIGATED** | Standard Classifier | No | No | Yes | No | No | **Supported** | Supported (sensitivity) | Deterministic / Deterministic | Event prob $p$, Hard decision | Log-loss / Deviance | `UNVERIFIED` |
| **FAIRBIAS_BM** | Tang et al. (2024) application-v1 | Yes (Greedy BM) | No | No | Yes | No (verified mapping) | **Supported** (requires B3 test) | Supported (downstream) | Stochastic MDS / Deterministic | Event prob $p$, Hard decision | Min $d_\phi$ geometry | `UNVERIFIED` |
| **REWEIGHING** | Kamiran & Calders (2012) AIF360 | No | Yes (Training sample weights) | Yes | Yes | No | **Supported** (multi-group) | `NOT_SUPPORTED` (unverified mix) | Deterministic / Deterministic | Event prob $p$, Hard decision | Parity reweighted log-loss | `UNVERIFIED` |
| **LFR_RECONSTRUCTED** | Zemel et al. (2013) AIF360 | Yes (Prototype representations) | No | Yes | Yes | Yes (API requirement) | **`NOT_SUPPORTED`** | `NOT_SUPPORTED` | Stochastic / Deterministic | Downstream $p$, Hard decision | Reconstruction + Fair loss | `UNVERIFIED` |
| **EG_DP** | Agarwal et al. (2018) Fairlearn | No | Yes (Reduction oracle costs) | Yes | Yes | No | **Supported** | `NOT_SUPPORTED` | Deterministic / **Stochastic $q$** | Decision prob $q$ | Error rate s.t. DP | `UNVERIFIED` |
| **EG_EO** | Agarwal et al. (2018) Fairlearn | No | Yes (Reduction oracle costs) | Yes | Yes | No | **Supported** | `NOT_SUPPORTED` | Deterministic / **Stochastic $q$** | Decision prob $q$ | Error rate s.t. EO | `UNVERIFIED` |
| **TO_EO** | Hardt et al. (2016) Fairlearn | No | No | Yes | Yes | **Yes** | **Supported** (requires positive support) | `NOT_SUPPORTED` | Deterministic / **Stochastic $q$** | Decision prob $q$, Base $p$ | Balanced Acc s.t. EO | `UNVERIFIED` |

### 6.1 Crucial Adapter Execution Contracts
1. **LFR Label & Score Preservation**:
   - AIF360 `LFR.transform()` modifies the `labels` and `scores` fields of `BinaryLabelDataset`.
   - Contract: The original ground-truth labels $y$ must be maintained in an isolated array and never overwritten by LFR transformed labels.
   - Inference Contract: Dummy placeholder labels passed during inference must be proven to have zero influence on transformed features.
2. **Reweighing (RW) Factor & Index Preservation**:
   - AIF360 `Reweighing.fit_transform()` returns a tuple `(X_trans, sample_weights)`.
   - Contract: `sample_weights` represents fairness balancing factors $W_{\text{fair}}$, completely separate from sampling weights $W_{\text{survey}}$. The record key index must be preserved intact.
3. **Exponentiated Gradient (EG) Oracle Routing & Decision Probability $q$**:
   - Contract: Oracle cost weights generated during reductions must be routed directly to base estimators that support `sample_weight`.
   - Contract: Output decision probability $q_i = \sum_m \beta_m h_m(x_i)$ is computed from the mixture of base classifier **hard decisions**, not continuous probabilities. $q$ represents randomized decision policy and must never be disguised as event risk $p$.
4. **ThresholdOptimizer (TO) Calibration Contract**:
   - Contract: Fitted exclusively on Set C with `prefit=True` and `predict_method='predict_proba'`.
   - Contract: Requires $A$ at prediction time. Unseen or missing $A$ levels must raise explicit exceptions rather than falling back to uncalibrated thresholds.

---

## 7. Explicit NOT_SUPPORTED Boundary Register

Any execution script or configuration requesting the following combinations must fail closed with status `NOT_SUPPORTED`:
1. **LFR on Arm 002 (`HISPALLP_A` 7 groups)**:
   - AIF360 LFR is mathematically and architecturally implemented for binary protected attributes (single privileged group vs single unprivileged group).
   - Splitting or binarizing 7 groups into White vs Non-White is strictly prohibited.
   - Status: `NOT_SUPPORTED`. This condition cell is retained in the benchmark matrix as an explicit structural void with scientific justification.
2. **Complex Survey Weighted Training for EG, TO, and LFR**:
   - Upstream library implementations do not support simultaneous survey design weights and fairness constraint weights.
   - Merging or multiplying survey weights into fairness reduction weights without mathematical formulation is prohibited.
   - Status: `NOT_SUPPORTED`.
