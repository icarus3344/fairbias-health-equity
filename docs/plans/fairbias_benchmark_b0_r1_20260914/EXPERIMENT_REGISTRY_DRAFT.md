# Experiment Registry and Computational Budget Draft (Benchmark V1 - B0-R1 Revision)

**Document ID**: `docs/plans/fairbias_benchmark_b0_r1_20260914/EXPERIMENT_REGISTRY_DRAFT.md`  
**Execution Timestamp**: `2026-09-14T15:55:00+08:00`  
**Active Gate**: `FAIRBIAS-BENCHMARK-B0-R1`  
**Authority Reference**: [Master Plan](file:///Users/lkc/Downloads/code_v_0_3/docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md) | [Supervisor Review of B0](file:///Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_B0_SUPERVISOR_REVIEW_20260914.md)  
**Supervisor**: Codex Supervisor  
**Implementation Worker**: Gemini Implementation Worker  

---

## 1. Experimental Design Overview

The primary scientific objective is to benchmark **FairBias-BM (application-v1)** against established algorithmic fairness paradigms (Reweighing, LFR Reconstructed Features, Exponentiated Gradient DP/EO, ThresholdOptimizer EO, and Unmitigated baselines) in identifying individuals experiencing cost-related healthcare delays (`MEDDL12M_A`) in the National Health Interview Survey (NHIS).

The benchmark rigorously enforces:
- **No Presumed Victory**: FairBias is the primary method under investigation, but experimental procedures, hyperparameter grids, and selection criteria are symmetric and neutral.
- **Pre-registered Hyperparameter Grids**: Grids are locked prior to running on real data; no grid tuning occurs on test set T.
- **Complex Survey Inference**: All primary evaluations and paired contrasts incorporate NHIS complex sampling design variables (`WTFA_A`, `PSTRAT`, `PPSU`) via rescaled PSU bootstrap.

---

## 2. Models, Predictors & Hyperparameter Grids

### 2.1 Base Classifier Backbones
1. **Primary Backbone: Logistic Regression (LR)**
   - Estimator: `sklearn.linear_model.LogisticRegression`
   - Solver: `lbfgs`
   - Maximum Iterations: `1000`
   - Class Weight: `None` (standard unweighted baseline; class reweighting is not implicitly applied)
   - Hyperparameter Grid:
     $$C \in \{0.01, 0.1, 1.0, 10.0\}$$ (4 configurations)
2. **Predictor Robustness Backbone: Gradient Boosted Decision Trees (GBDT)**
   - Estimator: `sklearn.ensemble.GradientBoostingClassifier` (exclusively; no alternative classes permitted per R02)
   - Learning Rate: `0.05`
   - Subsample: `1.0`
   - Hyperparameter Grid:
     $$\text{n\_estimators} \in \{100, 200\} \times \text{max\_depth} \in \{2, 3\}$$ (4 configurations)

### 2.2 Fairness Method Hyperparameter Grids
| Method ID | Hyperparameter Parameters & Grid Values | Configurations per Arm/Backbone | Inner Optimization & Execution Notes |
|---|---|---|---|
| **UNMITIGATED** | Backbone hyperparameters only ($C$ or depth/estimators). | **4** | Pure predictive baseline without fairness constraints. |
| **FAIRBIAS_BM** | Backbone grid $\times$ Epsilon Multiplier:$$\mu_\epsilon \in \{0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 4.00, 8.00\}$$where $\epsilon_{\text{cand}} = \mu_\epsilon \cdot \epsilon_{\text{ref}}$. | **32** | $\epsilon_{\text{ref}}$ is calculated strictly on the semantic representation of Partition F. Greedy search terminates on first feasible candidate or budget exhaustion. |
| **REWEIGHING** | Backbone grid $\times$ Standard Kamiran & Calders multi-group balancing factor:$$W_{\text{fair}}(A=a, Y=y) = \frac{P(A=a)P(Y=y)}{P(A=a, Y=y)}$$ | **4** | Standard reweighing factor has no intensity tuning parameter. Computed strictly on Partition F. |
| **LFR_RECONSTRUCTED** | Backbone grid $\times$ Prototype count $k \in \{5, 10\} \times$ Fair constraint weight $A_z \in \{0.1, 1.0, 10.0, 50.0\}$ with fixed $A_x = 0.01, A_y = 1.0$. Optimization limits: $\text{maxiter}=5000, \text{maxfun}=5000$. | **32** | Reconstructed continuous representations $\hat{X}$ feed downstream backbone classifiers. Applicable only to binary demographic arms (Arms 001, 003, 004). Arm 002 is `NOT_SUPPORTED`. |
| **EG_DP** | Backbone grid $\times$ Moment difference bound:$$\text{difference\_bound} \in \{0.005, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20\}$$Fixed EG parameters: $\text{eps} = 0.01, \text{max\_iter} = 50$. | **32** | Fairlearn Exponentiated Gradient reduction with Demographic Parity constraint. Outputs randomized mixture policy $q$. |
| **EG_EO** | Backbone grid $\times$ Moment difference bound:$$\text{difference\_bound} \in \{0.005, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20\}$$Fixed EG parameters: $\text{eps} = 0.01, \text{max\_iter} = 50$. | **32** | Fairlearn Exponentiated Gradient reduction with Equalized Odds constraint. Outputs randomized mixture policy $q$. |
| **TO_EO** | Backbone grid $\times$ Postprocessing on Set C:$$\text{objective} = \text{'balanced\_accuracy\_score'}, \quad \text{grid\_size} = 1000, \quad \text{flip} = \text{False}, \quad \text{prefit} = \text{True}$$ | **4** | Hardt et al. ThresholdOptimizer fitted exclusively on Set C using base model probabilities from Set F. Outputs randomized policy $q$. |

### 2.3 Seed Schedule
- **Algorithmic Random Seeds**: Fixed 5-seed schedule for stochastic algorithms (MDS random starts, LFR initializations, stochastic reduction oracles):
  $$\mathcal{S}_{\text{algo}} = [0, 7, 19, 37, 73]$$
- **PSU Partitioning Seed**: Fixed seed `20260913` for 2022 F/C splitting.
- **Survey Bootstrap Replicate Seed**: Fixed seed `20260914` for generating the $B=2000$ rescaled PSU replicate weight matrices.

---

## 3. Evaluation Metrics & Exact Formulas

All evaluations on Set S (selection) and Set T (final frozen evaluation) apply complex survey weights $w_i = \text{WTFA\_A}_i$. For deterministic classifiers, $q_i \in \{0, 1\}$. For randomized classifiers (EG, TO), $q_i \in [0, 1]$ represents the exact theoretical positive decision probability.

### 3.1 Survey-Weighted Confusion Matrix & Decision Metrics
For any eligible cohort or demographic subgroup $g$ with indicator $\mathbb{I}_g(i) \in \{0, 1\}$:
$$TP_g = \sum_{i \in \text{Cohort}} w_i \cdot y_i \cdot q_i \cdot \mathbb{I}_g(i)$$
$$FP_g = \sum_{i \in \text{Cohort}} w_i \cdot (1 - y_i) \cdot q_i \cdot \mathbb{I}_g(i)$$
$$FN_g = \sum_{i \in \text{Cohort}} w_i \cdot y_i \cdot (1 - q_i) \cdot \mathbb{I}_g(i)$$
$$TN_g = \sum_{i \in \text{Cohort}} w_i \cdot (1 - y_i) \cdot (1 - q_i) \cdot \mathbb{I}_g(i)$$

- **True Positive Rate (Sensitivity)**:
  $$TPR_g = \frac{TP_g}{TP_g + FN_g}$$
- **False Positive Rate (1 - Specificity)**:
  $$FPR_g = \frac{FP_g}{FP_g + TN_g}$$
- **Balanced Accuracy**:
  $$BA = \frac{TPR + (1 - FPR)}{2}$$
- **Selection Rate (Positive Decision Rate)**:
  $$SR_g = \frac{\sum_{i} w_i \cdot q_i \cdot \mathbb{I}_g(i)}{\sum_{i} w_i \cdot \mathbb{I}_g(i)}$$

### 3.2 Fairness Gap Metrics & Expected Groups Estimability (Addressing R05)
Primary fairness gaps are evaluated strictly across the **full set of expected demographic groups** $\mathcal{G}_{\text{expected}}$:
- Arm 001 (`SEX_A`): $\mathcal{G}_{\text{expected}} = \{1, 2\}$
- Arm 002 (`HISPALLP_A`): $\mathcal{G}_{\text{expected}} = \{1, 2, 3, 4, 5, 6, 7\}$
- Arms 003 & 004 (`DISAB3_A`): $\mathcal{G}_{\text{expected}} = \{1, 2\}$

- **Demographic Parity Gap**:
  $$DP\_gap = \max_{g \in \mathcal{G}_{\text{expected}}} SR_g - \min_{g \in \mathcal{G}_{\text{expected}}} SR_g$$
- **Equal Opportunity Gap (TPR Gap)**:
  $$EOpp\_gap = \max_{g \in \mathcal{G}_{\text{expected}}} TPR_g - \min_{g \in \mathcal{G}_{\text{expected}}} TPR_g$$
- **Equalized Odds Gap (Primary Fairness Metric)**:
  $$EO\_gap = \max\left( \max_{g \in \mathcal{G}_{\text{expected}}} TPR_g - \min_{g \in \mathcal{G}_{\text{expected}}} TPR_g, \quad \max_{g \in \mathcal{G}_{\text{expected}}} FPR_g - \min_{g \in \mathcal{G}_{\text{expected}}} FPR_g \right)$$

*Strict Estimability Invariant*: If any expected group $g \in \mathcal{G}_{\text{expected}}$ lacks positive cases ($TP_g + FN_g = 0$) or negative cases ($FP_g + TN_g = 0$), the primary metric cannot be estimated and returns `status = 'NOT_ESTIMABLE'` (with explicit reason). Imputing zero gap or dropping missing groups is strictly prohibited. Descriptive gaps evaluated across observed subsets are labeled `observed_EO_gap` and cannot be substituted for primary benchmark metrics.

### 3.3 Secondary Predictive Metrics (For Probability Models)
For estimators outputting predicted event probabilities $p_i = \hat{P}(Y=1 \mid X_i)$ (explicitly designated as model probability outputs, not assumed to be "true" or "calibrated"):
- **Survey-Weighted Average Precision (AP)**:
  $$AP = \sum_k (R_k - R_{k-1}) P_k$$
  calculated via sorted thresholds (explicitly distinguishing AP from trapezoidal PR-AUC).
- **Survey-Weighted AUROC**: Area under the ROC curve calculated via rank-order algorithm or weighted trapezoids.
- **Brier Score**:
  $$\text{Brier} = \frac{\sum_i w_i (p_i - y_i)^2}{\sum_i w_i}$$
- **Survey-Weighted Calibration Curve**: Evaluated non-parametrically across predicted probability deciles on Set T. (Spurious `logit(y) ~ logit(p)` regression is removed per R05).

---

## 4. Configuration Selection & Threshold Calibration on Set C & Set S

### 4.1 Common Calibration Rule on Set C (Addressing R07)
For all probability-outputting models, a global decision threshold $t$ is calibrated on Partition C:
1. Candidate thresholds $\mathcal{T}$ comprise all adjacent midpoints of unique sorted predicted probabilities on Set C, plus boundary points $\{0.0, 1.0\}$.
2. Objective: Maximize unweighted Balanced Accuracy on Set C:
   $$t^* = \arg\max_{t \in \mathcal{T}} \text{BA}(t; C)$$
3. Deterministic Tie-Breaking Hierarchy:
   - Tie-breaker 1: Closest to $0.5$ ($|t - 0.5|$ minimized).
   - Tie-breaker 2: Higher threshold ($t$ larger).

### 4.2 FairBias AE Inner Loop Specification (Addressing R07)
- **Inner Feedback Utility**: Evaluated strictly on Partition C using the common threshold calibration rule to compute Balanced Accuracy.
- **Strict Geometric Feasibility**: Enforces $\text{slack} = 0.0$ ($d_\phi \le \epsilon_{\text{ref}}$ strictly; historical $0.02$ relaxation cataloged as sensitivity analysis).
- **Inner Search Budgets**: Maximum 10 outer candidate submissions; maximum 500 candidate utility evaluations.
- **Candidate Ordering**: Deterministic sequence exported from runner configuration.

### 4.3 Pre-registered Operating Points on Set S (2023)
1. **Primary Operating Point**: $\text{EO\_gap} \le 0.10$
2. **Strict Sensitivity Operating Point**: $\text{EO\_gap} \le 0.05$
3. **Relaxed Sensitivity Operating Point**: $\text{EO\_gap} \le 0.20$

### 4.4 Selection Algorithm & Seed Completeness Policy (Addressing R08)
For each method, arm, and backbone:
1. **Seed Completeness Verification**:
   - For stochastic algorithms, verify all 5 seeds $\mathcal{S}_{\text{expected}} = [0, 7, 19, 37, 73]$ succeeded.
   - Any configuration where $\ge 1$ seed failed or timed out is marked `PARTIAL_FAILURE` and disqualified from winning configuration selection. Discarding failed seeds and averaging surviving seeds (survivor bias) is strictly prohibited.
2. **Budget Feasibility**:
   - Among complete configurations, retain those whose mean survey-weighted fairness metric satisfies the operating point budget:
     $$\overline{EO\_gap}_{\text{config}} \le \text{Budget}_{\text{fairness}}$$
   - Preserve both mean budget feasibility and individual seed feasibility flags in the candidate ledger.
3. **Utility Maximization**: Select configuration maximizing mean survey-weighted Balanced Accuracy on Set S:
   $$\text{config}^* = \arg\max_{\text{config} \in \mathcal{C}_{\text{feasible}}} \overline{BA}_{\text{config}}$$
4. **Deterministic Tie-Breaking Hierarchy**:
   - Tie-breaker 1: Lower mean $\overline{EO\_gap}$.
   - Tie-breaker 2: Lower computational complexity (fewer transformations/iterations).
   - Tie-breaker 3: Lexicographical order of `config_id`.
5. **Infeasible Configuration Rule**:
   - If $\mathcal{C}_{\text{feasible}} = \emptyset$, return status `NO_FEASIBLE_CONFIGURATION`. The registry also records the boundary fallback configuration (min $EO\_gap$, then max $BA$) strictly for descriptive reporting.
6. **Unmitigated Predictive Reference**:
   - Maximizes survey-weighted AP on Set S with zero fairness budget.
   - Tie-breakers: (1) higher BA, (2) lower computational complexity, (3) lexicographical `config_id`.
7. **Strict Non-Refitting Invariant**:
   - Selected models are **NEVER re-fitted on F+C+S**. Selected models from F and thresholds from C are deployed directly to Set T.

---

## 5. Complex Survey Design Inference (Addressing R06)

### 5.1 Design Variables & Domain Estimation
- Sampling Weight: `WTFA_A`
- Design Stratum: `PSTRAT`
- Design Cluster: `(PSTRAT, PPSU)` (nested within strata).
- Subpopulations (arms) are analyzed as domains within the master survey design to prevent variance underestimation.

### 5.2 Rescaled PSU Bootstrap Protocol
To estimate sampling variability for non-smooth extreme-gap metrics ($EO\_gap$) and paired contrasts ($\Delta\text{BA}, \Delta\text{EO}$):
1. **Replicate Count**: $B = 2000$ bootstrap replicates, seed `20260914`.
2. **Sampling Mechanism**: Within each stratum $h$ containing $m_h \ge 2$ PSUs, draw an independent simple random sample of $m_h - 1$ PSUs with replacement.
3. **Replicate Weights**: For respondent $i$ in PSU $j$ of stratum $h$:
   $$w_i^{(b)} = w_i \cdot \left[ \frac{m_h}{m_h - 1} \cdot k_{hj}^{(b)} \right]$$
   where $k_{hj}^{(b)}$ is the number of times PSU $(h, j)$ is selected in replicate $b$.
4. **Replicate Computation**: For each replicate $b$, recompute all group rates, extreme gaps $\max_g - \min_g$, and paired differences $\Delta^{(b)} = \theta_{\text{FairBias}}^{(b)} - \theta_{\text{other}}^{(b)}$ using identical replicate weights $w^{(b)}$ across all models.
5. **Sample Variance of Bootstrap Replicates**:
   $$V_{\text{boot}}(\hat{\Delta}) = \frac{1}{B_{\text{eff}} - 1} \sum_{b=1}^{B_{\text{eff}}} \left( \Delta^{(b)} - \bar{\Delta}^* \right)^2$$
   where $\bar{\Delta}^* = \frac{1}{B_{\text{eff}}} \sum_{b=1}^{B_{\text{eff}}} \Delta^{(b)}$.
6. **Degrees of Freedom**: $df = \sum_h m_h - H$ (total PSUs minus total strata in 2024 design).
7. **Singleton Stratum Policy**: If any stratum has $m_h < 2$ PSUs, variance is formally unidentifiable and the inference engine marks the estimate `DESIGN_NOT_ESTIMABLE`.
8. **Effective Replicate Ratio**: If $>5\%$ of bootstrap replicates encounter degenerate group cells ($B_{\text{eff}} < 1900$), the bootstrap confidence interval is marked invalid.
9. **Non-Smooth Gap Coverage Verification**: Extreme range statistics ($\max - \min$) are non-smooth. Gate B2 will evaluate empirical coverage on synthetic designs; if bootstrap coverage falls short of nominal coverage, intervals are downgraded to descriptive status.

### 5.3 Primary Confirmatory Contrast Family (Addressing R06)
The primary confirmatory inferential family comprises **exactly 20 pre-registered contrasts** on Set T (Year 2024):
- **Backbone**: Logistic Regression (LR).
- **Operating Point**: Primary point ($\text{EO\_gap} \le 0.10$).
- **Demographic Arms**: Arm 001 (`SEX_A`) and Arm 003 (`DISAB3_A` full).
- **Comparator Methods**: Exactly 5 external fairness methods:
  1. Reweighing (RW)
  2. LFR Reconstructed (LFR)
  3. Exponentiated Gradient DP (EG-DP)
  4. Exponentiated Gradient EO (EG-EO)
  5. ThresholdOptimizer EO (TO-EO)
  *(Unmitigated is a standalone predictive baseline reference evaluated in parallel, but NOT included in this 20-contrast family).*
- **Contrast Metrics**: $\Delta\text{BA} = \text{BA}_{\text{FairBias}} - \text{BA}_{\text{other}}$ and $\Delta\text{EO} = \text{EO}_{\text{FairBias}} - \text{EO}_{\text{other}}$.
- **Total Contrasts**:
  $$5 \text{ comparators} \times 2 \text{ metrics} (\Delta\text{BA}, \Delta\text{EO}) \times 2 \text{ arms} (\text{Arm 001}, \text{Arm 003}) = \mathbf{20} \text{ unique contrasts}.$$
- **Multiplicity Correction**: Family-wise error rate controlled via Bonferroni adjustment at $\alpha_{\text{family}} = 0.05$:
  $$\alpha_{\text{adjusted}} = \frac{0.05}{20} = 0.0025$$
  Yielding $99.75\%$ two-sided bootstrap confidence intervals:
  $$\hat{\Delta} \pm t_{df, 1 - 0.00125} \cdot \sqrt{V_{\text{boot}}}$$
  Unadjusted 95% intervals ($\hat{\Delta} \pm t_{df, 0.975} \cdot \sqrt{V_{\text{boot}}}$) are presented alongside for descriptive transparency.
- **Fixed Denominator Invariant**: If any external comparator method fails to complete, the Bonferroni denominator remains fixed at $M=20$.

---

## 6. Complete Condition Matrix & Computational Budget (Addressing R07 & R09)

### 6.1 Complete Condition Matrix Breakdown
The complete benchmark matrix contains **76 unique model fitting conditions**:
1. **Core Conditions (54 conditions)**:
   - 7 methods $\times$ 4 arms $- 1$ (LFR Arm 002 `NOT_SUPPORTED`) = 27 conditions per backbone.
   - 2 backbones (LR, `GradientBoostingClassifier`): $27 \times 2 = \mathbf{54}$ core conditions.
2. **AE Ablation Conditions (16 conditions)**:
   - FairBias-BM$\to$AE: 4 arms $\times$ 2 backbones = 8 conditions.
   - FairBias-Joint: 4 arms $\times$ 2 backbones = 8 conditions.
   - Subtotal Core + AE = **70 conditions** (matching the core/AE subtotal).
3. **Survey-Weighted Downstream Training Sensitivity (4 conditions)**:
   - Evaluated on Arm 003 and Arm 004, LR backbone:
     - Unmitigated with survey-weighted training (2 conditions).
     - FairBias-BM with survey-weighted training (2 conditions).
     - Subtotal = **4 conditions**.
4. **Arm 004 Geometric Path Sensitivity (2 conditions)**:
   - Evaluated on Arm 004, LR backbone, holding all other factors (F/C/S, seed, power grid, budget, reference epsilon) identical:
     - FairBias-BM with fixed 2D MDS (1 condition).
     - FairBias-Joint with fixed 2D MDS (1 condition).
     - Paired against the primary stress-elbow versions in core/AE.
     - Subtotal = **2 conditions**.
5. **Grand Total Unique Model Fitting Conditions**:
   $$\mathbf{54} \text{ (Core)} + \mathbf{16} \text{ (AE)} + \mathbf{4} \text{ (Weighted Train)} + \mathbf{2} \text{ (Path)} = \mathbf{76} \text{ conditions}.$$
6. **Evaluation Views (Zero Additional Model Fits)**:
   - Fixed $t=0.5$ threshold sensitivity evaluation on S and T (computed directly on cached probability predictions).
   - Unweighted sample metric evaluations (computed directly on cached predictions).

### 6.2 Computational Budget Limits (Addressing R09)
| Algorithm / Stage | Parameter Grid Size | Algorithmic Seeds | Max Inner Fits / Evaluator Calls | Wall-Clock Timeout per Fit | RAM Ceiling per Worker |
|---|---|---|---|---|---|
| **Unmitigated** | 4 configs | 1 (deterministic) | 4 base fits | 30 minutes | 4 GiB |
| **FairBias-BM** | 32 configs | 5 seeds | Max 50 submitted transforms; max 20,000 geometric calls | 30 minutes | 4 GiB |
| **Reweighing** | 4 configs | 1 (deterministic) | 4 base fits | 30 minutes | 4 GiB |
| **LFR** | 32 configs | 5 seeds | Max 5,000 optimization iterations ($\text{maxiter}=5000, \text{maxfun}=5000$) | 30 minutes | 4 GiB |
| **EG-DP / EG-EO** | 32 configs | 1 (deterministic reduction) | $\text{max\_iter}=50$ iterations (upper-bounded base estimator oracle fits) | 30 minutes | 4 GiB |
| **TO-EO** | 4 configs | 1 (deterministic postprocess) | 4 base fits on F + 4 postprocess on C | 30 minutes | 4 GiB |
| **FairBias AE Ablation** | 32 configs | 5 seeds | Max 10 outer submissions; max 500 candidate utility calls | 30 minutes | 4 GiB |

*Strict Resource Enforcement*: In accordance with the Master Plan, every condition operates under an authoritative limit of **30 minutes wall-clock timeout** and **4 GiB peak resident memory**. Any fit exceeding these limits terminates with status `BUDGET_EXHAUSTED`. Truncated runs are retained in execution logs but disqualified from winning configuration selection.

---

## 7. Literature Fidelity & Source Mapping (Addressing R01)

| Paradigm | Literature Reference | Benchmark Implementation | Algorithmic Source Mapping & Fidelity Status |
|---|---|---|---|
| **FairBias** | Tang et al. (2024), Title: "FairBias: Mitigating Bias in Machine Learning Models via Geometric Data Transformation" | `FairBias-BM application-v1` | • $H=1$ mapped to `mitigation.py:175`.<br>• Stress-elbow MDS mapped to `bias_metric.py:431`.<br>• Interleaved power grid mapped to `enhancement.py:DEFAULT_POLY_GRID`.<br>• NMI gate mapped to `mitigation.py:212, 271`.<br>• Fidelity Status: **`SOURCE_UNVERIFIED (pending Gate B3)`**. Specific publication venue marked `SOURCE_UNVERIFIED`. |
| **Reweighing** | Kamiran & Calders (2012) | AIF360 `aif360.sklearn.preprocessing.Reweighing` | • Multi-category group support.<br>• Sample weights $W_{\text{fair}}$ applied to base classifiers; survey weights $W_{\text{survey}}$ strictly reserved for external evaluation. |
| **LFR** | Zemel et al. (2013) | AIF360 `aif360.algorithms.preprocessing.LFR` | • Continuous reconstructed features $\hat{X}$ feeding common LR/GBDT backbones.<br>• Ground truth labels $y$ isolated from LFR internal modification.<br>• Arm 002 (7-group) rejected as `NOT_SUPPORTED`. |
| **Reductions** | Agarwal et al. (2018) | Fairlearn `fairlearn.reductions.ExponentiatedGradient` | • Reduction cost weights explicitly routed to base estimators.<br>• Evaluated via exact mixture decision probability $q$ rather than thresholded hard decisions. |
| **Threshold Optimizer** | Hardt et al. (2016) | Fairlearn `fairlearn.postprocessing.ThresholdOptimizer` | • Postprocessor fitted strictly on Set C with `prefit=True` on frozen Set F base classifiers.<br>• Requires protected attribute $A$ during inference; deployment trade-offs disclosed. |
