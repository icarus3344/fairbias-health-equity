# Experiment Registry and Computational Budget Draft (Benchmark V1)

**Document ID**: `docs/plans/fairbias_benchmark_b0_20260914/EXPERIMENT_REGISTRY_DRAFT.md`  
**Execution Timestamp**: `2026-09-14T13:36:35+08:00`  
**Active Gate**: `FAIRBIAS-BENCHMARK-B0`  
**Authority Reference**: [Master Plan](file:///Users/lkc/Downloads/code_v_0_3/docs/plans/FAIRBIAS_PRIMARY_BENCHMARK_MASTER_PLAN_20260913.md) | [Consolidated Audit](file:///Users/lkc/Downloads/code_v_0_3/docs/reports/FAIRBIAS_CONSOLIDATED_AUDIT_AND_APPLICATION_PLAN_20260913.md)  
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
   - Estimator: `sklearn.ensemble.HistGradientBoostingClassifier` (or verified `GradientBoostingClassifier`)
   - Learning Rate: `0.05`
   - Subsample: `1.0`
   - Hyperparameter Grid:
     $$\text{n\_estimators} \in \{100, 200\} \times \text{max\_depth} \in \{2, 3\}$$ (4 configurations)

### 2.2 Fairness Method Hyperparameter Grids
| Method ID | Hyperparameter Parameters & Grid Values | Configurations per Arm/Backbone | Inner Optimization & Execution Notes |
|---|---|---|---|
| **UNMITIGATED** | Backbone hyperparameters only ($C$ or depth/estimators). | **4** | Pure predictive baseline without fairness constraints. |
| **FAIRBIAS_BM** | Backbone grid $\times$ Epsilon Multiplier:$$\mu_\epsilon \in \{0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 4.00, 8.00\}$$where $\epsilon_{\text{cand}} = \mu_\epsilon \cdot \epsilon_{\text{ref}}$. | **32** | $\epsilon_{\text{ref}}$ is calculated strictly on the semantic representation of Partition F. Greedy search stops on first feasible state or budget exhaustion. |
| **REWEIGHING** | Backbone grid $\times$ Standard Kamiran & Calders multi-group balancing factor:$$W_{\text{fair}}(A=a, Y=y) = \frac{P(A=a)P(Y=y)}{P(A=a, Y=y)}$$ | **4** | Standard reweighing factor has no intensity tuning parameter. Computed strictly on Partition F. |
| **LFR_RECONSTRUCTED** | Backbone grid $\times$ Dimensionality $k \in \{5, 10\} \times$ Fair constraint weight $A_z \in \{0.1, 1.0, 10.0, 50.0\}$ with fixed $A_x = 0.01, A_y = 1.0$. | **32** | Reconstructed continuous representations $\hat{X}$ feed downstream backbone classifiers. Applicable only to binary demographic arms (Arms 001, 003, 004). Arm 002 is `NOT_SUPPORTED`. |
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

### 3.2 Fairness Gap Metrics
Using extreme range across all observed demographic categories $g \in \mathcal{G}$:
- **Demographic Parity Gap**:
  $$DP\_gap = \max_{g \in \mathcal{G}} SR_g - \min_{g \in \mathcal{G}} SR_g$$
- **Equal Opportunity Gap (TPR Gap)**:
  $$EOpp\_gap = \max_{g \in \mathcal{G}} TPR_g - \min_{g \in \mathcal{G}} TPR_g$$
- **Equalized Odds Gap (Primary Fairness Metric)**:
  $$EO\_gap = \max\left( \max_{g \in \mathcal{G}} TPR_g - \min_{g \in \mathcal{G}} TPR_g, \quad \max_{g \in \mathcal{G}} FPR_g - \min_{g \in \mathcal{G}} FPR_g \right)$$

*Estimability Rule*: If any demographic group has 0 true positive cases ($\sum w_i y_i = 0$) or 0 true negative cases ($\sum w_i (1 - y_i) = 0$), the respective rate is undefined, and the gap metric returns status `NOT_ESTIMABLE`. It is strictly forbidden to impute 0.0 or substitute an ad-hoc rate.

### 3.3 Secondary Predictive Metrics (For Probability Models)
For estimators outputting true calibrated event probabilities $p_i = P(Y=1 \mid X_i)$:
- **Survey-Weighted Average Precision (AP)**:
  $$AP = \sum_k (R_k - R_{k-1}) P_k$$
  calculated via sorted thresholds (explicitly distinguishing AP from trapezoidal PR-AUC).
- **Survey-Weighted AUROC**: Area under the ROC curve calculated via rank-order algorithm or weighted trapezoids.
- **Brier Score**:
  $$\text{Brier} = \frac{\sum_i w_i (p_i - y_i)^2}{\sum_i w_i}$$
- **Calibration Slope & Intercept**: Evaluated via survey-weighted logistic regression $\text{logit}(y) \sim \text{logit}(p)$.

---

## 4. Configuration Selection Rules on Set S (2023)

To prevent cross-year cherry-picking, configuration selection is frozen strictly on Set S (2023) prior to unlocking 2024 data:

### 4.1 Pre-registered Operating Points
1. **Primary Operating Point**: $\text{EO\_gap} \le 0.10$
2. **Strict Sensitivity Operating Point**: $\text{EO\_gap} \le 0.05$
3. **Relaxed Sensitivity Operating Point**: $\text{EO\_gap} \le 0.20$

### 4.2 Selection Algorithm
For each method, arm, and backbone:
1. **Filter Eligible Configurations**: Retain all configurations with execution status `VALID` whose mean survey-weighted fairness metric on Set S satisfies the operating point budget:
   $$\overline{EO\_gap}_{\text{config}} \le \text{Budget}_{\text{fairness}}$$
2. **Maximize Predictive Utility**: Among budget-feasible candidates, select the configuration that maximizes mean survey-weighted Balanced Accuracy on Set S:
   $$\text{config}^* = \arg\max_{\text{config} \in \mathcal{C}_{\text{feasible}}} \overline{BA}_{\text{config}}$$
3. **Deterministic Tie-Breaking Hierarchy**:
   - Tie-breaker 1: Lower mean $\overline{EO\_gap}$.
   - Tie-breaker 2: Lower computational complexity (fewer iterations / transformations).
   - Tie-breaker 3: Lexicographical order of `config_id`.
4. **Infeasible Configuration Rule**:
   - If $\mathcal{C}_{\text{feasible}} = \emptyset$ (no configuration satisfies the budget), the operating point returns status `NO_FEASIBLE_CONFIGURATION`.
   - The registry additionally records the "boundary fallback" configuration (minimizing $EO\_gap$, then maximizing $BA$) strictly for descriptive reporting.
5. **Unmitigated Predictive Reference**:
   - An independent unmitigated reference configuration is selected on Set S maximizing survey-weighted AP with zero fairness constraint. This guarantees a rigorous baseline that has not been handicapped by common fairness selection rules.

---

## 5. Complex Survey Design Inference

### 5.1 Design Structure
- Weight Variable: `WTFA_A`
- Stratum Variable: `PSTRAT`
- Cluster Variable: `(PSTRAT, PPSU)` (PSU identifiers are nested within strata and are not globally unique).
- Full Population Domain: Analysis is conducted over the entire survey design. Subpopulations (arms) are analyzed as domains within the master survey design to prevent variance underestimation.

### 5.2 Rescaled PSU Bootstrap Protocol
To estimate sampling variability for non-smooth extreme-gap metrics ($EO\_gap$) and paired contrasts ($\Delta\text{BA}, \Delta\text{EO}$):
1. **Replicate Count**: $B = 2000$ bootstrap replicates, seed `20260914`.
2. **Sampling Mechanism**: Within each stratum $h$ containing $m_h \ge 2$ PSUs, draw an independent simple random sample of $m_h - 1$ PSUs with replacement.
3. **Replicate Weights**: For respondent $i$ in PSU $j$ of stratum $h$:
   $$w_i^{(b)} = w_i \cdot \left[ \frac{m_h}{m_h - 1} \cdot k_{hj}^{(b)} \right]$$
   where $k_{hj}^{(b)}$ is the number of times PSU $(h, j)$ is selected in replicate $b$.
4. **Singleton Stratum Policy**: If any stratum has $m_h < 2$ PSUs, design variance is formally unidentifiable and the inference engine marks the estimate `DESIGN_NOT_ESTIMABLE`.
5. **Effective Replicate Ratio**: If $>5\%$ of bootstrap replicates encounter degenerate group cells (causing undefined rates), the bootstrap confidence interval is marked invalid.

### 5.3 Multiplicity Correction & Confirmatory Contrast Family
The primary inferential hypothesis family comprises **20 pre-registered contrasts** on Set T (Year 2024):
- Primary Backbone: Logistic Regression (LR).
- Primary Operating Point: $\text{EO\_gap} \le 0.10$.
- Primary Demographic Arms: Arm 001 (`SEX_A`) and Arm 003 (`DISAB3_A` full).
- Contrasts: FairBias-BM vs 5 External Methods (Unmitigated, Reweighing, LFR, EG-DP, EG-EO, TO-EO).
- Metrics: $\Delta\text{BA} = \text{BA}_{\text{FairBias}} - \text{BA}_{\text{other}}$ and $\Delta\text{EO} = \text{EO}_{\text{FairBias}} - \text{EO}_{\text{other}}$.
- Total Contrasts: $5 \text{ methods} \times 2 \text{ metrics} \times 2 \text{ arms} = 20$ contrasts.
- **Family-Wise Error Rate**: Controlled via Bonferroni correction at $\alpha_{\text{family}} = 0.05$:
  $$\alpha_{\text{adjusted}} = \frac{0.05}{20} = 0.0025$$
  Corresponding to $99.75\%$ two-sided bootstrap confidence intervals.

---

## 6. Complete Condition Matrix & Computational Budget

### 6.1 Condition Matrix Breakdown
- **Core Methods**: 7 methods (Unmitigated, FairBias-BM, RW, LFR, EG-DP, EG-EO, TO-EO).
- **Arms**: 4 arms (001, 002, 003, 004).
- **Structural Void**: LFR on Arm 002 is `NOT_SUPPORTED` ($-1$ cell).
- **Supported Conditions per Backbone**: $7 \times 4 - 1 = \mathbf{27}$ method-arm conditions.
- **Two Backbones (LR + GBDT)**: $27 \times 2 = \mathbf{54}$ core conditions.
- **Ablation Studies (FairBias-BM$\to$AE, FairBias-Joint)**: 2 ablations $\times 4 \text{ arms} \times 2 \text{ backbones} = \mathbf{16}$ ablation conditions.
- **Total Registered Conditions**: $\mathbf{70}$ conditions.

### 6.2 Computational Upper-Bound Limits
| Algorithm / Stage | Parameter Grid Size | Algorithmic Seeds | Max Inner Fits / Evaluator Calls | Wall-Clock Timeout per Fit | RAM Ceiling per Worker |
|---|---|---|---|---|---|
| **Unmitigated** | 4 configs | 1 (deterministic) | 4 fits | 5 minutes | 2 GiB |
| **FairBias-BM** | 32 configs | 5 seeds | Max 50 submitted transforms; max 20,000 geometric calls | 30 minutes | 4 GiB |
| **Reweighing** | 4 configs | 1 (deterministic) | 4 fits | 5 minutes | 2 GiB |
| **LFR** | 32 configs | 5 seeds | Max 5,000 optimization iterations | 30 minutes | 4 GiB |
| **EG-DP / EG-EO** | 32 configs | 1 (deterministic reduction) | Max 50 base estimator fits per reduction | 30 minutes | 4 GiB |
| **TO-EO** | 4 configs | 1 (deterministic postprocess) | 4 base fits on F + 4 postprocess on C | 10 minutes | 2 GiB |
| **FairBias AE Ablation** | 32 configs | 5 seeds | Max 10 outer submissions; max 500 candidate utility calls | 30 minutes | 4 GiB |

*Strict Resource Enforcement*: Any condition exceeding the 30-minute timeout or 4 GiB memory limit is terminated with status `BUDGET_EXHAUSTED`. Truncated runs are retained in execution logs but disqualified from winning configuration selection.

---

## 7. Literature Fidelity & Deviation Mapping

| Paradigm | Canonical Reference | Benchmark Implementation | Known Engineering & Scientific Deviations |
|---|---|---|---|
| **FairBias** | Tang et al. (2024), *Information Sciences* | `FairBias-BM application-v1` | 1. MDS dimensionality uses stress-elbow heuristic (ablation compares fixed 2D).<br>2. Evaluation uses survey-weighted EO gap rather than sample $d_\phi$.<br>3. Strict feasibility enforced ($\text{slack}=0$); historical $0.02$ relaxation cataloged as sensitivity analysis.<br>4. Multi-group HISP uses max-pair formulation aligned with Tang et al. (2024). |
| **Reweighing** | Kamiran & Calders (2012), *KAIS* | AIF360 `aif360.sklearn.preprocessing.Reweighing` | 1. Evaluated using multi-category group support.<br>2. Sample weights $W_{\text{fair}}$ applied to base classifiers; survey weights $W_{\text{survey}}$ strictly reserved for external evaluation. |
| **LFR** | Zemel et al. (2013), *ICML* | AIF360 `aif360.algorithms.preprocessing.LFR` | 1. Primary benchmark uses continuous reconstructed features $\hat{X}$ feeding common LR/GBDT backbones rather than native classification head.<br>2. Ground truth labels $y$ isolated from LFR internal modification.<br>3. Arm 002 (7-group) rejected as `NOT_SUPPORTED`. |
| **Reductions** | Agarwal et al. (2018), *ICML* | Fairlearn `fairlearn.reductions.ExponentiatedGradient` | 1. Reduction cost weights explicitly routed to base estimators.<br>2. Evaluated via exact mixture decision probability $q$ rather than thresholded hard decisions. |
| **Threshold Optimizer** | Hardt et al. (2016), *NeurIPS* | Fairlearn `fairlearn.postprocessing.ThresholdOptimizer` | 1. Postprocessor fitted strictly on Set C with `prefit=True` on frozen Set F base classifiers.<br>2. Requires protected attribute $A$ during inference; deployment trade-offs disclosed. |
