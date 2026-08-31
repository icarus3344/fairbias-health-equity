# Claims & Evidence Matrix: MEPS Longitudinal Fairness Research

This document establishes the binding scientific claims boundary for all manuscripts, reports, presentations, and publications originating from this repository.

---

## 1. Permitted Empirical Claims (Evidence-Bounded)

The following claims are directly supported by verified unit tests, cryptographic execution manifests, and statistical outputs:

1. **Methodological Pipeline Scalability**: The linear-complexity bias concentration estimation operates in $O(N \cdot D)$ linear time, strictly avoiding $O(N^2)$ sample-by-sample pairwise distance matrices, and executes within 5 seconds with $< 0.20$ GB RSS memory.
2. **Leakage-Free Partitioning**: The DUID-grouped 60/20/20 partitioning architecture guarantees zero household or record overlap across Train, Validation, and Calibration partitions ($\text{DUID}_{\text{train}} \cap \text{DUID}_{\text{val}} \cap \text{DUID}_{\text{cal}} = \emptyset$).
3. **Train-Only Preprocessing**: Missing-value imputations (training medians), continuous standardizations, and categorical encodings are fit strictly on the Training partition without test-set distribution leakage. Demographic variables (`RACETHX`, `SEX`) are strictly excluded from the baseline feature matrix $X$.
4. **Survey Design Structure**: All complex survey design structures (`LONGWT > 0`, `VARSTR` 2001–2117, `VARPSU` $\ge 1$) were verified, with Kish effective sample size $n_{\text{eff}} = 1,745.2$ on Panel 26.
5. **Statistical Power Finding**: On Panel 26 (HC-244), the working-age continuous-baseline cohort contains 2,882 individuals and exactly 136 positive coverage interruption events ($Y=1$, event rate 4.72%), falling below the prespecified 200-positive power threshold.
6. **Temporal Holdout Lock Integrity**: Panel 27 (HC-252) remains 100% locked under `LOCKED_UNDERPOWERED_STOP_CONDITION` because the conditional unlock prerequisite ($n_{\text{pos}} \ge 200$) was not satisfied. Zero Panel 27 outcome data, distributions, protected-group summaries, prevalence, or performance metrics were accessed.
7. **Exploratory Internal Development Evidence**: On the Panel 26 calibration partition (evaluated as apparent fit diagnostics across 5 random development seeds `20260828`–`20260832`), the unmitigated group-agnostic baseline achieved mean Weighted AUROC $0.5564 \pm 0.0519$ and Weighted AUPRC $0.0638 \pm 0.0245$, while the exploratory survey-weighted centering heuristic achieved Weighted AUROC $0.5580 \pm 0.0503$ and Weighted AUPRC $0.0750 \pm 0.0409$.
8. **Subgroup Cell Suppression Reality**: Under standard privacy and statistical reliability rules ($n < 100 \lor \text{pos} < 20 \lor \text{neg} < 20 \lor n_{\text{eff}} < 50$), 100% of race/ethnicity and sex subgroup cells on the evaluation partition were suppressed; all pairwise max TPR disparities and primary fairness endpoints are Not Estimable (`None`), and bootstrap fairness inference propagates non-estimability (`fairness_max_tpr_gap` status `NOT_ESTIMABLE_SUPPRESSED`) without fabricating a numeric zero or fairness confidence interval.

---

## 2. Strictly Prohibited Claims (Scientific & Ethical Violations)

The following claims are completely prohibited:

1. **PROHIBITED: Clinical or Underwriting Deployment Claim**: Claiming that this model is ready for unassisted real-world deployment in individual insurance denial, benefit cuts, or clinical underwriting. (The authorized scope is strictly supportive retention outreach).
2. **PROHIBITED: "Absolute Fairness" or "Zero Bias" Claim**: Claiming that the mitigated model achieves "complete algorithmic fairness" or eliminates all structural health inequities.
3. **PROHIBITED: Fabricated Holdout Generalization Claim**: Claiming that the model was tested or validated on Panel 27 outcomes. (Panel 27 remained strictly locked due to the underpowering stop condition).
4. **PROHIBITED: Causal Inference or Policy Intervention Claim**: Claiming that changing feature values causes retention improvements, or asserting causal treatment effects without a randomized or quasi-experimental design.
5. **PROHIBITED: Claiming Inherited Legacy Code Correctness**: Claiming that the inherited root files (`module_AE.py`, `module_BM.py`, `module_transform.py`, `eval.py`) correctly reproduced published literature or satisfied survey weighting. (They exhibited critical methodological deficiencies including test leakage, unweighted objectives, and $O(N^2)$ memory limits).
6. **PROHIBITED: Subgroup Generalization on Suppressed Cells**: Drawing substantive conclusions about racial/ethnic subgroups whose cell counts failed suppression thresholds ($n < 100 \lor \text{pos} < 20 \lor \text{neg} < 20 \lor n_{\text{eff}} < 50$).
7. **PROHIBITED: Unsupported Disparity Reduction or Fabricated Numeric Fairness Claims**: Asserting numerical disparity reduction (such as 0.142 to 0.096) or fabricating a 0.0 disparity / zero CI on evaluation partitions where 100% of subgroup cells are suppressed.
8. **PROHIBITED: Conference-Ready Result Package Claims**: Claiming that a conference-ready validation package or full empirical evaluation is complete. (Study stopped fail-closed at development stage).
9. **PROHIBITED: Faithful Paper Reconstruction Claims Without Evidence**: Claiming that the exploratory group-aware centering heuristic is a faithful reconstruction of Tang et al. (2024), or claiming that it operates without protected attributes at inference time.

