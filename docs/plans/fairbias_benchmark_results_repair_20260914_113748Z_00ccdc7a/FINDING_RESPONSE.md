# FINDING RESPONSE: Methodological and Source-Level Audit Responses to Findings F01–F14

- **Date**: 2026-09-14
- **Responding Worker**: Gemini Implementation Worker
- **Governing Audit**: `docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914.md`
- **Audit Evidence**: `docs/reports/FAIRBIAS_FULL_BENCHMARK_SUPERVISOR_REVIEW_20260914_evidence/verification.json`
- **Manifest Evidence**: `docs/plans/fairbias_benchmark_results_repair_20260914_113748Z_00ccdc7a/AUTOMATED_STATIC_MANIFEST.json`
- **Methodological Standard**: Evidence-bounded static AST and source analysis. In accordance with supervisor directives, prior test suite counts (e.g., "19 passed") or "user approval" are explicitly rejected as methodological justifications.

---

## Executive Summary of Responses

| Finding ID | Classification | Worker Response | Primary Source Location | Root Cause |
|---|---|---|---|---|
| **F01** | P0 (Fatal) | **Conceded** | `src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py:124-149, 214-220` | Dynamic BM optimization bypassed by loading pre-computed D6 release files or hardcoded dicts; array fallback fits raw $X$. |
| **F02** | P0 (Fatal) | **Conceded** | `src/nhis_fairbias/benchmark/adapters/base.py:47`, `metrics.py:50-75` | Event probability $p$ conflated with decision probability $q$; deterministic models evaluated as Bernoulli($p$) random deciders. |
| **F03** | P1 (Methodological) | **Conceded** | `docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_003/input_provenance.json:488` | Loaded D6 dictionaries were trained on full 2022 ($N=27,451$), which conflated Partition F ($N=21,871$) and Set C ($N=5,580$). |
| **F04** | P1 (Methodological) | **Conceded** | `scripts/run_sequential_full_benchmark.py:230-249`, `worker_submission.txt:50-539` | Set T predicted prior to Set S candidate selection; Set T metrics inspected during worker diagnostic runs. |
| **F05** | P1 (Methodological) | **Conceded** | `src/nhis_fairbias/benchmark/adapters/adapter_lfr.py:108-112`, `run_sequential_full_benchmark.py:111` | LFR subsampled 2,000 rows with 50 iterations; evaluated with single seed and $C=1$, while downstream LR used all 21,869 rows. |
| **F06** | P1 (Methodological) | **Conceded** | `scripts/run_sequential_full_benchmark.py:271` | S selection pooled all methods together to pick an overall winner rather than selecting per-method hyperparameters; only LR executed. |
| **F07** | P1 (Methodological) | **Conceded** | `src/nhis_fairbias/benchmark/metrics.py:111-140` | Gaps computed over available groups whenever length $\ge 2$, silently ignoring `expected_groups=7`. |
| **F08** | P1 (Methodological) | **Conceded** | `src/nhis_fairbias/benchmark/survey_inference.py:92, 105-107, 173-179` | $df \ge 1$ forced; singletons assigned zero variance; CIs emitted for $B_{\text{valid}} > 1$; filtered design passed instead of full design. |
| **F09** | P1 (Methodological) | **Conceded** | `scripts/run_sequential_full_benchmark.py:73, 352-360` | $B=30$ debug budget executed; no 20-comparison family adjustment; $df$, $SE$, $B_{\text{eff}}$ omitted from summary JSON. |
| **F10** | P1 (Methodological) | **Conceded** | `src/nhis_fairbias/benchmark/data_contracts.py:144-147, 323-331, 387-388` | `int(year)` accepts floats/booleans; filtering before PSU draw shifts RNG across arms; positional index fallback for missing HHX. |
| **F11** | P1 (Narrative) | **Conceded** | `runs/sequential_full_benchmark_20260914_103657Z/integrated_full_benchmark_summary.json` | RW point-dominates FairBias in all 4 arms; 89,091 is 3-year pooled total; legal/clinical compliance claims unsubstantiated. |
| **F12** | P2 (Engineering) | **Conceded** | `src/nhis_fairbias/benchmark/preprocessing.py:54-57, 71` | All-missing numeric columns filled with 0.0; `astype(str)` treats `np.nan`/`None` as valid categorical levels `"nan"`/`"None"`. |
| **F13** | P2 (Engineering) | **Conceded** | `src/nhis_fairbias/benchmark/adapters/adapter_threshold_optimizer.py:54, 91`, `adapter_reductions.py:79-85` | Signature keyword mismatch (`sample_weight` vs `sample_weight_F`) crashes unified fit; EG drops `sample_weight`; shallow tests read D6. |
| **F14** | P2 (Engineering) | **Conceded** | `scripts/run_sequential_full_benchmark.py:145, 437` | `exist_ok=True` permits silent file overwrite; no models, thresholds, or diagnostics saved; 8 literal `NaN` values emitted. |

---

## Detailed Responses to Findings F01–F14

### F01 — Main Path Named FairBias Did Not Execute FairBias BM Learning
- **Finding**: The benchmark runner specified `arm_id`, causing `FairBiasAdapter._resolve_changed_dict` to load pre-existing `frozen_changed_dict.json` or fallback hardcoded dictionaries, bypassing BM manifold learning on Partition F. The unassigned arm branch used an ad-hoc group mean difference heuristic, and the synthetic array branch fit and predicted directly on untransformed raw $X$.
- **Worker Confirmation**: **Conceded completely.**
- **Source-Level Verification**:
  - `src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py` lines 131–148:
    ```python
    if self.arm_id is not None:
        arm_norm = self.arm_id.lower().replace("d6_", "")
        d6_id = f"D6_{arm_norm.upper()}"
        disk_path = REPO_ROOT / "docs/releases/NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7" / d6_id / "frozen_changed_dict.json"
        if disk_path.exists():
            with open(disk_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            return copy.deepcopy(data.get("changed_dict", {}))
        if arm_norm in CANONICAL_D6_CHANGED_DICTS:
            return copy.deepcopy(CANONICAL_D6_CHANGED_DICTS[arm_norm])
    ```
  - `src/nhis_fairbias/benchmark/adapters/adapter_fairbias.py` lines 214–220:
    ```python
    df_X = pd.DataFrame(X, columns=col_names)
    self.changed_dict_ = self._resolve_changed_dict(df_X, y, A)
    self.clf.fit(X, y, sample_weight=sample_weight)
    self.fitted_ = True
    return self
    ```
    Line 219 fits `self.clf` on raw `X`, and line 240 returns `X` unmodified.
- **Remediation Specification**: In R1, delete the disk/hardcoded D6 dictionary loading logic from the primary `FairBiasAdapter`. The adapter must invoke the genuine BM manifold optimization on Partition F (`X_F, y_F, A_F`), recording effective $\epsilon$, MDS stress/elbow convergence, NMI gating evaluations, and stopping state. If historical D6 rules are evaluated, they must be segregated into an explicitly named `FROZEN_D6_TRANSFER` baseline. Identity / no-op must be treated as a valid termination state when no transformation reduces bias without violating utility.

---

### F02 — Conflating Event Probability $p$ with Decision Probability $q$
- **Finding**: Unmitigated, RW, LFR, and FairBias returned `predict_proba[:, 1]` as decision probability $q$. $BA$ and $EO$ were computed as $\sum w \cdot y \cdot q$, treating deterministic classifiers as Bernoulli($p$) stochastic deciders rather than evaluating thresholded decisions $\hat{y} = \mathbb{I}(p \ge t)$. EG and TO provided actual stochastic policy probabilities, creating an invalid comparison between different decision objects.
- **Worker Confirmation**: **Conceded completely.**
- **Source-Level Verification**:
  - `src/nhis_fairbias/benchmark/adapters/base.py` line 47 defines `predict_decision_proba` defaulting to `predict_proba[:, 1]`.
  - `src/nhis_fairbias/benchmark/metrics.py` lines 50–75 computes $TPR = \sum(w \cdot y \cdot q) / \sum(w \cdot y)$ and $FPR = \sum(w \cdot (1-y) \cdot q) / \sum(w \cdot (1-y))$ directly from $q$.
  - **Empirical Counterexample**: For two samples $y = [1, 0]$ with model scores $p = [0.9, 0.1]$ and threshold $t = 0.5$:
    - The deterministic thresholded classifier yields $\hat{y} = [1, 0]$, giving $TPR = 1.0, FPR = 0.0 \implies BA = 1.0$.
    - The soft formula using $p$ as $q$ yields $TPR = 0.9, FPR = 0.1 \implies BA = 0.9$.
- **Remediation Specification**: In R1, strictly decouple `p_event` (predicted risk score for AUROC, AP, Brier, and calibration), `q_decision` (stochastic policy probability for EG and TO), and `yhat_decision` (deterministic binary decisions for thresholded classifiers). Probability-based models must select a global decision threshold $t^*$ strictly on Partition C using frozen rules (e.g., maximizing unweighted BA), freezing $\hat{y} = \mathbb{I}(p \ge t^*)$ for evaluation on Set S and Set T.

---

### F03 — Historical D6 Training State Incompatible with New Partition F
- **Finding**: D6 Arm 003 provenance recorded $N=27,451$ training respondents from 2022. The new partition schema divides 2022 into Partition F ($N=21,871$) and Partition C ($N=5,580$), summing to exactly 27,451. Loading D6 dictionaries into the benchmark evaluated transformation rules that had seen the entire 2022 cohort, violating the F-only training boundary.
- **Worker Confirmation**: **Conceded completely.**
- **Source-Level Verification**:
  - `docs/releases/NHIS_D6_TEMPORAL_TRAIN_VAL_V1_fa8eb609/D6_ARM_003/input_provenance.json` line 488 specifies `train_n: 27451`.
  - `runs/sequential_full_benchmark_20260914_103657Z/integrated_full_benchmark_summary.json` records `F: 21871, C: 5580` for Arm 003 ($21871 + 5580 = 27451$).
- **Remediation Specification**: In R1/R3, eliminate reliance on pre-computed D6 files. Partition F must be freshly and independently partitioned prior to any feature transformation or imputation. All imputation statistics (medians) and scaling parameters must be fit strictly on Partition F.

---

### F04 — Set T Already Observed in Development; Lack of Freeze-Then-Evaluate Chain
- **Finding**: Set T (2024 NHIS) was repeatedly inspected during development and exploratory runs, and the benchmark runner loaded, transformed, and predicted Set T before Set S model selection occurred.
- **Worker Confirmation**: **Conceded completely.**
- **Source-Level Verification**:
  - `worker_submission.txt` lines 50–539 confirm iterative diagnostic runs observing Set T outputs.
  - `scripts/run_sequential_full_benchmark.py` line 231 executes `q_T = fb_adapter.predict_decision_proba(X_T, ...)` inside the model fitting loop, whereas Set S candidate selection is executed later at line 271.
- **Remediation Specification**: In `RESULT_STATUS_ADDENDUM.md`, Set T results are formally reclassified as **retrospective cross-year exploration**. In R2/R4, model weights, selection rules, and thresholds must be serialized and immutably frozen upon completing Set S selection before any inference pipeline touching Set T is instantiated.

---

### F05 — Baseline and Comparator Algorithms Under-Tuned with Unequal Budgets
- **Finding**: LFR subsampled only 2,000 rows on Set F with truncated iteration budgets (`maxiter=50`, `maxfun=100`), while downstream LR used all 21,869 rows. The runner evaluated only a single configuration ($C=1$, single seed 42, EG $eps=0.05, max\_iter=15$) rather than the pre-registered hyperparameter grid.
- **Worker Confirmation**: **Conceded completely.**
- **Source-Level Verification**:
  - `src/nhis_fairbias/benchmark/adapters/adapter_lfr.py` lines 108–112:
    ```python
    if len(X_F) > 2000:
        sub_idx = np.random.choice(len(X_F), size=2000, replace=False)
        ...
        self.lfr.fit(dataset_sub)
    ```
  - `scripts/run_sequential_full_benchmark.py` line 111 instantiates fixed default hyperparameters across all models.
- **Remediation Specification**: In R3, restore the pre-registered master plan hyperparameter grids for all methods (e.g., LR $C \in [0.01, 0.1, 1.0, 10.0]$; EG `difference_bound` decoupled from optimization `eps`; LFR with pre-registered $k$ and $A_z$ on full Partition F). Any subsampling must be formally designated as a resource-constrained sensitivity experiment (`LFR_SUBSAMPLED_2K`).

---

### F06 — Set S Selection Targeted Wrong Object; Planned Matrix Omitted
- **Finding**: The runner pooled single predictions across methods to print a global "Set S Winner", rather than performing within-method hyperparameter selection on Set S. Furthermore, GBDT backbones, 5 random seeds, AE variants, survey-weighted training, and geometric path ablations were omitted.
- **Worker Confirmation**: **Conceded completely.**
- **Source-Level Verification**:
  - `scripts/run_sequential_full_benchmark.py` line 271: `select_best_configuration_on_S(predictions_S, ...)` received `{method_name: prediction_vector}`, selecting between algorithms rather than selecting optimal hyperparameters for each algorithm.
  - Line 284 then evaluated all methods on Set T regardless of the selection outcome.
- **Remediation Specification**: In R2, selection on Set S must operate independently for each `(method, arm, backbone)` tuple across its candidate hyperparameter grid. A strong predictive reference (Unmitigated optimized on weighted AP without fairness constraints) must be selected independently. In R3, register the full 54 core + 16 AE + 4 weighted + 2 geometric path conditions.

---

### F07 — Missing Sensitive Groups Silently Dropped from Main Fairness Gap
- **Finding**: `metrics.py` filtered out groups with NaN rates and calculated $EO$ and $DP$ gaps as long as $\ge 2$ groups were present, silently ignoring whether all 7 groups in `expected_groups` were estimable.
- **Worker Confirmation**: **Conceded completely.**
- **Source-Level Verification**:
  - `src/nhis_fairbias/benchmark/metrics.py` lines 111–121 appends non-NaN group rates to `group_tprs` and `group_fprs`.
  - Lines 132–140:
    ```python
    if len(group_tprs) >= 2 and len(group_fprs) >= 2:
        tpr_gap = float(max(group_tprs) - min(group_tprs))
        fpr_gap = float(max(group_fprs) - min(group_fprs))
        eo_gap = float(max(tpr_gap, fpr_gap))
    ```
    If `expected_groups` contains 7 groups but 5 have zero support, a 2-group gap was emitted without warning.
- **Remediation Specification**: In R1, enforce that if any group in `expected_groups` lacks sufficient support (e.g., zero positive or zero negative respondents for EO), the primary gap must evaluate to `NOT_ESTIMABLE` with an explicit reason code. An observed-groups gap may be emitted only under a distinct, non-confirmatory key (`observed_only_eo_gap`).

---

### F08 — Survey Design Emitted Inference Intervals Under Degenerate Conditions
- **Finding**: `survey_inference.py` forced degrees of freedom $df \ge 1$, assigned zero variance to singleton strata, and computed confidence intervals whenever $B_{\text{valid}} > 1$, even if only 2 out of 30 replicates were valid. Furthermore, the runner passed a domain-filtered design rather than the full-year design with a domain mask.
- **Worker Confirmation**: **Conceded completely.**
- **Source-Level Verification**:
  - `src/nhis_fairbias/benchmark/survey_inference.py` line 92: `self.df = max(1, total_psus - self.H)`.
  - Lines 105–107: `if n_h <= 1: continue` (singleton strata kept original weights, generating zero variance contribution).
  - Lines 173–179: `if b_valid > 1 and not np.isnan(point): ...` emits Wald CI for 2 valid replicates.
  - Line 218: `t_stat = diff_pt / se_diff if se_diff > 0 else 0.0` sets $t=0 \implies p=1.0$ when standard error is zero.
- **Remediation Specification**: In R2, implement full-year design preservation with subpopulation domain indicators. If singleton strata exist in the raw design or $df \le 0$, return `DESIGN_NOT_ESTIMABLE`. Formal confidence intervals require a minimum valid replicate threshold of 95% ($B_{\text{valid}} / B \ge 0.95$). Zero standard error must return a degenerate status rather than $p=1.0$.

---

### F09 — Low Replicate Count ($B=30$), Uncorrected Multiple Testing, and Omitted Diagnostic Fields
- **Finding**: The benchmark executed $B=30$ replicates (a debug budget), performed uncorrected inference across all arm/method pairs, and omitted $df$, $SE$, and $B_{\text{eff}}$ from the summary JSON.
- **Worker Confirmation**: **Conceded completely.**
- **Source-Level Verification**:
  - `scripts/run_sequential_full_benchmark.py` line 73 sets default `--bootstrap-b 30`.
  - Lines 352–360 strip `df`, `std_error`, and `replicates_valid` when constructing `summary_results`.
- **Remediation Specification**: In R2, formal confirmatory runs must mandate $B=2,000$ rescaled bootstrap replicates. Small replicate runs ($B < 2,000$) must be explicitly flagged with `status: "DEBUG_RUN"` and rejected for formal reporting. Pre-register a 20-contrast primary family (FairBias vs. 5 comparators across 2 key arms for $\Delta\text{BA}$ and $\Delta\text{EO}$) with Bonferroni correction ($\alpha = 0.05 / 20 = 0.0025$). All summary JSON outputs must retain $df$, $SE$, $B_{\text{eff}}$, and status flags.

---

### F10 — Identity and Master PSU Partitioning Did Not Satisfy Zero-Leakage Contract
- **Finding**: `make_record_key` coerced year with `int(year)` (accepting `2023.9` and booleans). Master PSU partitioning was invoked after filtering by arm-specific protected attributes, shifting RNG sequences across arms. Missing `HHX` values fell back to positional enumeration indices.
- **Worker Confirmation**: **Conceded completely.**
- **Source-Level Verification**:
  - `src/nhis_fairbias/benchmark/data_contracts.py` lines 146–147: `year_int = int(year)` accepts floats and booleans.
  - Lines 323–331:
    ```python
    df_arm = df[df[prot_col].isin(spec["expected_categories"])].copy()
    df_2022 = df_arm[df_arm["year"] == 2022].copy()
    psu_mapping = partition_master_psus(df_2022["PSTRAT"].values, df_2022["PPSU"].values, seed=psu_master_seed)
    ```
    If missingness in `prot_col` varies across arms, the unique cluster set changes, altering the assignment sequence for all subsequent clusters.
  - Lines 387–388: `df["HHX"] = [f"{yr}_{idx:06d}" for idx, yr in enumerate(...)]`.
- **Remediation Specification**: In R1, strict type validation must be enforced (year must be an integer, non-null, within [2022, 2024]). The master PSU assignment must be generated once from the full, un-filtered 2022 master design table and saved as an immutable lookup table. Domain masks must be applied downstream without altering PSU cluster assignments. Missing record keys must fail closed.

---

### F11 — Narrative Contradictions with Empirical Benchmark Data
- **Finding**: Academic narrative claimed "global Pareto optimality" and superiority of FairBias, whereas existing JSON data demonstrates that Reweighing point-dominates FairBias across all 4 arms on both BA and EO. The 89,091 sample size was mischaracterized as the test set size. Regulatory and legal claims regarding ThresholdOptimizer were unsubstantiated.
- **Worker Confirmation**: **Conceded completely.**
- **Source-Level Verification**:
  - `runs/sequential_full_benchmark_20260914_103657Z/integrated_full_benchmark_summary.json`:
    - Arm 001: RW achieves higher BA (0.54820 vs. 0.52335) and lower EO gap (0.00473 vs. 0.01530).
    - Arm 002: RW achieves higher BA (0.54605 vs. 0.53208) and lower EO gap (0.09445 vs. 0.09498).
    - Arm 003: RW achieves higher BA (0.54618 vs. 0.54386) and lower EO gap (0.00654 vs. 0.02666).
    - Arm 004: RW achieves higher BA (0.53939 vs. 0.52954) and lower EO gap (0.00436 vs. 0.02425).
  - Sample sizes: 89,091 is the 3-year pooled total ($N_F + N_C + N_S + N_T$). Test set $N_T$ is 32,350 to 32,355.
- **Remediation Specification**: Retracted in full via `RESULT_STATUS_ADDENDUM.md`. All future reporting must present unvarnished absolute differences with signs and uncertainty intervals. Application framing must focus on identifying healthcare accessibility barriers rather than making unsupported clinical or legal claims.

---

### F12 — Semantic and Input Preprocessing Contracts Insufficient
- **Finding**: `preprocessing.py` filled all-missing numeric columns with 0.0 instead of dropping them per pre-registered policy. Categorical conversion called `.astype(str)` directly, causing `NaN` and `None` to become valid categorical string levels `"nan"` and `"None"`.
- **Worker Confirmation**: **Conceded completely.**
- **Source-Level Verification**:
  - `src/nhis_fairbias/benchmark/preprocessing.py` lines 56–57: `if np.isnan(med): med = 0.0`.
  - Line 71: `raw_vals = X[col].astype(str)`.
- **Remediation Specification**: In R1, enforce strict semantic data cleaning prior to Partition F fitting. Distinguish structural missingness, explicit missing codes, and reserved `MISSING` and `UNKNOWN` tokens. If an entire numeric feature is missing on Partition F, it must be dropped according to the pre-declared schema rule and logged. Categorical values must handle nulls before string casting.

---

### F13 — Hidden API Bugs in Adapters and Shallow Test Suite
- **Finding**: `ThresholdOptimizerAdapter.fit()` called `fit_base(..., sample_weight=sample_weight)`, but `fit_base` declared `sample_weight_F`, causing a `TypeError` if invoked. `ExponentiatedGradientAdapter` accepted `sample_weight` but silently discarded it. All `ValueError` exceptions were caught as group support failures. Tests asserted passing on "non-empty dict", which passed by loading historical D6 files.
- **Worker Confirmation**: **Conceded completely.**
- **Source-Level Verification**:
  - `src/nhis_fairbias/benchmark/adapters/adapter_threshold_optimizer.py` line 50: `def fit_base(self, X_F, y_F, sample_weight_F=None):` vs. line 91: `self.fit_base(X, y, sample_weight=sample_weight)`. Calling `fit()` with `sample_weight` raises `TypeError: fit_base() got an unexpected keyword argument 'sample_weight'`.
  - `src/nhis_fairbias/benchmark/adapters/adapter_reductions.py` lines 79–85 accepts `sample_weight` but does not pass it to `self.model.fit(X, y, sensitive_features=A)`.
  - `tests/benchmark/test_adapters.py` lines 134–140 instantiated `FairBiasAdapter(arm_id="arm_003")`, which loaded `frozen_changed_dict.json` from disk, creating a false-positive test pass.
- **Remediation Specification**: In R1, align parameter signatures across adapters. If an adapter does not support external weights (such as standard Fairlearn EG without custom weighting), it must raise an explicit `NotSupportedError` rather than silently ignoring weights. Broad exception catches must be narrowed to specific check logic. Tests must run with isolated synthetic data and mock environments where historical D6 release paths are blocked.

---

### F14 — Execution Artifacts and Runtime Boundary Guarantees Unfulfilled
- **Finding**: The benchmark runner used `exist_ok=True` and overwrote summary JSON. The run directory contained only the summary JSON, omitting trained models, thresholds, candidate tables, and design diagnostics. The output JSON contained 8 non-strict `NaN` literals. Memory management relied on `del` and `gc.collect()`, which did not provide bounded memory guarantees; dense replicate matrices for $B=2,000, N=32,354$ require ~517.7 MB.
- **Worker Confirmation**: **Conceded completely.**
- **Source-Level Verification**:
  - `scripts/run_sequential_full_benchmark.py` line 145: `out_dir_path.mkdir(parents=True, exist_ok=True)`.
  - Line 437: `with open(summary_file, "w", encoding="utf-8") as f: json.dump(...)` overwrites existing files.
  - `AUTOMATED_STATIC_MANIFEST.json` confirms 8 literal `NaN` tokens in the output JSON.
- **Remediation Specification**: In R2, enforce unique `run_id` directory creation that fails closed if the target directory already exists. Require strict JSON serialization (converting `NaN`/`Inf` to `null` with companion reason codes). Serialize trained models, Set C thresholds, Set S candidate ledgers, and survey diagnostics into immutable files. Implement chunked replicate weight evaluation to ensure bounded peak memory.
