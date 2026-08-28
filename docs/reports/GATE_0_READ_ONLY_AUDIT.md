# Gate 0: Read-Only Baseline Audit Report

## 1. Audit Overview & Integrity Verification

- **Objective**: Conduct a comprehensive read-only audit of the inherited `code_v_0_3` codebase without modifying the environment or filesystem.
- **Git State**: No Git repository initialized at audit time.
- **File Counts**:
  - Non-virtualenv files: 35 files
  - Total filesystem items (including virtualenv directories `.venv`, `.venv311`): 74,395 files
- **Filesystem Integrity**:
  - Pre-audit aggregate SHA-256: `220ba37302ce81fca10d27a588a23511ad2182bc4a63ad30bb3fcebab2ca1458`
  - Post-audit aggregate SHA-256: `220ba37302ce81fca10d27a588a23511ad2182bc4a63ad30bb3fcebab2ca1458`
  - Verified: **Zero writes, zero modifications** during the audit.

---

## 2. Inherited Benchmark Data Audit

| File | Total Lines | Header Rows | Data Rows | SHA-256 Hash |
|---|---|---|---|---|
| `data_COMPAS.csv` | 6,173 | 1 | 6,172 | `4b2bc1d55553f1c2061768bd245b725ee57b3f7e438f612b78c60f308848cb27` |
| `data_Credit_Card.csv` | 30,001 | 1 | 30,000 | `303cf916663273a345671688c96e9be7a83c76e39f0a2727f8d0ed33b8d6df1e` |

---

## 3. Verified Critical Methodological & Technical Flaws

1. **Final Evaluation Leakage (`main.py`)**: While `main.py` creates an initial train/test split, final evaluation trains on the full `transformed_df` (which contains test rows) and evaluates on `X_test`, causing severe test-set leakage.
2. **Full-Data Epsilon & Bias Mitigation (BM) Search**: Despite the existing train/test split, iterative searches for fairness bias metric $\epsilon$ and Bias Mitigation (BM) transformations receive the full dataset, including test rows.
3. **Train/Test Scaler Refitting**: Evaluation code invokes `scaler.fit_transform(X_test)`, modifying test distributions independently rather than applying frozen training parameters.
4. **Pre-Split Data-Dependent Preprocessing**: Column-type inference, label encoding, and protected/target binning are applied across entire tables before train/test partitioning.
5. **Unconditional Full DataFrame Printing**: `module_load.py` executes full un-truncated `print(df)` statements upon loading datasets, violating microdata privacy guardrails.
6. **Historical Result Overwriting**: Outputs write directly to unversioned files (`results/all_results.json`) without timestamped run IDs or execution provenance.
7. **Random Seed Inconsistency**: Random states are missing or inconsistently set across `main.py`, `classifiers.py`, and transformation modules, preventing reproducible execution.
8. **Incomplete Dependency Coverage**: While `requirements.txt` pins exact versions for all 8 packages it declares, dependency coverage is incomplete: unconditional `torch` (imported across classifiers/eval) and conditional `matplotlib` are undeclared; `pytorch_tabular` and Excel/Parquet backends are optional path-dependent dependencies.
9. **Quadratic Sample-Pair Scaling Risks (`num-c` / `num-d` in `eval.py`)**: While default execution uses `num-a` (which is linear and not quadratic), `eval.py` contains quadratic sample-pair computations: `num-c` constructs a cross-group difference matrix (around lines 953–954) and `num-d` executes `pairwise_distances` (around line 1026). If configured on full MEPS cohorts, these $O(N^2)$ operations risk memory and compute exhaustion without chunking or approximations.

---

## 4. Explicit Audit Corrections & Nuances

- **Figure Generation (`test02.png`)**: Code inspecting image generation revealed that the save call for `test02.png` resides inside an uncalled nested helper function and is not executed in default runs.
- **Historical Output Line Count**: `results/all_results.json` contains 524 newline characters (525 total lines), representing 5 historical benchmark iterations rather than a large multi-run database.
- **Scope of MEPS Analysis**: Gate 0 audited only inherited files. Exact MEPS HC-252 variable mappings, panel rounds (Panel 27 Rounds 1–5), longitudinal linkage variables, and survey design weights were not verified or mapped in Gate 0 and are deferred to the new research track.
