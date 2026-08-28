# 1. Freeze Inherited Codebase as Immutable Historical Baseline

Date: 2026-08-28

## Context

The repository was received containing an inherited fairness benchmarking codebase (`code_v_0_3`) comprising 14 legacy files (and existing virtual environments). A comprehensive read-only audit conducted in Gate 0 revealed several critical methodological and software engineering defects in this legacy codebase:
- Final evaluation flaw: while `main.py` creates an initial split, final evaluation trains classifiers on the full `transformed_df` (which contains test rows) and evaluates on `X_test`, resulting in test set leakage.
- Iterative bias metric (epsilon) and Bias Mitigation (BM) search performed on full datasets prior to evaluation.
- Test data distributions altered via separate scaler fitting (`scaler.fit_transform(X_test)`).
- Pre-split data transformations and target encoding introducing data leakage.
- Unconditional full DataFrame printing in data loaders risking microdata exposure.
- Execution results (`results/all_results.json`) overwritten without run manifests or seeds.
- Quadratic sample-pair matrix computations in `eval.py` (`num-c` cross-group matrix at lines 953–954; `num-d` `pairwise_distances` at line 1026) posing severe $O(N^2)$ scaling risks if selected over the default `num-a` branch.

To enable rigorous, scientifically valid research on the Medical Expenditure Panel Survey (MEPS HC-252 Panel 27 longitudinal cohort), the project requires a clean structural separation between the flawed legacy code and the new research track.

---

## Decision

1. **Freeze Inherited Baseline**: The initial state of the repository is formally committed and tagged as an immutable historical baseline.
   - **Commit SHA**: `038897e9f751edac6e36445b7706eec5fdb15988`
   - **Tree SHA**: `9e43047f69326a844cec1e7acdb6726af555dff3`
   - **Baseline Tag**: `inherited-code-v0.3-baseline-20260828`
   - **Tracked Files (15)**: `.gitignore`, `app.py`, `classifiers.py`, `config.py`, `data_COMPAS.csv`, `data_Credit_Card.csv`, `eval.py`, `main.py`, `module_AE.py`, `module_BM.py`, `module_load.py`, `module_transform.py`, `requirements.txt`, `results/all_results.json`, `start.sh`.
2. **Preserve Root Files for Auditability**: The 14 inherited root files remain in place at the repository root to ensure full provenance and traceability. They must not be modified, moved, renamed, reformatted, or deleted.
3. **Inherited Evidence Tier**: The inherited code and historical results (`results/all_results.json`) serve solely as evidence of a historical execution record, but provide no valid evidence of unbiased model performance, fairness improvement, reproduction of the paper, or MEPS validity.
4. **Additive Research Branch**: All subsequent development takes place on the dedicated branch `research/meps-hc252-longitudinal` using additive directories (`src/`, `configs/`, `docs/`, `scripts/`, `tests/`).

---

## Consequences

- **Positive**: Complete historical transparency and reproducible audit trail; zero risk of regression or silent tampering with baseline artifacts.
- **Positive**: Clear boundary isolating new MEPS longitudinal research from legacy methodological flaws.
- **Negative**: Root directory contains legacy files alongside new project documentation, requiring clear documentation mapping in `README.md`.
