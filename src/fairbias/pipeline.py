"""End-to-end FairBias pipeline orchestrator.

Leakage-free execution protocol (paper: training 64% / validation 16% /
test 20%):

- The TRAINING partition determines bias concentrations (d_phi) and the
  data transforms (greedy mitigation until d_phi < epsilon).
- The VALIDATION partition produces the per-iteration metrics used for
  early stopping and Pareto checkpoint selection.  The test partition
  never informs model or transform selection.
- The TEST partition is evaluated exactly once, after the best
  checkpoint has been locked in.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import hashlib
import json
import pathlib
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fairbias.config import FairBiasConfig
from fairbias.data import FairDataLoader
from fairbias.enhancement import FairAccuracyEnhancement
from fairbias.evaluator import FairEvaluator
from fairbias.mitigation import FairBiasMitigation
from fairbias.transform import FairTransform, calculate_nmi_dict


@dataclasses.dataclass
class FairBiasRunResult:
    """Encapsulates execution metrics, manifests, and best Pareto iteration state.

    ``initial_metrics`` and per-iteration ``metrics`` are computed on the
    VALIDATION partition; ``final_metrics`` is the single, locked-in
    evaluation on the TEST partition.
    """

    run_id: str
    config: Dict[str, Any]
    initial_metrics: Dict[str, Any]
    initial_epsilon: Dict[str, Dict[str, float]]
    epsilon_threshold: float
    iterations: List[Dict[str, Any]]
    best_iteration: int
    best_selection_reason: str
    final_metrics: Dict[str, Any]
    final_changed_dict: Dict[str, Any]
    # Strict-paper failure record: highest-d_phi attribute whose exhaustive
    # transform search could not reach the epsilon ball (None when the run
    # fully mitigated or ran to the iteration budget without failure).
    non_convergence: Optional[Dict[str, Any]] = None
    execution_time_seconds: float = 0.0
    output_file: str = ""


def _serialize_object(obj: Any) -> Any:
    """Recursively convert NumPy/Pandas objects to JSON-serializable types."""
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _serialize_object(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_object(item) for item in obj]
    return obj


def _sha256_of_file(path: str) -> Optional[str]:
    """SHA-256 of an input file (dataset provenance; empty string if missing)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _stratify_series(Y: pd.Series, enabled: bool) -> Optional[pd.Series]:
    if enabled and Y.nunique() > 1 and not Y.isna().any():
        return Y
    return None


def run_fairbias_pipeline(config: Optional[FairBiasConfig] = None) -> FairBiasRunResult:
    """
    Execute the complete FairBias benchmarking pipeline.
    """
    start_time = time.time()
    cfg = config or FairBiasConfig.compas_default()

    # 1. Load RAW data (encoding is deferred until after the split)
    loader = FairDataLoader(cfg)
    X_raw, Y_raw, O_raw, categorical_cols, numerical_cols = loader.prepare_data()
    dataset_sha = _sha256_of_file(cfg.dataset_path)

    # 2. Paper split: train 64% / validation 16% / test 20%, stratified on Y.
    # First hold out validation+test jointly, then carve test out of the
    # holdout in proportion test_size / (val_size + test_size) so that the
    # TEST partition truly receives 20% of the FULL dataset (the previous
    # two-stage split handed 80% to train and only 4% to test).
    holdout_size = cfg.val_size + cfg.test_size
    X_tr_raw, X_hold_raw, Y_tr_raw, Y_hold_raw, O_tr_raw, O_hold_raw = train_test_split(
        X_raw,
        Y_raw,
        O_raw,
        test_size=holdout_size,
        random_state=cfg.random_seed,
        stratify=_stratify_series(Y_raw, cfg.stratify_split),
    )
    test_fraction_of_holdout = cfg.test_size / max(1e-12, holdout_size)
    X_va_raw, X_te_raw, Y_va_raw, Y_te_raw, O_va_raw, O_te_raw = train_test_split(
        X_hold_raw,
        Y_hold_raw,
        O_hold_raw,
        test_size=test_fraction_of_holdout,
        random_state=cfg.random_seed,
        stratify=_stratify_series(Y_hold_raw, cfg.stratify_split),
    )

    # Defensive assertion on the OBSERVED row counts (the manifest used to
    # echo the configured fractions while the actual split was different).
    n_total = len(X_raw)
    if n_total > 0:
        observed = {
            "train": len(X_tr_raw) / n_total,
            "validation": len(X_va_raw) / n_total,
            "test": len(X_te_raw) / n_total,
        }
        configured = {
            "train": 1.0 - holdout_size,
            "validation": cfg.val_size,
            "test": cfg.test_size,
        }
        for part, frac in observed.items():
            if abs(frac - configured[part]) > 0.05:
                raise RuntimeError(
                    f"Observed {part} split fraction {frac:.4f} deviates from the "
                    f"configured {configured[part]:.4f}; refusing to continue with a "
                    "mis-partitioned dataset."
                )
        observed_fractions = observed
    else:
        observed_fractions = {"train": 0.0, "validation": 0.0, "test": 0.0}

    # 3. Fit encoders STRICTLY on the training partition, then transform
    loader.fit_encoders(X_tr_raw, Y_tr_raw, O_tr_raw)
    X_train, Y_train, O_train = loader.transform_partition(X_tr_raw, Y_tr_raw, O_tr_raw)
    X_val, Y_val, O_val = loader.transform_partition(X_va_raw, Y_va_raw, O_va_raw)
    X_test, Y_test, O_test = loader.transform_partition(X_te_raw, Y_te_raw, O_te_raw)

    # 4. Initialize Evaluator & Transformer
    evaluator = FairEvaluator(
        config=cfg,
        label_O=list(O_raw.columns),
        label_Y=Y_raw.name,
        cate_attrs=categorical_cols,
        num_attrs=numerical_cols,
    )
    transformer = FairTransform(
        n_bins=cfg.transform_n_bins,
        log_epsilon=cfg.transform_log_epsilon,
        x_max=cfg.transform_x_max,
    )

    # Bias concentrations and NMI strictly on the TRAINING partition
    nmi_org = calculate_nmi_dict(X_train, Y_train)
    init_epsilon = evaluator.calculate_epsilon(
        X_train, O_train, cate_attrs=categorical_cols, num_attrs=numerical_cols
    )

    # Initial performance/fairness metrics on the VALIDATION partition
    init_metrics = evaluator.evaluate(
        X_train, Y_train, O_train, X_val, Y_val, O_val
    )

    init_acc = float(init_metrics.get("ACC", 0.0))

    # Extract initial max and average epsilon
    all_init_eps = [
        val for group_dict in init_epsilon.values() for val in group_dict.values()
    ]
    init_avg_eps = float(np.mean(all_init_eps)) if all_init_eps else 0.0
    init_max_eps = float(np.max(all_init_eps)) if all_init_eps else 0.0

    epsilon_threshold = evaluator.compute_threshold(init_epsilon)
    acc_threshold = init_acc * (1.0 + cfg.threshold_accuracy)

    # 5. Initialize Mitigation and Enhancement Engines
    mitigation_engine = FairBiasMitigation(
        evaluator=evaluator,
        transformer=transformer,
        label_O=list(O_raw.columns),
        cate_attrs=categorical_cols,
        num_attrs=numerical_cols,
        phi_threshold=cfg.phi_threshold,
        poly_exponents=cfg.transform_poly_exponents,
        failed_attribute_mode=cfg.failed_attribute_mode,
    )
    enhancement_engine = FairAccuracyEnhancement(
        evaluator=evaluator,
        transformer=transformer,
        label_Y=Y_raw.name,
        cate_attrs=categorical_cols,
        num_attrs=numerical_cols,
    )

    changed_dict: Dict[str, Any] = {}
    history_iterations: List[Dict[str, Any]] = []

    # Record Initial state as iteration 0 for Pareto comparison
    # (validation metrics, same partition as every other checkpoint)
    initial_checkpoint = {
        "iteration": 0,
        "metrics": copy.deepcopy(init_metrics),
        "metrics_partition": "validation",
        "epsilon_values": copy.deepcopy(init_epsilon),
        "max_epsilon": init_max_eps,
        "avg_epsilon": init_avg_eps,
        "changed_dict": {},
        "selected_attributes": {"selected_label_O": None, "selected_attribute": None},
    }

    current_epsilon = copy.deepcopy(init_epsilon)

    # 6. Iterative Mitigation & Enhancement Loop
    for iter_idx in range(1, cfg.max_iterations + 1):
        iter_data: Dict[str, Any] = {
            "iteration": iter_idx,
            "selected_label_O": None,
            "selected_attribute": None,
        }

        # Step A: Bias Mitigation (searched on the training partition,
        # accepted only when the attribute's d_phi falls below epsilon)
        sel_attr: Optional[str] = None
        if cfg.use_bias_mitigation:
            (
                current_X_train,
                changed_dict,
                sel_o,
                sel_attr,
            ) = mitigation_engine.mitigate_step(
                X=X_train,
                Y=Y_train,
                O=O_train,
                nmi_org=nmi_org,
                changed_dict=changed_dict,
                current_epsilon=current_epsilon,
                epsilon_threshold=epsilon_threshold,
            )
            iter_data["selected_label_O"] = sel_o
            iter_data["selected_attribute"] = sel_attr

        # Step B: Accuracy Enhancement
        ae_attr: Optional[str] = None
        if cfg.use_accuracy_enhancement:
            current_X_train, changed_dict, ae_attr = enhancement_engine.enhance_step(
                X_train=X_train,
                Y_train=Y_train,
                changed_dict=changed_dict,
            )
            if ae_attr and not iter_data["selected_attribute"]:
                iter_data["selected_attribute"] = ae_attr

        if sel_attr is None and ae_attr is None:
            # No transform accepted this round: the state is terminal.  Under
            # the strict-paper "stop" failure mode this carries the recorded
            # non-convergence (highest attribute could not enter the epsilon
            # ball); under a fully mitigated state every attribute is inside
            # the ball and non_convergence stays None.
            break

        # Apply current transformations to all partitions
        transformed_X_train = transformer.transform_data(
            X_train, changed_dict, numerical_cols, categorical_cols
        )
        transformed_X_val = transformer.transform_data(
            X_val, changed_dict, numerical_cols, categorical_cols
        )

        # Per-iteration metrics on the VALIDATION partition (never test)
        metrics = evaluator.evaluate(
            transformed_X_train,
            Y_train,
            O_train,
            transformed_X_val,
            Y_val,
            O_val,
        )

        # Update current epsilon strictly on the training partition
        current_epsilon = evaluator.calculate_epsilon(
            transformed_X_train, O_train, categorical_cols, numerical_cols
        )
        all_eps = [
            val for group_dict in current_epsilon.values() for val in group_dict.values()
        ]
        curr_max_eps = float(np.max(all_eps)) if all_eps else 0.0
        curr_avg_eps = float(np.mean(all_eps)) if all_eps else 0.0

        iter_record = {
            "iteration": iter_idx,
            "metrics": copy.deepcopy(metrics),
            "metrics_partition": "validation",
            "epsilon_values": copy.deepcopy(current_epsilon),
            "max_epsilon": curr_max_eps,
            "avg_epsilon": curr_avg_eps,
            "changed_dict": copy.deepcopy(changed_dict),
            "selected_attributes": iter_data,
        }
        history_iterations.append(iter_record)

        # Early termination checks (training epsilon + validation accuracy)
        if cfg.use_bias_mitigation and curr_max_eps <= epsilon_threshold:
            break
        if cfg.use_accuracy_enhancement and metrics.get("ACC", 0.0) >= acc_threshold:
            break

    # 7. Pareto Checkpointing on VALIDATION metrics only
    # Selection rule: minimize fairness gap (EO or SP on validation) subject to
    # ACC >= initial validation ACC - tau.  The test partition is NOT consulted.
    candidates = [initial_checkpoint] + history_iterations
    min_acc_allowed = init_acc - cfg.accuracy_tolerance_tau

    feasible_candidates = [
        c for c in candidates if c["metrics"].get("ACC", 0.0) >= min_acc_allowed
    ]
    if not feasible_candidates:
        feasible_candidates = candidates

    metric_key = cfg.selection_metric.upper()  # "EO" or "SP"

    def get_fairness_penalty(record: Dict[str, Any]) -> float:
        m_dict = record["metrics"].get(metric_key, {})
        if isinstance(m_dict, dict):
            vals = [float(v) for v in m_dict.values()]
            return float(np.mean(vals)) if vals else 0.0
        return float(m_dict)

    best_record = min(feasible_candidates, key=get_fairness_penalty)
    best_iter_num = int(best_record["iteration"])
    best_changed_dict = copy.deepcopy(best_record["changed_dict"])
    best_fairness_val = get_fairness_penalty(best_record)

    best_reason = (
        f"Selected Iteration {best_iter_num} via Pareto Rule on VALIDATION metrics: "
        f"Minimizes {metric_key} ({best_fairness_val:.4f}) under accuracy constraint "
        f"(validation ACC {best_record['metrics'].get('ACC', 0.0):.4f} >= {min_acc_allowed:.4f}). "
        f"Test partition evaluated exactly once afterwards."
    )

    # 8. Final evaluation: single, locked-in evaluation on the TEST partition
    final_transformed_X_train = transformer.transform_data(
        X_train, best_changed_dict, numerical_cols, categorical_cols
    )
    final_transformed_X_test = transformer.transform_data(
        X_test, best_changed_dict, numerical_cols, categorical_cols
    )
    final_metrics = evaluator.evaluate(
        final_transformed_X_train,
        Y_train,
        O_train,
        final_transformed_X_test,
        Y_test,
        O_test,
    )

    exec_time = time.time() - start_time

    # 9. Create Unique Run ID and Output Artifacts
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"fairbias_{cfg.dataset_name}_seed{cfg.random_seed}_{ts}"
    out_dir = pathlib.Path(cfg.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"

    result_payload = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "config_parameters": dataclasses.asdict(cfg),
        "provenance": {
            "dataset_path": cfg.dataset_path,
            "dataset_sha256": dataset_sha,
            "code_note": "src/fairbias (see Git state in the gate report)",
        },
        "split": {
            "configured_fractions": {
                "train": 1.0 - cfg.test_size - cfg.val_size,
                "validation": cfg.val_size,
                "test": cfg.test_size,
            },
            "observed_fractions": observed_fractions,
            "row_counts": {
                "train": int(len(X_train)),
                "validation": int(len(X_val)),
                "test": int(len(X_test)),
            },
            "encoders_fitted_on": "train",
        },
        "selection_partition": "validation",
        "final_evaluation_partition": "test",
        "epsilon_threshold": epsilon_threshold,
        "failed_attribute_mode": cfg.failed_attribute_mode,
        "mitigation_non_convergence": (
            copy.deepcopy(mitigation_engine.non_convergence)
            if cfg.use_bias_mitigation else None
        ),
        "initial_metrics": init_metrics,
        "initial_metrics_partition": "validation",
        "initial_epsilon": init_epsilon,
        "iterations": history_iterations,
        "best_iteration": best_iter_num,
        "best_selection_reason": best_reason,
        "final_results": {
            "metrics": final_metrics,
            "metrics_partition": "test",
            "changed_dict": best_changed_dict,
        },
        "execution_time_seconds": exec_time,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(_serialize_object(result_payload), f, indent=2)

    return FairBiasRunResult(
        run_id=run_id,
        config=dataclasses.asdict(cfg),
        initial_metrics=init_metrics,
        initial_epsilon=init_epsilon,
        epsilon_threshold=epsilon_threshold,
        iterations=history_iterations,
        best_iteration=best_iter_num,
        best_selection_reason=best_reason,
        final_metrics=final_metrics,
        final_changed_dict=best_changed_dict,
        non_convergence=(
            copy.deepcopy(mitigation_engine.non_convergence)
            if cfg.use_bias_mitigation else None
        ),
        execution_time_seconds=exec_time,
        output_file=str(out_file),
    )
