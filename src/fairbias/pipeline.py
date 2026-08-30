"""End-to-end FairBias pipeline orchestrator with Pareto checkpointing and leakage-free execution."""

from __future__ import annotations

import copy
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union

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
    """Encapsulates execution metrics, manifests, and best Pareto iteration state."""

    run_id: str
    config: Dict[str, Any]
    initial_metrics: Dict[str, Any]
    initial_epsilon: Dict[str, Dict[str, float]]
    iterations: List[Dict[str, Any]]
    best_iteration: int
    best_selection_reason: str
    final_metrics: Dict[str, Any]
    final_changed_dict: Dict[str, Any]
    execution_time_seconds: float
    output_file: str


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


def run_fairbias_pipeline(config: Optional[FairBiasConfig] = None) -> FairBiasRunResult:
    """
    Execute the complete FairBias benchmarking pipeline.
    """
    start_time = time.time()
    cfg = config or FairBiasConfig.compas_default()

    # 1. Load Data
    loader = FairDataLoader(cfg)
    X, Y, O, categorical_cols, numerical_cols = loader.prepare_data()

    # 2. Stratified Train/Test Split
    stratify_target = Y if cfg.stratify_split and Y.nunique() > 1 else None
    X_train, X_test, Y_train, Y_test, O_train, O_test = train_test_split(
        X,
        Y,
        O,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=stratify_target,
    )

    # 3. Initialize Evaluator & Transformer
    evaluator = FairEvaluator(
        config=cfg,
        label_O=list(O.columns),
        label_Y=Y.name,
        cate_attrs=categorical_cols,
        num_attrs=numerical_cols,
    )
    transformer = FairTransform(
        n_bins=cfg.transform_n_bins,
        log_epsilon=cfg.transform_log_epsilon,
        x_max=cfg.transform_x_max,
    )

    # Calculate initial NMI strictly on training fold
    nmi_org = calculate_nmi_dict(X_train, Y_train)

    # Calculate initial Epsilon strictly on training fold
    init_epsilon = evaluator.calculate_epsilon(
        X_train, O_train, cate_attrs=categorical_cols, num_attrs=numerical_cols
    )

    # Compute initial performance and fairness metrics
    init_metrics = evaluator.evaluate(
        X_train, Y_train, O_train, X_test, Y_test, O_test
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

    # 4. Initialize Mitigation and Enhancement Engines
    mitigation_engine = FairBiasMitigation(
        evaluator=evaluator,
        transformer=transformer,
        label_O=list(O.columns),
        cate_attrs=categorical_cols,
        num_attrs=numerical_cols,
        phi_threshold=cfg.phi_threshold,
        poly_exponents=cfg.transform_poly_exponents,
    )
    enhancement_engine = FairAccuracyEnhancement(
        evaluator=evaluator,
        transformer=transformer,
        label_Y=Y.name,
        cate_attrs=categorical_cols,
        num_attrs=numerical_cols,
    )

    changed_dict: Dict[str, Any] = {}
    history_iterations: List[Dict[str, Any]] = []

    # Record Initial state as iteration 0 for Pareto comparison
    initial_checkpoint = {
        "iteration": 0,
        "metrics": copy.deepcopy(init_metrics),
        "epsilon_values": copy.deepcopy(init_epsilon),
        "max_epsilon": init_max_eps,
        "avg_epsilon": init_avg_eps,
        "changed_dict": {},
        "selected_attributes": {"selected_label_O": None, "selected_attribute": None},
    }

    current_X_train = X_train.copy()
    current_epsilon = copy.deepcopy(init_epsilon)

    # 5. Iterative Mitigation & Enhancement Loop
    for iter_idx in range(1, cfg.max_iterations + 1):
        iter_data: Dict[str, Any] = {
            "iteration": iter_idx,
            "selected_label_O": None,
            "selected_attribute": None,
        }

        # Step A: Bias Mitigation
        if cfg.use_bias_mitigation:
            # Rebin candidates are searched on the currently transformed frame so the
            # search space matches the epsilon ranking space
            search_X_train = transformer.transform_data(
                X_train, changed_dict, numerical_cols, categorical_cols
            )
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
                X_search=search_X_train,
            )
            iter_data["selected_label_O"] = sel_o
            iter_data["selected_attribute"] = sel_attr

        # Step B: Accuracy Enhancement
        if cfg.use_accuracy_enhancement:
            current_X_train, changed_dict, ae_attr = enhancement_engine.enhance_step(
                X_train=X_train,
                Y_train=Y_train,
                changed_dict=changed_dict,
            )
            if ae_attr and not iter_data["selected_attribute"]:
                iter_data["selected_attribute"] = ae_attr

        # Apply current transformations to both train and test partitions
        transformed_X_train = transformer.transform_data(
            X_train, changed_dict, numerical_cols, categorical_cols
        )
        transformed_X_test = transformer.transform_data(
            X_test, changed_dict, numerical_cols, categorical_cols
        )

        # Unified evaluation on test fold
        metrics = evaluator.evaluate(
            transformed_X_train,
            Y_train,
            O_train,
            transformed_X_test,
            Y_test,
            O_test,
        )

        # Update current epsilon strictly on training partition
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
            "epsilon_values": copy.deepcopy(current_epsilon),
            "max_epsilon": curr_max_eps,
            "avg_epsilon": curr_avg_eps,
            "changed_dict": copy.deepcopy(changed_dict),
            "selected_attributes": iter_data,
        }
        history_iterations.append(iter_record)

        # Early termination checks
        if cfg.use_bias_mitigation and curr_max_eps <= epsilon_threshold:
            break
        if cfg.use_accuracy_enhancement and metrics.get("ACC", 0.0) >= acc_threshold:
            break

    # 6. Pareto Checkpointing: Find Best Iteration
    # Selection rule: Select iteration minimizing fairness gap (EO or SP) subject to ACC >= initial_ACC - tau
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
        f"Selected Iteration {best_iter_num} via Pareto Rule: "
        f"Minimizes {metric_key} ({best_fairness_val:.4f}) under accuracy constraint "
        f"(ACC {best_record['metrics'].get('ACC', 0.0):.4f} >= {min_acc_allowed:.4f})."
    )

    # 7. Final Evaluation on Best State
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

    # 8. Create Unique Run ID and Output Artifacts
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"fairbias_{cfg.dataset_name}_seed{cfg.random_seed}_{ts}"
    out_dir = pathlib.Path(cfg.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "results.json"

    result_payload = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "config_parameters": dataclasses.asdict(cfg),
        "initial_metrics": init_metrics,
        "initial_epsilon": init_epsilon,
        "iterations": history_iterations,
        "best_iteration": best_iter_num,
        "best_selection_reason": best_reason,
        "final_results": {
            "metrics": final_metrics,
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
        iterations=history_iterations,
        best_iteration=best_iter_num,
        best_selection_reason=best_reason,
        final_metrics=final_metrics,
        final_changed_dict=best_changed_dict,
        execution_time_seconds=exec_time,
        output_file=str(out_file),
    )
