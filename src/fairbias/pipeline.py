"""End-to-end FairBias pipeline orchestrator.

Leakage-free execution protocol (paper: training 64% / validation 16% /
test 20%):

- The TRAINING partition determines bias concentrations (d_phi) and the
  data transforms (greedy mitigation until d_phi < epsilon).
- The VALIDATION partition produces the per-iteration metrics used for
  the ENGINEERING Pareto checkpoint selection.  The test partition never
  informs model or transform selection.
- The TEST partition is evaluated exactly once per REPORTED TERMINAL
  STATE.

Algorithm modes (Round 4.1; renamed twice — the second rename is the
Round 4.1 REPAIR-2 verdict of 2026-08-30 — see ``fairbias.config``):

- ``official_code_derived_monotone_cursor_unweighted``: an
  OFFICIAL-CODE-DERIVED VARIANT WITH A TERMINATION-SAFETY EXTENSION —
  derived from the official code repository's behavior, but NOT
  claimed to be behaviorally equivalent to it and NOT a paper-text
  method reproduction (the paper text prescribes elbow-plot MDS
  dimension selection; the fixed dim=2 is inherited official-code
  behavior).  Official interleaved power stream (order preserved,
  searched under a MONOTONE per-attribute stream cursor — the
  DELIBERATE deviation from the official restart-from-head search that
  guarantees termination of the budget-free loop), NO finite iteration
  budget, NO validation-Pareto rollback.  The greedy termination state
  is the SOLE reported state, under the key
  ``final_results_official_code_derived_monotone_cursor_unweighted``.

- ``engineering_bounded`` (default): the bounded configuration —
  automatic MDS dimension, six-value ascending power grid, finite
  ``max_iterations``, and two separately reported terminal states that
  must never be merged:

  1. ``final_results_configured_greedy_terminal``: the termination state
     of the configured greedy loop (the last accepted transform state,
     with no validation-based rollback).  This makes NO paper-alignment
     claim (hence the rename from the round-4 "paper_strict").
  2. ``final_results_pareto_engineering``: the validation-Pareto-selected
     checkpoint — an explicitly named ENGINEERING extension, NOT a paper
     output.
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

from fairbias.config import (
    ALGORITHM_MODE_ENGINEERING,
    ALGORITHM_MODE_OFFICIAL,
    FairBiasConfig,
)
from fairbias.data import FairDataLoader
from fairbias.enhancement import FairAccuracyEnhancement
from fairbias.evaluator import FairEvaluator
from fairbias.mitigation import FairBiasMitigation
from fairbias.transform import FairTransform, calculate_nmi_dict


@dataclasses.dataclass
class FairBiasRunResult:
    """Encapsulates execution metrics, manifests, and terminal states.

    ``initial_metrics`` and per-iteration ``metrics`` are computed on the
    VALIDATION partition.  ``greedy_terminal_metrics`` is the single TEST
    evaluation of the greedy algorithm's termination state — in
    ``official_code_derived_monotone_cursor_unweighted`` mode this is
    the official-code-derived output; in ``engineering_bounded`` mode
    it is the configured greedy terminal state (no paper-alignment
    claim).  ``pareto_engineering_metrics`` is the single TEST evaluation
    of the validation-Pareto-selected checkpoint and exists ONLY in
    ``engineering_bounded`` mode (None in the official-code-derived
    mode, which has no Pareto rollback).
    """

    run_id: str
    config: Dict[str, Any]
    algorithm_mode: str
    initial_metrics: Dict[str, Any]
    initial_epsilon: Dict[str, Dict[str, float]]
    epsilon_threshold: float
    iterations: List[Dict[str, Any]]
    best_iteration: int
    best_selection_reason: str
    greedy_terminal_metrics: Dict[str, Any]
    greedy_terminal_changed_dict: Dict[str, Any]
    # Termination semantics of the greedy loop: converged,
    # termination_reason, terminal_iteration, terminal_max_dphi,
    # epsilon_threshold.
    termination: Dict[str, Any]
    # Configured-grid failure record: highest-d_phi attribute whose
    # configured candidate-grid search could not reach the epsilon ball
    # (None when the run fully mitigated or ran to the iteration budget
    # without failure).  This is "configured grid exhausted", NOT a
    # paper-level non-convergence claim.
    non_convergence: Optional[Dict[str, Any]] = None
    # Pareto checkpoint (ENGINEERING mode only; None in official mode,
    # which has no validation-Pareto rollback).
    pareto_engineering_metrics: Optional[Dict[str, Any]] = None
    pareto_engineering_changed_dict: Optional[Dict[str, Any]] = None
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
    # bool must be checked BEFORE int: Python bools are ints, and the
    # termination record's ``converged`` must serialize as true/false.
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
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


def _termination_note(
    termination_reason: str,
    is_official: bool,
    non_convergence: Optional[Dict[str, Any]],
) -> str:
    """Termination note generated from the REAL recorded failure scope.

    Round 4.1 REPAIR-2 (Codex P1): in the official-code-derived mode the
    note used to unconditionally describe "official power stream
    exhaustion" even when the failing attribute was CATEGORICAL (whose
    search scope is the merge chain).  The note is now derived from the
    mitigation engine's recorded ``search_scope`` so a categorical
    failure is described as such.
    """
    if is_official:
        if termination_reason == "candidate_grid_exhausted":
            scope = (non_convergence or {}).get("search_scope")
            if scope == "categorical_merge_chain":
                return (
                    "official-code-derived mode: the categorical merge "
                    "chain for an attribute (bounded merges plus the "
                    "rejected terminal drop) was exhausted without "
                    "reaching the epsilon ball; there is no iteration "
                    "budget in this mode."
                )
            return (
                "official-code-derived mode: the finite OFFICIAL power "
                "stream was exhausted for a numeric attribute under the "
                "monotone per-attribute stream cursor (no position is "
                "retried) without reaching the epsilon ball; there is no "
                "iteration budget in this mode."
            )
        return (
            "official-code-derived mode: no finite iteration budget "
            "applies; max_iterations is ignored."
        )
    if termination_reason == "candidate_grid_exhausted":
        return (
            "candidate_grid_exhausted means the CONFIGURED transform grid "
            "was exhausted; it is not a paper-level non-convergence claim "
            "(the paper prescribes an increasing-order search without a "
            "stated finite bound)."
        )
    return ""


def run_fairbias_pipeline(config: Optional[FairBiasConfig] = None) -> FairBiasRunResult:
    """
    Execute the complete FairBias benchmarking pipeline.
    """
    start_time = time.time()
    cfg = config or FairBiasConfig.compas_default()

    # Round 4.1: resolve the algorithm mode ONCE so that every downstream
    # component (evaluator, mitigation engine, manifest) sees the concrete
    # effective configuration.  In the official-code-derived mode this
    # fixes the MDS dimension at 2, installs the official interleaved
    # power stream, and marks the run as having NO iteration budget and
    # NO Pareto rollback.
    algorithm_mode = cfg.algorithm_mode
    is_official = algorithm_mode == ALGORITHM_MODE_OFFICIAL
    cfg = cfg.resolved()

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
        preserve_exponent_order=is_official,
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

    # Greedy-loop exit tracking (refined into the public termination
    # record after the loop).  Exit points:
    #   - "no_transform_accepted": nothing applicable this round
    #   - "epsilon_reached": training max d_phi <= epsilon
    #   - "accuracy_threshold_reached": engineering enhancement stop
    #   - "iteration_budget_exhausted": for-loop ran out of max_iterations
    #     (ENGINEERING mode only — the official mode has no budget)
    exit_reason: Optional[str] = None

    # 6. Iterative Mitigation & Enhancement Loop
    # ENGINEERING mode: bounded by cfg.max_iterations.
    # OFFICIAL-CODE-DERIVED mode: NO finite budget — termination is
    # guaranteed by the MONOTONE per-attribute power-stream cursor in
    # the mitigation engine (Round 4.1 REPAIR: each searched stream
    # position is consumed exactly once per numeric attribute, so
    # revisits advance forward-only and can never oscillate between
    # powers) together with the bounded categorical merge chains; the
    # mere finiteness of the official stream is NOT by itself a
    # termination guarantee.
    iteration_budget: Optional[int] = None if is_official else cfg.max_iterations
    iter_idx = 0
    while True:
        if iteration_budget is not None and iter_idx >= iteration_budget:
            exit_reason = "iteration_budget_exhausted"
            break
        iter_idx += 1
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
            # No transform accepted this round: the state is terminal.
            # Under the strict "stop" failure mode this carries the
            # recorded configured-grid failure (highest attribute could
            # not enter the epsilon ball); under a fully mitigated state
            # every attribute is already inside the ball.
            exit_reason = "no_transform_accepted"
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
            exit_reason = "epsilon_reached"
            break
        if cfg.use_accuracy_enhancement and metrics.get("ACC", 0.0) >= acc_threshold:
            exit_reason = "accuracy_threshold_reached"
            break

    if exit_reason is None:
        # The loop ended without an explicit early-exit condition firing.
        # (Only reachable in engineering mode: the official mode has no
        # budget, so its loop always exits via break.)
        exit_reason = "iteration_budget_exhausted"

    # ------------------------------------------------------------------
    # Termination record for the greedy loop.
    # The terminal state is the LAST ACCEPTED transform state (iteration 0
    # = untransformed data when no transform was ever accepted).
    # ------------------------------------------------------------------
    terminal_record = history_iterations[-1] if history_iterations else initial_checkpoint
    terminal_iteration = int(terminal_record["iteration"])
    terminal_max_dphi = float(terminal_record["max_epsilon"])

    if exit_reason == "no_transform_accepted":
        if not cfg.use_bias_mitigation and not cfg.use_accuracy_enhancement:
            termination_reason = "mitigation_disabled"
        elif cfg.use_bias_mitigation and terminal_max_dphi <= epsilon_threshold:
            # No candidate above epsilon: every attribute already sits in
            # the epsilon ball (e.g. the very first step found nothing to
            # mitigate).
            termination_reason = "epsilon_reached"
        else:
            # The configured candidate grid (power exponents / category
            # merge chain / drop) could not bring the highest attribute
            # into the epsilon ball.  This is "configured grid exhausted"
            # — NOT a claim of paper-level algorithmic non-convergence
            # (the paper's power search has no stated finite bound).
            termination_reason = "candidate_grid_exhausted"
    else:
        termination_reason = exit_reason

    if not cfg.use_bias_mitigation:
        # The epsilon-ball convergence criterion is not applicable.
        converged: Optional[bool] = None
    else:
        converged = termination_reason == "epsilon_reached"

    termination_record = {
        "converged": converged,
        "termination_reason": termination_reason,
        "terminal_iteration": terminal_iteration,
        "terminal_max_dphi": terminal_max_dphi,
        "epsilon_threshold": epsilon_threshold,
        "algorithm_mode": algorithm_mode,
        # Auditable distinction: grid exhaustion is an implementation
        # budget outcome, not a paper-level convergence claim.  The note
        # text follows the REAL recorded search_scope (numeric power
        # stream vs categorical merge chain).
        "note": _termination_note(
            termination_reason,
            is_official,
            mitigation_engine.non_convergence if cfg.use_bias_mitigation else None,
        ),
    }

    # 7. Terminal state(s), reported SEPARATELY and never merged.
    #
    # 7a. Greedy termination state — the last ACCEPTED transform state
    # (iteration 0 = untransformed data when no transform was ever
    # accepted), with NO validation-based rollback.  Evaluated on the TEST
    # partition exactly once.
    #   - official-code-derived mode: this is the
    #     official_code_derived_monotone_cursor_unweighted output and the
    #     SOLE reported state.
    #   - engineering mode: this is the configured_greedy_terminal state
    #     (no paper-alignment claim).
    greedy_terminal_changed_dict = copy.deepcopy(terminal_record["changed_dict"])
    greedy_terminal_X_train = transformer.transform_data(
        X_train, greedy_terminal_changed_dict, numerical_cols, categorical_cols
    )
    greedy_terminal_X_test = transformer.transform_data(
        X_test, greedy_terminal_changed_dict, numerical_cols, categorical_cols
    )
    greedy_terminal_metrics = evaluator.evaluate(
        greedy_terminal_X_train,
        Y_train,
        O_train,
        greedy_terminal_X_test,
        Y_test,
        O_test,
    )

    # 7b. pareto_engineering: validation-Pareto-selected checkpoint — an
    # explicitly named ENGINEERING extension.  Selection rule: minimize
    # fairness gap (EO or SP on validation) subject to ACC >= initial
    # validation ACC - tau.  The test partition is NOT consulted during
    # selection.  THE OFFICIAL-CODE-DERIVED MODE HAS NO PARETO STATE: the
    # greedy termination state above is the sole reported output.
    if is_official:
        pareto_changed_dict: Optional[Dict[str, Any]] = None
        pareto_engineering_metrics: Optional[Dict[str, Any]] = None
        best_iter_num = terminal_iteration
        best_reason = (
            "N/A — algorithm_mode='official_code_derived_monotone_"
            "cursor_unweighted' has no validation-Pareto selection; the "
            "greedy termination state is the sole reported state."
        )
    else:
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
        pareto_changed_dict = copy.deepcopy(best_record["changed_dict"])
        best_fairness_val = get_fairness_penalty(best_record)

        best_reason = (
            f"Selected Iteration {best_iter_num} via Pareto Rule on VALIDATION metrics: "
            f"Minimizes {metric_key} ({best_fairness_val:.4f}) under accuracy constraint "
            f"(validation ACC {best_record['metrics'].get('ACC', 0.0):.4f} >= {min_acc_allowed:.4f}). "
            f"Test partition evaluated exactly once afterwards. "
            f"This is the ENGINEERING pareto_engineering state, not the paper "
            f"algorithm's termination state."
        )

        pareto_X_train = transformer.transform_data(
            X_train, pareto_changed_dict, numerical_cols, categorical_cols
        )
        pareto_X_test = transformer.transform_data(
            X_test, pareto_changed_dict, numerical_cols, categorical_cols
        )
        pareto_engineering_metrics = evaluator.evaluate(
            pareto_X_train,
            Y_train,
            O_train,
            pareto_X_test,
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
        "selection_partition": "validation" if not is_official else None,
        "final_evaluation_partition": "test",
        "algorithm_mode": algorithm_mode,
        "algorithm_mode_definition": (
            "official_code_derived_monotone_cursor_unweighted: an "
            "OFFICIAL-CODE-DERIVED VARIANT WITH A TERMINATION-SAFETY "
            "EXTENSION — derived from the official code repository's "
            "behavior, NOT claimed to be behaviorally equivalent to it "
            "and NOT a paper-text method reproduction (the paper text "
            "prescribes elbow-plot MDS dimension selection; MDS fixed "
            "at dim=2 is inherited official-code behavior): official "
            "interleaved power stream [3, 1/3, 5, 1/5, ..., 1999, "
            "1/1999] (order preserved, searched under a monotone "
            "per-attribute stream cursor — a deliberate termination-"
            "safety deviation from the official restart-from-head "
            "search), NO finite iteration budget, NO validation-Pareto "
            "rollback; the greedy termination state is the sole "
            "reported state."
            if is_official else
            "engineering_bounded: automatic stress-elbow MDS dimension, "
            "six-value ascending power grid, finite max_iterations budget, "
            "and a validation-Pareto checkpoint reported as an explicitly "
            "named ENGINEERING extension.  NO paper-alignment claim."
        ),
        "final_states": (
            ["official_code_derived_monotone_cursor_unweighted"]
            if is_official
            else ["configured_greedy_terminal", "pareto_engineering"]
        ),
        "epsilon_threshold": epsilon_threshold,
        "failed_attribute_mode": cfg.failed_attribute_mode,
        "termination": termination_record,
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
        "execution_time_seconds": exec_time,
    }

    if is_official:
        result_payload["final_results_official_code_derived_monotone_cursor_unweighted"] = {
            "metrics": greedy_terminal_metrics,
            "metrics_partition": "test",
            "changed_dict": greedy_terminal_changed_dict,
            "state": (
                "greedy termination state of the official-code-derived "
                "variant with termination-safety extension "
                "(official_code_derived_monotone_cursor_unweighted) — "
                "sole reported state; no validation-Pareto rollback "
                "exists in this mode"
            ),
            "termination": termination_record,
        }
    else:
        result_payload["final_results_configured_greedy_terminal"] = {
            "metrics": greedy_terminal_metrics,
            "metrics_partition": "test",
            "changed_dict": greedy_terminal_changed_dict,
            "state": (
                "termination state of the configured greedy loop (last "
                "accepted transform, no validation rollback); ENGINEERING "
                "configuration — no paper-alignment claim"
            ),
            "termination": termination_record,
        }
        result_payload["final_results_pareto_engineering"] = {
            "metrics": pareto_engineering_metrics,
            "metrics_partition": "test",
            "changed_dict": pareto_changed_dict,
            "best_iteration": best_iter_num,
            "selection_reason": best_reason,
            "state": (
                "validation-Pareto-selected checkpoint; ENGINEERING "
                "extension, not the paper algorithm's termination state"
            ),
        }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(_serialize_object(result_payload), f, indent=2)

    return FairBiasRunResult(
        run_id=run_id,
        config=dataclasses.asdict(cfg),
        algorithm_mode=algorithm_mode,
        initial_metrics=init_metrics,
        initial_epsilon=init_epsilon,
        epsilon_threshold=epsilon_threshold,
        iterations=history_iterations,
        best_iteration=best_iter_num,
        best_selection_reason=best_reason,
        greedy_terminal_metrics=greedy_terminal_metrics,
        greedy_terminal_changed_dict=greedy_terminal_changed_dict,
        pareto_engineering_metrics=pareto_engineering_metrics,
        pareto_engineering_changed_dict=pareto_changed_dict,
        termination=termination_record,
        non_convergence=(
            copy.deepcopy(mitigation_engine.non_convergence)
            if cfg.use_bias_mitigation else None
        ),
        execution_time_seconds=exec_time,
        output_file=str(out_file),
    )
