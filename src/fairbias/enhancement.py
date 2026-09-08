"""Accuracy Enhancement module with bounded fairness degradation, partition contract, and state-bound candidate audit."""

from __future__ import annotations

import collections
import copy
import hashlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fairbias.evaluator import FairEvaluator
from fairbias.enhancement_contracts import (
    CandidateAuditEvent,
    CandidateEvaluationResult,
    EnhancementStatus,
    EvaluationPartition,
    FairnessEvaluationResult,
    compute_configuration_fingerprint,
    evaluate_candidate_utility,
)
from fairbias.enhancement_state import (
    StatefulCandidateTracker,
    hash_transform_state,
    normalize_category_mapping,
    safe_compose_category_mapping,
)
from fairbias.transform import FairTransform, calculate_nmi_dict

DEFAULT_POLY_GRID: Tuple[float, ...] = (1 / 7, 1 / 5, 1 / 3, 3.0, 5.0, 7.0)


class FairAccuracyEnhancement:
    """Explores accuracy-enhancing feature transformations under explicit fairness degradation constraints."""

    def __init__(
        self,
        evaluator: FairEvaluator,
        transformer: FairTransform,
        label_Y: str,
        cate_attrs: List[str],
        num_attrs: List[str],
        max_fairness_degradation: float = 0.02,
        poly_exponents: Sequence[float] = DEFAULT_POLY_GRID,
        min_utility_gain: float = 0.0,
        run_id: str = "default_run",
        arm_id: str = "default_arm",
        condition: str = "default_condition",
    ):
        self.evaluator = evaluator
        self.transformer = transformer
        self.label_Y = label_Y
        self.cate_attrs = list(cate_attrs)
        self.num_attrs = list(num_attrs)

        if (
            max_fairness_degradation is None
            or np.isnan(max_fairness_degradation)
            or np.isinf(max_fairness_degradation)
            or max_fairness_degradation < 0
        ):
            raise ValueError(
                f"Invalid max_fairness_degradation: {max_fairness_degradation}; must be finite and non-negative"
            )
        self.max_fairness_degradation = float(max_fairness_degradation)

        if (
            min_utility_gain is None
            or np.isnan(min_utility_gain)
            or np.isinf(min_utility_gain)
            or min_utility_gain < 0
        ):
            raise ValueError(
                f"Invalid min_utility_gain: {min_utility_gain}; must be finite and non-negative"
            )
        self.min_utility_gain = float(min_utility_gain)

        self.poly_exponents = tuple(float(p) for p in poly_exponents)
        self.run_id = run_id
        self.arm_id = arm_id
        self.condition = condition

        # State and candidate tracking bound to transform states
        self.tracker = StatefulCandidateTracker()
        self.audit_trail: List[CandidateAuditEvent] = []
        self._cached_ranking: Optional[List[str]] = None
        self._ranking_cache: Dict[Tuple[str, str, str], List[str]] = {}
        self.total_model_fits: int = 0
        self.total_geometry_evals: int = 0

        # Legacy compatibility tracking properties
        self._current_skip_attrs: Set[str] = set()
        self._tried_exponents: Dict[str, Set[float]] = collections.defaultdict(set)
        self._tried_rebins: Dict[str, Set[Tuple[Any, Any]]] = collections.defaultdict(set)

    @property
    def skip_attr_list(self) -> Set[str]:
        return self._current_skip_attrs

    @property
    def tried_exponents(self) -> Dict[str, Set[float]]:
        return self._tried_exponents

    @property
    def tried_rebins(self) -> Dict[str, Set[Tuple[Any, Any]]]:
        return self._tried_rebins

    def configuration_fingerprint(self) -> str:
        """Deterministic fingerprint of all configuration settings materially affecting search/evaluation."""
        label_O = getattr(self.evaluator, "label_O", [])
        return compute_configuration_fingerprint(
            config=self.evaluator.config,
            max_fairness_degradation=self.max_fairness_degradation,
            min_utility_gain=self.min_utility_gain,
            poly_exponents=self.poly_exponents,
            label_Y=self.label_Y,
            label_O=label_O,
            cate_attrs=self.cate_attrs,
            num_attrs=self.num_attrs,
            transformer=self.transformer,
        )

    def find_target_correlated_attribute(
        self,
        X: pd.DataFrame,
        Y: pd.Series,
        changed_dict: Dict[str, Any],
        parent_state_hash: Optional[str] = None,
        exhausted_features: Optional[Set[str]] = None,
        partition_fit_fingerprint: Optional[str] = None,
        partition: Optional[EvaluationPartition] = None,
    ) -> Optional[str]:
        """Rank features by mutual information with target Y, returning highest unexhausted active feature."""
        if partition is not None:
            partition.verify_not_mutated()
            X = partition.fit_X
            Y = partition.fit_y
            partition_fit_fingerprint = partition.fit_fingerprint()

        if X.empty or len(Y) == 0:
            return None

        # Key ranking cache by fit partition fingerprint + transform state hash + config fingerprint
        fit_fp = partition_fit_fingerprint or hashlib.sha256(
            pd.util.hash_pandas_object(X, index=True).values.tobytes()
            + pd.util.hash_pandas_object(Y, index=True).values.tobytes()
        ).hexdigest()[:16]
        state_h = parent_state_hash or hash_transform_state(changed_dict)
        cfg_fp = self.configuration_fingerprint()
        cache_key = (fit_fp, state_h, cfg_fp)

        if cache_key not in self._ranking_cache:
            nmi_scores = calculate_nmi_dict(X, Y)
            self._ranking_cache[cache_key] = [
                col for col, _ in sorted(nmi_scores.items(), key=lambda item: item[1], reverse=True)
            ]

        ranking = self._ranking_cache[cache_key]
        self._cached_ranking = ranking

        if exhausted_features is None:
            exhausted_features = self._current_skip_attrs

        for col in ranking:
            # Skip dropped features
            if changed_dict.get(col) == "dropped":
                continue
            # Skip features that have exhausted candidates on this state
            if col in exhausted_features:
                continue
            return col
        return None

    def _is_fairness_acceptable(
        self,
        X_cand: pd.DataFrame,
        O_train: Optional[pd.DataFrame],
        epsilon_threshold: Optional[float],
        current_max_epsilon: Optional[float],
        fairness_guard_enabled: Optional[bool] = None,
    ) -> FairnessEvaluationResult:
        """
        Check whether candidate transformed data stays within fairness degradation bounds.

        Policy: legacy_epsilon_plus_absolute_slack.
        When fairness guard is disabled, returns explicit NOT_EVALUATED status with null candidate_max_dphi/cap.
        """
        guard_enabled = (
            fairness_guard_enabled
            if fairness_guard_enabled is not None
            else ((epsilon_threshold is not None) or (current_max_epsilon is not None))
        )
        if not guard_enabled:
            return FairnessEvaluationResult(
                is_acceptable=True,
                candidate_max_dphi=None,
                cap_applied=None,
                rejection_reason="",
                geometry_eval_count=0,
                evaluation_status=EnhancementStatus.NOT_EVALUATED,
            )

        if O_train is None:
            return FairnessEvaluationResult(
                is_acceptable=False,
                candidate_max_dphi=float("inf"),
                cap_applied=None,
                rejection_reason="MISSING_PROTECTED_DATA_WITH_ENABLED_GUARD",
                geometry_eval_count=0,
                evaluation_status="EVALUATED",
            )

        # Validate max_fairness_degradation
        if (
            self.max_fairness_degradation is None
            or np.isnan(self.max_fairness_degradation)
            or np.isinf(self.max_fairness_degradation)
            or self.max_fairness_degradation < 0
        ):
            return FairnessEvaluationResult(
                is_acceptable=False,
                candidate_max_dphi=float("inf"),
                cap_applied=0.0,
                rejection_reason="INVALID_MAX_FAIRNESS_DEGRADATION",
                geometry_eval_count=0,
            )

        # Validate protected attribute existence
        for p_col in self.evaluator.label_O:
            if p_col not in O_train.columns:
                return FairnessEvaluationResult(
                    is_acceptable=False,
                    candidate_max_dphi=float("inf"),
                    cap_applied=0.0,
                    rejection_reason=f"PROTECTED_ATTRIBUTE_MISSING: {p_col}",
                    geometry_eval_count=0,
                )

        try:
            cand_eps_dict = self.evaluator.calculate_epsilon(
                X_cand, O_train, cate_attrs=self.cate_attrs, num_attrs=self.num_attrs
            )
            self.total_geometry_evals += 1
        except Exception as exc:
            return FairnessEvaluationResult(
                is_acceptable=False,
                candidate_max_dphi=float("inf"),
                cap_applied=0.0,
                rejection_reason=f"GEOMETRY_EVAL_FAILED: {type(exc).__name__}: {exc}",
                geometry_eval_count=1,
            )

        # Check for partial geometry results across active features
        active_features = [f for f in (self.num_attrs + self.cate_attrs) if f in X_cand.columns]
        if not cand_eps_dict:
            return FairnessEvaluationResult(
                is_acceptable=False,
                candidate_max_dphi=float("inf"),
                cap_applied=0.0,
                rejection_reason="EMPTY_EPSILON_RESULTS",
                geometry_eval_count=1,
            )

        cand_all_eps: List[float] = []
        for p_col in self.evaluator.label_O:
            if p_col not in cand_eps_dict:
                return FairnessEvaluationResult(
                    is_acceptable=False,
                    candidate_max_dphi=float("inf"),
                    cap_applied=0.0,
                    rejection_reason="PARTIAL_GEOMETRY_RESULTS",
                    geometry_eval_count=1,
                )
            p_dict = cand_eps_dict[p_col]
            for feat in active_features:
                if feat not in p_dict:
                    return FairnessEvaluationResult(
                        is_acceptable=False,
                        candidate_max_dphi=float("inf"),
                        cap_applied=0.0,
                        rejection_reason="PARTIAL_GEOMETRY_RESULTS",
                        geometry_eval_count=1,
                    )
                val = p_dict[feat]
                if val is None or np.isnan(val) or np.isinf(val) or val < 0:
                    return FairnessEvaluationResult(
                        is_acceptable=False,
                        candidate_max_dphi=float("nan") if (val is not None and np.isnan(val)) else float("inf"),
                        cap_applied=0.0,
                        rejection_reason="NON_FINITE_CANDIDATE_DPHI",
                        geometry_eval_count=1,
                    )
                cand_all_eps.append(float(val))

        if not cand_all_eps:
            return FairnessEvaluationResult(
                is_acceptable=False,
                candidate_max_dphi=float("inf"),
                cap_applied=0.0,
                rejection_reason="EMPTY_EPSILON_RESULTS",
                geometry_eval_count=1,
            )

        cand_max_eps = float(max(cand_all_eps))

        reference_eps = epsilon_threshold if epsilon_threshold is not None else current_max_epsilon
        if reference_eps is None or np.isnan(reference_eps) or np.isinf(reference_eps) or reference_eps < 0:
            return FairnessEvaluationResult(
                is_acceptable=False,
                candidate_max_dphi=cand_max_eps,
                cap_applied=None,
                rejection_reason="INVALID_REFERENCE_EPSILON",
                geometry_eval_count=1,
                evaluation_status="EVALUATED",
            )

        upper_bound = float(reference_eps + self.max_fairness_degradation)
        if np.isinf(upper_bound) or np.isnan(upper_bound):
            return FairnessEvaluationResult(
                is_acceptable=False,
                candidate_max_dphi=cand_max_eps,
                cap_applied=None,
                rejection_reason="INVALID_FAIRNESS_CAP",
                geometry_eval_count=1,
                evaluation_status="EVALUATED",
            )

        is_ok = bool(cand_max_eps <= upper_bound)
        reason = None if is_ok else f"EXCEEDS_FAIRNESS_CAP: {cand_max_eps:.5f} > {upper_bound:.5f}"

        return FairnessEvaluationResult(
            is_acceptable=is_ok,
            candidate_max_dphi=cand_max_eps,
            cap_applied=upper_bound,
            rejection_reason=reason,
            geometry_eval_count=1,
            evaluation_status="EVALUATED",
        )

    def _evaluate_utility(
        self,
        X_train: pd.DataFrame,
        Y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        Y_val: Optional[pd.Series] = None,
        partition: Optional[EvaluationPartition] = None,
        changed_dict: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compatibility wrapper for utility evaluation."""
        if partition is None:
            if X_val is not None and Y_val is not None and len(X_val) > 0:
                partition = EvaluationPartition(fit_X=X_train, fit_y=Y_train, selection_X=X_val, selection_y=Y_val)
            else:
                stratify = Y_train if Y_train.nunique() > 1 else None
                x_tr, x_eval, y_tr, y_eval = train_test_split(
                    X_train, Y_train, test_size=0.3, random_state=getattr(self.evaluator.config, "random_seed", 42), stratify=stratify
                )
                partition = EvaluationPartition(fit_X=x_tr, fit_y=y_tr, selection_X=x_eval, selection_y=y_eval)
        else:
            partition.verify_not_mutated()

        res = evaluate_candidate_utility(
            partition=partition,
            changed_dict=changed_dict or {},
            num_attrs=self.num_attrs,
            cate_attrs=self.cate_attrs,
            transformer=self.transformer,
            evaluator=self.evaluator,
        )
        if not res.is_valid:
            raise RuntimeError(f"Candidate utility invalid: {res.validity_status}: {res.error_message}")
        return float(res.utility_score)

    def enhance_step(
        self,
        X_train: pd.DataFrame,
        Y_train: pd.Series,
        changed_dict: Dict[str, Any],
        O_train: Optional[pd.DataFrame] = None,
        epsilon_threshold: Optional[float] = None,
        current_epsilon: Optional[Dict[str, Dict[str, float]]] = None,
        X_val: Optional[pd.DataFrame] = None,
        Y_val: Optional[pd.Series] = None,
        iteration: int = 1,
        partition: Optional[EvaluationPartition] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[str]]:
        """
        Attempt a fairness-bounded accuracy enhancement step on training data.

        Enforces the candidate evaluation contract:
        - Both fit and selection partitions receive identical transformations.
        - Authoritative EvaluationPartition defines single fit/selection world.
        - Fail-closed validation on budget and partition mismatches.
        - AUROC evaluated on selection partition; invalid evaluations explicit.
        - Transformed state tracked without permanent feature pollution.

        Returns:
            Tuple of (transformed_X_train, new_changed_dict, selected_attribute_or_None).
        """
        # 1. Parameter and budget validation before substantive candidate search
        if (
            self.min_utility_gain is None
            or np.isnan(self.min_utility_gain)
            or np.isinf(self.min_utility_gain)
            or self.min_utility_gain < 0
        ):
            raise ValueError(f"Invalid min_utility_gain: {self.min_utility_gain}; must be finite and non-negative")

        if (
            self.max_fairness_degradation is None
            or np.isnan(self.max_fairness_degradation)
            or np.isinf(self.max_fairness_degradation)
            or self.max_fairness_degradation < 0
        ):
            raise ValueError(
                f"Invalid max_fairness_degradation: {self.max_fairness_degradation}; must be finite and non-negative"
            )

        if epsilon_threshold is not None and (
            np.isnan(epsilon_threshold) or np.isinf(epsilon_threshold) or epsilon_threshold < 0
        ):
            raise ValueError(f"Invalid epsilon_threshold: {epsilon_threshold}; must be finite and non-negative")

        if current_epsilon is not None:
            for gd in current_epsilon.values():
                for v in gd.values():
                    if v is not None and (np.isnan(v) or np.isinf(v) or v < 0):
                        raise ValueError(f"Invalid current_epsilon value: {v}")

        parent_state_hash = hash_transform_state(changed_dict)
        self.tracker.record_state_visit(parent_state_hash)

        # 2. Build or bind authoritative EvaluationPartition
        if partition is not None:
            partition.verify_not_mutated()
            X_train = partition.fit_X
            Y_train = partition.fit_y
            O_train = partition.protected_fit
        else:
            if X_val is not None and Y_val is not None and len(X_val) > 0:
                partition = EvaluationPartition(
                    fit_X=X_train,
                    fit_y=Y_train,
                    selection_X=X_val,
                    selection_y=Y_val,
                    protected_fit=O_train,
                )
            else:
                stratify = Y_train if Y_train.nunique() > 1 else None
                x_tr, x_eval, y_tr, y_eval = train_test_split(
                    X_train,
                    Y_train,
                    test_size=0.3,
                    random_state=getattr(self.evaluator.config, "random_seed", 42),
                    stratify=stratify,
                )
                partition = EvaluationPartition(
                    fit_X=x_tr,
                    fit_y=y_tr,
                    selection_X=x_eval,
                    selection_y=y_eval,
                    protected_fit=O_train.loc[x_tr.index] if O_train is not None else None,
                )
            X_train = partition.fit_X
            Y_train = partition.fit_y
            O_train = partition.protected_fit

        current_df = self.transformer.transform_data(
            X_train, changed_dict, self.num_attrs, self.cate_attrs
        )

        curr_max_eps: Optional[float] = None
        if current_epsilon is not None:
            all_eps = [float(v) for gd in current_epsilon.values() for v in gd.values()]
            curr_max_eps = float(max(all_eps)) if all_eps else None

        base_res = evaluate_candidate_utility(
            partition=partition,
            changed_dict=changed_dict,
            num_attrs=self.num_attrs,
            cate_attrs=self.cate_attrs,
            transformer=self.transformer,
            evaluator=self.evaluator,
        )

        self.total_model_fits += base_res.model_fit_count
        if not base_res.is_valid:
            raise RuntimeError(
                f"Baseline utility evaluation failed ({base_res.validity_status}): {base_res.error_message}"
            )

        current_utility = float(base_res.utility_score)

        # Iterate through candidate attributes
        exhausted_features: Set[str] = set()
        while True:
            target_attr = self.find_target_correlated_attribute(
                partition.fit_X,
                partition.fit_y,
                changed_dict,
                parent_state_hash,
                exhausted_features,
                partition_fit_fingerprint=partition.fit_fingerprint(),
            )
            if target_attr is None:
                return current_df, changed_dict, None

            is_categorical = (
                target_attr in self.cate_attrs
                or not pd.api.types.is_numeric_dtype(X_train[target_attr])
            )

            if is_categorical:
                accepted = self._try_categorical_enhancement(
                    target_attr=target_attr,
                    X_train=X_train,
                    Y_train=Y_train,
                    changed_dict=changed_dict,
                    current_utility=current_utility,
                    O_train=O_train,
                    epsilon_threshold=epsilon_threshold,
                    curr_max_eps=curr_max_eps,
                    partition=partition,
                    parent_state_hash=parent_state_hash,
                    iteration=iteration,
                )
            else:
                accepted = self._try_numerical_enhancement(
                    target_attr=target_attr,
                    X_train=X_train,
                    Y_train=Y_train,
                    changed_dict=changed_dict,
                    current_utility=current_utility,
                    O_train=O_train,
                    epsilon_threshold=epsilon_threshold,
                    curr_max_eps=curr_max_eps,
                    partition=partition,
                    parent_state_hash=parent_state_hash,
                    iteration=iteration,
                )

            if accepted is not None:
                transformed_df, new_changed = accepted
                return transformed_df, new_changed, target_attr

            exhausted_features.add(target_attr)
            self._current_skip_attrs.add(target_attr)

    def _try_numerical_enhancement(
        self,
        target_attr: str,
        X_train: pd.DataFrame,
        Y_train: pd.Series,
        changed_dict: Dict[str, Any],
        current_utility: float,
        O_train: Optional[pd.DataFrame],
        epsilon_threshold: Optional[float],
        curr_max_eps: Optional[float],
        partition: EvaluationPartition,
        parent_state_hash: str,
        iteration: int,
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Test candidate polynomial powers for a numerical feature under the contract."""
        base_power = 1.0
        if target_attr in changed_dict and isinstance(changed_dict[target_attr], dict):
            base_power = float(changed_dict[target_attr].get("power", 1.0))
        self._tried_exponents[target_attr].add(base_power)

        evaluated_candidates: List[Dict[str, Any]] = []

        for power in self.poly_exponents:
            self._tried_exponents[target_attr].add(power)
            cand_sig = f"{target_attr}:power={power:.4f}"
            if self.tracker.is_candidate_evaluated(parent_state_hash, cand_sig):
                continue
            self.tracker.mark_candidate_evaluated(parent_state_hash, cand_sig)

            cand_change = copy.deepcopy(changed_dict)
            cand_change[target_attr] = {"power": power}
            cand_state_hash = hash_transform_state(cand_change)

            # Cycle detection
            if self.tracker.is_cycle(cand_state_hash):
                self._record_audit_event(
                    iteration=iteration,
                    parent_state_hash=parent_state_hash,
                    candidate_state_hash=cand_state_hash,
                    selected_feature=target_attr,
                    proposed_transform={"power": power},
                    partition=partition,
                    utility_before=current_utility,
                    utility_cand=None,
                    utility_gain=None,
                    dphi_before=curr_max_eps,
                    dphi_cand=None,
                    final_eps=epsilon_threshold,
                    cap=None,
                    step_rebound=None,
                    accepted=False,
                    rejection_reason="CYCLE_DETECTED",
                    validity_status=EnhancementStatus.CYCLE_DETECTED,
                    model_fit_count=0,
                    geometry_eval_count=0,
                )
                continue

            if not self.transformer.check_transform_validity(
                X_train, target_attr, cand_change[target_attr], self.num_attrs, self.cate_attrs
            ):
                self._record_audit_event(
                    iteration=iteration,
                    parent_state_hash=parent_state_hash,
                    candidate_state_hash=cand_state_hash,
                    selected_feature=target_attr,
                    proposed_transform={"power": power},
                    partition=partition,
                    utility_before=current_utility,
                    utility_cand=None,
                    utility_gain=None,
                    dphi_before=curr_max_eps,
                    dphi_cand=None,
                    final_eps=epsilon_threshold,
                    cap=None,
                    step_rebound=None,
                    accepted=False,
                    rejection_reason="INVALID_TRANSFORM_SEMANTICS",
                    validity_status=EnhancementStatus.CANDIDATE_INVALID,
                    model_fit_count=0,
                    geometry_eval_count=0,
                )
                continue

            cand_X = self.transformer.transform_data(
                X_train, cand_change, self.num_attrs, self.cate_attrs
            )

            # 1. Check fairness degradation bound first
            fairness_res = self._is_fairness_acceptable(
                cand_X, O_train, epsilon_threshold, curr_max_eps
            )
            step_rebound = (
                (fairness_res.candidate_max_dphi - curr_max_eps)
                if (curr_max_eps is not None and fairness_res.candidate_max_dphi is not None)
                else None
            )

            if not fairness_res.is_acceptable:
                self._record_audit_event(
                    iteration=iteration,
                    parent_state_hash=parent_state_hash,
                    candidate_state_hash=cand_state_hash,
                    selected_feature=target_attr,
                    proposed_transform={"power": power},
                    partition=partition,
                    utility_before=current_utility,
                    utility_cand=None,
                    utility_gain=None,
                    dphi_before=curr_max_eps,
                    dphi_cand=fairness_res.candidate_max_dphi,
                    final_eps=epsilon_threshold,
                    cap=fairness_res.cap_applied,
                    step_rebound=step_rebound,
                    accepted=False,
                    rejection_reason=fairness_res.rejection_reason,
                    validity_status=EnhancementStatus.FAIRNESS_CAP_EXCEEDED,
                    model_fit_count=0,
                    geometry_eval_count=fairness_res.geometry_eval_count,
                )
                continue

            # 2. Evaluate candidate utility
            eval_res = evaluate_candidate_utility(
                partition=partition,
                changed_dict=cand_change,
                num_attrs=self.num_attrs,
                cate_attrs=self.cate_attrs,
                transformer=self.transformer,
                evaluator=self.evaluator,
            )
            self.total_model_fits += eval_res.model_fit_count

            if not eval_res.is_valid:
                self._record_audit_event(
                    iteration=iteration,
                    parent_state_hash=parent_state_hash,
                    candidate_state_hash=cand_state_hash,
                    selected_feature=target_attr,
                    proposed_transform={"power": power},
                    partition=partition,
                    utility_before=current_utility,
                    utility_cand=None,
                    utility_gain=None,
                    dphi_before=curr_max_eps,
                    dphi_cand=fairness_res.candidate_max_dphi,
                    final_eps=epsilon_threshold,
                    cap=fairness_res.cap_applied,
                    step_rebound=step_rebound,
                    accepted=False,
                    rejection_reason=eval_res.error_message,
                    validity_status=eval_res.validity_status,
                    model_fit_count=eval_res.model_fit_count,
                    geometry_eval_count=fairness_res.geometry_eval_count,
                )
                continue

            cand_utility = float(eval_res.utility_score)
            gain = cand_utility - current_utility

            evaluated_candidates.append({
                "cand_X": cand_X,
                "cand_change": cand_change,
                "cand_state_hash": cand_state_hash,
                "proposed_transform": {"power": power},
                "cand_utility": cand_utility,
                "gain": gain,
                "fairness_res": fairness_res,
                "eval_res": eval_res,
                "step_rebound": step_rebound,
            })

        if not evaluated_candidates:
            return None

        # Find best candidate by strictly highest utility gain
        best_idx = -1
        best_gain = self.min_utility_gain
        for idx, item in enumerate(evaluated_candidates):
            if item["gain"] > best_gain:
                best_gain = item["gain"]
                best_idx = idx

        best_cand: Optional[Tuple[pd.DataFrame, Dict[str, Any]]] = None
        for idx, item in enumerate(evaluated_candidates):
            if idx == best_idx:
                is_accepted = True
                rejection_reason = None
                best_cand = (item["cand_X"], item["cand_change"])
            elif item["gain"] > self.min_utility_gain:
                is_accepted = False
                rejection_reason = "ELIGIBLE_NOT_COMMITTED"
            else:
                is_accepted = False
                rejection_reason = f"INSUFFICIENT_GAIN: {item['gain']:.5f} <= {self.min_utility_gain:.5f}"

            self._record_audit_event(
                iteration=iteration,
                parent_state_hash=parent_state_hash,
                candidate_state_hash=item["cand_state_hash"],
                selected_feature=target_attr,
                proposed_transform=item["proposed_transform"],
                partition=partition,
                utility_before=current_utility,
                utility_cand=item["cand_utility"],
                utility_gain=item["gain"],
                dphi_before=curr_max_eps,
                dphi_cand=item["fairness_res"].candidate_max_dphi,
                final_eps=epsilon_threshold,
                cap=item["fairness_res"].cap_applied,
                step_rebound=item["step_rebound"],
                accepted=is_accepted,
                rejection_reason=rejection_reason,
                validity_status=EnhancementStatus.VALID,
                model_fit_count=item["eval_res"].model_fit_count,
                geometry_eval_count=item["fairness_res"].geometry_eval_count,
            )

        return best_cand

    def _try_categorical_enhancement(
        self,
        target_attr: str,
        X_train: pd.DataFrame,
        Y_train: pd.Series,
        changed_dict: Dict[str, Any],
        current_utility: float,
        O_train: Optional[pd.DataFrame],
        epsilon_threshold: Optional[float],
        curr_max_eps: Optional[float],
        partition: EvaluationPartition,
        parent_state_hash: str,
        iteration: int,
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Test category merging on the CURRENT transformed state under the contract."""
        curr_t_train = self.transformer.transform_data(
            X_train, changed_dict, self.num_attrs, self.cate_attrs
        )
        s = curr_t_train[target_attr]
        unique_cats = list(s.unique())
        if len(unique_cats) < 3:
            return None

        # Compute positive rates on fit partition of current transformed representation
        fit_t = self.transformer.transform_data(
            partition.fit_X, changed_dict, self.num_attrs, self.cate_attrs
        )
        s_fit = fit_t[target_attr]
        cat_target_rates = partition.fit_y.groupby(s_fit).mean().to_dict()
        sorted_cats = sorted(cat_target_rates.items(), key=lambda x: x[1])

        # Generate adjacent category pairs
        candidate_pairs = []
        for i in range(len(sorted_cats) - 1):
            pair = (sorted_cats[i][0], sorted_cats[i + 1][0])
            cand_sig = f"{target_attr}:merge({pair[1]}->{pair[0]})"
            norm_pair = (min(str(pair[0]), str(pair[1])), max(str(pair[0]), str(pair[1])))
            self._tried_rebins[target_attr].add(norm_pair)
            if not self.tracker.is_candidate_evaluated(parent_state_hash, cand_sig):
                diff = abs(sorted_cats[i][1] - sorted_cats[i + 1][1])
                candidate_pairs.append((pair, cand_sig, diff))

        candidate_pairs.sort(key=lambda x: x[2])

        evaluated_candidates: List[Dict[str, Any]] = []

        for pair, cand_sig, _ in candidate_pairs:
            self.tracker.mark_candidate_evaluated(parent_state_hash, cand_sig)
            rebin = {pair[1]: pair[0]}

            cand_change = copy.deepcopy(changed_dict)
            try:
                if target_attr in cand_change and isinstance(cand_change[target_attr], dict):
                    cand_change[target_attr] = safe_compose_category_mapping(
                        cand_change[target_attr], rebin, col_sample=X_train[target_attr]
                    )
                else:
                    cand_change[target_attr] = normalize_category_mapping(
                        rebin, col_sample=X_train[target_attr]
                    )
            except (ValueError, TypeError) as exc:
                # Expected category mapping/normalization conflict logged as explicit rejected audit event
                cand_state_hash = hash_transform_state({target_attr: rebin, "__parent__": parent_state_hash})
                self._record_audit_event(
                    iteration=iteration,
                    parent_state_hash=parent_state_hash,
                    candidate_state_hash=cand_state_hash,
                    selected_feature=target_attr,
                    proposed_transform=rebin,
                    partition=partition,
                    utility_before=current_utility,
                    utility_cand=None,
                    utility_gain=None,
                    dphi_before=curr_max_eps,
                    dphi_cand=None,
                    final_eps=epsilon_threshold,
                    cap=None,
                    step_rebound=None,
                    accepted=False,
                    rejection_reason=f"CATEGORY_MAPPING_CONFLICT: {type(exc).__name__}: {exc}",
                    validity_status=EnhancementStatus.CATEGORY_MAPPING_INVALID,
                    model_fit_count=0,
                    geometry_eval_count=0,
                )
                continue

            cand_state_hash = hash_transform_state(cand_change)

            # Cycle detection
            if self.tracker.is_cycle(cand_state_hash):
                self._record_audit_event(
                    iteration=iteration,
                    parent_state_hash=parent_state_hash,
                    candidate_state_hash=cand_state_hash,
                    selected_feature=target_attr,
                    proposed_transform=rebin,
                    partition=partition,
                    utility_before=current_utility,
                    utility_cand=None,
                    utility_gain=None,
                    dphi_before=curr_max_eps,
                    dphi_cand=None,
                    final_eps=epsilon_threshold,
                    cap=None,
                    step_rebound=None,
                    accepted=False,
                    rejection_reason="CYCLE_DETECTED",
                    validity_status=EnhancementStatus.CYCLE_DETECTED,
                    model_fit_count=0,
                    geometry_eval_count=0,
                )
                continue

            if not self.transformer.check_transform_validity(
                X_train, target_attr, cand_change[target_attr], self.num_attrs, self.cate_attrs
            ):
                self._record_audit_event(
                    iteration=iteration,
                    parent_state_hash=parent_state_hash,
                    candidate_state_hash=cand_state_hash,
                    selected_feature=target_attr,
                    proposed_transform=rebin,
                    partition=partition,
                    utility_before=current_utility,
                    utility_cand=None,
                    utility_gain=None,
                    dphi_before=curr_max_eps,
                    dphi_cand=None,
                    final_eps=epsilon_threshold,
                    cap=None,
                    step_rebound=None,
                    accepted=False,
                    rejection_reason="INVALID_TRANSFORM_SEMANTICS",
                    validity_status=EnhancementStatus.CANDIDATE_INVALID,
                    model_fit_count=0,
                    geometry_eval_count=0,
                )
                continue

            cand_X = self.transformer.transform_data(
                X_train, cand_change, self.num_attrs, self.cate_attrs
            )

            fairness_res = self._is_fairness_acceptable(
                cand_X, O_train, epsilon_threshold, curr_max_eps
            )
            step_rebound = (
                (fairness_res.candidate_max_dphi - curr_max_eps)
                if (curr_max_eps is not None and fairness_res.candidate_max_dphi is not None)
                else None
            )

            if not fairness_res.is_acceptable:
                self._record_audit_event(
                    iteration=iteration,
                    parent_state_hash=parent_state_hash,
                    candidate_state_hash=cand_state_hash,
                    selected_feature=target_attr,
                    proposed_transform=rebin,
                    partition=partition,
                    utility_before=current_utility,
                    utility_cand=None,
                    utility_gain=None,
                    dphi_before=curr_max_eps,
                    dphi_cand=fairness_res.candidate_max_dphi,
                    final_eps=epsilon_threshold,
                    cap=fairness_res.cap_applied,
                    step_rebound=step_rebound,
                    accepted=False,
                    rejection_reason=fairness_res.rejection_reason,
                    validity_status=EnhancementStatus.FAIRNESS_CAP_EXCEEDED,
                    model_fit_count=0,
                    geometry_eval_count=fairness_res.geometry_eval_count,
                )
                continue

            eval_res = evaluate_candidate_utility(
                partition=partition,
                changed_dict=cand_change,
                num_attrs=self.num_attrs,
                cate_attrs=self.cate_attrs,
                transformer=self.transformer,
                evaluator=self.evaluator,
            )
            self.total_model_fits += eval_res.model_fit_count

            if not eval_res.is_valid:
                self._record_audit_event(
                    iteration=iteration,
                    parent_state_hash=parent_state_hash,
                    candidate_state_hash=cand_state_hash,
                    selected_feature=target_attr,
                    proposed_transform=rebin,
                    partition=partition,
                    utility_before=current_utility,
                    utility_cand=None,
                    utility_gain=None,
                    dphi_before=curr_max_eps,
                    dphi_cand=fairness_res.candidate_max_dphi,
                    final_eps=epsilon_threshold,
                    cap=fairness_res.cap_applied,
                    step_rebound=step_rebound,
                    accepted=False,
                    rejection_reason=eval_res.error_message,
                    validity_status=eval_res.validity_status,
                    model_fit_count=eval_res.model_fit_count,
                    geometry_eval_count=fairness_res.geometry_eval_count,
                )
                continue

            cand_utility = float(eval_res.utility_score)
            gain = cand_utility - current_utility

            evaluated_candidates.append({
                "cand_X": cand_X,
                "cand_change": cand_change,
                "cand_state_hash": cand_state_hash,
                "proposed_transform": rebin,
                "cand_utility": cand_utility,
                "gain": gain,
                "fairness_res": fairness_res,
                "eval_res": eval_res,
                "step_rebound": step_rebound,
            })

        if not evaluated_candidates:
            return None

        best_idx = -1
        best_gain = self.min_utility_gain
        for idx, item in enumerate(evaluated_candidates):
            if item["gain"] > best_gain:
                best_gain = item["gain"]
                best_idx = idx

        best_cand: Optional[Tuple[pd.DataFrame, Dict[str, Any]]] = None
        for idx, item in enumerate(evaluated_candidates):
            if idx == best_idx:
                is_accepted = True
                rejection_reason = None
                best_cand = (item["cand_X"], item["cand_change"])
            elif item["gain"] > self.min_utility_gain:
                is_accepted = False
                rejection_reason = "ELIGIBLE_NOT_COMMITTED"
            else:
                is_accepted = False
                rejection_reason = f"INSUFFICIENT_GAIN: {item['gain']:.5f} <= {self.min_utility_gain:.5f}"

            self._record_audit_event(
                iteration=iteration,
                parent_state_hash=parent_state_hash,
                candidate_state_hash=item["cand_state_hash"],
                selected_feature=target_attr,
                proposed_transform=item["proposed_transform"],
                partition=partition,
                utility_before=current_utility,
                utility_cand=item["cand_utility"],
                utility_gain=item["gain"],
                dphi_before=curr_max_eps,
                dphi_cand=item["fairness_res"].candidate_max_dphi,
                final_eps=epsilon_threshold,
                cap=item["fairness_res"].cap_applied,
                step_rebound=item["step_rebound"],
                accepted=is_accepted,
                rejection_reason=rejection_reason,
                validity_status=EnhancementStatus.VALID,
                model_fit_count=item["eval_res"].model_fit_count,
                geometry_eval_count=item["fairness_res"].geometry_eval_count,
            )

        return best_cand

    def _record_audit_event(
        self,
        iteration: int,
        parent_state_hash: str,
        candidate_state_hash: str,
        selected_feature: str,
        proposed_transform: Any,
        partition: EvaluationPartition,
        utility_before: Optional[float],
        utility_cand: Optional[float],
        utility_gain: Optional[float],
        dphi_before: Optional[float],
        dphi_cand: Optional[float],
        final_eps: Optional[float],
        cap: Optional[float],
        step_rebound: Optional[float],
        accepted: bool,
        rejection_reason: Optional[str],
        validity_status: str,
        model_fit_count: int,
        geometry_eval_count: int,
    ) -> None:
        """Create and store an immutable audit event."""
        config_hash = self.configuration_fingerprint()

        event = CandidateAuditEvent(
            run_id=self.run_id,
            arm_id=self.arm_id,
            condition=self.condition,
            iteration=iteration,
            engine="AE",
            parent_state_hash=parent_state_hash,
            candidate_state_hash=candidate_state_hash,
            selected_feature=selected_feature,
            proposed_transform=proposed_transform,
            fit_partition_fingerprint=partition.fit_fingerprint(),
            selection_partition_fingerprint=partition.selection_fingerprint(),
            utility_metric="AUROC",
            utility_before=utility_before,
            utility_candidate=utility_cand,
            utility_gain=utility_gain,
            train_max_dphi_before=dphi_before,
            train_max_dphi_candidate=dphi_cand,
            final_epsilon=final_eps,
            effective_candidate_cap=cap,
            step_rebound=step_rebound,
            accepted=accepted,
            rejection_reason=rejection_reason,
            validity_status=validity_status,
            model_fit_count=model_fit_count,
            geometry_eval_count=geometry_eval_count,
            config_hash=config_hash,
        )
        self.audit_trail.append(event)
