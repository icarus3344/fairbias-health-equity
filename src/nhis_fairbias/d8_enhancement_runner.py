"""Exploratory Study D8: Accuracy Enhancement evaluation on the four NHIS temporal arms.

Evaluates whether fairness-bounded Accuracy Enhancement (AE) can alleviate or reverse
downstream classification utility collapse (AUROC/AUPRC loss and prediction suppression)
observed in Gate D6 / D7 across the four frozen arms:
- D6_ARM_001 (SEX_A, 21 predictors)
- D6_ARM_002 (HISPALLP_A, 21 predictors)
- D6_ARM_003 (DISAB3_A, 21 predictors)
- D6_ARM_004 (DISAB3_A, 15 predictors)

Adheres to D8-R1 repair contracts:
- Unified train/selection transformation contract (identical transform, scaling fit on train only)
- Dependency injection for adapter and canonical provider (sentinel prevents unintended parquet access)
- Explicit stopping reasons, dual event logging, candidate audit trail, and collision-free outputs.
"""

from __future__ import annotations

import copy
import json
import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    auc,
)
from sklearn.preprocessing import MinMaxScaler

import sklearn.base

from fairbias.config import (
    ALGORITHM_MODE_ENGINEERING,
    ALGORITHM_MODE_PAPER_FAITHFUL,
    FairBiasConfig,
)
from fairbias.evaluator import FairEvaluator
from fairbias.enhancement import FairAccuracyEnhancement
from fairbias.enhancement_contracts import (
    CandidateAuditEvent,
    EnhancementStatus,
    EvaluationPartition,
)
from fairbias.enhancement_state import changed_dict_hash
from fairbias.mitigation import FairBiasMitigation
from fairbias.transform import (
    FairTransform,
    calculate_nmi_dict,
)
from nhis_fairbias.d6_temporal_runner import FROZEN_D6_ARMS

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FEATURES_PARQUET_PATH = REPO_ROOT / "data" / "processed" / "nhis" / "nhis_2022_2024_features.parquet"
D6_RELEASE_DIR = REPO_ROOT / "docs" / "releases" / "NHIS_D6_TEMPORAL_TEST_V1_b2fd84e7"


def compute_group_fairness_gaps(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_vals: np.ndarray,
) -> Dict[str, float]:
    """Compute demographic parity difference and equal opportunity difference."""
    unique_groups = np.unique(protected_vals)
    if len(unique_groups) < 2:
        return {"demographic_parity_difference": 0.0, "equal_opportunity_difference": 0.0}

    selection_rates = []
    tprs = []

    for g in unique_groups:
        mask_g = (protected_vals == g)
        if np.sum(mask_g) > 0:
            selection_rates.append(float(np.mean(y_pred[mask_g])))
        pos_mask = mask_g & (y_true == 1)
        if np.sum(pos_mask) > 0:
            tprs.append(float(np.mean(y_pred[pos_mask])))

    dp_diff = float(max(selection_rates) - min(selection_rates)) if selection_rates else 0.0
    eo_diff = float(max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0

    return {
        "demographic_parity_difference": dp_diff,
        "equal_opportunity_difference": eo_diff,
    }


def evaluate_representation(
    model: Optional[Any],
    scaler: Optional[MinMaxScaler],
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_val_raw: pd.DataFrame,
    y_val: np.ndarray,
    X_test_raw: pd.DataFrame,
    y_test: np.ndarray,
    o_train: np.ndarray,
    o_val: np.ndarray,
    o_test: np.ndarray,
    changed_dict: Dict[str, Any],
    evaluator: FairEvaluator,
    transformer: FairTransform,
    cate_attrs: List[str],
    num_attrs: List[str],
    protected_attr: str,
) -> Dict[str, Any]:
    """Transform features, fit model on train, and evaluate utility + fairness across train/val/test."""
    # Transform partitions identically
    X_tr_t = transformer.transform_data(X_train_raw, changed_dict, num_attrs, cate_attrs)
    X_val_t = transformer.transform_data(X_val_raw, changed_dict, num_attrs, cate_attrs)
    X_te_t = transformer.transform_data(X_test_raw, changed_dict, num_attrs, cate_attrs)

    if X_tr_t.shape[1] == 0 or X_val_t.shape[1] == 0 or X_te_t.shape[1] == 0:
        raise ValueError("Cannot evaluate representation: all features dropped")

    # Enforce model-ready numeric representation contract
    for df, name in [(X_tr_t, "train"), (X_val_t, "val"), (X_te_t, "test")]:
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(
                    f"Non-numeric representation encountered in {name} column '{col}' with dtype '{df[col].dtype}'. "
                    "Features must have numeric representation prior to evaluation."
                )

    # Validate target label diversity
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("Single class encountered in target labels during evaluation")

    # Model resolution
    if model is not None:
        try:
            model_inst = sklearn.base.clone(model)
        except Exception:
            model_inst = copy.deepcopy(model)
    elif hasattr(evaluator, "model"):
        model_inst = sklearn.base.clone(evaluator.model)
    else:
        model_inst = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)

    # Require probabilistic predictor
    if not hasattr(model_inst, "predict_proba"):
        raise ValueError(
            f"{EnhancementStatus.MISSING_PROBABILITIES}: Model does not support predict_proba; "
            "probabilistic predictor required for utility evaluation."
        )

    # Scaler resolution
    if scaler is not None:
        try:
            scaler_inst = sklearn.base.clone(scaler)
        except Exception:
            scaler_inst = copy.deepcopy(scaler)
    elif hasattr(evaluator, "_get_scaler"):
        scaler_inst = evaluator._get_scaler()
    else:
        scaler_inst = MinMaxScaler(feature_range=(0, 1))

    # Scale strictly on train
    X_tr_s = scaler_inst.fit_transform(X_tr_t)
    X_val_s = scaler_inst.transform(X_val_t)
    X_te_s = scaler_inst.transform(X_te_t)

    # Fit model on train
    model_inst.fit(X_tr_s, y_train)

    # Predictions
    proba_tr = model_inst.predict_proba(X_tr_s)
    proba_val = model_inst.predict_proba(X_val_s)
    proba_te = model_inst.predict_proba(X_te_s)

    if proba_tr.shape[1] < 2 or proba_val.shape[1] < 2 or proba_te.shape[1] < 2:
        raise ValueError("Model predicted fewer than 2 probability classes")

    p_tr = proba_tr[:, 1]
    p_val = proba_val[:, 1]
    p_te = proba_te[:, 1]

    if np.isnan(p_tr).any() or np.isnan(p_val).any() or np.isnan(p_te).any():
        raise ValueError("NaN probabilities encountered during prediction")

    pred_tr = (p_tr >= 0.5).astype(int)
    pred_val = (p_val >= 0.5).astype(int)
    pred_te = (p_te >= 0.5).astype(int)

    # AUPRC helpers: trapezoidal AUC vs Average Precision
    def get_auprc_trap(y_t, p_s):
        prec, rec, _ = precision_recall_curve(y_t, p_s)
        return float(auc(rec, prec))

    def get_avg_prec(y_t, p_s):
        return float(average_precision_score(y_t, p_s))

    # Evaluate fairness epsilon
    O_tr_df = pd.DataFrame({protected_attr: o_train}, index=X_train_raw.index)
    O_val_df = pd.DataFrame({protected_attr: o_val}, index=X_val_raw.index)
    O_te_df = pd.DataFrame({protected_attr: o_test}, index=X_test_raw.index)

    eps_tr = evaluator.calculate_epsilon(X_tr_t, O_tr_df, cate_attrs=cate_attrs, num_attrs=num_attrs)
    eps_val = evaluator.calculate_epsilon(X_val_t, O_val_df, cate_attrs=cate_attrs, num_attrs=num_attrs)
    eps_te = evaluator.calculate_epsilon(X_te_t, O_te_df, cate_attrs=cate_attrs, num_attrs=num_attrs)

    def get_max_eps(ed):
        vals = [float(v) for gd in ed.values() for v in gd.values()]
        return float(max(vals)) if vals else 0.0

    # Group fairness gaps on validation and test
    val_gaps = compute_group_fairness_gaps(y_val, pred_val, o_val)
    test_gaps = compute_group_fairness_gaps(y_test, pred_te, o_test)

    return {
        "terminal_evaluation_performed": True,
        "changed_dict": copy.deepcopy(changed_dict),
        "num_transforms": len(changed_dict),
        "train": {
            "N": int(len(X_train_raw)),
            "n": int(len(X_train_raw)),
            "auroc": float(roc_auc_score(y_train, p_tr)),
            "auprc": get_avg_prec(y_train, p_tr),
            "auprc_trapezoidal": get_auprc_trap(y_train, p_tr),
            "average_precision": get_avg_prec(y_train, p_tr),
            "brier": float(brier_score_loss(y_train, p_tr)),
            "accuracy": float(accuracy_score(y_train, pred_tr)),
            "balanced_accuracy": float(balanced_accuracy_score(y_train, pred_tr)),
            "f1": float(f1_score(y_train, pred_tr, zero_division=0)),
            "predicted_positive_count": int(np.sum(pred_tr)),
            "count_predicted_positive": int(np.sum(pred_tr)),
            "predicted_positive_rate": float(np.mean(pred_tr)),
            "selection_rate": float(np.mean(pred_tr)),
            "count_outcome_positive": int(np.sum(y_train == 1)),
            "prevalence": float(np.mean(y_train == 1)),
            "max_dphi": get_max_eps(eps_tr),
        },
        "validation": {
            "N": int(len(X_val_raw)),
            "n": int(len(X_val_raw)),
            "auroc": float(roc_auc_score(y_val, p_val)),
            "auprc": get_avg_prec(y_val, p_val),
            "auprc_trapezoidal": get_auprc_trap(y_val, p_val),
            "average_precision": get_avg_prec(y_val, p_val),
            "brier": float(brier_score_loss(y_val, p_val)),
            "accuracy": float(accuracy_score(y_val, pred_val)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, pred_val)),
            "f1": float(f1_score(y_val, pred_val, zero_division=0)),
            "predicted_positive_count": int(np.sum(pred_val)),
            "count_predicted_positive": int(np.sum(pred_val)),
            "predicted_positive_rate": float(np.mean(pred_val)),
            "selection_rate": float(np.mean(pred_val)),
            "count_outcome_positive": int(np.sum(y_val == 1)),
            "prevalence": float(np.mean(y_val == 1)),
            "max_dphi": get_max_eps(eps_val),
            "demographic_parity_difference": val_gaps["demographic_parity_difference"],
            "demographic_parity_gap": val_gaps["demographic_parity_difference"],
            "equal_opportunity_difference": val_gaps["equal_opportunity_difference"],
            "equal_opportunity_gap": val_gaps["equal_opportunity_difference"],
        },
        "test": {
            "N": int(len(X_test_raw)),
            "n": int(len(X_test_raw)),
            "auroc": float(roc_auc_score(y_test, p_te)),
            "auprc": get_avg_prec(y_test, p_te),
            "auprc_trapezoidal": get_auprc_trap(y_test, p_te),
            "average_precision": get_avg_prec(y_test, p_te),
            "brier": float(brier_score_loss(y_test, p_te)),
            "accuracy": float(accuracy_score(y_test, pred_te)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred_te)),
            "f1": float(f1_score(y_test, pred_te, zero_division=0)),
            "predicted_positive_count": int(np.sum(pred_te)),
            "count_predicted_positive": int(np.sum(pred_te)),
            "predicted_positive_rate": float(np.mean(pred_te)),
            "selection_rate": float(np.mean(pred_te)),
            "count_outcome_positive": int(np.sum(y_test == 1)),
            "prevalence": float(np.mean(y_test == 1)),
            "max_dphi": get_max_eps(eps_te),
            "demographic_parity_difference": test_gaps["demographic_parity_difference"],
            "demographic_parity_gap": test_gaps["demographic_parity_difference"],
            "equal_opportunity_difference": test_gaps["equal_opportunity_difference"],
            "equal_opportunity_gap": test_gaps["equal_opportunity_difference"],
        },
    }


class D8EnhancementRunner:
    """Orchestrates the Accuracy Enhancement evaluation across the 4 frozen NHIS arms."""

    def __init__(
        self,
        adapter: Optional[Any] = None,
        canonical_provider: Optional[Callable[[str], Dict[str, Any]]] = None,
        smoke_test: bool = False,
        random_seed: int = 0,
        run_id: str = "d8_study",
        allow_real_data: bool = False,
        baseline_reproduction_only: bool = False,
    ):
        self.smoke_test = smoke_test
        self.random_seed = random_seed
        self.run_id = run_id
        self.allow_real_data = allow_real_data
        self.baseline_reproduction_only = baseline_reproduction_only
        self.canonical_provider = canonical_provider
        self._adapter = adapter
        self.audit_events: List[CandidateAuditEvent] = []

        if self.allow_real_data and not self.baseline_reproduction_only:
            raise RuntimeError(
                "Access to real NHIS microdata is restricted to --baseline-reproduction-only mode during Gate D8-R2. "
                "Full four-condition enhancement on real data is NOT authorized."
            )

    @property
    def adapter(self) -> Any:
        if self._adapter is not None:
            return self._adapter
        if not self.allow_real_data:
            raise RuntimeError(
                "Access to real NHIS parquet is prohibited without explicit allow_real_data authorization. "
                "In synthetic verification, an explicit adapter must be injected into D8EnhancementRunner."
            )
        from nhis_fairbias.adapter import NHISStudyAdapter
        self._adapter = NHISStudyAdapter(features_parquet_path=FEATURES_PARQUET_PATH)
        return self._adapter

    def load_canonical_d6_changed_dict(self, arm_id: str) -> Dict[str, Any]:
        """Load frozen canonical FairBias changed_dict from the D6 release or injected provider."""
        if self.canonical_provider is not None:
            return self.canonical_provider(arm_id)
        dict_path = D6_RELEASE_DIR / arm_id / "frozen_changed_dict.json"
        if not dict_path.exists():
            raise FileNotFoundError(f"D6 canonical changed_dict not found at {dict_path}")
        with open(dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["changed_dict"]

    def run_arm(self, arm_id: str) -> Dict[str, Any]:
        """Execute all 4 conditions on a single arm and return comparative evaluation."""
        if arm_id not in FROZEN_D6_ARMS:
            raise ValueError(f"Unknown arm: {arm_id}")
        arm_meta = FROZEN_D6_ARMS[arm_id]
        outcome = arm_meta["outcome"]
        protected_attr = arm_meta["protected_attribute"]
        disability_arm = arm_meta["disability_arm"]
        feature_set = arm_meta["feature_set"].lower()

        # 1. Load cohorts via adapter (must be injected in synthetic tests)
        X_train, y_train_s, o_train_s, _, _ = self.adapter.get_cohort(
            year=2022, outcome=outcome, protected_attribute=protected_attr,
            feature_set=feature_set, disability_arm=disability_arm
        )
        X_val, y_val_s, o_val_s, _, _ = self.adapter.get_cohort(
            year=2023, outcome=outcome, protected_attribute=protected_attr,
            feature_set=feature_set, disability_arm=disability_arm
        )
        X_test, y_test_s, o_test_s, _, _ = self.adapter.get_cohort(
            year=2024, outcome=outcome, protected_attribute=protected_attr,
            feature_set=feature_set, disability_arm=disability_arm
        )

        if self.smoke_test:
            n_tr = min(500, len(X_train))
            n_eval = min(300, len(X_val))
            X_train = X_train.iloc[:n_tr]
            y_train_s = y_train_s.iloc[:n_tr]
            o_train_s = o_train_s.iloc[:n_tr]
            X_val = X_val.iloc[:n_eval]
            y_val_s = y_val_s.iloc[:n_eval]
            o_val_s = o_val_s.iloc[:n_eval]
            X_test = X_test.iloc[:min(300, len(X_test))]
            y_test_s = y_test_s.iloc[:min(300, len(y_test_s))]
            o_test_s = o_test_s.iloc[:min(300, len(o_test_s))]

        y_tr = y_train_s.values.astype(int)
        y_v = y_val_s.values.astype(int)
        y_te = y_test_s.values.astype(int)
        o_tr = o_train_s.values
        o_v = o_val_s.values
        o_te = o_test_s.values

        # Feature groupings
        all_cats, all_nums = self.adapter.preprocessor.get_feature_family_lists(feature_set)
        active_cols = set(X_train.columns)
        cate_attrs = [c for c in all_cats if c in active_cols]
        num_attrs = [c for c in all_nums if c in active_cols]

        # Observed cohort sizes and outcome-positive counts directly from actual partitions
        observed_cohort = {
            "train_n": int(len(X_train)),
            "train_positives": int(np.sum(y_tr == 1)),
            "validation_n": int(len(X_val)),
            "validation_positives": int(np.sum(y_v == 1)),
            "test_n": int(len(X_test)),
            "test_positives": int(np.sum(y_te == 1)),
        }

        # Observed schema from actual execution configuration and data frame
        observed_schema = {
            "disability_arm": str(disability_arm),
            "feature_count": int(len(X_train.columns)),
            "feature_order": list(X_train.columns),
            "categorical_features": list(cate_attrs),
            "numerical_features": list(num_attrs),
            "outcome": str(outcome),
            "protected_attribute": str(protected_attr),
        }

        if self.baseline_reproduction_only:
            fb_config = FairBiasConfig(
                algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL,
                random_seed=self.random_seed,
                classifier="LR",
                eval_norm="min-max",
                label_O=(protected_attr,),
                label_Y=outcome,
                use_bias_mitigation=True,
                use_accuracy_enhancement=False,
                failed_attribute_mode="stop",
            ).resolved()
        else:
            fb_config = FairBiasConfig(
                algorithm_mode=ALGORITHM_MODE_ENGINEERING,
                random_seed=self.random_seed,
                classifier="LR",
                eval_norm="min-max",
                label_O=(protected_attr,),
                label_Y=outcome,
                use_bias_mitigation=True,
                use_accuracy_enhancement=True,
                failed_attribute_mode="stop",
                mds_fixed_components=2,
            ).resolved()

        evaluator = FairEvaluator(
            config=fb_config,
            label_O=[protected_attr],
            label_Y=outcome,
            cate_attrs=cate_attrs,
            num_attrs=num_attrs,
        )
        transformer = FairTransform(
            n_bins=fb_config.transform_n_bins,
            log_epsilon=fb_config.transform_log_epsilon,
            x_max=fb_config.transform_x_max,
        )

        O_tr_df = pd.DataFrame({protected_attr: o_tr}, index=X_train.index)
        O_val_df = pd.DataFrame({protected_attr: o_v}, index=X_val.index)
        init_eps_dict = evaluator.calculate_epsilon(X_train, O_tr_df, cate_attrs=cate_attrs, num_attrs=num_attrs)
        eps_thresh = float(evaluator.compute_threshold(init_eps_dict))

        # Build evaluation partition for training / validation search
        partition = EvaluationPartition(
            fit_X=X_train,
            fit_y=y_train_s,
            selection_X=X_val,
            selection_y=y_val_s,
            protected_fit=O_tr_df,
            protected_selection=O_val_df,
        )

        # Model & scaler prototypes
        def fresh_model():
            return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=self.random_seed)

        def fresh_scaler():
            return MinMaxScaler(feature_range=(0, 1))

        # -------------------------------------------------------------
        # Condition 1: Baseline (untransformed)
        # -------------------------------------------------------------
        try:
            res_baseline = evaluate_representation(
                model=fresh_model(),
                scaler=fresh_scaler(),
                X_train_raw=X_train, y_train=y_tr,
                X_val_raw=X_val, y_val=y_v,
                X_test_raw=X_test, y_test=y_te,
                o_train=o_tr, o_val=o_v, o_test=o_te,
                changed_dict={},
                evaluator=evaluator,
                transformer=transformer,
                cate_attrs=cate_attrs,
                num_attrs=num_attrs,
                protected_attr=protected_attr,
            )
            res_baseline["terminal_state"] = {}
            res_baseline["terminal_train_max_dphi"] = res_baseline["train"]["max_dphi"]
            res_baseline["final_epsilon"] = eps_thresh
            res_baseline["fairness_feasible"] = bool(res_baseline["train"]["max_dphi"] <= eps_thresh)
            res_baseline["termination_reason"] = "baseline_untransformed"
        except Exception as exc:
            res_baseline = {
                "terminal_evaluation_performed": False,
                "terminal_state": {},
                "num_transforms": 0,
                "train": None,
                "validation": None,
                "test": None,
                "terminal_train_max_dphi": None,
                "final_epsilon": eps_thresh,
                "fairness_feasible": False,
                "termination_reason": "evaluation_failed",
                "error": str(exc),
            }

        # -------------------------------------------------------------
        # Condition 2: Canonical FairBias (D6 mitigation only)
        # -------------------------------------------------------------
        d6_changed = self.load_canonical_d6_changed_dict(arm_id)
        try:
            res_canonical = evaluate_representation(
                model=fresh_model(),
                scaler=fresh_scaler(),
                X_train_raw=X_train, y_train=y_tr,
                X_val_raw=X_val, y_val=y_v,
                X_test_raw=X_test, y_test=y_te,
                o_train=o_tr, o_val=o_v, o_test=o_te,
                changed_dict=d6_changed,
                evaluator=evaluator,
                transformer=transformer,
                cate_attrs=cate_attrs,
                num_attrs=num_attrs,
                protected_attr=protected_attr,
            )
            res_canonical["terminal_state"] = copy.deepcopy(d6_changed)
            res_canonical["terminal_train_max_dphi"] = res_canonical["train"]["max_dphi"]
            res_canonical["final_epsilon"] = eps_thresh
            res_canonical["fairness_feasible"] = bool(res_canonical["train"]["max_dphi"] <= eps_thresh)
            res_canonical["termination_reason"] = "d6_canonical_frozen"
        except Exception as exc:
            res_canonical = {
                "terminal_evaluation_performed": False,
                "terminal_state": copy.deepcopy(d6_changed),
                "num_transforms": len(d6_changed),
                "train": None,
                "validation": None,
                "test": None,
                "terminal_train_max_dphi": None,
                "final_epsilon": eps_thresh,
                "fairness_feasible": False,
                "termination_reason": "evaluation_failed",
                "error": str(exc),
            }

        if self.baseline_reproduction_only:
            return {
                "arm_id": arm_id,
                "protected_attribute": protected_attr,
                "disability_arm": disability_arm,
                "feature_set": feature_set,
                "epsilon_threshold": eps_thresh,
                "initial_train_max_dphi": float(max([float(v) for gd in init_eps_dict.values() for v in gd.values()])) if init_eps_dict else 0.0,
                "conditions": {
                    "baseline": res_baseline,
                    "canonical_fairbias": res_canonical,
                },
                "audit": {
                    "baseline_reproduction_only": True,
                    "posthoc_enhancement_called": False,
                    "joint_enhancement_called": False,
                    "candidate_fits_performed": 0,
                },
                "observed_cohort": observed_cohort,
                "observed_schema": observed_schema,
            }

        # -------------------------------------------------------------
        # Condition 3: Post-Mitigation Bounded Accuracy Enhancement
        # -------------------------------------------------------------
        ae_engine_post = FairAccuracyEnhancement(
            evaluator=evaluator,
            transformer=transformer,
            label_Y=outcome,
            cate_attrs=cate_attrs,
            num_attrs=num_attrs,
            max_fairness_degradation=0.02,
            run_id=self.run_id,
            arm_id=arm_id,
            condition="posthoc_enhancement",
        )

        post_changed = copy.deepcopy(d6_changed)
        t_X_init = transformer.transform_data(X_train, post_changed, num_attrs, cate_attrs)
        current_eps_post = evaluator.calculate_epsilon(
            t_X_init, O_tr_df, cate_attrs=cate_attrs, num_attrs=num_attrs
        )
        post_runner_refresh_geometry_evals = 1

        ae_steps_accepted = 0
        post_term_reason = "budget_exhausted"
        max_post_steps = 5 if not self.smoke_test else 2
        for it in range(1, max_post_steps + 1):
            try:
                _, post_changed, accepted_attr = ae_engine_post.enhance_step(
                    X_train=X_train,
                    Y_train=y_train_s,
                    changed_dict=post_changed,
                    O_train=O_tr_df,
                    epsilon_threshold=eps_thresh,
                    current_epsilon=current_eps_post,
                    iteration=it,
                    partition=partition,
                )
            except (RuntimeError, ValueError) as exc:
                post_term_reason = "evaluation_failed"
                break

            if accepted_attr is None:
                post_term_reason = "candidate_exhausted"
                break
            ae_steps_accepted += 1
            t_X_curr = transformer.transform_data(X_train, post_changed, num_attrs, cate_attrs)
            current_eps_post = evaluator.calculate_epsilon(
                t_X_curr, O_tr_df, cate_attrs=cate_attrs, num_attrs=num_attrs
            )
            post_runner_refresh_geometry_evals += 1

        self.audit_events.extend(ae_engine_post.audit_trail)

        if post_term_reason == "evaluation_failed":
            res_posthoc_ae = {
                "terminal_evaluation_performed": False,
                "terminal_state": copy.deepcopy(post_changed),
                "num_transforms": len(post_changed),
                "train": None,
                "validation": None,
                "test": None,
                "terminal_train_max_dphi": None,
                "final_epsilon": eps_thresh,
                "fairness_feasible": False,
                "termination_reason": "evaluation_failed",
                "ae_steps_accepted": ae_steps_accepted,
                "model_fit_count": ae_engine_post.total_model_fits,
                "geometry_eval_count": ae_engine_post.total_geometry_evals,
                "geometry_eval_breakdown": {
                    "ae_guard_geometry_evals": ae_engine_post.total_geometry_evals,
                    "runner_state_refresh_geometry_evals": post_runner_refresh_geometry_evals,
                    "terminal_evaluation_geometry_evals": 0,
                    "bm_geometry_evals": None,
                    "total_observable_geometry_evals": ae_engine_post.total_geometry_evals + post_runner_refresh_geometry_evals,
                },
            }
        else:
            try:
                res_posthoc_ae = evaluate_representation(
                    model=fresh_model(),
                    scaler=fresh_scaler(),
                    X_train_raw=X_train, y_train=y_tr,
                    X_val_raw=X_val, y_val=y_v,
                    X_test_raw=X_test, y_test=y_te,
                    o_train=o_tr, o_val=o_v, o_test=o_te,
                    changed_dict=post_changed,
                    evaluator=evaluator,
                    transformer=transformer,
                    cate_attrs=cate_attrs,
                    num_attrs=num_attrs,
                    protected_attr=protected_attr,
                )
                res_posthoc_ae["terminal_state"] = copy.deepcopy(post_changed)
                res_posthoc_ae["terminal_train_max_dphi"] = res_posthoc_ae["train"]["max_dphi"]
                res_posthoc_ae["final_epsilon"] = eps_thresh
                res_posthoc_ae["fairness_feasible"] = bool(res_posthoc_ae["train"]["max_dphi"] <= eps_thresh)
                res_posthoc_ae["termination_reason"] = post_term_reason
                res_posthoc_ae["ae_steps_accepted"] = ae_steps_accepted
                res_posthoc_ae["model_fit_count"] = ae_engine_post.total_model_fits
                res_posthoc_ae["geometry_eval_count"] = ae_engine_post.total_geometry_evals
                res_posthoc_ae["geometry_eval_breakdown"] = {
                    "ae_guard_geometry_evals": ae_engine_post.total_geometry_evals,
                    "runner_state_refresh_geometry_evals": post_runner_refresh_geometry_evals,
                    "terminal_evaluation_geometry_evals": 3,
                    "bm_geometry_evals": None,
                    "total_observable_geometry_evals": ae_engine_post.total_geometry_evals + post_runner_refresh_geometry_evals + 3,
                }
            except Exception as exc:
                res_posthoc_ae = {
                    "terminal_evaluation_performed": False,
                    "terminal_state": copy.deepcopy(post_changed),
                    "num_transforms": len(post_changed),
                    "train": None,
                    "validation": None,
                    "test": None,
                    "terminal_train_max_dphi": None,
                    "final_epsilon": eps_thresh,
                    "fairness_feasible": False,
                    "termination_reason": "evaluation_failed",
                    "error": str(exc),
                    "ae_steps_accepted": ae_steps_accepted,
                    "model_fit_count": ae_engine_post.total_model_fits,
                    "geometry_eval_count": ae_engine_post.total_geometry_evals,
                    "geometry_eval_breakdown": {
                        "ae_guard_geometry_evals": ae_engine_post.total_geometry_evals,
                        "runner_state_refresh_geometry_evals": post_runner_refresh_geometry_evals,
                        "terminal_evaluation_geometry_evals": 0,
                        "bm_geometry_evals": None,
                        "total_observable_geometry_evals": ae_engine_post.total_geometry_evals + post_runner_refresh_geometry_evals,
                    },
                }

        # -------------------------------------------------------------
        # Condition 4: Joint Interleaved Mitigation + Enhancement
        # -------------------------------------------------------------
        mit_engine = FairBiasMitigation(
            evaluator=evaluator,
            transformer=transformer,
            label_O=[protected_attr],
            cate_attrs=cate_attrs,
            num_attrs=num_attrs,
            phi_threshold=fb_config.phi_threshold,
            poly_exponents=fb_config.transform_poly_exponents,
            failed_attribute_mode="stop",
            power_sequence_policy=fb_config.power_sequence_policy,
            power_revisit_policy=fb_config.power_revisit_policy,
        )
        ae_engine_joint = FairAccuracyEnhancement(
            evaluator=evaluator,
            transformer=transformer,
            label_Y=outcome,
            cate_attrs=cate_attrs,
            num_attrs=num_attrs,
            max_fairness_degradation=0.02,
            run_id=self.run_id,
            arm_id=arm_id,
            condition="joint_enhancement",
        )

        joint_changed: Dict[str, Any] = {}
        current_parent_state_hash = changed_dict_hash(joint_changed)
        committed_joint_states_set = {current_parent_state_hash}
        current_eps_joint = copy.deepcopy(init_eps_dict)
        nmi_org = calculate_nmi_dict(X_train, y_train_s)
        joint_runner_refresh_geometry_evals = 0

        bm_steps_accepted = 0
        ae_joint_steps_accepted = 0
        joint_events: List[Dict[str, Any]] = []
        joint_term_reason = "budget_exhausted"

        max_iter = 10 if not self.smoke_test else 3
        for it in range(1, max_iter + 1):
            # Step A: Mitigation
            parent_state_hash = current_parent_state_hash
            _, candidate_changed, sel_o, sel_attr = mit_engine.mitigate_step(
                X=X_train,
                Y=y_train_s,
                O=O_tr_df,
                nmi_org=nmi_org,
                changed_dict=joint_changed,
                current_epsilon=current_eps_joint,
                epsilon_threshold=eps_thresh,
                iteration=it,
            )
            if sel_attr is not None:
                bm_hash = changed_dict_hash(candidate_changed)
                if bm_hash in committed_joint_states_set:
                    joint_term_reason = "cycle_detected"
                    joint_events.append({
                        "iteration": it,
                        "engine": "BM",
                        "selected_attribute": sel_attr,
                        "parent_state_hash": parent_state_hash,
                        "resulting_state_hash": bm_hash,
                        "engine_accepted": True,
                        "trajectory_committed": False,
                        "cycle_detected": True,
                        "changed_dict_snapshot": copy.deepcopy(candidate_changed),
                    })
                    break
                committed_joint_states_set.add(bm_hash)
                bm_steps_accepted += 1
                joint_changed = candidate_changed
                current_parent_state_hash = bm_hash
                t_X_bm = transformer.transform_data(X_train, joint_changed, num_attrs, cate_attrs)
                current_eps_joint = evaluator.calculate_epsilon(
                    t_X_bm, O_tr_df, cate_attrs=cate_attrs, num_attrs=num_attrs
                )
                joint_runner_refresh_geometry_evals += 1
                joint_events.append({
                    "iteration": it,
                    "engine": "BM",
                    "selected_attribute": sel_attr,
                    "parent_state_hash": parent_state_hash,
                    "resulting_state_hash": bm_hash,
                    "engine_accepted": True,
                    "trajectory_committed": True,
                    "cycle_detected": False,
                    "changed_dict_snapshot": copy.deepcopy(joint_changed),
                })
            else:
                joint_events.append({
                    "iteration": it,
                    "engine": "BM",
                    "selected_attribute": None,
                    "parent_state_hash": parent_state_hash,
                    "resulting_state_hash": parent_state_hash,
                    "engine_accepted": False,
                    "trajectory_committed": False,
                    "cycle_detected": False,
                    "changed_dict_snapshot": copy.deepcopy(joint_changed),
                })

            # Step B: Enhancement (passes updated current_eps_joint)
            parent_state_hash = current_parent_state_hash
            try:
                _, candidate_changed, ae_attr = ae_engine_joint.enhance_step(
                    X_train=X_train,
                    Y_train=y_train_s,
                    changed_dict=joint_changed,
                    O_train=O_tr_df,
                    epsilon_threshold=eps_thresh,
                    current_epsilon=current_eps_joint,
                    iteration=it,
                    partition=partition,
                )
            except (RuntimeError, ValueError) as exc:
                joint_term_reason = "evaluation_failed"
                break

            if ae_attr is not None:
                ae_hash = changed_dict_hash(candidate_changed)
                if ae_hash in committed_joint_states_set:
                    joint_term_reason = "cycle_detected"
                    joint_events.append({
                        "iteration": it,
                        "engine": "AE",
                        "selected_attribute": ae_attr,
                        "parent_state_hash": parent_state_hash,
                        "resulting_state_hash": ae_hash,
                        "engine_accepted": True,
                        "trajectory_committed": False,
                        "cycle_detected": True,
                        "changed_dict_snapshot": copy.deepcopy(candidate_changed),
                    })
                    break
                committed_joint_states_set.add(ae_hash)
                ae_joint_steps_accepted += 1
                joint_changed = candidate_changed
                current_parent_state_hash = ae_hash
                t_X_ae = transformer.transform_data(X_train, joint_changed, num_attrs, cate_attrs)
                current_eps_joint = evaluator.calculate_epsilon(
                    t_X_ae, O_tr_df, cate_attrs=cate_attrs, num_attrs=num_attrs
                )
                joint_runner_refresh_geometry_evals += 1
                joint_events.append({
                    "iteration": it,
                    "engine": "AE",
                    "selected_attribute": ae_attr,
                    "parent_state_hash": parent_state_hash,
                    "resulting_state_hash": ae_hash,
                    "engine_accepted": True,
                    "trajectory_committed": True,
                    "cycle_detected": False,
                    "changed_dict_snapshot": copy.deepcopy(joint_changed),
                })
            else:
                joint_events.append({
                    "iteration": it,
                    "engine": "AE",
                    "selected_attribute": None,
                    "parent_state_hash": parent_state_hash,
                    "resulting_state_hash": parent_state_hash,
                    "engine_accepted": False,
                    "trajectory_committed": False,
                    "cycle_detected": False,
                    "changed_dict_snapshot": copy.deepcopy(joint_changed),
                })

            # Check convergence/termination
            if sel_attr is None and ae_attr is None:
                joint_term_reason = "candidate_exhausted"
                break

            all_eps = [float(v) for gd in current_eps_joint.values() for v in gd.values()]
            max_e = float(max(all_eps)) if all_eps else 0.0

            if max_e <= eps_thresh:
                joint_term_reason = "epsilon_reached"
                break

        self.audit_events.extend(ae_engine_joint.audit_trail)

        if joint_term_reason == "evaluation_failed":
            res_joint_ae = {
                "terminal_evaluation_performed": False,
                "terminal_state": copy.deepcopy(joint_changed),
                "num_transforms": len(joint_changed),
                "train": None,
                "validation": None,
                "test": None,
                "terminal_train_max_dphi": None,
                "final_epsilon": eps_thresh,
                "fairness_feasible": False,
                "termination_reason": "evaluation_failed",
                "bm_steps_accepted": bm_steps_accepted,
                "ae_steps_accepted": ae_joint_steps_accepted,
                "iteration_events": joint_events,
                "model_fit_count": ae_engine_joint.total_model_fits,
                "geometry_eval_count": ae_engine_joint.total_geometry_evals,
                "geometry_eval_breakdown": {
                    "ae_guard_geometry_evals": ae_engine_joint.total_geometry_evals,
                    "runner_state_refresh_geometry_evals": joint_runner_refresh_geometry_evals,
                    "terminal_evaluation_geometry_evals": 0,
                    "bm_geometry_evals": None,
                    "total_observable_geometry_evals": ae_engine_joint.total_geometry_evals + joint_runner_refresh_geometry_evals,
                },
            }
        else:
            try:
                res_joint_ae = evaluate_representation(
                    model=fresh_model(),
                    scaler=fresh_scaler(),
                    X_train_raw=X_train, y_train=y_tr,
                    X_val_raw=X_val, y_val=y_v,
                    X_test_raw=X_test, y_test=y_te,
                    o_train=o_tr, o_val=o_v, o_test=o_te,
                    changed_dict=joint_changed,
                    evaluator=evaluator,
                    transformer=transformer,
                    cate_attrs=cate_attrs,
                    num_attrs=num_attrs,
                    protected_attr=protected_attr,
                )
                res_joint_ae["terminal_state"] = copy.deepcopy(joint_changed)
                res_joint_ae["terminal_train_max_dphi"] = res_joint_ae["train"]["max_dphi"]
                res_joint_ae["final_epsilon"] = eps_thresh
                res_joint_ae["fairness_feasible"] = bool(res_joint_ae["train"]["max_dphi"] <= eps_thresh)
                res_joint_ae["termination_reason"] = joint_term_reason
                res_joint_ae["bm_steps_accepted"] = bm_steps_accepted
                res_joint_ae["ae_steps_accepted"] = ae_joint_steps_accepted
                res_joint_ae["iteration_events"] = joint_events
                res_joint_ae["model_fit_count"] = ae_engine_joint.total_model_fits
                res_joint_ae["geometry_eval_count"] = ae_engine_joint.total_geometry_evals
                res_joint_ae["geometry_eval_breakdown"] = {
                    "ae_guard_geometry_evals": ae_engine_joint.total_geometry_evals,
                    "runner_state_refresh_geometry_evals": joint_runner_refresh_geometry_evals,
                    "terminal_evaluation_geometry_evals": 3,
                    "bm_geometry_evals": None,
                    "total_observable_geometry_evals": ae_engine_joint.total_geometry_evals + joint_runner_refresh_geometry_evals + 3,
                }
            except Exception as exc:
                res_joint_ae = {
                    "terminal_evaluation_performed": False,
                    "terminal_state": copy.deepcopy(joint_changed),
                    "num_transforms": len(joint_changed),
                    "train": None,
                    "validation": None,
                    "test": None,
                    "terminal_train_max_dphi": None,
                    "final_epsilon": eps_thresh,
                    "fairness_feasible": False,
                    "termination_reason": "evaluation_failed",
                    "error": str(exc),
                    "bm_steps_accepted": bm_steps_accepted,
                    "ae_steps_accepted": ae_joint_steps_accepted,
                    "iteration_events": joint_events,
                    "model_fit_count": ae_engine_joint.total_model_fits,
                    "geometry_eval_count": ae_engine_joint.total_geometry_evals,
                    "geometry_eval_breakdown": {
                        "ae_guard_geometry_evals": ae_engine_joint.total_geometry_evals,
                        "runner_state_refresh_geometry_evals": joint_runner_refresh_geometry_evals,
                        "terminal_evaluation_geometry_evals": 0,
                        "bm_geometry_evals": None,
                        "total_observable_geometry_evals": ae_engine_joint.total_geometry_evals + joint_runner_refresh_geometry_evals,
                    },
                }

        return {
            "arm_id": arm_id,
            "protected_attribute": protected_attr,
            "outcome": outcome,
            "epsilon_threshold": eps_thresh,
            "conditions": {
                "baseline": res_baseline,
                "canonical_fairbias": res_canonical,
                "posthoc_enhancement": res_posthoc_ae,
                "joint_enhancement": res_joint_ae,
            },
            "observed_cohort": observed_cohort,
            "observed_schema": observed_schema,
        }
