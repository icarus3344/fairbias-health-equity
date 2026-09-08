"""Exploratory Study D8: Accuracy Enhancement evaluation on the four NHIS temporal arms.

Evaluates whether fairness-bounded Accuracy Enhancement (AE) can alleviate or reverse
downstream classification utility collapse (AUROC/AUPRC loss and prediction suppression)
observed in Gate D6 / D7 across the four frozen arms:
- D6_ARM_001 (SEX_A, 21 predictors)
- D6_ARM_002 (HISPALLP_A, 21 predictors)
- D6_ARM_003 (DISAB3_A, 21 predictors)
- D6_ARM_004 (DISAB3_A, 15 predictors)
"""

from __future__ import annotations

import copy
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    auc,
)
from sklearn.preprocessing import MinMaxScaler

from fairbias.config import (
    ALGORITHM_MODE_ENGINEERING,
    FairBiasConfig,
)
from fairbias.evaluator import FairEvaluator
from fairbias.enhancement import FairAccuracyEnhancement
from fairbias.mitigation import FairBiasMitigation
from fairbias.transform import (
    FairTransform,
    calculate_nmi_dict,
)
from nhis_fairbias.adapter import NHISStudyAdapter
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
    model: LogisticRegression,
    scaler: MinMaxScaler,
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
    # Transform
    X_tr_t = transformer.transform_data(X_train_raw, changed_dict, num_attrs, cate_attrs)
    X_val_t = transformer.transform_data(X_val_raw, changed_dict, num_attrs, cate_attrs)
    X_te_t = transformer.transform_data(X_test_raw, changed_dict, num_attrs, cate_attrs)

    # Scale strictly on train
    X_tr_s = scaler.fit_transform(X_tr_t)
    X_val_s = scaler.transform(X_val_t)
    X_te_s = scaler.transform(X_te_t)

    # Fit model on train
    model.fit(X_tr_s, y_train)

    # Predictions
    p_tr = model.predict_proba(X_tr_s)[:, 1]
    p_val = model.predict_proba(X_val_s)[:, 1]
    p_te = model.predict_proba(X_te_s)[:, 1]

    pred_tr = (p_tr >= 0.5).astype(int)
    pred_val = (p_val >= 0.5).astype(int)
    pred_te = (p_te >= 0.5).astype(int)

    # AUPRC helper
    def get_auprc(y_t, p_s):
        prec, rec, _ = precision_recall_curve(y_t, p_s)
        return float(auc(rec, prec))

    # Evaluate fairness epsilon
    O_tr_df = pd.DataFrame({protected_attr: o_train}, index=X_train_raw.index)
    O_val_df = pd.DataFrame({protected_attr: o_val}, index=X_val_raw.index)
    O_te_df = pd.DataFrame({protected_attr: o_test}, index=X_test_raw.index)

    eps_tr = evaluator.calculate_epsilon(X_tr_t, O_tr_df, cate_attrs=cate_attrs, num_attrs=num_attrs)
    eps_val = evaluator.calculate_epsilon(X_val_t, O_val_df, cate_attrs=cate_attrs, num_attrs=num_attrs)
    eps_te = evaluator.calculate_epsilon(X_te_t, O_te_df, cate_attrs=cate_attrs, num_attrs=num_attrs)

    def get_max_eps(ed):
        vals = [v for gd in ed.values() for v in gd.values()]
        return float(max(vals)) if vals else 0.0

    # Group fairness gaps on test
    test_gaps = compute_group_fairness_gaps(y_test, pred_te, o_test)

    return {
        "changed_dict": copy.deepcopy(changed_dict),
        "num_transforms": len(changed_dict),
        "train": {
            "auroc": float(roc_auc_score(y_train, p_tr)),
            "auprc": get_auprc(y_train, p_tr),
            "brier": float(brier_score_loss(y_train, p_tr)),
            "accuracy": float(accuracy_score(y_train, pred_tr)),
            "predicted_positive_count": int(np.sum(pred_tr)),
            "predicted_positive_rate": float(np.mean(pred_tr)),
            "max_dphi": get_max_eps(eps_tr),
        },
        "validation": {
            "auroc": float(roc_auc_score(y_val, p_val)),
            "auprc": get_auprc(y_val, p_val),
            "brier": float(brier_score_loss(y_val, p_val)),
            "accuracy": float(accuracy_score(y_val, pred_val)),
            "predicted_positive_count": int(np.sum(pred_val)),
            "predicted_positive_rate": float(np.mean(pred_val)),
            "max_dphi": get_max_eps(eps_val),
        },
        "test": {
            "auroc": float(roc_auc_score(y_test, p_te)),
            "auprc": get_auprc(y_test, p_te),
            "brier": float(brier_score_loss(y_test, p_te)),
            "accuracy": float(accuracy_score(y_test, pred_te)),
            "predicted_positive_count": int(np.sum(pred_te)),
            "predicted_positive_rate": float(np.mean(pred_te)),
            "max_dphi": get_max_eps(eps_te),
            "demographic_parity_difference": test_gaps["demographic_parity_difference"],
            "equal_opportunity_difference": test_gaps["equal_opportunity_difference"],
        },
    }


class D8EnhancementRunner:
    """Orchestrates the Accuracy Enhancement evaluation across the 4 frozen NHIS arms."""

    def __init__(
        self,
        adapter: Optional[NHISStudyAdapter] = None,
        smoke_test: bool = False,
        random_seed: int = 42,
    ):
        self.smoke_test = smoke_test
        self.random_seed = random_seed
        self.adapter = adapter or NHISStudyAdapter(features_parquet_path=FEATURES_PARQUET_PATH)

    def load_canonical_d6_changed_dict(self, arm_id: str) -> Dict[str, Any]:
        """Load frozen canonical FairBias changed_dict from the D6 release."""
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

        # 1. Load cohorts
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
            X_train = X_train.iloc[:500]
            y_train_s = y_train_s.iloc[:500]
            o_train_s = o_train_s.iloc[:500]
            X_val = X_val.iloc[:300]
            y_val_s = y_val_s.iloc[:300]
            o_val_s = o_val_s.iloc[:300]
            X_test = X_test.iloc[:300]
            y_test_s = y_test_s.iloc[:300]
            o_test_s = o_test_s.iloc[:300]

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
        init_eps_dict = evaluator.calculate_epsilon(X_train, O_tr_df, cate_attrs=cate_attrs, num_attrs=num_attrs)
        eps_thresh = float(evaluator.compute_threshold(init_eps_dict))

        # Model & scaler prototypes
        def fresh_model():
            return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=self.random_seed)

        def fresh_scaler():
            return MinMaxScaler(feature_range=(0, 1))

        # -------------------------------------------------------------
        # Condition 1: Baseline (untransformed)
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # Condition 2: Canonical FairBias (D6 mitigation only)
        # -------------------------------------------------------------
        d6_changed = self.load_canonical_d6_changed_dict(arm_id)
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

        # -------------------------------------------------------------
        # Condition 3: Post-Mitigation Bounded Accuracy Enhancement
        # (Takes D6 converged state and tests bounded AE steps)
        # -------------------------------------------------------------
        ae_engine_post = FairAccuracyEnhancement(
            evaluator=evaluator,
            transformer=transformer,
            label_Y=outcome,
            cate_attrs=cate_attrs,
            num_attrs=num_attrs,
            max_fairness_degradation=0.02,
        )

        post_changed = copy.deepcopy(d6_changed)
        current_eps_post = evaluator.calculate_epsilon(
            transformer.transform_data(X_train, post_changed, num_attrs, cate_attrs),
            O_tr_df, cate_attrs=cate_attrs, num_attrs=num_attrs
        )

        # Run up to 5 bounded AE steps
        ae_steps_accepted = 0
        for _ in range(5):
            _, post_changed, accepted_attr = ae_engine_post.enhance_step(
                X_train=X_train,
                Y_train=y_train_s,
                changed_dict=post_changed,
                O_train=O_tr_df,
                epsilon_threshold=eps_thresh,
                current_epsilon=current_eps_post,
                X_val=X_val,
                Y_val=y_val_s,
            )
            if accepted_attr is None:
                break
            ae_steps_accepted += 1
            current_eps_post = evaluator.calculate_epsilon(
                transformer.transform_data(X_train, post_changed, num_attrs, cate_attrs),
                O_tr_df, cate_attrs=cate_attrs, num_attrs=num_attrs
            )

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
        res_posthoc_ae["ae_steps_accepted"] = ae_steps_accepted

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
        )

        joint_changed: Dict[str, Any] = {}
        current_eps_joint = copy.deepcopy(init_eps_dict)
        nmi_org = calculate_nmi_dict(X_train, y_train_s)

        max_iter = 10 if not self.smoke_test else 3
        for it in range(1, max_iter + 1):
            # Step A: Mitigation
            _, joint_changed, sel_o, sel_attr = mit_engine.mitigate_step(
                X=X_train,
                Y=y_train_s,
                O=O_tr_df,
                nmi_org=nmi_org,
                changed_dict=joint_changed,
                current_epsilon=current_eps_joint,
                epsilon_threshold=eps_thresh,
                iteration=it,
            )

            # Step B: Enhancement
            _, joint_changed, ae_attr = ae_engine_joint.enhance_step(
                X_train=X_train,
                Y_train=y_train_s,
                changed_dict=joint_changed,
                O_train=O_tr_df,
                epsilon_threshold=eps_thresh,
                current_epsilon=current_eps_joint,
                X_val=X_val,
                Y_val=y_val_s,
            )

            # Re-evaluate epsilon
            t_X = transformer.transform_data(X_train, joint_changed, num_attrs, cate_attrs)
            current_eps_joint = evaluator.calculate_epsilon(t_X, O_tr_df, cate_attrs=cate_attrs, num_attrs=num_attrs)
            all_eps = [v for gd in current_eps_joint.values() for v in gd.values()]
            max_e = float(max(all_eps)) if all_eps else 0.0

            if sel_attr is None and ae_attr is None:
                break
            if max_e <= eps_thresh:
                break

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
        }
