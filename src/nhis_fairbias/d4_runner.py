"""Dedicated NHIS presplit FairBias runner and Gate D4.0 preflight orchestrator.

Guarantees:
1. Master Pooled Split Preservation:
   - Directly orchestrates NHISPooledAdapter, FairBiasConfig, FairEvaluator,
     FairTransform, FairBiasMitigation, and calculate_nmi_dict without re-splitting data.
   - NEVER calls run_fairbias_pipeline().
   - NEVER modifies generate_pooled_splits() or creates a second random split.
2. Training-Only FairBias Fitting:
   - Predictor families, initial d_phi, epsilon threshold, NMI dictionary, greedy
     ranking, and all transformations are learned strictly on TRAIN.
   - Validation and test data NEVER participate in transform fitting or epsilon determination.
   - Terminates strictly via paper-mode greedy criteria (no iteration budget, no Pareto rollback).
3. Test Partition Embargo:
   - allow_test_evaluation defaults to False.
   - Any attempt to evaluate, score, or predict on test while allow_test_evaluation=False
     raises RuntimeError immediately.
   - Manifest explicitly asserts test_evaluated = False and authorized_partitions = ['train', 'validation'].
4. Paired Baseline vs FairBias Comparison:
   - Baseline: fresh LR trained on original preprocessed TRAIN X, evaluated on VALIDATION.
   - FairBias: fresh LR trained on transformed TRAIN X, evaluated on transformed VALIDATION.
   - Identical fixed classifier specification and min-max scaling.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import pathlib
import subprocess
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from fairbias.config import (
    ALGORITHM_MODE_PAPER_FAITHFUL,
    FairBiasConfig,
    official_power_stream,
)
from fairbias.evaluator import FairEvaluator
from fairbias.mitigation import FairBiasMitigation
from fairbias.models import get_classifier
from fairbias.transform import (
    FairTransform,
    calculate_nmi_dict,
)
from fairbias.transform_trace import (
    FairBiasTransformStep,
    FairBiasTransformTrace,
)

from .audit import write_csv_atomic
from .download import (
    compute_sha256,
    new_run_id,
    utc_timestamp,
    write_json_atomic,
)
from .evaluation import (
    compute_evaluation_comparison,
    compute_fairness_gaps,
    compute_group_metrics,
    compute_multicategory_pairwise_differences,
    compute_utility_metrics,
    evaluate_predictions,
)
from .features import (
    DEFAULT_FEATURE_CONFIG,
    load_feature_registry,
)
from .pooled import (
    DEFAULT_POOLED_SEED,
    NHISPooledAdapter,
)
from .preprocessing import NHISLeakageError
from .schema import (
    DEFAULT_STUDY_CONFIG,
    load_study_config,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

FROZEN_D4_ARMS: Dict[str, Dict[str, Any]] = {
    "ARM_D3_001": {
        "arm_id": "ARM_D3_001",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "SEX_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
    },
    "ARM_D3_002": {
        "arm_id": "ARM_D3_002",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "HISPALLP_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
    },
    "ARM_D3_003": {
        "arm_id": "ARM_D3_003",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "DISAB3_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "full_feature",
        "expected_predictors": 21,
    },
    "ARM_D3_004": {
        "arm_id": "ARM_D3_004",
        "outcome": "MEDDL12M_A",
        "protected_attribute": "DISAB3_A",
        "feature_set": "PRIMARY_CORE",
        "disability_arm": "exclude_disability_components",
        "expected_predictors": 15,
    },
}


def get_git_commit(repo_root: Optional[pathlib.Path] = None) -> str:
    """Retrieve current git HEAD commit hash."""
    root = repo_root or _REPO_ROOT
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


class NHISD4Runner:
    """Dedicated orchestrator for Gate D4 substantive FairBias execution."""

    def __init__(
        self,
        adapter: Optional[NHISPooledAdapter] = None,
        features_parquet_path: Optional[Union[str, pathlib.Path]] = None,
        split_manifest_path: Optional[Union[str, pathlib.Path]] = None,
        d3_manifest_path: Optional[Union[str, pathlib.Path]] = None,
        study_config_path: Optional[Union[str, pathlib.Path]] = None,
        feature_config_path: Optional[Union[str, pathlib.Path]] = None,
        allow_test_evaluation: bool = False,
    ):
        self.allow_test_evaluation = bool(allow_test_evaluation)

        self.adapter = adapter or NHISPooledAdapter(
            features_parquet_path=features_parquet_path,
            split_manifest_path=split_manifest_path,
            study_config_path=study_config_path,
            feature_config_path=feature_config_path,
        )

        d3_path = d3_manifest_path or (_REPO_ROOT / "artifacts" / "nhis" / "d3" / "d3_manifest.json")
        self.d3_manifest_path = pathlib.Path(d3_path).resolve()

        split_path = split_manifest_path or (_REPO_ROOT / "artifacts" / "nhis" / "d3" / "pooled_split_manifest.csv")
        self.split_manifest_path = pathlib.Path(split_path).resolve()

    def get_arm_config(self, arm_id: str) -> Dict[str, Any]:
        """Retrieve frozen arm specification."""
        arm_key = arm_id.strip().upper()
        if arm_key not in FROZEN_D4_ARMS:
            raise ValueError(
                f"Unknown arm_id: {arm_id!r}. Supported frozen arms: {list(FROZEN_D4_ARMS.keys())}"
            )
        return copy.deepcopy(FROZEN_D4_ARMS[arm_key])

    def evaluate_test_embargo_guard(self) -> None:
        """Fail-closed assertion enforcing test partition embargo."""
        if not self.allow_test_evaluation:
            raise RuntimeError(
                "Test partition evaluation is strictly embargoed in Gate D4.0 "
                "(allow_test_evaluation=False). Cannot evaluate, predict, or score on test."
            )

    def run_preflight(
        self,
        arm_id: str = "ARM_D3_001",
        output_dir: Optional[Union[str, pathlib.Path]] = None,
        random_seed: int = 0,
        sample_weight: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute real-data preflight on TRAIN and VALIDATION partitions only.
        """
        start_time = time.time()
        arm_config = self.get_arm_config(arm_id)

        # Rejection of sample weights in paper mode
        if sample_weight is not None:
            raise NHISLeakageError(
                "tang2024_paper_faithful mode strictly forbids sample_weight. "
                "The Tang et al. (2024) baseline is unweighted."
            )

        run_id = new_run_id()
        if output_dir is not None:
            target_out_dir = pathlib.Path(output_dir).resolve()
        else:
            target_out_dir = (_REPO_ROOT / "runs" / "nhis_d4_preflight" / run_id).resolve()
        target_out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Obtain frozen cohort from NHISPooledAdapter
        cohorts = self.adapter.get_pooled_cohort(
            outcome=arm_config["outcome"],
            protected_attribute=arm_config["protected_attribute"],
            feature_set=arm_config["feature_set"].lower(),
            disability_arm=arm_config["disability_arm"],
        )
        X_train, y_train, o_train, _, meta_train = cohorts["train"]
        X_val, y_val, o_val, _, meta_val = cohorts["val"]

        # 2. Semantic predictor families from preprocessor
        feature_set_key = arm_config["feature_set"].lower()
        all_cats, all_nums = self.adapter.preprocessor.get_feature_family_lists(feature_set_key)
        active_feats = set(X_train.columns)
        cate_attrs = [f for f in all_cats if f in active_feats]
        num_attrs = [f for f in all_nums if f in active_feats]

        if len(active_feats) != arm_config["expected_predictors"]:
            raise ValueError(
                f"Feature count mismatch for {arm_id}: expected {arm_config['expected_predictors']}, "
                f"got {len(active_feats)} ({list(active_feats)})"
            )

        # 3. Resolve FairBiasConfig in tang2024_paper_faithful mode
        fb_config = FairBiasConfig(
            algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL,
            random_seed=random_seed,
            classifier="LR",
            eval_norm="min-max",
            label_O=(arm_config["protected_attribute"],),
            label_Y=arm_config["outcome"],
            use_bias_mitigation=True,
            use_accuracy_enhancement=False,
            failed_attribute_mode="stop",
            power_sequence_policy="official_stream",
            power_revisit_policy="restart",
        ).resolved()

        # 4. Initialize Evaluator and Transformer
        evaluator = FairEvaluator(
            config=fb_config,
            label_O=[arm_config["protected_attribute"]],
            label_Y=arm_config["outcome"],
            cate_attrs=cate_attrs,
            num_attrs=num_attrs,
        )
        transformer = FairTransform(
            n_bins=fb_config.transform_n_bins,
            log_epsilon=fb_config.transform_log_epsilon,
            x_max=fb_config.transform_x_max,
        )

        # 5. Determine initial d_phi, epsilon threshold, and NMI strictly on TRAIN
        O_train_df = pd.DataFrame({arm_config["protected_attribute"]: o_train})
        nmi_org = calculate_nmi_dict(X_train, y_train)
        init_epsilon_dict = evaluator.calculate_epsilon(
            X_train, O_train_df, cate_attrs=cate_attrs, num_attrs=num_attrs, sample_weight=None
        )
        epsilon_threshold = evaluator.compute_threshold(init_epsilon_dict)

        initial_dphi_raw = copy.deepcopy(init_epsilon_dict.get(arm_config["protected_attribute"], {}))
        initial_dphi = {k: float(v) for k, v in initial_dphi_raw.items()}
        initial_max_dphi = float(max(initial_dphi.values())) if initial_dphi else 0.0
        highest_initial_feature = (
            max(initial_dphi, key=initial_dphi.get) if initial_dphi else None
        )

        # 6. Initialize FairBiasMitigation engine
        mitigation_engine = FairBiasMitigation(
            evaluator=evaluator,
            transformer=transformer,
            label_O=[arm_config["protected_attribute"]],
            cate_attrs=cate_attrs,
            num_attrs=num_attrs,
            phi_threshold=fb_config.phi_threshold,
            poly_exponents=fb_config.transform_poly_exponents,
            failed_attribute_mode=fb_config.failed_attribute_mode,
            power_sequence_policy=fb_config.power_sequence_policy,
            power_revisit_policy=fb_config.power_revisit_policy,
        )

        # 7. Execute greedy mitigation loop strictly on TRAIN
        changed_dict: Dict[str, Any] = {}
        current_epsilon = copy.deepcopy(init_epsilon_dict)
        iter_idx = 0
        exit_reason: Optional[str] = None

        if initial_max_dphi <= epsilon_threshold:
            exit_reason = "epsilon_reached"
        else:
            while True:
                iter_idx += 1
                current_X_train, changed_dict, sel_o, sel_attr = mitigation_engine.mitigate_step(
                    X=X_train,
                    Y=y_train,
                    O=O_train_df,
                    nmi_org=nmi_org,
                    changed_dict=changed_dict,
                    current_epsilon=current_epsilon,
                    epsilon_threshold=epsilon_threshold,
                    iteration=iter_idx,
                )

                if sel_attr is None:
                    exit_reason = "no_transform_accepted"
                    break

                # Apply current transformations to training fold
                transformed_X_train = transformer.transform_data(
                    X_train, changed_dict, num_attrs, cate_attrs
                )

                # Recompute d_phi strictly on training fold
                current_epsilon = evaluator.calculate_epsilon(
                    transformed_X_train,
                    O_train_df,
                    cate_attrs=cate_attrs,
                    num_attrs=num_attrs,
                    sample_weight=None,
                )
                curr_max_eps = float(
                    max(val for gd in current_epsilon.values() for val in gd.values())
                ) if current_epsilon else 0.0

                if curr_max_eps <= epsilon_threshold:
                    exit_reason = "epsilon_reached"
                    break

        # Resolve termination semantics
        if exit_reason == "no_transform_accepted":
            final_eps_vals = [
                val for gd in current_epsilon.values() for val in gd.values()
            ]
            term_max = float(max(final_eps_vals)) if final_eps_vals else initial_max_dphi
            if term_max <= epsilon_threshold:
                termination_reason = "epsilon_reached"
                converged = True
            else:
                termination_reason = "candidate_grid_exhausted"
                converged = False
        elif exit_reason == "epsilon_reached":
            termination_reason = "epsilon_reached"
            converged = True
        else:
            termination_reason = exit_reason or "unknown"
            converged = (termination_reason == "epsilon_reached")

        # Compute final d_phi on transformed TRAIN
        transformed_X_train = transformer.transform_data(
            X_train, changed_dict, num_attrs, cate_attrs
        )
        final_epsilon_dict = evaluator.calculate_epsilon(
            transformed_X_train,
            O_train_df,
            cate_attrs=cate_attrs,
            num_attrs=num_attrs,
            sample_weight=None,
        )
        final_dphi_raw = copy.deepcopy(final_epsilon_dict.get(arm_config["protected_attribute"], {}))
        final_dphi = {k: float(v) for k, v in final_dphi_raw.items()}
        final_max_dphi = float(max(final_dphi.values())) if final_dphi else 0.0

        # Build FairBiasTransformTrace according to D3 contract schema
        trace = FairBiasTransformTrace(
            algorithm_mode=fb_config.algorithm_mode,
            protected_attribute=arm_config["protected_attribute"],
            epsilon_threshold=float(epsilon_threshold),
            steps=mitigation_engine.step_traces,
            final_status="CONVERGED" if converged else "TERMINATED",
            final_max_dphi=float(final_max_dphi),
        )

        # 8. Train & Evaluate Baseline Model (Original X)
        scaler_baseline = MinMaxScaler(feature_range=(0, 1))
        X_train_base_scaled = scaler_baseline.fit_transform(X_train)
        X_val_base_scaled = scaler_baseline.transform(X_val)

        model_baseline = get_classifier("LR", random_state=random_seed)
        model_baseline.fit(X_train_base_scaled, y_train.to_numpy())

        val_prob_base = model_baseline.predict_proba(X_val_base_scaled)[:, 1]
        val_pred_base = (val_prob_base >= 0.5).astype(int)

        val_eval_base = evaluate_predictions(
            y_true=y_train.iloc[0:0] if False else y_val.to_numpy(),
            y_pred=val_pred_base,
            y_prob=val_prob_base,
            o_group=o_val.to_numpy(),
        )

        # 9. Train & Evaluate FairBias Model (Transformed X)
        # Apply the frozen train-learned changed_dict to validation
        transformed_X_val = transformer.transform_data(
            X_val, changed_dict, num_attrs, cate_attrs
        )

        scaler_fb = MinMaxScaler(feature_range=(0, 1))
        X_train_fb_scaled = scaler_fb.fit_transform(transformed_X_train)
        X_val_fb_scaled = scaler_fb.transform(transformed_X_val)

        # Fresh model instance with exact same fixed parameters
        model_fairbias = get_classifier("LR", random_state=random_seed)
        model_fairbias.fit(X_train_fb_scaled, y_train.to_numpy())

        val_prob_fb = model_fairbias.predict_proba(X_val_fb_scaled)[:, 1]
        val_pred_fb = (val_prob_fb >= 0.5).astype(int)

        val_eval_fb = evaluate_predictions(
            y_true=y_val.to_numpy(),
            y_pred=val_pred_fb,
            y_prob=val_prob_fb,
            o_group=o_val.to_numpy(),
        )

        # 10. Paired Before-vs-After Comparison
        val_comparison = compute_evaluation_comparison(val_eval_base, val_eval_fb)

        # 11. Write Run Artifacts
        # 1. arm_config.json
        arm_config_payload = {
            **arm_config,
            "algorithm_mode": fb_config.algorithm_mode,
            "survey_weighting": "NONE",
            "classifier": {
                "type": "LR",
                "random_state": random_seed,
                "max_iter": 1000,
                "solver": "lbfgs",
            },
            "eval_norm": "min-max",
            "prediction_threshold": 0.5,
            "active_feature_count": len(active_feats),
            "categorical_count": len(cate_attrs),
            "numerical_count": len(num_attrs),
            "categorical_features": cate_attrs,
            "numerical_features": num_attrs,
        }
        write_json_atomic(target_out_dir / "arm_config.json", arm_config_payload)

        # 2. train_fairbias_trace.json
        write_json_atomic(target_out_dir / "train_fairbias_trace.json", trace.to_dict())

        # 3. train_dphi_before_after.json
        dropped_features = [k for k, v in changed_dict.items() if v == "dropped"]
        surviving_features = [f for f in active_feats if f not in dropped_features]
        dphi_record = {
            "arm_id": arm_config["arm_id"],
            "protected_attribute": arm_config["protected_attribute"],
            "epsilon_threshold": float(epsilon_threshold),
            "initial_max_dphi": float(initial_max_dphi),
            "final_max_dphi": float(final_max_dphi),
            "highest_initial_feature": highest_initial_feature,
            "initial_dphi": initial_dphi,
            "final_dphi": final_dphi,
            "termination_reason": termination_reason,
            "converged": bool(converged),
            "non_convergence": mitigation_engine.non_convergence,
            "final_changed_dict": changed_dict,
            "dropped_features": dropped_features,
            "surviving_features": surviving_features,
            "accepted_steps_count": len(mitigation_engine.step_traces),
        }
        write_json_atomic(target_out_dir / "train_dphi_before_after.json", dphi_record)

        # 4. final_changed_dict.json
        write_json_atomic(target_out_dir / "final_changed_dict.json", changed_dict)

        # 5. validation_metrics_baseline.json
        write_json_atomic(target_out_dir / "validation_metrics_baseline.json", val_eval_base)

        # 6. validation_metrics_fairbias.json
        write_json_atomic(target_out_dir / "validation_metrics_fairbias.json", val_eval_fb)

        # 7. validation_group_metrics_baseline.csv
        base_group_df = pd.DataFrame(val_eval_base["group_metrics"])
        write_csv_atomic(base_group_df, target_out_dir / "validation_group_metrics_baseline.csv", force=True)

        # 8. validation_group_metrics_fairbias.csv
        fb_group_df = pd.DataFrame(val_eval_fb["group_metrics"])
        write_csv_atomic(fb_group_df, target_out_dir / "validation_group_metrics_fairbias.csv", force=True)

        # 9. validation_comparison.json
        write_json_atomic(target_out_dir / "validation_comparison.json", val_comparison)

        # 12. Build and Write d4_preflight_manifest.json
        output_files = [
            "arm_config.json",
            "train_fairbias_trace.json",
            "train_dphi_before_after.json",
            "final_changed_dict.json",
            "validation_metrics_baseline.json",
            "validation_metrics_fairbias.json",
            "validation_group_metrics_baseline.csv",
            "validation_group_metrics_fairbias.csv",
            "validation_comparison.json",
        ]
        output_hashes = {}
        for fname in output_files:
            fpath = target_out_dir / fname
            output_hashes[fname] = {
                "sha256": compute_sha256(fpath),
                "size_bytes": fpath.stat().st_size,
            }

        features_pq_sha = (
            compute_sha256(self.adapter.features_parquet_path)
            if self.adapter.features_parquet_path.is_file()
            else "UNKNOWN"
        )
        split_manifest_sha = (
            compute_sha256(self.split_manifest_path)
            if self.split_manifest_path.is_file()
            else "UNKNOWN"
        )
        d3_manifest_sha = (
            compute_sha256(self.d3_manifest_path)
            if self.d3_manifest_path.is_file()
            else "UNKNOWN"
        )

        manifest_payload = {
            "schema_version": "nhis-fairbias-d4-preflight-1.0",
            "gate": "D4.0",
            "run_id": run_id,
            "timestamp": utc_timestamp(),
            "git_commit": get_git_commit(_REPO_ROOT),
            "input_provenance": {
                "d3_manifest_sha256": d3_manifest_sha,
                "features_parquet_sha256": features_pq_sha,
                "split_manifest_sha256": split_manifest_sha,
            },
            "algorithm_mode": fb_config.algorithm_mode,
            "algorithm_random_seed": random_seed,
            "classifier_specification": {
                "classifier": "LR",
                "random_state": random_seed,
                "max_iter": 1000,
                "solver": "lbfgs",
                "eval_norm": "min-max",
                "prediction_threshold": 0.5,
            },
            "arm_id": arm_config["arm_id"],
            "outcome": arm_config["outcome"],
            "protected_attribute": arm_config["protected_attribute"],
            "feature_set": arm_config["feature_set"],
            "disability_arm": arm_config["disability_arm"],
            "feature_count": len(active_feats),
            "categorical_feature_count": len(cate_attrs),
            "numerical_feature_count": len(num_attrs),
            "train_cohort_n_valid": len(X_train),
            "validation_cohort_n_valid": len(X_val),
            "test_evaluated": False,
            "authorized_partitions": ["train", "validation"],
            "termination_semantics": {
                "exit_reason": exit_reason,
                "termination_reason": termination_reason,
                "converged": bool(converged),
                "terminal_iteration": len(mitigation_engine.step_traces),
                "initial_epsilon_threshold": float(epsilon_threshold),
                "initial_max_dphi": float(initial_max_dphi),
                "final_max_dphi": float(final_max_dphi),
                "highest_initial_feature": highest_initial_feature,
                "non_convergence": mitigation_engine.non_convergence,
            },
            "output_artifacts": output_hashes,
            "execution_time_seconds": round(time.time() - start_time, 3),
            "status": "PASS",
        }

        manifest_path = target_out_dir / "d4_preflight_manifest.json"
        write_json_atomic(manifest_path, manifest_payload)

        return {
            "run_id": run_id,
            "output_dir": str(target_out_dir),
            "manifest_path": str(manifest_path),
            "manifest": manifest_payload,
            "arm_config": arm_config_payload,
            "dphi_record": dphi_record,
            "val_eval_baseline": val_eval_base,
            "val_eval_fairbias": val_eval_fb,
            "val_comparison": val_comparison,
        }
