"""Benchmark V1 execution orchestrator running F -> C -> S -> T pipeline.

Coordinates:
1. Synthetic cohort generation & arm-specific partitioning (F, C, S, T).
2. Leakage-free 3-tier preprocessing fit on F and applied to C, S, T.
3. Model fitting on F, threshold calibration on C.
4. Candidate evaluation and configuration selection on S.
5. Frozen evaluation on T with rescaled PSU bootstrap survey inference.
6. Paired contrast calculations against FairBias.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .adapters import (
    BaseMethodAdapter,
    ExponentiatedGradientAdapter,
    FairBiasAdapter,
    LFRAdapter,
    NotSupportedError,
    ReweighingAdapter,
    ThresholdOptimizerAdapter,
    UnmitigatedAdapter,
)
from .data_contracts import ARM_SPECS, PartitionDataset, load_arm_partitions
from .metrics import compute_survey_fairness_metrics
from .preprocessing import BenchmarkPreprocessor
from .predictions import FrozenDecisionPolicy
from .risk_metrics import compute_risk_metrics
from .selection import select_best_configuration_on_S
from .survey_inference import evaluate_with_survey_bootstrap


@dataclasses.dataclass
class ArmBenchmarkResult:
    """Benchmark execution result for a single experimental arm."""

    arm_id: str
    protected_attribute: str
    expected_categories: List[int]
    sample_sizes: Dict[str, int]
    method_results: Dict[str, Dict[str, Any]]
    paired_contrasts: Dict[str, Dict[str, Any]]
    execution_time_seconds: float


class BenchmarkRunner:
    """Benchmark orchestrator executing standardized multi-method fairness evaluations."""

    def __init__(
        self,
        bootstrap_B: int = 2000,
        budget_tau: float = 0.10,
        random_seed: int = 42,
        survey_seed: int = 20260914,
        method_factories: Optional[Dict[str, Callable[[], BaseMethodAdapter]]] = None,
    ):
        self.bootstrap_B = int(bootstrap_B)
        self.budget_tau = float(budget_tau)
        self.random_seed = int(random_seed)
        self.survey_seed = int(survey_seed)
        self.method_factories = method_factories

    def run_arm(
        self,
        df_cohort: pd.DataFrame,
        arm_id: str,
    ) -> ArmBenchmarkResult:
        """Execute full benchmark pipeline for a single experimental arm."""
        t0 = time.time()
        spec = ARM_SPECS[arm_id]
        prot_attr = spec["protected_attribute"]
        expected_cats = spec["expected_categories"]

        # Step 1: Master Survey Design Partitioning (F, C, S, T)
        partitions = load_arm_partitions(df_cohort, arm_id)
        p_F = partitions["fitting_F"]
        p_C = partitions["calibration_C"]
        p_S = partitions["selection_S"]
        p_T = partitions["evaluation_T"]
        if any(len(p) == 0 for p in partitions.values()):
            raise ValueError("NOT_ESTIMABLE: an F/C/S/T partition is empty under the frozen split")

        sample_sizes = {
            "F": len(p_F),
            "C": len(p_C),
            "S": len(p_S),
            "T": len(p_T),
        }

        # Step 2: 3-tier Preprocessing fit strictly on Partition F
        preprocessor = BenchmarkPreprocessor(spec["features"])
        preprocessor.fit(p_F.X_semantic)

        X_F = preprocessor.transform(p_F.X_semantic)
        X_C = preprocessor.transform(p_C.X_semantic)
        X_S = preprocessor.transform(p_S.X_semantic)

        # Step 3: Instantiate Benchmark Methods
        methods: Dict[str, BaseMethodAdapter] = {
            "UNMITIGATED": UnmitigatedAdapter(C=1.0, random_state=self.random_seed),
            "FAIRBIAS_BM": FairBiasAdapter(arm_id=arm_id, C=1.0, max_iterations=50, random_state=self.random_seed),
            "REWEIGHING": ReweighingAdapter(C=1.0, random_state=self.random_seed),
            "EG_DP": ExponentiatedGradientAdapter(
                constraint_type="demographic_parity", eps=0.05, max_iter=20, random_state=self.random_seed
            ),
            "EG_EO": ExponentiatedGradientAdapter(
                constraint_type="equalized_odds", eps=0.05, max_iter=20, random_state=self.random_seed
            ),
            "TO_EO": ThresholdOptimizerAdapter(C=1.0, random_state=self.random_seed),
        }

        # LFR only supported for binary arms
        if arm_id != "arm_002":
            methods["LFR_RECONSTRUCTED"] = LFRAdapter(k=5, random_state=self.random_seed)
        if self.method_factories is not None:
            methods = {name: factory() for name, factory in self.method_factories.items()}

        # Step 4: Model Fitting on F & Calibration on C
        fitted_models: Dict[str, BaseMethodAdapter] = {}
        method_statuses: Dict[str, str] = {}
        policies: Dict[str, FrozenDecisionPolicy] = {}

        for name, adapter in methods.items():
            try:
                if name in ("TO_EO", "OXONFAIR_EO"):
                    # Two-stage: fit base on F, calibrate thresholds on C
                    to_adapter: ThresholdOptimizerAdapter = adapter  # type: ignore
                    to_adapter.fit_base(X_F, p_F.y)
                    to_adapter.calibrate(X_C, p_C.y, p_C.A)
                    fitted_models[name] = to_adapter
                else:
                    if name == "FAIRBIAS_BM":
                        adapter.fit(X_F, p_F.y, p_F.A, X_semantic=p_F.X_semantic)
                        if not adapter.converged_:
                            method_statuses[name] = str(adapter.termination_reason_).upper()
                            continue
                    else:
                        adapter.fit(X_F, p_F.y, p_F.A)
                    fitted_models[name] = adapter
                policies[name] = FrozenDecisionPolicy().fit_calibration(
                    adapter, p_C.X_semantic if name == "FAIRBIAS_BM" else X_C,
                    p_C.y, p_C.A, metadata={"role": "calibration_C", "year": 2022})
                method_statuses[name] = "VALID"
            except NotSupportedError:
                fitted_models.pop(name, None)
                method_statuses[name] = "NOT_SUPPORTED"
            except Exception as e:
                fitted_models.pop(name, None)
                method_statuses[name] = f"ERROR: {str(e)}"

        if arm_id == "arm_002" and self.method_factories is None:
            method_statuses["LFR_RECONSTRUCTED"] = "NOT_SUPPORTED"

        # Step 5: Candidate Evaluation on Set S (2023)
        predictions_S: Dict[str, np.ndarray] = {}
        for name, model in fitted_models.items():
            bundle = policies[name].predict(p_S.X_semantic if name == "FAIRBIAS_BM" else X_S, A=p_S.A)
            predictions_S[name] = bundle.q_decision

        # Evaluate S selection
        # This bounded runner has one configuration per method. Never select
        # one method as a substitute for per-method hyperparameter selection.
        selections = {name: select_best_configuration_on_S(
            {name: prediction},
            p_S.y,
            p_S.A,
            p_S.WTFA_A,
            expected_groups=expected_cats,
            budget_tau=self.budget_tau,
        ) for name, prediction in predictions_S.items()}

        # Step 6: Frozen Forward Evaluation on Set T (2024)
        predictions_T: Dict[str, np.ndarray] = {}
        X_T = preprocessor.transform(p_T.X_semantic)
        bundles_T = {}
        for name, model in fitted_models.items():
            bundle = policies[name].predict(p_T.X_semantic if name == "FAIRBIAS_BM" else X_T, A=p_T.A)
            bundles_T[name] = bundle
            predictions_T[name] = bundle.q_decision

        # Step 7: Rescaled PSU Bootstrap Survey Inference
        design = p_T.annual_design
        if design is None:
            raise ValueError("Complete annual design is required for survey inference")
        survey_results = evaluate_with_survey_bootstrap(
            y_true=design.expand(p_T.y),
            predictions_dict={name: design.expand(q) for name, q in predictions_T.items()},
            A=design.expand(p_T.A),
            strata=design.strata,
            psus=design.psus,
            weights=design.weights,
            domain_mask=design.domain_mask,
            expected_groups=expected_cats,
            B=self.bootstrap_B,
            seed=self.survey_seed,
        )

        # Format method results
        method_summary: Dict[str, Dict[str, Any]] = {}
        for name in list(methods.keys()) + (["LFR_RECONSTRUCTED"] if arm_id == "arm_002" and self.method_factories is None else []):
            if method_statuses.get(name) == "NOT_SUPPORTED":
                method_summary[name] = {
                    "status": "NOT_SUPPORTED",
                    "balanced_accuracy": np.nan,
                    "ba_se": np.nan,
                    "ba_ci_lower": np.nan,
                    "ba_ci_upper": np.nan,
                    "dp_gap": np.nan,
                    "dp_se": np.nan,
                    "eo_gap": np.nan,
                    "eo_se": np.nan,
                    "eo_ci_lower": np.nan,
                    "eo_ci_upper": np.nan,
                }
            elif name in survey_results["method_inference"]:
                mi = survey_results["method_inference"][name]
                method_summary[name] = {
                    "status": "VALID" if all(v.status == "VALID" for v in mi.values()) else "NOT_ESTIMABLE",
                    "selection_status": selections[name].status,
                    "selection_scope": "single_configuration_smoke_only",
                    "threshold": policies[name].threshold,
                    "prediction_output_type": methods[name].output_type,
                    "risk_metrics": compute_risk_metrics(p_T.y, bundles_T[name], p_T.WTFA_A,
                                                          p_T.A, expected_cats),
                    "balanced_accuracy": mi["balanced_accuracy"].point_estimate,
                    "ba_se": mi["balanced_accuracy"].std_error,
                    "ba_ci_lower": mi["balanced_accuracy"].ci_lower,
                    "ba_ci_upper": mi["balanced_accuracy"].ci_upper,
                    "dp_gap": mi["dp_gap"].point_estimate,
                    "dp_se": mi["dp_gap"].std_error,
                    "eo_gap": mi["eo_gap"].point_estimate,
                    "eo_se": mi["eo_gap"].std_error,
                    "eo_ci_lower": mi["eo_gap"].ci_lower,
                    "eo_ci_upper": mi["eo_gap"].ci_upper,
                }
            else:
                method_summary[name] = {"status": survey_results.get("status") if name in fitted_models else method_statuses.get(name, "FAILED")}

        # Format paired contrasts
        contrast_summary: Dict[str, Dict[str, Any]] = {}
        for base_name, contrasts in survey_results["paired_contrasts"].items():
            contrast_summary[base_name] = {
                "delta_ba": contrasts["balanced_accuracy"].point_contrast,
                "delta_ba_se": contrasts["balanced_accuracy"].std_error,
                "delta_ba_ci_lower": contrasts["balanced_accuracy"].ci_lower,
                "delta_ba_ci_upper": contrasts["balanced_accuracy"].ci_upper,
                "delta_ba_p_value": contrasts["balanced_accuracy"].p_value,
                "delta_eo": contrasts["eo_gap"].point_contrast,
                "delta_eo_se": contrasts["eo_gap"].std_error,
                "delta_eo_ci_lower": contrasts["eo_gap"].ci_lower,
                "delta_eo_ci_upper": contrasts["eo_gap"].ci_upper,
                "delta_eo_p_value": contrasts["eo_gap"].p_value,
            }

        elapsed = time.time() - t0
        return ArmBenchmarkResult(
            arm_id=arm_id,
            protected_attribute=prot_attr,
            expected_categories=expected_cats,
            sample_sizes=sample_sizes,
            method_results=method_summary,
            paired_contrasts=contrast_summary,
            execution_time_seconds=elapsed,
        )
