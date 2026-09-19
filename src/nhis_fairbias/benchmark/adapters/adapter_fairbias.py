"""F-only FairBias BM adapter for the NHIS application benchmark."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fairbias.config import ALGORITHM_MODE_PAPER_FAITHFUL, FairBiasConfig
from fairbias.evaluator import FairEvaluator
from fairbias.mitigation import FairBiasMitigation
from fairbias.prediction_contracts import validate_and_extract_positive_probabilities
from fairbias.transform import FairTransform, calculate_nmi_dict
from nhis_fairbias.benchmark.preprocessing import BenchmarkPreprocessor, NUMERICAL_FEATURES

from .base import BaseMethodAdapter
from .estimators import make_estimator
from .geometry_audit import audited_mds


class _GeometryBudgetExceeded(RuntimeError):
    pass


class FairBiasAdapter(BaseMethodAdapter):
    """Learn FairBias BM on F, then fit a matched downstream classifier."""

    name: str = "FAIRBIAS_BM"
    literature_reference: str = "Tang, Lu & Li (2024) FairBias"
    upstream_implementation: str = "src/fairbias (mitigation, bias_metric, transform)"
    supports_arm2: bool = True
    requires_sensitive_at_predict: bool = False
    output_type: str = "event_probability_p"

    def __init__(self, arm_id: Optional[str] = None, C: float = 1.0, max_iterations: int = 5,
                 phi_threshold: float = 100.0, random_state: int = 42, epsilon_ratio: float = 0.5,
                 algorithm_version: str = "application_v1", backbone: str = "LR",
                 estimator_params: Optional[Dict[str, Any]] = None,
                 max_geometry_evaluations: int = 20000, geometry_profile: str = "stress_elbow"):
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if not np.isfinite(epsilon_ratio) or epsilon_ratio <= 0:
            raise ValueError("epsilon_ratio must be finite and positive")
        if max_geometry_evaluations <= 0:
            raise ValueError("max_geometry_evaluations must be positive")
        self.arm_id = str(arm_id).lower() if arm_id is not None else None
        self.C, self.max_iterations = float(C), int(max_iterations)
        self.phi_threshold, self.random_state = float(phi_threshold), int(random_state)
        self.epsilon_ratio, self.algorithm_version = float(epsilon_ratio), str(algorithm_version)
        self.backbone = str(backbone).upper()
        self.estimator_params = dict(estimator_params or {})
        self.max_geometry_evaluations = int(max_geometry_evaluations)
        if geometry_profile not in {"stress_elbow", "fixed2_same_epsilon", "fixed2_own_epsilon"}:
            raise ValueError("Unknown registered geometry profile")
        self.geometry_profile = geometry_profile
        self.clf = make_estimator(self.backbone, C=self.C, random_state=self.random_state,
                                  estimator_params=self.estimator_params)
        self.transformer: Optional[FairTransform] = None
        self.preprocessor_: Optional[BenchmarkPreprocessor] = None
        self.changed_dict_: Dict[str, Any] = {}
        self.fitted_ = False
        self.fit_manifest_: Dict[str, Any] = {}
        self.transform_trace_: list[Dict[str, Any]] = []
        self.termination_reason_: Optional[str] = None
        self.converged_ = False
        self.semantic_columns_: tuple[str, ...] = ()
        self.semantic_num_attrs_: tuple[str, ...] = ()
        self.semantic_cat_attrs_: tuple[str, ...] = ()
        self.numeric_medians_: Dict[str, float] = {}
        self.geometry_evaluations_: int = 0

    def _prepare_fit_semantic(self, X: pd.DataFrame) -> pd.DataFrame:
        """Freeze F-only semantic missingness handling before BM geometry."""
        if X.columns.has_duplicates:
            raise ValueError("X_semantic columns must be unique")
        forbidden = {"year", "WTFA_A", "PSTRAT", "PPSU", "record_key"} & set(X.columns)
        if forbidden:
            raise ValueError(f"Prediction/design columns cannot be FairBias features: {sorted(forbidden)}")
        out = X.copy()
        self.numeric_medians_ = {}
        drop_numeric = []
        for col in [c for c in out.columns if c in NUMERICAL_FEATURES]:
            values = pd.to_numeric(out[col], errors="coerce")
            bad_parse = values.isna() & out[col].notna()
            if bool(bad_parse.any()):
                raise ValueError(f"Non-numeric values in numeric feature {col!r}")
            if bool((~values.isna() & ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))).any()):
                raise ValueError(f"Non-finite values in numeric feature {col!r}")
            if values.notna().sum() == 0:
                drop_numeric.append(col)
                continue
            values = pd.Series(values.to_numpy(dtype=float, na_value=np.nan), index=out.index)
            median = float(values.median())
            self.numeric_medians_[col] = median
            out[col] = pd.Series(values.to_numpy(dtype=float, na_value=np.nan), index=out.index).fillna(median)
        if drop_numeric:
            out = out.drop(columns=drop_numeric)
        self.semantic_columns_ = tuple(out.columns)
        self.semantic_num_attrs_ = tuple(c for c in self.semantic_columns_ if c in self.numeric_medians_)
        self.semantic_cat_attrs_ = tuple(c for c in self.semantic_columns_ if c not in self.semantic_num_attrs_)
        for col in self.semantic_cat_attrs_:
            out[col] = out[col].astype(object).where(out[col].notna(), "MISSING").fillna("MISSING").astype(str)
        return out

    def _prepare_predict_semantic(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.semantic_columns_) - set(X.columns))
        if missing:
            raise ValueError(f"Prediction semantic data missing frozen F features: {missing}")
        forbidden = {"year", "WTFA_A", "PSTRAT", "PPSU", "record_key"} & set(X.columns)
        if forbidden:
            raise ValueError(f"Prediction/design columns cannot be FairBias features: {sorted(forbidden)}")
        out = X.loc[:, list(self.semantic_columns_)].copy()
        for col, median in self.numeric_medians_.items():
            values = pd.to_numeric(out[col], errors="coerce")
            bad_parse = values.isna() & out[col].notna()
            if bool(bad_parse.any()):
                raise ValueError(f"Non-numeric values in numeric feature {col!r}")
            if bool((~values.isna() & ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))).any()):
                raise ValueError(f"Non-finite values in numeric feature {col!r}")
            out[col] = pd.Series(values.to_numpy(dtype=float, na_value=np.nan), index=out.index).fillna(median)
        for col in self.semantic_cat_attrs_:
            out[col] = out[col].astype(object).where(out[col].notna(), "MISSING").fillna("MISSING").astype(str)
        return out

    def _encoder_num_attrs(self, transformed: pd.DataFrame) -> list[str]:
        nums = []
        for col in transformed.columns:
            change = self.changed_dict_.get(col)
            categorical_merge = isinstance(change, dict) and "power" not in change
            if col in self.semantic_num_attrs_ and not categorical_merge:
                nums.append(col)
        return nums

    @staticmethod
    def _fingerprint_frame(X: pd.DataFrame, y: np.ndarray, A: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
        h.update(np.asarray(y).tobytes())
        h.update(np.asarray(A).tobytes())
        return h.hexdigest()

    @staticmethod
    def _max_epsilon(epsilon_dict: Dict[str, Dict[str, float]]) -> float:
        values = [float(v) for group in epsilon_dict.values() for v in group.values() if np.isfinite(v) and v >= 0]
        return float(max(values)) if values else 0.0

    @staticmethod
    def _validate_epsilon(epsilon_dict: Dict[str, Dict[str, float]], stage: str) -> None:
        values = [float(v) for group in epsilon_dict.values() for v in group.values()]
        if not values or any(not np.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"NOT_ESTIMABLE: {stage} d_phi table is empty or contains non-finite/negative values")

    def _learn_transform(self, X_df: pd.DataFrame, y: np.ndarray, A: np.ndarray) -> pd.DataFrame:
        """Run the repository BM engine strictly on the fitting partition F."""
        features = list(X_df.columns)
        num_attrs = [f for f in features if f in self.semantic_num_attrs_]
        cate_attrs = [f for f in features if f in self.semantic_cat_attrs_]
        y_s = pd.Series(np.asarray(y, dtype=int), index=X_df.index, name="target")
        o_s = pd.Series(np.asarray(A), index=X_df.index, name="__protected__")
        config = FairBiasConfig(algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL, random_seed=self.random_state,
            classifier="LR", eval_norm="min-max", label_O=("__protected__",), label_Y="target",
            use_bias_mitigation=True, use_accuracy_enhancement=False, phi_threshold=self.phi_threshold,
            failed_attribute_mode="stop", power_sequence_policy="official_stream", power_revisit_policy="restart",
            multigroup_aggregation="author_max_pair").resolved()
        if self.geometry_profile != "stress_elbow":
            config = dataclasses.replace(config, mds_fixed_components=2)
        evaluator = FairEvaluator(config=config, label_O=["__protected__"], label_Y="target",
                                  cate_attrs=cate_attrs, num_attrs=num_attrs)
        original_calculate_epsilon = evaluator.calculate_epsilon
        self.geometry_evaluations_ = 0

        def counted_calculate_epsilon(*args, **kwargs):
            self.geometry_evaluations_ += 1
            if self.geometry_evaluations_ > self.max_geometry_evaluations:
                raise _GeometryBudgetExceeded(
                    f"FairBias geometry budget exhausted at {self.max_geometry_evaluations} evaluations"
                )
            return original_calculate_epsilon(*args, **kwargs)

        evaluator.calculate_epsilon = counted_calculate_epsilon  # type: ignore[method-assign]
        transformer = FairTransform(n_bins=config.transform_n_bins, log_epsilon=config.transform_log_epsilon,
                                    x_max=config.transform_x_max)
        O_df = pd.DataFrame({"__protected__": o_s})
        nmi_org = calculate_nmi_dict(X_df, y_s)
        initial_epsilon = evaluator.calculate_epsilon(X_df, O_df, cate_attrs=cate_attrs, num_attrs=num_attrs, sample_weight=None)
        try:
            self._validate_epsilon(initial_epsilon, "initial F")
        except ValueError:
            self.fit_manifest_ = {"status": "NOT_ESTIMABLE", "source_partition": "F", "stage": "initial"}
            raise
        initial_max = self._max_epsilon(initial_epsilon)
        reference_epsilon = initial_epsilon
        if self.geometry_profile == "fixed2_same_epsilon":
            # Same F, seed and stress-elbow reference as the anchor: only the
            # optimized geometry changes. The own-epsilon variant changes both.
            reference_evaluator = FairEvaluator(config=dataclasses.replace(config, mds_fixed_components=None),
                label_O=["__protected__"], label_Y="target", cate_attrs=cate_attrs, num_attrs=num_attrs)
            self.geometry_evaluations_ += 1
            if self.geometry_evaluations_ > self.max_geometry_evaluations:
                raise _GeometryBudgetExceeded("BUDGET_EXHAUSTED: reference geometry")
            reference_epsilon = reference_evaluator.calculate_epsilon(X_df, O_df, cate_attrs=cate_attrs, num_attrs=num_attrs)
            self._validate_epsilon(reference_epsilon, "stress-elbow reference F")
        ref = [float(v) for group in reference_epsilon.values() for v in group.values() if np.isfinite(v) and v >= 0]
        epsilon_reference = float(np.mean(ref)) if ref else 0.0
        if not ref:
            self.fit_manifest_ = {"status": "NOT_ESTIMABLE", "source_partition": "F"}
            raise ValueError("NOT_ESTIMABLE: F epsilon reference is empty")
        epsilon_threshold = epsilon_reference * self.epsilon_ratio
        if epsilon_threshold <= 0 and initial_max > 0:
            self.fit_manifest_ = {"status": "NOT_ESTIMABLE", "source_partition": "F"}
            raise ValueError("NOT_ESTIMABLE: nonzero F d_phi has zero epsilon reference")
        engine = FairBiasMitigation(evaluator=evaluator, transformer=transformer, label_O=["__protected__"],
            cate_attrs=cate_attrs, num_attrs=num_attrs, phi_threshold=self.phi_threshold,
            poly_exponents=config.transform_poly_exponents, failed_attribute_mode=config.failed_attribute_mode,
            power_sequence_policy=config.power_sequence_policy, power_revisit_policy=config.power_revisit_policy)
        changed: Dict[str, Any] = {}
        current_epsilon = copy.deepcopy(initial_epsilon)
        transformed = X_df.copy()
        termination = "epsilon_reached" if initial_max <= epsilon_threshold else None
        iteration = 0
        while termination is None:
            if iteration >= self.max_iterations:
                termination = "budget_exhausted"
                break
            iteration += 1
            try:
                transformed, changed, _, selected_attr = engine.mitigate_step(X=X_df, Y=y_s, O=O_df, nmi_org=nmi_org,
                    changed_dict=changed, current_epsilon=current_epsilon, epsilon_threshold=epsilon_threshold, iteration=iteration)
            except _GeometryBudgetExceeded:
                termination = "budget_exhausted"
                break
            if selected_attr is None:
                termination = "epsilon_reached" if self._max_epsilon(current_epsilon) <= epsilon_threshold else "candidate_exhausted"
                break
            transformed = transformer.transform_data(X_df, changed, num_attrs, cate_attrs)
            try:
                current_epsilon = evaluator.calculate_epsilon(transformed, O_df, cate_attrs=cate_attrs, num_attrs=num_attrs, sample_weight=None)
            except _GeometryBudgetExceeded:
                termination = "budget_exhausted"
                break
            try:
                self._validate_epsilon(current_epsilon, "intermediate F")
            except ValueError:
                self.fit_manifest_ = {"status": "NOT_ESTIMABLE", "source_partition": "F", "stage": "intermediate"}
                raise
            if self._max_epsilon(current_epsilon) <= epsilon_threshold:
                termination = "epsilon_reached"
        termination = termination or "candidate_exhausted"
        if termination == "budget_exhausted":
            final_epsilon = current_epsilon
        else:
            try:
                final_epsilon = evaluator.calculate_epsilon(transformed, O_df, cate_attrs=cate_attrs, num_attrs=num_attrs, sample_weight=None)
            except _GeometryBudgetExceeded:
                termination = "budget_exhausted"
                final_epsilon = current_epsilon
            try:
                self._validate_epsilon(final_epsilon, "final F")
            except ValueError:
                self.fit_manifest_ = {"status": "NOT_ESTIMABLE", "source_partition": "F", "stage": "final"}
                raise
        final_max = self._max_epsilon(final_epsilon)
        if final_max <= epsilon_threshold and termination != "budget_exhausted":
            termination = "epsilon_reached"
        elif termination == "epsilon_reached":
            termination = "candidate_exhausted"
        self.transformer = transformer
        self.changed_dict_ = copy.deepcopy(changed)
        self.transform_trace_ = [step.to_dict() for step in engine.step_traces]
        self.termination_reason_ = termination
        self.converged_ = termination == "epsilon_reached"
        self.fit_manifest_ = {"algorithm_version": self.algorithm_version, "algorithm_mode": config.algorithm_mode,
            "geometry_profile": self.geometry_profile, "reference_epsilon": reference_epsilon,
            "resolved_config": dataclasses.asdict(config),
            "geometry_classifier_field_unused": True,
            "downstream_backbone": self.backbone, "downstream_params": self.clf.get_params(),
            "arm_id": self.arm_id, "source_partition": "F", "input_fingerprint": self._fingerprint_frame(X_df, y, A),
            "epsilon_reference": epsilon_reference, "epsilon_ratio": self.epsilon_ratio,
            "epsilon_threshold": epsilon_threshold, "initial_max_dphi": initial_max,
            "final_max_dphi": final_max, "iterations": iteration,
            "geometry_evaluations": self.geometry_evaluations_,
            "geometry_source": "FairEvaluator.calculate_epsilon",
            "termination_reason": termination, "converged": self.converged_,
            "changed_dict": copy.deepcopy(changed), "transform_trace": copy.deepcopy(self.transform_trace_),
            "non_convergence": copy.deepcopy(engine.non_convergence)}
        return transformed

    def fit(self, X: np.ndarray, y: np.ndarray, A: np.ndarray, sample_weight: Optional[np.ndarray] = None,
            X_semantic: Optional[pd.DataFrame] = None) -> "FairBiasAdapter":
        if X_semantic is None and isinstance(X, pd.DataFrame):
            X_semantic = X
        self.fit_representation(X_semantic, y, A)
        return self.fit_predictor(X_semantic, y, sample_weight=sample_weight)

    def fit_representation(self, X_semantic: pd.DataFrame, y: np.ndarray, A: np.ndarray) -> "FairBiasAdapter":
        """Learn and freeze the F representation independently of the backbone grid."""
        self.fitted_ = False
        self.preprocessor_ = None
        self.transformer = None
        self.changed_dict_ = {}
        self.fit_manifest_ = {}
        self.transform_trace_ = []
        self.termination_reason_ = None
        self.converged_ = False
        if X_semantic is None or not isinstance(X_semantic, pd.DataFrame):
            raise TypeError("FairBiasAdapter requires X_semantic DataFrame; numeric arrays cannot be used for FairBias geometric fitting")
        if len(X_semantic) != len(y) or len(X_semantic) != len(A):
            raise ValueError("X_semantic, y, and A must have identical row counts")
        X_f_semantic = self._prepare_fit_semantic(X_semantic)
        self.mds_diagnostics_ = []
        try:
            with audited_mds(self.mds_diagnostics_, lambda: self.geometry_evaluations_):
                transformed_F = self._learn_transform(X_f_semantic, y, A)
        finally:
            self.fit_manifest_["mds_fits"] = self.mds_diagnostics_
        if self.fit_manifest_.get("termination_reason") == "budget_exhausted":
            raise RuntimeError("BUDGET_EXHAUSTED: FairBias fit did not produce a valid model")
        transformed_num = self._encoder_num_attrs(transformed_F)
        self.preprocessor_ = BenchmarkPreprocessor(list(transformed_F.columns), numerical_features=transformed_num)
        self.preprocessor_.fit(transformed_F)
        return self

    def fit_predictor(self, X_semantic: pd.DataFrame, y: np.ndarray,
                      sample_weight: Optional[np.ndarray] = None) -> "FairBiasAdapter":
        if self.preprocessor_ is None or self.transformer is None:
            raise RuntimeError("F representation must be fitted before the predictor")
        X_sem = self._prepare_predict_semantic(X_semantic)
        transformed_F = self.transformer.transform_data(X_sem, self.changed_dict_, list(self.semantic_num_attrs_), list(self.semantic_cat_attrs_))
        self.clf = make_estimator(self.backbone, C=self.C, random_state=self.random_state,
                                  estimator_params=self.estimator_params)
        self.clf.fit(self.preprocessor_.transform(transformed_F), y, sample_weight=sample_weight)
        self.fit_manifest_["downstream_backbone"] = self.backbone
        self.fit_manifest_["downstream_params"] = self.clf.get_params()
        self.fitted_ = True
        return self

    def _transform_input(self, X: np.ndarray, X_semantic: Optional[pd.DataFrame] = None) -> np.ndarray:
        if not self.fitted_ or self.preprocessor_ is None or self.transformer is None:
            raise RuntimeError("FairBiasAdapter must be fitted before predicting.")
        if X_semantic is None and isinstance(X, pd.DataFrame):
            X_semantic = X
        if X_semantic is None or not isinstance(X_semantic, pd.DataFrame):
            raise TypeError("FairBiasAdapter prediction requires X_semantic DataFrame")
        X_sem = self._prepare_predict_semantic(X_semantic)
        transformed = self.transformer.transform_data(X_sem, self.changed_dict_, list(self.semantic_num_attrs_), list(self.semantic_cat_attrs_))
        return self.preprocessor_.transform(transformed)

    def predict(self, X: np.ndarray, A: Optional[np.ndarray] = None, X_semantic: Optional[pd.DataFrame] = None) -> np.ndarray:
        return self.clf.predict(self._transform_input(X, X_semantic))

    def predict_proba(self, X: np.ndarray, A: Optional[np.ndarray] = None, X_semantic: Optional[pd.DataFrame] = None) -> np.ndarray:
        X_enc = self._transform_input(X, X_semantic)
        return self.clf.predict_proba(X_enc)

    def predict_decision_proba(self, X: np.ndarray, A: Optional[np.ndarray] = None, X_semantic: Optional[pd.DataFrame] = None) -> np.ndarray:
        X_enc = self._transform_input(X, X_semantic)
        return validate_and_extract_positive_probabilities(self.clf, X_enc, pos_label=1)
