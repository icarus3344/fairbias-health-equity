"""Benchmark adapter for the paper's FairBias BM, AE, and JOINT engines.

The adapter deliberately delegates transform search to :mod:`fairbias`: the
small amount of code here only fixes the benchmark partitions, classifier
encoding, and the unweighted balanced-accuracy utility contract.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from fairbias.config import ALGORITHM_MODE_PAPER_FAITHFUL, FairBiasConfig
from fairbias.enhancement import FairAccuracyEnhancement
from fairbias.enhancement_contracts import CandidateEvaluationResult, EvaluationPartition
from fairbias.evaluator import FairEvaluator
from fairbias.mitigation import FairBiasMitigation
from fairbias.transform import FairTransform, calculate_nmi_dict
from fairbias.prediction_contracts import validate_and_extract_positive_probabilities
from nhis_fairbias.benchmark.predictions import _best_balanced_accuracy_threshold
from nhis_fairbias.benchmark.adapters.base import BaseMethodAdapter, NotSupportedError
from nhis_fairbias.benchmark.preprocessing import BenchmarkPreprocessor
from nhis_fairbias.benchmark.adapters.adapter_fairbias import FairBiasAdapter
from .geometry_audit import audited_mds


class FairBiasAEAdapter(BaseMethodAdapter):
    """Frozen-development FairBias adapter.

    ``fit_development`` is the only training entry point.  It uses F for all
    geometry and model fitting and C only for candidate utility.  S and T are
    intentionally absent from the adapter API.
    """

    name = "fairbias_bm_ae"
    literature_reference = "Tang, Lu & Li (2024), FairBias"
    upstream_implementation = "src/fairbias/mitigation.py + enhancement.py"
    requires_sensitive_at_predict = False
    output_type = "event_probability_p"

    def __init__(
        self,
        mode: str = "BM_AE",
        backbone: str = "LR",
        numerical_features: Optional[Sequence[str]] = None,
        categorical_features: Optional[Sequence[str]] = None,
        protected_name: str = "A",
        max_outer_iterations: int = 10,
        max_utility_evaluations: int = 500,
        max_geometry_evaluations: int = 20_000,
        max_bm_steps: int = 50,
        epsilon_ratio: float = 1.0,
        poly_exponents: Sequence[float] = (1 / 7, 1 / 5, 1 / 3, 3.0, 5.0, 7.0),
        random_state: int = 0,
        estimator_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.mode = str(mode).upper()
        if self.mode not in {"BM_AE", "JOINT"}:
            raise ValueError("mode must be BM_AE or JOINT")
        self.name = "FAIRBIAS_" + self.mode
        self.backbone = str(backbone).upper()
        if self.backbone not in {"LR", "GBDT"}:
            raise ValueError("backbone must be LR or GBDT")
        for value in (max_outer_iterations, max_utility_evaluations, max_geometry_evaluations, max_bm_steps):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("all search limits must be positive integers")
        if not np.isfinite(epsilon_ratio) or epsilon_ratio <= 0:
            raise ValueError("epsilon_ratio must be finite and positive")
        self.numerical_features = None if numerical_features is None else tuple(numerical_features)
        self.categorical_features = None if categorical_features is None else tuple(categorical_features)
        self.protected_name = protected_name
        self.max_outer_iterations = int(max_outer_iterations)
        self.max_utility_evaluations = int(max_utility_evaluations)
        self.max_geometry_evaluations = int(max_geometry_evaluations)
        self.max_bm_steps = int(max_bm_steps)
        self.epsilon_ratio = float(epsilon_ratio)
        self.poly_exponents = tuple(float(x) for x in poly_exponents)
        self.random_state = int(random_state)
        self.estimator_params = dict(estimator_params or {})
        self.is_fitted_ = False

    def _estimator(self):
        params = dict(self.estimator_params)
        if self.backbone == "GBDT":
            params.setdefault("n_estimators", 100)
            params.setdefault("max_depth", 2)
            params.setdefault("learning_rate", 0.05)
            params.setdefault("subsample", 1.0)
            return GradientBoostingClassifier(random_state=self.random_state, **params)
        params.setdefault("max_iter", 1000)
        params.setdefault("C", 1.0)
        return LogisticRegression(random_state=self.random_state, **params)

    @staticmethod
    def _best_ba_threshold(p: np.ndarray, y: np.ndarray) -> float:
        return float(_best_balanced_accuracy_threshold(np.asarray(p, dtype=float), np.asarray(y, dtype=int))[0])

    @staticmethod
    def _max_eps(table: Dict[str, Dict[str, float]]) -> float:
        vals = [float(v) for group in table.values() for v in group.values()]
        if not vals or any(not np.isfinite(v) or v < 0 for v in vals):
            raise ValueError("non-finite or negative d_phi")
        return float(max(vals))

    def _encode_fit(self, X: pd.DataFrame) -> np.ndarray:
        nums = [c for c in self.num_attrs_ if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
        self.preprocessor_ = BenchmarkPreprocessor(list(X.columns), numerical_features=nums)
        return self.preprocessor_.fit_transform(X)

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        return self.preprocessor_.transform(X)

    def _utility(self, partition: EvaluationPartition, changed: Dict[str, Any]) -> CandidateEvaluationResult:
        self._utility_evaluations_ = getattr(self, "_utility_evaluations_", 0) + 1
        if self._utility_evaluations_ > self.max_utility_evaluations:
            self.termination_reason_ = "BUDGET_EXHAUSTED"
            raise RuntimeError("BUDGET_EXHAUSTED: FairBias AE utility evaluations")
        try:
            tr = self.transformer_.transform_data(partition.fit_X, changed, self.num_attrs_, self.cate_attrs_)
            ev = self.transformer_.transform_data(partition.selection_X, changed, self.num_attrs_, self.cate_attrs_)
            nums = [c for c in self.num_attrs_ if c in tr.columns and pd.api.types.is_numeric_dtype(tr[c])]
            enc = BenchmarkPreprocessor(list(tr.columns), numerical_features=nums)
            Xf = enc.fit_transform(tr)
            Xc = enc.transform(ev)
            model = self._estimator()
            model.fit(Xf, np.asarray(partition.fit_y, dtype=int))
            p = validate_and_extract_positive_probabilities(model, Xc, expected_classes=(0, 1), pos_label=1)
            if not np.all(np.isfinite(p)) or partition.selection_y.nunique() < 2:
                raise ValueError("non-finite probabilities or single-class C")
            threshold = self._best_ba_threshold(p, np.asarray(partition.selection_y, dtype=int))
            score = float(balanced_accuracy_score(partition.selection_y, (p >= threshold).astype(int)))
            return CandidateEvaluationResult("VALID", score, utility_metric="balanced_accuracy", model_fit_count=1)
        except Exception as exc:
            return CandidateEvaluationResult("MODEL_FIT_FAILED", None, utility_metric="balanced_accuracy", error_message=str(exc), model_fit_count=1)

    def fit_development(
        self, X_semantic_F, yF, AF, X_semantic_C, yC, AC, metadata=None,
    ):
        self.mds_diagnostics_ = []
        try:
            with audited_mds(self.mds_diagnostics_, lambda: getattr(self, "_geometry_evaluations_", 0)):
                return self._fit_development(X_semantic_F, yF, AF, X_semantic_C, yC, AC, metadata)
        finally:
            self.provenance_["mds_fits"] = self.mds_diagnostics_

    def _fit_development(
        self, X_semantic_F: pd.DataFrame, yF: Sequence[int], AF: pd.DataFrame,
        X_semantic_C: pd.DataFrame, yC: Sequence[int], AC: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "FairBiasAEAdapter":
        self.is_fitted_ = False
        self.model_ = None
        self.preprocessor_ = None
        self.termination_reason_ = None
        self.provenance_ = {}
        self.changed_dict_ = {}
        if metadata is not None and ("F_ids" in metadata or "C_ids" in metadata):
            if not {"F_ids", "C_ids"}.issubset(metadata):
                raise ValueError("Both F and C record identities are required")
            f_ids, c_ids = metadata["F_ids"], metadata["C_ids"]
            if len(f_ids) != len(X_semantic_F) or len(c_ids) != len(X_semantic_C) or set(f_ids) & set(c_ids):
                raise ValueError("F/C record identities overlap or have mismatched lengths")
        # Reuse the benchmark's frozen semantic missingness rules before any
        # FairBias geometry is computed; C is projected through the F schema.
        self._semantic_adapter = FairBiasAdapter(backbone=self.backbone, random_state=self.random_state)
        Xf = self._semantic_adapter._prepare_fit_semantic(X_semantic_F.copy(deep=True))
        Xc = self._semantic_adapter._prepare_predict_semantic(X_semantic_C.copy(deep=True)) if self._semantic_adapter.semantic_columns_ else X_semantic_C.copy(deep=True)
        for values in (yF, yC):
            arr = np.asarray(values)
            if arr.ndim != 1 or np.iscomplexobj(arr) or not np.isin(arr, [0, 1]).all():
                raise ValueError("F and C require one-dimensional binary labels before integer conversion")
        yf, yc = pd.Series(np.asarray(yF, dtype=int), index=Xf.index), pd.Series(np.asarray(yC, dtype=int), index=Xc.index)
        if len(Xf) != len(yf) or len(Xc) != len(yc) or len(AF) != len(Xf) or len(AC) != len(Xc):
            raise ValueError("X, y, and A lengths must agree within F and C")
        if set(yf.unique()) != {0, 1} or set(yc.unique()) != {0, 1}:
            raise ValueError("F and C must contain both binary classes")
        if list(Xf.columns) != list(Xc.columns):
            raise ValueError("F and C feature columns must match")
        self.cate_attrs_ = list(self.categorical_features or [c for c in Xf.columns if not pd.api.types.is_numeric_dtype(Xf[c])])
        self.num_attrs_ = list(self.numerical_features or [c for c in Xf.columns if c not in self.cate_attrs_])
        self.transformer_ = FairTransform()
        cfg = FairBiasConfig(algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL, classifier="LR", random_seed=self.random_state,
                             multigroup_aggregation="author_max_pair",
                             eval_norm="min-max", label_O=(self.protected_name,), label_Y="__y__",
                             failed_attribute_mode="stop", power_sequence_policy="official_stream",
                             power_revisit_policy="restart").resolved()
        self.evaluator_ = FairEvaluator(config=cfg, label_O=[self.protected_name], label_Y="__y__", cate_attrs=self.cate_attrs_, num_attrs=self.num_attrs_)
        O_f = AF.copy(deep=True) if isinstance(AF, pd.DataFrame) else pd.DataFrame({self.protected_name: np.asarray(AF).ravel()}, index=Xf.index)
        O_c = AC.copy(deep=True) if isinstance(AC, pd.DataFrame) else pd.DataFrame({self.protected_name: np.asarray(AC).ravel()}, index=Xc.index)
        if self.protected_name not in O_f.columns: O_f = pd.DataFrame({self.protected_name: np.asarray(AF).ravel()}, index=Xf.index)
        if self.protected_name not in O_c.columns: O_c = pd.DataFrame({self.protected_name: np.asarray(AC).ravel()}, index=Xc.index)
        eps0 = self.evaluator_.calculate_epsilon(Xf, O_f, cate_attrs=self.cate_attrs_, num_attrs=self.num_attrs_)
        self._max_eps(eps0)
        vals = [float(v) for d in eps0.values() for v in d.values()]
        self.reference_epsilon_ = float(np.mean(vals))
        self.epsilon_threshold_ = float(self.reference_epsilon_ * self.epsilon_ratio)
        self._utility_evaluations_ = 0
        self._geometry_evaluations_ = 1
        partition = EvaluationPartition(fit_X=Xf, fit_y=yf, selection_X=Xc, selection_y=yc, protected_fit=O_f, protected_selection=O_c, fit_source="F", selection_source="enhancement_eval")
        self.partition_ = partition
        self.changed_dict_ = {}
        self.provenance_ = {"mode": self.mode, "algorithm_mode": cfg.algorithm_mode, "backbone": self.backbone, "candidate_order": list(self.poly_exponents), "fit_source": "F", "selection_source": "enhancement_eval", "label_origin": {"F": "F", "C": "C"}, "reference_epsilon_mean_F": self.reference_epsilon_, "epsilon_threshold": self.epsilon_threshold_}
        self.provenance_.update(resolved_config=dataclasses.asdict(cfg), estimator_params=self._estimator().get_params(),
                                utility="unweighted C balanced accuracy after frozen F encoder and common C threshold", execution="one isolated process per fit")
        import fairbias.enhancement as enhancement_module
        original_utility = enhancement_module.evaluate_candidate_utility
        original_geometry = self.evaluator_.calculate_epsilon
        def bounded_geometry(*args, **kwargs):
            self._geometry_evaluations_ += 1
            if self._geometry_evaluations_ > self.max_geometry_evaluations:
                self.termination_reason_ = "BUDGET_EXHAUSTED"
                raise RuntimeError("BUDGET_EXHAUSTED: FairBias AE geometry evaluations")
            return original_geometry(*args, **kwargs)
        enhancement_module.evaluate_candidate_utility = lambda partition, changed_dict, num_attrs, cate_attrs, transformer, evaluator: self._utility(partition, changed_dict)
        self.evaluator_.calculate_epsilon = bounded_geometry
        try:
            bm = FairBiasMitigation(self.evaluator_, self.transformer_, [self.protected_name], self.cate_attrs_, self.num_attrs_, max_search_candidates=5, phi_threshold=100.0, poly_exponents=cfg.transform_poly_exponents, failed_attribute_mode="stop", power_sequence_policy="official_stream", power_revisit_policy="restart")
            ae = FairAccuracyEnhancement(self.evaluator_, self.transformer_, "__y__", self.cate_attrs_, self.num_attrs_, max_fairness_degradation=0.0, min_utility_gain=0.0, poly_exponents=self.poly_exponents, run_id="benchmark", arm_id=self.name, condition=self.mode)
            nmi = calculate_nmi_dict(Xf, yf)
            current_eps = eps0
            changed = {}
            traces = []
            bm_commits = 0
            ae_commits = 0
            def refresh():
                table = bounded_geometry(self.transformer_.transform_data(Xf, changed, self.num_attrs_, self.cate_attrs_),
                                         O_f, cate_attrs=self.cate_attrs_, num_attrs=self.num_attrs_)
                self._max_eps(table)
                return table
            def budget(label):
                self.termination_reason_ = "BUDGET_EXHAUSTED"
                raise RuntimeError("BUDGET_EXHAUSTED: " + label)
            def bm_step():
                nonlocal changed, current_eps, bm_commits
                if self._max_eps(current_eps) <= self.epsilon_threshold_:
                    return False
                if bm_commits >= self.max_bm_steps:
                    budget("BM commit limit")
                previous = copy.deepcopy(changed)
                _, candidate, _, attr = bm.mitigate_step(Xf, yf, O_f, nmi, changed, current_eps,
                    self.epsilon_threshold_, iteration=len(traces)+1)
                committed = attr is not None and candidate != previous
                traces.append({"engine": "BM", "iteration": len(traces)+1, "feature": attr, "committed": committed})
                if committed:
                    changed = copy.deepcopy(candidate)
                    bm_commits += 1
                    current_eps = refresh()
                return committed
            def ae_step():
                nonlocal changed, current_eps, ae_commits
                if ae_commits >= self.max_outer_iterations:
                    budget("AE commit limit; stopping criterion not yet established")
                previous = copy.deepcopy(changed)
                _, candidate, attr = ae.enhance_step(Xf, yf, changed, O_f, self.epsilon_threshold_, current_eps,
                                                   iteration=len(traces)+1, partition=partition)
                committed = attr is not None and candidate != previous
                traces.append({"engine": "AE", "iteration": len(traces)+1, "feature": attr, "committed": committed})
                if committed:
                    changed = copy.deepcopy(candidate)
                    ae_commits += 1
                    current_eps = refresh()
                    if self._max_eps(current_eps) > self.epsilon_threshold_:
                        raise RuntimeError("AE committed a geometrically infeasible state")
                return committed
            if self.mode == "BM_AE":
                while self._max_eps(current_eps) > self.epsilon_threshold_:
                    if not bm_step():
                        self.termination_reason_ = "CANDIDATE_EXHAUSTED"
                        raise RuntimeError("CANDIDATE_EXHAUSTED: BM_AE has no epsilon-feasible BM state")
                while ae_step():
                    pass
            else:
                while True:
                    bm_changed = bm_step()
                    ae_changed = ae_step()
                    if not bm_changed and not ae_changed:
                        break
            final_eps = refresh()
            if self._max_eps(final_eps) > self.epsilon_threshold_:
                self.termination_reason_ = "CANDIDATE_EXHAUSTED"
                raise RuntimeError("CANDIDATE_EXHAUSTED: final state remains infeasible")
            self.termination_reason_ = "STRICT_FEASIBLE_SEARCH_EXHAUSTED"
            self.changed_dict_ = copy.deepcopy(changed)
            self.bm_engine_ = bm; self.ae_engine_ = ae; self.candidate_traces_ = traces
            self.provenance_.update(bm_commits=bm_commits, ae_commits=ae_commits, final_max_dphi=self._max_eps(final_eps),
                termination_reason=self.termination_reason_, bm_candidate_merges_limit=bm.max_search_candidates,
                bm_trace=[s.to_dict() for s in bm.step_traces], ae_audit=[dataclasses.asdict(e) for e in ae.audit_trail])
        finally:
            enhancement_module.evaluate_candidate_utility = original_utility
            self.evaluator_.calculate_epsilon = original_geometry
        final_f = self.transformer_.transform_data(Xf, self.changed_dict_, self.num_attrs_, self.cate_attrs_)
        self.model_ = self._estimator(); self.model_.fit(self._encode_fit(final_f), yf.to_numpy())
        self.is_fitted_ = True
        self.provenance_.update({"changed_dict": copy.deepcopy(self.changed_dict_), "trace_count": len(self.candidate_traces_), "bm_model_fits": getattr(self.bm_engine_, "total_model_fits", 0), "ae_model_fits": getattr(self.ae_engine_, "total_model_fits", 0), "utility_evaluations": self._utility_evaluations_, "geometry_evaluations": self._geometry_evaluations_, "limits": {"outer": self.max_outer_iterations, "utility": self.max_utility_evaluations, "geometry": self.max_geometry_evaluations, "bm": self.max_bm_steps}})
        return self

    def fit(self, X, y, A, sample_weight=None):
        raise NotSupportedError("FairBiasAEAdapter requires disjoint F/C fit_development partitions")

    def predict_event_probability(self, X, A=None):
        if not self.is_fitted_: raise RuntimeError("adapter is not fitted")
        Xs = self._semantic_adapter._prepare_predict_semantic(pd.DataFrame(X))
        Xt = self.transformer_.transform_data(Xs, self.changed_dict_, self.num_attrs_, self.cate_attrs_)
        return validate_and_extract_positive_probabilities(self.model_, self._encode(Xt), expected_classes=(0, 1), pos_label=1)

    def predict_decision_proba(self, X, A=None): return self.predict_event_probability(X, A)
    def predict(self, X, A=None): return (self.predict_event_probability(X, A) >= 0.5).astype(int)
