"""Separately versioned scheduled-Joint extension with feasible incumbents.

This is a new search schedule, not an equivalent implementation of Joint or a
paper reproduction. Preparation follows FairBiasAEAdapter; its estimator,
utility, encoding and prediction helpers are reused without modifying it.
The cooperative search deadline excludes the final incumbent model refit.
One isolated process per fit is required by the inherited utility hook.
"""
from __future__ import annotations

import copy
import dataclasses
import time
from typing import Callable, Optional

import numpy as np
import pandas as pd

from fairbias.config import ALGORITHM_MODE_PAPER_FAITHFUL, FairBiasConfig
from fairbias.enhancement import FairAccuracyEnhancement
from fairbias.enhancement_contracts import EvaluationPartition
from fairbias.enhancement_state import hash_transform_state
from fairbias.evaluator import FairEvaluator
from fairbias.mitigation import FairBiasMitigation
from fairbias.transform import FairTransform, calculate_nmi_dict
from .adapter_fairbias import FairBiasAdapter
from .adapter_fairbias_ae import FairBiasAEAdapter
from .geometry_audit import audited_mds


class StopBudget(BaseException):
    """Typed resource stop that legacy candidate ``except Exception`` cannot hide."""

    def __init__(self, reason: str, *, geometry_incomplete: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.geometry_incomplete = geometry_incomplete


class _EvaluationFailure(BaseException):
    """Transport ordinary evaluation errors past legacy candidate rejection."""

    def __init__(self, original: Exception):
        super().__init__(str(original))
        self.original = original


def _is_mds_cap(exc: Exception) -> bool:
    # compute_bias_concentration wraps the audit error; inspect its cause chain.
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, RuntimeError) and "MDS_ITERATION_CAP" in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class ScheduledJointAdapter(FairBiasAEAdapter):
    """Run AE every k committed BM steps, upon feasibility, or as BM rescue."""

    name = "FAIRBIAS_SCHEDULED_JOINT"

    def __init__(self, mode="JOINT", *, ae_every_bm_commits=3,
                 search_seconds=1500.0, monotonic: Optional[Callable[[], float]] = None,
                 **kwargs):
        if str(mode).upper() != "JOINT":
            raise ValueError("ScheduledJointAdapter requires mode JOINT")
        if (isinstance(ae_every_bm_commits, (bool, np.bool_))
                or not isinstance(ae_every_bm_commits, (int, np.integer))
                or ae_every_bm_commits < 1):
            raise ValueError("ae_every_bm_commits must be a positive integer")
        if (isinstance(search_seconds, (bool, np.bool_))
                or not np.isfinite(search_seconds) or search_seconds <= 0):
            raise ValueError("search_seconds must be finite and positive")
        if monotonic is not None and not callable(monotonic):
            raise ValueError("monotonic must be callable")
        super().__init__(mode="JOINT", **kwargs)
        self.name = "FAIRBIAS_SCHEDULED_JOINT"
        self.ae_every_bm_commits = int(ae_every_bm_commits)
        self.search_seconds = float(search_seconds)
        self._monotonic = monotonic or time.monotonic

    def _check_deadline(self, stage):
        if self._monotonic() >= self._search_deadline:
            raise StopBudget("SEARCH_TIME_LIMIT: " + stage)

    def _utility(self, partition, changed):
        self._check_deadline("before utility")
        if self._utility_evaluations_ >= self.max_utility_evaluations:
            raise StopBudget("UTILITY_EVALUATION_LIMIT")
        result = super()._utility(partition, changed)
        if not result.is_valid or not np.isfinite(result.utility_score):
            raise _EvaluationFailure(RuntimeError(
                "UTILITY_EVALUATION_FAILED: " + str(result.error_message)))
        self._check_deadline("after utility")
        return result

    def _prepare_development(self, XF, yF, AF, XC, yC, AC, metadata):
        """Copy the registered adapter's F/C-only semantic preparation contract."""
        if metadata is not None and ("F_ids" in metadata or "C_ids" in metadata):
            if not {"F_ids", "C_ids"}.issubset(metadata):
                raise ValueError("Both F and C record identities are required")
            f_ids, c_ids = metadata["F_ids"], metadata["C_ids"]
            if len(f_ids) != len(XF) or len(c_ids) != len(XC) or set(f_ids) & set(c_ids):
                raise ValueError("F/C record identities overlap or have mismatched lengths")
        self._semantic_adapter = FairBiasAdapter(backbone=self.backbone, random_state=self.random_state)
        Xf = self._semantic_adapter._prepare_fit_semantic(XF.copy(deep=True))
        Xc = (self._semantic_adapter._prepare_predict_semantic(XC.copy(deep=True))
              if self._semantic_adapter.semantic_columns_ else XC.copy(deep=True))
        for values in (yF, yC):
            arr = np.asarray(values)
            if arr.ndim != 1 or np.iscomplexobj(arr) or not np.isin(arr, [0, 1]).all():
                raise ValueError("F and C require one-dimensional binary labels before integer conversion")
        yf = pd.Series(np.asarray(yF, dtype=int), index=Xf.index)
        yc = pd.Series(np.asarray(yC, dtype=int), index=Xc.index)
        if len(AF) != len(Xf) or len(AC) != len(Xc):
            raise ValueError("X, y, and A lengths must agree within F and C")
        if set(yf.unique()) != {0, 1} or set(yc.unique()) != {0, 1}:
            raise ValueError("F and C must contain both binary classes")
        if list(Xf.columns) != list(Xc.columns):
            raise ValueError("F and C feature columns must match")
        self.cate_attrs_ = list(self.categorical_features or [
            c for c in Xf.columns if not pd.api.types.is_numeric_dtype(Xf[c])])
        self.num_attrs_ = list(self.numerical_features or [c for c in Xf.columns if c not in self.cate_attrs_])
        self.transformer_ = FairTransform()
        cfg = FairBiasConfig(
            algorithm_mode=ALGORITHM_MODE_PAPER_FAITHFUL, classifier="LR", random_seed=self.random_state,
            multigroup_aggregation="author_max_pair", eval_norm="min-max", label_O=(self.protected_name,),
            label_Y="__y__", failed_attribute_mode="stop", power_sequence_policy="official_stream",
            power_revisit_policy="restart").resolved()
        self.evaluator_ = FairEvaluator(config=cfg, label_O=[self.protected_name], label_Y="__y__",
                                       cate_attrs=self.cate_attrs_, num_attrs=self.num_attrs_)
        def protected(values, index):
            if isinstance(values, pd.DataFrame) and self.protected_name in values.columns:
                return values.copy(deep=True)
            return pd.DataFrame({self.protected_name: np.asarray(values).ravel()}, index=index)
        Of, Oc = protected(AF, Xf.index), protected(AC, Xc.index)
        self.partition_ = EvaluationPartition(
            fit_X=Xf, fit_y=yf, selection_X=Xc, selection_y=yc,
            protected_fit=Of, protected_selection=Oc, fit_source="F", selection_source="enhancement_eval")
        self.provenance_.update(
            geometry_algorithm_mode=cfg.algorithm_mode, resolved_config=dataclasses.asdict(cfg),
            estimator_params=self._estimator().get_params(), candidate_order=list(self.poly_exponents),
            backbone=self.backbone, fit_source="F", selection_source="enhancement_eval",
            label_origin={"F": "F", "C": "C"}, utility_metric="balanced_accuracy",
            utility="unweighted C balanced accuracy after frozen F encoder and common C threshold")
        return Xf, yf, Of, cfg

    def fit_development(self, X_semantic_F, yF, AF, X_semantic_C, yC, AC, metadata=None):
        import fairbias.enhancement as enhancement_module
        self.is_fitted_, self.model_, self.preprocessor_ = False, None, None
        self.termination_reason_, self.budget_reason_ = None, None
        self.convergence_verified_, self.incumbent_available_ = False, False
        self.changed_dict_, self.candidate_traces_, self.mds_diagnostics_ = {}, [], []
        self._geometry_evaluations_, self._utility_evaluations_ = 0, 0
        self._bm_commits, self._ae_commits = 0, 0
        self.bm_engine_, self.ae_engine_ = None, None
        self._incumbent, self._current_changed, self._current_eps = None, {}, None
        self._search_started = self._monotonic()
        self._search_deadline = self._search_started + self.search_seconds
        self.provenance_ = {
            "mode": "SCHEDULED_JOINT", "algorithm_variant": "scheduled_joint_v1",
            "paper_equivalent": False, "ae_every_bm_commits": self.ae_every_bm_commits,
            "execution": "one isolated process per fit", "final_refit_outside_search_budget": True,
            "limits": {"outer": self.max_outer_iterations, "utility": self.max_utility_evaluations,
                       "geometry": self.max_geometry_evaluations, "bm": self.max_bm_steps,
                       "search_seconds": self.search_seconds}}
        try:
            with audited_mds(self.mds_diagnostics_, lambda: self._geometry_evaluations_):
                Xf, yf, Of, cfg = self._prepare_development(
                    X_semantic_F, yF, AF, X_semantic_C, yC, AC, metadata)
                original_geometry = self.evaluator_.calculate_epsilon
                original_utility = enhancement_module.evaluate_candidate_utility

                def bounded_geometry(*args, **kwargs):
                    self._check_deadline("before geometry")
                    if self._geometry_evaluations_ >= self.max_geometry_evaluations:
                        raise StopBudget("GEOMETRY_EVALUATION_LIMIT")
                    self._geometry_evaluations_ += 1
                    try:
                        result = original_geometry(*args, **kwargs)
                        frame = args[0] if args else kwargs["X"]
                        protected_frame = args[1] if len(args) > 1 else kwargs["O"]
                        if (set(result) != set(protected_frame.columns)
                                or any(set(group) != set(frame.columns) for group in result.values())):
                            raise ValueError("Incomplete protected-feature geometry")
                        self._max_eps(result)
                    except Exception as exc:
                        if _is_mds_cap(exc):
                            raise StopBudget("MDS_ITERATION_CAP", geometry_incomplete=True) from exc
                        raise _EvaluationFailure(exc) from exc
                    self._check_deadline("after geometry")
                    return result

                self.evaluator_.calculate_epsilon = bounded_geometry
                enhancement_module.evaluate_candidate_utility = (
                    lambda partition, changed_dict, **kwargs: self._utility(partition, changed_dict))
                try:
                    eps0 = bounded_geometry(Xf, Of, cate_attrs=self.cate_attrs_, num_attrs=self.num_attrs_)
                    self.reference_epsilon_ = float(np.mean([v for g in eps0.values() for v in g.values()]))
                    self.epsilon_threshold_ = self.reference_epsilon_ * self.epsilon_ratio
                    self.provenance_.update(reference_epsilon_mean_F=self.reference_epsilon_,
                                            epsilon_threshold=self.epsilon_threshold_)
                    self._current_eps = copy.deepcopy(eps0)
                    self._remember_feasible()
                    self.bm_engine_ = FairBiasMitigation(
                        self.evaluator_, self.transformer_, [self.protected_name], self.cate_attrs_, self.num_attrs_,
                        max_search_candidates=5, phi_threshold=100.0, poly_exponents=cfg.transform_poly_exponents,
                        failed_attribute_mode="stop", power_sequence_policy="official_stream", power_revisit_policy="restart")
                    self.ae_engine_ = FairAccuracyEnhancement(
                        self.evaluator_, self.transformer_, "__y__", self.cate_attrs_, self.num_attrs_,
                        max_fairness_degradation=0.0, min_utility_gain=0.0, poly_exponents=self.poly_exponents,
                        run_id="benchmark", arm_id=self.name, condition="SCHEDULED_JOINT")
                    self._search(Xf, yf, Of, calculate_nmi_dict(Xf, yf))
                except StopBudget as stop:
                    self.budget_reason_ = stop.reason
                    if self._incumbent is None:
                        self.termination_reason_ = ("NO_FEASIBLE_GEOMETRY_INCOMPLETE" if stop.geometry_incomplete
                                                    else "NO_FEASIBLE_BUDGET_LIMITED")
                        raise RuntimeError(self.termination_reason_ + ": " + stop.reason) from stop
                    self.termination_reason_ = ("FEASIBLE_GEOMETRY_INCOMPLETE" if stop.geometry_incomplete
                                                else "FEASIBLE_BUDGET_LIMITED")
                    self.convergence_verified_ = False
                except _EvaluationFailure as exc:
                    self.termination_reason_ = "EVALUATION_FAILED"
                    raise exc.original from exc
                finally:
                    enhancement_module.evaluate_candidate_utility = original_utility
                    self.evaluator_.calculate_epsilon = original_geometry
            # Only a fully verified incumbent reaches final fitting. In particular,
            # a timed-out candidate never replaces the previously committed state.
            self.partition_.verify_not_mutated()
            self.changed_dict_ = copy.deepcopy(self._incumbent["changed"])
            final_f = self.transformer_.transform_data(Xf, self.changed_dict_, self.num_attrs_, self.cate_attrs_)
            model = self._estimator()
            model.fit(self._encode_fit(final_f), yf.to_numpy())
            self.model_, self.is_fitted_ = model, True
            return self
        except (KeyboardInterrupt, SystemExit):
            self.termination_reason_ = "INTERRUPTED"
            self.convergence_verified_ = False
            self.is_fitted_, self.model_ = False, None
            raise
        except Exception:
            if self.termination_reason_ not in {"NO_FEASIBLE_BUDGET_LIMITED", "NO_FEASIBLE_GEOMETRY_INCOMPLETE",
                                               "CANDIDATE_EXHAUSTED"}:
                self.termination_reason_ = "EVALUATION_FAILED"
            self.is_fitted_, self.model_ = False, None
            self.convergence_verified_ = False
            raise
        finally:
            self._save_provenance()

    def _remember_feasible(self):
        if self._max_eps(self._current_eps) <= self.epsilon_threshold_:
            self._incumbent = {"changed": copy.deepcopy(self._current_changed),
                               "epsilon": copy.deepcopy(self._current_eps),
                               "bm_commits": self._bm_commits, "ae_commits": self._ae_commits}
            self.incumbent_available_ = True

    def _refresh(self, Xf, Of, changed):
        return self.evaluator_.calculate_epsilon(
            self.transformer_.transform_data(Xf, changed, self.num_attrs_, self.cate_attrs_),
            Of, cate_attrs=self.cate_attrs_, num_attrs=self.num_attrs_)

    def _step(self, engine, Xf, yf, Of, nmi):
        trace = {"engine": engine, "iteration": len(self.candidate_traces_) + 1,
                 "selected": False, "committed": False, "status": "STARTED", "feature": None}
        self.candidate_traces_.append(trace)
        self._check_deadline("before " + engine)
        if engine == "BM" and self._max_eps(self._current_eps) <= self.epsilon_threshold_:
            trace["status"] = "NOOP_FEASIBLE"
            return False
        limit, count = ((self.max_bm_steps, self._bm_commits) if engine == "BM"
                        else (self.max_outer_iterations, self._ae_commits))
        if count >= limit:
            trace["status"] = "COMMIT_LIMIT"
            raise StopBudget(engine + "_COMMIT_LIMIT")
        previous = copy.deepcopy(self._current_changed)
        if engine == "BM":
            _, candidate, _, attr = self.bm_engine_.mitigate_step(
                Xf, yf, Of, nmi, copy.deepcopy(previous), copy.deepcopy(self._current_eps),
                self.epsilon_threshold_, iteration=trace["iteration"])
        else:
            _, candidate, attr = self.ae_engine_.enhance_step(
                Xf, yf, copy.deepcopy(previous), Of, self.epsilon_threshold_, copy.deepcopy(self._current_eps),
                iteration=trace["iteration"], partition=self.partition_)
        self._check_deadline("after " + engine)
        trace.update(feature=attr, selected=attr is not None and candidate != previous)
        if not trace["selected"]:
            trace["status"] = "NO_CANDIDATE"
            return False
        trace["candidate_state_hash"] = hash_transform_state(candidate)
        trace["status"] = "PENDING_GEOMETRY_REFRESH"
        full_eps = self._refresh(Xf, Of, candidate)
        if engine == "AE" and self._max_eps(full_eps) > self.epsilon_threshold_:
            raise RuntimeError("AE selected a geometrically infeasible state")
        # Transaction boundary: all computation and deadline checks precede mutation.
        self._check_deadline("before " + engine + " commit")
        self._current_changed, self._current_eps = copy.deepcopy(candidate), copy.deepcopy(full_eps)
        if engine == "BM":
            self._bm_commits += 1
        else:
            self._ae_commits += 1
        trace.update(committed=True, status="COMMITTED")
        self._remember_feasible()
        return True

    def _search(self, Xf, yf, Of, nmi):
        since_ae = 0
        while True:
            bm_changed = self._step("BM", Xf, yf, Of, nmi)
            since_ae += int(bm_changed)
            feasible = self._max_eps(self._current_eps) <= self.epsilon_threshold_
            if not feasible and bm_changed and since_ae < self.ae_every_bm_commits:
                self.candidate_traces_.append({"engine": "AE", "iteration": len(self.candidate_traces_) + 1,
                    "selected": False, "committed": False, "status": "DEFERRED", "feature": None})
                continue
            ae_changed = self._step("AE", Xf, yf, Of, nmi)
            since_ae = 0
            if not bm_changed and not ae_changed:
                self.partition_.verify_not_mutated()
                final_eps = self._refresh(Xf, Of, self._current_changed)
                if self._max_eps(final_eps) > self.epsilon_threshold_:
                    self.termination_reason_ = "CANDIDATE_EXHAUSTED"
                    raise RuntimeError("CANDIDATE_EXHAUSTED: scheduled Joint has no feasible state")
                self._current_eps = copy.deepcopy(final_eps)
                self._remember_feasible()
                self.termination_reason_, self.convergence_verified_ = "SEARCH_EXHAUSTED", True
                return

    def _save_provenance(self):
        by_iteration = {t["iteration"]: t for t in self.candidate_traces_}
        audits = []
        for event in getattr(self.ae_engine_, "audit_trail", []):
            row = dataclasses.asdict(event)
            trace = by_iteration.get(row["iteration"], {})
            selected = bool(row["accepted"])
            committed = bool(selected and trace.get("committed")
                             and row["candidate_state_hash"] == trace.get("candidate_state_hash"))
            row.update(utility_metric="balanced_accuracy", selected=selected, committed=committed, accepted=committed)
            audits.append(row)
        bm_trace = []
        for event in getattr(self.bm_engine_, "step_traces", []):
            row = event.to_dict()
            row["controller_committed"] = bool(by_iteration.get(row["iteration"], {}).get("committed"))
            bm_trace.append(row)
        self.provenance_.update(
            termination_reason=self.termination_reason_, budget_reason=self.budget_reason_,
            convergence_verified=self.convergence_verified_, incumbent_available=self.incumbent_available_,
            is_fitted=self.is_fitted_, bm_commits=self._bm_commits, ae_commits=self._ae_commits,
            current_changed=copy.deepcopy(self._current_changed), changed_dict=copy.deepcopy(self.changed_dict_),
            feasible_incumbent=copy.deepcopy(self._incumbent), candidate_traces=copy.deepcopy(self.candidate_traces_),
            ae_audit=audits, bm_trace=bm_trace, mds_fits=copy.deepcopy(self.mds_diagnostics_),
            utility_evaluations=self._utility_evaluations_, geometry_evaluations=self._geometry_evaluations_,
            elapsed_seconds=float(self._monotonic() - self._search_started),
            final_max_dphi=(self._max_eps(self._incumbent["epsilon"]) if self._incumbent is not None else None))
