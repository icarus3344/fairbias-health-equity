"""Explicit F/C-only recovery composition; the registered worker is unchanged.

Reuses the admitted exact NumPy kernel, original strict adapter, scheduled
controller and optional MDS retry. This is a candidate runtime for a separate
registration, never permission to replace old receipts or evaluate S/T.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager, ExitStack
import functools
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

from fairbias.evaluator import FairEvaluator
from .base import BaseMethodAdapter, NotSupportedError
from .adapter_fairbias_ae import FairBiasAEAdapter
from .adapter_fairbias_scheduled import ScheduledJointAdapter
from ..joint_mds_numpy import numpy_mds_acceleration
from ..mds_budget_retry import mds_budget_retry


VERSION = "joint_ae_recovery_v1"
_ROOT = Path(__file__).resolve().parents[4]
_COMPONENT_HASHES = {
    "src/nhis_fairbias/benchmark/joint_budget_optimization.py": "349f59f9bc39c2a41718ec0ab81aaa66230ee3912b4575029689a5de7af373d8",
    "src/nhis_fairbias/benchmark/joint_mds_numpy.py": "2b7a773fa90fa507548f557a229a2f2db776c7845afa928e4e12087774ad11c1",
    "src/nhis_fairbias/benchmark/adapters/adapter_fairbias_scheduled.py": "7f13232bd8201e6380a1aa3c489a1f995106a9103f2ab5d9c2f7b0a06c173a14",
    "src/nhis_fairbias/benchmark/mds_budget_retry.py": "4d6a230630330f23f9884b9bfef38eebbfae1e067b2706d81ecd6a1b498c5a0e",
    "src/nhis_fairbias/benchmark/adapters/adapter_fairbias_ae.py": "004613b9b20134c6ec95a3a707fbbcd162cf1269e6af28621f8cf4998c998f84",
    "src/fairbias/evaluator.py": "27530345f30505c46ebe17c4fd50bfe7f58dc041c927c96d03465d05ea0811ab",
    "src/fairbias/enhancement.py": "2abd5688b8614e5af55b4c18409d29301418c03338fae29ba26d8d6ea35c0b9f",
}


def verify_recovery_component_sources():
    """Pin admitted components; the new worker must also check its registration."""
    observed = {name: hashlib.sha256((_ROOT / name).read_bytes()).hexdigest()
                for name in _COMPONENT_HASHES}
    if observed != _COMPONENT_HASHES:
        raise RecoveryAdmissionError("Recovery component source fingerprint changed")
    for name in _COMPONENT_HASHES:
        module = sys.modules.get(name[4:-3].replace("/", "."))
        origin = getattr(module, "__file__", None)
        if origin is None or Path(origin).resolve() != (_ROOT / name).resolve():
            raise RecoveryAdmissionError("Recovery component loaded outside the pinned source tree")
    return observed


class RecoveryAdmissionError(RuntimeError):
    """Returned model has insufficient evidence for the declared fit status."""


class _GeometryFailure(BaseException):
    def __init__(self, original):
        self.original = original


@contextmanager
def _strict_geometry_failure_transport():
    """Prevent legacy AE's except Exception from hiding incomplete geometry."""
    original = FairEvaluator.calculate_epsilon

    @functools.wraps(original)
    def checked(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            raise _GeometryFailure(exc) from exc

    FairEvaluator.calculate_epsilon = checked
    try:
        yield
    finally:
        FairEvaluator.calculate_epsilon = original


def classify_recovery_fit(adapter, *, controller, retry_stats=None):
    """Return a model-status contract; no result here is a formal-study VALID."""
    if not getattr(adapter, "is_fitted_", False):
        raise RecoveryAdmissionError("Adapter returned without a fitted model")
    provenance = getattr(adapter, "provenance_", {})
    maximum = provenance.get("final_max_dphi")
    threshold = getattr(adapter, "epsilon_threshold_", None)
    if (maximum is None or threshold is None or not np.isfinite(maximum)
            or not np.isfinite(threshold) or maximum < 0 or threshold < 0 or maximum > threshold):
        raise RecoveryAdmissionError("Returned model lacks verified final global feasibility")
    diagnostics = getattr(adapter, "mds_diagnostics_", [])
    incomplete = any(row.get("status") != "CONVERGED_BEFORE_CAP" for row in diagnostics)
    incomplete = incomplete or any(not row.get("complete", False)
                                  for row in (retry_stats or {}).get("fits", []))
    termination = getattr(adapter, "termination_reason_", None)
    if controller == "strict":
        if termination != "STRICT_FEASIBLE_SEARCH_EXHAUSTED" or incomplete:
            raise RecoveryAdmissionError("Strict recovery did not verify complete feasible search")
        status, converged = "COMPLETE_FEASIBLE", True
    elif controller == "scheduled":
        declared_convergence = bool(getattr(adapter, "convergence_verified_", False))
        if termination == "SEARCH_EXHAUSTED" and declared_convergence and not incomplete:
            status, converged = "COMPLETE_FEASIBLE", True
        elif termination == "FEASIBLE_BUDGET_LIMITED" and not declared_convergence and not incomplete:
            status, converged = "FEASIBLE_BUDGET_LIMITED", False
        elif termination == "FEASIBLE_GEOMETRY_INCOMPLETE" and not declared_convergence:
            status, converged = "FEASIBLE_GEOMETRY_INCOMPLETE", False
            incomplete = True
        else:
            raise RecoveryAdmissionError("Scheduled recovery termination evidence is inconsistent")
    else:
        raise ValueError("Unknown recovery controller")
    return {
        "status": status, "model_available": True, "final_geometry_feasible": True,
        "search_complete": converged, "geometry_search_incomplete": incomplete,
        "convergence_verified": converged, "termination_reason": termination,
        "final_max_dphi": float(maximum), "epsilon_threshold": float(threshold),
        "formal_study_admitted": False, "S_T_evaluated_by_recovery": False,
    }


class JointAERecoveryAdapter(BaseMethodAdapter):
    """Serializable adapter with explicit strict/scheduled and retry identities."""

    literature_reference = FairBiasAEAdapter.literature_reference
    upstream_implementation = "Existing FairBias engines plus explicit versioned recovery composition"

    def __init__(self, *, mode, controller="strict", use_mds_retry=False, ae_cap_retry=None,
                 ae_every_bm_commits=3, search_seconds=900.0, **adapter_params):
        if mode not in {"JOINT", "BM_AE"} or controller not in {"strict", "scheduled"}:
            raise ValueError("Explicit BM_AE/JOINT mode and strict/scheduled controller are required")
        if controller == "scheduled" and mode != "JOINT":
            raise ValueError("Scheduled controller is a Joint extension, not BM_AE recovery")
        if type(use_mds_retry) is not bool:
            raise ValueError("use_mds_retry must be an explicit boolean")
        if ae_cap_retry is not None and (
                type(ae_cap_retry) is not int or ae_cap_retry != 40
                or mode != "BM_AE" or controller != "strict"
                or adapter_params.get("max_outer_iterations", 10) != 10):
            raise ValueError("AE cap retry v1 requires strict BM_AE with registered cap10 and retry40")
        self.mode, self.controller, self.use_mds_retry = mode, controller, use_mds_retry
        self.ae_cap_retry = ae_cap_retry
        self._original_adapter_params = copy.deepcopy(adapter_params)
        self.name = f"FAIRBIAS_{mode}_{controller.upper()}_RECOVERY_V1" + ("_MDS_RETRY_V1" if use_mds_retry else "")
        if ae_cap_retry is not None:
            self.name += "_AE_CAP_RETRY_10_TO_40_V1"
        self.runtime_spec = {
            "version": VERSION, "mode": mode, "controller": controller,
            "exact_kernel": "joint_numpy_mds_v1", "mds_retry": "mds_budget_retry_v1" if use_mds_retry else None,
            "ae_every_bm_commits": ae_every_bm_commits if controller == "scheduled" else None,
            "search_seconds": search_seconds if controller == "scheduled" else None,
            "search_policy_changed": controller == "scheduled" or use_mds_retry or ae_cap_retry is not None,
            "search_order_changed": controller == "scheduled",
            "mds_iteration_budget_changed": use_mds_retry,
            "ae_commit_budget_changed": ae_cap_retry is not None,
            "ae_cap_retry": ae_cap_retry,
            "ae_cap_retry_policy": "fresh_same_seed_refit_after_exact_ae_commit_cap" if ae_cap_retry else None,
            "strict_geometry_errors_fail_closed": True,
        }
        if controller == "scheduled":
            self._delegate = ScheduledJointAdapter(
                mode=mode, ae_every_bm_commits=ae_every_bm_commits,
                search_seconds=search_seconds, **adapter_params)
        else:
            self._delegate = FairBiasAEAdapter(mode=mode, **adapter_params)
        self.runtime_spec["registered_search_limits"] = {
            "outer": self._delegate.max_outer_iterations, "utility": self._delegate.max_utility_evaluations,
            "geometry": self._delegate.max_geometry_evaluations, "bm": self._delegate.max_bm_steps,
        }
        self.is_fitted_ = False
        self.provenance_ = {"recovery": copy.deepcopy(self.runtime_spec)}

    def fit(self, X, y, A, sample_weight=None):
        raise NotSupportedError("Recovery requires disjoint F/C fit_development, never generic fit or weighted training")

    def fit_development(self, XF, yF, AF, XC, yC, AC, metadata=None):
        self.is_fitted_ = False
        if self.ae_cap_retry is not None:
            self._delegate = FairBiasAEAdapter(mode=self.mode, **copy.deepcopy(self._original_adapter_params))
        self.recovery_result_ = {"status": "FAILED_OR_INTERRUPTED", "formal_study_admitted": False}
        kernel_stats, retry_stats = None, None
        source_hashes = None
        self.fit_attempts_ = []
        try:
            source_hashes = verify_recovery_component_sources()
            with ExitStack() as scope:
                if self.use_mds_retry:
                    retry_stats = scope.enter_context(mds_budget_retry())
                kernel_stats = scope.enter_context(numpy_mds_acceleration())
                if self.controller == "strict":
                    scope.enter_context(_strict_geometry_failure_transport())
                for attempt in range(1, 3 if self.ae_cap_retry else 2):
                    record = {"attempt": attempt, "ae_commit_cap": self._delegate.max_outer_iterations,
                              "seed": self._delegate.random_state, "status": "STARTED"}
                    self.fit_attempts_.append(record)
                    started = time.monotonic()
                    try:
                        self._delegate.fit_development(XF, yF, AF, XC, yC, AC, metadata)
                        record["status"] = "FIT_RETURNED"
                        break
                    except BaseException as exc:
                        # Match the sole original AE-cap exit. No utility, geometry,
                        # MDS, model failure or interrupt enables this fresh refit.
                        at_ae_cap = (type(exc) is RuntimeError
                            and str(exc) == "BUDGET_EXHAUSTED: AE commit limit; stopping criterion not yet established"
                            and self._delegate.termination_reason_ == "BUDGET_EXHAUSTED")
                        eligible = self.ae_cap_retry is not None and attempt == 1 and at_ae_cap
                        record.update(status="AE_COMMIT_CAP" if at_ae_cap else "FAILED_OR_INTERRUPTED",
                                      error_type=type(exc).__name__, retry_eligible=eligible)
                        if not eligible:
                            if isinstance(exc, _GeometryFailure):
                                raise exc.original from exc
                            raise
                    finally:
                        record.update(elapsed_seconds=time.monotonic() - started,
                            termination_reason=getattr(self._delegate, "termination_reason_", None),
                            geometry_evaluations=getattr(self._delegate, "_geometry_evaluations_", 0),
                            utility_evaluations=getattr(self._delegate, "_utility_evaluations_", 0),
                            mds_fits=copy.deepcopy(getattr(self._delegate, "mds_diagnostics_", [])))
                    params = {**copy.deepcopy(self._original_adapter_params), "max_outer_iterations": self.ae_cap_retry}
                    self._delegate = FairBiasAEAdapter(mode=self.mode, **params)
                self.recovery_result_ = classify_recovery_fit(
                    self._delegate, controller=self.controller, retry_stats=retry_stats)
            self.is_fitted_ = True
            return self
        except BaseException as exc:
            self.recovery_result_["error_type"] = type(exc).__name__
            if getattr(self._delegate, "termination_reason_", None) == "BUDGET_EXHAUSTED":
                self.recovery_result_["status"] = "NO_MODEL_BUDGET_EXHAUSTED"
            raise
        finally:
            self.termination_reason_ = getattr(self._delegate, "termination_reason_", None)
            self.convergence_verified_ = self.recovery_result_.get("convergence_verified", False)
            self.provenance_ = copy.deepcopy(getattr(self._delegate, "provenance_", {}))
            self.provenance_["recovery"] = {
                **copy.deepcopy(self.runtime_spec), "result": copy.deepcopy(self.recovery_result_),
                "kernel": kernel_stats, "mds_retry_receipt": retry_stats,
                "component_source_hashes": source_hashes,
                "fit_attempts": copy.deepcopy(self.fit_attempts_),
            }

    def _require_model(self):
        if not self.is_fitted_:
            raise RuntimeError("Recovery model is not admitted for prediction")

    def predict_event_probability(self, X, A=None):
        self._require_model()
        return self._delegate.predict_event_probability(X, A)

    def predict_decision_proba(self, X, A=None):
        return self.predict_event_probability(X, A)

    def predict(self, X, A=None):
        self._require_model()
        return self._delegate.predict(X, A)


def make_joint_ae_recovery_adapter(config, seed, **recovery_options):
    """Explicit new-worker factory; does not patch the registered registry."""
    modes = {"FAIRBIAS_JOINT": "JOINT", "FAIRBIAS_BM_AE": "BM_AE"}
    if config.get("method") not in modes or seed not in config.get("seeds", []):
        raise ValueError("Recovery requires a registered Joint/BM_AE method and seed")
    if config.get("training_weighted", False):
        raise ValueError("The inherited F/C recovery contract is unweighted")
    params = copy.deepcopy(config["params"])
    mode = params.pop("mode", modes[config["method"]])
    if mode != modes[config["method"]]:
        raise ValueError("Registered method and mode disagree")
    return JointAERecoveryAdapter(mode=mode, backbone=config["backbone"],
                                  random_state=seed, **params, **recovery_options)


def fit_joint_ae_recovery_fc(config, seed, fitting_F, calibration_C, **recovery_options):
    """Fit one admitted development pair; accepts no S/T partition or grid.

    The caller verifies registration/input hashes and uses an isolated process
    with the external supervisor. This function never loads data or emits rows.
    """
    F, C = fitting_F, calibration_C
    for partition, role in ((F, "fitting_F"), (C, "calibration_C")):
        if partition.role != role or partition.year != 2022 or partition.arm_id != config["arm_id"]:
            raise ValueError("Recovery pilot requires the registered arm's F/C 2022 partitions")
        sizes = [len(partition.X_semantic), len(partition.y), len(partition.A), len(partition.record_keys)]
        if sizes[0] == 0 or len(set(sizes)) != 1 or len(set(partition.record_keys)) != sizes[0]:
            raise ValueError("F/C rows or unique record identities do not align")
    if set(F.record_keys) & set(C.record_keys):
        raise ValueError("F/C record identities overlap")
    adapter = make_joint_ae_recovery_adapter(config, seed, **recovery_options)
    return adapter.fit_development(F.X_semantic, F.y, F.A, C.X_semantic, C.y, C.A,
                                  metadata={"F_ids": F.record_keys, "C_ids": C.record_keys})
