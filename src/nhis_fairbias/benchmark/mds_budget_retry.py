"""Versioned MDS budget policy: one fresh seeded refit after a selected cap.

Install this context OUTSIDE the adapter's ``audited_mds`` context. No-cap
fits use exactly the original estimator call. Capped paths are a new budget
policy, not a bit-equivalent reproduction. The first incomplete attempt is
always retained; the existing outer audit still rejects an unresolved cap.
No input matrices, embeddings, or fitted models are retained in the receipt.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import pickle
import sys
import time

import numpy as np
import sklearn
from sklearn.manifold import MDS
from sklearn.manifold import _mds

from fairbias import bias_metric

VERSION = "mds_budget_retry_v1"
SUPPORTED_SKLEARN = "1.9.1"
_active = False


def _source_hashes():
    sources = {
        "retry_module": inspect.getsource(sys.modules[__name__]),
        "MDS_fit_transform": inspect.getsource(MDS.fit_transform),
        "MDS_fit": inspect.getsource(MDS.fit),
        "smacof": inspect.getsource(_mds.smacof),
        "smacof_single": inspect.getsource(_mds._smacof_single),
    }
    return {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}


def _skip_reason(estimator, fit_init, constructor_default):
    seed = estimator.random_state
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        return "RNG_NOT_FIXED_INTEGER"
    if fit_init is not None:
        return "EXPLICIT_FIT_INITIALIZATION"
    if constructor_default is not inspect.Parameter.empty:
        configured = getattr(estimator, "init", constructor_default)
        if type(configured) is not type(constructor_default) or configured != constructor_default:
            return "EXPLICIT_CONSTRUCTOR_INITIALIZATION"
    return None


def _distance_hash(estimator, X):
    # Narrow, read-only witness. Avoid conversion of arbitrary input providers.
    precomputed = (getattr(estimator, "dissimilarity", None) == "precomputed"
                   or getattr(estimator, "metric", None) == "precomputed")
    if precomputed and type(X) is np.ndarray and X.ndim == 2 and not X.dtype.hasobject:
        header = (X.dtype.str + ":" + str(X.shape) + ":").encode()
        return hashlib.sha256(header + np.ascontiguousarray(X).tobytes()).hexdigest()
    return None


def _attempt(estimator, X, y, init, *, records, logical_fit, attempt, fit_record):
    record = {"logical_fit": logical_fit, "attempt": attempt,
              "max_iter": int(estimator.max_iter), "iterations": None, "stress": None,
              "finite": None, "selected_capped": None, "status": "STARTED",
              "input_distance_sha256": _distance_hash(estimator, X)}
    records.append(record)
    started = time.monotonic()
    try:
        # Calling the captured base implementation avoids entering this retry
        # wrapper recursively and never invokes the outer audit prematurely.
        points = MDS.fit_transform(estimator, X, y=y, init=init)
        finite = bool(np.isfinite(estimator.stress_) and np.isfinite(points).all())
        capped = int(estimator.n_iter_) >= int(estimator.max_iter)
        record.update(iterations=int(estimator.n_iter_),
                      stress=float(estimator.stress_) if np.isfinite(estimator.stress_) else None,
                      finite=finite, selected_capped=capped,
                      status=("NONFINITE_MDS" if not finite else
                              "MDS_ITERATION_CAP" if capped else "CONVERGED_BEFORE_CAP"))
        fit_record["status"] = record["status"]
        return points
    except BaseException as exc:
        record.update(status="ERROR" if isinstance(exc, Exception) else "INTERRUPTED",
                      error_type=type(exc).__name__)
        fit_record["status"] = record["status"]
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started


@contextmanager
def mds_budget_retry():
    """Yield aggregate receipts while allowing one 2x-cap fresh seeded refit.

    Only the finite selected initialization reaching its cap triggers retry.
    Mutable/unset RNGs and explicit initialization are left untouched. If the
    retry remains capped or nonfinite, its actual result reaches the existing
    audit, which rejects it. There is no warm start or second retry.
    """
    global _active
    if _active:
        raise RuntimeError("MDS budget retry context is not reentrant")
    if sklearn.__version__ != SUPPORTED_SKLEARN:
        raise RuntimeError("MDS budget retry requires reviewed sklearn " + SUPPORTED_SKLEARN)
    if bias_metric.MDS is not MDS:
        raise RuntimeError("Install mds_budget_retry outside audited_mds; MDS class already replaced")
    original = bias_metric.MDS
    constructor = inspect.signature(original)
    constructor_default = (constructor.parameters["init"].default
                           if "init" in constructor.parameters else inspect.Parameter.empty)
    stats = {"version": VERSION, "policy": "one_fresh_seeded_refit_at_double_iteration_cap",
             "max_factor": 2, "attempts": [], "fits": [], "recoveries": 0,
             "source_hashes": _source_hashes(),
             "library_versions": {"sklearn": sklearn.__version__, "numpy": np.__version__}}

    class RetryMDS(original):
        def fit_transform(self, X, y=None, init=None):
            params = self.get_params(deep=False)
            original_cap = int(params["max_iter"])
            skip = _skip_reason(self, init, constructor_default)
            logical_fit = len(stats["fits"]) + 1
            row = {"logical_fit": logical_fit, "requested_max_iter": original_cap,
                   "actual_max_iter": original_cap, "retry_eligible": skip is None,
                   "skip_reason": skip, "retried": False, "complete": False,
                   "status": "STARTED", "input_distance_sha256": _distance_hash(self, X),
                   "n_components": int(self.n_components), "n_init": int(self.n_init),
                   "seed": int(self.random_state) if isinstance(self.random_state, (int, np.integer)) else None,
                   "parameters_sha256": hashlib.sha256(pickle.dumps(params, protocol=5)).hexdigest()}
            stats["fits"].append(row)
            points = _attempt(self, X, y, init, records=stats["attempts"],
                              logical_fit=logical_fit, attempt=1, fit_record=row)
            first = stats["attempts"][-1]
            row["complete"] = bool(first["finite"] and not first["selected_capped"])
            if skip is not None or not first["finite"] or not first["selected_capped"]:
                return points
            # Fresh constructor + original integer seed resets the complete
            # multi-initialization search. Only its per-initialization cap changes.
            retry_params = dict(params, max_iter=2 * original_cap)
            retry = original(**retry_params)
            row.update(retried=True, actual_max_iter=2 * original_cap)
            points = _attempt(retry, X, y, None, records=stats["attempts"],
                              logical_fit=logical_fit, attempt=2, fit_record=row)
            # The outer AuditedMDS instance must expose the cap actually used,
            # along with the second estimator's own embedding/stress/iterations.
            self.__dict__.update(retry.__dict__)
            second = stats["attempts"][-1]
            row["complete"] = bool(second["finite"] and not second["selected_capped"])
            if row["complete"]:
                stats["recoveries"] += 1
            return points

    _active = True
    try:
        bias_metric.MDS = RetryMDS
        yield stats
    finally:
        bias_metric.MDS = original
        _active = False
