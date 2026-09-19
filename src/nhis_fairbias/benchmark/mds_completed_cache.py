"""Exact completed-fit reuse for isolated, explicitly registered MDS replays.

This is not a search-state checkpoint or permission to resume an old wall-time
budget. Only an MDS fit whose selected initialization passes the existing audit
is retained. Install outside ``audited_mds``; logical audits and geometry budgets
still execute. The old ``mds_budget_retry`` wrapper is deliberately unsupported.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import uuid
import warnings

import numpy as np
import scipy
import sklearn
from joblib import effective_n_jobs
from sklearn.manifold import MDS, _mds
from sklearn.metrics import pairwise
from sklearn.utils import _array_api, extmath
from threadpoolctl import threadpool_info

from fairbias import bias_metric
from .joint_budget_optimization import make_checked_mds_distances
from .joint_mds_numpy import _numpy_self_distances


VERSION = "mds_completed_cache_v2"
_active = False
_MAX_RAM_ENTRIES = 128
_ARRAY_FIELDS = ("embedding_", "dissimilarity_matrix_")
_SCALAR_FIELDS = ("stress_", "n_iter_", "n_features_in_", "_init", "_metric", "_metric_mds")


class MDSCacheIntegrityError(RuntimeError):
    """An allegedly complete cache entry failed its integrity contract."""


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _typed(value):
    # Preserve parameter types rather than treating True, 1, and 1.0 as equal.
    if value is None:
        return ["None"]
    if isinstance(value, (bool, np.bool_)):
        return [type(value).__name__, bool(value)]
    if isinstance(value, (int, np.integer)):
        return [type(value).__name__, str(int(value))]
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return [type(value).__name__, float(value).hex()]
    if type(value) is str:
        return ["str", value]
    if type(value) in (tuple, list):
        return [type(value).__name__, [_typed(v) for v in value]]
    if type(value) is dict:
        return ["dict", [[_typed(k), _typed(v)] for k, v in value.items()]]
    raise TypeError("Unsupported cache parameter type")


def _source_functions():
    functions = {
        "MDS.fit": MDS.fit, "MDS.fit_transform": MDS.fit_transform,
        "smacof": _mds.smacof, "smacof_single": _mds._smacof_single,
        "public_distance": pairwise.euclidean_distances,
        "private_distance": pairwise._euclidean_distances,
        "numpy_distance": _numpy_self_distances,
        "checked_distance_factory": make_checked_mds_distances,
    }
    for name in ("row_norms", "safe_sparse_dot", "_modify_in_place_if_numpy",
                 "_fill_diagonal", "get_namespace_and_device"):
        functions["pairwise." + name] = getattr(pairwise, name)
    for name in ("get_namespace", "_is_numpy_namespace"):
        functions["extmath." + name] = getattr(extmath, name)
    functions["array_api._validate_diagonal_args"] = _array_api._validate_diagonal_args
    for name in ("check_symmetric", "check_array", "check_random_state", "validate_data"):
        functions["mds." + name] = getattr(_mds, name)
    return functions


def _kernel_identity():
    function = _mds.euclidean_distances
    if function is pairwise.euclidean_distances:
        return "sklearn_public_distance"
    probe = make_checked_mds_distances(pairwise.euclidean_distances, _numpy_self_distances, {})
    if getattr(function, "__code__", None) is not probe.__code__:
        return None
    closure = inspect.getclosurevars(function).nonlocals
    if closure.get("original") is not pairwise.euclidean_distances:
        return None
    if closure.get("kernel") is _numpy_self_distances:
        return "joint_numpy_mds_v1"
    if closure.get("kernel") is pairwise._euclidean_distances:
        return "joint_exact_geometry_v2"
    return None


def _environment_identity(library_hashes):
    pools = threadpool_info()
    blas = [p for p in pools if p.get("user_api") == "blas"]
    if not blas or any(p.get("num_threads") != 1 for p in blas):
        return None
    witness = []
    for pool in pools:
        path = Path(pool["filepath"])
        stat = path.stat()
        stamp = (str(path), stat.st_size, stat.st_mtime_ns)
        if stamp not in library_hashes:
            library_hashes[stamp] = _file_sha(path)
        witness.append({**pool, "library_sha256": library_hashes[stamp]})
    # Linux may enumerate the same loaded libraries in a different order in
    # another process. Their multiset is the identity, not enumeration order.
    # Keep every field and duplicate: actual library/config changes still miss.
    witness.sort(key=_json_bytes)
    cpu = np._core._multiarray_umath
    return {
        "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "sklearn_config": sklearn.get_config(), "numpy_error_policy": np.geterr(),
        "numpy_build": json.loads(_json_bytes(np.__config__.CONFIG)),
        "cpu_features": dict(cpu.__cpu_features__), "cpu_baseline": list(cpu.__cpu_baseline__),
        "cpu_dispatch": list(cpu.__cpu_dispatch__), "threadpools": witness,
        "thread_env": {name: os.environ.get(name) for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")},
    }


def _skip_reason(estimator, X, y, init):
    seed = estimator.random_state
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        return "RNG_NOT_FIXED_INTEGER"
    if (init is not None or y is not None or type(estimator.init) is not str
            or estimator.init not in ("warn", "random")):
        return "UNSUPPORTED_INITIALIZATION_OR_Y"
    precomputed = (estimator.dissimilarity == "precomputed"
                   and (estimator.metric is True or estimator.metric == "euclidean")) or (
                   estimator.dissimilarity == "deprecated" and estimator.metric == "precomputed")
    if not precomputed:
        return "NOT_PRECOMPUTED"
    effective_metric = estimator.metric if type(estimator.metric) is bool else estimator.metric_mds
    if effective_metric is not True:
        return "NONMETRIC_MDS_NOT_ADMITTED"
    if (type(X) is not np.ndarray or X.dtype != np.dtype("float64") or X.ndim != 2
            or not X.flags.c_contiguous or X.shape[0] != X.shape[1]
            or not 2 <= X.shape[0] <= 129 or not np.isfinite(X).all()
            or not np.array_equal(X, X.T) or np.any(X < 0)):
        return "UNSUPPORTED_DISTANCE_ARRAY"
    if (type(estimator.n_jobs) not in (type(None), int) or estimator.n_jobs not in (None, 1)
            or effective_n_jobs(estimator.n_jobs) != 1):
        return "NOT_SERIAL_JOBLIB"
    # A hit must not suppress a warning configured to become an exception.
    if any(entry[0] == "error" and issubclass(FutureWarning, entry[2]) for entry in warnings.filters):
        return "WARNING_ERROR_POLICY"
    if (type(estimator.n_components) is not int or not 1 <= estimator.n_components <= 64
            or type(estimator.n_init) is not int or estimator.n_init < 1
            or type(estimator.max_iter) is not int or estimator.max_iter < 1
            or type(estimator.eps) not in (int, float) or not np.isfinite(estimator.eps)
            or estimator.eps < 0):
        return "UNSUPPORTED_PARAMETERS"
    return None


def _copy_state(state):
    return {name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in state.items()}


def _validate_state(state, X, params):
    if set(state) != set(_ARRAY_FIELDS + _SCALAR_FIELDS):
        raise MDSCacheIntegrityError("Unexpected fitted-state schema")
    points = state["embedding_"]
    dist = state["dissimilarity_matrix_"]
    if (type(points) is not np.ndarray or points.dtype != np.dtype("float64")
            or points.shape != (len(X), params["n_components"]) or not np.isfinite(points).all()
            or type(dist) is not np.ndarray or dist.dtype != X.dtype or dist.shape != X.shape
            or dist.tobytes() != X.tobytes()):
        raise MDSCacheIntegrityError("Invalid or mismatched cached arrays")
    if (type(state["n_iter_"]) is not int or not 1 <= state["n_iter_"] < params["max_iter"]
            or type(state["n_features_in_"]) is not int or state["n_features_in_"] != X.shape[1]
            or type(state["stress_"]) is not float or not np.isfinite(state["stress_"])
            or state["stress_"] < 0 or state["_init"] != "random"
            or state["_metric"] != "precomputed" or type(state["_metric_mds"]) is not bool):
        raise MDSCacheIntegrityError("Cached fit fails selected-initialization admission")
    expected_metric = params["metric"] if type(params["metric"]) is bool else params["metric_mds"]
    if state["_metric_mds"] != expected_metric:
        raise MDSCacheIntegrityError("Cached effective metric differs from parameters")


def _read_entry(path, request, X, params):
    if not path.exists():
        return None
    try:
        manifest_path, payload_path = path / "manifest.json", path / "payload.npz"
        if path.is_symlink() or manifest_path.is_symlink() or payload_path.is_symlink():
            raise MDSCacheIntegrityError("Cache symlinks are not admitted")
        manifest = json.loads(manifest_path.read_bytes())
        if set(manifest) != {"version", "request", "payload_sha256", "state", "origin", "manifest_sha256"}:
            raise MDSCacheIntegrityError("Unexpected cache manifest schema")
        seal = manifest.pop("manifest_sha256")
        if seal != _sha_bytes(_json_bytes(manifest)):
            raise MDSCacheIntegrityError("Cache manifest checksum mismatch")
        if manifest["version"] != VERSION or manifest["request"] != request:
            raise MDSCacheIntegrityError("Cache request identity mismatch")
        origin = manifest["origin"]
        if (type(origin) is not dict or set(origin) != {
                "cache_key", "producer_version", "producer_id", "producer_logical_fit",
                "selected_initialization_status"}
                or origin["cache_key"] != path.name or origin["producer_version"] != VERSION
                or type(origin["producer_id"]) is not str or len(origin["producer_id"]) != 32
                or type(origin["producer_logical_fit"]) is not int or origin["producer_logical_fit"] < 1
                or origin["selected_initialization_status"] != "CONVERGED_BEFORE_CAP"):
            raise MDSCacheIntegrityError("Invalid completed-fit origin receipt")
        if _file_sha(payload_path) != manifest["payload_sha256"]:
            raise MDSCacheIntegrityError("Cache payload checksum mismatch")
        with np.load(payload_path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(_ARRAY_FIELDS):
                raise MDSCacheIntegrityError("Unexpected cache array schema")
            state = {**manifest["state"], **{name: arrays[name].copy() for name in _ARRAY_FIELDS}}
        _validate_state(state, X, params)
        return state, manifest["origin"]
    except MDSCacheIntegrityError:
        raise
    except Exception as exc:
        raise MDSCacheIntegrityError("Incomplete or unreadable completed MDS cache entry") from exc


def _write_entry(path, request, state, origin):
    temporary = Path(tempfile.mkdtemp(prefix=".partial-", dir=path.parent))
    try:
        payload = temporary / "payload.npz"
        with payload.open("wb") as handle:
            np.savez(handle, **{name: state[name] for name in _ARRAY_FIELDS})
            handle.flush()
            os.fsync(handle.fileno())
        manifest = {"version": VERSION, "request": request, "payload_sha256": _file_sha(payload),
                    "state": {name: state[name] for name in _SCALAR_FIELDS}, "origin": origin}
        manifest["manifest_sha256"] = _sha_bytes(_json_bytes(manifest))
        with (temporary / "manifest.json").open("wb") as handle:
            handle.write(_json_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        # A completed entry is one atomic directory publication. Orphan .partial
        # directories from a killed process are never considered cache entries.
        try:
            os.rename(temporary, path)
        except OSError:
            if not path.exists():
                raise
            existing = _read_entry(path, request, state["dissimilarity_matrix_"],
                                   {"n_components": state["embedding_"].shape[1],
                                    "max_iter": request["admission_max_iter"],
                                    "metric": state["_metric_mds"], "metric_mds": state["_metric_mds"]})
            if existing is None or any(
                    existing[0][name].tobytes() != state[name].tobytes() for name in _ARRAY_FIELDS
            ) or any(existing[0][name] != state[name] for name in _SCALAR_FIELDS):
                raise MDSCacheIntegrityError("Concurrent cache writer produced different fitted state")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


@contextmanager
def completed_mds_cache(cache_dir, namespace):
    """Yield cache receipts; cache only complete, serial, fixed-integer-seed fits.

    ``namespace`` is a stable caller-authorized F/C task-family identity, not a
    run identifier or the outer AE/wall-time budget. Environment and actual
    supported distance-kernel identity are captured for each logical fit.
    Informational warnings are not replayed on hits; error-warning policies
    bypass reuse. The returned stats never contain distance matrices/embeddings.
    """
    global _active
    if _active:
        raise RuntimeError("Completed MDS cache is not reentrant")
    if type(namespace) is not str or not namespace.strip():
        raise ValueError("A nonempty stable cache namespace is required")
    if sklearn.__version__ != "1.9.1" or bias_metric.MDS is not MDS:
        raise RuntimeError("Completed MDS cache requires original sklearn 1.9.1 MDS outside other class hooks")
    root = Path(cache_dir) / _sha_bytes(namespace.encode())
    root.mkdir(parents=True, exist_ok=True)
    functions = _source_functions()
    source_tokens = {name: (fn, getattr(fn, "__code__", None)) for name, fn in functions.items()}
    sources = {name: _sha_bytes(inspect.getsource(fn).encode()) for name, fn in functions.items()}
    sources["cache_module"] = _file_sha(__file__)
    source_digest = _sha_bytes(_json_bytes(sources))
    stats = {"version": VERSION, "namespace_sha256": _sha_bytes(namespace.encode()),
             "producer_id": uuid.uuid4().hex,
             "requests": 0, "hits": 0, "ram_hits": 0, "disk_hits": 0, "misses": 0,
             "bypasses": 0, "actual_fits": 0, "writes": 0, "uncacheable_results": 0,
             "max_ram_entries": _MAX_RAM_ENTRIES, "ram_entries": 0, "ram_evictions": 0,
             "logical_budget_counts_unchanged": True, "selected_initialization_only": True,
             "search_state_checkpoint": False, "source_hashes": sources, "fits": [],
             "environments": {}}
    ram, library_hashes = OrderedDict(), {}
    original = bias_metric.MDS

    class CachedMDS(original):
        def fit_transform(self, X, y=None, init=None):
            stats["requests"] += 1
            row = {"logical_fit": stats["requests"], "status": "STARTED", "actual_fit": False}
            stats["fits"].append(row)
            skip = _skip_reason(self, X, y, init)
            params = self.get_params(deep=False)
            kernel = _kernel_identity()
            current = {name: (fn, getattr(fn, "__code__", None)) for name, fn in _source_functions().items()}
            if current != source_tokens:
                raise MDSCacheIntegrityError("MDS source callable changed inside cache context")
            env = _environment_identity(library_hashes) if skip is None else None
            skip = skip or ("UNSUPPORTED_KERNEL" if kernel is None else None)
            skip = skip or ("NUMERICAL_ENVIRONMENT_NOT_ADMITTED" if env is None else None)
            try:
                parameters = _typed(params)
            except TypeError:
                skip = skip or "UNSUPPORTED_PARAMETERS"
            if skip:
                stats["bypasses"] += 1
                stats["actual_fits"] += 1
                row.update(status="BYPASS", reason=skip, actual_fit=True)
                try:
                    return super().fit_transform(X, y=y, init=init)
                except BaseException as exc:
                    row.update(status="ERROR" if isinstance(exc, Exception) else "INTERRUPTED",
                               error_type=type(exc).__name__)
                    raise
            environment = _sha_bytes(_json_bytes(env))
            stats["environments"][environment] = env
            request = {"namespace_sha256": stats["namespace_sha256"], "source_sha256": source_digest,
                       "environment_sha256": environment, "kernel": kernel,
                       "parameters": parameters, "dtype": X.dtype.str, "shape": list(X.shape),
                       "input_sha256": _sha_bytes(X.tobytes()), "admission_max_iter": int(self.max_iter)}
            key = _sha_bytes(_json_bytes(request))
            row.update(cache_key=key, environment_sha256=environment, kernel=kernel)
            try:
                cached = ram.get(key)
                tier = "ram" if cached is not None else "disk"
                if cached is None:
                    cached = _read_entry(root / key, request, X, params)
                if cached is not None:
                    state, origin = cached
                    _validate_state(state, X, params)
                    self.__dict__.update(_copy_state(state))
                    self.stress_ = np.float64(state["stress_"])
                    stats["hits"] += 1
                    stats[tier + "_hits"] += 1
                    row.update(status="CACHE_HIT", cache_tier=tier, origin=origin)
                else:
                    stats["misses"] += 1
                    stats["actual_fits"] += 1
                    row["actual_fit"] = True
                    points = super().fit_transform(X, y=y, init=init)
                    state = {name: getattr(self, name) for name in _ARRAY_FIELDS + _SCALAR_FIELDS}
                    state["stress_"] = float(state["stress_"])
                    state["n_iter_"] = int(state["n_iter_"])
                    state["n_features_in_"] = int(state["n_features_in_"])
                    try:
                        _validate_state(state, X, params)
                    except MDSCacheIntegrityError:
                        stats["uncacheable_results"] += 1
                        row["status"] = "RESULT_NOT_CACHEABLE"
                        return points
                    state = _copy_state(state)
                    origin = {"cache_key": key, "producer_version": VERSION,
                              "producer_id": stats["producer_id"],
                              "producer_logical_fit": row["logical_fit"],
                              "selected_initialization_status": "CONVERGED_BEFORE_CAP"}
                    _write_entry(root / key, request, state, origin)
                    stats["writes"] += 1
                    row.update(status="COMPUTED_AND_STORED", origin=origin)
                ram[key] = (state, origin)
                ram.move_to_end(key)
                while len(ram) > _MAX_RAM_ENTRIES:
                    ram.popitem(last=False)
                    stats["ram_evictions"] += 1
                stats["ram_entries"] = len(ram)
                return self.embedding_
            except BaseException as exc:
                row.update(status="ERROR" if isinstance(exc, Exception) else "INTERRUPTED",
                           error_type=type(exc).__name__)
                raise

    _active = True
    try:
        bias_metric.MDS = CachedMDS
        yield stats
    finally:
        bias_metric.MDS = original
        _active = False
