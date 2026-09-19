"""Opt-in exact geometry acceleration for isolated, versioned diagnostics.

The registered benchmark does not import this module. Geometry requests still
pass through the original adapter's budget counter. Only identical successful
deterministic geometry results are reused; candidate decisions are never cached.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import copy
import hashlib
import inspect
import pickle
import time

import numpy as np
import pandas as pd


def _geometry_key(arguments):
    """Hash logical values, independent of pandas' internal block layout.

    Pickling whole DataFrames can miss identical geometry when a transform
    builds a new block layout. Typed, length-delimited parts preserve exact
    floats and object types; no rounding or lossy row hash is used.
    """
    digest = hashlib.sha256()

    def part(tag, payload=b""):
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def walk(value):
        if isinstance(value, pd.DataFrame):
            part(b"frame")
            walk(value.index)
            walk(value.columns)
            for position in range(len(value.columns)):
                walk(value.iloc[:, position])
        elif isinstance(value, pd.Series):
            part(b"series")
            walk(value.name)
            walk(value.index)
            walk(value.dtype)
            walk(value.to_numpy())
        elif isinstance(value, pd.Index):
            part(b"index", type(value).__name__.encode())
            walk(value.names)
            walk(value.dtype)
            walk(value.to_numpy())
        elif isinstance(value, np.ndarray):
            part(b"array", str(value.shape).encode())
            walk(value.dtype)
            if not value.dtype.hasobject:
                part(b"values", np.ascontiguousarray(value).tobytes())
            elif all(type(v) is str for v in value.flat):
                # Lossless Unicode array, including lengths/width through dtype.
                walk(np.fromiter((len(v) for v in value.flat), dtype=np.uint64))
                walk(np.asarray(value.tolist(), dtype=np.str_))
            else:
                for item in value.flat:
                    walk(item)
        elif isinstance(value, dict):
            part(b"dict", str(len(value)).encode())
            for key, item in value.items():
                walk(key)
                walk(item)
        elif isinstance(value, (list, tuple)):
            part(b"sequence", type(value).__name__.encode())
            walk(len(value))
            for item in value:
                walk(item)
        else:
            part(b"scalar", pickle.dumps(value, protocol=5))
    walk(arguments)
    return digest.digest()


def memoized_geometry(original, stats, max_entries=128):
    """Fit-scoped, bounded result memo with full input/configuration identity."""
    if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
        raise ValueError("max_entries must be a positive integer")
    signature = inspect.signature(original)
    cache = OrderedDict()
    stats.update(requests=0, hits=0, misses=0, bypasses=0, entries=0, evictions=0,
                 key_seconds=0.0, reused_compute_seconds=0.0)

    def wrapped(*args, **kwargs):
        stats["requests"] += 1
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        seed = bound.arguments.get("random_state")
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            # None and mutable RNGs must retain their original state progression.
            stats["bypasses"] += 1
            return original(*args, **kwargs)
        tick = time.monotonic()
        # Includes complete frames, indices, dtypes, categories, protected values,
        # weights, feature order, geometry options and seed. No payload is saved.
        key = _geometry_key(bound.arguments)
        stats["key_seconds"] += time.monotonic() - tick
        if key in cache:
            value, elapsed = cache.pop(key)
            cache[key] = (value, elapsed)
            stats["hits"] += 1
            stats["reused_compute_seconds"] += elapsed
            return copy.deepcopy(value)
        stats["misses"] += 1
        tick = time.monotonic()
        value = original(*args, **kwargs)
        elapsed = time.monotonic() - tick
        expected = list(bound.arguments["X"].columns)
        try:
            cacheable = (isinstance(value, dict) and set(value) == set(expected)
                         and all(np.isfinite(v) and v >= 0 for v in value.values()))
        except (TypeError, ValueError):
            cacheable = False
        if cacheable:
            cache[key] = (copy.deepcopy(value), elapsed)
            if len(cache) > max_entries:
                cache.popitem(last=False)
                stats["evictions"] += 1
            stats["entries"] = len(cache)
        return value

    return wrapped


_active = False


def make_checked_mds_distances(original, kernel, stats):
    """Use the same arithmetic kernel after narrow ndarray validation.

    SMACOF repeatedly validates small internal float64 arrays through a general
    public dataframe/sparse/array-API dispatcher. This specialization preserves
    finiteness checks each iteration and delegates every other case unchanged.
    """
    stats.update(fast_calls=0, fallback_calls=0)

    def distances(X, Y=None, *, Y_norm_squared=None, squared=False, X_norm_squared=None):
        if (type(X) is np.ndarray and X.dtype == np.dtype("float64") and X.ndim == 2
                and 0 < X.shape[0] <= 129 and 0 < X.shape[1] <= 64
                and Y is None and Y_norm_squared is None and squared is False
                and X_norm_squared is None and np.isfinite(X).all()):
            stats["fast_calls"] += 1
            return kernel(X, X)
        stats["fallback_calls"] += 1
        return original(X, Y, Y_norm_squared=Y_norm_squared, squared=squared,
                        X_norm_squared=X_norm_squared)
    return distances


def _verified_mds_functions():
    import sklearn
    from sklearn.manifold import _mds
    from sklearn.metrics import pairwise
    expected = {
        "_smacof_single": "d5adae8a9f9dc2c1d6384a36b32f16466641c65084ae24589fcdf2db6b7b1323",
        "euclidean_distances": "ac81afba74d8a4e4a8835350cd25e49e35ff0b75fd3d50e87db4c886503a3d54",
        "_euclidean_distances": "c4cd3d6f23ad2a343f097128f4f4a0ae374f2150e079d9ace856b278ad980065",
    }
    functions = [_mds._smacof_single, pairwise.euclidean_distances, pairwise._euclidean_distances]
    observed = {f.__name__: hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for f in functions}
    if sklearn.__version__ != "1.9.1" or observed != expected:
        raise RuntimeError("MDS fast path requires reviewed sklearn 1.9.1 source fingerprints")
    if _mds.euclidean_distances is not pairwise.euclidean_distances:
        raise RuntimeError("MDS distance function already replaced")
    return _mds, pairwise, observed


@contextmanager
def exact_geometry_acceleration(*, use_plan=False, use_memo=False, max_entries=128):
    """Single-process scoped hook; never install into a running worker.

    The original audited MDS still runs on every cache miss with unchanged
    settings. Cache hits have explicit counters and do not fabricate MDS fits.
    """
    global _active
    if _active:
        raise RuntimeError("Geometry acceleration is not reentrant")
    from fairbias import bias_metric
    original_geometry = bias_metric.compute_bias_concentration
    original_shapley = bias_metric.compute_shapley_distance_matrix
    mds, pairwise, function_hashes = _verified_mds_functions()
    original_distances = mds.euclidean_distances
    stats = {"version": "joint_exact_geometry_v2", "geometry": {"enabled": bool(use_memo)},
             "logical_budget_counts_unchanged": True, "shapley_plan": bool(use_plan),
             "mds_distances": {}, "sklearn_function_hashes": function_hashes,
             "numpy_version": np.__version__, "pandas_version": pd.__version__}
    memo = memoized_geometry(original_geometry, stats["geometry"], max_entries) if use_memo else None
    distances = make_checked_mds_distances(original_distances, pairwise._euclidean_distances,
                                           stats["mds_distances"])
    plan = None
    if use_plan:
        from .joint_shapley_plan import make_planned_shapley
        plan = make_planned_shapley(original_shapley)
    _active = True
    try:
        if memo is not None:
            bias_metric.compute_bias_concentration = memo
        mds.euclidean_distances = distances
        if plan is not None:
            bias_metric.compute_shapley_distance_matrix = plan
        yield stats
    finally:
        bias_metric.compute_bias_concentration = original_geometry
        bias_metric.compute_shapley_distance_matrix = original_shapley
        mds.euclidean_distances = original_distances
        _active = False
