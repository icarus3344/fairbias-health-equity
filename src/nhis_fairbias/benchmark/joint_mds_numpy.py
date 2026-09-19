"""Opt-in NumPy specialization of the reviewed small MDS distance kernel.

This module changes no MDS settings, stopping rules, initializations, or RNG
operations. The original implementation remains the fallback for every input
outside the existing narrow float64 ndarray contract. Hooks are process scoped;
the registered benchmark does not import this module.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect

import numpy as np

from .joint_budget_optimization import (
    _verified_mds_functions,
    exact_geometry_acceleration,
    make_checked_mds_distances,
)


_DEPENDENCY_HASHES = {
    "row_norms": "f9d19a4bcb9966e7e0d2a4785f686f1af9dc310f9137e9656f3ee31f9cf36faa",
    "safe_sparse_dot": "c8259546787ab8ac484494c57e1e32550a42a55c9f67346521f77f56a0a13185",
    "_modify_in_place_if_numpy": "39e05807f791e1028110c8a41fb720fac9264b7bfd325122b59e92a421c65bf9",
    "_fill_diagonal": "8245acea466efe23f5b65bc6e46eea984ce035ef407ccd81d1434d33694621b7",
    "get_namespace_and_device": "0e7dc7e150c2ef6a2d38cc16c1dc78d194b24d01d1e650b94c71043b0a3b7318",
    "get_namespace": "0f3c6354958cd8685411109d82f96f4eb4f85a65744cc112223d33a8c0b3168e",
    "_is_numpy_namespace": "534aec2b244ae085faadab98b17f8a13b4fea599da28c818815d064ad4b861c2",
    "_validate_diagonal_args": "a09401099a29a8d705c031b6f12657e445a0a741380f0ddeea6e4a5219acf25b",
}


def _numpy_self_distances(X, Y):
    """Same reviewed float64 operations, in the same order, using NumPy.

    Called only after ``make_checked_mds_distances`` admits plain finite 2-D
    float64 arrays and supplies ``Y is X``. ``row_norms`` uses this einsum;
    ``safe_sparse_dot`` uses ndarray matmul. Do not replace the row reduction
    with sum, fuse the additions, or substitute another numerical backend.
    """
    XX = np.einsum("ij,ij->i", X, X)[:, None]
    distances = -2 * (X @ X.T)
    distances += XX
    distances += XX.T
    np.maximum(distances, np.asarray(0, dtype=distances.dtype), out=distances)
    np.fill_diagonal(distances, 0)
    np.sqrt(distances, out=distances)
    return distances


def make_numpy_mds_distances(original, stats):
    """Build the existing checked wrapper around the NumPy specialization."""
    return make_checked_mds_distances(original, _numpy_self_distances, stats)


def _verified_numpy_dependencies():
    """Reject unreviewed sklearn sources and altered imported aliases."""
    from sklearn.utils import _array_api, extmath

    mds, pairwise, base_hashes = _verified_mds_functions()
    dependencies = {
        "row_norms": extmath.row_norms,
        "safe_sparse_dot": extmath.safe_sparse_dot,
        "_modify_in_place_if_numpy": _array_api._modify_in_place_if_numpy,
        "_fill_diagonal": _array_api._fill_diagonal,
        "get_namespace_and_device": _array_api.get_namespace_and_device,
        "get_namespace": _array_api.get_namespace,
        "_is_numpy_namespace": _array_api._is_numpy_namespace,
        "_validate_diagonal_args": _array_api._validate_diagonal_args,
    }
    aliases = (
        (pairwise, "row_norms"),
        (pairwise, "safe_sparse_dot"),
        (pairwise, "_modify_in_place_if_numpy"),
        (pairwise, "_fill_diagonal"),
        (pairwise, "get_namespace_and_device"),
        (extmath, "get_namespace"),
        (extmath, "_is_numpy_namespace"),
    )
    if any(getattr(module, name) is not dependencies[name] for module, name in aliases):
        raise RuntimeError("NumPy MDS requires reviewed sklearn dependency identities")
    try:
        observed = {
            name: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
            for name, function in dependencies.items()
        }
    except (OSError, TypeError) as exc:
        raise RuntimeError("NumPy MDS dependency sources cannot be verified") from exc
    if observed != _DEPENDENCY_HASHES:
        raise RuntimeError("NumPy MDS requires reviewed sklearn dependency source fingerprints")
    return mds, pairwise, {**base_hashes, **observed}


@contextmanager
def numpy_mds_acceleration():
    """Install the exact small-array specialization in an isolated process.

    Whole-geometry memoization stays disabled. The enclosing v2 context owns
    non-reentrancy and final restoration even for BaseException. Source hashes,
    numerical environment and distance-call counters remain visible. Logical
    geometry budgets remain owned by the unchanged adapter.
    """
    mds, pairwise, source_hashes = _verified_numpy_dependencies()
    with exact_geometry_acceleration(use_plan=False, use_memo=False) as stats:
        previous_distances = mds.euclidean_distances
        stats["base_version"] = stats["version"]
        stats["version"] = "joint_numpy_mds_v1"
        stats["sklearn_function_hashes"] = source_hashes
        stats["numpy_kernel_source_sha256"] = hashlib.sha256(
            inspect.getsource(_numpy_self_distances).encode()
        ).hexdigest()
        stats["numerical_backend"] = "installed_numpy_same_operation_order"
        stats["mds_distances"] = {}
        mds.euclidean_distances = make_numpy_mds_distances(
            pairwise.euclidean_distances, stats["mds_distances"]
        )
        try:
            yield stats
        finally:
            mds.euclidean_distances = previous_distances
