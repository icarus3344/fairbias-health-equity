"""Optional exact H=1 plan for FairBias Shapley distance evaluation.

The public FairBias implementation remains the authority.  This module only
provides a wrapper factory: supported H=1 calls use a bounded, immutable
context/index plan and NumPy reductions; every other call is delegated to the
function supplied to the factory.  Supplying the original function explicitly
is intentional so installing the wrapper cannot recurse back into itself.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from fairbias.bias_metric import ORIGIN, get_subsets


@dataclass(frozen=True)
class _PairPlan:
    """Context pair indices for one upper-triangular matrix entry."""

    left: Tuple[int, ...]
    right: Tuple[int, ...]
    author_max_pair: bool


@dataclass(frozen=True)
class ShapleyH1Plan:
    """Immutable H=1 subset and matrix-entry plan."""

    features: Tuple[str, ...]
    schema: Tuple[str, ...]
    subset_rows: Tuple[Tuple[int, ...], ...]
    pair_plans: Tuple[_PairPlan, ...]
    node_pairs: Tuple[Tuple[int, int], ...]

    def evaluate(self, df_s: pd.DataFrame, multigroup_aggregation: str) -> Tuple[np.ndarray, List[str]]:
        """Evaluate the planned contexts with the original reduction order."""
        if multigroup_aggregation not in ("mean_pair", "author_max_pair"):
            raise ValueError(f"Unsupported multigroup_aggregation: {multigroup_aggregation}")

        sq = df_s.to_numpy(dtype=float) ** 2
        ncols = sq.shape[1]
        values: Dict[int, np.ndarray] = {}
        by_size: Dict[int, List[int]] = {}
        for sid, rows in enumerate(self.subset_rows):
            if rows:
                by_size.setdefault(len(rows), []).append(sid)
            else:
                values[sid] = np.zeros(ncols)

        # A separate batch per cardinality preserves each original subset's
        # row order and therefore its floating-point reduction order.
        for size, sids in by_size.items():
            stacked = np.stack([sq[list(self.subset_rows[sid])] for sid in sids], axis=0)
            reduced = np.sqrt(stacked.sum(axis=1) / size)
            for row, sid in zip(reduced, sids):
                values[sid] = row

        n_nodes = len(self.features) + 1
        dist = np.zeros((n_nodes, n_nodes))
        for plan, (i, j) in zip(self.pair_plans, self.node_pairs):
            pair_means = []
            for left_sid, right_sid in zip(plan.left, plan.right):
                v1 = values[left_sid]
                v2 = values[right_sid]
                if multigroup_aggregation == "author_max_pair":
                    max_v1 = float(np.max(v1)) if v1.size else 0.0
                    max_v2 = float(np.max(v2)) if v2.size else 0.0
                    pair_means.append(abs(max_v1 - max_v2))
                else:
                    diff = np.abs(v1 - v2)
                    pair_means.append(float(np.mean(diff)) if diff.size else 0.0)
            d_ab = float(np.mean(pair_means)) if pair_means else 0.0
            dist[i, j] = d_ab
            dist[j, i] = d_ab
        return dist, list(self.features) + [ORIGIN]


def build_shapley_h1_plan(features: Sequence[str], schema: Sequence[str]) -> ShapleyH1Plan:
    """Compile the exact H=1 context/index plan for one feature schema."""
    feats = tuple(features)
    schema_tuple = tuple(schema)
    feature_index = {name: i for i, name in enumerate(schema_tuple)}
    subset_ids: Dict[Tuple[str, ...], int] = {}
    subset_rows: List[Tuple[int, ...]] = []

    def subset_id(names: Sequence[str]) -> int:
        key = tuple(sorted(names))
        if key not in subset_ids:
            subset_ids[key] = len(subset_rows)
            subset_rows.append(tuple(feature_index[name] for name in key if name in feature_index))
        return subset_ids[key]

    node_pairs: List[Tuple[int, int]] = []
    pair_plans: List[_PairPlan] = []
    origin_index = len(feats)
    nodes = list(feats) + [ORIGIN]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if a != ORIGIN and b != ORIGIN:
                available = [f for f in feats if f != a and f != b]
                contexts = get_subsets(available, 1)
                left, right = [], []
                for context in contexts:
                    left.append(subset_id(list(context) + [a]))
                    right.append(subset_id(list(context) + [b]))
            else:
                m = a if b == ORIGIN else b
                available = [f for f in feats if f != m]
                contexts = get_subsets(available, 1)
                left, right = [], []
                for context in contexts:
                    left.append(subset_id(list(context) + [m]))
                    right.append(subset_id(list(context)))
            node_pairs.append((i, j))
            pair_plans.append(_PairPlan(tuple(left), tuple(right), False))
    return ShapleyH1Plan(feats, schema_tuple, tuple(subset_rows), tuple(pair_plans), tuple(node_pairs))


def make_planned_shapley_distance_matrix(
    original: Callable[..., Tuple[np.ndarray, List[str]]],
    max_cached_plans: int = 8,
) -> Callable[..., Tuple[np.ndarray, List[str]]]:
    """Return an exact H=1 wrapper with bounded plan caching.

    ``original`` must be the unwrapped implementation.  Unsupported H/order,
    invalid input, empty input, or missing schema entries are delegated so the
    original validation and fallback behavior remain authoritative.
    """
    if isinstance(max_cached_plans, bool) or int(max_cached_plans) < 1:
        raise ValueError("max_cached_plans must be a positive integer")
    cache: "OrderedDict[Tuple[Tuple[str, ...], Tuple[str, ...]], ShapleyH1Plan]" = OrderedDict()
    limit = int(max_cached_plans)

    def planned(df_s: pd.DataFrame, features: List[str], h_order: int = 1, multigroup_aggregation: str = "mean_pair"):
        if h_order != 1 or not isinstance(df_s, pd.DataFrame) or not isinstance(features, (list, tuple)):
            return original(df_s, features, h_order=h_order, multigroup_aggregation=multigroup_aggregation)
        if df_s.empty or df_s.shape[1] == 0 or df_s.index.has_duplicates:
            return original(df_s, features, h_order=h_order, multigroup_aggregation=multigroup_aggregation)
        if multigroup_aggregation not in ("mean_pair", "author_max_pair"):
            return original(df_s, features, h_order=h_order, multigroup_aggregation=multigroup_aggregation)
        feats, schema = tuple(features), tuple(df_s.index)
        if len(set(feats)) != len(feats) or any(f not in schema for f in feats):
            return original(df_s, features, h_order=h_order, multigroup_aggregation=multigroup_aggregation)
        arr = df_s.to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            return original(df_s, features, h_order=h_order, multigroup_aggregation=multigroup_aggregation)
        key = (feats, schema)
        plan = cache.get(key)
        if plan is None:
            plan = build_shapley_h1_plan(feats, schema)
            cache[key] = plan
            cache.move_to_end(key)
            while len(cache) > limit:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        return plan.evaluate(df_s, multigroup_aggregation)

    planned.plan_cache = cache  # type: ignore[attr-defined]
    return planned


def make_planned_shapley(original_function: Callable[..., Tuple[np.ndarray, List[str]]], max_cached_plans: int = 8):
    """Compatibility alias for the optional benchmark optimizer."""
    return make_planned_shapley_distance_matrix(original_function, max_cached_plans=max_cached_plans)
