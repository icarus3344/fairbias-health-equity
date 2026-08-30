"""Paper-level bias concentration metric (d_phi).

Implements the bias concentration pipeline described in the reference paper:

1. Group-wise feature divergences ``df_S`` for every protected-attribute group pair.
2. Subset values ``v(S) = sqrt(sum_{f in S} divergence_f^2)`` (baseline ``d1B`` aggregation).
3. Shapley-truncated distance matrix over features plus an ``origin`` node,
   ``dist(a, b) = mean_S |v(S + a) - v(S + b)|``.
4. Metric MDS embedding with stress-elbow dimension selection, translated so the
   origin node sits at (0, ..., 0).
5. ``d_phi``: Euclidean distance of each feature embedding to the origin.

Reference implementations in the frozen baseline: ``eval.py`` (calculate_epsilon)
and ``config.py`` (default divergence/scale/MDS parameters).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.manifold import MDS

ORIGIN = "origin"


def get_subsets(features: Sequence[str], h_order: int = 1) -> List[List[str]]:
    """
    Generate low-order interaction subsets of size k in [1, H] (truncated Shapley orders).

    Complexity is O(N^H) instead of the legacy O(2^N) high-order enumeration;
    H is capped at 3 as a safety bound.
    """
    feats = list(features)
    max_h = max(1, min(int(h_order), 3, len(feats)))
    subsets: List[List[str]] = []
    for k in range(1, max_h + 1):
        for comb in combinations(feats, k):
            subsets.append(list(comb))
    return subsets


def compute_pairwise_divergences(
    X: pd.DataFrame,
    o_series: pd.Series,
    cate_attrs: List[str],
    num_attrs: List[str],
    num_method: str = "num-a",
    cat_method: str = "cat-a",
    scale: str = "mean",
) -> pd.DataFrame:
    """
    Compute per-feature divergence between every pair of protected groups.

    Returns a DataFrame with features as index and one column per group pair
    (keyed ``"{p}_{n}"``). Divergence definitions follow the frozen baseline:

    - ``cat-a``: ``(1/K) * sum |prop_p - prop_n|`` over the union of categories.
    - ``num-a``: absolute difference of group means after within-pair min-max
      normalization of the feature.
    """
    o_col = "_prot_"
    df = pd.concat(
        [X.reset_index(drop=True), o_series.reset_index(drop=True).rename(o_col)],
        axis=1,
    )
    groups = sorted(df[o_col].dropna().unique())
    cate_set = set(cate_attrs)
    num_set = set(num_attrs)

    columns: Dict[str, pd.Series] = {}
    for p, n in combinations(groups, 2):
        mask = df[o_col].isin([p, n])
        sub = df[mask].copy()
        mask_p = sub[o_col] == p
        mask_n = sub[o_col] == n

        num_diff: Dict[str, float] = {}
        for col in num_attrs:
            if col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").astype(float)
            min_val, max_val = float(vals.min()), float(vals.max())
            if max_val > min_val:
                vals = (vals - min_val) / (max_val - min_val)
            p_vals = vals[mask_p].dropna()
            n_vals = vals[mask_n].dropna()
            if num_method == "num-a":
                if len(p_vals) == 0 or len(n_vals) == 0:
                    num_diff[col] = 0.0
                else:
                    num_diff[col] = float(abs(p_vals.mean() - n_vals.mean()))
            else:
                raise ValueError(f"Unsupported num divergence method: {num_method}")

        cat_diff: Dict[str, float] = {}
        for col in cate_attrs:
            if col not in sub.columns:
                continue
            p_counts = sub.loc[mask_p, col].value_counts()
            n_counts = sub.loc[mask_n, col].value_counts()
            if len(p_counts) == 0 or len(n_counts) == 0:
                cat_diff[col] = 0.0
                continue
            union_index = pd.Index(list(p_counts.index) + list(n_counts.index)).unique()
            p_counts = p_counts.reindex(union_index, fill_value=0)
            n_counts = n_counts.reindex(union_index, fill_value=0)
            if cat_method == "cat-a":
                k = max(1, len(union_index))
                cat_diff[col] = float(
                    (1.0 / k) * (p_counts / p_counts.sum() - n_counts / n_counts.sum()).abs().sum()
                )
            else:
                raise ValueError(f"Unsupported cat divergence method: {cat_method}")

        # Scale each divergence family separately (baseline 'mean' semantics)
        num_series = pd.Series(num_diff, dtype="float64")
        cat_series = pd.Series(cat_diff, dtype="float64")
        if scale == "mean":
            if not num_series.empty and float(num_series.mean()) > 1e-12:
                num_series = num_series / float(num_series.mean())
            if not cat_series.empty and float(cat_series.mean()) > 1e-12:
                cat_series = cat_series / float(cat_series.mean())
        elif scale != "none":
            raise ValueError(f"Unsupported divergence scale: {scale}")

        columns[f"{p}_{n}"] = pd.concat([num_series, cat_series])

    if not columns:
        return pd.DataFrame(index=list(X.columns))
    df_s = pd.DataFrame(columns).fillna(0.0)
    # Keep every feature row even if it had no divergence entry
    for col in X.columns:
        if col not in df_s.index:
            df_s.loc[col] = 0.0
    return df_s


def compute_shapley_distance_matrix(
    df_s: pd.DataFrame,
    features: List[str],
    h_order: int = 1,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the Shapley-truncated distance matrix over features plus the origin node.

    ``dist(a, b) = mean over pair-columns of mean over subsets S (|v(S+a) - v(S+b)|)``,
    where subsets S range over the empty set and all subsets of the remaining
    features of size <= h_order (subset values use the ``d1B`` norm).
    """
    nodes = list(features) + [ORIGIN]
    n_nodes = len(nodes)
    dist = np.zeros((n_nodes, n_nodes))

    if df_s.empty or df_s.shape[1] == 0:
        return dist, nodes

    df_s_sq = df_s ** 2
    value_cache: Dict[Tuple[str, ...], np.ndarray] = {}
    zero_vec = np.zeros(df_s.shape[1])

    def subset_value(subset: Sequence[str]) -> np.ndarray:
        key = tuple(sorted(subset))
        if not key:
            return zero_vec
        cached = value_cache.get(key)
        if cached is None:
            rows = [c for c in key if c in df_s_sq.index]
            if rows:
                cached = np.sqrt(df_s_sq.loc[rows].sum(axis=0).to_numpy(dtype=float))
            else:
                cached = zero_vec
            value_cache[key] = cached
        return cached

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            a, b = nodes[i], nodes[j]
            available = [f for f in features if f != a and f != b]
            subset_candidates: List[List[str]] = [[]] + get_subsets(available, h_order)

            pair_means = []
            for s in subset_candidates:
                s1 = sorted(s + [a]) if a != ORIGIN else sorted(s)
                s2 = sorted(s + [b]) if b != ORIGIN else sorted(s)
                diff = np.abs(subset_value(s1) - subset_value(s2))
                pair_means.append(float(np.mean(diff)) if diff.size else 0.0)

            d_ab = float(np.mean(pair_means)) if pair_means else 0.0
            dist[i, j] = d_ab
            dist[j, i] = d_ab

    return dist, nodes


def _find_optimal_mds_components(
    distance_matrix: np.ndarray,
    max_components: int,
    slope_threshold: float,
    random_state: int,
) -> int:
    """Stress-elbow selection of the MDS embedding dimension (baseline semantics)."""
    n_samples = distance_matrix.shape[0]
    max_components = max(1, min(max_components, n_samples - 1))
    if max_components == 1:
        return 1

    stress_list = []
    for n in range(1, max_components + 1):
        mds_temp = MDS(
            n_components=n,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=1,
            max_iter=20000,
            eps=1e-10,
            normalized_stress="auto",
        )
        mds_temp.fit(distance_matrix)
        stress_list.append(mds_temp.stress_)

    stress_arr = np.asarray(stress_list, dtype=float)
    span = stress_arr.max() - stress_arr.min()
    if span <= 1e-12:
        return 1
    stress_norm = (stress_arr - stress_arr.min()) / span
    optimal = max_components
    for i in range(1, len(stress_norm)):
        if abs(stress_norm[i] - stress_norm[i - 1]) < slope_threshold:
            optimal = i + 1
            break
    return optimal


def compute_bias_concentration(
    X: pd.DataFrame,
    o_series: pd.Series,
    cate_attrs: List[str],
    num_attrs: List[str],
    h_order: int = 1,
    mds_max_components: int = 15,
    mds_slope_threshold: float = 0.01,
    random_state: int = 0,
    num_method: str = "num-a",
    cat_method: str = "cat-a",
    scale: str = "mean",
) -> Dict[str, float]:
    """
    Compute d_phi (Euclidean distance to origin after metric MDS) for each feature.

    Returns ``{feature: d_phi}``. Falls back to all-zero concentrations when the
    embedding cannot be computed (e.g. degenerate zero distance matrix).
    """
    features = list(X.columns)
    if not features:
        return {}

    df_s = compute_pairwise_divergences(
        X, o_series, cate_attrs, num_attrs,
        num_method=num_method, cat_method=cat_method, scale=scale,
    )
    dist, nodes = compute_shapley_distance_matrix(df_s, features, h_order=h_order)

    try:
        if not np.any(dist > 0):
            return {f: 0.0 for f in features}

        optimal_n = _find_optimal_mds_components(
            dist, mds_max_components, mds_slope_threshold, random_state
        )
        mds = MDS(
            n_components=optimal_n,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=4,
            max_iter=10000,
            eps=1e-10,
            normalized_stress="auto",
        )
        pts = mds.fit_transform(dist)
        # Translate so the origin node sits at (0, ..., 0)
        pts = pts - pts[nodes.index(ORIGIN)]
        d_phi = np.sqrt((pts ** 2).sum(axis=1))
        return {node: float(d_phi[idx]) for idx, node in enumerate(nodes) if node != ORIGIN}
    except Exception:
        return {f: 0.0 for f in features}


def compute_dphi_matrix(
    X: pd.DataFrame,
    O: pd.DataFrame,
    cate_attrs: List[str],
    num_attrs: List[str],
    h_order: int = 1,
    mds_max_components: int = 15,
    mds_slope_threshold: float = 0.01,
    random_state: int = 0,
    num_method: str = "num-a",
    cat_method: str = "cat-a",
    scale: str = "mean",
) -> Dict[str, Dict[str, float]]:
    """Compute d_phi for every protected attribute column in O."""
    results: Dict[str, Dict[str, float]] = {}
    X_reset = X.reset_index(drop=True)
    for p_col in O.columns:
        o_series = pd.Series(np.asarray(O[p_col])).reset_index(drop=True)
        if o_series.nunique() < 2:
            results[p_col] = {col: 0.0 for col in X_reset.columns}
            continue
        results[p_col] = compute_bias_concentration(
            X_reset,
            o_series,
            cate_attrs,
            num_attrs,
            h_order=h_order,
            mds_max_components=mds_max_components,
            mds_slope_threshold=mds_slope_threshold,
            random_state=random_state,
            num_method=num_method,
            cat_method=cat_method,
            scale=scale,
        )
    return results
