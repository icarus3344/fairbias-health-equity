"""Paper-level bias concentration metric (d_phi).

Implements the bias concentration pipeline of Tang, Lu & Li (2024),
"Metric-Independent Mitigation of Unpredefined Bias in Machine
Classification", Intell. Comput. 2024;3:Article 0083:

1. Per-feature group-pair divergences ``g_m`` (Eq. 2): numerical
   attributes use the distance between the group centroids after
   min-max normalization (``num-a``); categorical attributes use the
   mean absolute frequency gap across the K categories (``cat-a``).
2. Set separating capability (Eq. 1, alpha = 2 RMS norm)::

       w_max(S) = sqrt( sum_{m in S} g_m^2 / |S| )

   (the previous implementation omitted the ``/ |S|`` normalization).
3. Sub-distance (Eq. 3)::

       d_{Xc}(xa, xb) = | w_max(Xc + {xa}) - w_max(Xc + {xb}) |

   aggregated over level-h exclusion contexts (Eq. 4): at level h,
   ``Xc`` excludes any h attributes besides xa/xb, so
   ``|Xc| = |X| - 2 - h``.  H therefore enumerates *near-full* contexts
   (H = 1 keeps the full remaining set and the sets missing exactly one
   attribute), not small subsets.  The origin distance (Eq. 5) is the
   analogous mean over contexts of ``X \\ {xm}`` excluding up to h
   attributes.  ``h_order >= |available|`` degrades to the full Shapley
   enumeration of all subsets.
4. Metric MDS embedding of the distance matrix with stress-elbow
   dimension selection; the origin node is translated to (0, ..., 0).
5. ``d_phi`` (Eq. 6): Euclidean distance of each attribute's embedding
   coordinates to the origin.

MDS failures are NOT masked: an embedding error raises instead of
returning a fake all-zero (perfectly fair) d_phi vector.

``mds_fixed_components`` (Round 4.1): when not None, the MDS embedding
dimension is FIXED at that value and the stress-elbow selection is
skipped entirely — this is inherited ``official_code_derived_monotone_
cursor_unweighted`` behavior (the official code fixes the embedding
dimension at 2; the paper TEXT instead prescribes elbow-plot dimension
selection, so this fixed-dimension shortcut is official-code-derived
behavior, not a paper-text method reproduction).  None keeps the
automatic elbow selection (engineering mode).

SURVEY-WEIGHTED EXTENSION POINT (pre-declared, Gate D scope): the ONLY
planned deviation for the survey-weighted FairBias variant replaces the
two empirical group statistics feeding Eq. (2) —

    mu_hat_{m,g}      = sum_{i: O_i=g} w_i * X_im / sum_{i: O_i=g} w_i
    p_hat_{m,k,g}     = sum_{i: O_i=g} w_i * 1(X_im = k) / sum_{i: O_i=g} w_i

— inside ``compute_pairwise_divergences`` (numerical group means and
categorical group frequencies).  The distance matrix, MDS embedding,
attribute ranking, and greedy transform search remain UNCHANGED.  With
equal weights (w_i = const) both estimators must degenerate EXACTLY to
the unweighted statistics implemented here; that degeneracy is a
mandatory unit test of the extension (see the round-4.1 report).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.manifold import MDS

ORIGIN = "origin"

# Pre-declared survey-weighted extension contract (see module docstring).
# Pinned as a module constant so the declaration itself is testable:
# any weighted implementation must (a) only replace the group statistics
# below, and (b) degenerate exactly to the unweighted statistics at equal
# weights.
SURVEY_WEIGHTED_EXTENSION_DECLARATION = (
    "Survey-weighted extension (Gate D, pre-declared): replace ONLY the "
    "empirical group statistics in compute_pairwise_divergences with "
    "weighted versions mu_hat_mg = sum(w_i * X_im)/sum(w_i) over i:O_i=g "
    "and p_hat_mkg = sum(w_i * 1(X_im=k))/sum(w_i) over i:O_i=g; keep the "
    "distance matrix, MDS, ranking, and greedy search unchanged; equal "
    "weights must degenerate EXACTLY to the unweighted statistics "
    "(mandatory unit test)."
)


def w_max(g_values: Sequence[float]) -> float:
    """
    Eq. (1) with alpha = 2: the RMS norm of the individual contributions.

    ``w_max(S) = sqrt(sum_{m in S} g_m^2 / |S|)``; ``w_max(empty) = 0``.
    """
    arr = np.asarray(list(g_values), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.sum(arr ** 2) / arr.size))


def get_subsets(features: Sequence[str], h_order: int = 1) -> List[List[str]]:
    """
    Level-h exclusion contexts (Eq. 4/5): every subset obtained from
    ``features`` by excluding at most ``h_order`` attributes.

    Level 0 keeps the full set, level 1 removes exactly one attribute,
    and so on, so ``H = 1`` produces near-full contexts.  When
    ``h_order >= len(features)`` this degenerates to the complete Shapley
    enumeration of all subsets (including the empty set).
    """
    feats = list(features)
    n = len(feats)
    h = max(0, min(int(h_order), n))
    subsets: List[List[str]] = []
    for k in range(n - h, n + 1):
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
) -> pd.DataFrame:
    """
    Compute per-feature divergence ``g_m`` (Eq. 2) between every pair of
    protected groups.

    Returns a DataFrame with features as index and one column per group pair
    (keyed ``"{p}_{n}"``):

    - ``cat-a`` (Eq. 2, categorical): ``(1/K) * sum |N_k^p/N^p - N_k^n/N^n|``
      over the union of categories (K categories, group-wise frequencies).
    - ``num-a`` (Eq. 2, numerical): absolute difference of the group means
      after within-pair min-max normalization of the feature.

    No additional family scaling is applied: the paper applies Eq. (1)
    directly to the raw ``g_m`` values.

    SURVEY-WEIGHTED EXTENSION POINT: the group means (``p_vals.mean()`` /
    ``n_vals.mean()``) and the group frequencies (``p_counts / p_counts.sum()``
    etc.) are exactly the two statistics the pre-declared weighted
    extension replaces (see ``SURVEY_WEIGHTED_EXTENSION_DECLARATION``).
    """
    o_col = "_prot_"
    df = pd.concat(
        [X.reset_index(drop=True), o_series.reset_index(drop=True).rename(o_col)],
        axis=1,
    )
    groups = sorted(df[o_col].dropna().unique())

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

        columns[f"{p}_{n}"] = pd.concat([
            pd.Series(num_diff, dtype="float64"),
            pd.Series(cat_diff, dtype="float64"),
        ])

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
    Build the paper distance matrix (Eq. 3-5) over features plus the origin.

    ``dist(a, b) = mean over level-h exclusion contexts S of
    |w_max(S + a) - w_max(S + b)|`` where S ranges over subsets of the
    remaining features obtained by excluding at most ``h_order``
    attributes (level 0..H exclusion, Eq. 4).  For the origin node the
    contexts range over subsets of ``features \\ {m}`` (Eq. 5) and the
    sub-distance is ``|w_max(S + m) - w_max(S)|``.  ``w_max`` is the RMS
    norm of Eq. (1).  With multiple protected groups the per-pair
    sub-distances are averaged (extension of the binary-o paper setting).
    """
    nodes = list(features) + [ORIGIN]
    n_nodes = len(nodes)
    dist = np.zeros((n_nodes, n_nodes))

    if df_s.empty or df_s.shape[1] == 0:
        return dist, nodes

    df_s_arr = df_s.to_numpy(dtype=float)
    if not np.isfinite(df_s_arr).all():
        # NaN divergences would be silently skipped by downstream sums and
        # could masquerade as "no bias"; fail loudly instead.
        raise ValueError(
            "Pairwise divergence table contains NaN/Inf entries; refusing to "
            "build the bias distance matrix"
        )

    df_s_sq = df_s ** 2
    value_cache: Dict[Tuple[str, ...], np.ndarray] = {}
    zero_vec = np.zeros(df_s.shape[1])

    def subset_value(subset: Sequence[str]) -> np.ndarray:
        """Eq. (1) applied per group-pair column: sqrt(sum g^2 / |S|)."""
        key = tuple(sorted(subset))
        if not key:
            return zero_vec
        cached = value_cache.get(key)
        if cached is None:
            rows = [c for c in key if c in df_s_sq.index]
            if rows:
                cached = np.sqrt(
                    df_s_sq.loc[rows].sum(axis=0).to_numpy(dtype=float) / len(rows)
                )
            else:
                cached = zero_vec
            value_cache[key] = cached
        return cached

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            a, b = nodes[i], nodes[j]
            if a != ORIGIN and b != ORIGIN:
                available = [f for f in features if f != a and f != b]
            else:
                m = a if b == ORIGIN else b
                available = [f for f in features if f != m]

            # Level 0..H exclusion contexts (near-full companion sets)
            contexts = get_subsets(available, h_order)

            pair_means = []
            for s in contexts:
                if b == ORIGIN or a == ORIGIN:
                    m = a if b == ORIGIN else b
                    s1 = list(s) + [m]
                    s2 = list(s)
                else:
                    s1 = list(s) + [a]
                    s2 = list(s) + [b]
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
    mds_fixed_components: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute d_phi (Eq. 6: Euclidean distance to the origin after metric MDS)
    for each feature.

    Returns ``{feature: d_phi}``.  A fully degenerate (all-zero) distance
    matrix legitimately means no measurable bias and returns all zeros; any
    MDS or input failure raises ``RuntimeError`` instead of being masked as
    a zero-bias result.

    ``mds_fixed_components``: when not None, fix the embedding dimension
    and skip the stress-elbow selection (official mode behavior).
    """
    features = list(X.columns)
    if not features:
        return {}

    df_s = compute_pairwise_divergences(
        X, o_series, cate_attrs, num_attrs,
        num_method=num_method, cat_method=cat_method,
    )
    dist, nodes = compute_shapley_distance_matrix(df_s, features, h_order=h_order)

    if np.isnan(dist).any() or np.isinf(dist).any():
        # NaN/Inf entries compare False against 0, so this check must run
        # BEFORE the all-zero early return below (a NaN matrix would
        # otherwise masquerade as "no bias").
        raise ValueError(
            "Bias distance matrix contains NaN/Inf entries; refusing to "
            f"compute d_phi (features={len(features)})"
        )
    if not np.any(dist > 0):
        # All group-pair divergences are exactly zero: no measurable bias.
        return {f: 0.0 for f in features}

    try:
        if mds_fixed_components is not None:
            # Official mode: fixed embedding dimension, NO elbow search.
            optimal_n = int(mds_fixed_components)
        else:
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
    except Exception as exc:
        raise RuntimeError(
            "MDS embedding of the bias-distance matrix failed; refusing to "
            "fabricate a zero-bias d_phi vector "
            f"(features={len(features)}, nodes={dist.shape[0]}): {exc!r}"
        ) from exc

    # Translate so the origin node sits at (0, ..., 0)
    pts = pts - pts[nodes.index(ORIGIN)]
    d_phi = np.sqrt((pts ** 2).sum(axis=1))
    return {node: float(d_phi[idx]) for idx, node in enumerate(nodes) if node != ORIGIN}


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
    mds_fixed_components: Optional[int] = None,
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
            mds_fixed_components=mds_fixed_components,
        )
    return results
