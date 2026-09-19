"""Shared PSU bootstrap from sufficient totals; no replicate-by-person matrix."""
from __future__ import annotations

import numpy as np

from .survey_inference import RescaledPSUBootstrapEngine


def bootstrap_metric_arrays(y, predictions, A, strata, psus, weights, groups, *,
                            domain_mask, B=2000, seed=20260914):
    """Return paired replicate BA/DP/EO per frozen model (B,3 arrays).

    The ratio denominators depend only on A/Y and the domain, so all numerator
    totals can be aggregated to PSU once. This is algebraically the same
    resampling as the row-level reference, not an approximation/subsample.
    """
    y, A, w = np.asarray(y), np.asarray(A), np.asarray(weights, dtype=float)
    if isinstance(B, bool) or not isinstance(B, (int, np.integer)) or not 2 <= B <= 10000:
        raise ValueError("B must be an integer in [2, 10000] under the evaluation resource budget")
    domain = np.asarray(domain_mask)
    if domain.dtype.kind != 'b' or domain.ndim != 1 or len(domain) != len(y):
        raise ValueError("domain_mask must be an aligned boolean vector")
    if not predictions or not groups or len(set(groups)) != len(groups):
        raise ValueError("predictions and unique expected groups required")
    if y.ndim != 1 or A.ndim != 1 or len(A) != len(y) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("aligned binary y and group vector required")
    if not set(A[domain]).issubset(set(groups)):
        raise ValueError("domain contains an unexpected group")
    engine = RescaledPSUBootstrapEngine(strata, psus, w, seed=seed)
    masks = [domain & (y == 1), domain & (y == 0)]
    masks += [domain & (A == g) for g in groups]
    masks += [domain & (A == g) & (y == 1) for g in groups]
    masks += [domain & (A == g) & (y == 0) for g in groups]
    masks = np.column_stack(masks).astype(float)
    keys = [(h, p) for h, plist in engine.stratum_psu_map.items() for p in plist]
    row_psu = np.empty(len(y), dtype=int)
    for i, key in enumerate(keys):
        row_psu[engine.psu_row_indices[key]] = i
    # First block holds common denominators; remaining blocks numerators.
    k = masks.shape[1]
    allocated = (len(keys) * (len(predictions)+1)*k + B * (len(keys)+(len(predictions)+1)*k)) * 8
    if allocated > 1024**3:
        raise ValueError("PSU aggregate evaluation would exceed its 1 GiB matrix allocation budget")
    totals = np.zeros((len(keys), (len(predictions)+1)*k))
    np.add.at(totals[:, :k], row_psu, masks * w[:, None])
    for index, (name, q) in enumerate(predictions.items()):
        q = np.asarray(q)
        if q.ndim != 1 or len(q) != len(y) or np.iscomplexobj(q) or not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)):
            raise ValueError("invalid frozen decision probability: " + name)
        np.add.at(totals[:, (index+1)*k:(index+2)*k], row_psu, masks * (w*q)[:, None])
    # Same RNG ordering as RescaledPSUBootstrapEngine.iter_replicate_weights.
    rng = np.random.default_rng(seed)
    factors = np.empty((B, len(keys)))
    for b in range(B):
        offset = 0
        for plist in engine.stratum_psu_map.values():
            n = len(plist)
            factors[b, offset:offset+n] = n / (n - 1) * np.bincount(rng.choice(n, size=n-1, replace=True), minlength=n)
            offset += n
    replicate_totals = factors @ totals
    k = masks.shape[1]
    denominators = replicate_totals[:, :k]
    results = {}
    for i, name in enumerate(predictions):
        with np.errstate(divide='ignore', invalid='ignore'):
            rates = replicate_totals[:, (i+1)*k:(i+2)*k] / denominators
        n = len(groups)
        ba = .5 * (rates[:, 0] + 1 - rates[:, 1])
        dp = np.ptp(rates[:, 2:2+n], axis=1)
        eo = np.maximum(np.ptp(rates[:, 2+n:2+2*n], axis=1), np.ptp(rates[:, 2+2*n:], axis=1))
        results[name] = np.column_stack((ba, dp, eo))
    return results
