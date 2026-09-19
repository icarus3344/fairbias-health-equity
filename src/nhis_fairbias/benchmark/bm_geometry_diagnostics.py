"""Opt-in aggregate diagnostics for FairBias BM geometry.

The diagnostic deliberately stops at F geometry.  An optional C frame is
accepted only for schema/row-count comparison; no C fit, model, S, or T
operation is performed.  It reports counts and exception classes, never rows
or feature values.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fairbias.bias_metric import compute_bias_concentration, compute_pairwise_divergences
from fairbias.evaluator import FairEvaluator


def _partition_summary(X: pd.DataFrame, A: pd.Series | np.ndarray) -> dict[str, Any]:
    a = np.asarray(A)
    groups, counts = np.unique(a[~pd.isna(a)], return_counts=True)
    return {
        "rows": int(len(X)),
        "columns": int(X.shape[1]),
        "column_names": list(map(str, X.columns)),
        "group_count": int(len(groups)),
        "group_counts": {str(g): int(n) for g, n in zip(groups.tolist(), counts.tolist())},
    }


def _feature_validity(X: pd.DataFrame, A: pd.Series | np.ndarray,
                      cate_attrs: Sequence[str], num_attrs: Sequence[str]) -> list[dict[str, Any]]:
    a = np.asarray(A)
    rows = []
    for feature in X.columns:
        kind = "categorical" if feature in cate_attrs else "numeric" if feature in num_attrs else "undeclared"
        values = pd.to_numeric(X[feature], errors="coerce") if kind == "numeric" else X[feature]
        by_group = {}
        for group in sorted(pd.Series(a).dropna().unique().tolist()):
            mask = a == group
            valid = values.notna().to_numpy() & mask
            by_group[str(group)] = {
                "rows": int(mask.sum()),
                "valid": int(valid.sum()),
                "missing": int(mask.sum() - valid.sum()),
                "category_count": int(pd.Series(values[mask][values[mask].notna()]).nunique()) if kind == "categorical" else None,
            }
        rows.append({"feature": str(feature), "kind": kind, "by_group": by_group})
    return rows


def diagnose_bm_geometry(
    X_F: pd.DataFrame,
    A_F: pd.Series | np.ndarray,
    *,
    cate_attrs: Sequence[str],
    num_attrs: Sequence[str],
    X_C: pd.DataFrame | None = None,
    A_C: pd.Series | np.ndarray | None = None,
    h_order: int = 1,
    mds_fixed_components: int | None = 2,
    random_state: int = 0,
    multigroup_aggregation: str = "author_max_pair",
) -> dict[str, Any]:
    """Return aggregate F/C diagnostics and a fail-closed geometry status."""
    if not isinstance(X_F, pd.DataFrame):
        raise TypeError("X_F must be a semantic DataFrame")
    if len(X_F) != len(A_F):
        raise ValueError("F feature/protected lengths differ")
    result: dict[str, Any] = {
        "source_partition": "F",
        "fit_operation": "geometry_only",
        "partitions": {"F": _partition_summary(X_F, A_F)},
        "feature_validity": _feature_validity(X_F, A_F, cate_attrs, num_attrs),
        "status": "STARTED",
    }
    if X_C is not None or A_C is not None:
        if not isinstance(X_C, pd.DataFrame) or A_C is None or len(X_C) != len(A_C):
            raise ValueError("C diagnostic requires a DataFrame and aligned protected values")
        result["partitions"]["C"] = _partition_summary(X_C, A_C)
        result["C_schema_match_F"] = list(X_C.columns) == list(X_F.columns)
    try:
        table = compute_pairwise_divergences(X_F, pd.Series(np.asarray(A_F), index=X_F.index),
                                             list(cate_attrs), list(num_attrs))
        result["pairwise"] = {
            "rows": int(table.shape[0]), "columns": int(table.shape[1]),
            "finite": bool(np.isfinite(table.to_numpy(dtype=float)).all()),
            "all_zero": bool(np.all(table.to_numpy(dtype=float) == 0)),
        }
    except Exception as exc:
        result.update(status="PAIRWISE_FAILURE", failure_class=_classify(str(exc)), error=str(exc))
        return result
    try:
        dphi = compute_bias_concentration(X_F, pd.Series(np.asarray(A_F), index=X_F.index),
                                          list(cate_attrs), list(num_attrs), h_order=h_order,
                                          random_state=random_state,
                                          mds_fixed_components=mds_fixed_components,
                                          multigroup_aggregation=multigroup_aggregation)
        values = np.asarray(list(dphi.values()), dtype=float)
        result.update(status="VALID", dphi_count=int(values.size), dphi_finite=bool(np.isfinite(values).all()),
                      dphi_all_zero=bool(np.all(values == 0)))
    except Exception as exc:
        result.update(status="GEOMETRY_FAILURE", failure_class=_classify(str(exc)), error=str(exc))
    return result


def _classify(message: str) -> str:
    lower = message.lower()
    if "budget" in lower or "iteration_cap" in lower or "max_geometry" in lower:
        return "BUDGET_OR_CONVERGENCE_UNVERIFIED"
    if "non-finite" in lower or "nan" in lower or "inf" in lower:
        return "NONFINITE_GEOMETRY"
    if "empty" in lower or "no valid" in lower or "no pair" in lower or "fewer than 2" in lower:
        return "EMPTY_OR_UNESTIMABLE_GEOMETRY"
    if "mds" in lower or "iteration_cap" in lower:
        return "MDS_FAILURE_OR_BUDGET"
    return "GEOMETRY_INPUT_OR_CONTRACT_FAILURE"


def replay_fairbias_fit(
    X_F: pd.DataFrame,
    A_F: pd.Series | np.ndarray,
    *,
    y_F: Sequence[int],
    config: Mapping[str, Any],
    seed: int,
    X_C: pd.DataFrame | None = None,
    A_C: pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    """Replay one FairBias BM representation fit and receipt every geometry call.

    ``config`` is the registered job config.  The adapter is created through
    the normal registry, while ``FairEvaluator.calculate_epsilon`` is wrapped
    only for this call.  Returned geometry tables are summarized by shape,
    finite count, min, max and key count; no per-feature values are retained.
    """
    if config.get("method") != "FAIRBIAS_BM":
        raise ValueError("replay_fairbias_fit supports FAIRBIAS_BM jobs only")
    from .experiment_registry import make_adapter
    from fairbias.transform import FairTransform
    adapter = make_adapter(dict(config), int(seed))
    calls: list[dict[str, Any]] = []
    transform_events: list[dict[str, Any]] = []
    original = FairEvaluator.calculate_epsilon
    original_transform = FairTransform.transform_data

    def wrapped(self, X, O, *args, **kwargs):
        record: dict[str, Any] = {"rows": int(len(X)), "columns": int(X.shape[1]),
                                  "column_names": [str(c) for c in X.columns]}
        try:
            table = original(self, X, O, *args, **kwargs)
            values = np.asarray([v for group in table.values() for v in group.values()], dtype=float)
            finite_values = values[np.isfinite(values)]
            record.update({"status": "RETURNED", "groups": int(len(table)), "keys": int(values.size),
                           "finite": bool(np.isfinite(values).all()) if values.size else False,
                           "nonfinite_count": int(np.count_nonzero(~np.isfinite(values))),
                           "negative_count": int(np.count_nonzero(values < 0)),
                           "min": float(finite_values.min()) if finite_values.size else None,
                           "max": float(finite_values.max()) if finite_values.size else None})
            return table
        except Exception as exc:
            record.update({"status": "RAISED", "failure_class": _classify(str(exc)),
                           "error": str(exc)})
            raise
        finally:
            record["call_index"] = len(calls) + 1
            calls.append(record)

    def wrapped_transform(self, X, changed_dict, num_attrs, cate_attrs):
        event: dict[str, Any] = {
            "input_columns": [str(c) for c in X.columns],
            "changed_keys": sorted(str(k) for k in changed_dict),
            "changed_count": int(len(changed_dict)),
        }
        try:
            out = original_transform(self, X, changed_dict, num_attrs, cate_attrs)
            event.update({"status": "RETURNED", "output_columns": [str(c) for c in out.columns],
                          "output_count": int(out.shape[1])})
            return out
        except Exception as exc:
            event.update({"status": "RAISED", "failure_class": _classify(str(exc)), "error": str(exc)})
            raise
        finally:
            transform_events.append(event)

    result: dict[str, Any] = {
        "candidate_id": config.get("candidate_id"), "seed": int(seed),
        "source_partition": "F", "fit_operation": "representation_only",
        "F": _partition_summary(X_F, A_F),
        "geometry_calls": calls,
        "transform_events": transform_events,
    }
    if X_C is not None and A_C is not None:
        result["C"] = _partition_summary(X_C, A_C)
    try:
        FairEvaluator.calculate_epsilon = wrapped
        FairTransform.transform_data = wrapped_transform
        adapter.fit_representation(X_F, np.asarray(y_F, dtype=int), np.asarray(A_F))
        result.update(status="FIT_REPRESENTATION_RETURNED", converged=bool(getattr(adapter, "converged_", False)))
    except Exception as exc:
        result.update(status="FIT_REPRESENTATION_RAISED", failure_class=_classify(str(exc)), error=str(exc))
    finally:
        FairEvaluator.calculate_epsilon = original
        FairTransform.transform_data = original_transform
    result["geometry_call_count"] = len(calls)
    result["transform_event_count"] = len(transform_events)
    if (result["status"] == "FIT_REPRESENTATION_RAISED" and calls
            and calls[-1].get("columns") == 0 and calls[-1].get("keys") == 0):
        # This proves exhaustion on this search path, not global infeasibility
        # of every possible representation under the requested constraint.
        result["terminal_diagnosis"] = "SEARCH_FEATURE_EXHAUSTED"
    result["fit_manifest_status"] = getattr(adapter, "fit_manifest_", {}).get("status")
    return result
