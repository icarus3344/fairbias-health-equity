"""Standardized application-level group fairness disparity metrics."""

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence
import numpy as np


def _validate_binary_hard_labels(arr: Any, name: str) -> np.ndarray:
    """Ensure array consists strictly of 1D binary hard labels {0, 1}."""
    a = np.asarray(arr)
    if a.ndim != 1:
        raise ValueError(f"{name} must be strictly 1-dimensional, got {a.ndim}D shape {a.shape}")
    if len(a) == 0:
        raise ValueError(f"{name} must not be empty")

    # Reject float/decimal predictions before integer casting
    if np.issubdtype(a.dtype, np.floating):
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} contains NaN or non-finite values")
        if not np.all(np.isin(a, [0.0, 1.0])):
            raise ValueError(
                f"{name} must contain strictly binary hard labels {{0, 1}}; fractional probabilities or non-binary values rejected (shape {a.shape})"
            )
        return a.astype(int)

    if not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.bool_)):
        raise ValueError(f"{name} must have integer or boolean dtype, got {a.dtype}")

    int_a = a.astype(int)
    if not np.all(np.isin(int_a, [0, 1])):
        raise ValueError(f"{name} contains out-of-range labels outside {{0, 1}} (shape {int_a.shape})")

    return int_a


def _validate_group_scalar(val: Any, context: str) -> Any:
    """Validate that group identifier is a supported finite scalar value."""
    if val is None:
        raise ValueError(f"{context} contains None")
    # Reject non-scalar types: containers, sets, frozensets, dicts, lists, tuples, custom classes
    if isinstance(val, (set, frozenset, list, dict, tuple, Sequence)) and not isinstance(val, (str, bytes)):
        raise TypeError(f"{context} contains non-scalar type {type(val).__name__}: groups must be finite scalar values")

    # Check finiteness first for any type with is_finite (e.g. Decimal Infinity / NaN)
    if hasattr(val, "is_finite") and callable(val.is_finite):
        try:
            is_fin = val.is_finite()
        except Exception:
            is_fin = True
        if not is_fin:
            raise ValueError(f"{context} contains non-finite value {val}")

    # Reject Decimal explicitly to prevent underflow / precision loss / lossy serialization (R6-05)
    from decimal import Decimal
    if isinstance(val, Decimal) or (hasattr(val, "is_finite") and callable(val.is_finite) and not isinstance(val, (float, np.floating))):
        raise TypeError(
            f"{context} contains Decimal or non-standard numeric type {type(val).__name__}: "
            "Decimal groups are rejected to ensure lossless JSON serialization and prevent representation underflow/collision."
        )

    # Normalize numpy scalars to standard Python scalars (R7-06)
    # Check bool first because in Python issubclass(bool, int) is True
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    elif isinstance(val, (int, np.integer)):
        return int(val)
    elif isinstance(val, (float, np.floating)):
        val_f = float(val)
        if not np.isfinite(val_f):
            raise ValueError(f"{context} contains non-finite float value {val}")
        return val_f
    elif isinstance(val, str):
        if val.strip().lower() in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
            raise ValueError(f"{context} contains non-finite string representation {val!r}")
        return str(val)
    else:
        raise TypeError(f"{context} contains unsupported non-scalar type {type(val).__name__}: groups must be finite scalar values")


def _json_group_key(g: Any) -> str:
    """Encode group identifier to its exact JSON object key representation (R7-06, R8-01)."""
    if isinstance(g, bool):
        return "true" if g else "false"
    if isinstance(g, str):
        return g
    return str(g)


def compute_application_group_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected_vals: np.ndarray,
    expected_groups: Optional[Sequence[Any]] = None,
    expected_groups_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute application-level demographic parity, equal opportunity, and equalized odds differences.

    Definitions:
    - Application Demographic Parity Difference (max-min):
      max_g P(y_pred=1 | g) - min_g P(y_pred=1 | g)
    - Application Equal Opportunity Difference (max-min TPR gap):
      max_g TPR_g - min_g TPR_g
    - Application Equalized Odds Difference (primary benchmark estimand):
      max(TPR_gap, FPR_gap)
    - Legacy Pairwise Disparities:
      mean over (g_a, g_b) pairs of |rate_a - rate_b|
    """
    # 1. First validate expected_groups before any early return (R2-08 / R3-01 / R4-07)
    if expected_groups is not None:
        if isinstance(expected_groups, (str, bytes)):
            raise TypeError(f"expected_groups must be a non-string sequence, got {type(expected_groups).__name__}")
        if not isinstance(expected_groups, (list, tuple, np.ndarray, Sequence, set)):
            raise TypeError(f"expected_groups must be a sequence, got {type(expected_groups).__name__}")
        try:
            target_groups = list(expected_groups)
        except Exception as e:
            raise TypeError(f"expected_groups could not be converted to list: {e}")
        if len(target_groups) == 0:
            raise ValueError("expected_groups must not be empty when provided")
        target_groups = [_validate_group_scalar(g, "expected_groups") for g in target_groups]
        if len(target_groups) != len(set(target_groups)):
            raise ValueError(f"Duplicate expected groups provided: {expected_groups}")
        # Verify no JSON string collision among groups (R7-06)
        str_keys = [_json_group_key(g) for g in target_groups]
        if len(str_keys) != len(set(str_keys)):
            raise ValueError(f"Group identifier collision under JSON string conversion: {target_groups}")
        try:
            target_groups = sorted(target_groups)
        except TypeError:
            target_groups = sorted(target_groups, key=str)
        primary_declared = True
    else:
        target_groups = None
        primary_declared = False

    # 2. Validate labels and protected attribute values
    y_t = _validate_binary_hard_labels(y_true, "y_true")
    y_p = _validate_binary_hard_labels(y_pred, "y_pred")
    prot_arr = np.asarray(protected_vals)
    if prot_arr.ndim != 1:
        raise ValueError(f"protected_vals must be strictly 1-dimensional, got {prot_arr.ndim}D shape {prot_arr.shape}")
    if len(prot_arr) == 0:
        raise ValueError("protected_vals must not be empty")

    prot = np.array([_validate_group_scalar(v, "protected_vals") for v in prot_arr.ravel()], dtype=object)
    if len(y_t) != len(y_p) or len(y_t) != len(prot):
        raise ValueError(
            f"Length mismatch: y_true={len(y_t)}, y_pred={len(y_p)}, protected_vals={len(prot)}"
        )

    try:
        unique_in_data = sorted(set(prot.tolist()))
    except TypeError:
        unique_in_data = sorted(set(prot.tolist()), key=str)

    # Verify no string collision among observed unique groups (e.g. 1 and '1', True and 'true') (R7-06)
    unique_str_keys = [_json_group_key(g) for g in unique_in_data]
    if len(unique_str_keys) != len(set(unique_str_keys)):
        raise ValueError(f"Group identifier collision under JSON string conversion: {unique_in_data}")

    if target_groups is not None:
        unexpected_groups = [g for g in unique_in_data if g not in target_groups]
        if unexpected_groups:
            raise ValueError(
                f"Observed unexpected groups not declared in expected_groups: {unexpected_groups}"
            )
    else:
        target_groups = unique_in_data

    if len(unique_in_data) < 2 or len(target_groups) < 2:
        return {
            "demographic_parity_difference": None,
            "equal_opportunity_difference": None,
            "equalized_odds_gap": None,
            "dp_estimable": False,
            "eo_estimable": False,
            "equalized_odds_estimable": False,
            "dp_status": "SINGLE_GROUP",
            "eo_status": "SINGLE_GROUP",
            "equalized_odds_status": "SINGLE_GROUP",
            "group_selection_rates": {},
            "group_tprs": {},
            "group_fprs": {},
            "is_primary_estimand": False,
            "expected_groups": list(target_groups) if primary_declared else None,
            "expected_groups_source": expected_groups_source,
            "primary_estimand_declared": primary_declared,
            "primary_result_eligible": False,
            "primary_eligibility_reason": "SINGLE_GROUP",
            "legacy_demographic_parity_mean_pair": None,
            "legacy_equal_opportunity_mean_pair": None,
        }

    # Check for missing expected groups
    missing_expected = [g for g in target_groups if g not in unique_in_data]
    if missing_expected:
        reason = f"MISSING_EXPECTED_GROUPS: {missing_expected}"
        return {
            "demographic_parity_difference": None,
            "equal_opportunity_difference": None,
            "equalized_odds_gap": None,
            "dp_estimable": False,
            "eo_estimable": False,
            "equalized_odds_estimable": False,
            "dp_status": reason,
            "eo_status": reason,
            "equalized_odds_status": reason,
            "group_selection_rates": {},
            "group_tprs": {},
            "group_fprs": {},
            "is_primary_estimand": False,
            "expected_groups": list(target_groups),
            "expected_groups_source": expected_groups_source,
            "primary_estimand_declared": True,
            "primary_result_eligible": False,
            "primary_eligibility_reason": reason,
            "legacy_demographic_parity_mean_pair": None,
            "legacy_equal_opportunity_mean_pair": None,
        }

    selection_rates: Dict[Any, float] = {}
    tprs: Dict[Any, Optional[float]] = {}
    fprs: Dict[Any, Optional[float]] = {}

    dp_possible = True
    eo_possible = True
    fpr_possible = True

    dp_unestimable_reason = None
    eo_unestimable_reason = None
    fpr_unestimable_reason = None

    for g in target_groups:
        mask_g = (prot == g)
        n_g = int(np.sum(mask_g))
        if n_g == 0:
            dp_possible = False
            eo_possible = False
            fpr_possible = False
            dp_unestimable_reason = f"EMPTY_GROUP: {g}"
            eo_unestimable_reason = f"EMPTY_GROUP: {g}"
            fpr_unestimable_reason = f"EMPTY_GROUP: {g}"
            break

        rate_g = float(np.mean(y_p[mask_g]))
        selection_rates[g] = rate_g

        pos_mask = mask_g & (y_t == 1)
        n_pos = int(np.sum(pos_mask))
        if n_pos > 0:
            tprs[g] = float(np.mean(y_p[pos_mask]))
        else:
            tprs[g] = None
            eo_possible = False
            if eo_unestimable_reason is None:
                eo_unestimable_reason = f"MISSING_POSITIVE_SAMPLES: group {g} has 0 positives"

        neg_mask = mask_g & (y_t == 0)
        n_neg = int(np.sum(neg_mask))
        if n_neg > 0:
            fprs[g] = float(np.mean(y_p[neg_mask]))
        else:
            fprs[g] = None
            fpr_possible = False
            if fpr_unestimable_reason is None:
                fpr_unestimable_reason = f"MISSING_NEGATIVE_SAMPLES: group {g} has 0 negatives"

    # Compute Demographic Parity Difference
    if dp_possible and len(selection_rates) >= 2:
        rates_list = list(selection_rates.values())
        dp_diff = float(max(rates_list) - min(rates_list))
        dp_status = "VALID"
        dp_pairs = [abs(rates_list[i] - rates_list[j]) for i, j in combinations(range(len(rates_list)), 2)]
        legacy_dp_mean = float(np.mean(dp_pairs)) if dp_pairs else 0.0
    else:
        dp_diff = None
        dp_status = dp_unestimable_reason or "UNESTIMABLE"
        legacy_dp_mean = None

    # Compute Equal Opportunity Difference (TPR gap)
    if eo_possible and len(tprs) >= 2 and all(v is not None for v in tprs.values()):
        tpr_list = [float(v) for v in tprs.values() if v is not None]
        eo_diff = float(max(tpr_list) - min(tpr_list))
        eo_status = "VALID"
        eo_pairs = [abs(tpr_list[i] - tpr_list[j]) for i, j in combinations(range(len(tpr_list)), 2)]
        legacy_eo_mean = float(np.mean(eo_pairs)) if eo_pairs else 0.0
    else:
        eo_diff = None
        eo_status = eo_unestimable_reason or "UNESTIMABLE"
        legacy_eo_mean = None

    # Compute FPR difference
    if fpr_possible and len(fprs) >= 2 and all(v is not None for v in fprs.values()):
        fpr_list = [float(v) for v in fprs.values() if v is not None]
        fpr_diff = float(max(fpr_list) - min(fpr_list))
    else:
        fpr_diff = None

    # Compute Equalized Odds Difference: max(TPR_gap, FPR_gap)
    if eo_diff is not None and fpr_diff is not None:
        eq_odds_diff = float(max(eo_diff, fpr_diff))
        eq_odds_status = "VALID"
    else:
        eq_odds_diff = None
        eq_odds_status = eo_unestimable_reason or fpr_unestimable_reason or "UNESTIMABLE"

    is_primary_eligible = (primary_declared and eq_odds_diff is not None)
    if is_primary_eligible:
        eligibility_reason = None
    elif primary_declared:
        eligibility_reason = eq_odds_status
    else:
        eligibility_reason = "NO_EXPECTED_GROUPS_DECLARED"

    return {
        "demographic_parity_difference": dp_diff,
        "equal_opportunity_difference": eo_diff,
        "equalized_odds_gap": eq_odds_diff,
        "dp_estimable": (dp_diff is not None),
        "eo_estimable": (eo_diff is not None),
        "equalized_odds_estimable": (eq_odds_diff is not None),
        "dp_status": dp_status,
        "eo_status": eo_status,
        "equalized_odds_status": eq_odds_status,
        "group_selection_rates": selection_rates,
        "group_tprs": tprs,
        "group_fprs": fprs,
        "is_primary_estimand": is_primary_eligible,
        "expected_groups": list(target_groups) if primary_declared else None,
        "expected_groups_source": expected_groups_source,
        "primary_estimand_declared": primary_declared,
        "primary_result_eligible": is_primary_eligible,
        "primary_eligibility_reason": eligibility_reason,
        "legacy_demographic_parity_mean_pair": legacy_dp_mean,
        "legacy_equal_opportunity_mean_pair": legacy_eo_mean,
    }
