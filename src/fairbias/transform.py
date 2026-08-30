"""Leakage-free feature transformation engine with chained category composition and polynomial power transforms.

Drop semantics: the sentinel ``"dropped"`` records an explicit, auditable
attribute exclusion (paper: merging the two categories of a binary
attribute, or numerical overflow beyond numpy.float32, is equivalent to
dropping the attribute).  A raw category mapping that merely collapses a
feature to a constant is still rejected by ``check_transform_validity``;
collapse must be represented through the recorded ``"dropped"`` state.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

# Paper overflow bound: transformed values beyond numpy.float32 are set
# uniformly to 1, i.e. the attribute is dropped.
FLOAT32_MAX = float(np.finfo(np.float32).max)


def calculate_nmi_dict(X: pd.DataFrame, Y: pd.Series) -> Dict[str, float]:
    """Calculate Normalized Mutual Information (NMI) between each feature in X and target Y."""
    nmi_dict: Dict[str, float] = {}
    y_arr = np.asarray(Y, dtype=int)
    for col in X.columns:
        x_col = X[col]
        # Discretize continuous feature into bins if float or high cardinality
        if pd.api.types.is_float_dtype(x_col) or (pd.api.types.is_numeric_dtype(x_col) and x_col.nunique() > 10):
            try:
                x_binned = pd.qcut(x_col, q=min(10, max(2, x_col.nunique())), labels=False, duplicates="drop")
            except Exception:
                x_binned = pd.cut(x_col, bins=min(10, max(2, x_col.nunique())), labels=False)
            score = normalized_mutual_info_score(np.asarray(x_binned).astype(int), y_arr)
        else:
            score = normalized_mutual_info_score(np.asarray(x_col).astype(str), y_arr)
        # Avoid exact zero or negative scores for numerical stability
        nmi_dict[col] = max(float(score), 1e-6)
    return nmi_dict


def compose_category_mapping(existing: Dict[Any, Any], new_change: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Compose an existing category mapping with a new merge so that chains stay consistent.

    Semantics: the existing mapping is applied first, then the new merge is applied
    to the result. This gives transitive closure behavior, e.g.
    ``{3: 1}`` composed with ``{1: 0}`` yields ``{3: 0, 1: 0}`` so previously merged
    categories follow subsequent merges instead of being stranded at stale codes.
    """
    composed: Dict[Any, Any] = {}
    for key, value in existing.items():
        composed[key] = new_change.get(value, value)
    for key, value in new_change.items():
        if key not in composed:
            composed[key] = value
    return composed


def apply_power_transform(s: pd.Series, power: float) -> pd.Series:
    """
    Apply the sign-preserving polynomial power transform ``sign(x) * |x|^p``
    (baseline ``poly`` semantics from module_transform._cal_beta_value).
    """
    s_float = pd.to_numeric(s, errors="coerce").astype(float)
    return np.sign(s_float) * (np.abs(s_float) ** float(power))


def power_transform_overflows(s: pd.Series, power: float) -> bool:
    """
    Paper overflow rule: if the maximum absolute transformed value exceeds
    numpy.float32 (≈3.4e38), the attribute's records are set uniformly to 1,
    which is equivalent to dropping the attribute.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = apply_power_transform(s, power).abs()
    arr = transformed.to_numpy(dtype=float)
    return bool(np.any(np.isinf(arr)) or np.any(arr > FLOAT32_MAX))


class FairTransform:
    """Transformation engine supporting simultaneous categorical rebinning and numerical scaling."""

    def __init__(
        self,
        n_bins: int = 10,
        log_epsilon: float = 1e-5,
        x_max: float = 1e9,
    ):
        self.n_bins = n_bins
        self.log_epsilon = log_epsilon
        self.x_max = x_max

    def check_transform_validity(
        self,
        df: pd.DataFrame,
        attr: str,
        change: Union[Dict[Any, Any], str],
        num_attrs: Optional[List[str]] = None,
        cate_attrs: Optional[List[str]] = None,
    ) -> bool:
        """
        Validate that a candidate transform does not collapse feature variance or introduce NaNs.

        ``change == "dropped"`` is always valid: it is the explicit, recorded
        exclusion state.  A category ``dict`` whose mapping collapses the
        feature to a constant is rejected here; such a collapse must instead
        be requested explicitly as the recorded ``"dropped"`` state.
        """
        if attr not in df.columns:
            return False

        if change == "dropped":
            return True

        num_attrs = num_attrs or []
        cate_attrs = cate_attrs or []

        s = df[attr].copy()
        initial_nunique = s.nunique()

        if attr in cate_attrs or not pd.api.types.is_numeric_dtype(s):
            if isinstance(change, dict):
                # Convert change keys/values to appropriate types
                # Perform simultaneous mapping
                mapped = s.map(lambda v: change.get(v, change.get(str(v), v)))
                
                # Check for NaNs
                if mapped.isna().any():
                    return False
                
                # Anti-collapse check: If feature had >= 2 unique values, it must not become constant (nunique == 1)
                if initial_nunique >= 2 and mapped.nunique() <= 1:
                    return False
                
                return True
            return False

        elif attr in num_attrs or pd.api.types.is_numeric_dtype(s):
            if isinstance(change, dict):
                power = float(change.get("power", 1.0))

                # Test numerical transformation (sign-preserving polynomial power)
                s_float = pd.to_numeric(s, errors="coerce").astype(float)
                transformed = np.sign(s_float) * (np.abs(s_float) ** power)

                if transformed.isna().any() or np.isinf(transformed).any():
                    return False
                if (transformed.abs() > self.x_max).any():
                    return False
                return True
            return False

        return False

    def transform_series(
        self,
        s: pd.Series,
        attr: str,
        change: Union[Dict[Any, Any], str],
        is_categorical: bool,
    ) -> Optional[pd.Series]:
        """Apply a validated transformation to a single Series."""
        if change == "dropped":
            return None

        if is_categorical or not pd.api.types.is_numeric_dtype(s):
            if isinstance(change, dict):
                # Simultaneous category mapping
                # Map using exact key or string representation of key
                def map_val(val: Any) -> Any:
                    if val in change:
                        return change[val]
                    s_val = str(val)
                    if s_val in change:
                        return change[s_val]
                    # Also check int conversions
                    try:
                        i_val = int(val)
                        if i_val in change:
                            return change[i_val]
                    except (ValueError, TypeError):
                        pass
                    return val

                return s.map(map_val)
            return s.copy()

        else:
            # Numerical feature: sign-preserving polynomial power transform
            if isinstance(change, dict):
                power = float(change.get("power", 1.0))
                s_float = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
                return np.sign(s_float) * (np.abs(s_float) ** power)
            return s.copy()

    def transform_data(
        self,
        df: pd.DataFrame,
        changed_dict: Dict[str, Any],
        num_attrs: Optional[List[str]] = None,
        cate_attrs: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Transform a full DataFrame according to changed_dict.
        """
        df_out = pd.DataFrame(index=df.index)
        num_set = set(num_attrs or [])
        cate_set = set(cate_attrs or [])

        for col in df.columns:
            if col in changed_dict:
                change = changed_dict[col]
                if change == "dropped":
                    continue  # Skip dropped columns
                is_cat = col in cate_set or not pd.api.types.is_numeric_dtype(df[col])
                t_series = self.transform_series(df[col], col, change, is_categorical=is_cat)
                if t_series is not None:
                    df_out[col] = t_series
            else:
                df_out[col] = df[col].copy()

        return df_out
