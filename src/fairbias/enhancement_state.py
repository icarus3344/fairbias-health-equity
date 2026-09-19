"""State management, canonical serialization, category normalization, and candidate tracking for FairBias enhancement.

Implements R1.2 requirements:
- Deterministic state hashing
- Unambiguous category mapping normalization (JSON string keys vs in-memory numeric keys)
- Transitive category composition ensuring previously merged categories are not split
- State-bound candidate evaluation caching and cycle detection.
"""

from __future__ import annotations

import collections
import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd


def canonical_json_dump(data: Any, *, version: str = "v2_lossless") -> str:
    """Serialize data structure to a deterministic canonical JSON string.

    Parameters
    ----------
    data : Any
        Object or dictionary to serialize.
    version : str, default "v2_lossless"
        - "v2_lossless": Lossless float serialization without precision rounding.
          Rejects non-finite values (NaN, Inf, -Inf) and unsupported types.
        - "v1_legacy_8dec": Historical D6/D8/R2 compatibility scheme that rounds floating point
          numbers to 8 decimal places. Reproduces authoritative frozen reference hashes.
    """
    if version not in ("v2_lossless", "v1_legacy_8dec"):
        raise ValueError(
            f"Invalid serialization version '{version}'; must be 'v2_lossless' or 'v1_legacy_8dec'"
        )

    def _canonical(obj: Any) -> Any:
        if isinstance(obj, dict):
            seen_keys: Set[str] = set()
            res = {}
            for k, v in sorted(obj.items(), key=lambda x: str(x[0])):
                sk = str(k)
                if sk in seen_keys:
                    raise ValueError(f"Ambiguous dictionary keys colliding under string conversion: {k!r}")
                seen_keys.add(sk)
                res[sk] = _canonical(v)
            return res
        if isinstance(obj, (list, tuple)):
            return [_canonical(v) for v in obj]
        if isinstance(obj, set):
            return [_canonical(v) for v in sorted(obj, key=lambda x: str(x))]
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):
                if version == "v1_legacy_8dec":
                    return None
                raise ValueError(f"Non-finite float value {obj} rejected in canonical serialization")
            if version == "v1_legacy_8dec":
                return round(float(obj), 8)
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, str):
            return str(obj)
        if obj is None:
            return None
        raise TypeError(f"Unsupported parameter type {type(obj).__name__} in canonical serialization: {obj!r}")

    return json.dumps(_canonical(data), sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_transform_state(changed_dict: Dict[str, Any], *, version: str = "v2_lossless") -> str:
    """Compute a deterministic 16-hex SHA-256 hash of a transform changed_dict.

    Defaults to version="v2_lossless" for all runtime parent/candidate/committed/cycle/cache identities.
    Supports version="v1_legacy_8dec" for explicit historical compatibility.
    """
    s = canonical_json_dump(changed_dict, version=version)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


changed_dict_hash = hash_transform_state


def hash_transform_state_lossless(changed_dict: Dict[str, Any]) -> str:
    """Compute lossless v2 hash of a transform changed_dict."""
    return hash_transform_state(changed_dict, version="v2_lossless")


def hash_transform_state_legacy(changed_dict: Dict[str, Any]) -> str:
    """Compute historical v1 (8-decimal) hash of a transform changed_dict."""
    return hash_transform_state(changed_dict, version="v1_legacy_8dec")


def transform_states_are_equivalent(
    state_a: Dict[str, Any],
    state_b: Dict[str, Any],
) -> bool:
    """Check semantic equivalence between two transformation states."""
    if not isinstance(state_a, dict) or not isinstance(state_b, dict):
        return state_a == state_b
    if set(state_a.keys()) != set(state_b.keys()):
        return False
    for k in state_a:
        v_a = state_a[k]
        v_b = state_b[k]
        if isinstance(v_a, dict) and isinstance(v_b, dict):
            if not transform_states_are_equivalent(v_a, v_b):
                return False
        elif isinstance(v_a, (float, np.floating)) or isinstance(v_b, (float, np.floating)):
            try:
                if not np.isclose(float(v_a), float(v_b), rtol=1e-7, atol=1e-9):
                    return False
            except Exception:
                if v_a != v_b:
                    return False
        else:
            if v_a != v_b:
                return False
    return True


def normalize_category_mapping(
    mapping: Dict[Any, Any],
    col_sample: Optional[pd.Series] = None,
) -> Dict[Any, Any]:
    """
    Normalize category mapping keys and values according to column schema.

    - Resolves string keys vs in-memory numeric keys.
    - Explicitly raises ValueError on conflicting keys (e.g. '3': 1 vs 3: 2).
    - Preserves exact string representation when column contains non-numeric strings (e.g. '01' != '1').
    """
    if not mapping:
        return {}

    # Check for string vs int conflict
    str_key_map: Dict[str, Tuple[Any, Any]] = {}
    for k, v in mapping.items():
        k_str = str(k)
        if k_str in str_key_map:
            prev_k, prev_v = str_key_map[k_str]
            if prev_v != v:
                raise ValueError(
                    f"Conflicting categorical mapping keys: {prev_k!r} -> {prev_v!r} "
                    f"conflicts with {k!r} -> {v!r}"
                )
        else:
            str_key_map[k_str] = (k, v)

    # If column sample provided, infer target type
    if col_sample is not None and not col_sample.empty:
        non_null = col_sample.dropna()
        if not non_null.empty:
            first_val = non_null.iloc[0]
            # If values are integers
            if pd.api.types.is_integer_dtype(non_null) or (
                isinstance(first_val, (int, np.integer)) and not isinstance(first_val, bool)
            ):
                try:
                    return {int(k): int(v) for k, v in mapping.items()}
                except (ValueError, TypeError):
                    pass
            # If values are strings
            if isinstance(first_val, str):
                return {str(k): str(v) for k, v in mapping.items()}

    # Fallback: check if all keys and values are clean integers
    all_int = True
    for k, v in mapping.items():
        try:
            k_int = int(k)
            v_int = int(v)
            if str(k_int) != str(k).strip() or str(v_int) != str(v).strip():
                all_int = False
                break
        except (ValueError, TypeError):
            all_int = False
            break

    if all_int:
        return {int(k): int(v) for k, v in mapping.items()}

    return {str(k): str(v) for k, v in mapping.items()}


def safe_compose_category_mapping(
    existing: Dict[Any, Any],
    new_merge: Dict[Any, Any],
    col_sample: Optional[pd.Series] = None,
) -> Dict[Any, Any]:
    """
    Compose existing category mapping with a new merge step.

    Maintains transitive closure: if existing maps A -> B and new maps B -> C,
    the resulting mapping maps A -> C and B -> C.
    Verifies that categories previously merged together are never split apart.
    """
    norm_existing = normalize_category_mapping(existing, col_sample)
    norm_new = normalize_category_mapping(new_merge, col_sample)

    composed: Dict[Any, Any] = {}
    for key, value in norm_existing.items():
        composed[key] = norm_new.get(value, value)
    for key, value in norm_new.items():
        if key not in composed:
            composed[key] = value

    # Verification: groups previously mapped to the same target must still map to the same target
    reverse_existing: Dict[Any, List[Any]] = collections.defaultdict(list)
    for k, v in norm_existing.items():
        reverse_existing[v].append(k)

    for target_group, members in reverse_existing.items():
        # Check all members map to the same final target in composed
        final_targets = {composed.get(m, m) for m in members}
        # Also check the target group itself if it is in composed
        if target_group in composed:
            final_targets.add(composed[target_group])
        if len(final_targets) > 1:
            raise ValueError(
                f"Transitive merge invariant violated: members of group {target_group} "
                f"split into multiple targets {final_targets}"
            )

    return normalize_category_mapping(composed, col_sample)


class StatefulCandidateTracker:
    """
    Tracks candidate evaluations bound to transform states to prevent stale cache pollution.

    - Caches evaluated candidates per parent state hash.
    - Prevents repeating identical candidates on the exact same state.
    - Allows features to be re-evaluated when a different feature transform changes the state.
    - Detects state trajectory cycles.
    """

    def __init__(self) -> None:
        # state_hash -> set of candidate signatures evaluated
        self._state_evaluated_candidates: Dict[str, Set[str]] = collections.defaultdict(set)
        # Sequence of visited state hashes to detect cycles
        self._state_history: List[str] = []
        self._context_state_histories: Dict[str, List[str]] = collections.defaultdict(list)
        # Count of total candidate evaluations
        self.total_evaluations: int = 0

    def record_state_visit(self, state_hash: str, context_fingerprint: str = "") -> None:
        """Record visit to a state hash, scoped to evaluation context if provided."""
        self._state_history.append(state_hash)
        if context_fingerprint:
            self._context_state_histories[context_fingerprint].append(state_hash)

    def is_cycle(self, candidate_state_hash: str, context_fingerprint: str = "") -> bool:
        """Check if transitioning to candidate_state_hash would revisit an earlier state in this context."""
        if context_fingerprint and context_fingerprint in self._context_state_histories:
            return candidate_state_hash in self._context_state_histories[context_fingerprint]
        return candidate_state_hash in self._state_history

    def reset_context_history(self, context_fingerprint: str = "") -> None:
        """Clear cycle history for a specific context or completely."""
        if context_fingerprint:
            self._context_state_histories[context_fingerprint].clear()
        else:
            self._state_history.clear()
            self._context_state_histories.clear()

    def is_candidate_evaluated(
        self, parent_state_hash: str, candidate_sig: str, context_fingerprint: str = ""
    ) -> bool:
        """Check if candidate signature was already evaluated on parent_state_hash with context."""
        key = f"{candidate_sig}:{context_fingerprint}" if context_fingerprint else candidate_sig
        return key in self._state_evaluated_candidates[parent_state_hash]

    def mark_candidate_evaluated(
        self, parent_state_hash: str, candidate_sig: str, context_fingerprint: str = ""
    ) -> None:
        """Record that candidate signature was evaluated on parent_state_hash with context."""
        key = f"{candidate_sig}:{context_fingerprint}" if context_fingerprint else candidate_sig
        self._state_evaluated_candidates[parent_state_hash].add(key)
        self.total_evaluations += 1

    def state_history(self, context_fingerprint: str = "") -> List[str]:
        if context_fingerprint and context_fingerprint in self._context_state_histories:
            return list(self._context_state_histories[context_fingerprint])
        return list(self._state_history)
