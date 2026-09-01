"""Bias Mitigation with paper-aligned acceptance criteria.

Paper semantics (Tang, Lu & Li 2024, "Implementation details and data
transforms"):

- When the maximum d_phi exceeds the tolerance threshold epsilon, the
  attribute with the LARGEST d_phi is transformed, and the transform
  search for that attribute continues until its d_phi falls below
  epsilon.  Acceptance is tied to the epsilon ball, not to an arbitrary
  marginal decrease.
- Numerical attributes: single sign-preserving polynomial terms at odd
  integer (3, 5, 7, ...) or odd fraction (1/3, 1/5, 1/7, ...) powers,
  searched in increasing order until d_phi < epsilon.  The paper's main
  text lists these values as EXAMPLES ("e.g.") and prescribes no finite
  upper bound; the finite grid configured here is an implementation
  budget, and exhausting it is recorded as "configured grid exhausted",
  NOT as a paper-level non-convergence claim.  Values beyond
  numpy.float32 are set uniformly to 1, which is equivalent to dropping
  the attribute (recorded explicitly).
- Categorical attributes: at each step the two (possibly already
  rebinned) categories with the largest positive and smallest negative
  frequency gap are rebinned into one; the search ends when d_phi <
  epsilon.  Merging the two categories of a binary attribute is
  equivalent to excluding the attribute and is recorded as an explicit,
  auditable ``"dropped"`` state.  Accidental collapse through a raw
  mapping is still rejected.

The baseline NMI information-loss gate (``module_BM.py`` phi semantics)
is retained as an engineering safety constraint on the accepted final
state.
"""

from __future__ import annotations

import copy
from itertools import combinations
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fairbias.evaluator import FairEvaluator
from fairbias.transform import (
    FairTransform,
    calculate_nmi_dict,
    compose_category_mapping,
    power_transform_overflows,
)
from fairbias.transform_trace import FairBiasTransformStep


class FairBiasMitigation:
    """Iterative feature rebinning / polynomial-power mitigation with
    epsilon-ball acceptance criteria (paper semantics)."""

    def __init__(
        self,
        evaluator: FairEvaluator,
        transformer: FairTransform,
        label_O: List[str],
        cate_attrs: List[str],
        num_attrs: List[str],
        max_search_candidates: int = 5,
        phi_threshold: float = 100.0,
        poly_exponents: Tuple[float, ...] = (1 / 7, 1 / 5, 1 / 3, 3.0, 5.0, 7.0),
        failed_attribute_mode: str = "stop",
        preserve_exponent_order: Optional[bool] = None,
        power_sequence_policy: str = "sorted_grid",
        power_revisit_policy: str = "restart",
    ):
        self.evaluator = evaluator
        self.transformer = transformer
        self.label_O = label_O
        self.cate_attrs = cate_attrs
        self.num_attrs = num_attrs
        self.max_search_candidates = max_search_candidates
        self.phi_threshold = float(phi_threshold)

        # Disentangle sequence policy from revisit policy (REPAIR 2)
        if preserve_exponent_order is not None:
            if preserve_exponent_order:
                power_sequence_policy = "official_stream"
                power_revisit_policy = "monotone_cursor"
            else:
                power_sequence_policy = "sorted_grid"
                power_revisit_policy = "restart"

        self.power_sequence_policy = power_sequence_policy
        self.power_revisit_policy = power_revisit_policy
        self.preserve_exponent_order = (power_sequence_policy == "official_stream")

        if power_sequence_policy == "official_stream":
            self.poly_exponents = tuple(float(p) for p in poly_exponents)
        else:
            self.poly_exponents = tuple(sorted(float(p) for p in poly_exponents))

        # Monotone per-attribute cursor is ONLY active when power_revisit_policy == "monotone_cursor"
        if power_revisit_policy == "monotone_cursor":
            self._exponent_stream_cursors: Dict[str, int] = {}
        else:
            self._exponent_stream_cursors = {}

        # Fail-closed cycle detection tracking visited transform states
        self.visited_states: set[str] = set()
        self.visited_states.add(json.dumps({}, sort_keys=True))

        if failed_attribute_mode not in ("stop", "next"):
            raise ValueError(
                f"failed_attribute_mode must be 'stop' (default: highest-d_phi "
                f"attribute failure stops the run) or 'next' (named "
                f"engineering extension), got {failed_attribute_mode!r}"
            )
        self.failed_attribute_mode = failed_attribute_mode
        self.failed_attribute_keys: set = set()
        self.non_convergence: Optional[Dict[str, Any]] = None
        self.step_traces: List[FairBiasTransformStep] = []

    def find_ranked_epsilon_attributes(
        self, df_epsilon: Dict[str, Dict[str, float]]
    ) -> List[Tuple[float, str, str]]:
        """Return sorted list of (epsilon, label_O, attribute) candidates in descending order.

        Only (protected, feature) pairs NOT already recorded as failed (under
        the "next" engineering mode) are ranked; failures never mask the same
        feature under a different protected attribute.
        """
        candidates = []
        for l_o, attr_dict in df_epsilon.items():
            for attr, eps in attr_dict.items():
                if (str(l_o), str(attr)) not in self.failed_attribute_keys:
                    candidates.append((float(eps), str(l_o), str(attr)))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates

    def compute_r1_rebin(
        self,
        df_feature: pd.Series,
        df_prot: pd.Series,
        zorder: int = 0,
    ) -> Optional[Dict[Any, Any]]:
        """
        Compute category rebinning pair using exact sample proportion differences.

        Paper semantics: the two categories with the largest positive and
        smallest negative frequency gap between the protected groups are
        rebinned into one new category.

        Uses df.sum() (total sample count in group) as denominator rather than len(df) (category count).
        """
        df_combo = pd.DataFrame({
            "feat": df_feature.reset_index(drop=True),
            "prot": df_prot.reset_index(drop=True),
        })
        unique_groups = list(df_combo["prot"].dropna().unique())
        if len(unique_groups) < 2:
            return None

        diff_candidates: List[Dict[str, Any]] = []

        for g1, g2 in combinations(unique_groups, 2):
            s_0 = df_combo[df_combo["prot"] == g1]["feat"].value_counts()
            s_1 = df_combo[df_combo["prot"] == g2]["feat"].value_counts()

            if len(s_0) == 0 or len(s_1) == 0:
                continue

            # Mathematically correct proportion denominator: total sample count in group
            total_0 = float(s_0.sum())
            total_1 = float(s_1.sum())
            if total_0 <= 0 or total_1 <= 0:
                continue

            prop_0 = s_0 / total_0
            prop_1 = s_1 / total_1

            all_cats = list(set(prop_0.index) | set(prop_1.index))
            prop_0 = prop_0.reindex(all_cats, fill_value=0.0)
            prop_1 = prop_1.reindex(all_cats, fill_value=0.0)

            diff_series = (prop_1 - prop_0).sort_values(ascending=False)
            if len(diff_series) < 2:
                continue

            # Identify category pairs with greatest divergence
            cats_sorted = list(diff_series.index)
            # Rebin candidate: merge the highest-disparity category into the lowest-disparity category
            for idx in range(min(len(cats_sorted) - 1, self.max_search_candidates)):
                cat_src = cats_sorted[idx]
                cat_dst = cats_sorted[-(idx + 1)]
                if cat_src != cat_dst:
                    diff_val = float(diff_series[cat_src] - diff_series[cat_dst])
                    diff_candidates.append({
                        "change": {cat_src: cat_dst},
                        "diff": abs(diff_val),
                    })

        if not diff_candidates:
            return None

        diff_candidates.sort(key=lambda x: x["diff"], reverse=True)
        if zorder < len(diff_candidates):
            return diff_candidates[zorder]["change"]
        return diff_candidates[0]["change"]

    # ------------------------------------------------------------------
    # Acceptance helpers
    # ------------------------------------------------------------------

    def _nmi_gate_ok(
        self,
        transformed_df: pd.DataFrame,
        Y: pd.Series,
        nmi_org: Dict[str, float],
        attr: str,
    ) -> bool:
        """Baseline NMI information-loss gate on the accepted final state."""
        nmi_new = calculate_nmi_dict(transformed_df, Y)
        nmi_before = float(nmi_org.get(attr, 1e-6))
        nmi_after = float(nmi_new.get(attr, 0.0))
        phi_loss = (nmi_before - nmi_after) / (nmi_before + 1e-10)
        return phi_loss <= self.phi_threshold

    def _epsilon_of(
        self, candidate_df: pd.DataFrame, O: pd.DataFrame, label_O: str, attr: str
    ) -> Optional[float]:
        """Recompute d_phi for `attr`; a dropped attribute has d_phi = 0."""
        if attr not in candidate_df.columns:
            return 0.0
        eps_new = self.evaluator.calculate_epsilon(
            candidate_df, O, self.cate_attrs, self.num_attrs
        )
        return eps_new.get(label_O, {}).get(attr)

    def _make_candidate(
        self,
        X: pd.DataFrame,
        changed_dict: Dict[str, Any],
        attr: str,
        change: Any,
        Y: pd.Series,
        O: pd.DataFrame,
        nmi_org: Dict[str, float],
        label_O: str,
        epsilon_threshold: float,
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any], float]]:
        """Validate + gate + epsilon-check one candidate transform.

        Returns (transformed_df, temp_changed, new_dphi) when the candidate
        passes the structural validity check, the NMI information-loss gate,
        and brings the attribute's d_phi below `epsilon_threshold`.
        """
        if not self.transformer.check_transform_validity(
            X, attr, change, self.num_attrs, self.cate_attrs
        ):
            return None

        temp_changed = copy.deepcopy(changed_dict)
        temp_changed[attr] = change
        candidate_df = self.transformer.transform_data(
            X, temp_changed, self.num_attrs, self.cate_attrs
        )

        new_dphi = self._epsilon_of(candidate_df, O, label_O, attr)
        if new_dphi is None:
            return None
        if new_dphi >= epsilon_threshold:
            return None
        if not self._nmi_gate_ok(candidate_df, Y, nmi_org, attr):
            return None
        return candidate_df, temp_changed, float(new_dphi)

    # ------------------------------------------------------------------
    # Per-attribute transform searches (paper semantics)
    # ------------------------------------------------------------------

    def _search_categorical(
        self,
        X: pd.DataFrame,
        Y: pd.Series,
        O: pd.DataFrame,
        nmi_org: Dict[str, float],
        changed_dict: Dict[str, Any],
        label_O: str,
        attr: str,
        epsilon_threshold: float,
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Iteratively rebin the extreme frequency-gap category pair until
        the attribute's d_phi < epsilon (paper categorical transform).

        Merging down to a single category (the binary-attribute terminal
        merge) is recorded as an explicit, auditable ``"dropped"`` state.
        """
        existing = changed_dict.get(attr)
        existing_map = dict(existing) if isinstance(existing, dict) else {}

        n_categories = int(X[attr].nunique())
        max_merges = max(1, n_categories - 1)

        for _ in range(max_merges):
            # Current (already rebinned) attribute series
            if existing_map:
                s_work = self.transformer.transform_series(
                    X[attr], attr, existing_map, is_categorical=True
                )
            else:
                s_work = X[attr]
            if s_work.nunique() <= 1:
                break

            rebin = self.compute_r1_rebin(s_work, O[label_O], zorder=0)
            if not rebin:
                break

            composed = compose_category_mapping(existing_map, rebin)
            mapped = self.transformer.transform_series(
                X[attr], attr, composed, is_categorical=True
            )

            if mapped.nunique() <= 1:
                # Terminal merge: every category collapsed into one.  Paper:
                # merging the two categories of a binary attribute is
                # equivalent to excluding the attribute.  Recorded explicitly.
                accepted = self._make_candidate(
                    X, changed_dict, attr, "dropped",
                    Y, O, nmi_org, label_O, epsilon_threshold,
                )
                if accepted is not None:
                    candidate_df, temp_changed, _ = accepted
                    return candidate_df, temp_changed
                return None  # drop rejected by the information-loss gate

            accepted = self._make_candidate(
                X, changed_dict, attr, composed,
                Y, O, nmi_org, label_O, epsilon_threshold,
            )
            if accepted is not None:
                candidate_df, temp_changed, _ = accepted
                return candidate_df, temp_changed

            # Below-epsilon not yet met: keep merging (search continues)
            existing_map = composed

        return None

    def _search_numerical(
        self,
        X: pd.DataFrame,
        Y: pd.Series,
        O: pd.DataFrame,
        nmi_org: Dict[str, float],
        changed_dict: Dict[str, Any],
        label_O: str,
        attr: str,
        epsilon_threshold: float,
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Search single sign-preserving polynomial powers in stream order
        until the attribute's d_phi < epsilon (paper numerical transform).
        Powers overflowing numpy.float32 map to the explicit ``"dropped"``
        state.

        Round 4.1 REPAIR (Codex P0): in the official-code-derived mode
        (``preserve_exponent_order=True``) the search resumes from the
        persisted per-attribute stream cursor and consumes every searched
        position permanently (monotone, forward-only advancement), so a
        revisited attribute can never reuse an earlier power or oscillate
        between powers.  The engineering mode keeps the legacy
        restart-from-head search (bounded by the caller's iteration
        budget).
        """
        existing = changed_dict.get(attr)
        current_power = float(existing.get("power", 1.0)) if isinstance(existing, dict) else 1.0

        start_index = 0
        if self.power_revisit_policy == "monotone_cursor":
            start_index = self._exponent_stream_cursors.get(attr, 0)

        for index in range(start_index, len(self.poly_exponents)):
            power = self.poly_exponents[index]
            if self.power_revisit_policy == "monotone_cursor":
                # Consume this stream position permanently: the cursor
                # only ever moves forward for this attribute.
                self._exponent_stream_cursors[attr] = index + 1
            if abs(power - current_power) < 1e-12 or abs(power - 1.0) < 1e-12:
                continue

            if power_transform_overflows(X[attr], power):
                # Paper: values beyond numpy.float32 are set uniformly to 1,
                # which is equivalent to dropping the attribute.
                accepted = self._make_candidate(
                    X, changed_dict, attr, "dropped",
                    Y, O, nmi_org, label_O, epsilon_threshold,
                )
                if accepted is not None:
                    candidate_df, temp_changed, _ = accepted
                    return candidate_df, temp_changed
                continue

            accepted = self._make_candidate(
                X, changed_dict, attr, {"power": power},
                Y, O, nmi_org, label_O, epsilon_threshold,
            )
            if accepted is not None:
                candidate_df, temp_changed, _ = accepted
                return candidate_df, temp_changed

        return None

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def mitigate_step(
        self,
        X: pd.DataFrame,
        Y: pd.Series,
        O: pd.DataFrame,
        nmi_org: Dict[str, float],
        changed_dict: Dict[str, Any],
        current_epsilon: Dict[str, Dict[str, float]],
        epsilon_threshold: float,
        X_search: Optional[pd.DataFrame] = None,
        iteration: int = 1,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[str], Optional[str]]:
        """
        Execute one bias mitigation step (paper greedy semantics).

        The attribute with the largest d_phi is selected and transforms are
        searched for it until its d_phi falls below ``epsilon_threshold``.
        Attributes already inside the epsilon ball are left untouched.

        Failure semantics (``failed_attribute_mode``):

        - "stop" (default): if the transform search for the CURRENT
          highest-d_phi attribute cannot reach the epsilon ball,
          the run is recorded as non-convergent
          (``self.non_convergence``) and no transform is applied this
          step; the caller must stop.  The paper's greedy loop always
          operates on the current highest attribute, so silently moving
          on to a lower-ranked one is not paper semantics.  In
          engineering mode exhausting the configured grid is an
          implementation-budget outcome, NOT a paper-level
          non-convergence claim (the paper's power search has no stated
          finite bound).  In the official-code-derived mode the failure
          means EITHER the finite OFFICIAL power stream was exhausted
          for a NUMERIC attribute under the monotone stream cursor
          (``search_scope="official_power_stream"``) OR the categorical
          merge chain (including its rejected terminal drop) was
          exhausted for a CATEGORICAL attribute
          (``search_scope="categorical_merge_chain"``).
        - "next" (named engineering extension): the failure is recorded keyed
          by (protected attribute, feature) and the next-ranked attribute is
          considered instead.

        Parameters
        ----------
        X: raw training frame (the frame ``changed_dict`` is applied to).
        epsilon_threshold: the paper bias tolerance epsilon.
        X_search: kept for API compatibility; rebin gaps are computed on the
            raw frame composed with the working mapping, which reproduces the
            transformed search space exactly.
        iteration: current mitigation step index (default 1).
        """
        ranked_candidates = self.find_ranked_epsilon_attributes(current_epsilon)
        if not ranked_candidates:
            return self._no_op(X, changed_dict)

        for eps, selected_label_O, selected_attribute in ranked_candidates:
            if eps <= epsilon_threshold:
                # Every remaining attribute already sits inside the epsilon ball
                break
            if selected_attribute not in X.columns:
                continue

            if (
                selected_attribute in self.cate_attrs
                or not pd.api.types.is_numeric_dtype(X[selected_attribute])
            ):
                accepted = self._search_categorical(
                    X, Y, O, nmi_org, changed_dict,
                    selected_label_O, selected_attribute, epsilon_threshold,
                )
            else:
                accepted = self._search_numerical(
                    X, Y, O, nmi_org, changed_dict,
                    selected_label_O, selected_attribute, epsilon_threshold,
                )

            if accepted is not None:
                candidate_df, temp_changed = accepted
                change = temp_changed.get(selected_attribute)

                # State-cycle detection check (fail-closed for author stream restart search)
                canonical_state = json.dumps(temp_changed, sort_keys=True)
                if (
                    self.power_sequence_policy == "official_stream"
                    and self.power_revisit_policy == "restart"
                    and canonical_state in self.visited_states
                ):
                    self.non_convergence = {
                        "label_O": selected_label_O,
                        "attribute": selected_attribute,
                        "d_phi": float(eps),
                        "search_scope": "state_cycle_detected",
                        "reason": (
                            f"state cycle detected for attribute {selected_attribute!r}: "
                            f"proposed transform reproduces previously visited state {canonical_state}; "
                            "stopping fail-closed without mutating candidate search trajectory."
                        ),
                    }
                    sem_type = (
                        "categorical"
                        if (
                            selected_attribute in self.cate_attrs
                            or not pd.api.types.is_numeric_dtype(X[selected_attribute])
                        )
                        else "numerical"
                    )
                    self.step_traces.append(
                        FairBiasTransformStep(
                            iteration=int(iteration),
                            selected_feature=str(selected_attribute),
                            feature_semantic_type=sem_type,
                            d_phi_before=float(eps),
                            epsilon=float(epsilon_threshold),
                            proposed_transformation=change,
                            accepted_transformation=None,
                            numerical_exponent=None,
                            categorical_merge_mapping=None,
                            d_phi_after=float(eps),
                            dropped=False,
                            stopped_reason=str(self.non_convergence["reason"]),
                        )
                    )
                    return self._no_op(X, changed_dict)

                self.visited_states.add(canonical_state)

                is_dropped = (change == "dropped")
                num_exp = (
                    float(change["power"])
                    if isinstance(change, dict) and "power" in change
                    else None
                )
                cat_map = (
                    {str(k): str(v) for k, v in change.items()}
                    if isinstance(change, dict) and "power" not in change
                    else None
                )
                sem_type = (
                    "categorical"
                    if (
                        selected_attribute in self.cate_attrs
                        or not pd.api.types.is_numeric_dtype(X[selected_attribute])
                    )
                    else "numerical"
                )
                new_dphi = self._epsilon_of(candidate_df, O, selected_label_O, selected_attribute)
                self.step_traces.append(
                    FairBiasTransformStep(
                        iteration=int(iteration),
                        selected_feature=str(selected_attribute),
                        feature_semantic_type=sem_type,
                        d_phi_before=float(eps),
                        epsilon=float(epsilon_threshold),
                        proposed_transformation=change,
                        accepted_transformation=change,
                        numerical_exponent=num_exp,
                        categorical_merge_mapping=cat_map,
                        d_phi_after=float(new_dphi) if new_dphi is not None else None,
                        dropped=is_dropped,
                        stopped_reason=None,
                    )
                )
                return candidate_df, temp_changed, selected_label_O, selected_attribute

            # The transform search could not bring this attribute below epsilon
            if self.failed_attribute_mode == "stop":
                # Failure semantics: report the search exhaustion on the
                # current highest attribute and stop the mitigation loop.
                is_categorical_search = (
                    selected_attribute in self.cate_attrs
                    or not pd.api.types.is_numeric_dtype(
                        X[selected_attribute]
                    )
                )
                if is_categorical_search:
                    self.non_convergence = {
                        "label_O": selected_label_O,
                        "attribute": selected_attribute,
                        "d_phi": float(eps),
                        "search_scope": "categorical_merge_chain",
                        "reason": (
                            "categorical merge chain exhausted for this "
                            "attribute (bounded merges plus the rejected "
                            "terminal drop could not reach the epsilon "
                            "ball); the loop has no iteration budget"
                        ),
                    }
                elif self.power_sequence_policy == "official_stream":
                    if self.power_revisit_policy == "monotone_cursor":
                        self.non_convergence = {
                            "label_O": selected_label_O,
                            "attribute": selected_attribute,
                            "d_phi": float(eps),
                            "search_scope": "official_power_stream",
                            "stream_positions_consumed": int(
                                self._exponent_stream_cursors.get(
                                    selected_attribute, 0
                                )
                            ),
                            "reason": (
                                "official power stream exhausted for this "
                                "attribute under the monotone per-attribute "
                                "stream cursor (no position is retried); the "
                                "official-code-derived loop has no "
                                "iteration budget"
                            ),
                        }
                    else:
                        self.non_convergence = {
                            "label_O": selected_label_O,
                            "attribute": selected_attribute,
                            "d_phi": float(eps),
                            "search_scope": "official_power_stream",
                            "reason": (
                                "author power stream exhausted for this "
                                "attribute under head restart search; the "
                                "paper reference loop has no iteration budget"
                            ),
                        }
                else:
                    self.non_convergence = {
                        "label_O": selected_label_O,
                        "attribute": selected_attribute,
                        "d_phi": float(eps),
                        "search_scope": "configured_grid",
                        "reason": (
                            "configured candidate grid exhausted; not a "
                            "paper-level non-convergence claim (paper power "
                            "search has no stated finite bound)"
                        ),
                    }
                if self.non_convergence:
                    sem_type = (
                        "categorical"
                        if (
                            selected_attribute in self.cate_attrs
                            or not pd.api.types.is_numeric_dtype(X[selected_attribute])
                        )
                        else "numerical"
                    )
                    self.step_traces.append(
                        FairBiasTransformStep(
                            iteration=int(iteration),
                            selected_feature=str(selected_attribute),
                            feature_semantic_type=sem_type,
                            d_phi_before=float(eps),
                            epsilon=float(epsilon_threshold),
                            proposed_transformation=None,
                            accepted_transformation=None,
                            numerical_exponent=None,
                            categorical_merge_mapping=None,
                            d_phi_after=float(eps),
                            dropped=False,
                            stopped_reason=str(self.non_convergence.get("reason")),
                        )
                    )
                break

            # Engineering extension ("next"): record keyed by
            # (protected attribute, feature) and try the next-ranked attribute
            self.failed_attribute_keys.add((selected_label_O, selected_attribute))

        return self._no_op(X, changed_dict)

    def _no_op(
        self, X: pd.DataFrame, changed_dict: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[str], Optional[str]]:
        current_df = self.transformer.transform_data(X, changed_dict, self.num_attrs, self.cate_attrs)
        return current_df, changed_dict, None, None
