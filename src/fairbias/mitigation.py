"""Bias Mitigation with paper-aligned acceptance criteria.

Candidates are accepted only when they pass the NMI information-loss gate
(baseline ``module_BM.py`` phi semantics) AND strictly decrease the feature's
d_phi bias concentration. Categorical rebin candidates are searched on the
currently transformed frame; numerical polynomial-power candidates are searched
on the raw frame with absolute exponent replacement.
"""

from __future__ import annotations

import copy
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fairbias.evaluator import FairEvaluator
from fairbias.transform import FairTransform, calculate_nmi_dict, compose_category_mapping

# Minimal strict-improvement margin for d_phi acceptance
_DPHI_TOL = 1e-12


class FairBiasMitigation:
    """Iterative feature rebinning / polynomial-power mitigation with evidence-checked acceptance."""

    def __init__(
        self,
        evaluator: FairEvaluator,
        transformer: FairTransform,
        label_O: List[str],
        cate_attrs: List[str],
        num_attrs: List[str],
        max_search_candidates: int = 5,
        phi_threshold: float = 100.0,
        poly_exponents: Tuple[float, ...] = (1 / 3, 1 / 2, 2 / 3, 3.0, 5.0),
    ):
        self.evaluator = evaluator
        self.transformer = transformer
        self.label_O = label_O
        self.cate_attrs = cate_attrs
        self.num_attrs = num_attrs
        self.max_search_candidates = max_search_candidates
        self.phi_threshold = float(phi_threshold)
        self.poly_exponents = tuple(float(p) for p in poly_exponents)
        self.failed_attributes: set[str] = set()

    def find_ranked_epsilon_attributes(
        self, df_epsilon: Dict[str, Dict[str, float]]
    ) -> List[Tuple[float, str, str]]:
        """Return sorted list of (epsilon, label_O, attribute) candidates in descending order."""
        candidates = []
        for l_o, attr_dict in df_epsilon.items():
            for attr, eps in attr_dict.items():
                if attr not in self.failed_attributes:
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

    def _build_candidates(
        self,
        attr: str,
        label_O: str,
        X: pd.DataFrame,
        X_search: pd.DataFrame,
        O: pd.DataFrame,
        changed_dict: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build the ordered candidate changed_dict list for one attribute."""
        candidates: List[Dict[str, Any]] = []

        if attr in self.cate_attrs or not pd.api.types.is_numeric_dtype(X[attr]):
            existing = changed_dict.get(attr)
            existing_map = existing if isinstance(existing, dict) else {}
            for zorder in range(self.max_search_candidates):
                rebin_change = self.compute_r1_rebin(
                    df_feature=X_search[attr],
                    df_prot=O[label_O],
                    zorder=zorder,
                )
                if not rebin_change:
                    continue
                # Chained composition keeps earlier merges consistent with new ones
                composed = compose_category_mapping(existing_map, rebin_change)
                if composed == existing_map:
                    continue
                temp_changed = copy.deepcopy(changed_dict)
                temp_changed[attr] = composed
                candidates.append(temp_changed)
        else:
            existing = changed_dict.get(attr)
            current_power = float(existing.get("power", 1.0)) if isinstance(existing, dict) else 1.0
            for power in self.poly_exponents:
                if abs(power - current_power) < 1e-12 or abs(power - 1.0) < 1e-12:
                    continue
                temp_changed = copy.deepcopy(changed_dict)
                temp_changed[attr] = {"power": power}
                candidates.append(temp_changed)

        return candidates

    def mitigate_step(
        self,
        X: pd.DataFrame,
        Y: pd.Series,
        O: pd.DataFrame,
        nmi_org: Dict[str, float],
        changed_dict: Dict[str, Any],
        current_epsilon: Dict[str, Dict[str, float]],
        X_search: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[str], Optional[str]]:
        """
        Execute one bias mitigation step.

        Acceptance criteria (baseline ``module_BM.py`` semantics):
        1. structural validity of the candidate transform;
        2. NMI information-loss gate: phi = (nmi_org - nmi_new) / nmi_org <= phi_threshold;
        3. the feature's d_phi must strictly decrease after the transform.

        Parameters
        ----------
        X: raw training frame (the frame ``changed_dict`` is applied to).
        X_search: currently transformed training frame used for rebin search
            (falls back to ``X`` when omitted).
        """
        if X_search is None:
            X_search = X

        ranked_candidates = self.find_ranked_epsilon_attributes(current_epsilon)
        if not ranked_candidates:
            return self.transformer.transform_data(X, changed_dict, self.num_attrs, self.cate_attrs), changed_dict, None, None

        for eps, selected_label_O, selected_attribute in ranked_candidates:
            if selected_attribute not in X.columns:
                continue

            current_dphi = current_epsilon.get(selected_label_O, {}).get(selected_attribute)
            candidate_dicts = self._build_candidates(
                selected_attribute, selected_label_O, X, X_search, O, changed_dict
            )

            for temp_changed in candidate_dicts:
                # 1. Structural validity
                if not self.transformer.check_transform_validity(
                    X,
                    selected_attribute,
                    temp_changed[selected_attribute],
                    self.num_attrs,
                    self.cate_attrs,
                ):
                    continue

                transformed_candidate_df = self.transformer.transform_data(
                    X, temp_changed, self.num_attrs, self.cate_attrs
                )

                # 2. NMI information-loss gate
                nmi_new = calculate_nmi_dict(transformed_candidate_df, Y)
                nmi_before = float(nmi_org.get(selected_attribute, 1e-6))
                nmi_after = float(nmi_new.get(selected_attribute, 1e-6))
                phi_loss = (nmi_before - nmi_after) / (nmi_before + 1e-10)
                if phi_loss > self.phi_threshold:
                    continue

                # 3. d_phi must strictly decrease
                eps_new = self.evaluator.calculate_epsilon(
                    transformed_candidate_df, O, self.cate_attrs, self.num_attrs
                )
                new_dphi = eps_new.get(selected_label_O, {}).get(selected_attribute)
                if current_dphi is None or new_dphi is None:
                    continue
                if new_dphi < current_dphi - _DPHI_TOL:
                    return (
                        transformed_candidate_df,
                        temp_changed,
                        selected_label_O,
                        selected_attribute,
                    )

            # All candidates for this attribute failed to improve d_phi
            self.failed_attributes.add(selected_attribute)

        # Fallback: return current state without modifications
        current_df = self.transformer.transform_data(X, changed_dict, self.num_attrs, self.cate_attrs)
        return current_df, changed_dict, None, None
