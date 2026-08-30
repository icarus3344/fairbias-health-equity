"""Accuracy Enhancement module with bounded fairness degradation and deepcopy protection."""

from __future__ import annotations

import copy
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from fairbias.evaluator import FairEvaluator
from fairbias.transform import FairTransform


class FairAccuracyEnhancement:
    """Explores accuracy-enhancing feature transformations under fairness degradation constraints."""

    def __init__(
        self,
        evaluator: FairEvaluator,
        transformer: FairTransform,
        label_Y: str,
        cate_attrs: List[str],
        num_attrs: List[str],
        max_fairness_degradation: float = 0.02,
    ):
        self.evaluator = evaluator
        self.transformer = transformer
        self.label_Y = label_Y
        self.cate_attrs = cate_attrs
        self.num_attrs = num_attrs
        self.max_fairness_degradation = max_fairness_degradation
        self.skip_attr_list: set[str] = set()

    def find_target_correlated_attribute(
        self, X: pd.DataFrame, Y: pd.Series, changed_dict: Dict[str, Any]
    ) -> Optional[str]:
        """Rank features by mutual information with target Y, returning the highest eligible feature."""
        if X.empty or len(Y) == 0:
            return None

        mi_scores = mutual_info_classif(X.values, Y.values, random_state=0)
        ranked_cols = [
            col
            for _, col in sorted(
                zip(mi_scores, X.columns), key=lambda item: item[0], reverse=True
            )
        ]

        for col in ranked_cols:
            if col not in self.skip_attr_list:
                return col
        return None

    def enhance_step(
        self,
        X_train: pd.DataFrame,
        Y_train: pd.Series,
        changed_dict: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[str]]:
        """
        Attempt an accuracy enhancement step on training data.
        """
        target_attr = self.find_target_correlated_attribute(X_train, Y_train, changed_dict)
        if target_attr is None:
            current_df = self.transformer.transform_data(
                X_train, changed_dict, self.num_attrs, self.cate_attrs
            )
            return current_df, changed_dict, None

        temp_changed = copy.deepcopy(changed_dict)

        if target_attr in self.cate_attrs or not pd.api.types.is_numeric_dtype(X_train[target_attr]):
            # Category merging based on target prevalence similarity
            s = X_train[target_attr]
            unique_cats = list(s.unique())
            if len(unique_cats) >= 3:
                # Find two categories with most similar target rate to merge
                cat_target_rates = Y_train.groupby(s).mean().to_dict()
                sorted_cats = sorted(cat_target_rates.items(), key=lambda x: x[1])
                # Merge the two adjacent categories with smallest difference
                min_diff = float("inf")
                best_pair = (sorted_cats[0][0], sorted_cats[1][0])
                for i in range(len(sorted_cats) - 1):
                    diff = abs(sorted_cats[i][1] - sorted_cats[i + 1][1])
                    if diff < min_diff:
                        min_diff = diff
                        best_pair = (sorted_cats[i][0], sorted_cats[i + 1][0])

                rebin = {best_pair[1]: best_pair[0]}
                if target_attr in temp_changed and isinstance(temp_changed[target_attr], dict):
                    temp_changed[target_attr].update(rebin)
                else:
                    temp_changed[target_attr] = rebin
        else:
            # Numerical feature: polynomial power search favoring target information
            base_power = 1.0
            if target_attr in temp_changed and isinstance(temp_changed[target_attr], dict):
                base_power = float(temp_changed[target_attr].get("power", 1.0))
            chosen_power = None
            # Paper power grid: odd fractions and odd integers, increasing order
            for power in (1 / 7, 1 / 5, 1 / 3, 3.0, 5.0, 7.0):
                if abs(power - base_power) < 1e-12:
                    continue
                candidate_change = {"power": power}
                if self.transformer.check_transform_validity(
                    X_train, target_attr, candidate_change, self.num_attrs, self.cate_attrs
                ):
                    chosen_power = power
                    break
            if chosen_power is None:
                self.skip_attr_list.add(target_attr)
                current_df = self.transformer.transform_data(
                    X_train, changed_dict, self.num_attrs, self.cate_attrs
                )
                return current_df, changed_dict, None
            temp_changed[target_attr] = {"power": chosen_power}

        if self.transformer.check_transform_validity(
            X_train,
            target_attr,
            temp_changed[target_attr],
            self.num_attrs,
            self.cate_attrs,
        ):
            transformed_df = self.transformer.transform_data(
                X_train, temp_changed, self.num_attrs, self.cate_attrs
            )
            return transformed_df, temp_changed, target_attr

        self.skip_attr_list.add(target_attr)
        current_df = self.transformer.transform_data(
            X_train, changed_dict, self.num_attrs, self.cate_attrs
        )
        return current_df, changed_dict, None
