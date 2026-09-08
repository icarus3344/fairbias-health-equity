"""Accuracy Enhancement module with bounded fairness degradation and deepcopy protection."""

from __future__ import annotations

import collections
import copy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from fairbias.evaluator import FairEvaluator
from fairbias.transform import FairTransform, calculate_nmi_dict, compose_category_mapping

DEFAULT_POLY_GRID: Tuple[float, ...] = (1 / 7, 1 / 5, 1 / 3, 3.0, 5.0, 7.0)


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
        poly_exponents: Sequence[float] = DEFAULT_POLY_GRID,
    ):
        self.evaluator = evaluator
        self.transformer = transformer
        self.label_Y = label_Y
        self.cate_attrs = cate_attrs
        self.num_attrs = num_attrs
        self.max_fairness_degradation = max_fairness_degradation
        self.poly_exponents = tuple(float(p) for p in poly_exponents)
        self.skip_attr_list: set[str] = set()
        self.tried_exponents: Dict[str, Set[float]] = collections.defaultdict(set)
        self.tried_rebins: Dict[str, Set[Tuple[Any, Any]]] = collections.defaultdict(set)
        self._cached_ranking: Optional[List[str]] = None

    def find_target_correlated_attribute(
        self, X: pd.DataFrame, Y: pd.Series, changed_dict: Dict[str, Any]
    ) -> Optional[str]:
        """Rank features by mutual information with target Y, returning the highest eligible feature."""
        if X.empty or len(Y) == 0:
            return None

        if self._cached_ranking is None:
            nmi_scores = calculate_nmi_dict(X, Y)
            self._cached_ranking = [
                col for col, _ in sorted(nmi_scores.items(), key=lambda item: item[1], reverse=True)
            ]

        for col in self._cached_ranking:
            if col in self.skip_attr_list:
                continue
            # Do not touch dropped features
            if changed_dict.get(col) == "dropped":
                continue
            return col
        return None

    def _evaluate_utility(
        self,
        X_train: pd.DataFrame,
        Y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        Y_val: Optional[pd.Series] = None,
    ) -> float:
        """Fit model and return utility metric (AUROC if binary with probabilities, ACC otherwise)."""
        try:
            if X_val is not None and Y_val is not None and len(X_val) > 0:
                y_pred, y_prob = self.evaluator.fit_and_predict(X_train, Y_train, X_val)
                y_true = Y_val.values
            else:
                stratify = Y_train if Y_train.nunique() > 1 else None
                x_tr, x_eval, y_tr, y_eval = train_test_split(
                    X_train,
                    Y_train,
                    test_size=0.3,
                    random_state=self.evaluator.config.random_seed,
                    stratify=stratify,
                )
                y_pred, y_prob = self.evaluator.fit_and_predict(x_tr, y_tr, x_eval)
                y_true = y_eval.values

            if len(np.unique(y_true)) >= 2 and y_prob is not None:
                try:
                    return float(roc_auc_score(y_true, y_prob))
                except Exception:
                    pass
            return float(accuracy_score(y_true, y_pred))
        except Exception:
            return 0.0

    def _is_fairness_acceptable(
        self,
        X_cand: pd.DataFrame,
        O_train: Optional[pd.DataFrame],
        epsilon_threshold: Optional[float],
        current_max_epsilon: Optional[float],
    ) -> Tuple[bool, float]:
        """Check whether candidate transformed data stays within fairness degradation bounds."""
        if O_train is None or (epsilon_threshold is None and current_max_epsilon is None):
            return True, 0.0

        cand_eps_dict = self.evaluator.calculate_epsilon(
            X_cand, O_train, cate_attrs=self.cate_attrs, num_attrs=self.num_attrs
        )
        cand_all_eps = [
            v for gd in cand_eps_dict.values() for v in gd.values()
        ]
        cand_max_eps = float(max(cand_all_eps)) if cand_all_eps else 0.0

        # Bound: cannot exceed epsilon_threshold (or current max) plus allowed degradation
        reference_eps = epsilon_threshold if epsilon_threshold is not None else current_max_epsilon
        upper_bound = reference_eps + self.max_fairness_degradation
        return (cand_max_eps <= upper_bound), cand_max_eps

    def enhance_step(
        self,
        X_train: pd.DataFrame,
        Y_train: pd.Series,
        changed_dict: Dict[str, Any],
        O_train: Optional[pd.DataFrame] = None,
        epsilon_threshold: Optional[float] = None,
        current_epsilon: Optional[Dict[str, Dict[str, float]]] = None,
        X_val: Optional[pd.DataFrame] = None,
        Y_val: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[str]]:
        """Attempt an accuracy enhancement step on training data under fairness bounds.

        Returns:
            Tuple of (transformed_X_train, new_changed_dict, selected_attribute_or_None).
        """
        current_df = self.transformer.transform_data(
            X_train, changed_dict, self.num_attrs, self.cate_attrs
        )

        curr_max_eps = None
        if current_epsilon is not None:
            all_eps = [v for gd in current_epsilon.values() for v in gd.values()]
            curr_max_eps = float(max(all_eps)) if all_eps else None

        current_utility = self._evaluate_utility(current_df, Y_train, X_val, Y_val)

        # Loop through eligible attributes until an accepted transformation is found or all fail
        while True:
            target_attr = self.find_target_correlated_attribute(X_train, Y_train, changed_dict)
            if target_attr is None:
                return current_df, changed_dict, None

            is_categorical = (
                target_attr in self.cate_attrs
                or not pd.api.types.is_numeric_dtype(X_train[target_attr])
            )

            if is_categorical:
                accepted = self._try_categorical_enhancement(
                    target_attr=target_attr,
                    X_train=X_train,
                    Y_train=Y_train,
                    changed_dict=changed_dict,
                    current_utility=current_utility,
                    O_train=O_train,
                    epsilon_threshold=epsilon_threshold,
                    curr_max_eps=curr_max_eps,
                    X_val=X_val,
                    Y_val=Y_val,
                )
            else:
                accepted = self._try_numerical_enhancement(
                    target_attr=target_attr,
                    X_train=X_train,
                    Y_train=Y_train,
                    changed_dict=changed_dict,
                    current_utility=current_utility,
                    O_train=O_train,
                    epsilon_threshold=epsilon_threshold,
                    curr_max_eps=curr_max_eps,
                    X_val=X_val,
                    Y_val=Y_val,
                )

            if accepted is not None:
                transformed_df, new_changed = accepted
                return transformed_df, new_changed, target_attr

            # No transformation accepted for this attribute; mark as exhausted
            self.skip_attr_list.add(target_attr)

    def _try_numerical_enhancement(
        self,
        target_attr: str,
        X_train: pd.DataFrame,
        Y_train: pd.Series,
        changed_dict: Dict[str, Any],
        current_utility: float,
        O_train: Optional[pd.DataFrame],
        epsilon_threshold: Optional[float],
        curr_max_eps: Optional[float],
        X_val: Optional[pd.DataFrame],
        Y_val: Optional[pd.Series],
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Test candidate polynomial powers monotonically for a numerical feature."""
        base_power = 1.0
        if target_attr in changed_dict and isinstance(changed_dict[target_attr], dict):
            base_power = float(changed_dict[target_attr].get("power", 1.0))
        self.tried_exponents[target_attr].add(base_power)

        best_cand: Optional[Tuple[pd.DataFrame, Dict[str, Any]]] = None
        best_gain = 0.0

        for power in self.poly_exponents:
            if power in self.tried_exponents[target_attr]:
                continue
            self.tried_exponents[target_attr].add(power)

            cand_change = copy.deepcopy(changed_dict)
            cand_change[target_attr] = {"power": power}

            if not self.transformer.check_transform_validity(
                X_train, target_attr, cand_change[target_attr], self.num_attrs, self.cate_attrs
            ):
                continue

            cand_X = self.transformer.transform_data(
                X_train, cand_change, self.num_attrs, self.cate_attrs
            )

            fairness_ok, _ = self._is_fairness_acceptable(
                cand_X, O_train, epsilon_threshold, curr_max_eps
            )
            if not fairness_ok:
                continue

            cand_utility = self._evaluate_utility(cand_X, Y_train, X_val, Y_val)
            # Require strict utility improvement
            if cand_utility > current_utility:
                gain = cand_utility - current_utility
                if gain > best_gain:
                    best_gain = gain
                    best_cand = (cand_X, cand_change)

        return best_cand

    def _try_categorical_enhancement(
        self,
        target_attr: str,
        X_train: pd.DataFrame,
        Y_train: pd.Series,
        changed_dict: Dict[str, Any],
        current_utility: float,
        O_train: Optional[pd.DataFrame],
        epsilon_threshold: Optional[float],
        curr_max_eps: Optional[float],
        X_val: Optional[pd.DataFrame],
        Y_val: Optional[pd.Series],
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Test category merging based on target rate similarity."""
        s = X_train[target_attr]
        unique_cats = list(s.unique())
        if len(unique_cats) < 3:
            return None

        cat_target_rates = Y_train.groupby(s).mean().to_dict()
        sorted_cats = sorted(cat_target_rates.items(), key=lambda x: x[1])

        # Generate adjacent category pairs
        candidate_pairs = []
        for i in range(len(sorted_cats) - 1):
            pair = (sorted_cats[i][0], sorted_cats[i + 1][0])
            norm_pair = (min(str(pair[0]), str(pair[1])), max(str(pair[0]), str(pair[1])))
            if norm_pair not in self.tried_rebins[target_attr]:
                diff = abs(sorted_cats[i][1] - sorted_cats[i + 1][1])
                candidate_pairs.append((pair, norm_pair, diff))

        candidate_pairs.sort(key=lambda x: x[2])

        best_cand: Optional[Tuple[pd.DataFrame, Dict[str, Any]]] = None
        best_gain = 0.0

        for pair, norm_pair, _ in candidate_pairs:
            self.tried_rebins[target_attr].add(norm_pair)
            rebin = {pair[1]: pair[0]}

            cand_change = copy.deepcopy(changed_dict)
            if target_attr in cand_change and isinstance(cand_change[target_attr], dict):
                cand_change[target_attr] = compose_category_mapping(cand_change[target_attr], rebin)
            else:
                cand_change[target_attr] = rebin

            if not self.transformer.check_transform_validity(
                X_train, target_attr, cand_change[target_attr], self.num_attrs, self.cate_attrs
            ):
                continue

            cand_X = self.transformer.transform_data(
                X_train, cand_change, self.num_attrs, self.cate_attrs
            )

            fairness_ok, _ = self._is_fairness_acceptable(
                cand_X, O_train, epsilon_threshold, curr_max_eps
            )
            if not fairness_ok:
                continue

            cand_utility = self._evaluate_utility(cand_X, Y_train, X_val, Y_val)
            if cand_utility > current_utility:
                gain = cand_utility - current_utility
                if gain > best_gain:
                    best_gain = gain
                    best_cand = (cand_X, cand_change)

        return best_cand
