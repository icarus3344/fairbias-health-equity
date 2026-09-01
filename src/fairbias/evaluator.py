"""Leakage-free evaluator with train-only scaling, baseline-aligned fairness metrics, and paper-level d_phi epsilon."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from fairbias.bias_metric import compute_dphi_matrix, get_subsets  # noqa: F401 (get_subsets re-exported for API compat)
from fairbias.config import ALGORITHM_MODE_PAPER_FAITHFUL, FairBiasConfig
from fairbias.models import get_classifier


class FairEvaluator:
    """Evaluates classifier performance and demographic fairness disparities without test leakage."""

    def __init__(
        self,
        config: Optional[FairBiasConfig] = None,
        label_O: Optional[List[str]] = None,
        label_Y: Optional[str] = None,
        cate_attrs: Optional[List[str]] = None,
        num_attrs: Optional[List[str]] = None,
    ):
        self.config = config or FairBiasConfig.compas_default()
        self.label_O = label_O or list(self.config.label_O)
        self.label_Y = label_Y or self.config.label_Y
        self.cate_attrs = cate_attrs or []
        self.num_attrs = num_attrs or []
        self.model: BaseEstimator = get_classifier(
            self.config.classifier, random_state=self.config.random_seed
        )
        self.scaler: Optional[Union[MinMaxScaler, StandardScaler]] = None

    def _get_scaler(self) -> Optional[Union[MinMaxScaler, StandardScaler]]:
        if self.config.eval_norm == "min-max":
            return MinMaxScaler(feature_range=(0, 1))
        elif self.config.eval_norm == "z-score":
            return StandardScaler()
        return None

    def fit_and_predict(
        self,
        X_train: pd.DataFrame,
        Y_train: pd.Series,
        X_test: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scaler and model STRICTLY on training data; predict on test data.
        """
        scaler = self._get_scaler()
        if scaler is not None:
            # Fit scaler STRICTLY on X_train
            X_tr_scaled = scaler.fit_transform(X_train)
            X_te_scaled = scaler.transform(X_test)
            self.scaler = scaler
        else:
            X_tr_scaled = X_train.values
            X_te_scaled = X_test.values

        # Fit model on training fold
        self.model.fit(X_tr_scaled, Y_train.values)

        # Predict on test fold
        y_pred = self.model.predict(X_te_scaled)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_te_scaled)
            y_prob = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
        else:
            y_prob = y_pred.astype(float)

        return y_pred, y_prob

    @staticmethod
    def _binary_stats(y_true_g: np.ndarray, y_pred_g: np.ndarray, pos_class: int) -> Dict[str, float]:
        """
        One-vs-rest confusion-matrix rates for a single group, mirroring the frozen
        baseline ``Evaluator._binary_confusion_counts`` (eval.py).
        """
        y_t = (y_true_g == pos_class).astype(int)
        y_p = (y_pred_g == pos_class).astype(int)

        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        n = len(y_t)

        return {
            "TPR": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            "PPV": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "NPV": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
            "ACC": (tp + tn) / n if n > 0 else 0.0,
            "PPOS": (tp + fp) / n if n > 0 else 0.0,
            "FDR": fp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "FOR": fn / (tn + fn) if (tn + fn) > 0 else 0.0,
            "FNR": fn / (tp + fn) if (tp + fn) > 0 else 0.0,
        }

    def compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        O_df: pd.DataFrame,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute performance and group fairness metrics on the evaluation partition.

        Fairness disparity definitions follow the reference paper (Eqs. 8-9)
        and the frozen baseline ``eval.py`` (multi-class one-vs-rest
        formulations, averaged over group pairs). SP is the Eq. (8)
        positive-prediction rate gap; EO is the Eq. (9) sum
        ``|TPR gap| + |FPR gap|`` (range [0, 2], NOT a max). BNC/BPC are
        score-based and use the predicted probability ``y_prob`` when
        available.
        """
        y_t = np.asarray(y_true, dtype=int).ravel()
        y_p = np.asarray(y_pred, dtype=int).ravel()
        y_s = np.asarray(y_prob, dtype=float).ravel() if y_prob is not None else y_p.astype(float)

        acc = float(accuracy_score(y_t, y_p))
        f1 = float(f1_score(y_t, y_p, zero_division=0))
        prec = float(precision_score(y_t, y_p, zero_division=0))
        rec = float(recall_score(y_t, y_p, zero_division=0))

        metric_names = [
            "SP", "EO", "EOpp", "CUAE", "OAE", "BNC", "BPC",
            "FDRP", "FORP", "FNRB", "FPRB", "NPVP", "PPVP",
        ]
        metrics: Dict[str, Any] = {"ACC": acc, "F1": f1, "Precision": prec, "Recall": rec}
        for name in metric_names:
            metrics[name] = {}

        all_classes = sorted(set(y_t.tolist()))

        # Calculate subgroup fairness disparities for each protected attribute
        for p_col in O_df.columns:
            prot_series = O_df[p_col].reset_index(drop=True)
            prot_vals = prot_series.values
            groups = sorted(prot_series.unique())

            if len(groups) < 2:
                for name in metric_names:
                    metrics[name][p_col] = 0.0
                continue

            group_masks = {g: (prot_vals == g) for g in groups}
            # Per-group, per-class one-vs-rest confusion-matrix rates
            stats: Dict[Any, Dict[int, Dict[str, float]]] = {}
            for g in groups:
                mask = group_masks[g]
                if mask.sum() == 0:
                    continue
                stats[g] = {
                    c: self._binary_stats(y_t[mask], y_p[mask], c) for c in all_classes
                }

            def rate_gap(stat_key: str) -> float:
                """Baseline pattern: max over classes, averaged over group pairs."""
                pair_vals = []
                for g_a, g_b in combinations(list(stats.keys()), 2):
                    max_diff = 0.0
                    for c in all_classes:
                        if c not in stats[g_a] or c not in stats[g_b]:
                            continue
                        diff = abs(stats[g_a][c][stat_key] - stats[g_b][c][stat_key])
                        max_diff = max(max_diff, diff)
                    pair_vals.append(max_diff)
                return float(np.mean(pair_vals)) if pair_vals else 0.0

            def conditional_gap(keys: Tuple[str, ...]) -> float:
                """Max over the given rates per class (CUAE/EO style), averaged over pairs."""
                pair_vals = []
                for g_a, g_b in combinations(list(stats.keys()), 2):
                    max_diff = 0.0
                    for c in all_classes:
                        if c not in stats[g_a] or c not in stats[g_b]:
                            continue
                        diff = max(
                            abs(stats[g_a][c][k] - stats[g_b][c][k]) for k in keys
                        )
                        max_diff = max(max_diff, diff)
                    pair_vals.append(max_diff)
                return float(np.mean(pair_vals)) if pair_vals else 0.0

            def score_gap(select_true_class: bool) -> float:
                """BNC/BPC: mean predicted-score gap on true-class samples (baseline _calculate_bnc/_calculate_bpc)."""
                pair_vals = []
                for g_a, g_b in combinations(list(stats.keys()), 2):
                    max_diff = 0.0
                    for c in all_classes:
                        mask_a = group_masks[g_a] & (y_t == c)
                        mask_b = group_masks[g_b] & (y_t == c)
                        m_a = float(np.mean(y_s[mask_a])) if mask_a.sum() > 0 else 0.0
                        m_b = float(np.mean(y_s[mask_b])) if mask_b.sum() > 0 else 0.0
                        max_diff = max(max_diff, abs(m_a - m_b))
                    pair_vals.append(max_diff)
                return float(np.mean(pair_vals)) if pair_vals else 0.0

            def eo_gap() -> float:
                """Paper Eq. (9): per one-vs-rest class, |TPR gap| + |FPR gap|
                (binary case: ΔEO = |TPR gap| + |FPR gap|, range [0, 2]);
                max over classes, averaged over group pairs."""
                pair_vals = []
                for g_a, g_b in combinations(list(stats.keys()), 2):
                    max_sum = 0.0
                    for c in all_classes:
                        if c not in stats[g_a] or c not in stats[g_b]:
                            continue
                        tpr_diff = abs(stats[g_a][c]["TPR"] - stats[g_b][c]["TPR"])
                        fpr_diff = abs(stats[g_a][c]["FPR"] - stats[g_b][c]["FPR"])
                        max_sum = max(max_sum, tpr_diff + fpr_diff)
                    pair_vals.append(max_sum)
                return float(np.mean(pair_vals)) if pair_vals else 0.0

            tpr_gap = rate_gap("TPR")
            fpr_gap = rate_gap("FPR")

            metrics["SP"][p_col] = rate_gap("PPOS")  # Eq. (8)
            metrics["EO"][p_col] = eo_gap()  # Eq. (9): |TPR gap| + |FPR gap|, range [0, 2]
            metrics["EOpp"][p_col] = tpr_gap
            metrics["CUAE"][p_col] = conditional_gap(("PPV", "NPV"))
            metrics["OAE"][p_col] = rate_gap("ACC")
            metrics["BNC"][p_col] = score_gap(select_true_class=False)
            metrics["BPC"][p_col] = score_gap(select_true_class=True)
            metrics["FDRP"][p_col] = rate_gap("FDR")
            metrics["FORP"][p_col] = rate_gap("FOR")
            metrics["FNRB"][p_col] = rate_gap("FNR")
            metrics["FPRB"][p_col] = fpr_gap
            metrics["NPVP"][p_col] = rate_gap("NPV")
            metrics["PPVP"][p_col] = rate_gap("PPV")

        return metrics

    def evaluate(
        self,
        X_train: pd.DataFrame,
        Y_train: pd.Series,
        O_train: pd.DataFrame,
        X_test: pd.DataFrame,
        Y_test: pd.Series,
        O_test: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Unified evaluation entry point executing fit on train fold and metric calculation on test fold.
        """
        if X_train.empty or X_test.empty:
            raise ValueError("Evaluation received empty train or test partition")

        y_pred, y_prob = self.fit_and_predict(X_train, Y_train, X_test)
        return self.compute_metrics(Y_test, y_pred, O_test, y_prob=y_prob)

    def calculate_epsilon(
        self,
        X: pd.DataFrame,
        O: pd.DataFrame,
        cate_attrs: Optional[List[str]] = None,
        num_attrs: Optional[List[str]] = None,
        sample_weight: Optional[pd.Series | np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute the paper-level bias concentration d_phi for each feature.

        Pipeline: group-pair feature divergences -> Shapley-truncated (H-order)
        distance matrix with an origin node -> metric MDS embedding -> Euclidean
        distance of each feature to the origin. See ``fairbias.bias_metric``.
        When the config fixes ``mds_fixed_components`` (official mode), the
        embedding dimension is fixed and the stress-elbow search is skipped.
        """
        cate_list = list(self.cate_attrs) if cate_attrs is None else list(cate_attrs)
        num_list = list(self.num_attrs) if num_attrs is None else list(num_attrs)
        cfg = self.config
        if cfg.algorithm_mode == ALGORITHM_MODE_PAPER_FAITHFUL and sample_weight is not None:
            raise ValueError(
                "algorithm_mode='tang2024_paper_faithful' strictly forbids sample_weight. "
                "The original Tang et al. (2024) baseline is strictly unweighted."
            )

        return compute_dphi_matrix(
            X,
            O,
            cate_list,
            num_list,
            h_order=cfg.h_order,
            mds_max_components=cfg.mds_max_components,
            mds_slope_threshold=cfg.mds_slope_threshold,
            random_state=cfg.random_seed,
            num_method=cfg.eval_divergence_num,
            cat_method=cfg.eval_divergence_cat,
            mds_fixed_components=cfg.mds_fixed_components,
            sample_weight=sample_weight,
        )

    def compute_threshold(
        self, epsilon_dict: Dict[str, Dict[str, float]]
    ) -> float:
        """Compute the global epsilon threshold based on config method."""
        all_eps = [
            float(val)
            for group_dict in epsilon_dict.values()
            for val in group_dict.values()
            if not np.isnan(val) and val >= 0
        ]
        if not all_eps:
            return 0.1

        if self.config.adaptive_threshold_method == "kmeans_125":
            return compute_adaptive_threshold_kmeans_125(all_eps)
        else:
            # Legacy ratio method
            return float(np.mean(all_eps) * self.config.threshold_epsilon)


def round_up_125(value: float) -> float:
    """
    Round up a positive float to the nearest number in {1, 2, 5, 10} * 10^k according to the 1-2-5 rule.
    """
    if value <= 0 or np.isnan(value) or np.isinf(value):
        return 1e-4

    exp = int(np.floor(np.log10(value)))
    base = 10.0 ** exp
    mantissa = value / base

    # 1-2-5 step intervals
    if mantissa <= 1.0 + 1e-9:
        step = 1.0
    elif mantissa <= 2.0 + 1e-9:
        step = 2.0
    elif mantissa <= 5.0 + 1e-9:
        step = 5.0
    else:
        step = 10.0

    return float(np.round(step * base, decimals=abs(min(0, exp)) + 4))


def compute_adaptive_threshold_kmeans_125(distances: Sequence[float]) -> float:
    """
    Compute adaptive bias concentration threshold epsilon using 2-clustering (KMeans k=2)
    on 1D feature distances to identify the low-bias group, then applying the 1-2-5 ceiling rule.
    """
    valid_dists = np.array([float(d) for d in distances if not np.isnan(d) and d >= 0])
    if len(valid_dists) == 0:
        return 1e-4

    if len(valid_dists) == 1 or float(np.std(valid_dists)) < 1e-8:
        return round_up_125(float(np.mean(valid_dists)))

    sorted_d = np.sort(valid_dists)
    n = len(sorted_d)

    # 1D optimal 2-clustering via minimum within-cluster sum of squares
    best_wcss = float("inf")
    best_split_idx = 1

    for i in range(1, n):
        c1 = sorted_d[:i]
        c2 = sorted_d[i:]
        wcss = float(np.sum((c1 - np.mean(c1)) ** 2) + np.sum((c2 - np.mean(c2)) ** 2))
        if wcss < best_wcss:
            best_wcss = wcss
            best_split_idx = i

    low_cluster = sorted_d[:best_split_idx]
    mean_low = float(np.mean(low_cluster))

    return round_up_125(mean_low)
