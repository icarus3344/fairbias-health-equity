"""Tests for Gate 10: Predictive Modeling, Exploratory Group-Aware Centering & Survey-Weighted Extension."""

from __future__ import annotations

import pathlib
import sys
import unittest

_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd

from meps_fairness.data.cohort import (
    ALL_BASELINE_PREDICTOR_COLUMNS,
    extract_meps_cohort,
)
from meps_fairness.data.preprocess import MEPSPreprocessor
from meps_fairness.data.split import split_panel26_duid_grouped
from meps_fairness.models.baseline import (
    WeightedGradientBoostingClassifier,
    WeightedLogisticClassifier,
    WeightedRandomForestClassifier,
)
from meps_fairness.models.mitigation import (
    ExploratoryGroupAwareCenteringMitigation,
    ExploratorySurveyWeightedCenteringExtension,
    compute_bias_concentration_epsilon,
)


class TestGate10Models(unittest.TestCase):
    """Unit and integration tests for Gate 10 predictive models and fairness mitigation."""

    def setUp(self) -> None:
        self.repo_root = pathlib.Path(__file__).resolve().parents[1]
        self.dta_path = self.repo_root / "data/interim/meps/h244/h244.dta"

    def test_weighted_logistic_classifier_synthetic(self) -> None:
        """Verify weighted logistic classifier on synthetic data."""
        rng = np.random.RandomState(20260828)
        X = pd.DataFrame(rng.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(rng.binomial(1, 0.3, size=100))
        w = pd.Series(rng.uniform(10.0, 100.0, size=100))

        clf = WeightedLogisticClassifier(random_state=20260828)
        clf.fit(X, y, sample_weight=w)

        probs = clf.predict_proba(X)
        self.assertEqual(len(probs), 100)
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

        preds = clf.predict(X, threshold=0.5)
        self.assertEqual(len(preds), 100)
        self.assertTrue(set(np.unique(preds)).issubset({0.0, 1.0}))

    def test_bias_concentration_epsilon_calculation(self) -> None:
        """Verify O(N*D) linear epsilon calculation."""
        df_X = pd.DataFrame({
            "f0": [10.0, 10.0, 20.0, 20.0],
            "f1": [5.0, 5.0, 5.0, 5.0],
        })
        prot = pd.Series([1, 1, 2, 2])
        w = pd.Series([1.0, 1.0, 1.0, 1.0])

        eps = compute_bias_concentration_epsilon(df_X, prot, sample_weight=w)
        self.assertAlmostEqual(eps["f0"], 10.0)
        self.assertAlmostEqual(eps["f1"], 0.0)

    def test_exploratory_centering_and_survey_weighted_mitigation(self) -> None:
        """Verify exploratory group-aware centering and survey-weighted extension reduce disparity."""
        rng = np.random.RandomState(20260828)
        n = 200
        prot = pd.Series(rng.choice([1, 2], size=n))
        f0 = rng.randn(n) + (prot.values * 2.0)
        f1 = rng.randn(n)
        X = pd.DataFrame({"f0": f0, "f1": f1})
        y = pd.Series(rng.binomial(1, 0.2, size=n))
        w = pd.Series(rng.uniform(10.0, 50.0, size=n))

        initial_eps = compute_bias_concentration_epsilon(X, prot, w)["f0"]

        # Exploratory group-aware centering mitigation
        mit = ExploratoryGroupAwareCenteringMitigation(shrinkage_intensity=0.8, random_state=20260828)
        mit.fit(X, y, prot, sample_weight=w)
        X_mit = mit.transform(X, prot)
        mit_eps = compute_bias_concentration_epsilon(X_mit, prot, w)["f0"]

        self.assertLess(mit_eps, initial_eps)

        # Exploratory survey-weighted extension
        ext = ExploratorySurveyWeightedCenteringExtension(shrinkage_intensity=0.8, random_state=20260828)
        ext.fit(X, y, prot, sample_weight=w)
        X_ext = ext.transform(X, prot)
        ext_eps = compute_bias_concentration_epsilon(X_ext, prot, w)["f0"]

        self.assertLess(ext_eps, initial_eps)

        # Predict proba
        probs = ext.predict_proba(X, prot)
        self.assertEqual(len(probs), n)
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_mitigation_inference_contract_and_predictor_isolation(self) -> None:
        """Verify inference contracts: baseline is group-agnostic, exploratory mitigation requires protected series."""
        self.assertNotIn("RACETHX", ALL_BASELINE_PREDICTOR_COLUMNS)
        self.assertNotIn("SEX", ALL_BASELINE_PREDICTOR_COLUMNS)

        # Baseline classifier accepts only X
        clf = WeightedLogisticClassifier(random_state=20260828)
        X_syn = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [0.5, 1.5, 2.5]})
        y_syn = pd.Series([0, 1, 0])
        clf.fit(X_syn, y_syn)
        p_base = clf.predict_proba(X_syn)
        self.assertEqual(len(p_base), 3)

        # Exploratory mitigation requires protected series for group centering
        mit = ExploratoryGroupAwareCenteringMitigation(random_state=20260828)
        prot_syn = pd.Series([1, 2, 1])
        mit.fit(X_syn, y_syn, prot_syn)
        p_mit = mit.predict_proba(X_syn, prot_syn)
        self.assertEqual(len(p_mit), 3)

    def test_real_panel26_model_training_integration(self) -> None:
        """Integration test: Train models on preprocessed Panel 26 training data."""
        if not self.dta_path.is_file():
            self.skipTest(f"Missing Panel 26 Stata file: {self.dta_path}")

        df = pd.read_stata(self.dta_path, convert_categoricals=False)
        cohort = extract_meps_cohort(df, panel_number=26, allow_target=True)
        split = split_panel26_duid_grouped(cohort, seed=20260828)

        prep = MEPSPreprocessor()
        X_train_trans = prep.fit_transform(split.train.X)
        X_val_trans = prep.transform(split.val.X)

        # Train Baseline Weighted Logistic (group-agnostic)
        clf = WeightedLogisticClassifier(random_state=20260828)
        clf.fit(X_train_trans, split.train.y, sample_weight=split.train.design["LONGWT"])
        val_probs = clf.predict_proba(X_val_trans)

        self.assertEqual(len(val_probs), split.val.record_count)
        self.assertTrue((val_probs >= 0.0).all() and (val_probs <= 1.0).all())

        # Train the exploratory survey-weighted group-aware centering heuristic.
        mit_ext = ExploratorySurveyWeightedCenteringExtension(shrinkage_intensity=0.5, random_state=20260828)
        mit_ext.fit(
            X_train_trans,
            split.train.y,
            protected_series=split.train.audit["RACETHX"],
            sample_weight=split.train.design["LONGWT"],
        )
        mit_val_probs = mit_ext.predict_proba(X_val_trans, protected_series=split.val.audit["RACETHX"])
        self.assertEqual(len(mit_val_probs), split.val.record_count)


if __name__ == "__main__":
    unittest.main()
