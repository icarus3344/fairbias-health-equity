"""Tests for Gate 11: Smoke Workflow, Calibration, Threshold Freezing, Metrics & Pre-Unlock Manifest."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import unittest

_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd

from meps_fairness.evaluation.calibration import (
    SurveyWeightedPlattCalibrator,
    find_weighted_capacity_threshold,
)
from meps_fairness.evaluation.inference import stratified_psu_bootstrap_inference
from meps_fairness.evaluation.metrics import (
    capacity_metrics,
    fixed_threshold_metrics,
    primary_fairness_endpoint,
    subgroup_audit_metrics,
    weighted_auprc,
    weighted_auroc,
    weighted_brier_score,
    weighted_calibration_stats,
)
from meps_fairness.pipeline import run_pipeline


class TestGate11Smoke(unittest.TestCase):
    """Unit and integration tests for Gate 11 evaluation metrics, calibration, and smoke pipeline."""

    def setUp(self) -> None:
        self.repo_root = pathlib.Path(__file__).resolve().parents[1]
        self.dta_path = self.repo_root / "data/interim/meps/h244/h244.dta"

    def test_weighted_metrics_synthetic(self) -> None:
        """Verify weighted discrimination and calibration calculations on synthetic data."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.4, 0.2, 0.85, 0.15])
        weights = np.array([100, 200, 150, 120, 180, 210, 140, 190, 160, 250], dtype=float)

        auroc = weighted_auroc(y_true, y_prob, weights)
        self.assertGreater(auroc, 0.90)

        auprc = weighted_auprc(y_true, y_prob, weights)
        self.assertGreater(auprc, 0.80)

        brier = weighted_brier_score(y_true, y_prob, weights)
        self.assertLess(brier, 0.20)

        cal_stats = weighted_calibration_stats(y_true, y_prob, weights)
        self.assertIn("intercept", cal_stats)
        self.assertIn("slope", cal_stats)
        self.assertIn("ece", cal_stats)

        cap_res = capacity_metrics(y_true, y_prob, weights, capacity_fractions=(0.10, 0.20))
        self.assertIn("recall_at_10pct", cap_res)
        self.assertIn("precision_at_10pct", cap_res)

    def test_subgroup_suppression_logic(self) -> None:
        """Verify subgroup suppression when n < 100 or positives < 20 and assert None for max_tpr_gap."""
        # Small group of 50 records (under min_n=100)
        y_small = np.zeros(50)
        y_small[:5] = 1
        p_small = np.random.uniform(0, 1, size=50)
        prot_small = np.array(["SmallGroup"] * 50)
        w_small = np.ones(50)

        res = subgroup_audit_metrics(
            y_small, p_small, prot_small, w_small, min_n=100, min_pos=20
        )
        self.assertEqual(res["subgroups"]["SmallGroup"]["status"], "Suppressed (Insufficient Sample/Power)")
        self.assertIsNone(res["subgroups"]["SmallGroup"]["tpr_at_capacity"])
        self.assertIsNone(res["max_tpr_gap"], "max_tpr_gap must be None when fewer than 2 subgroups unsuppressed")
        self.assertEqual(res["unsuppressed_subgroups_count"], 0)

        audit_df = pd.DataFrame({"RACETHX": prot_small, "SEX": [1] * 50})
        pf = primary_fairness_endpoint(y_small, p_small, audit_df, w_small, capacity=0.10)
        self.assertIsNone(pf["primary_fairness_max_tpr_gap"], "primary_fairness_max_tpr_gap must be None")

    def test_platt_calibrator_and_threshold_freezing(self) -> None:
        """Verify survey-weighted Platt calibration and 10% threshold freezing."""
        rng = np.random.RandomState(20260828)
        raw_probs = rng.beta(0.5, 5.0, size=200)
        y = rng.binomial(1, raw_probs)
        w = rng.uniform(10.0, 100.0, size=200)

        calibrator = SurveyWeightedPlattCalibrator(random_state=20260828)
        calibrator.fit(raw_probs, y, sample_weight=w, freeze_capacity_fraction=0.10)

        self.assertTrue(calibrator.is_fitted_)
        self.assertIsNotNone(calibrator.frozen_threshold_10pct_)
        self.assertGreater(calibrator.frozen_threshold_10pct_, 0.0)

        cal_probs = calibrator.predict_proba(raw_probs)
        self.assertEqual(len(cal_probs), 200)
        self.assertTrue((cal_probs >= 0.0).all() and (cal_probs <= 1.0).all())

    def test_smoke_pipeline_end_to_end_integration(self) -> None:
        """Integration test: Run full smoke pipeline and verify pre-unlock manifest and no-clobber run."""
        if not self.dta_path.is_file():
            self.skipTest(f"Missing Panel 26 Stata file: {self.dta_path}")

        output = run_pipeline(
            mode="smoke",
            seed=20260828,
            repo_root=self.repo_root,
            n_bootstraps=5,
            output_root="runs",
        )

        self.assertEqual(output.mode, "smoke")
        self.assertEqual(output.seed, 20260828)
        self.assertLess(output.runtime_seconds, 1800.0)  # Smoke under 30 minutes
        self.assertLess(output.peak_rss_gb, 16.0)        # Peak RSS under 16 GB

        # Check pre-unlock manifest
        manifest = output.pre_unlock_manifest
        self.assertIn("manifest_version", manifest)
        self.assertIn("development_panel", manifest)
        self.assertLess(
            manifest["development_panel"]["positive_events_count"],
            manifest["development_panel"]["minimum_positive_events"],
        )
        self.assertFalse(manifest["development_panel"]["power_threshold_met"])
        self.assertFalse(manifest["holdout_unlock_prerequisites"]["prerequisites_satisfied"])

        # Check holdout lock preserved
        self.assertIn("LOCKED_UNDERPOWERED_STOP_CONDITION", output.panel27_status)

        # Check output files exist
        run_path = pathlib.Path(output.output_directory)
        self.assertTrue((run_path / "pre_unlock_manifest.json").is_file())
        self.assertTrue((run_path / "pipeline_results.json").is_file())

        # Verify calibration evaluation nature
        self.assertEqual(
            output.panel26_metrics["metric_evaluation_nature"],
            "apparent_calibration_fit_diagnostics_not_out_of_sample_validation",
        )

    def test_trapezoid_integration_backward_compatibility(self) -> None:
        """Verify _trapezoid integration helper works and matches expected area."""
        from meps_fairness.evaluation.metrics import _trapezoid
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([1.0, 1.0, 1.0])
        area = _trapezoid(y, x)
        self.assertAlmostEqual(area, 1.0)

        # Triangle area
        y_tri = np.array([0.0, 1.0, 0.0])
        area_tri = _trapezoid(y_tri, x)
        self.assertAlmostEqual(area_tri, 0.5)

    def test_platt_calibrator_numerical_stability_on_extreme_probs(self) -> None:
        """Verify Platt scaling remains numerically stable without separation overflow on extreme logits."""
        extreme_probs = np.array([1e-6, 1e-5, 0.9999, 0.99999, 0.5])
        y = np.array([0, 0, 1, 1, 1])
        w = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

        cal = SurveyWeightedPlattCalibrator(random_state=20260828)
        cal.fit(extreme_probs, y, sample_weight=w)
        cal_probs = cal.predict_proba(extreme_probs)
        self.assertTrue(np.isfinite(cal_probs).all())
        self.assertTrue((cal_probs >= 0.0).all() and (cal_probs <= 1.0).all())

    def test_cli_subprocess_alternate_cwd(self) -> None:
        """Verify run_meps_pipeline.py CLI runs --help from alternate cwd (/tmp)."""
        cli_path = self.repo_root / "scripts/run_meps_pipeline.py"
        res = subprocess.run(
            [sys.executable, str(cli_path), "--help"],
            cwd="/tmp",
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Execute MEPS longitudinal coverage prediction", res.stdout)


if __name__ == "__main__":
    unittest.main()
