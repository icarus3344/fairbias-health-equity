"""End-to-end integration tests for FairBias pipeline execution, three-way
partition boundaries, and validation-based Pareto checkpointing."""

import json
import os
import shutil
import unittest
import numpy as np
import pandas as pd

from fairbias.config import FairBiasConfig
from fairbias.pipeline import run_fairbias_pipeline


class TestFairBiasPipeline(unittest.TestCase):
    """Verifies end-to-end pipeline execution, validation-based checkpoint
    selection, single test evaluation, and output generation."""

    def setUp(self):
        self.test_output_dir = "runs/test_pipeline_tmp"
        os.makedirs(self.test_output_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_pipeline_compas_end_to_end(self):
        config = FairBiasConfig.compas_default(
            max_iterations=3,
            output_dir=self.test_output_dir,
            random_seed=0,
        )
        res = run_fairbias_pipeline(config)

        # Check result object fields
        self.assertIsNotNone(res.run_id)
        self.assertIn("ACC", res.initial_metrics)
        self.assertIn("ACC", res.final_metrics)
        self.assertGreaterEqual(res.best_iteration, 0)
        self.assertLessEqual(res.best_iteration, 3)
        self.assertTrue(os.path.exists(res.output_file))

        # Check that iterations history was logged
        self.assertGreater(len(res.iterations), 0)
        for it in res.iterations:
            self.assertIn("metrics", it)
            self.assertIn("epsilon_values", it)
            self.assertIn("max_epsilon", it)
            self.assertIn("avg_epsilon", it)
            self.assertEqual(it["metrics_partition"], "validation")

        # Manifest must document the partition boundaries
        with open(res.output_file, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["selection_partition"], "validation")
        self.assertEqual(payload["final_evaluation_partition"], "test")
        self.assertEqual(payload["final_results"]["metrics_partition"], "test")
        fractions = payload["split"]["fractions"]
        self.assertAlmostEqual(fractions["train"], 0.64, places=9)
        self.assertAlmostEqual(fractions["validation"], 0.16, places=9)
        self.assertAlmostEqual(fractions["test"], 0.20, places=9)
        row_counts = payload["split"]["row_counts"]
        total = sum(row_counts.values())
        self.assertEqual(total, len(pd.read_csv(config.dataset_path)))
        for key in ("train", "validation", "test"):
            self.assertGreater(row_counts[key], 0)

    def test_pareto_checkpoint_selection_rule(self):
        # Verify that the Pareto rule prefers lower EO/SP disparity on the
        # VALIDATION partition under the accuracy constraint.
        config = FairBiasConfig.compas_default(
            max_iterations=5,
            accuracy_tolerance_tau=0.01,
            selection_metric="EO",
            output_dir=self.test_output_dir,
            random_seed=0,
        )
        res = run_fairbias_pipeline(config)

        # The selected best iteration must satisfy the accuracy constraint on
        # the SAME partition used for selection (validation).
        if res.best_iteration == 0:
            best_metrics = res.initial_metrics
        else:
            best_metrics = next(
                it["metrics"] for it in res.iterations if it["iteration"] == res.best_iteration
            )
        self.assertGreaterEqual(
            best_metrics["ACC"], res.initial_metrics["ACC"] - 0.01
        )

        # Selection reason string should be informative and validation-based
        self.assertIn("Pareto", res.best_selection_reason)
        self.assertIn("VALIDATION", res.best_selection_reason)

    def test_multi_seed_determinism(self):
        cfg_0_a = FairBiasConfig.compas_default(max_iterations=2, random_seed=42, output_dir=self.test_output_dir)
        cfg_0_b = FairBiasConfig.compas_default(max_iterations=2, random_seed=42, output_dir=self.test_output_dir)

        res_a = run_fairbias_pipeline(cfg_0_a)
        res_b = run_fairbias_pipeline(cfg_0_b)

        self.assertAlmostEqual(res_a.initial_metrics["ACC"], res_b.initial_metrics["ACC"])
        self.assertAlmostEqual(res_a.final_metrics["ACC"], res_b.final_metrics["ACC"])
        self.assertEqual(res_a.best_iteration, res_b.best_iteration)
        self.assertEqual(res_a.epsilon_threshold, res_b.epsilon_threshold)


if __name__ == "__main__":
    unittest.main()
