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
        self.assertEqual(res.algorithm_mode, "engineering_bounded")
        self.assertIn("ACC", res.initial_metrics)
        self.assertIn("ACC", res.greedy_terminal_metrics)
        self.assertIn("ACC", res.pareto_engineering_metrics)
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

        # Manifest must document the partition boundaries — BOTH the
        # configured fractions and the fractions OBSERVED from the actual
        # row counts (a previous bug echoed the configured numbers while the
        # real split was 80/16/4).
        with open(res.output_file, encoding="utf-8") as f:
            payload = json.load(f)
        self.assertEqual(payload["selection_partition"], "validation")
        self.assertEqual(payload["final_evaluation_partition"], "test")
        self.assertEqual(payload["algorithm_mode"], "engineering_bounded")
        # The terminal states must be reported SEPARATELY — the merged
        # single ``final_results`` key must no longer exist, and the
        # engineering greedy terminal state must NOT carry a paper_strict
        # name (round-4.1 rename: configured_greedy_terminal).
        self.assertNotIn("final_results", payload)
        self.assertNotIn("final_results_paper_strict", payload)
        self.assertIn("final_results_configured_greedy_terminal", payload)
        self.assertIn("final_results_pareto_engineering", payload)
        self.assertEqual(
            payload["final_states"],
            ["configured_greedy_terminal", "pareto_engineering"],
        )
        self.assertEqual(
            payload["final_results_configured_greedy_terminal"]["metrics_partition"],
            "test",
        )
        self.assertEqual(
            payload["final_results_pareto_engineering"]["metrics_partition"], "test"
        )
        configured = payload["split"]["configured_fractions"]
        self.assertAlmostEqual(configured["train"], 0.64, places=9)
        self.assertAlmostEqual(configured["validation"], 0.16, places=9)
        self.assertAlmostEqual(configured["test"], 0.20, places=9)
        row_counts = payload["split"]["row_counts"]
        total = sum(row_counts.values())
        self.assertEqual(total, len(pd.read_csv(config.dataset_path)))
        for key in ("train", "validation", "test"):
            self.assertGreater(row_counts[key], 0)

        # The OBSERVED fractions (row_counts / total) must match the paper
        # 64/16/20 split — this is the assertion the old manifest-only check
        # could not make.
        observed = payload["split"]["observed_fractions"]
        for part in ("train", "validation", "test"):
            self.assertAlmostEqual(
                observed[part], row_counts[part] / total, places=9,
                msg="observed_fractions must be derived from the real row counts",
            )
            self.assertAlmostEqual(
                observed[part], configured[part], delta=0.05,
                msg=(
                    f"actual {part} fraction {observed[part]:.4f} deviates from "
                    f"the paper 64/16/20 split (configured {configured[part]:.4f})"
                ),
            )

        # Strict-paper failure semantics must be reported explicitly
        self.assertIn("failed_attribute_mode", payload)
        self.assertIn("mitigation_non_convergence", payload)

        # Termination semantics must be recorded (converged / reason /
        # terminal state metrics) — a budget-exhausted run must NOT be
        # readable as converged.
        termination = payload["termination"]
        for key in (
            "converged", "termination_reason", "terminal_iteration",
            "terminal_max_dphi", "epsilon_threshold", "algorithm_mode",
        ):
            self.assertIn(key, termination)
        self.assertEqual(termination["algorithm_mode"], "engineering_bounded")
        self.assertIn(
            termination["termination_reason"],
            (
                "epsilon_reached", "candidate_grid_exhausted",
                "iteration_budget_exhausted", "mitigation_disabled",
                "accuracy_threshold_reached",
            ),
        )

    def test_greedy_terminal_state_is_greedy_terminal_state(self):
        # The greedy terminal state must be the greedy loop's TERMINATION
        # state (last accepted transform), never a validation-Pareto
        # rollback of it.
        config = FairBiasConfig.compas_default(
            max_iterations=5,
            output_dir=self.test_output_dir,
            random_seed=0,
        )
        res = run_fairbias_pipeline(config)

        self.assertGreater(len(res.iterations), 0)
        last_iteration = res.iterations[-1]
        self.assertEqual(
            res.greedy_terminal_changed_dict,
            last_iteration["changed_dict"],
            "greedy terminal state must equal the last ACCEPTED iteration state",
        )
        self.assertEqual(
            res.termination["terminal_iteration"],
            last_iteration["iteration"],
        )
        self.assertAlmostEqual(
            res.termination["terminal_max_dphi"],
            last_iteration["max_epsilon"],
            places=12,
        )
        # The Pareto checkpoint may roll back to an earlier iteration; it
        # must never be AHEAD of the greedy terminal iteration.
        self.assertLessEqual(res.best_iteration, res.termination["terminal_iteration"])

    def test_iteration_budget_exhausted_is_recorded_as_not_converged(self):
        # COMPAS seed 0 needs 4 greedy iterations to enter the epsilon
        # ball; capping the budget at 3 must terminate with
        # iteration_budget_exhausted and converged=False (never null).
        config = FairBiasConfig.compas_default(
            max_iterations=3,
            output_dir=self.test_output_dir,
            random_seed=0,
        )
        res = run_fairbias_pipeline(config)

        self.assertEqual(res.termination["termination_reason"], "iteration_budget_exhausted")
        self.assertFalse(res.termination["converged"])
        self.assertEqual(res.termination["terminal_iteration"], 3)
        self.assertGreater(
            res.termination["terminal_max_dphi"],
            res.termination["epsilon_threshold"],
            "budget-exhausted run must still be OUTSIDE the epsilon ball",
        )
        # The greedy terminal state still reports the budget-exhausted
        # terminal state (iteration 3), NOT the converged iteration 4 that
        # never ran.
        self.assertEqual(res.greedy_terminal_changed_dict, res.iterations[-1]["changed_dict"])

    def test_epsilon_reached_is_recorded_as_converged(self):
        # With a sufficient budget the COMPAS run enters the epsilon ball;
        # the termination record must then say so explicitly.
        config = FairBiasConfig.compas_default(
            max_iterations=10,
            output_dir=self.test_output_dir,
            random_seed=0,
        )
        res = run_fairbias_pipeline(config)

        if res.termination["terminal_max_dphi"] <= res.termination["epsilon_threshold"]:
            self.assertEqual(res.termination["termination_reason"], "epsilon_reached")
            self.assertTrue(res.termination["converged"])
        else:
            # If the configured grid could not finish, it must be named as
            # a budget/grid outcome, never silently "converged".
            self.assertFalse(res.termination["converged"])
            self.assertIn(
                res.termination["termination_reason"],
                ("candidate_grid_exhausted", "iteration_budget_exhausted"),
            )

    def test_mitigation_disabled_termination_reason(self):
        config = FairBiasConfig.compas_default(
            max_iterations=3,
            use_bias_mitigation=False,
            use_accuracy_enhancement=False,
            output_dir=self.test_output_dir,
            random_seed=0,
        )
        res = run_fairbias_pipeline(config)

        self.assertEqual(res.termination["termination_reason"], "mitigation_disabled")
        self.assertIsNone(res.termination["converged"])
        self.assertEqual(res.termination["terminal_iteration"], 0)
        self.assertEqual(res.greedy_terminal_changed_dict, {})
        self.assertEqual(len(res.iterations), 0)

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
        self.assertAlmostEqual(res_a.greedy_terminal_metrics["ACC"], res_b.greedy_terminal_metrics["ACC"])
        self.assertAlmostEqual(
            res_a.pareto_engineering_metrics["ACC"],
            res_b.pareto_engineering_metrics["ACC"],
        )
        self.assertEqual(res_a.best_iteration, res_b.best_iteration)
        self.assertEqual(res_a.epsilon_threshold, res_b.epsilon_threshold)
        self.assertEqual(res_a.termination, res_b.termination)


if __name__ == "__main__":
    unittest.main()
