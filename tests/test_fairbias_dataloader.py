"""Unit tests for FairDataLoader schema semantics, train-only encoder
fitting, unseen-category handling, and column exclusion."""

import unittest
import numpy as np
import pandas as pd

from fairbias.config import FairBiasConfig
from fairbias.data import FairDataLoader, DATASET_SCHEMAS


class TestFairDataLoader(unittest.TestCase):
    """Verifies schema definitions, target integrity, and leakage-free
    (train-fitted) encoding in FairDataLoader."""

    def test_schema_definitions_exist(self):
        self.assertIn("compas", DATASET_SCHEMAS)
        self.assertIn("credit", DATASET_SCHEMAS)
        compas_schema = DATASET_SCHEMAS["compas"]
        self.assertEqual(compas_schema["target"], "two_year_recid")
        self.assertEqual(compas_schema["protected"], ["sex"])
        self.assertIn("compas_screening_date", compas_schema["excluded"])
        self.assertIn("dob", compas_schema["excluded"])

    def test_compas_data_loading_and_exclusion(self):
        config = FairBiasConfig.compas_default()
        loader = FairDataLoader(config)
        X, Y, O, categorical_cols, numerical_cols = loader.prepare_data()

        # Check shapes and types
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(Y))
        self.assertEqual(len(X), len(O))

        # Check excluded columns are absent from X
        for excluded in loader.excluded_columns:
            self.assertNotIn(excluded, X.columns)

        # Check Target Y is binary 0/1 without NaNs (numeric binary passthrough)
        self.assertEqual(set(Y.unique()), {0, 1})
        self.assertFalse(Y.isna().any())

        # Check Protected Attribute O is present and absent from X
        self.assertIn("sex", O.columns)
        self.assertNotIn("sex", X.columns)  # Protected attribute must not be in X

        # Raw (unencoded) frame: string categoricals must NOT be integer codes yet
        self.assertFalse(np.issubdtype(X["race"].dtype, np.integer))

        # Fit encoders on the TRAINING-like partition only, then transform
        n = len(X)
        train_idx = X.index[: int(n * 0.8)]
        eval_idx = X.index[int(n * 0.8):]
        loader.fit_encoders(X.loc[train_idx], Y.loc[train_idx], O.loc[train_idx])
        X_tr, Y_tr, O_tr = loader.transform_partition(
            X.loc[train_idx], Y.loc[train_idx], O.loc[train_idx]
        )
        X_ev, Y_ev, O_ev = loader.transform_partition(
            X.loc[eval_idx], Y.loc[eval_idx], O.loc[eval_idx]
        )

        # Encoded categorical columns have integer codes on both partitions
        for cat_col in categorical_cols:
            self.assertTrue(np.issubdtype(X_tr[cat_col].dtype, np.integer), cat_col)
            self.assertTrue(np.issubdtype(X_ev[cat_col].dtype, np.integer), cat_col)

        # Encoding mappings are recorded at fit time (train categories only)
        self.assertIn("race", loader.encoding_mappings)
        self.assertIn("c_charge_degree", loader.encoding_mappings)

        # Encoded target and protected attribute stay binary 0/1
        self.assertEqual(set(Y_tr.unique()) <= {0, 1}, True)
        self.assertEqual(set(Y_ev.unique()) <= {0, 1}, True)

    def test_unseen_categories_get_new_deterministic_codes(self):
        # Split BEFORE encoding: every category absent from the training data
        # must map to the FIXED unseen-sentinel code defined at train fit time
        # (len(train classes)) — the SAME code in every partition, regardless
        # of which categories each evaluation partition happens to contain.
        df_synth = pd.DataFrame({
            "age": [25, 30, 45, 50, 60, 35],
            "gender": ["M", "F", "F", "M", "F", "M"],
            "income_cat": ["low", "high", "medium", "low", "high", "medium"],
            "target": [0, 1, 1, 0, 1, 0],
        })
        cfg = FairBiasConfig(
            dataset_name="synthetic",
            label_Y="target",
            label_O=("gender",),
        )
        loader = FairDataLoader(cfg)
        X, Y, O, cats, nums = loader.prepare_data(df_synth)

        self.assertEqual(len(X), 6)
        self.assertIn("income_cat", cats)
        self.assertIn("age", nums)
        self.assertNotIn("target", X.columns)
        self.assertNotIn("gender", X.columns)

        # Train sees only {low, high}; eval introduces the unseen "medium"
        train_rows = [0, 1, 3, 4]   # income_cat: low, high, low, high
        eval_rows = [2, 5]          # income_cat: medium, medium
        X_train = X.iloc[train_rows]
        X_eval = X.iloc[eval_rows]
        loader.fit_encoders(X_train, Y.iloc[train_rows], O.iloc[train_rows])
        X_tr, _, _ = loader.transform_partition(X_train, Y.iloc[train_rows], O.iloc[train_rows])
        X_ev, _, _ = loader.transform_partition(X_eval, Y.iloc[eval_rows], O.iloc[eval_rows])

        train_codes = sorted(set(X_tr["income_cat"]))
        self.assertEqual(train_codes, [0, 1])  # only train categories encoded
        # The sentinel is fixed at fit time: len(train classes) == 2
        self.assertEqual(loader.unseen_sentinels["income_cat"], 2)
        # Unseen category maps to the fixed sentinel (not a partition-specific
        # newly allocated code)
        self.assertEqual(set(X_ev["income_cat"]), {2})

    def test_unseen_sentinel_is_shared_across_partitions(self):
        # Two DIFFERENT unseen categories appearing in validation vs test
        # must receive the SAME sentinel code: the encoding space never
        # depends on the contents of an evaluation partition.
        df_synth = pd.DataFrame({
            "gender": ["M", "F"] * 5,
            "income_cat": (
                ["low", "high"] * 3          # train rows (0..5)
                + ["medium", "medium"]        # validation rows (6..7)
                + ["extreme", "other"]        # test rows (8..9)
            ),
            "target": [0, 1] * 5,
        })
        cfg = FairBiasConfig(
            dataset_name="synthetic",
            label_Y="target",
            label_O=("gender",),
        )
        loader = FairDataLoader(cfg)
        X, Y, O, cats, nums = loader.prepare_data(df_synth)

        n_train, n_val = 6, 2
        X_tr = X.iloc[:n_train]
        X_va = X.iloc[n_train:n_train + n_val]
        X_te = X.iloc[n_train + n_val:]
        loader.fit_encoders(X_tr, Y.iloc[:n_train], O.iloc[:n_train])
        X_tr_t, _, _ = loader.transform_partition(X_tr, Y.iloc[:n_train], O.iloc[:n_train])
        X_va_t, _, _ = loader.transform_partition(X_va, Y.iloc[n_train:n_train + n_val], O.iloc[n_train:n_train + n_val])
        X_te_t, _, _ = loader.transform_partition(X_te, Y.iloc[n_train + n_val:], O.iloc[n_train + n_val:])

        sentinel = loader.unseen_sentinels["income_cat"]
        self.assertEqual(sentinel, 2)
        # train encoding contains no sentinel
        self.assertNotIn(sentinel, set(X_tr_t["income_cat"]))
        # validation's "medium" and test's "extreme"/"other" all share the
        # single fixed sentinel — identical encoding space in both partitions
        self.assertEqual(set(X_va_t["income_cat"]), {sentinel})
        self.assertEqual(set(X_te_t["income_cat"]), {sentinel})

    def test_continuous_target_binarized_on_train_median(self):
        df_synth = pd.DataFrame({
            "gender": ["M", "F", "F", "M", "F", "M"],
            "income": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "target": [1.5, 2.5, 3.5, 10.5, 11.5, 12.5],
        })
        cfg = FairBiasConfig(
            dataset_name="synthetic",
            label_Y="target",
            label_O=("gender",),
        )
        loader = FairDataLoader(cfg)
        X, Y, O, cats, nums = loader.prepare_data(df_synth)

        # Train (first 3 rows) median = 2.5 -> threshold fit on train only
        loader.fit_encoders(X.iloc[:3], Y.iloc[:3], O.iloc[:3])
        _, Y_tr, _ = loader.transform_partition(X.iloc[:3], Y.iloc[:3], O.iloc[:3])
        _, Y_ev, _ = loader.transform_partition(X.iloc[3:], Y.iloc[3:], O.iloc[3:])

        self.assertEqual(list(Y_tr), [0, 0, 1])   # 1.5, 2.5, 3.5 vs median 2.5
        self.assertEqual(list(Y_ev), [1, 1, 1])   # all above the train median


if __name__ == "__main__":
    unittest.main()
