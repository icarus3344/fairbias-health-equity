"""Tests for Gate 8: DUID-Grouped Partitioning, Target Stratification & Leakage Audit."""

from __future__ import annotations

import pathlib
import sys
import unittest

_SRC_DIR = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
import pandas as pd

from meps_fairness.data.cohort import extract_meps_cohort
from meps_fairness.data.split import PartitionedCohort, split_panel26_duid_grouped


class TestGate8Split(unittest.TestCase):
    """Unit and integration tests for Gate 8 DUID-grouped split and anti-leakage audit."""

    def setUp(self) -> None:
        self.repo_root = pathlib.Path(__file__).resolve().parents[1]
        self.dta_path = self.repo_root / "data/interim/meps/h244/h244.dta"

    def test_synthetic_duid_grouped_split_properties(self) -> None:
        """Verify DUID-grouped split on synthetic multi-person households."""
        n_households = 50
        persons_per_du = 3
        n_total = n_households * persons_per_du

        duids = []
        for du in range(1001, 1001 + n_households):
            duids.extend([du] * persons_per_du)

        data = {
            "DUPERSID": [f"ID{i}" for i in range(n_total)],
            "DUID": duids,
            "PID": list(range(101, 101 + persons_per_du)) * n_households,
            "PANEL": [26] * n_total,
            "YEARIND": [1] * n_total,
            "ALL5RDS": [1] * n_total,
            "LONGWT": [1000.0 + (i % 10) * 100 for i in range(n_total)],
            "VARSTR": [2001 + (i % 5) for i in range(n_total)],
            "VARPSU": [1 + (i % 2) for i in range(n_total)],
            "AGEY1X": [30 + (i % 20) for i in range(n_total)],
            "RACETHX": [1 + (i % 5) for i in range(n_total)],
            "SEX": [1 + (i % 2) for i in range(n_total)],
            "POVCATY1": [1 + (i % 5) for i in range(n_total)],
            "POVLEVY1": [100.0] * n_total,
            "EDUCYR": [12] * n_total,
            "FAMSZEY1": [3] * n_total,
            "HIDEG": [1] * n_total,
            "MARRY1X": [1] * n_total,
            "MARRY2X": [1] * n_total,
            "MARRYY1X": [1] * n_total,
            "REGION1": [1] * n_total,
            "REGION2": [1] * n_total,
            "REGIONY1": [1] * n_total,
            "FAMINCY1": [50000.0] * n_total,
            "TTLPY1X": [40000.0] * n_total,
            "WAGEPY1X": [35000.0] * n_total,
            "HOUR1": [40] * n_total,
            "HOUR2": [40] * n_total,
            "NUMEMP1": [100] * n_total,
            "NUMEMP2": [100] * n_total,
            "EMPST1": [1] * n_total,
            "EMPST2": [1] * n_total,
            "OFFER1X": [1] * n_total,
            "OFFER2X": [1] * n_total,
            "HELD1X": [1] * n_total,
            "HELD2X": [1] * n_total,
            "CHOIC1": [1] * n_total,
            "CHOIC2": [1] * n_total,
            "SELFCM1": [2] * n_total,
            "SELFCM2": [2] * n_total,
            "UNION1": [2] * n_total,
            "UNION2": [2] * n_total,
            "RTHLTH1": [1] * n_total,
            "RTHLTH2": [1] * n_total,
            "MNHLTH1": [1] * n_total,
            "MNHLTH2": [1] * n_total,
            "HIBPDXY1": [2] * n_total,
            "DIABDXY1_M18": [2] * n_total,
            "ASTHDXY1": [2] * n_total,
            "CHDDXY1": [2] * n_total,
            "ANGIDXY1": [2] * n_total,
            "MIDXY1": [2] * n_total,
            "OHRTDXY1": [2] * n_total,
            "STRKDXY1": [2] * n_total,
            "EMPHDXY1": [2] * n_total,
            "CHBRON1": [2] * n_total,
            "CHOLDXY1": [2] * n_total,
            "CANCERY1": [2] * n_total,
            "ARTHDXY1": [2] * n_total,
            "JTPAIN1_M18": [2] * n_total,
            "ACTLIM1": [2] * n_total,
            "ADLHLP1": [2] * n_total,
            "IADLHP1": [2] * n_total,
            "WLKLIM1": [2] * n_total,
            "SOCLIM1": [2] * n_total,
            "COGLIM1": [2] * n_total,
            "OBTOTVY1": [0] * n_total,
            "OBDRVY1": [0] * n_total,
            "OPTOTVY1": [0] * n_total,
            "ERTOTY1": [0] * n_total,
            "IPDISY1": [0] * n_total,
            "RXTOTY1": [0] * n_total,
            "TOTEXPY1": [0.0] * n_total,
            "TOTSLFY1": [0.0] * n_total,
            "HAVEUS2": [1] * n_total,
            "LOCATN2": [1] * n_total,
            "PROVTY2_M18": [1] * n_total,
            "INSCOVY1": [1] * n_total,
            "PRIEUY1": [1] * n_total,
            "PRINGY1": [2] * n_total,
            "PUBY1X": [2] * n_total,
            "MCAIDY1X": [2] * n_total,
            "MCAREY1X": [2] * n_total,
            "TRICRY1X": [2] * n_total,
            "VAPROGY1": [2] * n_total,
        }

        from meps_fairness.data.cohort import BASELINE_INSURANCE_MONTHS, FOLLOWUP_INSURANCE_MONTHS
        for m in BASELINE_INSURANCE_MONTHS:
            data[m] = [1] * n_total
        for m in FOLLOWUP_INSURANCE_MONTHS:
            data[m] = [1] * n_total

        # Set some positive outcomes
        data["INSJAY2X"][0] = 2
        data["INSJAY2X"][10] = 2
        data["INSJAY2X"][20] = 2

        df = pd.DataFrame(data)
        cohort = extract_meps_cohort(df, panel_number=26, allow_target=True)

        split_result = split_panel26_duid_grouped(cohort, seed=20260828)

        # 1. Total row count preserved
        self.assertEqual(
            split_result.train.record_count + split_result.val.record_count + split_result.cal.record_count,
            cohort.eligible_record_count
        )

        # 2. Zero DUID overlap
        train_du = set(split_result.train.duids)
        val_du = set(split_result.val.duids)
        cal_du = set(split_result.cal.duids)
        self.assertEqual(len(train_du & val_du), 0)
        self.assertEqual(len(train_du & cal_du), 0)
        self.assertEqual(len(val_du & cal_du), 0)

        # 3. Refit partition contains exactly train + val
        self.assertEqual(
            split_result.train_val.record_count,
            split_result.train.record_count + split_result.val.record_count
        )
        self.assertEqual(
            set(split_result.train_val.duids),
            train_du | val_du
        )

        # 4. Deterministic hash invariance
        split_result_2 = split_panel26_duid_grouped(cohort, seed=20260828)
        self.assertEqual(split_result.split_assignment_hash, split_result_2.split_assignment_hash)

        # 5. Different seed yields different hash
        split_result_3 = split_panel26_duid_grouped(cohort, seed=20260829)
        self.assertNotEqual(split_result.split_assignment_hash, split_result_3.split_assignment_hash)

    def test_real_panel26_split_integration(self) -> None:
        """Integration test: DUID split on real Panel 26 data."""
        if not self.dta_path.is_file():
            self.skipTest(f"Missing Panel 26 Stata file: {self.dta_path}")

        df = pd.read_stata(self.dta_path, convert_categoricals=False)
        cohort = extract_meps_cohort(df, panel_number=26, allow_target=True)
        split_result = split_panel26_duid_grouped(cohort, seed=20260828)

        # Check partition ratios (~60/20/20)
        total_n = cohort.eligible_record_count
        train_frac = split_result.train.record_count / total_n
        val_frac = split_result.val.record_count / total_n
        cal_frac = split_result.cal.record_count / total_n

        self.assertAlmostEqual(train_frac, 0.60, delta=0.05)
        self.assertAlmostEqual(val_frac, 0.20, delta=0.05)
        self.assertAlmostEqual(cal_frac, 0.20, delta=0.05)

        # Zero leakage
        split_result.validate_zero_leakage()

        # Both positive and negative outcomes in all partitions
        self.assertGreater(split_result.train.positive_count, 0)
        self.assertGreater(split_result.val.positive_count, 0)
        self.assertGreater(split_result.cal.positive_count, 0)
        self.assertGreater(split_result.train.negative_count, 0)
        self.assertGreater(split_result.val.negative_count, 0)
        self.assertGreater(split_result.cal.negative_count, 0)


if __name__ == "__main__":
    unittest.main()
