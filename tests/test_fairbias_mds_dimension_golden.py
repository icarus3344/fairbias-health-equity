"""Golden contrast test for the MDS dimension selection (Codex round-4
verdict item 4).

The stress-elbow dimension selection in ``fairbias.bias_metric`` previously
carried no supplementary/golden contrast.  The paper's Supplementary
Materials were not accessible (publisher returned HTTP 403; the official
repository contains no supplementary PDF), so the OFFICIAL repository's
published UCI Adult distance matrix (step 0, before any mitigation
transform) is used as the golden reference (see
``tests/fixtures/official_uciadult/PROVENANCE.json``).

Key contrast (documented, not asserted as agreement):

- The OFFICIAL implementation (MachineClassifer_BiasMitigation.py) draws
  the stress elbow plot for dims 1..8 for VISUAL inspection only and then
  computes the actual embedding with the dimension FIXED AT 2.
- The fairbias implementation (``_find_optimal_mds_components``) selects
  the dimension automatically: on the official matrix it returns 3 (the
  first normalized stress delta below the slope threshold occurs between
  dim 2 and dim 3, and the implementation returns the dimension AT which
  the curve has flattened).

This is a RECORDED DEVIATION with baseline semantics.  The tests below
pin the deterministic behavior of our selection on the golden matrix and
make the deviation explicit so it can never be silently presented as
paper-aligned.  Whether to align the elbow rule with the official
fixed-dim-2 behavior is a supervisor decision, not a worker-level change.
"""

import hashlib
import json
import os
import unittest

import numpy as np
import pandas as pd

from fairbias.bias_metric import _find_optimal_mds_components

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures", "official_uciadult"
)
FIXTURE_CSV = os.path.join(FIXTURE_DIR, "distance_matrix_step_0.csv")
FIXTURE_PROVENANCE = os.path.join(FIXTURE_DIR, "PROVENANCE.json")

# Integrity pin: the fixture must be byte-identical to the official
# repository file recorded in PROVENANCE.json.
EXPECTED_SHA256 = "5f9fd4f81766ce724bd9866bd139503d252f68c9bc7e5065ff34ace0f57228e1"

# Golden record: our stress-elbow rule on the official step-0 matrix.
# The official implementation instead FIXES the embedding dimension at 2.
GOLDEN_FAIRBIAS_DIM = 3
OFFICIAL_FIXED_DIM = 2


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestMDSDimensionGoldenContrast(unittest.TestCase):
    """Contrast fairbias' automatic elbow selection with the official
    implementation's fixed dim=2 on the official UCI Adult distance
    matrix."""

    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_CSV, encoding="utf-8") as f:
            cls.df = pd.read_csv(f, index_col=0)
        cls.dist = cls.df.values.astype(float)

    def test_fixture_integrity_matches_provenance_manifest(self):
        # Atomic-download protocol: checksum validation before use.
        self.assertEqual(_sha256_of_file(FIXTURE_CSV), EXPECTED_SHA256)
        with open(FIXTURE_PROVENANCE, encoding="utf-8") as f:
            provenance = json.load(f)
        self.assertEqual(
            provenance["download"]["sha256"], EXPECTED_SHA256,
            "fixture checksum must match the provenance manifest",
        )

    def test_fixture_is_a_valid_distance_matrix(self):
        # 13 features + the origin node = 14 nodes, symmetric, zero diagonal.
        self.assertEqual(self.dist.shape, (14, 14))
        self.assertIn("origin", list(self.df.index))
        self.assertIn("origin", list(self.df.columns))
        self.assertTrue(np.allclose(self.dist, self.dist.T))
        self.assertTrue(np.allclose(np.diag(self.dist), 0.0))
        self.assertFalse(np.isnan(self.dist).any())
        self.assertFalse(np.isinf(self.dist).any())

    def test_fairbias_elbow_selection_on_official_matrix_is_deterministic(self):
        # Golden snapshot: the fairbias rule selects dim 3 on the official
        # step-0 matrix, deterministically across seeds.
        for random_state in (0, 1, 7):
            selected = _find_optimal_mds_components(
                self.dist,
                max_components=15,
                slope_threshold=0.01,
                random_state=random_state,
            )
            self.assertEqual(selected, GOLDEN_FAIRBIAS_DIM)

    def test_recorded_deviation_from_official_fixed_dimension(self):
        # Explicit, auditable deviation record: our automatic elbow
        # selection (dim 3) does NOT reproduce the official
        # implementation's fixed dim=2 embedding on the same matrix.
        # This assertion pins the DIFFERENCE so the deviation is a tested
        # fact rather than an undocumented claim of alignment.
        selected = _find_optimal_mds_components(
            self.dist, max_components=15, slope_threshold=0.01, random_state=0,
        )
        self.assertNotEqual(
            selected, OFFICIAL_FIXED_DIM,
            "If this fails, the elbow rule now reproduces the official "
            "fixed dimension — update PROVENANCE.json and the round-4 "
            "report deviation record accordingly.",
        )


if __name__ == "__main__":
    unittest.main()
