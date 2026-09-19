"""Small synthetic audit; no NHIS records, training runs or model selection.

Run with PYTHONPATH=src and --output pointing to a new JSON evidence file.
These counterexamples demonstrate limitations, not prevalence in NHIS.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fairbias.bias_metric import (
    compute_dphi_matrix,
    compute_pairwise_divergences,
    compute_shapley_distance_matrix,
)
from fairbias.mitigation import FairBiasMitigation
from fairbias.transform import FairTransform, calculate_nmi_dict
from nhis_fairbias.benchmark.metrics import compute_survey_fairness_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Four equally sized intersection groups, both labels in every group.
    cells = np.array(list(itertools.product((0, 1), (0, 1), (0, 1))))
    a, b, y = cells.T
    q = np.bitwise_xor(a, b)
    intersection = 2 * a + b
    weights = np.ones(len(y))

    def eo(group, expected):
        return compute_survey_fairness_metrics(
            y, q, group, weights, expected_groups=expected
        )["eo_gap"]

    separate_eo = [eo(a, [0, 1]), eo(b, [0, 1])]
    intersection_eo = eo(intersection, [0, 1, 2, 3])
    assert separate_eo == [0.0, 0.0] and intersection_eo == 1.0

    # Marginal protected columns do not create intersection groups implicitly.
    x_proxy = pd.DataFrame({"proxy": q})
    o = pd.DataFrame({"A": a, "B": b})
    marginal_geometry = compute_dphi_matrix(
        x_proxy, o, ["proxy"], [], multigroup_aggregation="author_max_pair"
    )
    assert all(v == 0.0 for d in marginal_geometry.values() for v in d.values())
    composite_divergence = compute_pairwise_divergences(
        x_proxy, pd.Series(intersection), ["proxy"], []
    )
    assert composite_divergence.shape[1] == 6
    assert float(composite_divergence.to_numpy().max()) == 1.0

    # A separate limitation: joint information in X can escape univariate g_m.
    x_joint = pd.DataFrame({"x1": a, "x2": b})
    dphi = compute_dphi_matrix(
        x_joint, pd.DataFrame({"protected": q}), ["x1", "x2"], [],
        multigroup_aggregation="author_max_pair",
    )
    divergence = compute_pairwise_divergences(x_joint, pd.Series(q), ["x1", "x2"], [])
    distance, _ = compute_shapley_distance_matrix(
        divergence, ["x1", "x2"], multigroup_aggregation="author_max_pair"
    )
    assert not np.any(distance) and all(v == 0.0 for v in dphi["protected"].values())
    assert eo(q, [0, 1]) == 1.0

    # A missing registered group must not disappear from the EO denominator.
    missing_group_eo = eo(intersection, [0, 1, 2, 3, 4])
    assert np.isnan(missing_group_eo)

    # Current normalized information-loss gate uses a threshold of 100.
    predictive = pd.DataFrame({"feature": y})
    dropped = predictive.drop(columns=["feature"])
    before = calculate_nmi_dict(predictive, pd.Series(y))
    after = calculate_nmi_dict(dropped, pd.Series(y))
    engine = FairBiasMitigation(None, FairTransform(), ["A"], ["feature"], [], phi_threshold=100.0)
    passes = engine._nmi_gate_ok(dropped, pd.Series(y), before, "feature")
    assert before["feature"] > 0.99 and after.get("feature", 0.0) == 0.0 and passes

    report = {
        "scope": "synthetic counterexamples only; not NHIS effect estimates or complete adapter validation",
        "marginal_EO": separate_eo,
        "intersection_EO": intersection_eo,
        "marginal_geometry": marginal_geometry,
        "four_group_pair_count": int(composite_divergence.shape[1]),
        "composite_max_feature_divergence": float(composite_divergence.to_numpy().max()),
        "predictor_interaction_dphi": dphi,
        "predictor_interaction_decision_EO": eo(q, [0, 1]),
        "missing_expected_group_EO": "NaN / not estimable",
        "NMI_before": before["feature"],
        "NMI_after_drop_default": after.get("feature", 0.0),
        "full_NMI_loss_passes_threshold_100": bool(passes),
        "source_sha256": {},
    }
    root = Path(__file__).resolve().parents[1]
    for name in [
        "scripts/audit_nhis_intersection_feasibility_20260918.py",
        "src/fairbias/bias_metric.py", "src/fairbias/mitigation.py",
        "src/nhis_fairbias/benchmark/metrics.py",
        "src/nhis_fairbias/benchmark/adapters/adapter_fairbias_ae.py",
    ]:
        report["source_sha256"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
