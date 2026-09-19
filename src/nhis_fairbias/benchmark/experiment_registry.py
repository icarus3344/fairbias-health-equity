"""Versioned F/C/S experiment matrix; no data or performance enters enumeration."""
from __future__ import annotations

import hashlib
import itertools
import json

SEEDS = (0, 7, 19, 37, 73)
EPSILON_RATIOS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0)
DIFFERENCE_BOUNDS = (0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2)
OPERATING_POINTS = (0.10, 0.05, 0.20)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def enumerate_candidates(*, arms=("arm_001", "arm_002", "arm_003", "arm_004"),
                         backbones=("LR", "GBDT"), include_recent=True,
                         weighted_sensitivity=True):
    """Enumerate complete configurations; seeds are required, never selected.

    Classical LR solvers are deterministic. GBDT feature-permutation ties and
    stochastic FairBias/LFR representations retain all five registered seeds.
    MLP/fairret is a separately matched extension with one fixed architecture.
    """
    configs = []
    methods = ("UNMITIGATED", "FAIRBIAS_BM", "REWEIGHING", "LFR_RECONSTRUCTED", "EG_DP", "EG_EO", "TO_EO")
    for arm, backbone in itertools.product(arms, backbones):
        grids = [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)] if backbone == "LR" else [
            {"C": 1.0, "estimator_params": {"n_estimators": n, "max_depth": d}} for n, d in itertools.product((100, 200), (2, 3))]
        for method in methods + (("OXONFAIR_EO",) if include_recent else ()):
            if method == "LFR_RECONSTRUCTED" and arm == "arm_002":
                configs.append({"arm_id": arm, "backbone": backbone, "method": method,
                                "status": "NOT_SUPPORTED", "reason": "official LFR is binary-group only", "seeds": []})
                continue
            if method == "FAIRBIAS_BM":
                method_grid = [{"epsilon_ratio": r, "max_iterations": 50, "max_geometry_evaluations": 20000} for r in EPSILON_RATIOS]
            elif method == "LFR_RECONSTRUCTED":
                method_grid = [{"k": k, "Az": az, "Ax": 0.01, "Ay": 1.0, "maxiter": 5000, "maxfun": 5000}
                               for k, az in itertools.product((5, 10), (0.1, 1.0, 10.0, 50.0))]
            elif method in ("EG_DP", "EG_EO"):
                method_grid = [{"difference_bound": b, "eps": 0.01, "max_iter": 50,
                                "constraint_type": "dp" if method == "EG_DP" else "eo"} for b in DIFFERENCE_BOUNDS]
            elif method == "OXONFAIR_EO":
                method_grid = [{"bound": b, "grid_width": 2 if arm == "arm_002" else 6} for b in (0.01, 0.05, 0.10, 0.20)]
            else:
                method_grid = [{}]
            weight_modes = (False, True) if weighted_sensitivity and method in ("UNMITIGATED", "FAIRBIAS_BM") else (False,)
            for estimator, params, weighted in itertools.product(grids, method_grid, weight_modes):
                complexity = float(estimator.get("C", 1)) if backbone == "LR" else float(estimator["estimator_params"]["n_estimators"] * 2**estimator["estimator_params"]["max_depth"])
                configs.append({"arm_id": arm, "backbone": backbone, "method": method,
                    "training_weighted": weighted, "params": {**estimator, **params}, "complexity": complexity,
                    "seeds": list(SEEDS if backbone != "LR" or method in ("FAIRBIAS_BM", "LFR_RECONSTRUCTED") else (0,)),
                    "status": "REGISTERED"})
    if include_recent:
        for arm, method in itertools.product(arms, ("UNMITIGATED", "FAIRBIAS_BM", "FAIRRET_EO")):
            params_grid = [{}] if method == "UNMITIGATED" else ([{"epsilon_ratio": r, "max_iterations": 50, "max_geometry_evaluations": 20000} for r in EPSILON_RATIOS]
                if method == "FAIRBIAS_BM" else [{"fairness_coefficient": c, "epochs": 100, "learning_rate": 0.001} for c in (0.01, 0.1, 1.0, 10.0)])
            for params in params_grid:
                configs.append({"arm_id": arm, "backbone": "MLP", "method": method, "training_weighted": False,
                    "params": params, "complexity": 1.0, "seeds": list(SEEDS), "status": "REGISTERED"})
    for config in configs:
        config["candidate_id"] = identity(config)[:20]
    return configs


def make_adapter(config, seed):
    from .adapters import (UnmitigatedAdapter, FairBiasAdapter, ReweighingAdapter,
                           LFRAdapter, ExponentiatedGradientAdapter, ThresholdOptimizerAdapter)
    params = dict(config["params"])
    params.update(backbone=config["backbone"], random_state=seed)
    method = config["method"]
    constructors = {"UNMITIGATED": UnmitigatedAdapter, "FAIRBIAS_BM": FairBiasAdapter,
        "REWEIGHING": ReweighingAdapter, "LFR_RECONSTRUCTED": LFRAdapter,
        "EG_DP": ExponentiatedGradientAdapter, "EG_EO": ExponentiatedGradientAdapter,
        "TO_EO": ThresholdOptimizerAdapter}
    if method == "FAIRBIAS_BM" or method.startswith("FAIRBIAS_GEOMETRY_"):
        params["arm_id"] = config["arm_id"]
        return FairBiasAdapter(**params)
    if method in ("FAIRBIAS_BM_AE", "FAIRBIAS_JOINT"):
        from .adapters.adapter_fairbias_ae import FairBiasAEAdapter
        return FairBiasAEAdapter(**params)
    if method == "FRAPPE_EO":
        from .adapters.adapter_frappe import FrappeAdapter
        return FrappeAdapter(**params)
    if method == "FAIRGBM_EO":
        from .adapters.adapter_fairgbm import FairGBMAdapter
        params.pop("backbone")
        return FairGBMAdapter(**params)
    if method == "OXONFAIR_EO":
        from .adapters.adapter_oxonfair import OxonFairAdapter
        return OxonFairAdapter(**params)
    if method == "FAIRRET_EO":
        from .adapters.adapter_fairret import FairretAdapter
        params.pop("backbone")
        return FairretAdapter(**params)
    return constructors[method](**params)
