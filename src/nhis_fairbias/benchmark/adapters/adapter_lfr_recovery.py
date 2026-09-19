"""Opt-in analytic-gradient recovery; the registered LFR adapter stays frozen.

Use ``LFRAnalyticRecoveryAdapter`` explicitly in an isolated worker process.
It inherits the original training/transform/label-isolation implementation and
replaces only the derivative provider during that fit. Like the original
adapter, its temporary module patch is NOT safe for concurrent thread fits.
Recovery outputs require their own run identity and are never ALL70 results.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .adapter_lfr import LFRAdapter
from ..lfr_analytic_objective import (
    GRADIENT_ID, OBJECTIVE_ID, lfr_objective_and_gradient,
)


# Full installed-source fingerprints used by the fidelity tests. Fail closed
# on package source drift instead of silently applying an unverified gradient.
_UPSTREAM_SOURCE_SHA256 = {
    "lfr": "12f708ad8c927e688cb9fded5e7b6b002a3fafe6c95e169ffd03b26f5f1afdf2",
    "lfr_helpers": "6ffe841453339ecad19dc69fcf752e767d9feda4997d6e96fa239905de83d2ed",
}


class LFRAnalyticRecoveryAdapter(LFRAdapter):
    """Same objective, initialization, bounds and budgets; analytic derivative."""

    name = "LFR_RECONSTRUCTED_ANALYTIC_RECOVERY_V1"
    recovery_variant = "lfr_analytic_recovery_v1"

    def fit(self, X, y, A, sample_weight=None):
        import aif360.algorithms.preprocessing.lfr as upstream

        source_hashes = {
            key: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for key, module in [("lfr", upstream), ("lfr_helpers", upstream.lfr_helpers)]
        }
        if source_hashes != _UPSTREAM_SOURCE_SHA256:
            raise RuntimeError("LFR recovery upstream source differs from the verified AIF360 implementation")
        original_optimizer = upstream.optim.fmin_l_bfgs_b
        expected_objective = upstream.lfr_helpers.LFR_optim_objective
        evaluation_count = 0
        corner_evaluations = 0
        initial_loss = None
        last_diagnostics = {}
        final_diagnostics = {}

        def analytic_optimizer(func, *args, **kwargs):
            if func is not expected_objective or not kwargs.get("approx_grad"):
                raise RuntimeError("LFR recovery expected the upstream finite-difference objective")
            if kwargs.get("fprime") is not None:
                raise RuntimeError("LFR recovery refuses an unexpected upstream derivative")
            reference_initial_loss = expected_objective(kwargs["x0"], *kwargs["args"])

            def combined(parameters, *objective_args):
                nonlocal evaluation_count, corner_evaluations, initial_loss
                result = lfr_objective_and_gradient(
                    parameters, *objective_args, diagnostics=last_diagnostics
                )
                evaluation_count += 1
                corner_evaluations += int(last_diagnostics["primitive_corner_present"])
                if initial_loss is None:
                    initial_loss = result[0]
                    if not np.isclose(initial_loss, reference_initial_loss, rtol=1e-12, atol=1e-12):
                        raise RuntimeError("LFR recovery initial loss disagrees with upstream")
                return result

            updated = dict(kwargs)
            updated["approx_grad"] = False
            # All other options (including x0, bounds, stopping rules and
            # budgets) are passed through exactly as received from AIF360.
            result = original_optimizer(combined, *args, **updated)
            final_loss, final_gradient = lfr_objective_and_gradient(
                result[0], *kwargs.get("args", ()), diagnostics=final_diagnostics
            )
            if not np.isclose(final_loss, result[1], rtol=1e-12, atol=1e-12):
                raise RuntimeError("LFR optimizer final objective disagrees with returned parameters")
            reference_final_loss = expected_objective(result[0], *kwargs["args"])
            if not np.isclose(final_loss, reference_final_loss, rtol=1e-12, atol=1e-12):
                raise RuntimeError("LFR recovery final loss disagrees with upstream")
            final_diagnostics["gradient_inf_norm"] = float(np.max(np.abs(final_gradient)))
            projected_gradient = final_gradient.copy()
            for index, (lower, upper) in enumerate(kwargs["bounds"]):
                if ((lower is not None and result[0][index] <= lower and final_gradient[index] > 0)
                        or (upper is not None and result[0][index] >= upper and final_gradient[index] < 0)):
                    projected_gradient[index] = 0
            final_diagnostics["projected_gradient_inf_norm"] = float(np.max(np.abs(projected_gradient)))
            return result

        with patch.object(upstream, "optim", SimpleNamespace(fmin_l_bfgs_b=analytic_optimizer)):
            super().fit(X, y, A, sample_weight=sample_weight)
        self.recovery_variant_ = self.recovery_variant
        self.optimization_result_.update({
            "recovery_variant": self.recovery_variant,
            "objective_id": OBJECTIVE_ID, "gradient_id": GRADIENT_ID,
            "upstream_source_sha256": source_hashes,
            "upstream_loss_verification_evaluations": 2,
            "postfit_gradient_verification_evaluations": 1,
            "initial_objective": initial_loss,
            "analytic_objective_evaluations": evaluation_count,
            "primitive_corner_evaluations": corner_evaluations,
            "final_derivative_diagnostics": final_diagnostics,
            "convergence_scope": "scipy_termination_only_no_global_or_nonsmooth_optimality_claim",
        })
        return self
