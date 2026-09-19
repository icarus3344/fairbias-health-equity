"""Scoped numerical admission for application FairBias geometry only."""
from contextlib import contextmanager
import numpy as np
from fairbias import bias_metric


@contextmanager
def audited_mds(records, geometry_call):
    """Capture stress/iterations and reject unverified iteration-cap exits.

    The benchmark runs one fit per process. Historical evaluator APIs retain
    their prior implementation. The fitted MDS objects are never serialized.
    """
    original = bias_metric.MDS

    class AuditedMDS(original):
        def fit_transform(self, *args, **kwargs):
            points = super().fit_transform(*args, **kwargs)
            finite = np.isfinite(self.stress_) and np.all(np.isfinite(points))
            capped = int(self.n_iter_) >= int(self.max_iter)
            records.append({"geometry_call": int(geometry_call()), "components": int(self.n_components),
                "n_init": int(self.n_init), "iterations": int(self.n_iter_), "max_iter": int(self.max_iter),
                "stress": float(self.stress_), "seed": int(self.random_state),
                "status": "MDS_ITERATION_CAP" if capped else ("CONVERGED_BEFORE_CAP" if finite else "NONFINITE_MDS")})
            if not finite:
                raise RuntimeError("NOT_ESTIMABLE: nonfinite MDS output")
            if capped:
                raise RuntimeError("BUDGET_EXHAUSTED: MDS_ITERATION_CAP; convergence not verified")
            return points

    try:
        bias_metric.MDS = AuditedMDS
        yield
    finally:
        bias_metric.MDS = original
