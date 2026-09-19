"""Bounded F/C-only diagnostic for a frozen EG/GBDT job.

This intentionally writes aggregate diagnostics only.  It never evaluates S/T
and never serializes the fitted adapter or prediction arrays.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from nhis_fairbias.benchmark.experiment_registry import make_adapter


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main(job_path: str, output_dir: str) -> None:
    job_file = Path(job_path).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    job = json.loads(job_file.read_text())
    data_file = Path(job["data_path"]).resolve()
    if sha(data_file) != job["data_sha256"]:
        raise RuntimeError("prepared data SHA256 mismatch")
    data = joblib.load(data_file)
    if data.get("data_identity") != job.get("data_identity"):
        raise RuntimeError("prepared data identity mismatch")
    partitions = data["partitions"]
    if "evaluation_T" in partitions:
        raise RuntimeError("diagnostic refuses data containing evaluation_T")
    F, C = partitions["fitting_F"], partitions["calibration_C"]
    config, seed = job["config"], int(job["seed"])
    started = time.time()
    adapter = make_adapter(config, seed)
    adapter.fit(data["X_F"], F.y, F.A)
    raw = np.asarray(adapter.predict_decision_proba(data["X_C"], A=C.A), dtype=float)
    weights = np.asarray(adapter.model.weights_, dtype=float)
    base_unique = []
    for estimator in adapter.model._hs:
        values = np.asarray(estimator(data["X_C"])).reshape(-1)
        base_unique.append(np.unique(values).tolist())
    below, above = raw[raw < 0], raw[raw > 1]
    finite = raw[np.isfinite(raw)]
    max_violation = max(float(-below.min()) if len(below) else 0.0,
                        float((above - 1).max()) if len(above) else 0.0)
    result = {
        "diagnostic_only": True,
        "candidate_id": config["candidate_id"], "seed": seed,
        "method": config["method"], "backbone": config["backbone"],
        "job_path": str(job_file), "prepared_path": str(data_file),
        "prepared_sha256": sha(data_file), "data_identity": data["data_identity"],
        "source_identity": job["source_identity"],
        "partition_rows": {"F": len(F), "C": len(C)},
        "evaluation_T_present": "evaluation_T" in partitions,
        "q": {"finite_count": int(np.isfinite(raw).sum()), "count": int(raw.size),
              "min": float(raw.min()) if raw.size else None,
              "max": float(raw.max()) if raw.size else None,
              "below_zero_count": int(len(below)), "above_one_count": int(len(above)),
              "max_absolute_bound_violation": max_violation},
        "eg_weights": {"count": int(weights.size), "sum": float(weights.sum()),
                       "min": float(weights.min()), "max": float(weights.max()),
                       "finite": bool(np.isfinite(weights).all())},
        "mixture_base_prediction_unique_values_on_C": base_unique,
        "elapsed_seconds": time.time() - started,
        "runtime": {"python": sys.executable, "numpy": np.__version__,
                     "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}},
        "source_hashes": {"experiment_registry.py": sha(Path("src/nhis_fairbias/benchmark/experiment_registry.py")),
                          "adapter_reductions.py": sha(Path("src/nhis_fairbias/benchmark/adapters/adapter_reductions.py")),
                          "predictions.py": sha(Path("src/nhis_fairbias/benchmark/predictions.py"))},
    }
    (out / "eg_gbdt_bounds_aggregate.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
