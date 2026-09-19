"""Tiny isolated FRAPPE fit/reload witness; no NHIS data."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np

from nhis_fairbias.benchmark.adapters.adapter_frappe import FrappeAdapter


def main() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(32, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    A = np.array([0, 1] * 16)
    adapter = FrappeAdapter(hidden_units=(2,), epochs=2, batch_size=2)
    adapter.fit(X[:16], y[:16], A[:16], X_calibration=X[16:], y_calibration=y[16:], A_calibration=A[16:])
    before = adapter.predict_decision_proba(X[16:20])
    repeat = FrappeAdapter(hidden_units=(2,), epochs=2, batch_size=2)
    repeat.fit(X[:16], y[:16], A[:16], X_calibration=X[16:], y_calibration=y[16:], A_calibration=A[16:])
    repeat_delta = float(np.max(np.abs(before - repeat.predict_decision_proba(X[16:20]))))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "frappe.joblib"
        joblib.dump(adapter, path)
        child = (
            "import joblib, numpy as np, sys; "
            "a=joblib.load(sys.argv[1]); "
            "x=np.load(sys.argv[2]); "
            "print(','.join(map(str,a.predict_decision_proba(x))))"
        )
        x_path = Path(directory) / "x.npy"
        np.save(x_path, X[16:20])
        env = dict(os.environ)
        child_out = subprocess.check_output(
            [sys.executable, "-c", child, str(path), str(x_path)], env=env, text=True
        ).strip()
        after = np.fromstring(child_out, sep=",")
        restored = joblib.load(path)
    delta = float(np.max(np.abs(before - after)))
    assert delta == 0.0, delta
    print(f"fit_reload_ok max_reload_delta={delta:.1f} repeat_seed_delta={repeat_delta:.6f} output_type={restored.output_type}")


if __name__ == "__main__":
    main()
