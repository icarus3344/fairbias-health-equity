"""Aggregate environment and tiny synthetic reproducibility witness for a replica."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic-frappe', action='store_true')
    parser.add_argument('--one-fit', action='store_true', help='Mirror one fit per isolated formal worker')
    args = parser.parse_args()
    result = {'hostname': platform.node(), 'python': sys.version,
              'executable': sys.executable, 'machine': platform.machine(),
              'packages': {}}
    for name in ('numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib', 'fairlearn',
                 'aif360', 'tensorflow', 'tensorflow-model-remediation'):
        try:
            result['packages'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result['packages'][name] = None
    for name in ('cpu.max', 'memory.max', 'memory.current', 'cpuset.cpus.effective'):
        path = Path('/sys/fs/cgroup') / name
        result[name] = path.read_text().strip() if path.exists() else None
    if args.synthetic_frappe:
        # Same CPU worker environment as the registered run, including unset oneDNN flag.
        import numpy as np
        from nhis_fairbias.benchmark.adapters.adapter_frappe import FrappeAdapter
        rng = np.random.default_rng(4)
        X = rng.normal(size=(32, 3))
        y = (X[:, 0] + .5 * X[:, 1] > 0).astype(int)
        A = np.array([0, 1] * 16)
        outputs = []
        start = time.monotonic()
        for _ in range(1 if args.one_fit else 2):
            adapter = FrappeAdapter(hidden_units=(2,), epochs=2, batch_size=2)
            adapter.fit(X[:16], y[:16], A[:16], X_calibration=X[16:],
                        y_calibration=y[16:], A_calibration=A[16:])
            outputs.append(adapter.predict_decision_proba(X[16:20]))
        delta = None if args.one_fit else float(np.max(np.abs(outputs[0] - outputs[1])))
        result['synthetic'] = {'elapsed_seconds': time.monotonic() - start,
                               'repeat_max_abs_delta': delta,
                               'fits_in_process': len(outputs),
                               'decision_probabilities': outputs[0].tolist(),
                               'note': '32 generated rows only; no NHIS input'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'hostname': result['hostname'], 'written': str(args.output),
                      'synthetic': result.get('synthetic')}), flush=True)
    if args.synthetic_frappe and not args.one_fit:
        assert result['synthetic']['repeat_max_abs_delta'] == 0.0, result['synthetic']

if __name__ == '__main__':
    main()
