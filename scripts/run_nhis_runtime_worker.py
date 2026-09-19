"""Isolated, hash-verified FRAPPE pipeline recovery; original worker is unchanged."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
VARIANT = 'frappe_pipeline_deterministic_v1_20260917'
REQUIRED_SOURCES = {
    'scripts/run_nhis_runtime_worker.py',
    'src/nhis_fairbias/benchmark/adapters/adapter_frappe_pipeline.py',
    'src/nhis_fairbias/benchmark/adapters/adapter_frappe.py',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['worker'])
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    if job.get('runtime_variant') != VARIANT or job['config']['method'] != 'FRAPPE_EO':
        raise ValueError('runtime worker accepts only the declared FRAPPE recovery')
    if job.get('runtime_environment') != {'TF_DETERMINISTIC_OPS': '1'}:
        raise ValueError('missing declared deterministic startup environment')
    if os.environ.get('TF_DETERMINISTIC_OPS') != '1':
        raise ValueError('determinism must be enabled before Python startup')
    files = job.get('runtime_source_files', {})
    if set(files) != REQUIRED_SOURCES:
        raise ValueError('runtime source manifest is incomplete')
    for relative, expected in files.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'runtime source hash mismatch: {relative}')
    from nhis_fairbias.benchmark import experiment_worker as worker
    from nhis_fairbias.benchmark.adapters.adapter_frappe_pipeline import DeterministicPipelineFrappeAdapter
    from nhis_fairbias.benchmark.experiment_registry import identity
    if identity({'variant': VARIANT, 'files': files, 'environment': job['runtime_environment']}) != job.get('runtime_source_identity'):
        raise ValueError('runtime source identity mismatch')
    def make_adapter(config, seed):
        if config['method'] != 'FRAPPE_EO':
            raise ValueError('unexpected method in isolated recovery worker')
        parameters = dict(config['params'])
        parameters.update(backbone=config['backbone'], random_state=seed)
        return DeterministicPipelineFrappeAdapter(**parameters)
    # This binding exists only in this isolated worker process. No source file,
    # global TensorFlow module, or other running worker is modified.
    worker.make_adapter = make_adapter
    result = worker.execute_job(args.job)
    if result['status'] != 'VALID':
        raise SystemExit(1)

if __name__ == '__main__':
    main()
