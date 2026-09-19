"""Read failure metadata only; never open prepared data or model objects."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def collect(root: Path) -> dict:
    registration = root / 'registration.json'
    counts, failures = Counter(), []
    for directory in sorted((root / 'jobs').iterdir()):
        result_path, job_path = directory / 'result.json', directory / 'job.json'
        if not result_path.exists() or not job_path.exists():
            continue
        result, job = json.loads(result_path.read_text()), json.loads(job_path.read_text())
        config = job['config']
        if config['method'] == 'FRAPPE_EO':
            continue  # superseded by a separate deterministic runtime
        counts[(config['method'], result['status'])] += 1
        if result['status'] == 'VALID':
            continue
        key = job.get('representation_key')
        cache_path = root / 'representation_cache' / f'{key}.json'
        cache = json.loads(cache_path.read_text()) if key and cache_path.exists() else {}
        failures.append({
            'job_id': directory.name, 'job_path': str(job_path),
            'job_sha256': hashlib.sha256(job_path.read_bytes()).hexdigest(),
            'result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
            'config': config, 'seed': job['seed'], 'source_identity': job['source_identity'],
            'data_identity': job['data_identity'], 'data_sha256': job['data_sha256'],
            'representation_key': key, 'status': result['status'],
            'error': result.get('error'), 'error_type': result.get('error_type'),
            'elapsed_seconds': result.get('elapsed_seconds'),
            'cache_status': cache.get('status'),
            'optimization_result': cache.get('optimization_result'),
        })
    return {
        'schema': 'failure_recovery_inventory_v1', 'metadata_only': True,
        'original_run': str(root),
        'registration_sha256': hashlib.sha256(registration.read_bytes()).hexdigest(),
        'counts': [{'method': method, 'status': status, 'n': count}
                   for (method, status), count in sorted(counts.items())],
        'failed_seed_jobs': len(failures),
        'unique_failed_representation_keys': len({r['representation_key'] for r in failures if r['representation_key']}),
        'failures': failures,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.run.resolve()), indent=2))
