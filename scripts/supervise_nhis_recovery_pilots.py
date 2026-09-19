"""Run one explicitly registered lane of bounded F/C diagnostics, then exit."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from supervise_joint_pilot import supervise_command


def run(plan_path, lane):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads(Path(plan_path).read_text())
    selected = [x for x in plan['pilots'] if x['lane'] == lane]
    if not selected or len({x['output'] for x in plan['pilots']}) != len(plan['pilots']):
        raise ValueError('Unknown lane or duplicate pilot ownership')
    for relative, expected in plan['source_files'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Recovery source changed: ' + relative)
    for job in selected:
        if hashlib.sha256(Path(job['job']).read_bytes()).hexdigest() != job['job_sha256']:
            raise ValueError('Pilot job hash changed')
        output = Path(job['output'])
        output.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONPATH=str(root / 'src'), CUDA_VISIBLE_DEVICES='')
        supervise_command(
            [sys.executable, '-B', str(root / 'scripts/pilot_nhis_numerical_recovery.py'),
             '--job', job['job'], '--output', job['output'], '--variant', job['variant']],
            receipt_path=output.with_suffix('.supervisor.json'),
            stdout_path=output.with_suffix('.stdout.log'),
            elapsed_limit_seconds=job['wall_seconds'], memory_limit_bytes=4 * 1024 ** 3,
            environment=env, cwd=root,
            receipt_metadata={'plan_sha256': hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
                              'lane': lane, 'diagnostic_only': True,
                              'runtime_source_files': plan['source_files']},
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--lane', required=True)
    args = parser.parse_args()
    run(args.plan, args.lane)
