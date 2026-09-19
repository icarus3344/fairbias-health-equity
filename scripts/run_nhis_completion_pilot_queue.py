"""Own a finite F/C queue, optionally continuing all80 after10 verified pilots.

All fits use the identical manifest. Pilot models count once in that matrix;
no S/T evaluation or formal-study admission is performed by this controller.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_nhis_fairbias_completion import load_manifest, sha


def atomic(path, value):
    temporary = path.with_suffix('.part')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def limit(path):
    value = json.loads(path.read_text())['concurrency']
    if type(value) is not int or not 1 <= value <= 56:
        raise ValueError('Concurrency must be an integer in [1,56]')
    return value


def verify_result(output, job_id, manifest_seal, returncode):
    path = output / 'result.json'
    if not path.exists():
        return {'status': 'INCOMPLETE_NO_RECEIPT', 'returncode': returncode}
    result = json.loads(path.read_text())
    passed = (returncode == 0 and result.get('status') == 'FC_COMPLETE_FEASIBLE'
              and result.get('job_id') == job_id
              and result.get('manifest_sha256') == manifest_seal
              and result.get('reload_exact') is True
              and result.get('formal_benchmark_admission') is False
              and result.get('S_T_evaluated') is False)
    if passed:
        model = output / 'policy.joblib'
        passed = model.is_file() and sha(model) == result.get('model_sha256')
    return {'status': 'FC_PILOT_VERIFIED' if passed else 'REQUIRES_REVIEW',
            'returncode': returncode, 'reported_status': result.get('status'),
            'result_sha256': sha(path), 'elapsed_seconds': result.get('elapsed_seconds')}


def stage_transition(completed, pilot_ids):
    if not set(pilot_ids) <= completed.keys():
        return 'PILOTS_RUNNING'
    return 'FAMILY' if all(completed[j]['status'] == 'FC_PILOT_VERIFIED' for j in pilot_ids) else 'PILOT_GATE_FAILED'


def memory_allows_dispatch():
    maximum, current = Path('/sys/fs/cgroup/memory.max'), Path('/sys/fs/cgroup/memory.current')
    if maximum.exists() and current.exists():
        ceiling = maximum.read_text().strip()
        if ceiling != 'max':
            return int(current.read_text()) < int(ceiling) * .80
    return True


def run(root, output, *, continue_family=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = load_manifest(root / 'control/manifest.json')
    plan_path = root / 'control/pilot_plan.json'
    plan = json.loads(plan_path.read_text())
    jobs = {j['job_id']: j for j in manifest['jobs']}
    pending = [p['job_id'] for p in plan['pilots']]
    pilot_ids = list(pending)
    reserved = sorted(set(jobs) - set(pending)) if continue_family else []
    if (plan['manifest_sha256'] != manifest['manifest_sha256']
            or len(pending) != 10 or len(set(pending)) != 10 or not set(pending) <= jobs.keys()):
        raise ValueError('Pilot ownership/manifest mismatch')
    for arm in {jobs[k]['config']['arm_id'] for k in pending}:
        expected = next(j['prepared']['sha256'] for j in jobs.values() if j['config']['arm_id'] == arm)
        if sha(root / 'prepared' / (arm + '.joblib')) != expected:
            raise ValueError('Pilot input mismatch')
    output.mkdir(parents=True, exist_ok=False)
    lock = (output / 'owner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cap = output / 'resources.json'
    atomic(cap, {'concurrency': 2})
    env = dict(os.environ, PYTHONPATH=str(root / 'src'), CUDA_VISIBLE_DEVICES='')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    active, completed, phase = {}, {}, 'PILOTS'
    meta = {'controller_pid': os.getpid(), 'queue_source_sha256': sha(__file__),
            'manifest_sha256': manifest['manifest_sha256'], 'pilot_plan_sha256': sha(plan_path),
            'started_unix': time.time(), 'wall_timeout_seconds': None,
            'continue_family_after_pilot_gate': continue_family,
            'total_registered_fits': len(pending) + len(reserved),
            'formal_benchmark_admission': False}
    atomic(output / 'launch.json', meta)
    while pending or active or reserved:
        for job_id, info in list(active.items()):
            process, log = info['process'], info['log']
            code = process.poll()
            if code is not None:
                log.close()
                completed[job_id] = verify_result(output / job_id, job_id, manifest['manifest_sha256'], code)
                del active[job_id]
        if phase == 'PILOTS' and not pending and not active:
            phase = stage_transition(completed, pilot_ids)
            if phase == 'PILOT_GATE_FAILED':
                atomic(output / 'queue_receipt.json', {**meta, 'completed': completed,
                    'status': phase, 'blocked_jobs': reserved, 'ended_unix': time.time()})
                return
            load_manifest(root / 'control/manifest.json')
            pending, reserved = reserved, []
        maximum = min(limit(cap), 10 if phase == 'PILOTS' else 56)
        while (pending and len(active) < maximum and memory_allows_dispatch()
               and shutil.disk_usage(root).free > 5 * 1024**3):
            job_id = pending.pop(0)
            job = jobs[job_id]
            target = output / job_id
            log = (output / (job_id + '.log')).open('xb')
            command = [sys.executable, '-B', str(root / 'scripts/run_nhis_fairbias_completion.py'),
                'pilot', '--manifest', str(root / 'control/manifest.json'), '--job-id', job_id,
                '--prepared', str(root / 'prepared' / (job['config']['arm_id'] + '.joblib')),
                '--output', str(target), '--cache-dir', str(root / 'mds_cache')]
            process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active[job_id] = {'process': process, 'log': log, 'started_unix': time.time()}
        atomic(output / 'state.json', {**meta, 'updated_unix': time.time(), 'concurrency': maximum,
            'phase': phase, 'awaiting_pilot_gate': reserved, 'pending': pending,
            'active': {k: {'pid': v['process'].pid, 'started_unix': v['started_unix']}
                                         for k,v in active.items()}, 'completed': completed})
        if pending and not active and shutil.disk_usage(root).free <= 5 * 1024**3:
            raise RuntimeError('Insufficient disk to dispatch pending pilots')
        if pending or active:
            time.sleep(5)
    load_manifest(root / 'control/manifest.json')
    atomic(output / 'queue_receipt.json', {**meta, 'completed': completed,
        'status': 'ALL_FC_FITS_VERIFIED' if all(r['status']=='FC_PILOT_VERIFIED' for r in completed.values())
                  else 'REQUIRES_REVIEW', 'ended_unix': time.time()})


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--continue-family', action='store_true')
    a=p.parse_args()
    run(a.root,a.output,continue_family=a.continue_family)
