"""Read-only F/C artifact audit and sealed backup of the completed westb run.

Does not refit, select models, read S values, evaluate T, or shut down the host.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tarfile
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix('.part')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def run(base, output):
    base, output = base.resolve(), output.resolve()
    root = base/'fairbias_completion_v2_20260917'
    controls = [base/'fairbias_completion_parallel_20260917',
                base/'fairbias_completion_parallel64_20260917']
    output.mkdir(exist_ok=False)
    sys.path[:0] = [str(root/'src'), str(root/'scripts')]
    from run_nhis_fairbias_completion import load_manifest
    from pilot_scheduled_joint import runtime_modules
    import joblib
    import numpy as np

    manifest = load_manifest(root/'control/manifest.json')
    state = json.loads((controls[-1]/'state.json').read_text())
    receipt = json.loads((controls[-1]/'queue_receipt.json').read_text())
    jobs = {j['job_id']: j for j in manifest['jobs']}
    assert not state['active'] and not state['pending']
    assert receipt['status'] == 'FINISHED_REQUIRES_SUPERVISOR_REVIEW'
    assert receipt['manifest_sha256'] == manifest['manifest_sha256']
    assert receipt['completed'] == state['completed']
    assert set(receipt['completed']) == set(jobs) and len(jobs) == receipt['jobs'] == 80
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        assert str(root/'scripts/run_nhis_fairbias_completion.py').encode() not in args
        assert str(controls[-1]/'run_nhis_completion_takeover_v2.py').encode() not in args

    sources = {k: v for k, v in manifest['sources'].items() if k.startswith('src/')}
    data = {}
    for arm in sorted({j['config']['arm_id'] for j in jobs.values()}):
        job = next(j for j in jobs.values() if j['config']['arm_id'] == arm)
        path = root/'prepared'/(arm+'.joblib')
        assert sha(path) == job['prepared']['sha256']
        payload = joblib.load(path)
        assert payload['data_identity'] == job['prepared']['data_identity']
        assert set(payload['partitions']) == {'fitting_F', 'calibration_C', 'selection_S'}
        assert 'X_T' not in payload
        # The registered container is deserialized; only C is retained or used.
        calibration = payload['partitions']['calibration_C']
        del payload
        assert calibration.role == 'calibration_C' and calibration.year == 2022
        assert calibration.arm_id == arm
        data[arm] = calibration

    outcomes = []
    for job_id, job in jobs.items():
        candidates = [root/'runs/full80_v2'/job_id] + [p/'jobs'/job_id for p in controls]
        found = [p for p in candidates if (p/'result.json').is_file()]
        assert len(found) == 1, 'Missing or duplicate output: '+job_id
        path = found[0]
        result_path, model_path = path/'result.json', path/'policy.joblib'
        r = json.loads(result_path.read_text())
        record = receipt['completed'][job_id]
        assert sha(result_path) == record['result_sha256']
        assert r['status'] == 'FC_COMPLETE_FEASIBLE' and r['reload_exact'] is True
        assert r['job_id'] == job_id and r['config'] == job['config'] and r['seed'] == job['seed']
        assert r['manifest_sha256'] == manifest['manifest_sha256']
        assert r['prepared_sha256'] == job['prepared']['sha256']
        assert r['data_identity'] == job['prepared']['data_identity']
        assert r['formal_benchmark_admission'] is False and r['S_T_evaluated'] is False
        assert r['partitions_used_for_fitting'] == ['fitting_F', 'calibration_C']
        assert r['completion']['status'] == 'COMPLETE_FEASIBLE'
        assert all(importlib.metadata.version(k) == v for k, v in r['packages'].items())
        for entry in r['loaded_modules'].values():
            assert sources[entry['source']] == entry['sha256']
        assert sha(model_path) == r['model_sha256']
        policy = joblib.load(model_path)
        assert policy.is_calibrated and policy._adapter.is_fitted_
        assert policy._adapter.mode == job['config']['method'].removeprefix('FAIRBIAS_')
        assert policy._adapter.adapter_params['random_state'] == job['seed']
        assert policy._adapter.provenance_['status'] == 'COMPLETE_FEASIBLE'
        C = data[job['config']['arm_id']]
        first = policy.predict(C.X_semantic, C.A)
        del policy
        second_policy = joblib.load(model_path)
        second = second_policy.predict(C.X_semantic, C.A)
        assert first.p_event is not None and second.p_event is not None
        assert first.n_samples == second.n_samples == len(C.X_semantic)
        assert np.array_equal(first.p_event, second.p_event)
        assert np.array_equal(first.q_decision, second.q_decision)
        row = {'job_id': job_id, 'method': job['config']['method'], 'arm': job['config']['arm_id'],
               'backbone': job['config']['backbone'], 'seed': job['seed'], 'output': str(path),
               'result_sha256': sha(result_path), 'model_sha256': r['model_sha256'],
               'elapsed_seconds': r['elapsed_seconds'], 'calibration_rows': first.n_samples,
               'C_p_sha256': hashlib.sha256(first.p_event.tobytes()).hexdigest(),
               'C_q_sha256': hashlib.sha256(first.q_decision.tobytes()).hexdigest(),
               'independent_reload_prediction_exact': True,
               'exit_code_observed': record.get('exit_code_observed', record.get('returncode') is not None)}
        outcomes.append(row)
        del second_policy, first, second
        if len(outcomes) % 10 == 0:
            print(json.dumps({'verified_models': len(outcomes)}), flush=True)

    loaded = runtime_modules(root, sources)
    load_manifest(root/'control/manifest.json')
    audit = {'status': 'ALL_80_FC_ARTIFACTS_AND_RELOADS_VERIFIED', 'finished_unix': time.time(),
             'auditor_sha256': sha(Path(__file__).resolve()),
             'manifest_sha256': manifest['manifest_sha256'], 'queue_receipt_sha256': sha(controls[-1]/'queue_receipt.json'),
             'jobs': outcomes, 'loaded_modules': loaded, 'S_T_evaluated': False,
             'formal_benchmark_admission': False, 'all_paper_work_complete': False}
    write(output/'audit.json', audit)
    files = {}
    for directory in [root]+controls:
        for p in sorted(directory.rglob('*')):
            rel = p.relative_to(base)
            if any(x in rel.parts for x in ('mds_cache', '__pycache__', '.pytest_cache')):
                continue
            if p.is_symlink():
                raise ValueError('Unexpected backup symlink: '+str(rel))
            if p.is_file():
                files[str(rel)] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
    index = {'version': 'completed80_backup_v1', 'created_unix': time.time(), 'source_base': str(base),
             'files': files, 'file_count': len(files), 'total_bytes': sum(v['bytes'] for v in files.values()),
             'audit_sha256': sha(output/'audit.json'), 'excludes': ['mds_cache', '__pycache__', '.pytest_cache']}
    write(output/'files_manifest.json', index)
    print(json.dumps({'archive_files': len(files), 'archive_source_bytes': index['total_bytes']}), flush=True)
    archive = output/'completed80.tar.gz'
    temporary = output/'completed80.tar.gz.part'
    with tarfile.open(temporary, 'w:gz', compresslevel=1) as tar:
        for name in files:
            tar.add(base/name, arcname=name, recursive=False)
    for name, record in files.items():
        p = base/name
        assert p.stat().st_size == record['bytes'] and sha(p) == record['sha256']
    temporary.replace(archive)
    final = {'status': 'REMOTE_BACKUP_SEALED', 'archive_sha256': sha(archive),
             'archive_bytes': archive.stat().st_size, 'files_manifest_sha256': sha(output/'files_manifest.json'),
             'audit_sha256': sha(output/'audit.json'), 'files': len(files), 'bytes': index['total_bytes'],
             'finished_unix': time.time(), 'shutdown_performed': False}
    write(output/'archive_receipt.json', final)
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.base, args.output)
