"""Independent F/C pilot of strict kernel / explicit AE and MDS budget recovery."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import time
from pilot_nhis_numerical_recovery import ROOT, json_default, sha, validate_job


class ExternalPilotStop(BaseException):
    pass


def run(job_path, output, *, ae_cap_retry=None, use_mds_retry=False):
    job, registered_run = validate_job(job_path, output)
    hashes = {str(p.relative_to(ROOT)): sha(p)
              for package in ('fairbias', 'nhis_fairbias')
              for p in (ROOT / 'src' / package).rglob('*.py')}
    import joblib
    import numpy as np
    import pilot_scheduled_joint as helper
    import pilot_nhis_numerical_recovery as numerical
    scripts = [Path(__file__).resolve(), Path(helper.__file__).resolve(), Path(numerical.__file__).resolve()]
    expected = [ROOT / 'scripts' / name for name in (
        'pilot_nhis_joint_ae_recovery.py', 'pilot_scheduled_joint.py', 'pilot_nhis_numerical_recovery.py')]
    if scripts != [p.resolve() for p in expected]:
        raise ValueError('Pilot helpers loaded outside the source checkout')
    script_hashes = {str(p.relative_to(ROOT)): sha(p) for p in scripts}
    runtime_modules = helper.runtime_modules
    from nhis_fairbias.benchmark.adapters.adapter_joint_ae_recovery import make_joint_ae_recovery_adapter
    from nhis_fairbias.benchmark.predictions import FrozenDecisionPolicy
    runtime_modules(ROOT, hashes)
    data = joblib.load(job['data_path'])
    runtime_modules(ROOT, hashes)
    if data['data_identity'] != job['data_identity'] or 'evaluation_T' in data['partitions'] or 'X_T' in data:
        raise ValueError('Prepared identity or partition mismatch')
    F, C = data['partitions']['fitting_F'], data['partitions']['calibration_C']
    del data
    for p, role in ((F, 'fitting_F'), (C, 'calibration_C')):
        if (p.role != role or p.year != 2022 or p.arm_id != job['config']['arm_id']
                or len(p) != len(p.X_semantic) or len(p) != len(p.A)
                or len(p) != len(set(p.record_keys)) or not len(p)):
            raise ValueError('Development partition contract mismatch')
    if set(F.record_keys) & set(C.record_keys):
        raise ValueError('F/C overlap')
    adapter = make_joint_ae_recovery_adapter(job['config'], job['seed'],
        controller='strict', ae_cap_retry=ae_cap_retry, use_mds_retry=use_mds_retry)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {'status': 'STARTED', 'diagnostic_only': True, 'formal_benchmark_admission': False,
              'job_id': job_path.parent.name, 'job_sha256': sha(job_path),
              'config': job['config'], 'seed': job['seed'],
              'data_sha256': job['data_sha256'], 'source_identity': job['source_identity'],
              'data_identity': job['data_identity'], 'registration_sha256': sha(registered_run / 'registration.json'),
              'project_source_hashes': hashes, 'script_sha256': sha(__file__),
              'script_source_hashes': script_hashes,
              'recovery_options': {'ae_cap_retry': ae_cap_retry, 'use_mds_retry': use_mds_retry},
              'S_T_evaluated': False, 'partitions_used': ['fitting_F', 'calibration_C']}
    def save():
        pending = output / 'result.json.part'
        pending.write_text(json.dumps(record, indent=2, allow_nan=False, default=json_default)+'\n')
        pending.replace(output / 'result.json')
    def stop(signum, frame):
        raise ExternalPilotStop()
    previous = signal.signal(signal.SIGTERM, stop)
    save()
    try:
        adapter.fit_development(F.X_semantic, F.y, F.A, C.X_semantic, C.y, C.A,
                                metadata={'F_ids': F.record_keys, 'C_ids': C.record_keys})
        policy = FrozenDecisionPolicy().fit_calibration(adapter, C.X_semantic, C.y, C.A)
        before = policy.predict(C.X_semantic, C.A)
        path = output / 'policy.joblib'
        joblib.dump(policy, path)
        after = joblib.load(path).predict(C.X_semantic, C.A)
        if not np.array_equal(before.q_decision, after.q_decision):
            raise ValueError('Decision reload mismatch')
        if before.p_event is not None and not np.array_equal(before.p_event, after.p_event):
            raise ValueError('Risk reload mismatch')
        record.update(status='FC_PILOT_PASS', reload_exact=True, model_sha256=sha(path))
    except BaseException as exc:
        record.update(status='FAILED' if isinstance(exc, Exception) else 'INTERRUPTED', error_type=type(exc).__name__)
    finally:
        signal.signal(signal.SIGTERM, previous)
        record.update(elapsed_seconds=time.monotonic()-started, recovery=adapter.provenance_.get('recovery', {}))
        try:
            record['loaded_project_modules'] = runtime_modules(ROOT, hashes)
            if any(sha(ROOT / p) != digest for p, digest in script_hashes.items()):
                raise ValueError('Pilot script source changed during fit')
        except Exception as exc:
            record.update(status='FAILED', error_type=type(exc).__name__, failure_phase='source_integrity')
        save()
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ae-cap-retry', type=int, choices=[40])
    parser.add_argument('--use-mds-retry', action='store_true')
    args = parser.parse_args()
    result = run(args.job.resolve(), args.output.resolve(), ae_cap_retry=args.ae_cap_retry,
                 use_mds_retry=args.use_mds_retry)
    raise SystemExit(0 if result['status'] == 'FC_PILOT_PASS' else 1)
