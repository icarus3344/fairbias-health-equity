"""Hash-verified F-only replay of a failed BM representation search."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
from pilot_nhis_numerical_recovery import ROOT, json_default, sha, validate_job


def run(job_path, output, *, recover=False, use_mds_retry=False):
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('BM replay requires the original Python hash seed 0')
    job, _ = validate_job(job_path, output)
    if job['config']['method'] != 'FAIRBIAS_BM':
        raise ValueError('BM replay requires a registered BM candidate')
    hashes = {str(p.relative_to(ROOT)): sha(p)
              for package in ('fairbias', 'nhis_fairbias')
              for p in (ROOT / 'src' / package).rglob('*.py')}
    import joblib
    from pilot_scheduled_joint import runtime_modules
    from nhis_fairbias.benchmark.bm_geometry_diagnostics import replay_fairbias_fit
    from nhis_fairbias.benchmark.joint_mds_numpy import numpy_mds_acceleration
    from nhis_fairbias.benchmark.adapters.adapter_bm_recovery import BMRecoveryAdapter
    runtime_modules(ROOT, hashes)
    data = joblib.load(job['data_path'])
    runtime_modules(ROOT, hashes)
    if data['data_identity'] != job['data_identity'] or 'evaluation_T' in data['partitions']:
        raise ValueError('Prepared data contract mismatch')
    F = data['partitions']['fitting_F']
    if F.role != 'fitting_F' or F.year != 2022 or F.arm_id != job['config']['arm_id']:
        raise ValueError('F identity mismatch')
    del data
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {'diagnostic_only': True, 'formal_benchmark_admission': False,
              'job_sha256': sha(job_path), 'job_id': job_path.parent.name,
              'data_sha256': job['data_sha256'], 'source_identity': job['source_identity'],
              'project_source_hashes': hashes, 'script_sha256': sha(__file__),
              'S_T_evaluated': False, 'partitions_used': ['fitting_F'], 'status': 'STARTED'}
    record['recovery_options'] = {'recover': recover, 'use_mds_retry': use_mds_retry}
    (output / 'result.json').write_text(json.dumps(record, indent=2)+'\n')
    try:
        if recover:
            params = dict(job['config']['params'], backbone=job['config']['backbone'],
                          arm_id=job['config']['arm_id'], random_state=job['seed'])
            adapter = BMRecoveryAdapter(**params, use_mds_retry=use_mds_retry)
            try:
                adapter.fit_representation(F.X_semantic, F.y, F.A)
                if adapter.converged_:
                    path = output / 'representation.joblib'
                    joblib.dump(adapter, path)
                    loaded = joblib.load(path)
                    if loaded.changed_dict_ != adapter.changed_dict_:
                        raise ValueError('Representation reload mismatch')
                    record.update(status='FC_REPRESENTATION_PASS', model_sha256=sha(path))
                else:
                    record['status'] = 'SEARCH_STOPPED_INFEASIBLE'
            except Exception as exc:
                record.update(status=getattr(adapter, 'recovery_receipt_', {}).get('status', 'FAILED'),
                              error_type=type(exc).__name__)
            finally:
                record['recovery'] = getattr(adapter, 'recovery_receipt_', {})
                record['fit_manifest'] = adapter.fit_manifest_
        else:
            with numpy_mds_acceleration() as acceleration:
                replay = replay_fairbias_fit(F.X_semantic, F.A, y_F=F.y,
                                            config=job['config'], seed=job['seed'])
            record.update(status='REPLAY_COMPLETED', replay=replay, acceleration=acceleration)
    except Exception as exc:
        record.update(status='REPLAY_ERROR', error_type=type(exc).__name__)
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        record['loaded_project_modules'] = runtime_modules(ROOT, hashes)
        (output / 'result.json').write_text(json.dumps(record, indent=2, allow_nan=False, default=json_default)+'\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recover', action='store_true')
    parser.add_argument('--use-mds-retry', action='store_true')
    args = parser.parse_args()
    run(args.job.resolve(), args.output.resolve(), recover=args.recover, use_mds_retry=args.use_mds_retry)
