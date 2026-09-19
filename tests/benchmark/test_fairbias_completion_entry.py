"""New completion entry: synthetic 80-task plan, F/C-only fit and provenance."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

import joblib
import pytest

from test_joint_ae_recovery import data

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_nhis_fairbias_completion as entry


@pytest.fixture
def setup_plan(tmp_path, monkeypatch):
    XF, yF, AF, XC, yC, AC = data()
    prepared, configs = {}, []
    original = tmp_path / 'original'
    original.mkdir()
    for arm in range(1, 5):
        arm_id = f'arm_{arm:03d}'
        def partition(X, y, A, role):
            return SimpleNamespace(X_semantic=X, y=y, A=A, role=role, year=2022, arm_id=arm_id,
                                   record_keys=[f'{role}_{i}' for i in range(len(y))])
        payload = {'data_identity': arm_id, 'partitions': {
            'fitting_F': partition(XF, yF, AF, 'fitting_F'),
            'calibration_C': partition(XC, yC, AC, 'calibration_C'),
            'selection_S': {'not_used': True}}}
        path = tmp_path / (arm_id + '.joblib')
        joblib.dump(payload, path)
        prepared[arm_id] = {'path': str(path), 'sha256': entry.sha(path), 'data_identity': arm_id}
        for method in entry.METHODS:
            for backbone in ('LR', 'GBDT'):
                cid = f'{arm_id}_{method}_{backbone}'
                c = dict(candidate_id=cid, arm_id=arm_id, method=method, backbone=backbone,
                         seeds=list(entry.SEEDS), training_weighted=False,
                         params=dict(mode='JOINT' if method.endswith('JOINT') else 'BM_AE',
                                     epsilon_ratio=.75, max_outer_iterations=10))
                configs.append(c)
                for seed in entry.SEEDS:
                    job = dict(config=c, seed=seed, source_identity='synthetic',
                               data_identity=arm_id, data_sha256=entry.sha(path))
                    p = original / 'jobs' / f'{cid}_s{seed}' / 'job.json'
                    p.parent.mkdir(parents=True)
                    p.write_text(json.dumps(job))
    r = original / 'registration.json'
    r.write_text(json.dumps(dict(candidates=configs, prepared=prepared,
                                source_identity='synthetic', source_files=entry.source_hashes())))
    monkeypatch.setattr(entry, 'REGISTRATION_SHA256', entry.sha(r))
    manifest = tmp_path / 'manifest.json'
    value = entry.prepare(r, manifest)
    return manifest, value


def test_plan_contains_all80_no_old_sources_or_inputs_modified(setup_plan):
    path, value = setup_plan
    assert len(value['jobs']) == 80
    assert entry.load_manifest(path) == value
    assert value['wall_timeout_seconds'] is None and value['max_replays_per_session'] is None
    assert value['S_T_evaluation_authorized'] is False and value['T_already_known'] is True
    broken = copy.deepcopy(value['jobs'])
    broken[0] = broken[1]
    with pytest.raises(ValueError, match='80 unique'):
        entry.validate_coverage(broken)
    with pytest.raises(ValueError, match='80 unique'):
        entry.validate_coverage(value['jobs'][:-1])


def test_manifest_tamper_or_changed_sources_prevents_input_load(setup_plan, tmp_path, monkeypatch):
    path, value = setup_plan
    value['initial_ae_cap'] = 100
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='integrity'):
        entry.load_manifest(path)
    value['initial_ae_cap'] = 40
    path.write_text(json.dumps(value))
    monkeypatch.setattr(entry, 'source_hashes', lambda: {})
    with pytest.raises(ValueError, match='source closure'):
        entry.pilot(path, value['jobs'][0]['job_id'], '/does/not/exist', tmp_path / 'out', tmp_path / 'cache')


@pytest.mark.parametrize('mode', ['BM_AE', 'JOINT'])
def test_actual_synthetic_fc_completion_save_reload(setup_plan, tmp_path, monkeypatch, mode):
    path, value = setup_plan
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(name, '1')
    job = next(j for j in value['jobs'] if j['config']['method'] == 'FAIRBIAS_' + mode and j['seed'] == 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        result = entry.pilot(path, job['job_id'], job['prepared']['path'], tmp_path / 'output', tmp_path / 'cache')
    assert result['status'] == 'FC_COMPLETE_FEASIBLE'
    assert result['reload_exact'] and result['S_T_evaluated'] is False
    assert result['completion']['attempts'][0]['limits']['max_outer_iterations'] == 40
    assert result['completion']['status'] == 'COMPLETE_FEASIBLE'
    assert result['formal_benchmark_admission'] is False
    assert entry.sha(tmp_path / 'output' / 'policy.joblib') == result['model_sha256']
    assert json.loads((tmp_path / 'output' / 'result.json').read_text())['status'] == 'FC_COMPLETE_FEASIBLE'


def test_pilot_interrupt_retained_without_admitting_model(setup_plan, tmp_path, monkeypatch):
    from nhis_fairbias.benchmark.fairbias_completion import FairBiasCompletionAdapter
    path, value = setup_plan
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(name, '1')
    def interrupt(self, *a, **kw):
        self.provenance_ = {'status': 'INTERRUPTED'}
        raise KeyboardInterrupt()
    monkeypatch.setattr(FairBiasCompletionAdapter, 'fit_development', interrupt)
    job = value['jobs'][0]
    result = entry.pilot(path, job['job_id'], job['prepared']['path'], tmp_path / 'out', tmp_path / 'cache')
    assert result['status'] == 'INTERRUPTED_REPLAY_REQUIRED'
    assert 'model_sha256' not in result and not (tmp_path / 'out' / 'policy.joblib').exists()
    assert result['S_T_evaluated'] is False


def test_prepared_hash_checked_before_unpickle(setup_plan, tmp_path, monkeypatch):
    path, value = setup_plan
    job = value['jobs'][0]
    Path(job['prepared']['path']).write_bytes(b'changed')
    def forbidden(*a, **k): raise AssertionError('unpickle must not run')
    monkeypatch.setattr(joblib, 'load', forbidden)
    with pytest.raises(ValueError, match='prepared input hash'):
        entry.pilot(path, job['job_id'], job['prepared']['path'], tmp_path / 'out', tmp_path / 'cache')
