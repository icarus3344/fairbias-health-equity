"""Generated-only F/C pilot integration and fail-closed admission checks."""

import importlib.util
import json
from pathlib import Path
import sys
import types

import joblib
import numpy as np
import pandas as pd
import pytest

from nhis_fairbias.benchmark.data_contracts import PartitionDataset
from nhis_fairbias.benchmark.adapters.adapter_lfr_recovery import LFRAnalyticRecoveryAdapter


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'numerical_recovery_pilot_tested', ROOT / 'scripts/pilot_nhis_numerical_recovery.py'
)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class SelectionAccessGuard(dict):
    """Pickle-compatible payload whose S entries cannot be retrieved."""

    def __getitem__(self, key):
        if key in {'selection_S', 'X_S'}:
            raise AssertionError('Selection partition was accessed')
        return super().__getitem__(key)

    def get(self, key, *args):
        if key in {'selection_S', 'X_S'}:
            raise AssertionError('Selection partition was accessed')
        return super().get(key, *args)


class SelectionAttributeGuard:
    @property
    def y(self):
        raise AssertionError('Selection labels were accessed')

    @property
    def A(self):
        raise AssertionError('Selection groups were accessed')

    @property
    def year(self):
        raise AssertionError('Selection metadata were accessed')


def _partition(role, n, seed):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n, 6))
    y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(int)
    a = np.array([1] * (17 if n == 48 else n // 2) + [2] * (31 if n == 48 else n - n // 2))
    partition = PartitionDataset(
        role=role, year=2022,
        record_keys=np.array([role + ':' + str(i) for i in range(n)]),
        X_semantic=pd.DataFrame(x), y=y, A=a,
        WTFA_A=np.ones(n), PSTRAT=np.ones(n, dtype=int), PPSU=np.arange(n),
        feature_names=tuple('f' + str(i) for i in range(6)), arm_id='ARM_SYNTHETIC', metadata={},
    )
    return partition, x


@pytest.fixture
def case(tmp_path, monkeypatch):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(name, '1')
    monkeypatch.setattr(sys, 'path', list(sys.path))
    run = tmp_path / 'registered_synthetic'
    job_file = run / 'jobs' / 'synthetic_s7' / 'job.json'
    job_file.parent.mkdir(parents=True)
    prepared = run / 'prepared' / 'ARM_SYNTHETIC.joblib'
    prepared.parent.mkdir()
    F, xf = _partition('fitting_F', 48, 9)
    C, xc = _partition('calibration_C', 24, 19)
    data = SelectionAccessGuard(
        data_identity='generated-only', X_F=xf, X_C=xc, X_S=SelectionAttributeGuard(),
        partitions=SelectionAccessGuard(fitting_F=F, calibration_C=C, selection_S=SelectionAttributeGuard()),
    )
    joblib.dump(data, prepared)
    config = {
        'candidate_id': 'synthetic_lfr', 'method': 'LFR_RECONSTRUCTED', 'arm_id': 'ARM_SYNTHETIC',
        'seeds': [7], 'backbone': 'LR', 'training_weighted': False,
        'params': {'k': 3, 'Ax': 0.01, 'Ay': 1.0, 'Az': 0.5, 'maxiter': 200, 'maxfun': 300},
    }
    source = 'src/nhis_fairbias/benchmark/adapters/adapter_lfr.py'
    registration = {
        'candidates': [config], 'source_identity': 'synthetic-source',
        'source_files': {source: pilot.sha(ROOT / source)},
        'prepared': {'ARM_SYNTHETIC': {'sha256': pilot.sha(prepared), 'data_identity': 'generated-only'}},
    }
    job = {'config': config, 'seed': 7, 'source_identity': 'synthetic-source',
           'data_identity': 'generated-only', 'data_sha256': pilot.sha(prepared), 'data_path': str(prepared)}

    def write():
        (run / 'registration.json').write_text(json.dumps(registration))
        job_file.write_text(json.dumps(job))

    write()
    return types.SimpleNamespace(run=run, job_file=job_file, output=tmp_path / 'fresh_pilot',
                                 prepared=prepared, job=job, registration=registration,
                                 data=data, write=write, F=F, C=C, xf=xf, xc=xc)


def test_real_lfr_fit_calibration_saved_reload_and_numpy_json_without_s_access(case, monkeypatch):
    actual_fit = LFRAnalyticRecoveryAdapter.fit
    calls = []

    def observed_fit(self, x, y, a, sample_weight=None):
        calls.append((np.array(x), np.array(y), np.array(a)))
        actual_fit(self, x, y, a, sample_weight=sample_weight)
        self.optimization_result_['synthetic_numpy_diagnostics'] = {
            'integer': np.int64(3), 'boolean': np.bool_(True), 'array': np.array([0.2, 0.3]),
            'float': np.float32(0.25),
        }
        return self

    monkeypatch.setattr(LFRAnalyticRecoveryAdapter, 'fit', observed_fit)
    report = pilot.run_pilot(case.job_file, case.output, 'lfr_analytic_v1')
    assert report['status'] == 'FC_PILOT_PASS', report
    assert len(calls) == 1
    for actual, expected in zip(calls[0], (case.xf, case.F.y, case.F.A)):
        np.testing.assert_array_equal(actual, expected)
    persisted = json.loads((case.output / 'result.json').read_text())
    assert persisted['optimization']['synthetic_numpy_diagnostics'] == {
        'integer': 3, 'boolean': True, 'array': [0.2, 0.3], 'float': 0.25,
    }
    assert persisted['optimization']['training_n'] == len(case.F)
    assert persisted['partitions_used'] == ['fitting_F', 'calibration_C']
    assert persisted['rows'] == {'F': 48, 'C': 24}
    assert persisted['S_T_evaluated'] is False and persisted['reload_exact'] is True
    assert persisted['formal_benchmark_admission'] is False
    assert persisted['loaded_project_modules_before_load']
    assert persisted['loaded_project_modules_before_fit']
    assert persisted['loaded_project_modules']
    policy_path = case.output / 'policy.joblib'
    policy = joblib.load(policy_path)
    assert isinstance(policy._adapter, LFRAnalyticRecoveryAdapter)
    assert policy._adapter.lfr.learned_model is not None
    assert policy.is_calibrated
    assert policy.predict(case.xc, case.C.A).q_decision.shape == (24,)
    assert pilot.sha(policy_path) == persisted['model_sha256']
    assert not (case.output / 'result.json.part').exists()


def test_real_eg_fit_and_saved_reload_without_s_access(case):
    config = case.job['config']
    config.update(method='EG_DP', candidate_id='synthetic_eg', params={
        'constraint_type': 'demographic_parity', 'difference_bound': 0.2, 'max_iter': 3,
    })
    case.write()
    report = pilot.run_pilot(case.job_file, case.output, 'eg_roundoff_v1')
    assert report['status'] == 'FC_PILOT_PASS', report
    assert report['reload_exact'] and report['roundoff']['all_interior_unchanged']
    policy = joblib.load(case.output / 'policy.joblib')
    assert len(policy._adapter.model.predictors_) > 0
    assert policy.predict(case.xc, case.C.A).p_event is None


@pytest.mark.parametrize('bad', [
    'variant', 'method', 'source_identity', 'source_hash', 'unknown_source', 'absolute_source',
    'escape_source', 'empty_sources', 'data_hash', 'data_path', 'candidate', 'seed',
    'weighted', 'existing_output', 'output_inside_run',
])
def test_admission_rejects_invalid_method_source_inputs_and_output_before_load(case, monkeypatch, bad):
    variant, output = 'lfr_analytic_v1', case.output
    if bad == 'variant':
        variant = 'unknown_variant'
    elif bad == 'method':
        case.job['config']['method'] = 'UNMITIGATED'
    elif bad == 'source_identity':
        case.job['source_identity'] = 'unknown-source'
    elif bad == 'source_hash':
        source = next(iter(case.registration['source_files']))
        case.registration['source_files'][source] = '0' * 64
    elif bad == 'unknown_source':
        case.registration['source_files'] = {'src/not_a_registered_source.py': '0' * 64}
    elif bad == 'absolute_source':
        source = ROOT / 'src/nhis_fairbias/benchmark/adapters/adapter_lfr.py'
        case.registration['source_files'] = {str(source): pilot.sha(source)}
    elif bad == 'escape_source':
        case.registration['source_files'] = {'../outside.py': '0' * 64}
    elif bad == 'empty_sources':
        case.registration['source_files'] = {}
    elif bad == 'data_hash':
        case.prepared.write_bytes(case.prepared.read_bytes() + b'corruption')
    elif bad == 'data_path':
        alternative = case.prepared.parent / 'unregistered.joblib'
        alternative.write_bytes(case.prepared.read_bytes())
        case.job['data_path'] = str(alternative)
    elif bad == 'candidate':
        case.registration['candidates'] = []
    elif bad == 'seed':
        case.job['seed'] = 99
    elif bad == 'weighted':
        case.job['config']['training_weighted'] = True
    elif bad == 'existing_output':
        output.mkdir()
    else:
        output = case.run / 'forbidden_pilot'
    case.write()
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('Rejected admission loaded data'))
    with pytest.raises(ValueError):
        pilot.run_pilot(case.job_file, output, variant)
    assert not (output / 'result.json').exists()


@pytest.mark.parametrize('bad', ['F_role', 'C_year', 'arm', 'rows', 'overlap', 'evaluation_T'])
def test_invalid_fc_payload_rejected_before_fit(case, monkeypatch, bad):
    from dataclasses import replace

    if bad == 'F_role':
        case.data['partitions']['fitting_F'] = replace(case.F, role='selection_S')
    elif bad == 'C_year':
        case.data['partitions']['calibration_C'] = replace(case.C, year=2023)
    elif bad == 'arm':
        case.data['partitions']['fitting_F'] = replace(case.F, arm_id='unknown')
    elif bad == 'rows':
        case.data['X_F'] = case.xf[:-1]
    elif bad == 'overlap':
        case.data['partitions']['calibration_C'] = replace(case.C, record_keys=case.F.record_keys[:24])
    else:
        case.data['partitions']['evaluation_T'] = SelectionAttributeGuard()
    joblib.dump(case.data, case.prepared)
    digest = pilot.sha(case.prepared)
    case.job['data_sha256'] = digest
    case.registration['prepared']['ARM_SYNTHETIC']['sha256'] = digest
    case.write()
    monkeypatch.setattr(LFRAnalyticRecoveryAdapter, 'fit', lambda *a, **k: pytest.fail('Invalid payload was fitted'))
    with pytest.raises(ValueError):
        pilot.run_pilot(case.job_file, case.output, 'lfr_analytic_v1')
    assert not case.output.exists()


@pytest.mark.parametrize('kind', ['outside', 'unknown', 'no_origin'])
def test_loaded_shadow_source_is_rejected_before_prepared_load(case, monkeypatch, tmp_path, kind):
    fake = types.ModuleType('nhis_fairbias.pilot_shadow')
    if kind == 'outside':
        fake.__file__ = str(tmp_path / 'outside.py')
    elif kind == 'unknown':
        fake.__file__ = str(ROOT / 'src/nhis_fairbias/nonexistent_recovery_source.py')
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('Source rejection loaded data'))
    with pytest.raises(ValueError, match='outside source tree|source mismatch|no verifiable source'):
        pilot.run_pilot(case.job_file, case.output, 'lfr_analytic_v1')
    assert not case.output.exists()


def test_source_introduced_during_unpickle_is_rejected_before_fit(case, monkeypatch, tmp_path):
    actual_load = joblib.load

    def load_with_shadow(*args, **kwargs):
        data = actual_load(*args, **kwargs)
        fake = types.ModuleType('nhis_fairbias.unpickle_shadow')
        fake.__file__ = str(tmp_path / 'outside.py')
        monkeypatch.setitem(sys.modules, fake.__name__, fake)
        return data

    monkeypatch.setattr(joblib, 'load', load_with_shadow)
    monkeypatch.setattr(LFRAnalyticRecoveryAdapter, 'fit', lambda *a, **k: pytest.fail('Unverified source was fitted'))
    with pytest.raises(ValueError, match='outside source tree'):
        pilot.run_pilot(case.job_file, case.output, 'lfr_analytic_v1')
    assert not case.output.exists()


def test_foreign_runtime_helper_is_rejected_before_load(case, monkeypatch, tmp_path):
    helper = types.ModuleType('pilot_scheduled_joint')
    helper.__file__ = str(tmp_path / 'foreign_helper.py')
    monkeypatch.setitem(sys.modules, 'pilot_scheduled_joint', helper)
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('Foreign helper loaded data'))
    with pytest.raises(ValueError, match='source helper loaded outside'):
        pilot.run_pilot(case.job_file, case.output, 'lfr_analytic_v1')


def test_source_failure_during_fit_is_saved_as_failed_not_a_pass(case, monkeypatch, tmp_path):
    actual_fit = LFRAnalyticRecoveryAdapter.fit

    def fit_with_shadow(self, *args, **kwargs):
        actual_fit(self, *args, **kwargs)
        fake = types.ModuleType('nhis_fairbias.fit_shadow')
        fake.__file__ = str(tmp_path / 'outside.py')
        monkeypatch.setitem(sys.modules, fake.__name__, fake)
        return self

    monkeypatch.setattr(LFRAnalyticRecoveryAdapter, 'fit', fit_with_shadow)
    report = pilot.run_pilot(case.job_file, case.output, 'lfr_analytic_v1')
    persisted = json.loads((case.output / 'result.json').read_text())
    assert report['status'] == persisted['status'] == 'FAILED'
    assert persisted['failure_phase'] == 'source_integrity'
    assert persisted['formal_benchmark_admission'] is False


def test_incomplete_optimizer_is_saved_without_policy_artifact(case):
    case.job['config']['params'].update(maxiter=1, maxfun=1)
    case.write()
    report = pilot.run_pilot(case.job_file, case.output, 'lfr_analytic_v1')
    persisted = json.loads((case.output / 'result.json').read_text())
    assert report['status'] == persisted['status'] == 'OPTIMIZATION_INCOMPLETE'
    assert persisted['optimization']['warnflag'] != 0
    assert not (case.output / 'policy.joblib').exists()


def test_numpy_json_is_numeric_and_rejects_nonfinite_or_unknown_values():
    assert json.loads(json.dumps({'x': np.int32(2)}, default=pilot.json_default)) == {'x': 2}
    with pytest.raises(ValueError):
        json.dumps({'x': np.float32(np.nan)}, allow_nan=False, default=pilot.json_default)
    with pytest.raises(TypeError):
        json.dumps({'x': object()}, default=pilot.json_default)
