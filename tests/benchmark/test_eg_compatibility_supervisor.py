"""Independent admission probes for the legacy-EG compatibility audit."""
import json

import joblib
import numpy as np
import pytest

from scripts.audit_nhis_eg_compatibility import audit_eg_archive, _exact_equal
from nhis_fairbias.benchmark.adapters.adapter_reductions import ExponentiatedGradientAdapter
from nhis_fairbias.benchmark.adapters.adapter_reductions_numerical import NumericallyRecoveredExponentiatedGradientAdapter
from nhis_fairbias.benchmark.experiment_worker import file_sha
from test_eg_compatibility import _make_archive, _write_json


def seal(directory):
    path = directory / 'receipt.json'
    receipt = json.loads(path.read_text())
    for name in receipt['files']:
        receipt['files'][name] = file_sha(directory / name)
    _write_json(path, receipt)


@pytest.mark.parametrize('fault', ['seed', 'data', 'weights', 'empty_q', 'bad_q', 'result_data', 'missing_rc', 'escape'])
def test_rejects_real_admission_faults(tmp_path, fault):
    archive, root, registration = _make_archive(tmp_path)
    directory = archive / 'jobs/eg1_s0'
    if fault in {'seed', 'data', 'weights'}:
        artifact = joblib.load(directory / 'model.joblib')
        if fault == 'seed': artifact['policy']._adapter.random_state = 7
        if fault == 'data': artifact['data_identity'] = 'different'
        if fault == 'weights': artifact['policy']._adapter.model.weights_ = np.array([-.1, 1.1])
        joblib.dump(artifact, directory / 'model.joblib')
        seal(directory)
    elif fault in {'empty_q', 'bad_q'}:
        np.savez(directory / 'predictions_S.npz', q=np.array([] if fault == 'empty_q' else [1.01]))
        seal(directory)
    elif fault == 'result_data':
        result = json.loads((directory / 'result.json').read_text())
        result['data_identity'] = 'different'
        _write_json(directory / 'result.json', result)
        seal(directory)
    else:
        receipt = json.loads((directory / 'receipt.json').read_text())
        if fault == 'missing_rc': receipt.pop('returncode')
        if fault == 'escape': receipt['files']['../result.json'] = '0' * 64
        _write_json(directory / 'receipt.json', receipt)
    with pytest.raises(ValueError):
        audit_eg_archive(archive, root, file_sha(registration), 1, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_exact_numeric_constructor_normalization_is_not_tolerance():
    assert _exact_equal({'C': 1.0}, {'C': 1})
    assert not _exact_equal({'C': 1.00000001}, {'C': 1})
    assert not _exact_equal(True, 1)


def test_real_generated_eg_fit_is_unchanged_by_recovery_subclass():
    rng = np.random.default_rng(91)
    X = rng.normal(size=(80, 3))
    y = np.tile([0, 1, 1, 0], 20)
    A = np.tile([0, 0, 1, 1], 20)
    params = dict(constraint_type='demographic_parity', difference_bound=.1,
                  eps=.01, max_iter=5, C=1, backbone='LR', random_state=7)
    original = ExponentiatedGradientAdapter(**params)
    recovery = NumericallyRecoveredExponentiatedGradientAdapter(**params)
    assert type(recovery).fit is type(original).fit
    assert original.model.estimator.get_params() == recovery.model.estimator.get_params()
    np.random.seed(2)
    original.fit(X, y, A)
    np.random.seed(2)
    recovery.fit(X, y, A)
    assert np.array_equal(original.model.weights_.to_numpy(), recovery.model.weights_.to_numpy())
    raw = original.predict_decision_proba(X)
    recovered = recovery.predict_decision_proba(X)
    assert np.array_equal(raw[(raw >= 0) & (raw <= 1)], recovered[(raw >= 0) & (raw <= 1)])
