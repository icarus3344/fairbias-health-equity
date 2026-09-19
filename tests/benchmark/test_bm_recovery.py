"""Generated contracts for explicit BM terminal outcomes, no NHIS data."""
from contextlib import contextmanager
import numpy as np
import pytest
from nhis_fairbias.benchmark.adapters import adapter_bm_recovery as recovery


def test_only_empty_committed_intermediate_geometry_is_search_exhaustion():
    validate = recovery.BMRecoveryAdapter._validate_epsilon
    with pytest.raises(recovery.SearchFeatureExhausted):
        validate({'protected': {}}, 'intermediate F')
    for table, stage in [({}, 'intermediate F'), ({'p': {}}, 'initial F'),
                         ({'p': {'x': np.nan}}, 'intermediate F'),
                         ({'p': {'x': -1}}, 'intermediate F')]:
        with pytest.raises(ValueError) as caught:
            validate(table, stage)
        assert not isinstance(caught.value, recovery.SearchFeatureExhausted)
    validate({'p': {'x': 0}}, 'intermediate F')


def test_exhausted_fit_has_no_model_and_records_restored_context(monkeypatch):
    calls = []
    @contextmanager
    def context():
        calls.append('enter')
        try:
            yield {'synthetic': True}
        finally:
            calls.append('exit')
    def fit(self, *args):
        self._validate_epsilon({'p': {}}, 'intermediate F')
    monkeypatch.setattr(recovery, 'numpy_mds_acceleration', context)
    monkeypatch.setattr(recovery.FairBiasAdapter, 'fit_representation', fit)
    adapter = recovery.BMRecoveryAdapter()
    with pytest.raises(recovery.SearchFeatureExhausted):
        adapter.fit_representation(None, None, None)
    assert calls == ['enter', 'exit']
    assert adapter.recovery_receipt_['status'] == 'SEARCH_FEATURE_EXHAUSTED'
    assert not adapter.recovery_receipt_['global_infeasibility_claim']
    assert not adapter.fitted_ and not adapter.converged_


@pytest.mark.parametrize('flag', [1, 'true', None])
def test_retry_is_explicit_boolean(flag):
    with pytest.raises(ValueError):
        recovery.BMRecoveryAdapter(use_mds_retry=flag)
