"""Generated metadata fault probes against the actual recovery queue methods."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

import scripts.run_nhis_numerical_recovery_worker as worker
from test_numerical_recovery_queue import _fixture


def _state(scheduler, spec, *, status='VALID', returncode=None, termination=None):
    spec.path.mkdir(parents=True)
    job = scheduler._job_payload(spec)
    (spec.path / 'job.json').write_text(json.dumps(job))
    result = {'candidate_id': spec.candidate_id, 'seed': spec.seed, 'status': status,
              'source_identity': spec.source_identity, 'data_identity': spec.data['data_identity'],
              'reload_verified': status == 'VALID'}
    (spec.path / 'result.json').write_text(json.dumps(result))
    if status == 'VALID':
        (spec.path / 'model.joblib').write_bytes(b'generated unit-test model artifact')
        (spec.path / 'predictions_S.npz').write_bytes(b'generated unit-test prediction artifact')
    selected = scheduler.selected[spec.path.name]
    receipt = {
        'runtime_variant': scheduler.runtime_plan['runtime_variant'],
        'runtime_source_identity': scheduler._runtime_identity,
        'registration_sha256': worker.file_sha(scheduler.registration_path),
        'original_job_sha256': selected['original_job_sha256'],
        'original_result_sha256': selected['original_result_sha256'],
        'candidate_id': spec.candidate_id, 'seed': spec.seed, 'status': status,
        'worker_result_status': status, 'runtime_checks_passed': True,
        'files': {name: worker.file_sha(spec.path / name)
                  for name in ('job.json', 'result.json', 'model.joblib', 'predictions_S.npz')
                  if (spec.path / name).is_file()},
    }
    (spec.path / 'runtime_receipt.json').write_text(json.dumps(receipt))
    return SimpleNamespace(
        spec=spec, process=SimpleNamespace(returncode=(0 if status == 'VALID' else 1)
                                         if returncode is None else returncode),
        termination=termination, log_handle=(spec.path / 'worker.log').open('w'),
        started=time.monotonic(), last_rss_bytes=0,
    )


@pytest.fixture
def fault_case(tmp_path):
    scheduler, _, _, _ = _fixture(tmp_path, count=2)
    jobs = scheduler.discover()
    states = []

    def make(index=0, **kwargs):
        state = _state(scheduler, jobs[index], **kwargs)
        states.append(state)
        return state

    yield scheduler, make
    for state in states:
        state.log_handle.close()


def _rewrite_receipt(state, change):
    path = state.spec.path / 'runtime_receipt.json'
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_expected_exit_one_budget_receipt_is_accepted(fault_case):
    scheduler, make = fault_case
    state = make(status='BUDGET_EXHAUSTED', returncode=1)
    assert scheduler._receipt_error(state) is None
    result = scheduler._finalize(state)
    assert result['status'] == 'BUDGET_EXHAUSTED'
    assert not (state.spec.path / 'result.worker_unadmitted.json').exists()


@pytest.mark.parametrize('status', ['FAILED', 'BUDGET_EXHAUSTED', 'RUNTIME_FAILED'])
def test_nonvalid_runtime_receipt_cannot_admit_actual_valid_result(fault_case, status):
    scheduler, make = fault_case
    state = make(returncode=1)
    _rewrite_receipt(state, lambda r: r.update(status=status, worker_result_status=status))
    assert scheduler._receipt_error(state) is not None
    original = (state.spec.path / 'result.json').read_bytes()
    result = scheduler._finalize(state)
    assert result['status'] == 'SCHEDULER_RUNTIME_FAILED'
    assert (state.spec.path / 'result.worker_unadmitted.json').read_bytes() == original


@pytest.mark.parametrize('raw', [b'{"status":', b'[]', b'null', b'"not an object"'])
def test_malformed_or_nonobject_receipt_is_rejected_and_original_bytes_preserved(fault_case, raw):
    scheduler, make = fault_case
    state = make(returncode=1)
    path = state.spec.path / 'runtime_receipt.json'
    path.write_bytes(raw)
    assert scheduler._receipt_error(state) is not None
    result = scheduler._finalize(state)
    assert result['status'] == 'SCHEDULER_RUNTIME_FAILED'
    assert (state.spec.path / 'runtime_receipt.worker_invalid.json').read_bytes() == raw


@pytest.mark.parametrize('status', [[], {}])
def test_malformed_receipt_status_is_rejected_without_throwing(fault_case, status):
    scheduler, make = fault_case
    state = make(returncode=1)
    _rewrite_receipt(state, lambda r: r.update(status=status))
    assert scheduler._receipt_error(state) is not None


@pytest.mark.parametrize('termination,returncode', [('TIME_LIMIT', -15), (None, -9), (None, 1)])
def test_terminated_or_nonzero_valid_worker_is_not_admitted(fault_case, termination, returncode):
    scheduler, make = fault_case
    state = make(termination=termination, returncode=returncode)
    assert scheduler._receipt_error(state) is not None
    result = scheduler._finalize(state)
    assert result['status'] == 'SCHEDULER_RUNTIME_FAILED'
    neighbor = make(1)
    assert scheduler._finalize(neighbor)['status'] == 'VALID'
    assert scheduler._finalized_count == 2


@pytest.mark.parametrize('raw', [b'{"status":"VALID",', b'[]', b'{}', b'{"status":[]}'])
def test_malformed_result_is_archived_and_neighbor_can_finalize(fault_case, raw):
    scheduler, make = fault_case
    state = make(returncode=1)
    path = state.spec.path / 'result.json'
    path.write_bytes(raw)
    _rewrite_receipt(state, lambda r: r['files'].update({'result.json': worker.file_sha(path)}))
    result = scheduler._finalize(state)
    assert result['status'] == 'SCHEDULER_RUNTIME_FAILED'
    assert (state.spec.path / 'result.worker_unadmitted.json').read_bytes() == raw
    neighbor = make(1)
    assert scheduler._finalize(neighbor)['status'] == 'VALID'
    assert scheduler._finalized_count == 2 and scheduler._failed_count == 1


@pytest.mark.parametrize('has_model', [False, True])
def test_verified_failed_or_valid_cache_sidecar_is_preserved_without_orphan(fault_case, has_model):
    scheduler, make = fault_case
    state = make(status='FAILED')
    job = scheduler._job_payload(state.spec)
    cache, key = scheduler.cache_path, state.spec.representation_key
    status = {'key': key, 'status': 'VALID' if has_model else 'BUDGET_EXHAUSTED_OR_OPTIMIZATION_FAILURE'}
    (cache / (key + '.json')).write_text(json.dumps(status))
    if has_model:
        (cache / (key + '.joblib')).write_bytes(b'generated cache model bytes')
    sidecar = cache / (key + '.runtime.json')
    sidecar.write_text(json.dumps(worker._cache_evidence(job, key)))
    before = {path.name: worker.file_sha(path) for path in cache.iterdir() if path.is_file()}
    scheduler._seal_failed_cache(state.spec, 'TIME_LIMIT')
    assert worker.verify_cache_entry(job, key) is True
    assert not (cache / (key + '.scheduler_orphan.json')).exists()
    assert before == {path.name: worker.file_sha(path) for path in cache.iterdir() if path.is_file()}


def test_unverified_valid_cache_is_marked_orphan_without_being_blessed(fault_case):
    scheduler, make = fault_case
    state = make(status='FAILED')
    cache, key = scheduler.cache_path, state.spec.representation_key
    (cache / (key + '.json')).write_text(json.dumps({'status': 'VALID', 'key': key}))
    (cache / (key + '.joblib')).write_bytes(b'unaccepted representation')
    scheduler._seal_failed_cache(state.spec, 'TIME_LIMIT')
    assert (cache / (key + '.scheduler_orphan.json')).is_file()
    assert not (cache / (key + '.runtime.json')).exists()
    with pytest.raises(ValueError, match='without runtime provenance'):
        worker.verify_cache_entry(scheduler._job_payload(state.spec), key)


def test_timeout_before_worker_cache_init_keeps_namespace_valid(fault_case):
    scheduler, make = fault_case
    state = make(status='FAILED', termination='TIME_LIMIT', returncode=-15)
    assert (scheduler.cache_path / worker.CACHE_NAMESPACE_FILE).is_file()
    scheduler._seal_failed_cache(state.spec, 'TIME_LIMIT')
    job = scheduler._job_payload(state.spec)
    worker.initialize_cache(job)
    assert worker.verify_cache_entry(job, state.spec.representation_key)
