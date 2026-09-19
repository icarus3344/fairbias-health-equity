import json
import joblib
import numpy as np
from nhis_fairbias.benchmark.data_contracts import generate_synthetic_nhis_cohort, load_arm_partitions, ARM_SPECS
from nhis_fairbias.benchmark.preprocessing import BenchmarkPreprocessor
from nhis_fairbias.benchmark.experiment_registry import enumerate_candidates, identity
from nhis_fairbias.benchmark.experiment_worker import execute_job, write_json, file_sha


def test_registered_matrix_has_complete_seeds_and_separate_weighted_variants():
    matrix = enumerate_candidates()
    assert len({c['candidate_id'] for c in matrix}) == len(matrix)
    fair = [c for c in matrix if c['method'] == 'FAIRBIAS_BM']
    assert all(c['seeds'] == [0, 7, 19, 37, 73] for c in fair)
    assert {c['training_weighted'] for c in fair} == {True, False}
    assert all(c['status'] == 'NOT_SUPPORTED' for c in matrix if c['method'] == 'LFR_RECONSTRUCTED' and c['arm_id'] == 'arm_002')


def test_isolated_fit_reloads_and_rejects_evaluation_partition(tmp_path):
    cohort = generate_synthetic_nhis_cohort(n_records_per_year=140, n_strata=10, seed=5)
    parts = load_arm_partitions(cohort, 'arm_001')
    t = parts.pop('evaluation_T')
    F, C, S = [parts[k] for k in ('fitting_F', 'calibration_C', 'selection_S')]
    prep = BenchmarkPreprocessor(ARM_SPECS['arm_001']['features']).fit(F.X_semantic)
    data = {'partitions': parts, 'preprocessor': prep, 'data_identity': 'synthetic',
            'X_F': prep.transform(F.X_semantic), 'X_C': prep.transform(C.X_semantic), 'X_S': prep.transform(S.X_semantic)}
    data_path = tmp_path / 'data.joblib'
    joblib.dump(data, data_path)
    (tmp_path / 'cache').mkdir()
    config = next(c for c in enumerate_candidates(include_recent=False) if c['method'] == 'UNMITIGATED' and not c.get('training_weighted'))
    for name, leak in [('valid', False), ('reject', True)]:
        path = tmp_path / name
        path.mkdir()
        if leak:
            data['partitions']['evaluation_T'] = t
            joblib.dump(data, data_path)
        job = {'config': config, 'seed': 0, 'data_path': str(data_path), 'data_identity': 'synthetic',
               'data_sha256': file_sha(data_path), 'cache_path': str(tmp_path / 'cache'), 'source_identity': 'synthetic'}
        write_json(path / 'job.json', job)
        result = execute_job(path / 'job.json')
        assert result['status'] == ('FAILED' if leak else 'VALID'), result.get('traceback')
        if not leak:
            assert result['reload_verified']
            predictions = np.load(path / 'predictions_S.npz')
            assert set(predictions.files) == {'q', 'p'}
            assert set(predictions['q']).issubset({0., 1.})
        else:
            assert 'refuses evaluation data' in result['error']
