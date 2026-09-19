"""Independent generated contracts for the separate completed-80 evaluation.

No NHIS file, S/T result artifact, prepared container, or fitted model is read.
Coverage/release checks use generated dictionaries; statistics use tiny annual
survey designs with explicit domain rows and shared PSU replicate factors.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from nhis_fairbias.benchmark import catalog_evaluation as catalog
from nhis_fairbias.benchmark.metrics import compute_survey_fairness_metrics
from nhis_fairbias.benchmark.survey_batch import bootstrap_metric_arrays
from nhis_fairbias.benchmark.survey_linearization import linearized_survey_inference
from nhis_fairbias.benchmark.data_contracts import AnnualSurveyDesign
from test_frozen_evaluation_contract import _annual


@pytest.fixture
def completion():
    from nhis_fairbias.benchmark import completion_evaluation
    return completion_evaluation


@pytest.fixture
def completion_jobs():
    jobs = []
    for method in ('FAIRBIAS_BM_AE', 'FAIRBIAS_JOINT'):
        for arm in ('arm_001', 'arm_002', 'arm_003', 'arm_004'):
            for backbone in ('LR', 'GBDT'):
                cid = f'generated_{arm}_{backbone}_{method}'
                config = dict(candidate_id=cid, method=method, arm_id=arm, backbone=backbone,
                    seeds=[0, 7, 19, 37, 73], params={'epsilon_ratio': .75, 'max_outer_iterations': 10},
                    status='REGISTERED', training_weighted=False, complexity=1)
                for seed in config['seeds']:
                    jobs.append({'job_id': f'{cid}_s{seed}', 'config': copy.deepcopy(config), 'seed': seed})
    return jobs


def test_complete_80_preserves_exact_original_seed_matrix(completion_jobs, completion):
    before = copy.deepcopy(completion_jobs)
    completion.validate_job_coverage(completion_jobs)
    completion.validate_job_coverage(list(reversed(completion_jobs)))
    assert completion_jobs == before
    assert len(completion_jobs) == 80
    assert len({(j['config']['method'], j['config']['arm_id'], j['config']['backbone']) for j in completion_jobs}) == 16
    assert completion.BRANCH != 'registered_budget_main'
    assert tuple(completion.SEEDS) == (0, 7, 19, 37, 73)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'extra', 'wrong_seed', 'bool_seed', 'float_seed',
    'wrong_method', 'wrong_arm', 'wrong_backbone', 'wrong_job_id', 'two_candidates_one_cell',
    'different_config_one_cell', 'config_seed_missing', 'config_seed_duplicate', 'config_wrong_seed'])
def test_seed_and_one_original_config_coverage_rejected(completion_jobs, fault, completion):
    jobs = copy.deepcopy(completion_jobs)
    if fault == 'missing': jobs.pop()
    elif fault == 'duplicate': jobs[-1] = copy.deepcopy(jobs[0])
    elif fault == 'extra': jobs.append(copy.deepcopy(jobs[0]))
    elif fault == 'wrong_seed': jobs[0]['seed'] = 1
    elif fault == 'bool_seed': jobs[0]['seed'] = False
    elif fault == 'float_seed': jobs[0]['seed'] = 0.0
    elif fault == 'wrong_method': jobs[0]['config']['method'] = 'FAIRBIAS_BM'
    elif fault == 'wrong_arm': jobs[0]['config']['arm_id'] = 'arm_005'
    elif fault == 'wrong_backbone': jobs[0]['config']['backbone'] = 'MLP'
    elif fault == 'wrong_job_id': jobs[0]['job_id'] += '_different'
    elif fault == 'two_candidates_one_cell':
        jobs[0]['config']['candidate_id'] += '_second_candidate'
        jobs[0]['job_id'] = f"{jobs[0]['config']['candidate_id']}_s{jobs[0]['seed']}"
    elif fault == 'different_config_one_cell': jobs[0]['config']['params']['epsilon_ratio'] = .25
    elif fault == 'config_seed_missing': jobs[0]['config']['seeds'].pop()
    elif fault == 'config_seed_duplicate': jobs[0]['config']['seeds'][-1] = 0
    elif fault == 'config_wrong_seed': jobs[0]['config']['seeds'] = [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        completion.validate_job_coverage(jobs)


@pytest.fixture
def known_t_release(completion):
    study = {'study_branch': completion.BRANCH, 'known_T': True,
             'selection': {'path': '/generated/selection.json', 'sha256': '2' * 64}}
    release = {'schema_version': 'completion_t_release_v1', 'decision': 'SUPERVISOR_RELEASED_COMPLETION_T',
               'study_branch': completion.BRANCH, 'study_sha256': '1' * 64,
               'selection_sha256': '2' * 64, 'arm_ids': ['arm_001', 'arm_002', 'arm_003', 'arm_004'],
               'known_T': True}
    return study, release


def test_release_requires_separate_known_t_branch(known_t_release, completion):
    study, release = known_t_release
    before = copy.deepcopy((study, release))
    completion.validate_release(study, '1' * 64, release, ['arm_001'])
    completion.validate_release(study, '1' * 64, release, list(release['arm_ids']))
    assert (study, release) == before


@pytest.mark.parametrize('fault', ['study_sha', 'selection_sha', 'decision', 'schema', 'release_branch', 'study_branch',
    'known_T_false', 'known_T_missing', 'known_T_number', 'study_known_T_false', 'unreleased_arm', 'duplicate_arm', 'unknown_arm'])
def test_release_binding_faults_are_rejected(known_t_release, fault, completion):
    study, release = copy.deepcopy(known_t_release)
    if fault == 'study_sha': release['study_sha256'] = '3' * 64
    elif fault == 'selection_sha': release['selection_sha256'] = '3' * 64
    elif fault == 'decision': release['decision'] = 'SUPERVISOR_RELEASED_T'
    elif fault == 'schema': release['schema_version'] = 'nhis_catalog_t_release_v1'
    elif fault == 'release_branch': release['study_branch'] = 'registered_budget_main'
    elif fault == 'study_branch': study['study_branch'] = 'registered_budget_main'
    elif fault == 'known_T_false': release['known_T'] = False
    elif fault == 'known_T_missing': release.pop('known_T')
    elif fault == 'known_T_number': release['known_T'] = 1
    elif fault == 'study_known_T_false': study['known_T'] = False
    elif fault == 'unreleased_arm': release['arm_ids'] = ['arm_002']
    elif fault == 'duplicate_arm': release['arm_ids'].append('arm_001')
    elif fault == 'unknown_arm': release['arm_ids'].append('arm_999')
    with pytest.raises(ValueError):
        completion.validate_release(study, '1' * 64, release, ['arm_001'])


def _full_annual():
    T = _annual()
    mask = np.r_[np.ones(8, dtype=bool), np.zeros(4, dtype=bool)]
    T.annual_design = AnnualSurveyDesign(2024, np.arange(12), np.r_[T.PSTRAT, [3, 3, 3, 3]],
        np.r_[T.PPSU, [30, 30, 31, 31]], np.ones(12), mask)
    return T


def test_common_annual_psu_design_gives_zero_identical_paired_variance():
    T = _full_annual()
    design = catalog._normalized_annual_design(T)
    q = np.array([.1, .2, .1, .6, .1, .4, .1, .8])
    predictions = {'AE': design.expand(q), 'JOINT': design.expand(q.copy())}
    args = (design.expand(T.y), predictions, design.expand(T.A), design.strata, design.psus, design.weights, [1, 2])
    linear = linearized_survey_inference(*args, domain_mask=design.domain_mask, reference_method='AE')
    assert linear['degrees_of_freedom'] == 3
    assert linear['methods']['AE']['balanced_accuracy'].std_error > 0
    paired = linear['paired']['JOINT']['balanced_accuracy']
    assert paired.point_estimate == paired.std_error == paired.ci_lower == paired.ci_upper == 0
    reps = bootstrap_metric_arrays(*args, domain_mask=design.domain_mask, B=2000, seed=20260914)
    assert reps['AE'].shape == (2000, 3)
    np.testing.assert_array_equal(reps['AE'], reps['JOINT'])
    difference = reps['AE'] - reps['JOINT']
    np.testing.assert_array_equal(difference[:, 0], np.zeros(2000))
    assert np.all(difference[np.isfinite(difference)] == 0)
    np.testing.assert_array_equal(np.isnan(reps['AE']), np.isnan(reps['JOINT']))


def test_mean_seed_eo_is_not_eo_of_mean_q():
    T = _annual()
    q0 = np.tile([1., 1., 0., 0.], 2)
    q1 = 1 - q0
    metrics = [compute_survey_fairness_metrics(T.y, q, T.A, T.WTFA_A, [1, 2]) for q in (q0, q1)]
    ensemble = compute_survey_fairness_metrics(T.y, (q0 + q1) / 2, T.A, T.WTFA_A, [1, 2])
    assert np.mean([m['eo_gap'] for m in metrics]) == 1.
    assert ensemble['eo_gap'] == 0.
    assert np.mean([m['balanced_accuracy'] for m in metrics]) == ensemble['balanced_accuracy']
    d = T.annual_design
    linear = linearized_survey_inference(T.y, {'seed_0': q0, 'seed_7': q1}, T.A, d.strata, d.psus, d.weights, [1, 2])
    projection = catalog._project_mean(linear, ['seed_0', 'seed_7'], 'eo_gap')
    bands = [linear['methods'][k]['eo_gap_interval'] for k in ('seed_0', 'seed_7')]
    assert projection['lower'] == np.mean([b[0] for b in bands])
    assert projection['upper'] == np.mean([b[1] for b in bands])


@pytest.mark.parametrize('field', ['record_keys', 'strata', 'psus', 'weights', 'domain_mask'])
def test_annual_row_misalignment_rejected(field):
    T = _full_annual()
    array = getattr(T.annual_design, field)
    if field == 'domain_mask': array[0], array[8] = False, True
    else: array[0] += 100
    with pytest.raises(ValueError, match='alignment'):
        catalog._normalized_annual_design(T)


def test_singleton_and_missing_support_remain_not_estimable():
    for T, status in ((_annual(singleton=True), 'DESIGN_NOT_ESTIMABLE'), (_annual(missing_group=True), 'VALID')):
        d = catalog._normalized_annual_design(T)
        q = np.tile([.2, .8], 4)
        result = linearized_survey_inference(T.y, {'AE': q, 'JOINT': q}, T.A, d.strata, d.psus, d.weights, [1, 2], reference_method='AE')
        assert result['status'] == status
        if status == 'VALID':
            assert result['methods']['AE']['eo_gap_interval'][2] == 'NOT_ESTIMABLE'
            assert result['paired']['JOINT']['eo_gap_interval'][2] == 'NOT_ESTIMABLE'


def _statistics_case(*, opposite_seeds=False, missing=False, singleton=False):
    T = _annual(singleton=True) if singleton else _full_annual()
    d = catalog._normalized_annual_design(T)
    selections, q_models, individual = [], {}, {}
    seeds = (0, 7, 19, 37, 73)
    for short, method in (('ae', 'FAIRBIAS_BM_AE'), ('joint', 'FAIRBIAS_JOINT')):
        keys = []
        for position, seed in enumerate(seeds):
            key = f'{short}_seed_{seed}'
            if opposite_seeds:
                q = np.tile([1., 1., 0., 0.], 2) if position % 2 == 0 else np.tile([0., 0., 1., 1.], 2)
                if position == 4: q = np.full(8, .5)
            else:
                q = np.array([.1, .2, .1, .6, .1, .4, .1, .8])
                if short == 'joint': q = np.full(8, .5)
            keys.append(key)
            q_models[key] = d.expand(q)
            metrics = compute_survey_fairness_metrics(T.y, q, T.A, T.WTFA_A, [1, 2])
            individual[key] = {'weighted': metrics, 'unweighted': dict(metrics), 'risk': {}, 'unweighted_risk': {}}
        selections.append({'selection_id': short, 'arm_id': 'arm_001', 'backbone': 'LR', 'method': method,
            'training_weighted': False, 'tau': .10, 'status': 'NO_FEASIBLE_CONFIGURATION',
            'study_branch': 'fairbias_adaptive_completion_v1', 'model_ids': keys})
    contrasts = [{'contrast_id': 'ae_vs_joint', 'reference_selection_id': 'ae', 'comparison_selection_id': 'joint', 'family': 'primary'}]
    if missing:
        selections.append({**selections[-1], 'selection_id': 'missing', 'model_ids': [], 'status': 'NO_VALID_MODEL'})
        contrasts.append({'contrast_id': 'ae_vs_missing', 'reference_selection_id': 'ae', 'comparison_selection_id': 'missing', 'family': 'secondary'})
    return T, selections, q_models, individual, contrasts


def test_new_statistics_keeps_named_pairs_global_families_and_missing_slots(tmp_path, monkeypatch):
    from nhis_fairbias.benchmark import completion_statistics as module
    from scipy.stats import t
    T, selections, qs, metrics, contrasts = _statistics_case(missing=True)
    before = copy.deepcopy((selections, qs, metrics, contrasts))
    calls = []
    bootstrap = module.bootstrap_metric_arrays
    def shared(*args, **kwargs):
        calls.append(set(args[1]))
        assert len(args[0]) == 12 and int(kwargs['domain_mask'].sum()) == 8
        return bootstrap(*args, **kwargs)
    monkeypatch.setattr(module, 'bootstrap_metric_arrays', shared)
    rows, pairs = module.evaluate_statistics(T, 'arm_001', selections, qs, metrics, tmp_path / 'stats',
        contrasts=contrasts, family_sizes={'primary': 20, 'secondary': 2}, B=2000)
    assert calls == [set(qs)]
    assert len(rows) == 3 and len(pairs) == 2
    valid, unavailable = pairs
    assert valid['reference_method'] == 'FAIRBIAS_BM_AE' and valid['comparison_method'] == 'FAIRBIAS_JOINT'
    assert valid['difference_direction'] == 'reference_minus_comparison'
    assert valid['feasible_on_S'] is False  # Retain S-infeasible frozen boundary models.
    assert valid['family_endpoint_slots'] == 20 and valid['union_endpoint_slots'] == 22
    ba = valid['delta_balanced_accuracy']
    estimate, se = ba['taylor_95']['estimate'], ba['taylor_95']['se']
    assert se > 0
    for field, size in (('family_interval', 20), ('union_family_interval', 22)):
        assert ba[field]['lower'] == pytest.approx(estimate - t.ppf(1 - .05 / (2 * size), 3) * se)
        assert ba[field]['upper'] == pytest.approx(estimate + t.ppf(1 - .05 / (2 * size), 3) * se)
    assert ba['union_family_interval']['upper'] > ba['family_interval']['upper']
    assert unavailable['evaluation_status'] == 'NO_VALID_MODEL'
    assert unavailable['delta_eo_gap']['estimate'] is None
    assert unavailable['family_endpoint_slots'] == 2 and unavailable['union_endpoint_slots'] == 22
    assert selections == before[0] and metrics == before[2] and contrasts == before[3]
    for key in qs: np.testing.assert_array_equal(qs[key], before[1][key])
    with pytest.raises(ValueError, match='already exist'):
        module.evaluate_statistics(T, 'arm_001', selections, qs, metrics, tmp_path / 'stats', contrasts=contrasts,
            family_sizes={'primary': 20, 'secondary': 2}, B=2000)


def test_new_statistics_five_seed_eo_and_identical_paired_ba(tmp_path):
    from nhis_fairbias.benchmark import completion_statistics as module
    T, selections, qs, metrics, contrasts = _statistics_case(opposite_seeds=True)
    rows, pairs = module.evaluate_statistics(T, 'arm_001', selections, qs, metrics, tmp_path,
        contrasts=contrasts, family_sizes={'primary': 20, 'secondary': 0}, B=2000)
    assert rows[0]['eo_gap']['mean'] == .8
    ensemble = np.mean([qs[k][T.annual_design.domain_mask] for k in selections[0]['model_ids']], axis=0)
    assert compute_survey_fairness_metrics(T.y, ensemble, T.A, T.WTFA_A, [1, 2])['eo_gap'] == 0
    ba = pairs[0]['delta_balanced_accuracy']['taylor_95']
    assert ba['estimate'] == ba['se'] == ba['lower'] == ba['upper'] == 0
    assert pairs[0]['delta_eo_gap']['estimate'] == 0
    assert rows[0]['average_precision']['status'] == 'NOT_ESTIMABLE'


@pytest.mark.parametrize('fault', ['missing_prediction', 'missing_metrics', 'extra_prediction', 'duplicate_model', 'duplicate_selection',
    'mixed_branch', 'wrong_arm', 'duplicate_contrast', 'same_contrast_endpoints', 'unknown_selection', 'condition_mismatch',
    'undersized_family', 'missing_family', 'negative_family', 'bool_family', 'bad_B'])
def test_new_statistics_rejects_incomplete_or_mixed_inputs(tmp_path, fault):
    from nhis_fairbias.benchmark import completion_statistics as module
    T, selections, qs, metrics, contrasts = _statistics_case(missing=True)
    families, B = {'primary': 20, 'secondary': 2}, 2000
    if fault == 'missing_prediction': qs.pop(next(iter(qs)))
    elif fault == 'missing_metrics': metrics.pop(next(iter(metrics)))
    elif fault == 'extra_prediction': qs['unfrozen'] = np.zeros(12)
    elif fault == 'duplicate_model': selections[0]['model_ids'].append(selections[0]['model_ids'][0])
    elif fault == 'duplicate_selection': selections.append(copy.deepcopy(selections[0]))
    elif fault == 'mixed_branch': selections[0]['study_branch'] = 'registered_budget_main'
    elif fault == 'wrong_arm': selections[0]['arm_id'] = 'arm_002'
    elif fault == 'duplicate_contrast': contrasts.append(copy.deepcopy(contrasts[0]))
    elif fault == 'same_contrast_endpoints': contrasts.append({**contrasts[0], 'contrast_id': 'duplicate_endpoints'})
    elif fault == 'unknown_selection': contrasts[0]['comparison_selection_id'] = 'unknown'
    elif fault == 'condition_mismatch': selections[0]['tau'] = .20
    elif fault == 'undersized_family': families['secondary'] = 0
    elif fault == 'missing_family': families.pop('primary')
    elif fault == 'negative_family': families['secondary'] = -1
    elif fault == 'bool_family': families['secondary'] = True
    elif fault == 'bad_B': B = True
    with pytest.raises(ValueError):
        module.evaluate_statistics(T, 'arm_001', selections, qs, metrics, tmp_path / fault,
            contrasts=contrasts, family_sizes=families, B=B)
    assert not (tmp_path / fault / 'arm_001_individual_model_metrics.json').exists()


def test_new_statistics_retains_singleton_pairs(tmp_path):
    from nhis_fairbias.benchmark import completion_statistics as module
    T, selections, qs, metrics, contrasts = _statistics_case(singleton=True)
    rows, pairs = module.evaluate_statistics(T, 'arm_001', selections, qs, metrics, tmp_path,
        contrasts=contrasts, family_sizes={'primary': 20, 'secondary': 0}, B=2000)
    assert all(r['evaluation_status'] == 'DESIGN_NOT_ESTIMABLE' for r in rows)
    assert pairs[0]['evaluation_status'] == 'DESIGN_NOT_ESTIMABLE'
    assert pairs[0]['delta_balanced_accuracy']['taylor_95']['status'] == 'NOT_ESTIMABLE'
    assert pairs[0]['delta_eo_gap']['projection_family']['status'] == 'NOT_ESTIMABLE'


@pytest.fixture
def admission_case(tmp_path, monkeypatch, completion, completion_jobs):
    """Actual metadata builder; only generated opaque policy/prepared bytes."""
    from pathlib import Path
    backup, main = tmp_path / 'backup', tmp_path / 'parent_archive'
    runtime = backup / 'snapshot/fairbias_completion_v2_20260917'
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2))
        return path
    source_name = 'src/nhis_fairbias/benchmark/data_contracts.py'
    source_copy = runtime / source_name
    source_copy.parent.mkdir(parents=True)
    source_copy.write_bytes((completion.ROOT / source_name).read_bytes())
    prepared = {}
    for arm in completion.ARMS:
        path = runtime / 'prepared' / (arm + '.joblib')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('GENERATED OPAQUE PREPARED ' + arm).encode())
        prepared[arm] = {'path': '/generated/unused/' + arm, 'sha256': completion.sha(path), 'data_identity': 'generated_' + arm}
    jobs = copy.deepcopy(completion_jobs)
    for job in jobs:
        job['original_job_sha256'] = '8' * 64
        job['prepared'] = prepared[job['config']['arm_id']]
    training = {'version': 'generated_completion_manifest', 'jobs': jobs,
        'sources': {source_name: completion.sha(source_copy)}, 'T_already_known': True,
        'S_T_evaluation_authorized': False, 'formal_benchmark_admission': False}
    training['manifest_sha256'] = completion.digest(training)
    monkeypatch.setattr(completion, 'TRAINING_SEAL', training['manifest_sha256'])
    write(runtime / 'control/manifest.json', training)
    audit_jobs = []
    for job in jobs:
        directory = backup / 'snapshot/generated_completion_jobs' / job['job_id']
        directory.mkdir(parents=True)
        model = directory / 'policy.joblib'
        model.write_bytes(('GENERATED OPAQUE POLICY ' + job['job_id']).encode())
        result = {'config': job['config'], 'seed': job['seed'], 'job_id': job['job_id'],
            'status': 'FC_COMPLETE_FEASIBLE', 'completion': {'status': 'COMPLETE_FEASIBLE'},
            'manifest_sha256': training['manifest_sha256'], 'reload_exact': True,
            'S_T_evaluated': False, 'formal_benchmark_admission': False,
            'model_sha256': completion.sha(model), 'prepared_sha256': job['prepared']['sha256'],
            'data_identity': job['prepared']['data_identity']}
        result_path = write(directory / 'result.json', result)
        audit_jobs.append({'job_id': job['job_id'], 'output': '/root/autodl-tmp/generated_completion_jobs/' + job['job_id'],
            'result_sha256': completion.sha(result_path), 'model_sha256': completion.sha(model),
            'C_q_sha256': '6' * 64, 'C_p_sha256': '7' * 64})
    audit = write(backup / 'control/audit.json', {'jobs': audit_jobs})
    write(backup / 'control/supervisor_stop_ready.json', {'status': 'SUPERVISOR_ACCEPTED_COMPUTE_STOP_READY', 'audit_sha256': completion.sha(audit)})
    placeholder = tmp_path / 'generated_t_placeholder.txt'
    placeholder.write_text('Generated fixture bytes only; no NHIS records.\n')
    study_registry = write(tmp_path / 'configs/nhis/study.json',
        {'years': {'2024': {'local_csv_file': placeholder.name, 'expected_raw_rows': 12}}})
    features = write(tmp_path / 'configs/nhis/features.json', {'generated_fixture': True})
    provenance = {'raw_sources': {'2024': {'path': placeholder.name, 'sha256': completion.sha(placeholder), 'rows': 12}},
        'identity': 'official HHX + survey year', 'study_registry_sha256': completion.sha(study_registry),
        'feature_registry_sha256': completion.sha(features)}
    comparators, parent_configs = [], []
    seed_zero_id = 'parent_seed_zero'
    seed_zero_candidate = None
    for arm in completion.ARMS:
        for backbone in completion.BACKBONES:
            for method, tau in [(m, .1) for m in completion.COMPARATORS] + [('FAIRBIAS_BM', None)]:
                cid = f'original_{arm}_{backbone}_{method}_{tau}'
                unsupported = arm == 'arm_002' and method in {'LFR_RECONSTRUCTED', 'FRAPPE_EO'}
                only_zero = arm == 'arm_001' and backbone == 'LR' and method == 'UNMITIGATED'
                parent_configs.append({'candidate_id': cid, 'arm_id': arm, 'backbone': backbone, 'method': method,
                    'training_weighted': False, 'status': 'NOT_SUPPORTED' if unsupported else 'REGISTERED',
                    'reason': 'generated unsupported multi-group condition' if unsupported else None,
                    'seeds': [] if unsupported else [0] if only_zero else list(completion.SEEDS), 'params': {}})
                if unsupported:
                    continue
                row = {'selection_id': f'parent_{arm}_{backbone}_{method}_{tau}',
                    'arm_id': arm, 'backbone': backbone, 'method': method, 'tau': tau, 'training_weighted': False,
                    'status': 'NO_VALID_FIXED_ABLATION' if tau is None else 'NOT_ESTIMABLE',
                    'study_branch': 'registered_budget_main', 'analysis_version': 'registered_v1',
                    'model_ids': [], 'selected_candidate_id': None, 'boundary_candidate_id': None}
                if only_zero:
                    row.update(status='FEASIBLE', model_ids=[seed_zero_id], selected_candidate_id=cid)
                    seed_zero_candidate = cid
                comparators.append(row)
    parent_registration = write(main / 'original_registration.json', {'candidates': parent_configs})
    old_model = {'model_id': seed_zero_id, 'seed': 0, 'candidate_id': seed_zero_candidate, 'arm_id': 'arm_001',
        'backbone': 'LR', 'method': 'UNMITIGATED', 'training_weighted': False,
        'output_type': 'event_probability_p', 'artifacts': {'model.joblib': {'sha256': '9' * 64}}}
    old = {'selected_models': [old_model], 'selections': comparators, 't_source_provenance': provenance,
        'registration': completion.bound(parent_registration),
        'analysis_files': {source_name: completion.sha(completion.ROOT / source_name)}}
    old_path = write(main / 'main_analysis_v3/study_freeze.json', old)
    monkeypatch.setattr(completion, 'MAIN_STUDY_SHA', completion.sha(old_path))
    for arm in completion.ARMS:
        directory = main / 'main_evaluation_v2' / arm
        T = _full_annual()
        q = T.y.astype(float)
        metric_values = compute_survey_fairness_metrics(T.y, q, T.A, T.WTFA_A, [1, 2])
        individual = {seed_zero_id: {'model_id': seed_zero_id, 'weighted': metric_values,
            'unweighted': metric_values, 'risk': {}, 'unweighted_risk': {}}} if arm == 'arm_001' else {}
        metrics = write(directory / (arm + '_individual_model_metrics.json'), individual)
        artifacts = {metrics.name: completion.sha(metrics)}
        if arm == 'arm_001':
            prediction = directory / (arm + '_' + seed_zero_id + '_predictions_T.npz')
            np.savez(prediction, q=q, p=.2 + .6 * q)
            artifacts[prediction.name] = completion.sha(prediction)
        write(directory / 'evaluation_manifest.json', {'study_freeze_sha256': completion.sha(old_path),
            'status': 'COMPLETE', 'arm_ids': [arm], 'source_provenance': provenance, 'artifacts': artifacts})
    path = tmp_path / 'admission.json'
    value = completion.build_admission(backup, main, path, parent_registration=parent_registration)
    return {'path': path, 'value': value, 'backup': backup, 'main': main, 'runtime': runtime,
        'write': write, 'root': tmp_path, 'source_copy': source_copy, 'parent_registration': parent_registration,
        'parent_study': old_path, 'seed_zero_candidate': seed_zero_candidate}


def test_actual_admission_builder_is_metadata_only(admission_case, completion, monkeypatch):
    import joblib
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('metadata must not deserialize a policy or F/C/S container'))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('metadata must not decode prior T predictions'))
    monkeypatch.setattr(completion, 'load_local_nhis_cohort', lambda *a, **k: pytest.fail('metadata must not load T'))
    value = completion.validate_admission(admission_case['path'], completion.sha(admission_case['path']))
    assert value['known_T'] is True and value['study_branch'] == completion.BRANCH
    assert value['evaluation_authorized'] is False
    assert len(value['jobs']) == 80 and len(value['comparators']) == 80 and len(value['contrasts']) == 168
    assert value['family_sizes'] == {'primary': 20, 'secondary': 316, 'union': 336}
    unsupported = [r for r in value['comparators'] if r['status'] == 'NOT_SUPPORTED']
    assert len(unsupported) == 4
    assert all(r['arm_id'] == 'arm_002' and r['model_ids'] == [] and r['prediction_refs'] == [] for r in unsupported)
    singleton = next(r for r in value['comparators'] if r['model_ids'])
    assert [r['seed'] for r in singleton['prediction_refs']] == [0]
    with pytest.raises((ValueError, FileExistsError)):
        completion.build_admission(admission_case['backup'], admission_case['main'], admission_case['path'],
            parent_registration=admission_case['parent_registration'])


@pytest.mark.parametrize('field', ['policy', 'result', 'prepared_file', 'training_manifest', 'audit', 'parent_study', 'parent_metrics', 'admission_sha', 'runtime_source'])
def test_bound_file_corruption_rejected_before_any_unpickle(admission_case, completion, monkeypatch, field):
    import joblib
    from pathlib import Path
    c = admission_case
    expected = completion.sha(c['path'])
    if field in ('policy', 'result', 'prepared_file'):
        path = Path(c['value']['jobs'][0][field]['path'])
    elif field == 'runtime_source':
        path = c['source_copy']
    elif field == 'parent_metrics':
        path = Path(c['value']['parent_shards']['arm_001']['individual']['path'])
    else:
        path = c['path'] if field == 'admission_sha' else Path(c['value'][field]['path'])
    path.write_bytes(path.read_bytes() + b' changed')
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('invalid admission must not deserialize'))
    monkeypatch.setattr(completion, 'load_local_nhis_cohort', lambda *a, **k: pytest.fail('invalid admission must not load T'))
    with pytest.raises(ValueError):
        completion.validate_admission(c['path'], expected)


def test_statistics_explicit_union_and_fixed_bm_policy_exception(tmp_path):
    from nhis_fairbias.benchmark import completion_statistics as module
    T, selections, qs, metrics, contrasts = _statistics_case()
    selections[1].update(method='FAIRBIAS_BM', tau=None, status='FIXED_ABLATION')
    contrasts[0]['kind'] = 'parent_fixed_BM'
    _, pairs = module.evaluate_statistics(T, 'arm_001', selections, qs, metrics, tmp_path / 'fixed',
        contrasts=contrasts, family_sizes={'primary': 20, 'secondary': 316, 'union': 336}, B=2000)
    assert pairs[0]['comparison_method'] == 'FAIRBIAS_BM' and pairs[0]['union_endpoint_slots'] == 336
    for fault in ('union', 'undeclared_kind', 'wrong_method', 'wrong_status', 'wrong_reference_tau'):
        rows, pairspec = copy.deepcopy(selections), copy.deepcopy(contrasts)
        families = {'primary': 20, 'secondary': 316, 'union': 336}
        if fault == 'union': families['union'] = 20
        elif fault == 'undeclared_kind': pairspec[0]['kind'] = 'unspecified'
        elif fault == 'wrong_method': rows[1]['method'] = 'UNMITIGATED'
        elif fault == 'wrong_status': rows[1]['status'] = 'FEASIBLE'
        elif fault == 'wrong_reference_tau': rows[0]['tau'] = .20
        with pytest.raises(ValueError):
            module.evaluate_statistics(T, 'arm_001', rows, qs, metrics, tmp_path / fault,
                contrasts=pairspec, family_sizes=families, B=2000)


@pytest.mark.parametrize('fault', ['missing_comparator', 'duplicate_comparator', 'comparator_method', 'comparator_tau',
    'contrast_endpoint', 'contrast_family_swap', 't_provenance', 'reload_evidence'])
def test_admission_semantic_mapping_is_reconstructed(admission_case, completion, fault):
    c = admission_case
    value = copy.deepcopy(c['value'])
    if fault == 'missing_comparator': value['comparators'].pop()
    elif fault == 'duplicate_comparator': value['comparators'][0] = copy.deepcopy(value['comparators'][1])
    elif fault == 'comparator_method': value['comparators'][0]['method'] = 'UNREGISTERED_METHOD'
    elif fault == 'comparator_tau': value['comparators'][0]['tau'] = .2
    elif fault == 'contrast_endpoint': value['contrasts'][0]['comparison_selection_id'] = 'parent:unregistered'
    elif fault == 'contrast_family_swap':
        first = next(c for c in value['contrasts'] if c['family'] == 'primary')
        second = next(c for c in value['contrasts'] if c['family'] == 'secondary')
        first['family'], second['family'] = 'secondary', 'primary'
    elif fault == 't_provenance': value['t_source_provenance']['raw_sources']['2024']['sha256'] = 'f' * 64
    elif fault == 'reload_evidence': value['jobs'][0]['C_q_sha256'] = 'f' * 64
    path = c['write'](c['root'] / (fault + '.json'), value)
    # Rehash deliberately: this tests semantic provenance, not merely a stale
    # caller checksum. A new SHA cannot convert a wrong mapping into evidence.
    with pytest.raises(ValueError):
        completion.validate_admission(path, completion.sha(path))


@pytest.mark.parametrize('fault', ['wrong_release_hash', 'wrong_decision', 'wrong_study', 'wrong_branch', 'known_T_false', 'existing_output'])
def test_evaluation_entry_rejects_before_any_unpickle_or_t_io(tmp_path, monkeypatch, completion, known_t_release, fault):
    import joblib
    study, release = copy.deepcopy(known_t_release)
    # Authorization-order unit seam: metadata rebuilding is independently
    # exercised above; no model/T loader is replaced by a permissive stub.
    monkeypatch.setattr(completion, 'validate_study', lambda *args: (study, {'jobs': []}))
    monkeypatch.setattr(joblib, 'load', lambda *a, **k: pytest.fail('unreleased model/prepared unpickle'))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('unreleased prior T prediction read'))
    monkeypatch.setattr(completion, 'load_local_nhis_cohort', lambda *a, **k: pytest.fail('unreleased T cohort read'))
    monkeypatch.setattr(completion.ev, '_verify_t_inputs', lambda *a, **k: pytest.fail('unreleased T input hashing'))
    if fault == 'wrong_decision': release['decision'] = 'NOT_RELEASED'
    elif fault == 'wrong_study': release['study_sha256'] = 'f' * 64
    elif fault == 'wrong_branch': release['study_branch'] = 'registered_budget_main'
    elif fault == 'known_T_false': release['known_T'] = False
    path = tmp_path / 'release.json'
    path.write_text(json.dumps(release))
    expected = 'f' * 64 if fault == 'wrong_release_hash' else completion.sha(path)
    output = tmp_path / 'evaluation'
    if fault == 'existing_output':
        output.mkdir()
        (output / 'existing.json').write_text('preserve this artifact')
    with pytest.raises(ValueError):
        completion.evaluate_completion_arm(tmp_path / 'unused_study.json', '1' * 64, path, expected,
            output, arm_id='arm_001', data_root=tmp_path / 'no_data')
    if fault == 'existing_output':
        assert (output / 'existing.json').read_text() == 'preserve this artifact'
        assert list(output.iterdir()) == [output / 'existing.json']
    else:
        assert not output.exists()


def _rewrite_parent_fixture(case, completion, monkeypatch, old):
    """Reseal only generated parent metadata after a deliberate fixture change."""
    old['registration'] = completion.bound(case['parent_registration'])
    case['write'](case['parent_study'], old)
    parent_sha = completion.sha(case['parent_study'])
    monkeypatch.setattr(completion, 'MAIN_STUDY_SHA', parent_sha)
    for arm in completion.ARMS:
        path = case['main'] / 'main_evaluation_v2' / arm / 'evaluation_manifest.json'
        shard = completion.read(path)
        shard['study_freeze_sha256'] = parent_sha
        shard['source_provenance'] = old['t_source_provenance']
        case['write'](path, shard)


@pytest.mark.parametrize('fault', ['missing_registered_seed', 'wrong_registered_seed', 'missing_unsupported_registration',
    'missing_unsupported_slot', 'registration_hash'])
def test_parent_registered_seed_and_unsupported_coverage(admission_case, completion, monkeypatch, fault):
    c = admission_case
    if fault == 'missing_unsupported_slot':
        value = copy.deepcopy(c['value'])
        value['comparators'] = [r for r in value['comparators'] if not
            (r['arm_id'] == 'arm_002' and r['backbone'] == 'LR' and r['method'] == 'LFR_RECONSTRUCTED')]
        path = c['write'](c['root'] / 'missing_slot.json', value)
        with pytest.raises(ValueError, match='COMPARATOR_COVERAGE_CHANGED'):
            completion.validate_admission(path, completion.sha(path))
        return
    old = completion.read(c['parent_study'])
    registry = completion.read(c['parent_registration'])
    if fault == 'missing_registered_seed':
        next(r for r in registry['candidates'] if r['candidate_id'] == c['seed_zero_candidate'])['seeds'] = [0, 7]
    elif fault == 'wrong_registered_seed':
        old['selected_models'][0]['seed'] = 7
    elif fault == 'missing_unsupported_registration':
        registry['candidates'] = [r for r in registry['candidates'] if not
            (r['arm_id'] == 'arm_002' and r['backbone'] == 'LR' and r['method'] == 'LFR_RECONSTRUCTED')]
    else:
        registry['changed_without_rebinding'] = True
    c['write'](c['parent_registration'], registry)
    if fault != 'registration_hash':
        _rewrite_parent_fixture(c, completion, monkeypatch, old)
    with pytest.raises(ValueError):
        completion.build_admission(c['backup'], c['main'], c['root'] / 'bad_admission.json',
            parent_registration=c['parent_registration'])


@pytest.fixture
def synthetic_selection_case(admission_case, completion, monkeypatch, request):
    """Execute the real S/freeze path with generated in-memory prediction seams."""
    from pathlib import Path
    import hashlib
    from nhis_fairbias.benchmark.predictions import PredictionBundle
    from types import SimpleNamespace
    import pandas as pd
    c = admission_case
    q = _annual().y.astype(float)
    p = .2 + .6 * q
    audit_path = c['backup'] / 'control/audit.json'
    audit = completion.read(audit_path)
    for row in audit['jobs']:
        row['C_q_sha256'] = hashlib.sha256(q.tobytes()).hexdigest()
        row['C_p_sha256'] = hashlib.sha256(p.tobytes()).hexdigest()
    c['write'](audit_path, audit)
    c['write'](c['backup'] / 'control/supervisor_stop_ready.json',
        {'status': 'SUPERVISOR_ACCEPTED_COMPUTE_STOP_READY', 'audit_sha256': completion.sha(audit_path)})
    admission_path = c['root'] / 'synthetic_s_admission.json'
    c['value'] = completion.build_admission(c['backup'], c['main'], admission_path,
        parent_registration=c['parent_registration'])
    c['path'] = admission_path
    loaded, predicted = [], []
    def prepared_loader(path):
        path = Path(path)
        assert path.parent == c['runtime'] / 'prepared'
        arm = path.stem
        C, S = _annual(), _annual()
        C.year, C.role, C.arm_id = 2022, 'calibration_C', arm
        S.year, S.role, S.arm_id = 2023, 'selection_S', arm
        if not getattr(request, 'param', {}).get('missing_support', False):
            groups = completion.ARM_SPECS[arm]['expected_categories']
            n = 4 * len(groups)
            S = SimpleNamespace(year=2023, role='selection_S', arm_id=arm,
                X_semantic=pd.DataFrame({'x': np.arange(n)}), A=np.repeat(groups, 4),
                y=np.tile([0, 1], n // 2), WTFA_A=np.ones(n))
        loaded.append(arm)
        return {'data_identity': 'generated_' + arm,
            'partitions': {'fitting_F': object(), 'calibration_C': C, 'selection_S': S}}
    class GeneratedPolicy:
        def predict(self, X, A):
            assert len(X) == len(A)
            predicted.append(len(X))
            return PredictionBundle(np.resize(p, len(X)), np.resize(q, len(X)))
        def fit(self, *a, **k):
            pytest.fail('S/T evaluation must not fit')
    monkeypatch.setattr(completion.joblib, 'load', prepared_loader)
    monkeypatch.setattr(completion, '_load_policy', lambda job: GeneratedPolicy())
    monkeypatch.setattr(completion, 'load_local_nhis_cohort', lambda *a, **k: pytest.fail('selection/study must not load T'))
    selection_path = c['root'] / 'selection.json'
    if getattr(request, 'param', {}).get('missing_support', False):
        with pytest.raises(ValueError, match='Out of range float values are not JSON compliant'):
            completion.freeze_completion_selection(admission_path, completion.sha(admission_path), selection_path)
        c['rejected_selection_path'] = selection_path
        return c
    selected = completion.freeze_completion_selection(admission_path, completion.sha(admission_path), selection_path)
    study_path = c['root'] / 'study.json'
    study = completion.freeze_completion_study(selection_path, completion.sha(selection_path), study_path)
    c.update(selection_path=selection_path, selection=selected, study_path=study_path, study=study,
        prepared_loads=loaded, policy_predictions=predicted, q=q, p=p)
    return c


def test_actual_s_prediction_to_study_freeze_without_t_access(synthetic_selection_case, completion):
    c = synthetic_selection_case
    assert c['prepared_loads'] == list(completion.ARMS)
    assert len(c['policy_predictions']) == 160  # one C verification and one S prediction per frozen model
    assert len(c['selection']['individual_S']) == len(c['selection']['cross_platform_C_reload']) == 80
    assert len(c['selection']['selections']) == 16 and len(c['selection']['S_sensitivity']) == 48
    assert all(r['C_q_exact'] and r['C_p_bytes_equal'] for r in c['selection']['cross_platform_C_reload'])
    assert c['study']['known_T'] is True and c['study']['evaluation_authorized'] is False
    assert c['study']['study_branch'] == completion.BRANCH
    assert len(c['study']['selections']) == 96 and len(c['study']['contrasts']) == 168
    assert c['study']['bootstrap_replicates'] == 2000 and c['study']['bootstrap_seed'] == 20260914
    assert completion.validate_study(c['study_path'], completion.sha(c['study_path']))[0] == c['study']
    before = c['selection_path'].read_bytes()
    with pytest.raises(ValueError, match='SELECTION_OUTPUT_EXISTS'):
        completion.freeze_completion_selection(c['path'], completion.sha(c['path']), c['selection_path'])
    assert c['selection_path'].read_bytes() == before


@pytest.mark.parametrize('synthetic_selection_case', [{'missing_support': True}], indirect=True)
def test_missing_s_group_support_fails_closed_without_valid_freeze(synthetic_selection_case, completion):
    # Current frozen implementation deliberately does not serialize NaN. A
    # partially written file must never be accepted as a completed selection.
    c = synthetic_selection_case
    with pytest.raises(ValueError):
        completion.freeze_completion_study(c['rejected_selection_path'], completion.sha(c['rejected_selection_path']),
            c['root'] / 'must_not_exist_study.json')
    assert not (c['root'] / 'must_not_exist_study.json').exists()


@pytest.mark.parametrize('field,value', [('known_T', False), ('study_branch', 'registered_budget_main'),
    ('bootstrap_replicates', 100), ('bootstrap_seed', 9), ('arm_ids', ['arm_001'])])
def test_rehashed_study_cannot_change_frozen_policy(synthetic_selection_case, completion, field, value):
    c = synthetic_selection_case
    changed = copy.deepcopy(c['study'])
    changed[field] = value
    path = c['write'](c['root'] / ('changed_' + field + '.json'), changed)
    with pytest.raises(ValueError):
        completion.validate_study(path, completion.sha(path))


def test_generated_selection_freeze_release_and_arm_evaluation(synthetic_selection_case, completion, monkeypatch):
    import dataclasses
    import pandas as pd
    from nhis_fairbias.benchmark import completion_statistics
    c = synthetic_selection_case
    T = _full_annual()
    # The real data contract constructs source:year:id strings. Older survey
    # unit fixtures use positional NumPy integers, which are not JSON keys.
    record_keys = np.array([f'generated:2024:{i}' for i in range(12)], dtype=object)
    T.record_keys = record_keys[:8].copy()
    T.annual_design = dataclasses.replace(T.annual_design, record_keys=record_keys)
    cohort = pd.DataFrame({'year': [2024] * 12, 'WTFA_A': np.ones(12),
        'MEDDL12M_A': np.resize(T.y, 12), 'SEX_A': np.resize(T.A, 12),
        'HISPALLP_A': np.resize(T.A, 12), 'DISAB3_A': np.resize(T.A, 12)})
    cohort.attrs['source_provenance'] = copy.deepcopy(c['study']['t_source_provenance'])
    called = []
    def load_generated(root, *, years):
        assert root == c['root'] and years == (2024,)
        called.append(years)
        return cohort
    monkeypatch.setattr(completion, 'load_local_nhis_cohort', load_generated)
    monkeypatch.setattr(completion, 'load_arm_partitions', lambda loaded, arm: {'evaluation_T': T})
    bootstrap = completion_statistics.bootstrap_metric_arrays
    designs = []
    def record_bootstrap(*args, **kwargs):
        designs.append((len(args[0]), set(args[1]), kwargs.get('B')))
        return bootstrap(*args, **kwargs)
    monkeypatch.setattr(completion_statistics, 'bootstrap_metric_arrays', record_bootstrap)
    release = {'schema_version': 'completion_t_release_v1', 'decision': 'SUPERVISOR_RELEASED_COMPLETION_T',
        'study_branch': completion.BRANCH, 'known_T': True, 'study_sha256': completion.sha(c['study_path']),
        'selection_sha256': completion.sha(c['selection_path']), 'arm_ids': ['arm_001']}
    release_path = c['write'](c['root'] / 'release.json', release)
    output = c['root'] / 'evaluation'
    summary = completion.evaluate_completion_arm(c['study_path'], completion.sha(c['study_path']),
        release_path, completion.sha(release_path), output, arm_id='arm_001', data_root=c['root'])
    manifest = completion.read(output / 'evaluation_manifest.json')
    expected = {m for r in c['study']['selections'] if r['arm_id'] == 'arm_001' for m in r['model_ids']}
    assert called == [(2024,)] and len(expected) == 21
    assert designs == [(12, expected, 2000)]  # all model pairs use one full-year bootstrap design
    assert manifest['status'] == 'COMPLETE' and manifest['known_T'] is summary['known_T'] is True
    assert summary['study_branch'] == manifest['study_branch'] == completion.BRANCH
    assert manifest['study_sha256'] == completion.sha(c['study_path'])
    assert manifest['models'] == sorted(expected) and manifest['reused_parent_models'] == ['parent_seed_zero']
    assert len(summary['selections']) == 24 and len(summary['paired_contrasts']) == 42
    assert all(r['sample_n'] == 8 and r['design_n'] == 12 and r['df'] == 3 for r in summary['selections'])
    assert all(p['union_endpoint_slots'] == 336 for p in summary['paired_contrasts'])
    assert any(p['evaluation_status'] == 'NO_VALID_MODEL' for p in summary['paired_contrasts'])
    assert all(completion.sha(output / name) == value for name, value in manifest['artifacts'].items())
    with pytest.raises(ValueError, match='EVALUATION_OUTPUT_EXISTS'):
        completion.evaluate_completion_arm(c['study_path'], completion.sha(c['study_path']), release_path,
            completion.sha(release_path), output, arm_id='arm_001', data_root=c['root'])
    assert called == [(2024,)]
