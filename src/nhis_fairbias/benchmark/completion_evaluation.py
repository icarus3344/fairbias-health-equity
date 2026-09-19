"""Separate, hash-bound evaluation of all 80 adaptive-compute FairBias anchors.

Historical studies are immutable inputs. Admission is metric blind. S and T
have separate explicit entry points; no training or threshold fitting occurs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import joblib
import numpy as np

from . import catalog_evaluation as ev
from .data_contracts import ARM_SPECS, load_arm_partitions, load_local_nhis_cohort
from .experiment_selection import aggregate_configuration, select_aggregate
from .metrics import compute_survey_fairness_metrics
from .risk_metrics import compute_risk_metrics
from .predictions import FrozenDecisionPolicy, PredictionBundle

ROOT = Path(__file__).resolve().parents[3]
BRANCH = 'fairbias_adaptive_completion_v1'
METHODS = ('FAIRBIAS_BM_AE', 'FAIRBIAS_JOINT')
SEEDS = (0, 7, 19, 37, 73)
ARMS = tuple(f'arm_{i:03d}' for i in range(1, 5))
BACKBONES = ('LR', 'GBDT')
TAUS = (.05, .10, .20)
COMPARATORS = ('UNMITIGATED', 'FAIRBIAS_BM', 'REWEIGHING', 'LFR_RECONSTRUCTED',
               'EG_DP', 'EG_EO', 'TO_EO', 'FRAPPE_EO', 'OXONFAIR_EO')
PRIMARY = ('REWEIGHING', 'LFR_RECONSTRUCTED', 'EG_DP', 'EG_EO', 'TO_EO')
MAIN_STUDY_SHA = 'b9531ebba6c95e66489255eb0a232ce2a67b585e48799673d8d0c6dad1a10b1e'
TRAINING_SEAL = '6607256f5efa027e0c954e7e0a0021d886074bf6dbfde0a52f1e5c5ba69ba0f0'
FAMILY_SIZES = {'primary': 20, 'secondary': 316, 'union': 336}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), 'JSON_OBJECT_REQUIRED')
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def bound(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def verify(item):
    require(isinstance(item, dict) and set(item) == {'path', 'sha256'}, 'INVALID_BOUND_FILE')
    p = Path(item['path'])
    require(p.is_absolute() and p.is_file() and sha(p) == item['sha256'], 'BOUND_FILE_HASH_MISMATCH')
    return p


def fresh(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def analysis_files():
    paths = [p for pkg in ('fairbias', 'nhis_fairbias') for p in (ROOT/'src'/pkg).rglob('*.py')]
    entry = ROOT/'scripts/evaluate_nhis_completion.py'
    if entry.exists():
        paths.append(entry)
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in {'fairbias', 'nhis_fairbias'} and getattr(module, '__file__', None):
            require(Path(module.__file__).resolve().is_relative_to(ROOT/'src'), 'IMPORT_ORIGIN_MISMATCH')
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def validate_job_coverage(jobs):
    require(isinstance(jobs, list) and len(jobs) == 80, 'EXACT_80_JOBS_REQUIRED')
    expected = {(a, b, m, s) for a in ARMS for b in BACKBONES for m in METHODS for s in SEEDS}
    keys, ids, cells = [], [], defaultdict(set)
    for job in jobs:
        c, seed = job['config'], job['seed']
        require(type(seed) is int and seed in SEEDS, 'INVALID_SEED')
        require(isinstance(c['seeds'], list) and all(type(s) is int for s in c['seeds'])
                and sorted(c['seeds']) == list(SEEDS), 'CONFIG_SEED_SET_CHANGED')
        key = (c['arm_id'], c['backbone'], c['method'], seed)
        require(job['job_id'] == f"{c['candidate_id']}_s{seed}", 'JOB_ID_MISMATCH')
        keys.append(key)
        ids.append(job['job_id'])
        cells[key[:3]].add((c['candidate_id'], digest(c)))
    require(set(keys) == expected and len(set(keys)) == 80 and len(set(ids)) == 80,
            'MISSING_OR_DUPLICATE_JOB')
    require(all(len(v) == 1 for v in cells.values()), 'MIXED_CANDIDATES_IN_CELL')
    return True


def validate_release(study, study_sha, release, arms):
    require(study.get('study_branch') == BRANCH and study.get('known_T') is True,
            'SEPARATE_KNOWN_T_BRANCH_REQUIRED')
    require(release.get('schema_version') == 'completion_t_release_v1'
            and release.get('decision') == 'SUPERVISOR_RELEASED_COMPLETION_T'
            and release.get('study_branch') == BRANCH and release.get('known_T') is True
            and release.get('study_sha256') == study_sha
            and release.get('selection_sha256') == study['selection']['sha256'], 'COMPLETION_T_RELEASE_REQUIRED')
    ids = release.get('arm_ids')
    require(isinstance(ids, list) and len(ids) == len(set(ids)) and set(arms) <= set(ids) <= set(ARMS),
            'ARM_NOT_RELEASED')
    return True


def _new_id(arm, backbone, method):
    return f'completion:{arm}:{backbone}:{method}'


def contrast_registry(comps):
    contrasts = []
    def add(ref, comparison, family, kind):
        contrasts.append({'contrast_id': 'contrast_'+digest([ref, comparison]), 'reference_selection_id': ref,
                          'comparison_selection_id': comparison, 'family': family, 'kind': kind})
    for arm in ARMS:
        for backbone in BACKBONES:
            for method in METHODS:
                for comp in comps:
                    if (comp['arm_id'], comp['backbone']) != (arm, backbone):
                        continue
                    primary = (method == 'FAIRBIAS_JOINT' and arm in ('arm_001', 'arm_003') and backbone == 'LR'
                               and comp['method'] in PRIMARY and comp['origin'] == 'parent_tau10')
                    add(_new_id(arm, backbone, method), comp['selection_id'],
                        'primary' if primary else 'secondary', comp['origin'])
            add(_new_id(arm, backbone, 'FAIRBIAS_JOINT'), _new_id(arm, backbone, 'FAIRBIAS_BM_AE'),
                'secondary', 'joint_vs_sequential')
    return contrasts


def _parent_row(old, registry, arm, backbone, method, tau):
    found = [r for r in old['selections'] if r['arm_id'] == arm and r['backbone'] == backbone
             and r['method'] == method and r['tau'] == tau and r['training_weighted'] is False]
    require(len(found) <= 1, 'COMPARATOR_SLOT_DUPLICATED')
    if found:
        return found[0]
    unsupported = [r for r in registry.values() if r['arm_id'] == arm and r['backbone'] == backbone
                   and r['method'] == method and r.get('training_weighted', False) is False
                   and r.get('status') == 'NOT_SUPPORTED']
    require(tau == .1 and len(unsupported) == 1 and unsupported[0]['seeds'] == [], 'COMPARATOR_SLOT_UNEXPLAINED')
    row = unsupported[0]
    return {'selection_id': f'unsupported:{arm}:{backbone}:{method}', 'arm_id': arm,
            'backbone': backbone, 'method': method, 'tau': tau, 'training_weighted': False,
            'status': 'NOT_SUPPORTED', 'model_ids': [], 'selected_candidate_id': None,
            'registered_fixed_candidate_id': row['candidate_id'],
            'reason': row.get('reason', row.get('notes')), 'study_branch': 'registered_budget_main'}


def build_admission(backup_root, main_archive_root, output, *, parent_registration=None):
    """Bind complete artifacts and a comparator registry, without model/S/T reads."""
    backup, main = Path(backup_root).resolve(), Path(main_archive_root).resolve()
    audit_path = backup/'control/audit.json'
    proof = read(backup/'control/supervisor_stop_ready.json')
    require(proof['status'] == 'SUPERVISOR_ACCEPTED_COMPUTE_STOP_READY'
            and sha(audit_path) == proof['audit_sha256'], 'COMPLETION_AUDIT_NOT_ACCEPTED')
    audit = read(audit_path)
    runtime = backup/'snapshot/fairbias_completion_v2_20260917'
    training = read(runtime/'control/manifest.json')
    unsigned = {k: v for k, v in training.items() if k != 'manifest_sha256'}
    require(digest(unsigned) == training['manifest_sha256'] == TRAINING_SEAL, 'TRAINING_MANIFEST_CHANGED')
    validate_job_coverage(training['jobs'])
    for name, value in training['sources'].items():
        require(sha(runtime/name) == value and sha(ROOT/name) == value, 'TRAINING_SOURCE_COMPATIBILITY_FAILED')
    by_job = {r['job_id']: r for r in audit['jobs']}
    require(len(by_job) == len(audit['jobs']) == 80, 'AUDIT_JOB_DUPLICATE')
    records = []
    for job in training['jobs']:
        record = by_job[job['job_id']]
        rel = Path(record['output']).relative_to('/root/autodl-tmp')
        directory = backup/'snapshot'/rel
        result = read(directory/'result.json')
        require(result['config'] == job['config'] and result['seed'] == job['seed']
                and result['job_id'] == job['job_id'] and result['status'] == 'FC_COMPLETE_FEASIBLE'
                and result['completion']['status'] == 'COMPLETE_FEASIBLE'
                and result['manifest_sha256'] == TRAINING_SEAL
                and result['reload_exact'] is True and result['S_T_evaluated'] is False
                and result['formal_benchmark_admission'] is False, 'INVALID_COMPLETED_RESULT')
        require(sha(directory/'result.json') == record['result_sha256']
                and sha(directory/'policy.joblib') == result['model_sha256'] == record['model_sha256'],
                'COMPLETED_ARTIFACT_MISMATCH')
        prepared = bound(runtime/'prepared'/(job['config']['arm_id']+'.joblib'))
        require(prepared['sha256'] == job['prepared']['sha256'] == result['prepared_sha256']
                and result['data_identity'] == job['prepared']['data_identity'], 'PREPARED_IDENTITY_MISMATCH')
        records.append({**job, 'model_id': 'completion_model_'+job['job_id'],
                        'policy': bound(directory/'policy.joblib'), 'result': bound(directory/'result.json'),
                        'prepared_file': prepared, 'C_q_sha256': record['C_q_sha256'],
                        'C_p_sha256': record['C_p_sha256']})
    study_path = main/'main_analysis_v3/study_freeze.json'
    require(sha(study_path) == MAIN_STUDY_SHA, 'UNACCEPTED_PARENT_STUDY')
    old = read(study_path)
    registration_path = Path(parent_registration or ROOT/'scratch/fairbias_analysis_20260917/admission_draft/inputs/registration.json')
    require(sha(registration_path) == old['registration']['sha256'], 'PARENT_REGISTRATION_CHANGED')
    registry = {c['candidate_id']: c for c in read(registration_path)['candidates']}
    old_refs = {r['model_id']: r for r in old['selected_models']}
    # Equal raw bytes and loader code reproduce the old within-domain row order.
    for name, value in old['analysis_files'].items():
        if name.endswith(('data_contracts.py', 'features.py', 'schema.py', 'harmonize.py', 'survey.py')):
            require(sha(ROOT/name) == value, 'OLD_DOMAIN_LOADER_CHANGED')
    comps, shards = [], {}
    for arm in ARMS:
        directory = main/'main_evaluation_v2'/arm
        shard_path = directory/'evaluation_manifest.json'
        shard = read(shard_path)
        require(shard['study_freeze_sha256'] == MAIN_STUDY_SHA and shard['status'] == 'COMPLETE'
                and shard['arm_ids'] == [arm] and shard['source_provenance'] == old['t_source_provenance'],
                'INVALID_PARENT_SHARD')
        shards[arm] = {'manifest': bound(shard_path),
                       'individual': bound(directory/(arm+'_individual_model_metrics.json'))}
        require(shards[arm]['individual']['sha256'] == shard['artifacts'][arm+'_individual_model_metrics.json'],
                'PARENT_METRICS_HASH_MISMATCH')
        for backbone in BACKBONES:
            specs = [(m, .1) for m in COMPARATORS] + [('FAIRBIAS_BM', None)]
            for method, tau in specs:
                row = _parent_row(old, registry, arm, backbone, method, tau)
                model_refs = [old_refs[k] for k in row['model_ids']]
                if model_refs:
                    cid = model_refs[0]['candidate_id']
                    require(all(r['candidate_id'] == cid for r in model_refs)
                            and sorted(r['seed'] for r in model_refs) == sorted(registry[cid]['seeds']), 'OLD_SEEDS_INCOMPLETE')
                refs = []
                for r in model_refs:
                    name = arm+'_'+r['model_id']+'_predictions_T.npz'
                    binding = bound(directory/name)
                    require(binding['sha256'] == shard['artifacts'][name], 'OLD_PREDICTION_HASH_MISMATCH')
                    refs.append({'model_id': r['model_id'], 'seed': r['seed'], 'output_type': r['output_type'],
                                 'model_sha256': r['artifacts']['model.joblib']['sha256'], 'predictions': binding})
                comps.append({**row, 'selection_id': 'parent:'+row['selection_id'], 'parent_selection_id': row['selection_id'],
                              'origin': 'parent_fixed_BM' if tau is None else 'parent_tau10',
                              'prediction_refs': refs, 'study_branch': BRANCH})
    contrasts = contrast_registry(comps)
    require(Counter(r['family'] for r in contrasts) == {'primary': 10, 'secondary': 158}, 'CONTRAST_REGISTRY_MISMATCH')
    plan = {'schema_version': 'completion_admission_v1', 'decision': 'SUPERVISOR_ADMITTED_COMPLETION',
            'study_branch': BRANCH, 'known_T': True, 'evaluation_authorized': False,
            'training_manifest': bound(runtime/'control/manifest.json'), 'training_sources_root': str(runtime),
            'audit': bound(audit_path), 'parent_study': bound(study_path), 'parent_shards': shards,
            'parent_registration': bound(registration_path),
            'policy_document': bound(ROOT/'docs/plans/FAIRBIAS_COMPLETION_EVALUATION_POLICY_20260918.md'),
            'jobs': records, 'comparators': comps, 'contrasts': contrasts, 'family_sizes': FAMILY_SIZES,
            't_source_provenance': old['t_source_provenance'], 'taus': list(TAUS), 'reporting_tau': .1,
            'analysis_files': analysis_files(), 'environment': ev._environment()}
    fresh(output, plan)
    return plan


def validate_admission(path, expected_sha):
    value = read(verify({'path': str(Path(path).resolve()), 'sha256': expected_sha}))
    require(value['schema_version'] == 'completion_admission_v1'
            and value['decision'] == 'SUPERVISOR_ADMITTED_COMPLETION'
            and value['study_branch'] == BRANCH and value['known_T'] is True
            and value['evaluation_authorized'] is False, 'INVALID_ADMISSION_POLICY')
    require(value['analysis_files'] == analysis_files() and value['environment'] == ev._environment(),
            'ANALYSIS_SOURCE_OR_ENVIRONMENT_CHANGED')
    require(value['family_sizes'] == FAMILY_SIZES and value['taus'] == list(TAUS) and value['reporting_tau'] == .1,
            'ENDPOINT_POLICY_CHANGED')
    validate_job_coverage(value['jobs'])
    for field in ('training_manifest', 'audit', 'parent_study', 'policy_document', 'parent_registration'):
        verify(value[field])
    training = read(verify(value['training_manifest']))
    require(training['manifest_sha256'] == TRAINING_SEAL
            and digest({k: v for k, v in training.items() if k != 'manifest_sha256'}) == TRAINING_SEAL,
            'TRAINING_SEAL_INVALID')
    for name, value_sha in training['sources'].items():
        require(sha(Path(value['training_sources_root'])/name) == value_sha
                and sha(ROOT/name) == value_sha, 'TRAINING_SOURCE_COMPATIBILITY_FAILED')
    original = {r['job_id']: r for r in training['jobs']}
    audit = {r['job_id']: r for r in read(verify(value['audit']))['jobs']}
    for job in value['jobs']:
        require(all(job[k] == original[job['job_id']][k] for k in original[job['job_id']]), 'JOB_CONFIG_CHANGED')
        record = audit[job['job_id']]
        require(job['model_id'] == 'completion_model_'+job['job_id']
                and job['C_q_sha256'] == record['C_q_sha256'] and job['C_p_sha256'] == record['C_p_sha256']
                and job['policy']['sha256'] == record['model_sha256']
                and job['result']['sha256'] == record['result_sha256']
                and job['prepared_file']['sha256'] == job['prepared']['sha256'], 'AUDIT_BINDING_CHANGED')
        for key in ('policy', 'result', 'prepared_file'):
            verify(job[key])
    old = read(verify(value['parent_study']))
    require(value['parent_study']['sha256'] == MAIN_STUDY_SHA
            and value['t_source_provenance'] == old['t_source_provenance'], 'PARENT_STUDY_CHANGED')
    require(value['parent_registration']['sha256'] == old['registration']['sha256'], 'PARENT_REGISTRATION_CHANGED')
    registry = {c['candidate_id']: c for c in read(verify(value['parent_registration']))['candidates']}
    old_models = {r['model_id']: r for r in old['selected_models']}
    expected_slots = {(a, b, m, t) for a in ARMS for b in BACKBONES
                      for m, t in [(m, .1) for m in COMPARATORS]+[('FAIRBIAS_BM', None)]}
    require(len(value['comparators']) == 80 and set(value['parent_shards']) == set(ARMS)
            and {(c['arm_id'], c['backbone'], c['method'], c['tau']) for c in value['comparators']} == expected_slots,
            'COMPARATOR_COVERAGE_CHANGED')
    for comp in value['comparators']:
        oldrow = _parent_row(old, registry, comp['arm_id'], comp['backbone'], comp['method'], comp['tau'])
        expected_row = {**oldrow, 'selection_id': 'parent:'+oldrow['selection_id'],
                        'parent_selection_id': oldrow['selection_id'], 'study_branch': BRANCH,
                        'origin': 'parent_fixed_BM' if oldrow['tau'] is None else 'parent_tau10'}
        require({k: v for k, v in comp.items() if k != 'prediction_refs'} == expected_row
                and comp['training_weighted'] is False, 'PARENT_SELECTION_MAPPING_CHANGED')
        shard_binding = value['parent_shards'][comp['arm_id']]
        shard = read(verify(shard_binding['manifest']))
        metrics_path = verify(shard_binding['individual'])
        require(shard['study_freeze_sha256'] == MAIN_STUDY_SHA and shard['status'] == 'COMPLETE'
                and shard['arm_ids'] == [comp['arm_id']] and shard['source_provenance'] == old['t_source_provenance']
                and shard['artifacts'][metrics_path.name] == shard_binding['individual']['sha256'], 'PARENT_SHARD_CHANGED')
        require([r['model_id'] for r in comp['prediction_refs']] == comp['model_ids'], 'PARENT_MODEL_MAPPING_CHANGED')
        if comp['model_ids']:
            refs = [old_models[k] for k in comp['model_ids']]
            cid = refs[0]['candidate_id']
            require(all(r['candidate_id'] == cid for r in refs)
                    and sorted(r['seed'] for r in refs) == sorted(registry[cid]['seeds']), 'OLD_SEEDS_INCOMPLETE')
        for ref in comp['prediction_refs']:
            model = old_models[ref['model_id']]
            prediction_path = verify(ref['predictions'])
            require(ref['seed'] == model['seed'] and ref['output_type'] == model['output_type']
                    and ref['model_sha256'] == model['artifacts']['model.joblib']['sha256']
                    and prediction_path.name == comp['arm_id']+'_'+ref['model_id']+'_predictions_T.npz'
                    and shard['artifacts'][prediction_path.name] == ref['predictions']['sha256'], 'PARENT_PREDICTION_MAPPING_CHANGED')
    require(value['contrasts'] == contrast_registry(value['comparators']), 'CONTRAST_MAPPING_CHANGED')
    require(len({r['contrast_id'] for r in value['contrasts']}) == 168
            and Counter(r['family'] for r in value['contrasts']) == {'primary': 10, 'secondary': 158},
            'CONTRAST_COVERAGE_CHANGED')
    return value


def _load_policy(job):
    trained = read(verify(job['result']))['packages']
    for package in ('numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib'):
        require(importlib.metadata.version(package) == trained[package], 'CORE_PREDICTION_PACKAGE_CHANGED')
    policy = joblib.load(verify(job['policy']))
    require(type(policy) is FrozenDecisionPolicy and policy.is_calibrated
            and policy._adapter.is_fitted_ and policy._adapter.provenance_['status'] == 'COMPLETE_FEASIBLE'
            and policy._adapter.mode == job['config']['method'].removeprefix('FAIRBIAS_')
            and policy._adapter.adapter_params['random_state'] == job['seed'], 'POLICY_IDENTITY_MISMATCH')
    return policy


def selection_rows(jobs, metrics):
    require(set(metrics) == {j['job_id'] for j in jobs}, 'INCOMPLETE_S_METRICS')
    groups = defaultdict(list)
    for job in jobs:
        require(metrics[job['job_id']]['seed'] == job['seed']
                and metrics[job['job_id']]['status'] == 'VALID', 'S_MODEL_IDENTITY_CHANGED')
        groups[job['config']['candidate_id']].append(job)
    rows, all_taus = [], []
    for cid, members in groups.items():
        config = members[0]['config']
        aggregate = aggregate_configuration(config, [metrics[j['job_id']] for j in members])
        identity = {k: config[k] for k in ('arm_id', 'backbone', 'method', 'training_weighted')}
        for tau in TAUS:
            row = {**select_aggregate([aggregate], tau), **identity,
                   'selection_id': _new_id(config['arm_id'], config['backbone'], config['method']),
                   'study_branch': BRANCH, 'origin': 'adaptive_completed80', 'analysis_version': BRANCH,
                   'registered_fixed_candidate_id': cid, 'model_ids': [j['model_id'] for j in members]}
            all_taus.append(row)
            if tau == .1:
                rows.append(row)
    return rows, all_taus


def validate_selection(selection, admission):
    require(selection['schema_version'] == 'completion_selection_v1' and selection['study_branch'] == BRANCH
            and selection['known_T'] is True and selection['evaluation_authorized'] is False
            and selection['analysis_files'] == admission['analysis_files']
            and selection['environment'] == admission['environment'], 'INVALID_SELECTION_FREEZE')
    rows, all_taus = selection_rows(admission['jobs'], selection['individual_S'])
    require(selection['selections'] == rows and selection['S_sensitivity'] == all_taus
            and len(rows) == 16 and len(all_taus) == 48, 'S_SELECTION_MAPPING_CHANGED')
    reloads = selection['cross_platform_C_reload']
    require(len(reloads) == 80 and {r['job_id'] for r in reloads} == {j['job_id'] for j in admission['jobs']}
            and all(r['C_q_exact'] is True and type(r['C_p_bytes_equal']) is bool for r in reloads),
            'INCOMPLETE_C_RELOAD_EVIDENCE')


def freeze_completion_selection(admission_path, expected_sha, output):
    admission = validate_admission(admission_path, expected_sha)
    require(not Path(output).exists(), 'SELECTION_OUTPUT_EXISTS')
    partitions, metrics, reloads = {}, {}, []
    for job in admission['jobs']:
        arm = job['config']['arm_id']
        if arm not in partitions:
            payload = joblib.load(verify(job['prepared_file']))
            require(payload['data_identity'] == job['prepared']['data_identity']
                    and set(payload['partitions']) == {'fitting_F', 'calibration_C', 'selection_S'}
                    and 'X_T' not in payload, 'SELECTION_INPUT_IDENTITY_INVALID')
            partitions[arm] = (payload['partitions']['calibration_C'], payload['partitions']['selection_S'])
            del payload
        C, S = partitions[arm]
        require(C.role == 'calibration_C' and C.year == 2022 and S.role == 'selection_S' and S.year == 2023,
                'INCORRECT_DEVELOPMENT_PARTITION')
        policy = _load_policy(job)
        check = policy.predict(C.X_semantic, C.A)
        qsha = hashlib.sha256(check.q_decision.tobytes()).hexdigest()
        require(qsha == job['C_q_sha256'], 'C_DECISION_CHANGED_ACROSS_PLATFORMS')
        reloads.append({'job_id': job['job_id'], 'C_q_exact': True,
                        'C_p_bytes_equal': hashlib.sha256(check.p_event.tobytes()).hexdigest() == job['C_p_sha256']})
        bundle = policy.predict(S.X_semantic, S.A)
        spec = ARM_SPECS[arm]
        metrics[job['job_id']] = {
            'seed': job['seed'], 'status': 'VALID',
            'metrics_S': compute_survey_fairness_metrics(S.y, bundle.q_decision, S.A, S.WTFA_A, spec['expected_categories']),
            'risk_S': compute_risk_metrics(S.y, bundle, S.WTFA_A, S.A, spec['expected_categories'])}
        if len(metrics) % 10 == 0:
            print(f'S predicted and C verified: {len(metrics)}/80', flush=True)
        del policy, bundle, check
    rows, all_taus = selection_rows(admission['jobs'], metrics)
    require(len(rows) == 16 and len(all_taus) == 48, 'SELECTION_COVERAGE_MISMATCH')
    result = {'schema_version': 'completion_selection_v1', 'study_branch': BRANCH, 'known_T': True,
              'evaluation_authorized': False, 'admission': bound(admission_path), 'selections': rows,
              'S_sensitivity': all_taus, 'individual_S': metrics, 'cross_platform_C_reload': reloads,
              'analysis_files': admission['analysis_files'], 'environment': admission['environment']}
    validate_selection(result, admission)
    validate_admission(admission_path, expected_sha)
    fresh(output, result)
    return result


def freeze_completion_study(selection_path, expected_sha, output):
    selection = read(verify({'path': str(Path(selection_path).resolve()), 'sha256': expected_sha}))
    admission = validate_admission(selection['admission']['path'], selection['admission']['sha256'])
    validate_selection(selection, admission)
    study = {'schema_version': 'completion_study_v1', 'study_branch': BRANCH, 'known_T': True,
             'evaluation_authorized': False, 'selection': bound(selection_path), 'admission': selection['admission'],
             'selections': selection['selections']+admission['comparators'], 'contrasts': admission['contrasts'],
             'family_sizes': FAMILY_SIZES, 't_source_provenance': admission['t_source_provenance'],
             'analysis_files': admission['analysis_files'], 'environment': admission['environment'],
             'bootstrap_replicates': 2000, 'bootstrap_seed': 20260914, 'arm_ids': list(ARMS)}
    require(len({r['selection_id'] for r in study['selections']}) == 96, 'DUPLICATE_SELECTION_ID')
    fresh(output, study)
    return study


def validate_study(path, expected_sha):
    study = read(verify({'path': str(Path(path).resolve()), 'sha256': expected_sha}))
    require(study['schema_version'] == 'completion_study_v1' and study['study_branch'] == BRANCH
            and study['known_T'] is True and study['evaluation_authorized'] is False, 'INVALID_STUDY')
    admission = validate_admission(study['admission']['path'], study['admission']['sha256'])
    selection = read(verify(study['selection']))
    validate_selection(selection, admission)
    require(selection['admission'] == study['admission']
            and study['selections'] == selection['selections']+admission['comparators']
            and study['contrasts'] == admission['contrasts'] and study['family_sizes'] == FAMILY_SIZES
            and study['t_source_provenance'] == admission['t_source_provenance']
            and study['analysis_files'] == analysis_files() and study['environment'] == ev._environment()
            and study['bootstrap_replicates'] == 2000 and study['bootstrap_seed'] == 20260914
            and study['arm_ids'] == list(ARMS),
            'STUDY_NO_LONGER_MATCHES_FREEZE')
    return study, admission


def evaluate_completion_arm(study_path, expected_sha, release_path, release_sha, output, *, arm_id, data_root=None):
    study, admission = validate_study(study_path, expected_sha)
    release = read(verify({'path': str(Path(release_path).resolve()), 'sha256': release_sha}))
    validate_release(study, expected_sha, release, [arm_id])
    require(not Path(output).exists(), 'EVALUATION_OUTPUT_EXISTS')
    root = Path(data_root or ROOT).resolve()
    ev._verify_t_inputs(study, root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cohort = load_local_nhis_cohort(root, years=(2024,))
    ev._validate_observed_provenance(study['t_source_provenance'], cohort.attrs.get('source_provenance'))
    T = load_arm_partitions(cohort, arm_id)['evaluation_T']
    design, spec = ev._normalized_annual_design(T), ARM_SPECS[arm_id]
    require(T.year == design.year == 2024 and T.role == 'evaluation_T', 'INCORRECT_T_DOMAIN')
    selections = [r for r in study['selections'] if r['arm_id'] == arm_id]
    q_models, individual = {}, {}
    for job in admission['jobs']:
        if job['config']['arm_id'] != arm_id:
            continue
        policy = _load_policy(job)
        bundle = policy.predict(T.X_semantic, T.A)
        key = job['model_id']
        require(bundle.n_samples == len(T), 'NEW_PREDICTION_LENGTH_MISMATCH')
        q_models[key] = design.expand(bundle.q_decision)
        individual[key] = dict(model_id=key,
            weighted=compute_survey_fairness_metrics(T.y, bundle.q_decision, T.A, T.WTFA_A, spec['expected_categories']),
            unweighted=compute_survey_fairness_metrics(T.y, bundle.q_decision, T.A, np.ones(len(T)), spec['expected_categories']),
            risk=compute_risk_metrics(T.y, bundle, T.WTFA_A, T.A, spec['expected_categories']),
            unweighted_risk=compute_risk_metrics(T.y, bundle, np.ones(len(T)), T.A, spec['expected_categories']),
            threshold_05=compute_survey_fairness_metrics(T.y, (bundle.p_event >= .5).astype(float), T.A, T.WTFA_A, spec['expected_categories']))
        np.savez_compressed(output/(key+'_predictions_T.npz'), p=bundle.p_event, q=bundle.q_decision)
        del policy, bundle
    prior = read(verify(admission['parent_shards'][arm_id]['individual']))
    reused = set()
    for selection in selections:
        for ref in selection.get('prediction_refs', []):
            key = ref['model_id']
            if key in reused:
                continue
            with np.load(verify(ref['predictions']), allow_pickle=False) as arrays:
                q = arrays['q']
                require(q.ndim == 1 and len(q) == len(T) and np.isfinite(q).all()
                        and np.all((q >= 0) & (q <= 1)), 'OLD_DOMAIN_PREDICTION_MISMATCH')
                q_models[key] = design.expand(q)
                individual[key] = prior[key]
                observed = compute_survey_fairness_metrics(T.y, q, T.A, T.WTFA_A, spec['expected_categories'])
                for metric in ('balanced_accuracy', 'eo_gap', 'dp_gap'):
                    require(np.isclose(observed[metric], individual[key]['weighted'][metric], rtol=1e-10, atol=1e-12),
                            'OLD_PREDICTION_METRIC_PARITY_FAILED')
                require(ref['output_type'] == 'decision_probability_q' or 'p' in arrays, 'OLD_RISK_SEMANTICS_MISMATCH')
            reused.add(key)
    expected = {key for r in selections for key in r['model_ids']}
    require(set(q_models) == expected, 'INCOMPLETE_ARM_MODELS')
    ids = {r['selection_id'] for r in selections}
    contrasts = [r for r in study['contrasts'] if r['reference_selection_id'] in ids]
    from .completion_statistics import evaluate_statistics
    summaries, paired = evaluate_statistics(T, arm_id, selections, q_models, individual, output,
                                             contrasts=contrasts, family_sizes=FAMILY_SIZES, B=2000)
    from .cohort_reporting import summarize_partition, summarize_eligibility
    ev.write_json(output/'cohort_T.json', summarize_partition(T, spec['expected_categories']))
    ev.write_json(output/'eligibility_T.json', summarize_eligibility(cohort))
    validate_study(study_path, expected_sha)
    ev._verify_t_inputs(study, root)
    verify({'path': str(Path(release_path).resolve()), 'sha256': release_sha})
    manifest = {'schema_version': 'completion_arm_evaluation_v1', 'status': 'COMPLETE',
                'known_T': True,
                'study_branch': BRANCH, 'study_sha256': expected_sha, 'arm_id': arm_id,
                'release': bound(release_path), 'models': sorted(q_models), 'reused_parent_models': sorted(reused),
                'source_provenance': cohort.attrs['source_provenance'], 'family_sizes': FAMILY_SIZES,
                'domain_record_order_sha256': digest(list(T.record_keys)),
                'annual_record_order_sha256': digest(list(design.record_keys)),
                'artifacts': {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    fresh(output/'evaluation_manifest.json', manifest)
    summary = {'schema_version': 'completion_summary_T_v1', 'study_branch': BRANCH,
               'known_T': True,
               'study_sha256': expected_sha, 'evaluation_manifest_sha256': sha(output/'evaluation_manifest.json'),
               'arm_id': arm_id, 'selections': summaries, 'paired_contrasts': paired}
    ev.write_json(output/'summary_T.json', summary)
    return summary


def merge_completion(study_path, study_sha, arm_directories, output):
    study, admission = validate_study(study_path, study_sha)
    require(len(arm_directories) == 4 and not Path(output).exists(), 'FOUR_NEW_ARM_SHARDS_REQUIRED')
    seen, selections, pairs, receipts = set(), [], [], []
    for directory in map(Path, arm_directories):
        manifest_path, summary_path = directory/'evaluation_manifest.json', directory/'summary_T.json'
        manifest, summary = read(manifest_path), read(summary_path)
        arm = manifest['arm_id']
        require(arm in ARMS and arm not in seen and summary['arm_id'] == arm, 'DUPLICATE_OR_WRONG_ARM')
        seen.add(arm)
        require(manifest['status'] == 'COMPLETE' and manifest['known_T'] is summary['known_T'] is True
                and manifest['study_branch'] == summary['study_branch'] == BRANCH
                and manifest['study_sha256'] == summary['study_sha256'] == study_sha
                and summary['evaluation_manifest_sha256'] == sha(manifest_path)
                and manifest['source_provenance'] == study['t_source_provenance']
                and manifest['family_sizes'] == FAMILY_SIZES, 'SHARD_IDENTITY_CHANGED')
        validate_release(study, study_sha, read(verify(manifest['release'])), [arm])
        for name, value in manifest['artifacts'].items():
            require(Path(name).name == name and sha(directory/name) == value, 'SHARD_ARTIFACT_CHANGED')
        expected_rows = [r for r in study['selections'] if r['arm_id'] == arm]
        expected_pairs = [c for c in study['contrasts'] if c['reference_selection_id'] in {r['selection_id'] for r in expected_rows}]
        require(len(summary['selections']) == len(expected_rows) and len(summary['paired_contrasts']) == len(expected_pairs)
                and all(all(actual[k] == v for k, v in expected.items()) for actual, expected in zip(summary['selections'], expected_rows))
                and all(all(actual[k] == v for k, v in expected.items()) for actual, expected in zip(summary['paired_contrasts'], expected_pairs))
                and set(manifest['models']) == {m for r in expected_rows for m in r['model_ids']}, 'SHARD_COVERAGE_CHANGED')
        selections.extend(summary['selections'])
        pairs.extend(summary['paired_contrasts'])
        receipts.append({'manifest': bound(manifest_path), 'summary': bound(summary_path)})
    require(len(selections) == 96 and len(pairs) == 168 and seen == set(ARMS), 'MERGED_COVERAGE_CHANGED')
    merged = {'schema_version': 'completion_merged_summary_v1', 'status': 'COMPLETE', 'study_branch': BRANCH,
              'known_T': True, 'study': bound(study_path), 'shards': receipts, 'family_sizes': FAMILY_SIZES,
              'selections': selections, 'paired_contrasts': pairs,
              'limitations': ['Retrospective computational extension after original T was known',
                              'Fixed FairBias anchors and original comparator selection budgets differ',
                              'Inference conditions on the fitted models; seed variation is separate']}
    fresh(output, merged)
    return merged
