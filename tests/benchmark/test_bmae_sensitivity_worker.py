"""Generated full registration; bounded production fits under the new policy."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import FunctionTransformer

import scripts.run_nhis_bmae_sensitivity_worker as worker
from scripts.run_nhis_bmae_sensitivity_queue import build_sensitivity_manifest
from test_fairbias_recovery_worker import _part
from test_joint_ae_recovery import _synthetic_commits
from nhis_fairbias.benchmark import experiment_worker
from nhis_fairbias.benchmark.adapters.adapter_fairbias_ae import FairBiasAEAdapter
from nhis_fairbias.benchmark.adapters.adapter_joint_ae_recovery import JointAERecoveryAdapter


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def sensitivity_case(tmp_path, monkeypatch, request):
    assert sys.flags.hash_randomization == 0, 'Run with PYTHONHASHSEED=0'
    for name, value in worker.THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    epsilon = getattr(request, 'param', {}).get('epsilon_ratio', .75)
    original = tmp_path / 'original'
    prepared = original / 'prepared' / 'arm_synthetic.joblib'
    prepared.parent.mkdir(parents=True)
    parts = {r: _part(r, n, y) for r, n, y in [('fitting_F',48,2022),('calibration_C',48,2022),('selection_S',32,2023)]}
    data = {'data_identity': 'generated_bmae_sensitivity', 'partitions': parts,
            'X_F': np.zeros((48,1)), 'X_C': np.zeros((48,1)), 'X_S': np.zeros((32,1)),
            'preprocessor': FunctionTransformer().fit(np.zeros((48,1)))}
    joblib.dump(data, prepared)
    old_cache = original / 'representation_cache'
    old_cache.mkdir()
    (old_cache / 'preserved').write_bytes(b'old cache untouched')
    configs=[]
    for i in range(8):
        config={'method':'FAIRBIAS_BM_AE','backbone':'LR','arm_id':'arm_synthetic',
            'status':'REGISTERED','training_weighted':False,'seeds':[0,7,19,37,73],
            'params':{'mode':'BM_AE','epsilon_ratio':epsilon,'max_bm_steps':50,
                'max_geometry_evaluations':20000,'max_outer_iterations':10,
                'max_utility_evaluations':500,'estimator_params':{'C':1.+i}}}
        config['candidate_id']=worker.identity(config)[:20]
        configs.append(config)
    frozen=['src/nhis_fairbias/benchmark/experiment_worker.py',
            'src/nhis_fairbias/benchmark/adapters/adapter_fairbias_ae.py']
    registration={'candidates':configs,'source_identity':'generated_original_source',
        'source_files':{p:worker.file_sha(worker.ROOT/p) for p in frozen},
        'prepared':{'arm_synthetic':{'path':str(prepared),'sha256':worker.file_sha(prepared),
                                   'data_identity':data['data_identity']}}}
    regpath=original/'registration.json'
    save(regpath,registration)
    names=[]
    for config in configs:
        for seed in config['seeds']:
            name=f"{config['candidate_id']}_s{seed}"
            index=len(names);names.append(name)
            old=original/'jobs'/name/'job.json'
            payload={'config':config,'seed':seed,'source_identity':registration['source_identity'],
                'data_path':str(prepared),'data_sha256':worker.file_sha(prepared),'data_identity':data['data_identity'],
                'cache_path':str(old_cache),'representation_key':None}
            save(old,payload)
            status='VALID' if index<19 else 'TIME_LIMIT' if index>=35 else 'BUDGET_EXHAUSTED'
            save(old.parent/'result.json',{'candidate_id':config['candidate_id'],'seed':seed,'status':status,
                'error':'MDS_ITERATION_CAP' if index in (33,34) else 'AE commit limit' if index>=19 else None})
    manifest_path=tmp_path/'sensitivity_manifest.json'
    save(manifest_path,build_sensitivity_manifest(original))
    files=worker.build_runtime_source_manifest()
    rid=worker.runtime_identity(files)
    run=tmp_path/('bmae_test_'+rid)
    name=names[0]
    old=original/'jobs'/name/'job.json'
    job={**json.loads(old.read_text()),'runtime_variant':worker.VARIANT,'runtime_source_identity':rid,
        'runtime_source_files':files,'runtime_environment':worker.THREAD_ENVIRONMENT,
        'recovery_policy':worker.POLICY,'execution_budget':worker.DEFAULT_EXECUTION_BUDGET,
        'recovery_run_path':str(run),'cache_path':str(run/'representation_cache'),
        'registration_path':str(regpath),'registration_sha256':worker.file_sha(regpath),
        'original_job_path':str(old),'original_job_sha256':worker.file_sha(old),
        'original_result_sha256':worker.file_sha(old.parent/'result.json'),'original_representation_key':None,
        'sensitivity_manifest_path':str(manifest_path),'sensitivity_manifest_sha256':worker.file_sha(manifest_path)}
    path=run/'jobs'/name/'job.json'
    save(path,job)
    snapshot={str(p):worker.file_sha(p) for p in original.rglob('*') if p.is_file()}
    return SimpleNamespace(original=original,registration=registration,regpath=regpath,prepared=prepared,
        data=data,run=run,path=path,job=job,files=files,runtime_id=rid,manifest=manifest_path,
        old_cache=old_cache,before=snapshot)


def assert_valid(case, receipt):
    assert receipt['status']=='VALID' and receipt['runtime_checks_passed'],receipt
    assert receipt['reload_exact_C'] and receipt['model_admission']['status']=='COMPLETE_FEASIBLE'
    assert receipt['original_valid_job_refit'] and not receipt['legacy_valid_results_admitted']
    assert receipt['representation_key'] is None and not receipt['representation_cache_reused']
    for name,digest in receipt['files'].items():
        assert worker.file_sha(case.path.parent/name)==digest
    artifact=joblib.load(case.path.parent/'model.joblib')
    S=case.data['partitions']['selection_S']
    predicted=artifact['policy'].predict(S.X_semantic,S.A)
    with np.load(case.path.parent/'predictions_S.npz') as persisted:
        np.testing.assert_array_equal(predicted.q_decision,persisted['q'])
        np.testing.assert_array_equal(predicted.p_event,persisted['p'])
    assert case.before=={str(p):worker.file_sha(p) for p in case.original.rglob('*') if p.is_file()}
    return artifact['policy']._adapter


def test_original_valid_is_refit_and_normal_cap10_does_not_retry(sensitivity_case):
    factory=experiment_worker.make_adapter
    receipt=worker.execute_recovery_job(sensitivity_case.path)
    adapter=assert_valid(sensitivity_case,receipt)
    assert [(a['ae_commit_cap'],a['status']) for a in adapter.fit_attempts_]==[(10,'FIT_RETURNED')]
    assert experiment_worker.make_adapter is factory
    with pytest.raises(ValueError,match='already has output'):
        worker.execute_recovery_job(sensitivity_case.path)


@pytest.mark.parametrize('sensitivity_case',[{'epsilon_ratio':100}],indirect=True)
def test_actual_commit_cap10_then_fresh40_is_complete(sensitivity_case,monkeypatch):
    calls=_synthetic_commits(monkeypatch,stop_after=12)
    receipt=worker.execute_recovery_job(sensitivity_case.path)
    adapter=assert_valid(sensitivity_case,receipt)
    assert calls==list(range(1,11))+list(range(1,14))
    assert [(a['ae_commit_cap'],a['status']) for a in adapter.fit_attempts_]==[(10,'AE_COMMIT_CAP'),(40,'FIT_RETURNED')]
    assert adapter._delegate.max_utility_evaluations==500 and adapter._delegate.max_geometry_evaluations==20000


@pytest.mark.parametrize('sensitivity_case',[{'epsilon_ratio':100}],indirect=True)
def test_cap40_exhaustion_remains_no_model(sensitivity_case,monkeypatch):
    _synthetic_commits(monkeypatch)
    receipt=worker.execute_recovery_job(sensitivity_case.path)
    assert receipt['status']=='BUDGET_EXHAUSTED' and receipt['runtime_checks_passed']
    assert not (sensitivity_case.path.parent/'model.joblib').exists()
    assert not (sensitivity_case.path.parent/'predictions_S.npz').exists()


@pytest.mark.parametrize('reason',['MDS_ITERATION_CAP','FairBias AE utility evaluations','FairBias AE geometry evaluations'])
def test_other_budgets_never_trigger_ae_retry(sensitivity_case,monkeypatch,reason):
    caps=[]
    def fail(self,*args,**kwargs):
        caps.append(self.max_outer_iterations)
        self.termination_reason_='BUDGET_EXHAUSTED'
        raise RuntimeError('BUDGET_EXHAUSTED: '+reason)
    monkeypatch.setattr(FairBiasAEAdapter,'fit_development',fail)
    receipt=worker.execute_recovery_job(sensitivity_case.path)
    assert receipt['status']=='BUDGET_EXHAUSTED' and caps==[10]
    assert not (sensitivity_case.path.parent/'model.joblib').exists()


@pytest.mark.parametrize('bad',['policy','budget','source','original_result','manifest','old_cache','seed','method','geometry_budget'])
def test_invalid_inputs_fail_before_fit(sensitivity_case,monkeypatch,bad):
    c=sensitivity_case;job=copy.deepcopy(c.job)
    if bad=='policy':job['recovery_policy']='joint_strict_exact_v1'
    elif bad=='budget':job['execution_budget']['fit_seconds']=1800
    elif bad=='source':job['runtime_source_files'][worker.WORKER_SOURCE]='0'*64
    elif bad=='original_result':job['original_result_sha256']='0'*64
    elif bad=='manifest':job['sensitivity_manifest_sha256']='0'*64
    elif bad=='old_cache':job['cache_path']=str(c.old_cache)
    elif bad=='seed':job['seed']=999
    elif bad=='method':job['config']['method']='FAIRBIAS_JOINT'
    else:job['config']['params']['max_geometry_evaluations']=20001
    save(c.path,job)
    monkeypatch.setattr(experiment_worker,'execute_job',lambda *a:pytest.fail('Invalid input reached fit'))
    with pytest.raises((ValueError,FileNotFoundError)):
        worker.execute_recovery_job(c.path)
    assert not (c.path.parent/'runtime_started.json').exists()


def test_incomplete_geometry_cannot_reach_calibration_or_S(sensitivity_case,monkeypatch):
    original=JointAERecoveryAdapter.fit_development
    def corrupt(self,*a,**kw):
        original(self,*a,**kw)
        self.fit_attempts_[0]['mds_fits'][0]['status']='MDS_ITERATION_CAP'
        return self
    monkeypatch.setattr(JointAERecoveryAdapter,'fit_development',corrupt)
    receipt=worker.execute_recovery_job(sensitivity_case.path)
    assert receipt['status']=='FAILED' and receipt['runtime_checks_passed']
    assert not (sensitivity_case.path.parent/'predictions_S.npz').exists()


def test_real_cli_process_generates_bound_S_outputs(sensitivity_case):
    child=subprocess.run([sys.executable,'-B',str(worker.ROOT/worker.WORKER_SOURCE),'worker','--job',str(sensitivity_case.path)],
        env={**os.environ,**worker.THREAD_ENVIRONMENT,'PYTHONPATH':str(worker.ROOT/'src')},capture_output=True,text=True,timeout=60)
    assert child.returncode==0,child.stderr+child.stdout
    receipt=json.loads((sensitivity_case.path.parent/'runtime_receipt.json').read_text())
    assert_valid(sensitivity_case,receipt)


def test_import_preserves_both_existing_worker_policies():
    import scripts.run_nhis_fairbias_recovery_worker as old
    import scripts.run_nhis_numerical_recovery_worker as numerical
    assert old.VARIANT=='fairbias_recovery_v1_20260917'
    assert set(old.POLICIES)=={'joint_strict_exact_v1','bm_mds_retry_v1'}
    assert numerical.VARIANT=='eg_lfr_numerical_recovery_v1_20260917'
    assert worker._core is not old and worker._core._base is not old._base
