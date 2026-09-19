"""Complete matrix ownership, predecessor drain, resource and real-worker wiring."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_nhis_bmae_sensitivity_queue as queue
from test_bmae_sensitivity_worker import sensitivity_case, save
from test_fairbias_recovery_queue import supervisor

worker,base,GiB=queue.worker,queue.base,queue.GiB


def scheduler(case,tmp_path):
    plan={'schema_version':queue.PLAN_SCHEMA,'runtime_variant':worker.VARIANT,
        'recovery_policy':worker.POLICY,'recovery_policy_spec':worker.policy_spec(),
        'runtime_environment':worker.THREAD_ENVIRONMENT,'execution_budget':worker.DEFAULT_EXECUTION_BUDGET,
        'runtime_source_files':case.files,'runtime_source_identity':case.runtime_id,'worker_script':worker.WORKER_SOURCE,
        'registration_sha256':base.file_sha(case.regpath),'sensitivity_manifest_path':str(case.manifest),
        'sensitivity_manifest_sha256':base.file_sha(case.manifest)}
    planpath=tmp_path/'plan.json';save(planpath,plan)
    policy=tmp_path/'policy.json'
    save(policy,{'status':'SUPERVISOR_AUTHORIZED_PARALLEL_DEVELOPMENT','registration_sha256':plan['registration_sha256'],
        'admitted_runtime_identities':[case.runtime_id],'max_workers':28,'max_total_rss_bytes':48*GiB,
        'external_supervisors':[]})
    return queue.BMAESensitivityScheduler(tmp_path/('queue_'+case.runtime_id),original_run=case.original,
        selected=queue.sensitivity_jobs(case.original,case.manifest),runtime_plan=plan,runtime_plan_path=planpath,
        external_runs=(),worker=queue.ROOT/worker.WORKER_SOURCE,source_script=Path(queue.__file__),
        limits=base.ResourceLimits(workers=28,max_workers=28,total_rss_bytes=48*GiB,worker_rss_bytes=4*GiB,fit_seconds=7200),
        methods=['FAIRBIAS_BM_AE'],policy_path=policy)


def test_exact40_including19_original_valid_share_one_budget_policy(sensitivity_case,tmp_path):
    s=scheduler(sensitivity_case,tmp_path)
    assert queue.runtime_plan(s.runtime_plan_path)['runtime_source_identity']==sensitivity_case.runtime_id
    assert len(s.selected)==40
    assert sum(x['original_status']=='VALID' for x in s.selected.values())==19
    jobs=s.discover()
    assert len({j.path.name for j in jobs})==40
    assert all(j.config['method']=='FAIRBIAS_BM_AE' and j.representation_key is None for j in jobs)
    for j in jobs:
        value=s._job_payload(j)
        assert value['execution_budget']==worker.DEFAULT_EXECUTION_BUDGET
        assert value['sensitivity_manifest_sha256']==base.file_sha(sensitivity_case.manifest)
        assert value['config']==s.selected[j.path.name]['config']
    assert not (s.run_path/'registration.json').exists()
    with pytest.raises(base.SchedulerError,match='not fresh'):
        s.discover()


@pytest.mark.parametrize('bad',['missing','extra','non_bmae','seed','job_hash','result_hash','config'])
def test_manifest_subset_or_rebound_evidence_refused(sensitivity_case,bad):
    c=sensitivity_case;manifest=base._json(c.manifest);name=next(iter(manifest['jobs']));row=manifest['jobs'][name]
    if bad=='missing':manifest['jobs'].pop(name)
    elif bad=='extra':manifest['jobs']['unregistered']=copy.deepcopy(row)
    elif bad=='non_bmae':row['config']['method']='FAIRBIAS_JOINT'
    elif bad=='seed':row['seed']=999
    elif bad=='job_hash':row['original_job_sha256']='0'*64
    elif bad=='result_hash':row['original_result_sha256']='0'*64
    else:row['config']['params']['max_outer_iterations']=40
    save(c.manifest,manifest)
    with pytest.raises(base.SchedulerError,match='exactly bind all 40'):
        queue.sensitivity_jobs(c.original,c.manifest)


def predecessor(tmp_path):
    root=tmp_path/'previous';rid='a'*64;joint=root/('fairbias_joint_strict_exact_v1_'+rid)
    save(joint/'parallel_scheduler_live_status.json',{'status':'COMPLETE','scheduled_count':38,
        'finalized_count':38,'active_count':0,'run':str(joint)})
    save(root/'queue_completion.json',{'status':'COMPLETE','fit_success_established':False,
        'phases':[{'runtime_source_identity':rid,'scheduled':38,'completed':38,'failed':26}]})
    return {'queue_root':str(root),'joint_run':str(joint),'joint_runtime_source_identity':rid,
        'completion_sha256':base.file_sha(root/'queue_completion.json'),'controller_pid':777,'pilot_supervisors':[]}


@pytest.mark.parametrize('bad',['controller_live','unsealed','joint_active','pilot_live','training_live'])
def test_predecessor_gate_rejects_residual_work(tmp_path,bad):
    binding=predecessor(tmp_path)
    probe=lambda pid:None
    training=lambda:[]
    if bad=='controller_live':probe=lambda pid:{'command':['old_controller']}
    elif bad=='unsealed':(Path(binding['queue_root'])/'queue_completion.json').unlink()
    elif bad=='joint_active':
        path=Path(binding['joint_run'])/'parallel_scheduler_live_status.json'
        value=base._json(path);value['active_count']=1;save(path,value)
    elif bad=='pilot_live':
        _,item,_,process=supervisor(tmp_path/'pilot')
        binding['pilot_supervisors']=[item]
        probe=lambda pid:process if pid==123 else None
    else:training=lambda:[123]
    with pytest.raises(base.SchedulerError):
        queue.predecessor_ready(binding,probe=probe,training=training)


def test_complete_failed_predecessor_is_not_mislabelled_success(tmp_path):
    binding=predecessor(tmp_path)
    result=queue.predecessor_ready(binding,probe=lambda pid:None,training=lambda:[])
    assert result['controller_absent'] and result['other_training_absent']
    assert 'fit_success_established' not in result


def test_process_scan_counts_worker_and_pilot_but_excludes_self(tmp_path):
    proc=tmp_path/'proc';run=tmp_path/'own'
    commands={1:['python',str(queue.ROOT/'scripts/pilot_nhis_joint_ae_recovery.py'),'--job','/old/job.json'],
        2:['python',str(queue.ROOT/worker.WORKER_SOURCE),'worker','--job',str(run/'jobs/x/job.json')],
        3:['python',str(queue.ROOT/'scripts/run_nhis_fairbias_recovery_worker.py'),'worker','--job','/external/jobs/x/job.json']}
    for pid,command in commands.items():
        p=proc/str(pid);p.mkdir(parents=True)
        (p/'stat').write_text(f'{pid} (python) S '+'0 '*25)
        (p/'cmdline').write_bytes(b'\0'.join(v.encode() for v in command)+b'\0')
    assert queue.live_training(exclude_run=run,proc_root=proc)==[1,3]


def test_memory_headroom_slots_and_external_race(sensitivity_case,tmp_path,monkeypatch):
    s=scheduler(sensitivity_case,tmp_path);s.discover()
    monkeypatch.setattr(queue,'live_training',lambda **k:[])
    monkeypatch.setattr(base,'external_worker_count',lambda runs:0)
    monkeypatch.setattr(base,'cgroup_memory',lambda:(0,60*GiB))
    assert s._launchable_index(s.jobs,28)==0
    s._running={i:SimpleNamespace(last_rss_bytes=2*GiB) for i in range(23)}
    assert s._launchable_index(s.jobs,5) is None
    s._running={};monkeypatch.setattr(base,'cgroup_memory',lambda:(45*GiB,60*GiB))
    assert s._launchable_index(s.jobs,28) is None
    monkeypatch.setattr(base,'cgroup_memory',lambda:(0,60*GiB))
    monkeypatch.setattr(queue,'live_training',lambda **k:[999])
    assert s._launchable_index(s.jobs,28) is None


def test_new_scientific_variant_has_independent_but_permanent_claims(sensitivity_case,tmp_path):
    s=scheduler(sensitivity_case,tmp_path)
    old=sensitivity_case.original.parent/'fairbias_recovery_ownership';old.mkdir()
    (old/'preserved').write_bytes(b'historical claim')
    queue.claim_jobs(s.original_run,s.selected,s.runtime_plan,s.run_path)
    with pytest.raises(base.SchedulerError,match='already has an owner'):
        queue.claim_jobs(s.original_run,s.selected,s.runtime_plan,tmp_path/'duplicate')
    assert (old/'preserved').read_bytes()==b'historical claim'


def test_generated_real_worker_through_scheduler(sensitivity_case,tmp_path,monkeypatch):
    c=sensitivity_case;s=scheduler(c,tmp_path)
    name=c.path.parent.name
    # One production subprocess suffices to verify dispatch wiring; formal main
    # obtains its complete 40-job set only through sensitivity_jobs above.
    s.selected={name:s.selected[name]};s._selected_names={name}
    s.limits=queue._q.replace(s.limits,workers=1,poll_seconds=.02)
    monkeypatch.setattr(queue,'live_training',lambda **k:[])
    monkeypatch.setattr(base,'external_worker_count',lambda runs:0)
    monkeypatch.setattr(base,'cgroup_memory',lambda:(0,60*GiB))
    result=s.run()
    assert result=={'scheduled':1,'completed':1,'resumed':0,'failed':0}
    spec=s.jobs[0];state=SimpleNamespace(spec=spec,termination=None,process=SimpleNamespace(returncode=0))
    assert s._receipt_error(state) is None
    path=spec.path/'runtime_receipt.json';receipt=base._json(path)
    assert receipt['original_status']=='VALID' and receipt['original_valid_job_refit']
    receipt['model_admission']['search_complete']=False;save(path,receipt)
    assert s._receipt_error(state)=='BM_AE sensitivity VALID lacks complete model/C evidence'
    assert c.before=={str(p):base.file_sha(p) for p in c.original.rglob('*') if p.is_file()}
