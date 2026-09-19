"""Resource-only takeover contracts; synthetic receipts, no NHIS input."""
import sys
import json
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import run_nhis_completion_takeover_v2 as q


def test_partition_never_dispatches_an_adopted_or_finished_task():
    ids=[str(i) for i in range(80)]
    assert q.owned_partition(ids,ids[:10],ids[10:13])==sorted(ids[13:])
    assert len(q.owned_partition(ids,ids[:10],[]))==70
    with pytest.raises(ValueError):q.owned_partition(ids,['0'],['0'])
    with pytest.raises(ValueError):q.owned_partition(ids,['outside'],[])


@pytest.mark.parametrize('state,start,expected',[('R',12,True),('S',12,True),('Z',12,False),('R',13,False)])
def test_pid_reuse_and_zombies_not_treated_as_owned_live_work(monkeypatch,state,start,expected):
    monkeypatch.setattr(q,'process_identity',lambda p:{'pid':p,'start_ticks':start,'state':state})
    assert q.process_alive({'pid':20,'start_ticks':12}) is expected


def test_adoption_preserves_unknown_exit_code_and_requires_bound_model(tmp_path):
    model=tmp_path/'policy.joblib';model.write_bytes(b'synthetic')
    record={'status':'FC_COMPLETE_FEASIBLE','job_id':'job','manifest_sha256':'seal',
      'reload_exact':True,'formal_benchmark_admission':False,'S_T_evaluated':False,
      'completion':{'status':'COMPLETE_FEASIBLE'},'model_sha256':q.sha(model)}
    q.atomic(tmp_path/'result.json',record)
    result=q.check_artifact(tmp_path,'job','seal',None,adopted=True)
    assert result['status']=='FC_ARTIFACT_READY_FOR_REVIEW'
    assert result['exit_code'] is None and result['exit_code_observed'] is False
    assert q.check_artifact(tmp_path,'job','seal',None,adopted=False)['status']=='REQUIRES_REVIEW'
    assert q.check_artifact(tmp_path,'job','seal',1,adopted=False)['status']=='REQUIRES_REVIEW'
    assert q.check_artifact(tmp_path,'job','wrong-seal',0,adopted=False)['status']=='REQUIRES_REVIEW'
    model.write_bytes(b'changed')
    assert q.check_artifact(tmp_path,'job','seal',None,adopted=True)['status']=='REQUIRES_REVIEW'


def test_no_receipt_is_never_completed(tmp_path):
    assert q.check_artifact(tmp_path,'job','seal',None,adopted=True)['status']=='INCOMPLETE_NO_RECEIPT'


def test_actual_controller_adopts56_across_two_outputs_and_dispatches21_once(tmp_path,monkeypatch):
    import run_nhis_fairbias_completion as entry
    root=tmp_path/'frozen';old=root/'runs/full80_v2';old.mkdir(parents=True)
    (root/'prepared').mkdir();prepared=root/'prepared/arm_001.joblib';prepared.write_bytes(b'synthetic')
    control=tmp_path/'control';control.mkdir()
    prior=tmp_path/'previous/jobs';prior.mkdir(parents=True)
    ids=[f'job{i:02d}' for i in range(80)]
    jobs=[{'job_id':j,'config':{'arm_id':'arm_001'},'prepared':{'sha256':q.sha(prepared)}} for j in ids]
    monkeypatch.setattr(entry,'load_manifest',lambda p:{'manifest_sha256':'seal','jobs':jobs})
    q.atomic(control/'resources.json',{'concurrency':64})
    q.atomic(control/'handoff.json',{'manifest_sha256':'seal','new_controller_sha256':q.sha(q.__file__),
      'old_run':str(old),'old_controller':{'pid':999,'start_ticks':1},
      'completed':{j:{'status':'FC_PILOT_VERIFIED'} for j in ids[:3]},
      'adopted_outputs':{j:str((old if i<7 else prior)/j) for i,j in enumerate(ids[3:59])},
      'adopted':{j:{'pid':i+1,'start_ticks':1} for i,j in enumerate(ids[3:59])},'new_jobs':ids[59:]})
    def artifact(path,job):
        path.mkdir();model=path/'policy.joblib';model.write_bytes(b'synthetic model')
        q.atomic(path/'result.json',{'status':'FC_COMPLETE_FEASIBLE','job_id':job,'manifest_sha256':'seal',
          'reload_exact':True,'formal_benchmark_admission':False,'S_T_evaluated':False,
          'completion':{'status':'COMPLETE_FEASIBLE'},'model_sha256':q.sha(model)})
    for i,job in enumerate(ids[3:59]):artifact((old if i<7 else prior)/job,job)
    dispatched=[]
    monkeypatch.setattr(q,'process_alive',lambda info:info['pid']!=999 and len(dispatched)<8)
    monkeypatch.setattr(q,'process_identity',lambda p:{'pid':p,'start_ticks':1,'state':'R'})
    monkeypatch.setattr(q.time,'sleep',lambda t:None)
    monkeypatch.setattr(q.shutil,'disk_usage',lambda p:type('Usage',(),{'free':20*1024**3})())
    class Process:
        def __init__(self,args,**kw):
            job=args[args.index('--job-id')+1];dispatched.append(job);self.pid=100+len(dispatched)
            artifact(Path(args[args.index('--output')+1]),job)
        def poll(self):return 0
    monkeypatch.setattr(q.subprocess,'Popen',Process)
    q.run(root,control)
    assert len(dispatched)==len(set(dispatched))==21
    assert set(dispatched)==set(ids[59:])
    receipt=json.loads((control/'queue_receipt.json').read_text())
    assert receipt['jobs']==80 and not receipt['formal_benchmark_admission']
    assert all(v['status']=='FC_ARTIFACT_READY_FOR_REVIEW' for j,v in receipt['completed'].items() if j not in ids[:3])
    assert all(receipt['completed'][j]['exit_code'] is None for j in ids[3:59])
