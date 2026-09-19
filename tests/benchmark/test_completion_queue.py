"""F/C queue gates: no success filling, no duplicate pilot/family fitting."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from run_nhis_completion_pilot_queue import atomic, limit, stage_transition, verify_result, sha


def test_all_pilots_must_pass_before_releasing_family():
    ids = [f'job{i}' for i in range(10)]
    done = {j: {'status':'FC_PILOT_VERIFIED'} for j in ids[:-1]}
    assert stage_transition(done, ids) == 'PILOTS_RUNNING'
    done[ids[-1]] = {'status':'REQUIRES_REVIEW'}
    assert stage_transition(done, ids) == 'PILOT_GATE_FAILED'
    done[ids[-1]]['status'] = 'FC_PILOT_VERIFIED'
    assert stage_transition(done, ids) == 'FAMILY'


@pytest.mark.parametrize('cap',[0,57,True,2.5])
def test_invalid_resource_limit_rejected(tmp_path,cap):
    p=tmp_path/'resources.json';atomic(p,{'concurrency':cap})
    with pytest.raises(ValueError):limit(p)


def test_missing_or_wrong_model_never_counts_as_verified(tmp_path):
    assert verify_result(tmp_path,'a','m',-9)['status']=='INCOMPLETE_NO_RECEIPT'
    model=tmp_path/'policy.joblib';model.write_bytes(b'synthetic-only')
    result={'status':'FC_COMPLETE_FEASIBLE','job_id':'a','manifest_sha256':'m',
            'reload_exact':True,'formal_benchmark_admission':False,'S_T_evaluated':False,
            'model_sha256':sha(model)}
    atomic(tmp_path/'result.json',result)
    assert verify_result(tmp_path,'a','m',0)['status']=='FC_PILOT_VERIFIED'
    assert verify_result(tmp_path,'a','other',0)['status']=='REQUIRES_REVIEW'
    assert verify_result(tmp_path,'a','m',1)['status']=='REQUIRES_REVIEW'
    model.write_bytes(b'changed')
    assert verify_result(tmp_path,'a','m',0)['status']=='REQUIRES_REVIEW'


@pytest.mark.parametrize('pilot_failure',[False, True])
def test_controller_runs80_once_or_blocks70_after_failed_pilot(tmp_path,monkeypatch,pilot_failure):
    import run_nhis_completion_pilot_queue as q
    root=tmp_path/'runtime';(root/'control').mkdir(parents=True);(root/'prepared').mkdir()
    prepared=root/'prepared'/'arm_001.joblib';prepared.write_bytes(b'synthetic input')
    ids=[f'job{i:02d}' for i in range(80)]
    jobs=[{'job_id':j,'config':{'arm_id':'arm_001'},'prepared':{'sha256':sha(prepared)}} for j in ids]
    manifest={'manifest_sha256':'test-seal','jobs':jobs}
    atomic(root/'control'/'pilot_plan.json',{'manifest_sha256':'test-seal','pilots':[{'job_id':j} for j in ids[:10]]})
    monkeypatch.setattr(q,'load_manifest',lambda p:manifest)
    monkeypatch.setattr(q,'memory_allows_dispatch',lambda:True)
    monkeypatch.setattr(q.shutil,'disk_usage',lambda p:type('Usage',(),{'free':20*1024**3})())
    monkeypatch.setattr(q.time,'sleep',lambda t:None)
    dispatched=[]
    class Process:
        def __init__(self,args,**kwargs):
            job=args[args.index('--job-id')+1];out=Path(args[args.index('--output')+1])
            if job not in ids[:10]:assert set(ids[:10])<=set(dispatched)
            dispatched.append(job);self.pid=100+len(dispatched);self.code=1 if pilot_failure and job==ids[0] else 0
            out.mkdir();model=out/'policy.joblib';model.write_bytes(b'synthetic fixture model')
            atomic(out/'result.json',{'status':'FC_COMPLETE_FEASIBLE','job_id':job,'manifest_sha256':'test-seal',
                  'reload_exact':True,'formal_benchmark_admission':False,'S_T_evaluated':False,'model_sha256':sha(model)})
        def poll(self):return self.code
    monkeypatch.setattr(q.subprocess,'Popen',Process)
    out=tmp_path/'run'
    q.run(root,out,continue_family=True)
    receipt=json.loads((out/'queue_receipt.json').read_text())
    assert len(dispatched)==len(set(dispatched))==(10 if pilot_failure else 80)
    assert receipt['status']==('PILOT_GATE_FAILED' if pilot_failure else 'ALL_FC_FITS_VERIFIED')
    if pilot_failure:assert set(receipt['blocked_jobs'])==set(ids[10:])
