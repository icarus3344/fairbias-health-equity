"""Resource-only handoff: adopt live F/C fits, dispatch remaining registered jobs.

This controller lives OUTSIDE the frozen99-file runtime. It never edits source,
moves live fits, fabricates an inherited exit status, or evaluates S/T.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_suffix('.part')
    with temp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def process_identity(pid):
    p=Path('/proc')/str(pid)
    try:
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'start_ticks':int(fields[19]),'state':fields[0],
                'argv':[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]}
    except (FileNotFoundError,ProcessLookupError):return None


def process_alive(identity):
    now=process_identity(identity['pid'])
    return bool(now and now['start_ticks']==identity['start_ticks'] and now['state']!='Z')


def check_artifact(path,job_id,manifest_seal,returncode,*,adopted):
    result_path=path/'result.json'
    if not result_path.exists():return {'status':'INCOMPLETE_NO_RECEIPT','exit_code':returncode}
    value=json.loads(result_path.read_text())
    valid=(value.get('status')=='FC_COMPLETE_FEASIBLE' and value.get('job_id')==job_id
           and value.get('manifest_sha256')==manifest_seal and value.get('reload_exact') is True
           and value.get('formal_benchmark_admission') is False and value.get('S_T_evaluated') is False
           and value.get('completion',{}).get('status')=='COMPLETE_FEASIBLE'
           and ((adopted and returncode is None) or returncode==0))
    model=path/'policy.joblib'
    valid=bool(valid and model.is_file() and sha(model)==value.get('model_sha256'))
    return {'status':'FC_ARTIFACT_READY_FOR_REVIEW' if valid else 'REQUIRES_REVIEW',
            'exit_code':returncode,'exit_code_observed':returncode is not None,
            'adopted_process':adopted,'result_sha256':sha(result_path),
            'reported_status':value.get('status'),'elapsed_seconds':value.get('elapsed_seconds')}


def owned_partition(job_ids,adopted_ids,completed_ids):
    jobs,active,done=set(job_ids),set(adopted_ids),set(completed_ids)
    if active&done or not active|done <= jobs:raise ValueError('Invalid or duplicate handoff ownership')
    return sorted(jobs-active-done)


def run(root,control):
    root,control=Path(root).resolve(),Path(control).resolve()
    sys.path.insert(0,str(root/'scripts'))
    from run_nhis_fairbias_completion import load_manifest
    manifest=load_manifest(root/'control/manifest.json')
    jobs={j['job_id']:j for j in manifest['jobs']}
    handoff_path=control/'handoff.json';handoff=json.loads(handoff_path.read_text())
    if (handoff['manifest_sha256']!=manifest['manifest_sha256']
            or handoff['new_controller_sha256']!=sha(__file__)
            or handoff['old_run']!=str(root/'runs/full80_v2')):
        raise ValueError('Handoff identity mismatch')
    if process_alive(handoff['old_controller']):raise ValueError('Old scheduler must be stopped before takeover')
    q=root/'runs/full80_v2'
    lock=(q/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    launch=control/'launch.json'
    with launch.open('x') as f:
        json.dump({'controller_pid':os.getpid(),'source_sha256':sha(__file__),
                   'handoff_sha256':sha(handoff_path),'started_unix':time.time()},f)
    completed=dict(handoff['completed'])
    active={j:{'identity':i,'adopted':True} for j,i in handoff['adopted'].items()}
    pending=owned_partition(jobs,active,completed)
    expected_new=set(handoff['new_jobs'])
    if set(pending)!=expected_new:raise ValueError('New-job partition differs from handoff')
    for arm in {j['config']['arm_id'] for j in jobs.values()}:
        expected=next(j['prepared']['sha256'] for j in jobs.values() if j['config']['arm_id']==arm)
        if sha(root/'prepared'/(arm+'.joblib'))!=expected:raise ValueError('Input hash mismatch')
    env=dict(os.environ,PYTHONPATH=str(root/'src'),CUDA_VISIBLE_DEVICES='')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[key]='1'
    new_outputs=control/'jobs';new_outputs.mkdir()
    def save():
        atomic(control/'state.json',{'controller_pid':os.getpid(),'updated_unix':time.time(),
            'manifest_sha256':manifest['manifest_sha256'],'pending':pending,'completed':completed,
            'active':{j:{'pid':v['identity']['pid'],'start_ticks':v['identity']['start_ticks'],
                         'adopted':v['adopted'],'output':str(q/j if v['adopted'] else new_outputs/j)}
                      for j,v in active.items()},'requested_concurrency':maximum,'registered_jobs':80})
    while pending or active:
        for job,info in list(active.items()):
            if info['adopted']:
                if process_alive(info['identity']):continue
                code=None;output=q/job
            else:
                code=info['process'].poll()
                if code is None:continue
                info['log'].close();output=new_outputs/job
            completed[job]=check_artifact(output,job,manifest['manifest_sha256'],code,adopted=info['adopted'])
            del active[job]
        maximum=json.loads((control/'resources.json').read_text())['concurrency']
        if type(maximum) is not int or not 1<=maximum<=56:raise ValueError('Invalid concurrency')
        while pending and len(active)<maximum:
            mem=Path('/sys/fs/cgroup/memory.max');cur=Path('/sys/fs/cgroup/memory.current')
            if mem.exists() and mem.read_text().strip()!='max' and int(cur.read_text())>=int(mem.read_text())*.8:break
            if shutil.disk_usage(root).free<5*1024**3:break
            job=pending.pop(0);target=new_outputs/job
            if target.exists():raise ValueError('Refusing duplicate output')
            log=(new_outputs/(job+'.log')).open('xb')
            command=[sys.executable,'-B',str(root/'scripts/run_nhis_fairbias_completion.py'),
                'pilot','--manifest',str(root/'control/manifest.json'),'--job-id',job,
                '--prepared',str(root/'prepared'/(jobs[job]['config']['arm_id']+'.joblib')),
                '--output',str(target),'--cache-dir',str(root/'mds_cache')]
            process=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity=process_identity(process.pid)
            if identity is None:identity={'pid':process.pid,'start_ticks':-1}
            active[job]={'identity':identity,'adopted':False,'process':process,'log':log}
            save()  # persist every claimed PID, not only a whole batch
        save()
        if pending and not active:raise RuntimeError('No resource headroom to dispatch remaining jobs')
        if pending or active:time.sleep(5)
    load_manifest(root/'control/manifest.json')
    atomic(control/'queue_receipt.json',{'status':'FINISHED_REQUIRES_SUPERVISOR_REVIEW',
        'manifest_sha256':manifest['manifest_sha256'],'completed':completed,
        'jobs':len(completed),'formal_benchmark_admission':False,'S_T_evaluated':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True);a=p.parse_args();run(a.root,a.control)
