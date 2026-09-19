"""Exercise the actual recovery worker with generated F/C/S data only."""
from pathlib import Path
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

def main():
    import joblib
    import numpy as np
    from nhis_fairbias.benchmark.data_contracts import generate_synthetic_nhis_cohort, load_arm_partitions, ARM_SPECS
    from nhis_fairbias.benchmark.preprocessing import BenchmarkPreprocessor
    from nhis_fairbias.benchmark.experiment_registry import identity
    output = ROOT/'artifacts/nhis/cpu_sharding_20260917/production_worker_synthetic'
    output.mkdir(parents=True, exist_ok=False)
    parts = load_arm_partitions(generate_synthetic_nhis_cohort(n_records_per_year=400, n_strata=10, seed=5), 'arm_001')
    parts.pop('evaluation_T')
    C = parts['calibration_C']
    parts['calibration_C'] = dataclasses.replace(C, y=(np.arange(len(C))//2)%2, A=np.arange(len(C))%2)
    F,C,S = [parts[x] for x in ('fitting_F','calibration_C','selection_S')]
    prep = BenchmarkPreprocessor(ARM_SPECS['arm_001']['features']).fit(F.X_semantic)
    data = {'partitions':parts,'preprocessor':prep,'data_identity':'generated_recovery_worker_20260917',
            'X_F':prep.transform(F.X_semantic),'X_C':prep.transform(C.X_semantic),'X_S':prep.transform(S.X_semantic)}
    data_path = output/'generated.joblib'; joblib.dump(data,data_path)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    files = {p:sha(ROOT/p) for p in ('scripts/run_nhis_runtime_worker.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_frappe_pipeline.py',
        'src/nhis_fairbias/benchmark/adapters/adapter_frappe.py')}
    variant = 'frappe_pipeline_deterministic_v1_20260917'; environment={'TF_DETERMINISTIC_OPS':'1'}
    runtime_id=identity({'variant':variant,'files':files,'environment':environment})
    env=dict(os.environ);env.update(environment);env['PYTHONHASHSEED']='0';env['CUDA_VISIBLE_DEVICES']=''
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[k]='1'
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+str(ROOT/'.venv311/lib/python3.11/site-packages')
    env.pop('TF_ENABLE_ONEDNN_OPTS',None)
    python=ROOT/'artifacts/nhis/benchmark_dependencies_20260916/frappe_env/.venv/bin/python'
    records=[]
    for backbone in ('LR','GBDT'):
        params={'C':1.,'hidden_units':[2],'epochs':2,'batch_size':4}
        if backbone=='GBDT':params['estimator_params']={'n_estimators':10,'max_depth':2}
        config={'arm_id':'arm_001','backbone':backbone,'method':'FRAPPE_EO','params':params,
                'status':'REGISTERED','seeds':[0],'training_weighted':False}
        config['candidate_id']=identity(config)[:20]
        arrays=[]
        for repeat in range(2):
            folder=output/f'{backbone}_{repeat}';folder.mkdir()
            job={'config':config,'seed':0,'data_path':str(data_path),'data_sha256':sha(data_path),
                 'data_identity':data['data_identity'],'source_identity':'synthetic_only',
                 'cache_path':str(output/'cache'),'runtime_variant':variant,'runtime_environment':environment,
                 'runtime_source_files':files,'runtime_source_identity':runtime_id}
            (folder/'job.json').write_text(json.dumps(job,indent=2))
            start=time.monotonic()
            with (folder/'worker.log').open('x') as log:
                process=subprocess.run([str(python),'-B',str(ROOT/'scripts/run_nhis_runtime_worker.py'),
                    'worker','--job',str(folder/'job.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            result=json.loads((folder/'result.json').read_text())
            assert process.returncode==0 and result['status']=='VALID', (backbone,result.get('error'))
            assert result['reload_verified']
            assert result['algorithm_manifest']['runtime_variant']==variant
            assert result['algorithm_manifest']['pipeline_options_applied']
            with np.load(folder/'predictions_S.npz') as x:arrays.append({k:x[k].copy() for k in x.files})
            records.append({'backbone':backbone,'repeat':repeat,'status':result['status'],
                            'elapsed_seconds':time.monotonic()-start,'reload_verified':True})
        assert set(arrays[0])==set(arrays[1])
        for key in arrays[0]:assert np.array_equal(arrays[0][key],arrays[1][key]), (backbone,key)
    summary={'scope':'generated data only; no NHIS records','runtime_source_identity':runtime_id,
             'runtime_source_files':files,'results':records,'fresh_process_predictions_exact':True,
             'sample_sizes':{'F':len(F),'C':len(C),'S':len(S)}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
