"""Execute disjoint residual FRAPPE shards alongside an unchanged LFR tail."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_nhis_cpu_capacity import ShardedScheduler as CapacityScheduler
from nhis_fairbias.benchmark.parallel_execution import (
    ResourceLimits, SchedulerError, _effective_cpu_count, file_sha,
    write_json_once, verify_completed_job,
)


def validate_execution_partition(plan, universe):
    """The old CPU300 plus the two new subsets must equal all480 exactly."""
    execution = plan.get('execution_jobs', {})
    if set(execution) != {'source', 'cpu'}:
        raise SchedulerError('missing residual execution partition')
    retained = plan.get('retained_cpu_jobs', [])
    groups = [retained, execution['source'], execution['cpu']]
    if any(not isinstance(g, list) or len(g) != len(set(g)) for g in groups):
        raise SchedulerError('invalid or duplicate residual jobs')
    groups = [set(g) for g in groups]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise SchedulerError('residual jobs overlap')
    expected = {n for n, (c, _) in universe.items() if c['method'] == 'FRAPPE_EO'}
    if set.union(*groups) != expected:
        raise SchedulerError('residual jobs do not cover registered FRAPPE exactly')
    for role in ('source', 'cpu'):
        if not set(execution[role]) <= set(plan['assignments'][role]):
            raise SchedulerError('execution outside declared owner')
    if not groups[0] <= set(plan['assignments']['cpu']):
        raise SchedulerError('retained CPU ownership changed')
    # Keep all seeds of a candidate on the same host and execution phase.
    candidate_owner = {}
    for index, group in enumerate(groups):
        for name in group:
            candidate = universe[name][0]['candidate_id']
            if candidate in candidate_owner and candidate_owner[candidate] != index:
                raise SchedulerError('candidate seeds split between execution phases')
            candidate_owner[candidate] = index


def external_worker_count(run):
    if run is None:
        return 0
    count = 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z':
                continue
            argv = (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if 'worker' not in argv or '--job' not in argv:
                continue
            job = Path(argv[argv.index('--job') + 1])
            if job.name == 'job.json' and job.parent.parent == run / 'jobs':
                count += 1
        except (FileNotFoundError, ProcessLookupError):
            continue
    return count


class BalancedScheduler(CapacityScheduler):
    def __init__(self, *args, external_run=None, **kwargs):
        self.external_run = Path(external_run).resolve() if external_run else None
        super().__init__(*args, **kwargs)
        validate_execution_partition(self.plan, self._universe()[0])
        self._owned_names = set(self.plan['execution_jobs'][self.role])

    def _launchable_index(self, pending, slots):
        other = external_worker_count(self.external_run)
        if len(self._running) + other >= self.limits.workers:
            return None
        return super()._launchable_index(pending, slots)

    def _write_shard_audit(self):
        super()._write_shard_audit()
        target = self.run_path / 'balanced_dispatch_activation.json'
        payload = {'plan_sha256':file_sha(self.plan_path), 'role':self.role,
            'execution_jobs': sorted(self._owned_names), 'pid':os.getpid(),
            'wrapper_sha256':file_sha(Path(__file__)),
            'capacity_wrapper_sha256':file_sha(ROOT/'scripts/run_nhis_cpu_capacity.py'),
            'policy_sha256':file_sha(self.policy_path),
            'external_run':str(self.external_run) if self.external_run else None,
            'external_work_counts_toward_cpu_limit':True}
        if target.exists():
            old = json.loads(target.read_text())
            if any(old.get(k) != payload[k] for k in ('plan_sha256','role','execution_jobs','wrapper_sha256','capacity_wrapper_sha256','policy_sha256')):
                raise SchedulerError('balanced dispatch audit conflict')
        else:
            write_json_once(target, payload)


def await_predecessor(scheduler, run):
    """Server-side wait, then verify all retained CPU jobs before extra work."""
    run = Path(run).resolve()
    if file_sha(run/'registration.json') != file_sha(scheduler.registration_path):
        raise SchedulerError('predecessor registration mismatch')
    deadline = time.monotonic() + 6*3600
    while True:
        live = json.loads((run/'parallel_scheduler_live_status.json').read_text())
        if live['status'] == 'COMPLETE' and external_worker_count(run) == 0:
            break
        if live['status'] in {'INTERRUPTED','FAILED'}:
            raise SchedulerError('predecessor requires recovery')
        if time.monotonic() >= deadline:
            raise SchedulerError('predecessor wait exceeded six hours')
        time.sleep(10)
    universe = scheduler._universe()[0]
    for name in scheduler.plan['retained_cpu_jobs']:
        config, seed = universe[name]
        result = verify_completed_job(run/'jobs'/name, config, seed, scheduler.registration)
        if result is None:
            raise SchedulerError('predecessor job missing: '+name)
        job = json.loads((run/'jobs'/name/'job.json').read_text())
        if job.get('runtime_source_identity') != scheduler.plan['runtime_source_identity']:
            raise SchedulerError('predecessor runtime identity mismatch')
    write_json_once(scheduler.run_path/'predecessor_verified.json', {
        'run':str(run), 'jobs_verified':len(scheduler.plan['retained_cpu_jobs']),
        'timestamp':time.time(), 'source_identity':scheduler.plan['runtime_source_identity']})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','plan','activation','policy'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--role',choices=['source','cpu'],required=True)
    p.add_argument('--workers',type=int,required=True)
    p.add_argument('--external-run',type=Path)
    p.add_argument('--predecessor-run',type=Path)
    args=p.parse_args()
    policy=json.loads(args.policy.read_text())
    if args.workers > _effective_cpu_count():
        raise SchedulerError('worker limit exceeds host quota')
    total=int(policy['max_total_rss_bytes'])
    maximum=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    if maximum!='max' and total>int(maximum)-6*1024**3:
        raise SchedulerError('insufficient host memory reserve')
    limits=ResourceLimits(workers=args.workers,max_workers=args.workers,total_rss_bytes=total)
    scheduler=BalancedScheduler(args.run,runner=ROOT/'scripts/run_nhis_runtime_worker.py',
        source_script=ROOT/'scripts/run_nhis_benchmark_parallel.py',limits=limits,
        policy_path=args.policy,plan_path=args.plan,activation_path=args.activation,
        role=args.role,methods=['FRAPPE_EO'],external_run=args.external_run)
    scheduler._validate_parallel_policy()
    if args.predecessor_run:
        if args.role!='cpu':raise SchedulerError('predecessor is CPU-only')
        await_predecessor(scheduler,args.predecessor_run)
    print(json.dumps(scheduler.run()),flush=True)


if __name__=='__main__':main()
