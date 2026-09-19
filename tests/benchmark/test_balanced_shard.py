import copy
from types import SimpleNamespace
import pytest
from scripts.run_nhis_balanced_shard import validate_execution_partition, BalancedScheduler
from nhis_fairbias.benchmark.parallel_execution import SchedulerError


def fixture():
    universe={n:({'candidate_id':n.split('_')[0],'method':'FRAPPE_EO'},0)
              for n in ['a_s0','a_s1','b_s0','b_s1','c_s0','c_s1']}
    plan={'retained_cpu_jobs':['a_s0','a_s1'],
          'execution_jobs':{'source':['b_s0','b_s1'],'cpu':['c_s0','c_s1']},
          'assignments':{'source':['b_s0','b_s1'],'cpu':['a_s0','a_s1','c_s0','c_s1']}}
    return plan,universe


def test_complete_disjoint_whole_candidates():
    validate_execution_partition(*fixture())


@pytest.mark.parametrize('kind',['overlap','missing','wrong_owner','split_seeds'])
def test_bad_residual_plan_rejected(kind):
    plan,universe=fixture()
    if kind=='overlap':plan['execution_jobs']['cpu'].append('a_s0')
    if kind=='missing':plan['execution_jobs']['source'].pop()
    if kind=='wrong_owner':plan['assignments']['cpu'].remove('c_s0')
    if kind=='split_seeds':
        plan['execution_jobs']['cpu'].append(plan['execution_jobs']['source'].pop())
        plan['assignments']['cpu'].append(plan['assignments']['source'].pop())
    with pytest.raises(SchedulerError):validate_execution_partition(plan,universe)


def test_other_run_consumes_available_slots(monkeypatch):
    import scripts.run_nhis_balanced_shard as mod
    fake=SimpleNamespace(external_run='/synthetic',_running={x:None for x in range(10)},
                         limits=SimpleNamespace(workers=15))
    monkeypatch.setattr(mod,'external_worker_count',lambda p:5)
    assert BalancedScheduler._launchable_index(fake,[object()],5) is None


def test_no_external_load_releases_admission(monkeypatch):
    import scripts.run_nhis_balanced_shard as mod
    scheduler=object.__new__(BalancedScheduler)
    scheduler.external_run=None;scheduler._running={};scheduler.limits=SimpleNamespace(workers=15)
    monkeypatch.setattr(mod,'external_worker_count',lambda p:0)
    monkeypatch.setattr(mod.CapacityScheduler,'_launchable_index',lambda *a:0)
    assert scheduler._launchable_index([object()],15)==0
