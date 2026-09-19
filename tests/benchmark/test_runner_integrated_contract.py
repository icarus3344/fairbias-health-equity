import numpy as np
import pytest
from nhis_fairbias.benchmark.data_contracts import generate_synthetic_nhis_cohort
from nhis_fairbias.benchmark.runner import BenchmarkRunner
from nhis_fairbias.benchmark import runner as runner_module
from nhis_fairbias.benchmark.adapters.adapter_unmitigated import UnmitigatedAdapter


def test_actual_lr_policy_and_full_design_reach_survey_evaluator(monkeypatch):
    data = generate_synthetic_nhis_cohort(n_records_per_year=80, n_strata=10, seed=8)
    for feature in runner_module.ARM_SPECS['arm_001']['features']:
        data[feature] = 1.
    for year in (2022, 2023, 2024):
        m = data.year == year
        data.loc[m, 'PSTRAT'] = np.repeat(np.arange(101, 111), 8)
        data.loc[m, 'PPSU'] = np.tile(np.repeat([1, 2], 4), 10)
        data.loc[m, 'SEX_A'] = np.tile([1, 1, 2, 2], 20)
        data.loc[m, 'MEDDL12M_A'] = np.tile([0, 1, 0, 1], 20)
        data.loc[m, 'agep_a'] = np.tile([20., 80., 20., 80.], 20)
    # One entire test PSU has no domain contribution, but stays in design.
    outside = (data.year == 2024) & (data.PSTRAT == 101) & (data.PPSU == 1)
    data.loc[outside, 'SEX_A'] = 9
    captured = {}
    original = runner_module.evaluate_with_survey_bootstrap
    def witness(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(runner_module, 'evaluate_with_survey_bootstrap', witness)
    run = BenchmarkRunner(bootstrap_B=20, method_factories={'UNMITIGATED': lambda: UnmitigatedAdapter(C=1e4)})
    result = run.run_arm(data, 'arm_001')
    assert captured['seed'] == 20260914
    assert len(captured['weights']) == 80
    assert captured['domain_mask'].sum() == 76
    q = captured['predictions_dict']['UNMITIGATED']
    assert set(q).issubset({0., 1.})
    assert np.all(q[~captured['domain_mask']] == 0)
    assert len(set(zip(captured['strata'], captured['psus']))) == 20
    method = result.method_results['UNMITIGATED']
    assert method['balanced_accuracy'] == pytest.approx(1.)
    assert method['risk_metrics']['auroc'] == pytest.approx(1.)
    assert method['risk_metrics']['brier'] < .01
    assert method['threshold'] is not None
