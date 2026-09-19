import numpy as np
import pytest
import pickle
import subprocess
import sys
import json

from nhis_fairbias.benchmark.adapters.adapter_fairgbm import FairGBMAdapter, OrdinaryFairGBMClassifier
from nhis_fairbias.benchmark.adapters.base import NotSupportedError


def _tiny():
    X = np.array([[0.0], [1.0], [0.1], [0.9], [0.2], [0.8], [0.3], [0.7]])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    A = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    return X, y, A


def test_contract_declares_native_eo_and_prediction_semantics():
    adapter = FairGBMAdapter(constraint_type="FNR,FPR", n_estimators=2)
    caps = adapter.get_capabilities()
    assert adapter.output_type == "event_probability_p"
    assert caps["constraint_type"] == "FNR,FPR"
    assert caps["requires_A_fit"] is True
    assert caps["requires_A_predict"] is False
    assert adapter.supports_arm2 is True


def test_constraint_validation_and_multigroup_fit_boundary():
    with pytest.raises(ValueError):
        FairGBMAdapter(constraint_type="FPR,FPR")
    with pytest.raises(ValueError):
        FairGBMAdapter(constraint_type="DP")
    X, y, A = _tiny()
    adapter = FairGBMAdapter(n_estimators=2)
    # If the isolated native build is on PYTHONPATH, exercise the real
    # multigroup fit; otherwise failure must be explicit rather than silently
    # becoming ordinary LightGBM.
    try:
        adapter.fit(X, y, A)
    except NotSupportedError:
        return
    assert adapter.model is not None
    assert adapter.predict_event_probability(X).shape == (len(X),)


def test_invalid_inputs_rejected_before_native_import():
    X, y, A = _tiny()
    with pytest.raises(ValueError):
        FairGBMAdapter().fit(X, y.astype(float) + .2, A)
    with pytest.raises(ValueError):
        FairGBMAdapter().fit(X, y, A.astype(float) + .5)
    with pytest.raises(ValueError):
        FairGBMAdapter().fit(X, y, A, sample_weight=np.ones(7))


def test_native_multigroup_pickle_reload_and_matched_objectives(tmp_path):
    X = np.column_stack([np.linspace(0, 1, 28), np.tile([0.0, 1.0], 14)])
    y = np.tile([0, 1], 14)
    A = np.repeat(np.arange(7), 4)
    constrained = FairGBMAdapter(n_estimators=10, num_leaves=7, constraint_type="FPR,FNR")
    ordinary = OrdinaryFairGBMClassifier(n_estimators=10, num_leaves=7)
    try:
        constrained.fit(X, y, A)
        ordinary.fit(X, y, A)
    except NotSupportedError:
        pytest.skip("isolated native FairGBM site is not on PYTHONPATH")
    assert len(constrained.classes_) == 2
    assert constrained.get_params()["constraint_type"] == "FPR,FNR"
    assert ordinary.get_params()["objective"] == "binary"
    assert constrained.model._objective == "constrained_cross_entropy"
    assert ordinary.model._objective == "binary"
    path = tmp_path / "fairgbm_adapter.pkl"
    path.write_bytes(pickle.dumps(constrained))
    code = "import pickle,sys,numpy as np; a=pickle.load(open(sys.argv[1],'rb')); X=np.array([[.15,0.],[.85,1.]]); print(a.predict_event_probability(X).tolist())"
    out = subprocess.check_output([sys.executable, "-c", code, str(path)], text=True)
    assert len(json.loads(out.strip().splitlines()[-1])) == 2
