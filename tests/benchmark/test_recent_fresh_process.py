"""Synthetic process-boundary audit for the recent neural adapters."""

import json
import os
import subprocess
import sys
import textwrap

import joblib
import numpy as np
import pytest

from nhis_fairbias.benchmark.adapters.adapter_tabm import TabMClassifier
from nhis_fairbias.benchmark.adapters.adapter_frappe import FrappeAdapter


def _tabm_data():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(24, 3)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_tabm_seed_repeat_and_prediction_batch_invariance():
    X, y = _tabm_data()
    a = TabMClassifier(epochs=2, batch_size=8, random_state=11).fit(X, y)
    b = TabMClassifier(epochs=2, batch_size=8, random_state=11).fit(X, y)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X), rtol=0, atol=1e-7)
    whole = a.predict_proba(X)[:, 1]
    chunks = np.concatenate([a.predict_proba(X[:7])[:, 1], a.predict_proba(X[7:])[:, 1]])
    np.testing.assert_allclose(whole, chunks, rtol=0, atol=1e-7)


def test_tabm_joblib_reload_in_fresh_process(tmp_path):
    X, y = _tabm_data()
    path = tmp_path / "tabm.joblib"
    joblib.dump(TabMClassifier(epochs=2, batch_size=8, random_state=13).fit(X, y), path)
    script = textwrap.dedent(
        """
        import json, joblib, numpy as np, sys
        obj = joblib.load(sys.argv[1])
        X = np.asarray(json.loads(sys.argv[2]), dtype=np.float32)
        print(json.dumps(obj.predict_proba(X)[:,1].tolist()))
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(["src", "artifacts/nhis/benchmark_dependencies_20260916/site-packages"])
    run = subprocess.run([sys.executable, "-c", script, str(path), json.dumps(X.tolist())], cwd=os.getcwd(), env=env, capture_output=True, text=True, check=True)
    child = np.asarray(json.loads(run.stdout), dtype=float)
    np.testing.assert_allclose(child, TabMClassifier.__name__ and joblib.load(path).predict_proba(X)[:, 1], rtol=0, atol=1e-7)


def test_frappe_fresh_process_contract_in_pinned_runtime(tmp_path):
    tf_python = os.path.join(os.getcwd(), "artifacts/nhis/benchmark_dependencies_20260916/frappe_env/.venv/bin/python")
    if not os.path.exists(tf_python):
        pytest.skip("pinned FRAPPE interpreter is unavailable")
    data = tmp_path / "data.npz"
    rng = np.random.default_rng(2)
    X = rng.normal(size=(32, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    A = np.array([0, 1] * 16)
    np.savez(data, X=X, y=y, A=A)
    model = tmp_path / "frappe.joblib"
    script = textwrap.dedent(
        """
        import joblib, numpy as np, sys
        from nhis_fairbias.benchmark.adapters.adapter_frappe import FrappeAdapter
        d=np.load(sys.argv[2]); X,y,A=d['X'],d['y'],d['A']
        if sys.argv[1]=='train':
            m=FrappeAdapter(epochs=2,batch_size=4,hidden_units=(4,),random_state=5)
            m.fit(X[:16],y[:16],A[:16],X_calibration=X[16:],y_calibration=y[16:],A_calibration=A[16:])
            p=m.predict_decision_proba(X[16:])
            chunks=np.concatenate([m.predict_decision_proba(X[16:24]),m.predict_decision_proba(X[24:])])
            joblib.dump((m,p,chunks),sys.argv[3])
            print('train_p', ' '.join(map(str,p.tolist())))
        else:
            m,p_before,chunks_before=joblib.load(sys.argv[3]); p=m.predict_decision_proba(X[16:])
            chunks=np.concatenate([m.predict_decision_proba(X[16:24]),m.predict_decision_proba(X[24:])])
            print(' '.join(map(str,p.tolist()))); print('max_reload_delta',float(np.max(np.abs(p-p_before)))); print('max_batch_delta',float(np.max(np.abs(chunks-chunks_before))))
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(["src", ".venv311/lib/python3.11/site-packages"])
    first_train = subprocess.run([tf_python, "-c", script, "train", str(data), str(model)], cwd=os.getcwd(), env=env, check=True, capture_output=True, text=True)
    run = subprocess.run([tf_python, "-c", script, "load", str(data), str(model)], cwd=os.getcwd(), env=env, check=True, capture_output=True, text=True)
    lines = run.stdout.strip().splitlines()
    p = np.fromstring(lines[-3], sep=" ")
    assert p.shape == (16,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert float(lines[-2].split()[-1]) <= 1e-7
    assert float(lines[-1].split()[-1]) <= 1e-7
    repeat = subprocess.run([tf_python, "-c", script, "train", str(data), str(model)], cwd=os.getcwd(), env=env, check=True, capture_output=True, text=True)
    # Same seed and same tiny F/C partitions must reproduce the saved fit.
    first_p = np.fromstring(first_train.stdout.strip().splitlines()[-1].split(" ", 1)[1], sep=" ")
    repeat_p = np.fromstring(repeat.stdout.strip().splitlines()[-1].split(" ", 1)[1], sep=" ")
    np.testing.assert_allclose(repeat_p, first_p, rtol=0, atol=1e-6)
    assert FrappeAdapter(epochs=2).supports_arm2 is False
