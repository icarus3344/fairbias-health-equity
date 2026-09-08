"""Independent synthetic review: assertions describe required behavior, not current bugs."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
BLOCKED = []
def data_guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        p = Path(args[0]).resolve()
        if p.is_relative_to(ROOT / 'data') or p in [ROOT/'data_COMPAS.csv', ROOT/'data_Credit_Card.csv'] or p.suffix == '.parquet':
            BLOCKED.append(p.name)
            raise RuntimeError('CODEX_REVIEW_DATA_ACCESS_BLOCKED')
sys.addaudithook(data_guard)

import contextlib
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from fairbias.config import FairBiasConfig
from fairbias.evaluator import FairEvaluator
from fairbias.transform import FairTransform
from fairbias.enhancement import FairAccuracyEnhancement
from fairbias.enhancement_contracts import EvaluationPartition, CandidateEvaluationResult, evaluate_candidate_utility
import scripts.run_nhis_enhancement_study as cli

class SyntheticReview(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(210)
        self.X = pd.DataFrame({'x': rng.uniform(.2, 2, 160), 'z': rng.uniform(.2, 2, 160)})
        self.Y = pd.Series(((self.X.x > 1.1) ^ (self.X.z > 1.1)).astype(int), name='y')
        self.V = pd.DataFrame({'x': rng.uniform(.2, 2, 80), 'z': rng.uniform(.2, 2, 80)}, index=range(1000,1080))
        self.W = pd.Series(((self.V.x > 1.1) ^ (self.V.z > 1.1)).astype(int), name='y')
        self.O = pd.DataFrame({'o': np.arange(len(self.X)) % 2}, index=self.X.index)
        self.cfg = FairBiasConfig(algorithm_mode='engineering_bounded', classifier='LR', random_seed=42).resolved()
        self.ev = FairEvaluator(self.cfg, ['o'], 'y', [], ['x','z'])
        self.tr = FairTransform()
        self.part = EvaluationPartition(self.X,self.Y,self.V,self.W,protected_fit=self.O)
        self.ae = FairAccuracyEnhancement(self.ev,self.tr,'y',[],['x','z'],poly_exponents=(1/3,3))

    def test_missing_protected_data_cannot_disable_enabled_guard(self):
        r = self.ae._is_fairness_acceptable(self.X,None,.1,.1)
        self.assertFalse(r.is_acceptable)

    def test_partial_geometry_is_invalid(self):
        with patch.object(self.ev,'calculate_epsilon',return_value={'o':{'x':.001}}):
            self.assertFalse(self.ae._is_fairness_acceptable(self.X,self.O,.1,.1).is_acceptable)

    def test_nan_after_finite_entry_is_invalid(self):
        with patch.object(self.ev,'calculate_epsilon',return_value={'o':{'x':.001,'z':float('nan')}}):
            self.assertFalse(self.ae._is_fairness_acceptable(self.X,self.O,.1,.1).is_acceptable)

    def test_infinite_slack_is_invalid(self):
        self.ae.max_fairness_degradation = float('inf')
        with patch.object(self.ev,'calculate_epsilon',return_value={'o':{'x':100.,'z':100.}}):
            self.assertFalse(self.ae._is_fairness_acceptable(self.X,self.O,.1,.1).is_acceptable)

    def test_invalid_baseline_must_reach_caller_as_failure(self):
        bad = self.W.copy(); bad[:] = 1
        with self.assertRaisesRegex((RuntimeError,ValueError),'SINGLE_CLASS|single|invalid'):
            self.ae.enhance_step(self.X,self.Y,{},X_val=self.V,Y_val=bad)

    def test_configured_classifier_matches_utility_oracle(self):
        cfg = FairBiasConfig(algorithm_mode='engineering_bounded',classifier='DT',random_seed=42).resolved()
        ev = FairEvaluator(cfg,['o'],'y',[],['x','z'])
        _,score = ev.fit_and_predict(self.X,self.Y,self.V)
        wanted = roc_auc_score(self.W,score)
        result = evaluate_candidate_utility(self.part,{},['x','z'],[],self.tr,ev)
        self.assertAlmostEqual(result.utility_score,wanted,places=10)

    def test_missing_probabilities_not_replaced_with_hard_predictions(self):
        class HardOnly:
            def fit(self,X,y): return self
            def predict(self,X): return np.arange(len(X))%2
        result = evaluate_candidate_utility(self.part,{},['x','z'],[],self.tr,self.ev,model_factory=HardOnly)
        self.assertFalse(result.is_valid)

    def test_protected_index_alignment_is_validated(self):
        wrong = self.O.iloc[::-1]
        with self.assertRaises(ValueError):
            EvaluationPartition(self.X,self.Y,self.V,self.W,protected_fit=wrong)

    def test_partition_fingerprint_binds_values_and_labels(self):
        changed_y = 1-self.Y
        changed_x = self.X.copy(); changed_x.iloc[30,0] += .5
        other = EvaluationPartition(changed_x,changed_y,self.V,self.W)
        self.assertNotEqual(self.part.fit_fingerprint(),other.fit_fingerprint())

    def test_feature_ranking_uses_only_fit_partition(self):
        fitX=self.X.iloc[:120]; fitY=self.Y.iloc[:120]
        selX=self.X.iloc[120:]; selY=self.Y.iloc[120:]
        part=EvaluationPartition(fitX,fitY,selX,selY)
        seen=[]
        def ranking(X,y):
            seen.append(len(y)); return {'x':1.,'z':.5}
        ae=FairAccuracyEnhancement(self.ev,self.tr,'y',[],['x','z'],poly_exponents=())
        with patch('fairbias.enhancement.calculate_nmi_dict',side_effect=ranking):
            ae.enhance_step(self.X,self.Y,{},partition=part)
        self.assertEqual(seen,[120])

    def test_only_committed_candidate_logged_accepted(self):
        results=[CandidateEvaluationResult('VALID',v,model_fit_count=1) for v in [.5,.6,.7]]
        with patch('fairbias.enhancement.evaluate_candidate_utility',side_effect=results):
            _,changed,attr=self.ae.enhance_step(self.X,self.Y,{},partition=self.part)
        actual=[ev for ev in self.ae.audit_trail if ev.accepted]
        self.assertEqual(len(actual),1,msg=f'committed={changed}; accepted_events={len(actual)}')

    def test_existing_directory_rejected_by_real_cli(self):
        class Dummy:
            def __init__(self,**kwargs): pass
            def run_arm(self,arm): raise RuntimeError('SYNTHETIC_INJECTED_STOP')
        with tempfile.TemporaryDirectory(dir=OUT) as td:
            d=Path(td)/'existing'; d.mkdir()
            (d/'execution_manifest.json').write_text('{"status":"failed"}')
            with patch.object(sys,'argv',['run_d8','--arm','D6_ARM_001','--output-dir',str(d)]),patch.object(cli,'D8EnhancementRunner',Dummy):
                with self.assertRaises(FileExistsError): cli.main()

    def test_real_cli_records_midrun_failure_manifest(self):
        class Dummy:
            def __init__(self,**kwargs): pass
            def run_arm(self,arm): raise RuntimeError('SYNTHETIC_INJECTED_STOP')
        with tempfile.TemporaryDirectory(dir=OUT) as td:
            d=Path(td)/'new'
            with patch.object(sys,'argv',['run_d8','--arm','D6_ARM_001','--output-dir',str(d)]),patch.object(cli,'D8EnhancementRunner',Dummy):
                with self.assertRaisesRegex(RuntimeError,'SYNTHETIC_INJECTED_STOP'): cli.main()
            self.assertTrue((d/'execution_manifest.json').exists())

    def test_original_drop_bug_fixed_with_arithmetic_oracle(self):
        state={'z':'dropped','x':{'power':1/3}}
        result=evaluate_candidate_utility(self.part,state,['x','z'],[],self.tr,self.ev)
        fit=self.X[['x']].pow(1/3); sel=self.V[['x']].pow(1/3)
        scaler=MinMaxScaler(); a=scaler.fit_transform(fit); b=scaler.transform(sel)
        model=LogisticRegression(max_iter=1000,solver='lbfgs',random_state=42).fit(a,self.Y)
        self.assertTrue(result.is_valid)
        self.assertAlmostEqual(result.utility_score,roc_auc_score(self.W,model.predict_proba(b)[:,1]),places=12)

if __name__=='__main__':
    with (OUT/'adversarial_review.log').open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticReview)
        res=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    result={'tests_run':res.testsRun,'passed':res.testsRun-len(res.failures)-len(res.errors),
            'failures':[{'test':str(t),'traceback':e} for t,e in res.failures],
            'errors':[{'test':str(t),'traceback':e} for t,e in res.errors],
            'blocked_read_attempts':BLOCKED}
    (OUT/'adversarial_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'tests_run':res.testsRun,'passed':result['passed'],'failures':len(res.failures),'errors':len(res.errors),'blocked_read_attempts':BLOCKED}))
