"""Development-only completion policy for the full BM_AE/Joint search.

This is a NEW adaptive-compute variant, not the registered-budget estimator.
Administrative search ceilings cause a deterministic replay at a larger limit;
completed MDS fits can be reused. Neither a ceiling nor an interruption counts
as convergence. No S/T data or performance is accepted by this interface.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from fairbias import bias_metric
from .adapters.adapter_fairbias_ae import FairBiasAEAdapter
from .adapters.base import NotSupportedError
from .adapters.adapter_joint_ae_recovery import (
    _GeometryFailure, _strict_geometry_failure_transport,
    classify_recovery_fit, verify_recovery_component_sources,
)
from .joint_mds_numpy import numpy_mds_acceleration

VERSION = 'fairbias_adaptive_completion_v1'
_MDS_ACTIVE = False
_LIMITS = {
    'BUDGET_EXHAUSTED: AE commit limit; stopping criterion not yet established': 'max_outer_iterations',
    'BUDGET_EXHAUSTED: BM commit limit': 'max_bm_steps',
    'BUDGET_EXHAUSTED: FairBias AE utility evaluations': 'max_utility_evaluations',
    'BUDGET_EXHAUSTED: FairBias AE geometry evaluations': 'max_geometry_evaluations',
}


def _emit(path, event):
    """Durable aggregate progress; never include inputs, predictions or exception text."""
    if path is not None:
        with Path(path).open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())


@contextmanager
def adaptive_mds_completion(*, progress_path=None):
    """Fresh same-seed 2x-cap refits until the existing selected-fit audit passes.

    Tolerance, dimensions, initialization count/order and input never change.
    This changes the iteration-budget policy and records every incomplete fit.
    It may enclose the adapter audit and must be installed after the completed
    fit cache. It does not silently relax a numerical convergence requirement.
    """
    global _MDS_ACTIVE
    if _MDS_ACTIVE:
        raise RuntimeError('MDS completion context is not reentrant')
    original = bias_metric.MDS
    stats = {'policy': 'fresh_same_seed_geometric_iteration_extension', 'attempts': [], 'extensions': 0}

    class CompletionMDS(original):
        def fit_transform(self, X, y=None, init=None):
            seed = self.random_state
            # These are the actual FairBias calls; unsupported inputs retain
            # the original single-attempt semantics rather than consuming RNG twice.
            if type(seed) is not int or init is not None:
                return super().fit_transform(X, y=y, init=init)
            requested_cap = int(self.max_iter)
            while True:
                tick = time.monotonic()
                points = super().fit_transform(X, y=y, init=init)
                finite = bool(np.isfinite(self.stress_) and np.isfinite(points).all())
                capped = int(self.n_iter_) >= int(self.max_iter)
                row = {'requested_cap': requested_cap, 'actual_cap': int(self.max_iter),
                       'iterations': int(self.n_iter_), 'components': int(self.n_components),
                       'n_init': int(self.n_init), 'seed': seed,
                       'stress': float(self.stress_) if finite else None,
                       'elapsed_seconds': time.monotonic() - tick,
                       'status': 'NONFINITE_MDS' if not finite else ('ITERATION_CAP' if capped else 'COMPLETE_SELECTED_FIT')}
                stats['attempts'].append(row)
                if not finite:
                    _emit(progress_path, {'event': 'MDS_NONFINITE_REQUIRES_REPAIR', **row})
                    raise RuntimeError('NOT_ESTIMABLE: nonfinite MDS output')
                if not capped:
                    return points
                _emit(progress_path, {'event': 'MDS_BUDGET_EXTENSION', **row})
                stats['extensions'] += 1
                # Integer random_state reinitializes the same RNG on each call.
                # A larger max_iter can change the chosen initialization, so
                # retain n_init and refit all of them instead of a warm start.
                self.set_params(max_iter=2 * int(self.max_iter))

    try:
        _MDS_ACTIVE = True
        bias_metric.MDS = CompletionMDS
        yield stats
    finally:
        bias_metric.MDS = original
        _MDS_ACTIVE = False


class FairBiasCompletionAdapter:
    """Explicit F/C-only adaptive search, with old methods/sources unmodified."""
    requires_sensitive_at_predict = False
    output_type = 'event_probability_p'

    def __init__(self, *, mode, cache_dir, progress_path=None, namespace='fairbias_completion_v1',
                 max_replays_per_session=None, **adapter_params):
        if mode not in {'BM_AE', 'JOINT'}:
            raise ValueError('Completion requires BM_AE or JOINT')
        if max_replays_per_session is not None and (type(max_replays_per_session) is not int or max_replays_per_session < 1):
            raise ValueError('Session replay allowance must be positive or None')
        self.mode, self.cache_dir, self.progress_path = mode, str(cache_dir), progress_path
        self.namespace = namespace
        self.max_replays_per_session = max_replays_per_session
        self.adapter_params = copy.deepcopy(adapter_params)
        self.name = 'FAIRBIAS_' + mode + '_ADAPTIVE_COMPLETION_V1'
        self.is_fitted_ = False

    def fit(self, X, y, A, sample_weight=None):
        raise NotSupportedError('Completion requires disjoint F/C fit_development')

    def fit_development(self, XF, yF, AF, XC, yC, AC, metadata=None):
        from .mds_completed_cache import completed_mds_cache
        self.is_fitted_ = False
        if self.progress_path is not None:
            Path(self.progress_path).parent.mkdir(parents=True, exist_ok=True)
        params = {'max_outer_iterations': 40, 'max_bm_steps': 50,
                  'max_geometry_evaluations': 20_000, 'max_utility_evaluations': 500,
                  **copy.deepcopy(self.adapter_params)}
        # Seed must be explicit and immutable across all replays.
        if type(params.get('random_state')) is not int:
            raise ValueError('Completion requires an explicit integer random_state')
        sources = verify_recovery_component_sources()
        self.attempts_ = []
        self.provenance_ = {'version': VERSION, 'mode': self.mode, 'component_sources': sources,
                            'status': 'RUNNING', 'attempts': self.attempts_,
                            'search_order_changed': False, 'adaptive_compute_policy': True,
                            'evaluation_authorized': False}
        started = time.monotonic()
        kernel = cache = mds = None
        try:
            with threadpool_limits(limits=1), numpy_mds_acceleration() as kernel:
                with completed_mds_cache(self.cache_dir, self.namespace) as cache:
                    with adaptive_mds_completion(progress_path=self.progress_path) as mds:
                        with _strict_geometry_failure_transport():
                            while True:
                                self._delegate = FairBiasAEAdapter(mode=self.mode, **params)
                                attempt = {'attempt': len(self.attempts_) + 1,
                                           'limits': {k: params[k] for k in set(_LIMITS.values())},
                                           'status': 'STARTED'}
                                self.attempts_.append(attempt)
                                _emit(self.progress_path, {'event': 'SEARCH_ATTEMPT_STARTED', **attempt})
                                tick = time.monotonic()
                                try:
                                    self._delegate.fit_development(XF, yF, AF, XC, yC, AC, metadata)
                                except (RuntimeError, _GeometryFailure) as exc:
                                    original_error = exc.original if isinstance(exc, _GeometryFailure) else exc
                                    field = _LIMITS.get(str(original_error)) if type(original_error) is RuntimeError else None
                                    if field is None or self._delegate.termination_reason_ != 'BUDGET_EXHAUSTED':
                                        attempt['status'] = 'ALGORITHM_ERROR_REQUIRES_REPAIR'
                                        raise
                                    attempt.update(status='REPLAY_REQUIRED', exhausted_limit=field)
                                    _emit(self.progress_path, {'event': 'SEARCH_BUDGET_EXTENSION', **attempt})
                                    params[field] *= 2
                                    if self.max_replays_per_session is not None and len(self.attempts_) >= self.max_replays_per_session:
                                        self.provenance_['status'] = 'SUSPENDED_REPLAY_REQUIRED'
                                        self.provenance_['next_limits'] = {k: params[k] for k in set(_LIMITS.values())}
                                        raise RuntimeError('SUSPENDED_REPLAY_REQUIRED: session allowance reached') from exc
                                except BaseException as exc:
                                    attempt['status'] = 'ALGORITHM_ERROR_REQUIRES_REPAIR' if isinstance(exc, Exception) else 'INTERRUPTED'
                                    raise
                                else:
                                    attempt['status'] = 'FIT_RETURNED'
                                    self.recovery_result_ = classify_recovery_fit(self._delegate, controller='strict')
                                    self.provenance_['status'] = 'COMPLETE_FEASIBLE'
                                    self.is_fitted_ = True
                                    break
                                finally:
                                    attempt.update(elapsed_seconds=time.monotonic() - tick,
                                        geometry_evaluations=getattr(self._delegate, '_geometry_evaluations_', None),
                                        utility_evaluations=getattr(self._delegate, '_utility_evaluations_', None),
                                        termination_reason=getattr(self._delegate, 'termination_reason_', None))
                                    _emit(self.progress_path, {'event': 'SEARCH_ATTEMPT_ENDED', **attempt})
                    self.provenance_.update(cache=copy.deepcopy(cache), kernel=copy.deepcopy(kernel), mds=copy.deepcopy(mds))
            self.provenance_['model_provenance'] = copy.deepcopy(self._delegate.provenance_)
            return self
        except BaseException as exc:
            self.is_fitted_ = False
            original_error = exc.original if isinstance(exc, _GeometryFailure) else exc
            if self.provenance_['status'] != 'SUSPENDED_REPLAY_REQUIRED':
                self.provenance_['status'] = 'INTERRUPTED' if not isinstance(original_error, Exception) else 'ALGORITHM_ERROR_REQUIRES_REPAIR'
            self.provenance_['failure'] = {
                'error_type': type(original_error).__name__,
                'frames': [{'file': Path(f.filename).name, 'line': f.lineno, 'function': f.name}
                           for f in traceback.extract_tb(original_error.__traceback__)],
            }
            _emit(self.progress_path, {'event': self.provenance_['status'], 'error_type': type(original_error).__name__})
            if isinstance(exc, _GeometryFailure):
                raise exc.original from exc
            raise
        finally:
            for name, stats in [('kernel', kernel), ('cache', cache), ('mds', mds)]:
                if stats is not None:
                    self.provenance_[name] = copy.deepcopy(stats)
            self.provenance_['elapsed_seconds'] = time.monotonic() - started

    def predict_event_probability(self, X, A=None):
        if not self.is_fitted_:
            raise RuntimeError('Completion has no admitted fitted model')
        return self._delegate.predict_event_probability(X, A)

    def predict_decision_proba(self, X, A=None):
        return self.predict_event_probability(X, A)

    def predict(self, X, A=None):
        return (self.predict_event_probability(X, A) >= .5).astype(int)


def make_completion_adapter(config, seed, *, cache_dir, progress_path=None, namespace):
    """Apply one uniform new budget policy to an unchanged registered F/C task."""
    modes = {'FAIRBIAS_BM_AE': 'BM_AE', 'FAIRBIAS_JOINT': 'JOINT'}
    if config.get('method') not in modes or type(seed) is not int or seed not in config.get('seeds', []):
        raise ValueError('Completion requires a registered full-method task and seed')
    if config.get('training_weighted') is not False:
        raise ValueError('Completion retains the unweighted development-training contract')
    params = copy.deepcopy(config['params'])
    mode = params.pop('mode', modes[config['method']])
    if mode != modes[config['method']]:
        raise ValueError('Registered mode differs from method')
    params['max_outer_iterations'] = 40
    return FairBiasCompletionAdapter(mode=mode, backbone=config['backbone'],
        random_state=seed, cache_dir=cache_dir, progress_path=progress_path,
        namespace=namespace, **params)
