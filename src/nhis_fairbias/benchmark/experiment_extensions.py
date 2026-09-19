"""Data-independent, capacity-matched extension and ablation registration.

All configurations use F/C/S under the common selection contract. The fixed
AE/geometry ablations are labelled separately and are not a tuned-method claim.
"""
from __future__ import annotations

import itertools
from .experiment_registry import identity, SEEDS, EPSILON_RATIOS

ARMS = ('arm_001', 'arm_002', 'arm_003', 'arm_004')


def enumerate_extensions():
    records = []

    def add(family, arm, backbone, method, params, *, status='REGISTERED', notes=(), complexity=1.):
        row = dict(family=family, arm_id=arm, backbone=backbone, method=method,
                   params=params, seeds=list(SEEDS) if status == 'REGISTERED' else [],
                   status=status, training_weighted=False, complexity=float(complexity), notes=list(notes))
        row['candidate_id'] = identity(row)[:20]
        records.append(row)

    for arm, n, leaves in itertools.product(ARMS, (100, 200), (15, 31)):
        base = dict(n_estimators=n, num_leaves=leaves, learning_rate=.05, n_jobs=1)
        add('recent_fairgbm', arm, 'FAIRGBM_BASE', 'UNMITIGATED', {'estimator_params': base}, complexity=n*leaves)
        for ratio in EPSILON_RATIOS:
            add('recent_fairgbm', arm, 'FAIRGBM_BASE', 'FAIRBIAS_BM',
                dict(estimator_params=base, epsilon_ratio=ratio, max_iterations=50, max_geometry_evaluations=20000), complexity=n*leaves)
        # Equal FPR/FNR slack, a six-point ONE-dimensional training grid.
        # This is not an EO max-gap parameter; common S evaluation uses EO.
        for tolerance in (.005, .01, .02, .05, .1, .2):
            add('recent_fairgbm', arm, 'FAIRGBM_BASE', 'FAIRGBM_EO',
                dict(**base, constraint_type='FPR,FNR', multiplier_learning_rate=.1,
                     constraint_fpr_tolerance=tolerance, constraint_fnr_tolerance=tolerance), complexity=n*leaves)

    for arm, backbone in itertools.product(ARMS, ('LR', 'GBDT')):
        if arm == 'arm_002':
            add('recent_frappe', arm, backbone, 'FRAPPE_EO', {}, status='NOT_SUPPORTED',
                notes=('Official source is a binary-group MinDiff postprocessor; no HISP7 reinterpretation.',))
        else:
            grid = [dict(C=c) for c in (.01, .1, 1., 10.)] if backbone == 'LR' else [
                dict(estimator_params=dict(n_estimators=n, max_depth=d, learning_rate=.05, subsample=1.))
                for n, d in itertools.product((100, 200), (2, 3))]
            for base, penalty in itertools.product(grid, (.01, .1, 1., 10.)):
                complexity = base['C'] if backbone == 'LR' else base['estimator_params']['n_estimators'] * 2**base['estimator_params']['max_depth']
                add('recent_frappe', arm, backbone, 'FRAPPE_EO',
                    dict(**base, hidden_units=[16], epochs=100, batch_size=32, learning_rate=.01,
                         mindiff_weight=penalty, mmd_kernel_decay_length=.1, regularization_strength=1.),
                    complexity=complexity, notes=('Pinned official postprocessor mathematics with registered compact hidden width 16.',))
        base = dict(C=1.) if backbone == 'LR' else dict(n_estimators=100, max_depth=2, learning_rate=.05, subsample=1.)
        for mode in ('BM_AE', 'JOINT'):
            add('fixed_ae_ablation', arm, backbone, 'FAIRBIAS_' + mode,
                dict(mode=mode, estimator_params=base, epsilon_ratio=1., max_outer_iterations=10,
                     max_bm_steps=50, max_utility_evaluations=500, max_geometry_evaluations=20000),
                complexity=1. if backbone == 'LR' else 400.,
                notes=('Fixed ablation anchor, not a separately tuned full AE grid. Retain corresponding BM anchor for evaluation.',))

    tabm = dict(k=4, n_blocks=2, d_block=64, dropout=0., epochs=100, batch_size=1024, learning_rate=.001)
    for arm in ARMS:
        add('compact_tabm', arm, 'TABM', 'UNMITIGATED', dict(estimator_params=tabm),
            notes=('Compact CPU TabM variant; not paper-default-capacity replication.',))
        for ratio in EPSILON_RATIOS:
            add('compact_tabm', arm, 'TABM', 'FAIRBIAS_BM',
                dict(estimator_params=tabm, epsilon_ratio=ratio, max_iterations=50, max_geometry_evaluations=20000),
                notes=('Same compact TabM architecture as the matched ordinary baseline.',))

    for backbone in ('LR', 'GBDT'):
        base = dict(C=1.) if backbone == 'LR' else dict(estimator_params=dict(n_estimators=100, max_depth=2, learning_rate=.05, subsample=1.))
        for profile in ('stress_elbow', 'fixed2_same_epsilon', 'fixed2_own_epsilon'):
            add('fixed_geometry_ablation', 'arm_004', backbone, 'FAIRBIAS_GEOMETRY_' + profile.upper(),
                dict(**base, geometry_profile=profile, epsilon_ratio=1., max_iterations=50, max_geometry_evaluations=20000),
                notes=('Fixed Arm-004 anchor; report path traces and absolute epsilon, not only outcome gaps.',))
    return records
