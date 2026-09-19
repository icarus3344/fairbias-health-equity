"""Opt-in exact BM acceleration, explicit empty-search outcome and MDS retry.

An exhausted search path is not a proof that every representation is infeasible.
No model is admitted when the last feature has been removed. The unchanged BM
search remains responsible for all proposals and stopping thresholds.
"""
from __future__ import annotations
from contextlib import ExitStack
from .adapter_fairbias import FairBiasAdapter
from ..joint_mds_numpy import numpy_mds_acceleration
from ..mds_budget_retry import mds_budget_retry


class SearchFeatureExhausted(ValueError):
    """The committed BM path removed every feature before yielding a model."""


class BMRecoveryAdapter(FairBiasAdapter):
    recovery_variant = 'bm_exact_kernel_explicit_outcomes_v1'

    def __init__(self, *args, use_mds_retry=False, **kwargs):
        if type(use_mds_retry) is not bool:
            raise ValueError('use_mds_retry must be an explicit boolean')
        self.use_mds_retry = use_mds_retry
        super().__init__(*args, **kwargs)
        self.name = 'FAIRBIAS_BM_RECOVERY_V1' + ('_MDS_RETRY' if use_mds_retry else '')

    @staticmethod
    def _validate_epsilon(epsilon_dict, stage):
        if (stage == 'intermediate F' and epsilon_dict
                and all(isinstance(group, dict) and not group for group in epsilon_dict.values())):
            raise SearchFeatureExhausted('SEARCH_FEATURE_EXHAUSTED: committed F representation has no features')
        return FairBiasAdapter._validate_epsilon(epsilon_dict, stage)

    def fit_representation(self, X_semantic, y, A):
        acceleration, retry = None, None
        outcome = 'FAILED_OR_INTERRUPTED'
        try:
            with ExitStack() as scope:
                if self.use_mds_retry:
                    retry = scope.enter_context(mds_budget_retry())
                acceleration = scope.enter_context(numpy_mds_acceleration())
                result = super().fit_representation(X_semantic, y, A)
            outcome = 'COMPLETE_FEASIBLE' if self.converged_ else 'SEARCH_STOPPED_INFEASIBLE'
            return result
        except SearchFeatureExhausted:
            self.termination_reason_ = 'SEARCH_FEATURE_EXHAUSTED'
            self.converged_ = False
            self.fitted_ = False
            outcome = 'SEARCH_FEATURE_EXHAUSTED'
            self.fit_manifest_['status'] = outcome
            self.fit_manifest_['termination_reason'] = outcome
            raise
        finally:
            self.recovery_receipt_ = {
                'variant': self.recovery_variant, 'status': outcome,
                'mds_budget_changed': self.use_mds_retry, 'kernel': acceleration,
                'mds_retry': retry, 'global_infeasibility_claim': False,
            }
            self.fit_manifest_['recovery'] = self.recovery_receipt_
