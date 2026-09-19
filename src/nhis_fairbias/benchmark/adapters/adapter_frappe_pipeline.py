"""Versioned FRAPPE runtime with stable repeated MinDiff dataset iteration.

The frozen FRAPPE adapter remains the scientific implementation.  This adapter
only disables autotuning on its packed training dataset.  With TensorFlow 2.14,
the default asynchronous pipeline can consume different numbers of elements
from the repeated shuffled MinDiff streams at epoch boundaries.  Generated-data
diagnostics found equal initial weights and first-epoch updates, followed by
different second-epoch MinDiff batches under otherwise identical seeds.

No upstream module is modified: temporary delegation objects intercept only
``pack_min_diff_data``.  Shuffle seeds, per-iteration reshuffling, repetition,
batching, losses, optimizer, epochs and partitions retain their frozen values.
"""

from __future__ import annotations

from typing import Any

from .adapter_frappe import FrappeAdapter


RUNTIME_VARIANT = "frappe_pipeline_deterministic_v1_20260917"
PIPELINE_OPTIONS = {"autotune.enabled": False}


class _DelegatingProxy:
    def __init__(self, target: Any, **overrides: Any) -> None:
        self._target = target
        self._overrides = overrides

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._target, name)


class DeterministicPipelineFrappeAdapter(FrappeAdapter):
    """The registered FRAPPE method with a separately identified data runtime."""

    runtime_variant = RUNTIME_VARIANT

    def _require_tensorflow(self):
        tf, min_diff = super()._require_tensorflow()

        def pack_min_diff_data(*args: Any, **kwargs: Any):
            dataset = min_diff.keras.utils.pack_min_diff_data(*args, **kwargs)
            options = tf.data.Options()
            options.autotune.enabled = False
            packed = dataset.with_options(options)
            self._pipeline_options_applied_ = True
            return packed

        utilities = _DelegatingProxy(
            min_diff.keras.utils, pack_min_diff_data=pack_min_diff_data
        )
        keras = _DelegatingProxy(min_diff.keras, utils=utilities)
        proxy = _DelegatingProxy(min_diff, keras=keras)
        self.runtime_policy_ = {
            **self.runtime_policy_,
            "runtime_variant": RUNTIME_VARIANT,
            "tf_data_options": dict(PIPELINE_OPTIONS),
        }
        return tf, proxy

    def fit(self, *args: Any, **kwargs: Any):
        self._pipeline_options_applied_ = False
        result = super().fit(*args, **kwargs)
        if not self._pipeline_options_applied_:
            raise RuntimeError("FRAPPE deterministic pipeline options were not applied")
        self.fit_manifest_.update(
            runtime_variant=RUNTIME_VARIANT,
            tf_data_options=dict(PIPELINE_OPTIONS),
            pipeline_options_applied=True,
        )
        return result
