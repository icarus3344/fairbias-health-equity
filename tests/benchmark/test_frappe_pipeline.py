"""Contracts for the additive FRAPPE dataset runtime; no TensorFlow required."""

from types import SimpleNamespace

import pytest

from nhis_fairbias.benchmark.adapters.adapter_frappe import FrappeAdapter
from nhis_fairbias.benchmark.adapters.adapter_frappe_pipeline import (
    DeterministicPipelineFrappeAdapter,
    PIPELINE_OPTIONS,
    RUNTIME_VARIANT,
)


def _runtime(monkeypatch):
    calls = []
    derived = object()

    class Dataset:
        def with_options(self, options):
            calls.append(("options", options))
            return derived

    dataset = Dataset()

    def pack(*args, **kwargs):
        calls.append(("pack", args, kwargs))
        return dataset

    build = object()
    models = object()
    losses = object()
    utilities = SimpleNamespace(pack_min_diff_data=pack, build_min_diff_dataset=build)
    tfmr = SimpleNamespace(keras=SimpleNamespace(utils=utilities, models=models), losses=losses)
    tf = SimpleNamespace(data=SimpleNamespace(
        Options=lambda: SimpleNamespace(autotune=SimpleNamespace(enabled=None))
    ))

    def require(self):
        self.runtime_policy_ = {"tf_seed": self.random_state, "deterministic_ops": True}
        return tf, tfmr

    monkeypatch.setattr(FrappeAdapter, "_require_tensorflow", require)
    return tf, tfmr, calls, derived


def test_only_packing_is_wrapped_without_mutating_upstream(monkeypatch):
    tf, original, calls, derived = _runtime(monkeypatch)
    adapter = DeterministicPipelineFrappeAdapter(random_state=19)
    actual_tf, proxy = adapter._require_tensorflow()
    original_pack = original.keras.utils.pack_min_diff_data
    original_data, correction_data = object(), object()
    assert proxy.keras.utils.pack_min_diff_data(
        original_data, min_diff_dataset=correction_data
    ) is derived
    assert actual_tf is tf
    assert calls[0] == ("pack", (original_data,), {"min_diff_dataset": correction_data})
    assert calls[1][1].autotune.enabled is False
    assert adapter._pipeline_options_applied_ is True
    assert original.keras.utils.pack_min_diff_data is original_pack
    assert proxy.keras.utils.build_min_diff_dataset is original.keras.utils.build_min_diff_dataset
    assert proxy.keras.models is original.keras.models
    assert proxy.losses is original.losses
    assert adapter.runtime_policy_ == {
        "tf_seed": 19, "deterministic_ops": True,
        "runtime_variant": RUNTIME_VARIANT, "tf_data_options": PIPELINE_OPTIONS,
    }


def test_fit_manifest_records_applied_variant_and_preserves_base_manifest(monkeypatch):
    _runtime(monkeypatch)
    supplied = object()

    def frozen_fit(self, *args, **kwargs):
        assert args == (supplied,)
        assert kwargs == {"calibration_marker": supplied}
        _, tfmr = self._require_tensorflow()
        tfmr.keras.utils.pack_min_diff_data(original_dataset=supplied)
        self.fit_manifest_ = {"base_partition": "F", "correction_partition": "C"}
        return self

    monkeypatch.setattr(FrappeAdapter, "fit", frozen_fit)
    adapter = DeterministicPipelineFrappeAdapter()
    assert adapter.fit(supplied, calibration_marker=supplied) is adapter
    assert adapter.fit_manifest_ == {
        "base_partition": "F", "correction_partition": "C",
        "runtime_variant": RUNTIME_VARIANT, "tf_data_options": PIPELINE_OPTIONS,
        "pipeline_options_applied": True,
    }


def test_success_without_packing_fails_closed_instead_of_claiming_application(monkeypatch):
    monkeypatch.setattr(FrappeAdapter, "fit", lambda self: self)
    adapter = DeterministicPipelineFrappeAdapter()
    with pytest.raises(RuntimeError, match="options were not applied"):
        adapter.fit()


def test_original_fit_error_is_preserved(monkeypatch):
    def fail(self):
        raise ValueError("original fit failed")

    monkeypatch.setattr(FrappeAdapter, "fit", fail)
    with pytest.raises(ValueError, match="original fit failed"):
        DeterministicPipelineFrappeAdapter().fit()
