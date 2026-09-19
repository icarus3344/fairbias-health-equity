"""Synthetic contract checks for the bounded FRAPPE adapter."""

import numpy as np
import pytest

from nhis_fairbias.benchmark.adapters.adapter_frappe import FrappeAdapter
from nhis_fairbias.benchmark.adapters.base import NotSupportedError


def test_frappe_declares_probability_and_two_group_scope():
    adapter = FrappeAdapter(hidden_units=(2,), epochs=1)
    caps = adapter.get_capabilities()
    assert caps["output_type"] == "event_probability_p"
    assert caps["supports_arm2"] is False
    assert adapter.upstream_implementation.endswith("@dbddc6626ce5363f48139d461bd8d131216d5722")


def test_frappe_rejects_hisp7_before_optional_runtime():
    adapter = FrappeAdapter(hidden_units=(2,), epochs=1)
    X = np.arange(24, dtype=float).reshape(8, 3)
    y = np.array([0, 1] * 4)
    with pytest.raises(NotSupportedError, match="exactly two sensitive groups"):
        adapter.fit(
            X,
            y,
            np.arange(8),
            X_calibration=X,
            y_calibration=y,
            A_calibration=np.arange(8),
        )


def test_frappe_requires_explicit_f_and_c_partitions():
    adapter = FrappeAdapter(hidden_units=(2,), epochs=1)
    X = np.arange(12, dtype=float).reshape(4, 3)
    with pytest.raises(ValueError, match="explicit F training and C calibration"):
        adapter.fit(X, np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]))


def test_frappe_fails_closed_when_pinned_tensorflow_stack_is_unavailable():
    adapter = FrappeAdapter(hidden_units=(2,), epochs=1)
    X = np.arange(24, dtype=float).reshape(8, 3)
    y = np.array([0, 1] * 4)
    A = np.array([0, 1] * 4)
    try:
        adapter.fit(X, y, A, X_calibration=X, y_calibration=y, A_calibration=A)
    except NotSupportedError as exc:
        assert "tensorflow" in str(exc).lower()
    else:
        # On a host with the exact optional stack, fitting is allowed; the
        # output contract remains an event probability p.
        assert adapter.postprocessor_ is not None
        assert adapter.fit_manifest_["output_type"] == "event_probability_p"

