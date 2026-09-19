"""Synthetic upstream-loss, smooth-gradient and explicit corner checks."""

import numpy as np
import pytest
from aif360.algorithms.preprocessing.lfr_helpers import helpers

from nhis_fairbias.benchmark.lfr_analytic_objective import lfr_objective_and_gradient


def _upstream(p, *args):
    helpers.LFR_optim_objective.steps = 0
    return helpers.LFR_optim_objective(p, *args, print_interval=0, verbose=0)


def _fixture(seed=18):
    rng = np.random.RandomState(seed)
    xu, xp = rng.normal(size=(7, 4)), rng.normal(loc=0.8, size=(11, 4))
    yu, yp = rng.randint(2, size=7), rng.randint(2, size=11)
    k = 3
    p = np.r_[rng.uniform(0.15, 0.85, size=k), rng.normal(size=k * 4)]
    return p, (xu, xp, yu, yp, k)


def _central(fun, p, step=1e-6):
    out = np.empty_like(p)
    for index in range(len(p)):
        offset = np.zeros_like(p)
        offset[index] = step
        out[index] = (fun(p + offset) - fun(p - offset)) / (2 * step)
    return out


@pytest.mark.parametrize("seed", [0, 7, 18])
@pytest.mark.parametrize("weights", [(0.01, 1.0, 50.0), (1.0, 0.0, 0.0),
                                     (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
def test_loss_matches_upstream_and_gradient_matches_central_differences(seed, weights):
    p, args = _fixture(seed)
    args += weights
    diagnostics = {}
    loss, gradient = lfr_objective_and_gradient(p, *args, diagnostics=diagnostics)
    assert loss == _upstream(p, *args)
    assert not diagnostics["primitive_corner_present"]
    expected = _central(lambda theta: _upstream(theta, *args), p)
    np.testing.assert_allclose(gradient, expected, rtol=2e-5, atol=2e-8)


def test_swapping_sensitive_groups_preserves_loss_and_gradient():
    p, (xu, xp, yu, yp, k) = _fixture()
    left = lfr_objective_and_gradient(p, xu, xp, yu, yp, k)
    right = lfr_objective_and_gradient(p, xp, xu, yp, yu, k)
    # Swapping groups reverses the concatenated pooled-loss summation order.
    # Linux BLAS/NumPy may differ by one ulp; this is mathematical symmetry,
    # whereas upstream parity tests retain the original operation order.
    assert abs(left[0] - right[0]) <= 8 * np.finfo(float).eps * max(1.0, abs(left[0]))
    np.testing.assert_allclose(left[1], right[1], atol=1e-15)


def test_wide_features_and_feasible_weight_bounds_match_directional_differences():
    rng = np.random.RandomState(61)
    xu, xp = rng.normal(size=(13, 230)), rng.normal(loc=0.3, size=(23, 230))
    yu, yp = rng.randint(2, size=13), rng.randint(2, size=23)
    p = np.r_[[0.0, 1.0, 0.2, 0.5, 0.8], rng.normal(size=5 * 230)]
    args = (xu, xp, yu, yp, 5, 0.01, 1.0, 50.0)
    loss, gradient = lfr_objective_and_gradient(p, *args)
    assert loss == _upstream(p, *args)
    for _ in range(5):
        direction = rng.normal(size=len(p))
        # Keep central probes within the original box at weight endpoints.
        direction[:2] = 0
        direction /= np.linalg.norm(direction)
        h = 1e-5
        expected = (_upstream(p + h * direction, *args)
                    - _upstream(p - h * direction, *args)) / (2 * h)
        assert gradient @ direction == pytest.approx(expected, rel=2e-5, abs=2e-8)


@pytest.mark.parametrize("value", [-0.25, 0.0, 1.0, 1.25])
def test_clipped_probability_plateaus_match_upstream_and_have_zero_derivative(value):
    p = np.array([value, 0.7])
    args = (np.array([[0.2], [0.3]]), np.array([[1.2]]),
            np.array([0, 1]), np.array([1]), 1, 0.0, 1.0, 0.0)
    stats = {}
    loss, grad = lfr_objective_and_gradient(p, *args, diagnostics=stats)
    assert loss == _upstream(p, *args)
    np.testing.assert_array_equal(grad, np.zeros_like(p))
    assert stats["clipped_probabilities"] == 3
    assert stats["clip_corners"] == 0
    # A finite-difference stencil must stay on the clipped plateau. At w=0/1
    # a usual 1e-6 step crosses the tiny plateau and is not a local derivative.
    if value in [-0.25, 1.25]:
        np.testing.assert_allclose(_central(lambda t: _upstream(t, *args), p), grad)


@pytest.mark.parametrize("value,step,rtol", [
    (4 * np.finfo(float).eps, np.finfo(float).eps / 1024, 1e-6),
    (1 - 1e-8, 1e-11, 2e-5),
])
def test_log_gradients_near_clip_regions_use_local_finite_difference(value, step, rtol):
    p = np.array([value, 0.7])
    args = (np.array([[0.2]]), np.array([[1.2], [1.3]]),
            np.array([0]), np.array([1, 1]), 1, 0.0, 1.0, 0.0)
    loss, grad = lfr_objective_and_gradient(p, *args)
    assert loss == _upstream(p, *args)
    offset = np.array([step, 0.0])
    numerical = (_upstream(p + offset, *args) - _upstream(p - offset, *args)) / (2 * step)
    assert grad[0] == pytest.approx(numerical, rel=rtol)


@pytest.mark.parametrize("value", [np.finfo(float).eps, 1 - np.finfo(float).eps])
def test_exact_clip_corners_are_reported_without_claiming_a_classical_gradient(value):
    p = np.array([value, 0.7])
    args = (np.array([[0.2]]), np.array([[1.2]]), np.array([0]), np.array([1]),
            1, 0.0, 1.0, 0.0)
    stats = {}
    loss, gradient = lfr_objective_and_gradient(p, *args, diagnostics=stats)
    assert loss == _upstream(p, *args)
    assert stats["clip_corners"] == 2
    assert gradient[0] == 0
    # An inward direction leaves the clip, so it has a nonzero one-sided
    # derivative; the reported zero is a convention, not a classical one.
    direction = 1 if value < 0.5 else -1
    moved = p.copy()
    moved[0] += direction * np.finfo(float).eps
    assert _upstream(moved, *args) < loss


def test_abs_fairness_corner_has_distinct_one_sided_derivatives():
    # Coincident prototypes have equal mean membership even though groups
    # differ; separating the prototypes creates a genuine fairness cusp.
    p = np.array([0.3, 0.7, 0.4, 0.4])
    args = (np.array([[-1.0], [-0.5]]), np.array([[1.0], [2.0]]),
            np.array([0, 1]), np.array([1, 0]), 2, 0.0, 0.0, 1.0)
    stats = {}
    loss, gradient = lfr_objective_and_gradient(p, *args, diagnostics=stats)
    assert loss == _upstream(p, *args) == 0
    assert stats["absolute_value_corners"] == 2
    np.testing.assert_array_equal(gradient, np.zeros_like(p))
    h = np.array([0.0, 0.0, 1e-6, 0.0])
    right = (_upstream(p + h, *args) - loss) / 1e-6
    left = (loss - _upstream(p - h, *args)) / 1e-6
    assert right > 0.1 and left < -0.1
    np.testing.assert_allclose(_central(lambda t: _upstream(t, *args), p), gradient, atol=1e-9)


def test_zero_euclidean_distance_matches_loss_and_reports_genuine_cusp():
    p = np.array([0.2, 0.8, 0.0, 2.0])
    args = (np.array([[0.0]]), np.array([[0.7], [1.2]]),
            np.array([1]), np.array([0, 1]), 2, 0.0, 1.0, 0.0)
    stats = {}
    loss, gradient = lfr_objective_and_gradient(p, *args, diagnostics=stats)
    assert loss == _upstream(p, *args)
    assert stats["zero_distance_pairs"] == 1
    h = np.array([0.0, 0.0, 1e-6, 0.0])
    right = (_upstream(p + h, *args) - loss) / 1e-6
    left = (loss - _upstream(p - h, *args)) / 1e-6
    assert abs(right - left) > 0.1
    assert gradient[2] == pytest.approx((right + left) / 2, abs=1e-6)


@pytest.mark.parametrize("bad", ["empty_group", "nonfinite", "label", "dimension"])
def test_invalid_objective_inputs_fail_closed(bad):
    p, args = _fixture()
    xu, xp, yu, yp, k = args
    if bad == "empty_group":
        xu, yu = xu[:0], yu[:0]
    elif bad == "nonfinite":
        p[0] = np.nan
    elif bad == "label":
        yu[0] = 2
    else:
        p = p[:-1]
    with pytest.raises(ValueError):
        lfr_objective_and_gradient(p, xu, xp, yu, yp, k)
