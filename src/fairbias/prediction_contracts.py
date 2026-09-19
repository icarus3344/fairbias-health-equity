"""Contracts and validators for probabilistic binary classification predictions."""

from typing import Any, Optional, Sequence, Tuple
import numpy as np


class ProbabilityValidationError(ValueError):
    """Raised when model predictions violate probabilistic contracts."""
    pass


def validate_and_extract_positive_probabilities(
    model: Any,
    X: Any,
    expected_classes: Sequence[Any] = (0, 1),
    pos_label: Any = 1,
    row_sum_tol: float = 1e-4,
) -> np.ndarray:
    """Validate binary model probability outputs and extract positive-class probabilities.

    Enforces the following contracts:
    - Model must provide predict_proba
    - Model classes_ must have exactly 2 classes matching expected_classes
    - Dynamic positive class column indexing (handles classes_=[1, 0] as well as [0, 1])
    - Output must be a 2D numpy array of shape (n_samples, 2)
    - All probability values must be finite and within [0.0, 1.0]
    - Row sums must equal 1.0 within row_sum_tol

    Returns
    -------
    np.ndarray
        1D float array of positive class probabilities of length n_samples.
    """
    if isinstance(row_sum_tol, (bool, np.bool_)) or not isinstance(row_sum_tol, (int, float, np.floating, np.integer)):
        raise ProbabilityValidationError(f"row_sum_tol must be a finite float, got {type(row_sum_tol).__name__}")
    if not np.isfinite(row_sum_tol) or row_sum_tol < 0.0 or row_sum_tol > 0.5:
        raise ProbabilityValidationError(f"row_sum_tol must be finite, non-negative, and <= 0.5, got {row_sum_tol}")

    exp_list = list(expected_classes)
    if len(exp_list) != 2 or len(set(exp_list)) != 2:
        raise ProbabilityValidationError(f"expected_classes must contain exactly 2 distinct classes, got {expected_classes}")

    if not hasattr(model, "predict_proba"):
        raise ProbabilityValidationError(
            f"Model {type(model).__name__} does not provide predict_proba; probabilistic predictor required"
        )

    if not hasattr(model, "classes_"):
        raise ProbabilityValidationError(
            f"Model {type(model).__name__} does not have classes_ attribute"
        )

    classes = list(model.classes_)
    if len(classes) != 2:
        raise ProbabilityValidationError(
            f"Binary probability evaluation requires exactly 2 classes in model.classes_, "
            f"got {len(classes)}: {classes}"
        )

    exp_set = set(exp_list)
    cls_set = set(classes)
    if cls_set != exp_set:
        raise ProbabilityValidationError(
            f"Model classes_ {classes} do not match expected binary classes {exp_list}"
        )

    if pos_label not in classes:
        raise ProbabilityValidationError(
            f"Positive label {pos_label!r} not found in model classes_ {classes}"
        )

    pos_idx = classes.index(pos_label)

    n_samples = len(X) if hasattr(X, "__len__") else 0
    if n_samples == 0:
        return np.empty((0,), dtype=float)

    try:
        proba = model.predict_proba(X)
    except Exception as exc:
        raise ProbabilityValidationError(f"model.predict_proba failed: {type(exc).__name__}: {exc}") from exc

    if not isinstance(proba, np.ndarray):
        proba = np.asarray(proba)

    if not np.issubdtype(proba.dtype, np.number) or np.iscomplexobj(proba):
        raise ProbabilityValidationError(f"Predicted probabilities must be real numeric floats, got {proba.dtype}")

    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ProbabilityValidationError(
            f"predict_proba output must be 2D array with shape (n, 2), got shape {proba.shape}"
        )

    if proba.shape[0] != n_samples:
        raise ProbabilityValidationError(
            f"predict_proba output length ({proba.shape[0]}) does not match input length ({n_samples})"
        )

    if not np.all(np.isfinite(proba)):
        raise ProbabilityValidationError("Predicted probabilities contain NaN, Inf, or non-finite values")

    if np.any(proba < 0.0) or np.any(proba > 1.0):
        min_v = float(np.min(proba))
        max_v = float(np.max(proba))
        raise ProbabilityValidationError(
            f"Predicted probabilities contain values outside [0.0, 1.0]: min={min_v}, max={max_v}"
        )

    row_sums = np.sum(proba, axis=1)
    max_dev = float(np.max(np.abs(row_sums - 1.0)))
    if max_dev > row_sum_tol:
        raise ProbabilityValidationError(
            f"Predicted probabilities row sums deviate from 1.0 beyond tolerance {row_sum_tol}: max_dev={max_dev}"
        )

    return np.asarray(proba[:, pos_idx], dtype=float)
