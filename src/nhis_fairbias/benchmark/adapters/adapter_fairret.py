"""Matched CPU MLP and the pinned fairret 0.1.3 regularized adapter."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn

from fairbias.prediction_contracts import validate_and_extract_positive_probabilities

from .base import BaseMethodAdapter, NotSupportedError

try:
    from fairret.loss import NormLoss
    from fairret.statistic import FalsePositiveRate, PositiveRate, TruePositiveRate
except ImportError as exc:  # pragma: no cover - exercised by environment setup
    raise ImportError(
        "FairretAdapter requires the pinned fairret artifact on PYTHONPATH"
    ) from exc


class _TwoLayerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: Sequence[int] = (64, 32)):
        super().__init__()
        if tuple(hidden_layers) != (64, 32):
            raise ValueError("benchmark MLP architecture is fixed at two layers (64, 32)")
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TorchMLPClassifier:
    """Small sklearn-ish full-batch CPU classifier shared by both MLP arms."""

    def __init__(
        self,
        input_dim: Optional[int] = None,
        *,
        hidden_layers: Sequence[int] = (64, 32),
        learning_rate: float = 0.001,
        epochs: int = 100,
        random_state: int = 42,
        fairness_loss: Optional[nn.Module] = None,
        fairness_coefficient: float = 0.0,
        fairness_variant: Optional[str] = None,
    ):
        if input_dim is not None:
            if isinstance(input_dim, (bool, np.bool_)) or not isinstance(input_dim, (int, np.integer)) or int(input_dim) <= 0:
                raise ValueError("input_dim must be a positive exact integer or None")
        self.input_dim = None if input_dim is None else int(input_dim)
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = float(learning_rate)
        if isinstance(epochs, (bool, np.bool_)) or not isinstance(epochs, (int, np.integer)):
            raise ValueError("epochs must be an exact integer")
        if int(epochs) != epochs or int(epochs) <= 0:
            raise ValueError("epochs must be a positive exact integer")
        if not np.isfinite(learning_rate) or float(learning_rate) <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        self.epochs = int(epochs)
        self.random_state = int(random_state)
        self.fairness_loss = fairness_loss
        self.fairness_coefficient = float(fairness_coefficient)
        self.fairness_variant = fairness_variant
        self.model_: Optional[_TwoLayerMLP] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.loss_history_: list[float] = []
        self.final_loss_: Optional[float] = None
        self.finite_loss_count_: int = 0
        self.n_epochs_: int = 0

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_layers": self.hidden_layers,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "random_state": self.random_state,
            "fairness_loss": self.fairness_loss,
            "fairness_coefficient": self.fairness_coefficient,
            "fairness_variant": self.fairness_variant,
        }

    def set_params(self, **params: Any) -> "TorchMLPClassifier":
        valid = set(self.get_params())
        unknown = set(params) - valid
        if unknown:
            raise ValueError(f"Invalid TorchMLPClassifier parameters: {sorted(unknown)}")
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _check_X(self, X: Any) -> np.ndarray:
        raw = np.asarray(X)
        if raw.ndim != 2 or np.iscomplexobj(raw):
            raise ValueError("X must be a finite two-dimensional real numeric array")
        try:
            arr = np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must be a finite two-dimensional real numeric array") from exc
        if not np.all(np.isfinite(arr)):
            raise ValueError("X must be a finite two-dimensional numeric array")
        expected_dim = self.n_features_in_ if self.n_features_in_ is not None else self.input_dim
        if expected_dim is not None and arr.shape[1] != expected_dim:
            raise ValueError("X feature count does not match input_dim/n_features_in_")
        return arr

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sensitive_onehot: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "TorchMLPClassifier":
        # A failed refit must invalidate the previous fitted model.
        self.model_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.loss_history_ = []
        self.final_loss_ = None
        self.finite_loss_count_ = 0
        self.n_epochs_ = 0
        if sample_weight is not None:
            raise NotSupportedError(
                "TorchMLPClassifier does not accept survey weights; weighted BCE is a separate sensitivity"
            )
        X_arr = self._check_X(X)
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim != 1 or len(y_arr) != len(X_arr) or not np.all(np.isfinite(y_arr)):
            raise ValueError("y must be a finite vector matching X")
        if not np.all((y_arr == 0.0) | (y_arr == 1.0)):
            raise ValueError("y must be binary")
        if not np.any(y_arr == 0.0) or not np.any(y_arr == 1.0):
            raise ValueError("y must contain both classes")
        if self.fairness_loss is not None:
            if sensitive_onehot is None:
                raise ValueError("fairness training requires one-hot sensitive features")
            if sensitive_onehot.ndim != 2 or sensitive_onehot.shape[0] != len(X_arr):
                raise ValueError("invalid sensitive feature shape")
            if not np.all(np.isfinite(sensitive_onehot)):
                raise ValueError("sensitive features must be finite")

        self.n_features_in_ = int(X_arr.shape[1])
        if self.fairness_coefficient < 0.0 or not np.isfinite(self.fairness_coefficient):
            raise ValueError("fairness_coefficient must be finite and non-negative")
        # fork_rng restores the caller's CPU RNG state after deterministic
        # initialization/training; no global seed leakage between methods.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                self.model_ = _TwoLayerMLP(self.n_features_in_, self.hidden_layers)
                optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
                x_t = torch.from_numpy(X_arr)
                y_t = torch.from_numpy(y_arr.reshape(-1, 1))
                sens_t = None if sensitive_onehot is None else torch.from_numpy(
                    np.asarray(sensitive_onehot, dtype=np.float32)
                )
                bce = nn.BCEWithLogitsLoss()
                self.model_.train()
                for _ in range(self.epochs):
                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model_(x_t)
                    loss = bce(logits, y_t)
                    if self.fairness_loss is not None and self.fairness_coefficient > 0.0:
                        if self.fairness_variant == "EO":
                            fairness_value = self.fairness_loss(logits, sens_t, y_t)
                        else:
                            fairness_value = self.fairness_loss(logits, sens_t)
                        loss = loss + self.fairness_coefficient * fairness_value
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite TorchMLP training loss")
                    loss.backward()
                    optimizer.step()
                    value = float(loss.detach().cpu().item())
                    self.loss_history_.append(value)
                    self.finite_loss_count_ += 1
                    self.n_epochs_ += 1
                self.final_loss_ = self.loss_history_[-1]
            finally:
                torch.set_num_threads(previous_threads)
        self.classes_ = np.asarray([0, 1])
        return self

    def _predict_logits(self, X: Any) -> np.ndarray:
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("TorchMLPClassifier must be fitted before prediction")
        x_arr = self._check_X(X)
        with torch.no_grad():
            self.model_.eval()
            logits = self.model_(torch.from_numpy(x_arr)).cpu().numpy().reshape(-1)
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("TorchMLP produced non-finite logits")
        return logits

    def predict_proba(self, X: Any) -> np.ndarray:
        logits = self._predict_logits(X)
        p = torch.sigmoid(torch.from_numpy(logits)).numpy()
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _MLPAdapterBase(BaseMethodAdapter):
    output_type = "event_probability_p"
    requires_sensitive_at_predict = False
    supports_arm2 = True

    def __init__(
        self,
        *,
        epochs: int = 100,
        learning_rate: float = 0.001,
        random_state: int = 42,
        fairness_coefficient: float = 0.0,
        fairness_variant: str = "EO",
    ):
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self.fairness_coefficient = float(fairness_coefficient)
        self.fairness_variant = str(fairness_variant).upper()
        if self.fairness_variant not in ("EO", "DP"):
            raise ValueError("fairness_variant must be 'EO' or 'DP'")
        self.classifier_: Optional[TorchMLPClassifier] = None
        self.sensitive_groups_: Optional[np.ndarray] = None
        self.loss_instance_: Optional[nn.Module] = None
        self.training_metadata_: Dict[str, Any] = {}

    def _make_loss(self) -> Optional[nn.Module]:
        return None

    def _encode_sensitive(self, A: Any, y: np.ndarray) -> np.ndarray:
        values = np.asarray(A)
        if values.ndim != 1 or len(values) != len(y):
            raise ValueError("A must be a vector matching y")
        if not np.all(np.isfinite(values)):
            raise ValueError("A must be finite")
        groups = np.unique(values)
        if len(groups) < 2:
            raise NotSupportedError("fairret requires at least two sensitive groups")
        if self.fairness_variant == "EO":
            for group in groups:
                group_y = y[values == group]
                if not np.any(group_y == 0) or not np.any(group_y == 1):
                    raise NotSupportedError(
                        "Fairret EO requires positive and negative support in every F group"
                    )
        self.sensitive_groups_ = groups
        return (values[:, None] == groups[None, :]).astype(np.float32)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        A: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "_MLPAdapterBase":
        if sample_weight is not None:
            raise NotSupportedError(
                "Fairret MLP does not accept survey weights; weighted BCE is a separate sensitivity"
            )
        y_arr = np.asarray(y)
        if y_arr.ndim != 1 or not np.all(np.isfinite(y_arr)) or not np.all((y_arr == 0) | (y_arr == 1)):
            raise ValueError("y must be finite binary labels")
        sens = self._encode_sensitive(A, y_arr.astype(int)) if self.fairness_coefficient > 0 else None
        self.loss_instance_ = self._make_loss() if self.fairness_coefficient > 0 else None
        self.training_metadata_ = {
            "architecture": {"hidden_layers": (64, 32), "activation": "ReLU"},
            "optimizer": "Adam",
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "fairness_variant": self.fairness_variant,
            "fairness_coefficient": self.fairness_coefficient,
            "loss_instance": None if self.loss_instance_ is None else type(self.loss_instance_).__name__,
        }
        if isinstance(self.loss_instance_, _SummedFairretLoss):
            self.training_metadata_["statistics"] = ("TruePositiveRate", "FalsePositiveRate")
            self.training_metadata_["norm_p"] = (self.loss_instance_.tpr_loss.p, self.loss_instance_.fpr_loss.p)
        elif self.loss_instance_ is not None:
            self.training_metadata_["statistics"] = ("PositiveRate",)
            self.training_metadata_["norm_p"] = (self.loss_instance_.p,)
        self.classifier_ = TorchMLPClassifier(
            input_dim=np.asarray(X).shape[1],
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            random_state=self.random_state,
            fairness_loss=self.loss_instance_,
            fairness_coefficient=self.fairness_coefficient,
            fairness_variant=self.fairness_variant if self.loss_instance_ is not None else None,
        )
        self.classifier_.fit(X, y_arr, sensitive_onehot=sens)
        self.training_metadata_["final_loss"] = self.classifier_.final_loss_
        self.training_metadata_["finite_loss_count"] = self.classifier_.finite_loss_count_
        return self

    def predict(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        if self.classifier_ is None:
            raise RuntimeError("MLP adapter must be fitted before prediction")
        return self.classifier_.predict(X)

    def predict_proba(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        if self.classifier_ is None:
            raise RuntimeError("MLP adapter must be fitted before prediction")
        return self.classifier_.predict_proba(X)

    def predict_event_probability(
        self, X: np.ndarray, A: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self.classifier_ is None:
            raise RuntimeError("MLP adapter must be fitted before prediction")
        return validate_and_extract_positive_probabilities(self.classifier_, X, pos_label=1)

    def predict_decision_proba(
        self, X: np.ndarray, A: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.predict_event_probability(X, A=A)


class UnmitigatedMLPAdapter(_MLPAdapterBase):
    """Matched plain MLP baseline using the same architecture and seed."""

    name = "UNMITIGATED_MLP"
    literature_reference = "Matched two-hidden-layer MLP baseline"
    upstream_implementation = "torch.nn (CPU)"

    def __init__(self, **kwargs: Any):
        coefficient = float(kwargs.get("fairness_coefficient", 0.0))
        if coefficient != 0.0:
            raise ValueError("UnmitigatedMLPAdapter cannot use fairness regularization")
        kwargs["fairness_coefficient"] = 0.0
        super().__init__(**kwargs)


class FairretAdapter(_MLPAdapterBase):
    """Fairret 0.1.3 NormLoss regularization over full-F batches."""

    name = "FAIRRET_MLP"
    literature_reference = "Fairret (ICLR 2024), paper id NnyD0Rjx2B"
    upstream_implementation = "fairret==0.1.3"

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("fairness_coefficient", 1.0)
        super().__init__(**kwargs)

    def _make_loss(self) -> nn.Module:
        if self.fairness_variant == "DP":
            return NormLoss(PositiveRate())
        if self.fairness_variant == "EO":
            return _SummedFairretLoss(NormLoss(TruePositiveRate()), NormLoss(FalsePositiveRate()))
        raise ValueError("fairness_variant must be 'EO' or 'DP'")


class _SummedFairretLoss(nn.Module):
    """Sum the official TPR and FPR NormLoss penalties for EO."""

    def __init__(self, tpr_loss: nn.Module, fpr_loss: nn.Module):
        super().__init__()
        self.tpr_loss = tpr_loss
        self.fpr_loss = fpr_loss

    def forward(self, logits: torch.Tensor, sens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.tpr_loss(logits, sens, labels) + self.fpr_loss(logits, sens, labels)
