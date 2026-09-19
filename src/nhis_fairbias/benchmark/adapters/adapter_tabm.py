"""Small, source-pinned TabM predictor adapter.

The neural model itself is loaded from the official Yandex TabM source snapshot
recorded under ``artifacts/nhis/benchmark_dependencies_20260916/tabm_source``.
This wrapper fixes a compact numerical-only configuration and supplies the
benchmark's sklearn-like fit/predict contract.
"""

from __future__ import annotations

import importlib.util
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from fairbias.prediction_contracts import validate_and_extract_positive_probabilities

from .base import BaseMethodAdapter, NotSupportedError


_TABM_COMMIT = "28e47ae301c92ec37787dde1ce923a0793f405b4"
_TABM_SOURCE_RELATIVE = Path("artifacts/nhis/benchmark_dependencies_20260916/tabm_source/tabm.py")
_RTDL_SITE_RELATIVE = Path("artifacts/nhis/benchmark_dependencies_20260916/site-packages")


def _load_official_tabm() -> Any:
    """Load the pinned official ``TabM`` class without requiring installation."""
    repo_root = Path(__file__).resolve().parents[4]
    site_path = repo_root / _RTDL_SITE_RELATIVE
    source_path = repo_root / _TABM_SOURCE_RELATIVE
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != "fc654af6a16bac53d893a8265c79d7af4ebddcb95ad0d600cc6b6bc6b7317ade":
        raise ImportError("Pinned TabM source hash mismatch")
    if str(site_path) not in sys.path:
        sys.path.insert(0, str(site_path))
    module_name = "_fairbias_official_tabm"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load pinned TabM source at {source_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module.TabM


class TabMClassifier:
    """Compact official TabM numerical classifier with fixed F-only epochs."""

    def __init__(
        self,
        input_dim: Optional[int] = None,
        *,
        k: int = 4,
        n_blocks: int = 2,
        d_block: int = 64,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 1024,
        random_state: int = 42,
    ):
        if input_dim is not None and (isinstance(input_dim, (bool, np.bool_)) or int(input_dim) != input_dim or int(input_dim) <= 0):
            raise ValueError("input_dim must be a positive integer or None")
        if isinstance(epochs, (bool, np.bool_)) or not isinstance(epochs, (int, np.integer)) or int(epochs) != epochs or int(epochs) <= 0:
            raise ValueError("epochs must be a positive exact integer")
        if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)) or int(batch_size) <= 0:
            raise ValueError("batch_size must be a positive exact integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        self.input_dim = None if input_dim is None else int(input_dim)
        self.k = int(k)
        self.n_blocks = int(n_blocks)
        self.d_block = int(d_block)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.model_: Optional[torch.nn.Module] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.loss_history_: list[float] = []
        self.final_loss_: Optional[float] = None
        self.finite_loss_count_: int = 0

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "k": self.k,
            "n_blocks": self.n_blocks,
            "d_block": self.d_block,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
        }

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_model_state_dict"] = None if self.model_ is None else self.model_.state_dict()
        state.pop("model_")
        return state

    def __setstate__(self, state):
        weights = state.pop("_model_state_dict")
        self.__dict__.update(state)
        self.model_ = None
        if weights is not None:
            with torch.random.fork_rng(devices=[]):
                self.model_ = _load_official_tabm().make(n_num_features=self.n_features_in_, cat_cardinalities=[],
                    d_out=1, k=self.k, n_blocks=self.n_blocks, d_block=self.d_block, dropout=self.dropout, arch_type="tabm-packed")
                self.model_.load_state_dict(weights)
                self.model_.eval()

    def set_params(self, **params: Any) -> "TabMClassifier":
        unknown = set(params) - set(self.get_params())
        if unknown:
            raise ValueError(f"Invalid TabMClassifier parameters: {sorted(unknown)}")
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _check_X(self, X: Any) -> np.ndarray:
        raw = np.asarray(X)
        if raw.ndim != 2 or np.iscomplexobj(raw):
            raise ValueError("X must be a finite two-dimensional real array")
        try:
            arr = np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must be a finite two-dimensional real array") from exc
        if not np.all(np.isfinite(arr)):
            raise ValueError("X must contain only finite values")
        expected = self.n_features_in_ if self.n_features_in_ is not None else self.input_dim
        if expected is not None and arr.shape[1] != expected:
            raise ValueError("X feature count does not match input_dim/n_features_in_")
        return arr

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "TabMClassifier":
        self.model_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.loss_history_ = []
        self.final_loss_ = None
        self.finite_loss_count_ = 0
        if sample_weight is not None:
            raise NotSupportedError("TabMClassifier does not support sample_weight in this registered condition")
        X_arr = self._check_X(X)
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim != 1 or len(y_arr) != len(X_arr) or not np.all(np.isfinite(y_arr)):
            raise ValueError("y must be a finite vector matching X")
        if not np.all((y_arr == 0.0) | (y_arr == 1.0)):
            raise ValueError("y must contain binary labels")
        if not np.any(y_arr == 0.0) or not np.any(y_arr == 1.0):
            raise ValueError("y must contain both classes")
        self.n_features_in_ = int(X_arr.shape[1])

        TabM = _load_official_tabm()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            old_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                self.model_ = TabM.make(
                    n_num_features=self.n_features_in_,
                    cat_cardinalities=[],
                    d_out=1,
                    k=self.k,
                    n_blocks=self.n_blocks,
                    d_block=self.d_block,
                    dropout=self.dropout,
                    arch_type="tabm-packed",
                )
                optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
                criterion = nn.BCEWithLogitsLoss()
                x_t = torch.from_numpy(X_arr)
                y_t = torch.from_numpy(y_arr.reshape(-1, 1))
                self.model_.train()
                for _ in range(self.epochs):
                    epoch_losses = []
                    order = torch.randperm(len(X_arr))
                    for start in range(0, len(X_arr), self.batch_size):
                        stop = min(start + self.batch_size, len(X_arr))
                        batch = order[start:stop]
                        optimizer.zero_grad(set_to_none=True)
                        logits = self.model_(x_num=x_t[batch]).squeeze(-1)
                        if logits.ndim != 2:
                            raise RuntimeError(f"official TabM returned unexpected shape {tuple(logits.shape)}")
                        loss = criterion(logits, y_t[batch].expand_as(logits))
                        if not torch.isfinite(loss):
                            raise FloatingPointError("non-finite TabM training loss")
                        loss.backward()
                        optimizer.step()
                        epoch_losses.append(float(loss.detach().cpu().item()))
                    epoch_loss = float(np.mean(epoch_losses))
                    self.loss_history_.append(epoch_loss)
                    self.finite_loss_count_ += 1
                self.final_loss_ = self.loss_history_[-1]
            finally:
                torch.set_num_threads(old_threads)
        self.classes_ = np.asarray([0, 1])
        return self

    def _predict_p(self, X: Any) -> np.ndarray:
        if self.model_ is None or self.classes_ is None:
            raise RuntimeError("TabMClassifier must be fitted before prediction")
        X_arr = self._check_X(X)
        x_t = torch.from_numpy(X_arr)
        chunks = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(X_arr), self.batch_size):
                logits = self.model_(x_num=x_t[start : start + self.batch_size]).squeeze(-1)
                chunks.append(torch.sigmoid(logits).mean(dim=1).cpu().numpy())
        p = np.concatenate(chunks) if chunks else np.empty(0, dtype=float)
        if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
            raise FloatingPointError("TabM produced invalid event probabilities")
        return p

    def predict_proba(self, X: Any) -> np.ndarray:
        p = self._predict_p(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TabMAdapter(BaseMethodAdapter):
    """Benchmark adapter exposing official TabM as event-risk p."""

    name = "TABM"
    literature_reference = "Gorishniy et al., ICLR 2025, TabM"
    upstream_implementation = f"yandex-research/tabm commit {_TABM_COMMIT}"
    output_type = "event_probability_p"
    supports_arm2 = True
    requires_sensitive_at_predict = False

    def __init__(self, **kwargs: Any):
        self.classifier = TabMClassifier(**kwargs)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        A: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "TabMAdapter":
        self.classifier.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        return self.classifier.predict(X)

    def predict_proba(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        return self.classifier.predict_proba(X)

    def predict_event_probability(
        self, X: np.ndarray, A: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return validate_and_extract_positive_probabilities(self.classifier, X, pos_label=1)

    def predict_decision_proba(
        self, X: np.ndarray, A: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.predict_event_probability(X, A=A)
