"""Bounded FRAPPE/Best-of-both-worlds post-processing adapter.

This adapter deliberately keeps the upstream TensorFlow dependency optional.  The
official implementation is a binary, two-group MinDiff postprocessor: a frozen
F-trained predictor supplies ``probs`` and a C-trained additive logit correction
is fitted with separate Y=0 and Y=1 MMD losses. It returns event probability
scores (p); calibration is evaluated, not assumed from this output type.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .base import BaseMethodAdapter, NotSupportedError
from .estimators import make_estimator


class FrappeAdapter(BaseMethodAdapter):
    name = "FRAPPE_EO"
    literature_reference = "FRAPPÉ: A Group Fairness Framework for Post-Processing Everything (ICML 2024)"
    upstream_implementation = (
        "google-research/google-research/postproc_fairness@dbddc6626ce5363f48139d461bd8d131216d5722"
    )
    supports_arm2 = False
    requires_sensitive_at_predict = False
    output_type = "event_probability_p"

    def __init__(
        self,
        *,
        backbone: str = "LR",
        C: float = 1.0,
        random_state: int = 42,
        estimator_params: Optional[Mapping[str, Any]] = None,
        hidden_units: tuple[int, ...] = (16,),
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        mindiff_weight: float = 1.0,
        mmd_kernel_decay_length: float = 0.1,
        regularization_strength: float = 1.0,
    ) -> None:
        self.backbone = str(backbone)
        self.C = float(C)
        self.random_state = int(random_state)
        self.estimator_params = dict(estimator_params or {})
        self.hidden_units = tuple(int(v) for v in hidden_units)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.mindiff_weight = float(mindiff_weight)
        self.mmd_kernel_decay_length = float(mmd_kernel_decay_length)
        self.regularization_strength = float(regularization_strength)
        self.base_estimator_ = None
        self.postprocessor_ = None
        self.n_features_in_: Optional[int] = None
        self.compatibility_patch_ = None
        self.fit_manifest_: dict[str, Any] = {}

    @staticmethod
    def _binary_groups(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.unique(np.asarray(A))
        if values.size != 2:
            raise NotSupportedError(
                "FRAPPE supports exactly two sensitive groups; HISP7/Arm-002 is unsupported."
            )
        return values[0], values[1]

    @staticmethod
    def _as_matrix(X: Any) -> np.ndarray:
        values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("FRAPPE requires a non-empty two-dimensional F/C feature matrix")
        if not np.isfinite(values).all():
            raise ValueError("FRAPPE feature matrix contains non-finite values")
        return values

    @staticmethod
    def _as_binary(values: Any, name: str) -> np.ndarray:
        result = np.asarray(values).reshape(-1)
        if result.size == 0 or not np.isin(result, [0, 1]).all():
            raise ValueError(f"FRAPPE requires binary {name} values in {{0, 1}}")
        return result.astype(np.float32)

    def _require_tensorflow(self):
        try:
            import tensorflow as tf  # type: ignore
            from tensorflow_model_remediation import min_diff  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional platform stack
            raise NotSupportedError(
                "FRAPPE requires official tensorflow==2.14.0 and "
                "tensorflow-model-remediation==0.1.7.1 in an isolated environment"
            ) from exc
        # The pinned upstream source uses tf.log.  TF 2.x exposes the public,
        # equivalent tf.math.log only; keep this compatibility alias local to
        # the optional runtime and record it in the fit manifest.
        if not hasattr(tf, "log"):
            tf.log = tf.math.log
            self.compatibility_patch_ = "tf.log -> tf.math.log"
        tf.keras.utils.set_random_seed(self.random_state)
        try:
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            if hasattr(tf.config.experimental, "enable_op_determinism"):
                tf.config.experimental.enable_op_determinism()
        except RuntimeError:
            pass
        self.runtime_policy_ = {
            "tf_seed": self.random_state,
            "deterministic_ops": True,
            "inter_op_threads": 1,
            "intra_op_threads": 1,
        }
        return tf, min_diff

    def _create_postprocessor(self, tf, n_features: int):
        # Faithful transcription of official models_lib.create_postproc_model
        # for base_model=None, including EPSILON=1e-3 and additive logits.
        epsilon = 1e-3
        names = ["probs"] + [f"f_{i}" for i in range(n_features)]
        inputs = {name: tf.keras.Input(shape=(1,), name=name) for name in names}
        base_logits = tf.log(
            epsilon + inputs["probs"] / (epsilon + 1.0 - inputs["probs"])
        )
        x = tf.keras.layers.concatenate(
            [value for key, value in inputs.items() if key not in ("probs", "log_probs")],
            axis=1,
        )
        for index, units in enumerate(self.hidden_units):
            x = tf.keras.layers.Dense(units, activation=tf.nn.relu, name=f"pp_hidden_{index}")(x)
        multiplier = tf.keras.layers.Dense(1, activation=None, name="pp_multiplier")(x)
        pp_logits = tf.keras.layers.Add(name="pp_logits")([base_logits, multiplier])
        pp_outputs = tf.keras.layers.Activation("sigmoid", name="pp_outputs")(pp_logits)
        model = tf.keras.Model(inputs=inputs, outputs=pp_outputs)
        base_outputs = inputs["probs"]
        kl = tf.reduce_mean(
            base_outputs * tf.log(base_outputs / (pp_outputs + epsilon) + epsilon)
            + (1.0 - base_outputs)
            * tf.log((1.0 - base_outputs) / (1.0 - pp_outputs + epsilon) + epsilon)
        )
        model.add_loss(self.regularization_strength * kl)
        return model

    @staticmethod
    def _features(tf, X: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
        data = {"probs": probs.reshape(-1, 1).astype(np.float32)}
        data.update({f"f_{i}": X[:, i].reshape(-1, 1).astype(np.float32) for i in range(X.shape[1])})
        return data

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        A: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        *,
        X_calibration: Optional[np.ndarray] = None,
        y_calibration: Optional[np.ndarray] = None,
        A_calibration: Optional[np.ndarray] = None,
    ) -> "FrappeAdapter":
        if sample_weight is not None:
            raise NotSupportedError("The pinned FRAPPE MinDiff source has no survey-weight contract")
        if X_calibration is None or y_calibration is None or A_calibration is None:
            raise ValueError("FRAPPE requires explicit F training and C calibration partitions")
        X_f = self._as_matrix(X)
        y_f = self._as_binary(y, "y")
        X_c = self._as_matrix(X_calibration)
        y_c = self._as_binary(y_calibration, "y_calibration")
        A_c = np.asarray(A_calibration).reshape(-1)
        self._binary_groups(A_c)
        if len(X_f) != len(y_f) or len(X_c) != len(y_c) or len(X_c) != len(A_c):
            raise ValueError("FRAPPE F/C feature, label, and sensitive lengths must match")

        tf, tfmr = self._require_tensorflow()
        if not getattr(self, "_calibration_only", False):
            self.fit_base(X_f, y_f)
        from fairbias.prediction_contracts import validate_and_extract_positive_probabilities

        p_c = validate_and_extract_positive_probabilities(self.base_estimator_, X_c)
        self.n_features_in_ = X_c.shape[1]
        post = self._create_postprocessor(tf, self.n_features_in_)

        features = self._features(tf, X_c, p_c)
        original = (
            tf.data.Dataset.from_tensor_slices((features, y_c))
            .shuffle(5000, seed=self.random_state, reshuffle_each_iteration=True)
            .batch(self.batch_size)
        )
        # Official eqodds construction uses independent Y=0 and Y=1 MMD terms.
        sensitive: dict[str, Any] = {}
        nonsensitive: dict[str, Any] = {}
        for key, target in (("neg", 0.0), ("pos", 1.0)):
            mask = y_c == target
            groups = np.unique(A_c[mask])
            if groups.size != 2 or any(np.sum(mask & (A_c == group)) == 0 for group in groups):
                raise NotSupportedError(
                    f"FRAPPE C partition must contain both sensitive groups for y={int(target)}"
                )
            left = mask & (A_c == groups[0])
            right = mask & (A_c == groups[1])
            sensitive[key] = tf.data.Dataset.from_tensor_slices(
                (self._features(tf, X_c[left], p_c[left]), y_c[left])
            ).shuffle(5000, seed=self.random_state, reshuffle_each_iteration=True).repeat().batch(
                self.batch_size, drop_remainder=True
            )
            nonsensitive[key] = tf.data.Dataset.from_tensor_slices(
                (self._features(tf, X_c[right], p_c[right]), y_c[right])
            ).shuffle(5000, seed=self.random_state, reshuffle_each_iteration=True).repeat().batch(
                self.batch_size, drop_remainder=True
            )
        try:
            min_diff_data = tfmr.keras.utils.build_min_diff_dataset(
                sensitive_group_dataset=sensitive,
                nonsensitive_group_dataset=nonsensitive,
            )
            packed = tfmr.keras.utils.pack_min_diff_data(
                original_dataset=original,
                min_diff_dataset=min_diff_data,
            )
            losses = {
                    key: tfmr.losses.MMDLoss(
                    tfmr.losses.GaussianKernel(
                        kernel_length=self.mmd_kernel_decay_length
                    )
                )
                for key in ("neg", "pos")
            }
            model = tfmr.keras.models.min_diff_model.MinDiffModel(
                original_model=post, loss=losses, loss_weight=self.mindiff_weight
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate),
                loss=None,
            )
            model.fit(packed, epochs=self.epochs, verbose=0)
        except Exception as exc:
            raise NotSupportedError(
                "The installed tensorflow-model-remediation runtime does not expose "
                "the pinned eqodds MinDiff API"
            ) from exc
        self.postprocessor_ = model
        self.fit_manifest_ = {
            "official_source_commit": self.upstream_implementation.rsplit("@", 1)[1],
            "base_partition": "F",
            "correction_partition": "C",
            "output_type": self.output_type,
            "sensitive_groups": [str(v) for v in np.unique(A_c)],
            "regularizer": "kl",
            "epsilon": 1e-3,
            "compatibility_patch": self.compatibility_patch_,
            "runtime_policy": getattr(self, "runtime_policy_", None),
        }
        return self

    def fit_base(self, X_F: np.ndarray, y_F: np.ndarray) -> "FrappeAdapter":
        """Fit and retain the frozen F base predictor."""
        X_f = self._as_matrix(X_F)
        y_f = self._as_binary(y_F, "y_F")
        self._fit_X_ = X_f.copy()
        self._fit_y_ = y_f.copy()
        self.base_estimator_ = make_estimator(
            self.backbone, C=self.C, random_state=self.random_state, estimator_params=self.estimator_params
        )
        self.base_estimator_.fit(X_f, y_f.astype(int))
        return self

    def calibrate(self, X_C: np.ndarray, y_C: np.ndarray, A_C: np.ndarray) -> "FrappeAdapter":
        """Apply the C correction using the already declared F partition."""
        if not hasattr(self, "_fit_X_") or self.base_estimator_ is None:
            raise RuntimeError("fit_base must run before calibrate")
        self._calibration_only = True
        try:
            return self.fit(
                self._fit_X_, self._fit_y_, np.zeros(len(self._fit_y_), dtype=int),
                X_calibration=X_C, y_calibration=y_C, A_calibration=A_C,
            )
        finally:
            self._calibration_only = False

    def predict_decision_proba(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        if self.postprocessor_ is None or self.n_features_in_ is None:
            raise RuntimeError("FRAPPE adapter is not fitted")
        values = self._as_matrix(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("FRAPPE prediction feature count differs from fitted F/C schema")
        tf, _ = self._require_tensorflow()
        from fairbias.prediction_contracts import validate_and_extract_positive_probabilities

        p = validate_and_extract_positive_probabilities(self.base_estimator_, values)
        out = np.asarray(self.postprocessor_(self._features(tf, values, p), training=False)).reshape(-1)
        if out.shape[0] != values.shape[0] or not np.isfinite(out).all() or np.any((out < 0.0) | (out > 1.0)):
            raise ValueError("FRAPPE event probability p is not finite or lies outside [0, 1]")
        return out.astype(float)

    def predict(self, X: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        return (self.predict_decision_proba(X, A=A) >= 0.5).astype(int)

    def save_state(self, path: str | Path) -> None:
        """Persist the fitted F estimator, C postprocessor weights, and contract."""
        if self.postprocessor_ is None or self.base_estimator_ is None or self.n_features_in_ is None:
            raise RuntimeError("FRAPPE adapter is not fitted")
        prefix = Path(path)
        with open(prefix.with_suffix(".base.pkl"), "wb") as handle:
            pickle.dump(self.base_estimator_, handle, protocol=pickle.HIGHEST_PROTOCOL)
        post_model = getattr(self.postprocessor_, "original_model", self.postprocessor_)
        post_model.save_weights(prefix.with_suffix(".weights.h5"))
        payload = {
            "n_features_in": self.n_features_in_,
            "fit_manifest": self.fit_manifest_,
            "backbone": self.backbone,
            "hidden_units": list(self.hidden_units),
        }
        prefix.with_suffix(".json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def load_state(self, path: str | Path) -> "FrappeAdapter":
        """Reload a state written by :meth:`save_state` in the same runtime."""
        prefix = Path(path)
        tf, _ = self._require_tensorflow()
        with open(prefix.with_suffix(".base.pkl"), "rb") as handle:
            self.base_estimator_ = pickle.load(handle)
        payload = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
        self.n_features_in_ = int(payload["n_features_in"])
        self.fit_manifest_ = dict(payload["fit_manifest"])
        post_model = self._create_postprocessor(tf, self.n_features_in_)
        post_model.load_weights(prefix.with_suffix(".weights.h5"))
        self.postprocessor_ = post_model
        return self

    def __getstate__(self) -> dict[str, Any]:
        """Serialize without pickling TensorFlow/Keras objects."""
        state = dict(self.__dict__)
        post = state.pop("postprocessor_", None)
        if post is not None:
            original = getattr(post, "original_model", post)
            state["_serialized_post_weights"] = [np.asarray(w) for w in original.get_weights()]
        else:
            state["_serialized_post_weights"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        weights = state.pop("_serialized_post_weights", None)
        self.__dict__.update(state)
        self.postprocessor_ = None
        if weights is not None:
            if self.n_features_in_ is None:
                raise ValueError("Serialized FRAPPE state is missing feature schema")
            tf, _ = self._require_tensorflow()
            post = self._create_postprocessor(tf, int(self.n_features_in_))
            post.set_weights(weights)
            self.postprocessor_ = post
