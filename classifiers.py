"""
Classifiers Module

This module provides various classification models for tabular data, including:
- sklearn-style wrappers for pytorch_tabular models
- Pure PyTorch implementations of CTR models (DeepFM, Wide&Deep, DCN, xDeepFM)
- Custom neural network layers (CrossNetwork, CIN)

These classifiers follow sklearn's API with fit(), predict(), and predict_proba() methods.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union, Any

# MODELS
# scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# sklearn base (optional)
try:
    from sklearn.base import BaseEstimator
except ImportError:
    BaseEstimator = object

# pytorch_tabular (optional)
try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models import (
        TabTransformerConfig,
        TabNetModelConfig,
        AutoIntConfig,
        CategoryEmbeddingModelConfig,
        FTTransformerConfig,
        GatedAdditiveTreeEnsembleConfig,
        DANetConfig,
        GANDALFConfig,
        NodeConfig,
    )
    PYTORCH_TABULAR_AVAILABLE = True
except ImportError:
    PYTORCH_TABULAR_AVAILABLE = False

# PyTorch for CTR models (DeepFM, Wide&Deep, DCN, xDeepFM)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TabularEstimator(BaseEstimator):
    """
    A thin sklearn-style wrapper around pytorch_tabular.TabularModel.

    - Supports multiple backbone models (TabNet, TabTransformer, CategoryEmbedding,
      FTTransformer, GANDALF, NODE, etc.)
    - Exposes fit(X, y), predict(X), predict_proba(X) similar to sklearn classifiers.
    """

    _MODEL_MAP = {
        "tabtransformer": TabTransformerConfig,
        "tab_transformer": TabTransformerConfig,
        "tabnet": TabNetModelConfig,
        "autoint": AutoIntConfig,
        "category_embedding": CategoryEmbeddingModelConfig,
        "fttransformer": FTTransformerConfig,
        "ft_transformer": FTTransformerConfig,
        "gate": GatedAdditiveTreeEnsembleConfig,
        "gated_additive_tree": GatedAdditiveTreeEnsembleConfig,
        "danet": DANetConfig,
        "gandalf": GANDALFConfig,
        "node": NodeConfig,
    }

    def __init__(
        self,
        task: str = "classification",
        model_name: str = "category_embedding",
        target_col: str = "target",
        categorical_cols: Optional[List[str]] = None,
        continuous_cols: Optional[List[str]] = None,
        data_config: Optional[DataConfig] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        trainer_config: Optional[TrainerConfig] = None,
        data_config_kwargs: Optional[Dict] = None,
        model_config_kwargs: Optional[Dict] = None,
        trainer_config_kwargs: Optional[Dict] = None,
        optimizer_config_kwargs: Optional[Dict] = None,
    ):
        self.task = task  # "classification" or "regression"
        self.model_name = model_name
        self.target_col = target_col

        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols

        self._user_data_config = data_config
        self._user_optimizer_config = optimizer_config
        self._user_trainer_config = trainer_config

        self.data_config_kwargs = data_config_kwargs or {}
        self.model_config_kwargs = model_config_kwargs or {}
        self.trainer_config_kwargs = trainer_config_kwargs or {}
        self.optimizer_config_kwargs = optimizer_config_kwargs or {}

        self.model_: Optional[TabularModel] = None
        self._fitted_: bool = False

    # -------------------- internal helpers -------------------- #
    def _ensure_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)

    def _infer_columns_if_needed(self, X: pd.DataFrame):
        # If user did not specify columns, infer by dtype
        if self.continuous_cols is None or self.categorical_cols is None:
            numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
            all_cols = X.columns.tolist()
            if self.target_col in all_cols:
                all_cols.remove(self.target_col)
            if self.continuous_cols is None:
                self.continuous_cols = [c for c in numeric_cols if c in all_cols]
            if self.categorical_cols is None:
                self.categorical_cols = [c for c in all_cols if c not in self.continuous_cols]

    def _get_model_config_class(self):
        key = self.model_name.lower()
        if key not in self._MODEL_MAP:
            raise ValueError(
                f"Unknown tabular model_name={self.model_name}. "
                f"Available: {list(self._MODEL_MAP.keys())}"
            )
        return self._MODEL_MAP[key]

    def _build_configs(self, train_df: pd.DataFrame):
        # DataConfig
        if self._user_data_config is not None:
            data_config = self._user_data_config
        else:
            self._infer_columns_if_needed(train_df)
            data_config = DataConfig(
                target=[self.target_col],
                continuous_cols=self.continuous_cols,
                categorical_cols=self.categorical_cols,
                **self.data_config_kwargs,
            )

        # ModelConfig
        model_cls = self._get_model_config_class()
        model_config = model_cls(
            task=self.task,
            **self.model_config_kwargs,
        )

        # OptimizerConfig
        if self._user_optimizer_config is not None:
            optimizer_config = self._user_optimizer_config
        else:
            optimizer_config = OptimizerConfig(**self.optimizer_config_kwargs)

        # TrainerConfig
        if self._user_trainer_config is not None:
            trainer_config = self._user_trainer_config
        else:
            # 防止 rich 进度条在某些 IDE/终端产生 IndexError
            if "progress_bar" not in self.trainer_config_kwargs:
                self.trainer_config_kwargs["progress_bar"] = "none"
            trainer_config = TrainerConfig(**self.trainer_config_kwargs)

        return data_config, model_config, optimizer_config, trainer_config

    # -------------------- sklearn-like API -------------------- #
    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
    ):
        """
        Fit the underlying TabularModel.

        Parameters
        ----------
        X : array-like or DataFrame
            Training features
        y : array-like
            Training labels
        X_val, y_val : optional validation set
        """
        train_X = self._ensure_dataframe(X)
        train_X[self.target_col] = y

        if X_val is not None and y_val is not None:
            val_X = self._ensure_dataframe(X_val)
            val_X[self.target_col] = y_val
        else:
            val_X = None

        data_config, model_config, optimizer_config, trainer_config = self._build_configs(train_X)

        self.model_ = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )

        self.model_.fit(train=train_X, validation=val_X)
        self._fitted_ = True
        return self

    def predict(self, X):
        """
        Return point predictions.

        - For classification: class labels
        - For regression: continuous predictions
        """
        if not self._fitted_:
            raise RuntimeError("TabularEstimator is not fitted yet. Call `fit` first.")

        X_df = self._ensure_dataframe(X)
        pred_df = self.model_.predict(X_df)

        cols = list(pred_df.columns)

        if self.task == "classification":
            # 1) Standard case: "prediction"
            if "prediction" in pred_df.columns:
                return pred_df["prediction"].values

            # 2) Common alt: "<target>_prediction"
            alt_col = f"{self.target_col}_prediction"
            if alt_col in pred_df.columns:
                return pred_df[alt_col].values

            # 3) Fallback: if there is exactly one *_prediction column
            pred_cols = [c for c in pred_df.columns if c.endswith("_prediction")]
            if len(pred_cols) == 1:
                return pred_df[pred_cols[0]].values

            # 4) Last resort: derive from probability columns via argmax
            prob_cols = [c for c in pred_df.columns if c.endswith("_probability")]
            if prob_cols:
                prob = pred_df[prob_cols].values
                class_idx = prob.argmax(axis=1)
                return class_idx

            raise RuntimeError(
                f"Cannot infer prediction column from pytorch_tabular output. "
                f"Columns: {cols}"
            )

        elif self.task == "regression":
            col = f"{self.target_col}_prediction"
            if col in pred_df.columns:
                return pred_df[col].values

            pred_cols = [c for c in pred_df.columns if c.endswith("_prediction")]
            if len(pred_cols) == 1:
                return pred_df[pred_cols[0]].values

            raise RuntimeError(
                f"Cannot find regression prediction column in pytorch_tabular output. "
                f"Columns: {cols}"
            )

        else:
            raise ValueError(f"Unknown task={self.task}, must be 'classification' or 'regression'.")

    def predict_proba(self, X):
        """
        Return class probabilities for classification.
        """
        if self.task != "classification":
            raise RuntimeError("predict_proba is only available for classification task.")

        if not self._fitted_:
            raise RuntimeError("TabularEstimator is not fitted yet. Call `fit` first.")

        X_df = self._ensure_dataframe(X)
        pred_df = self.model_.predict(X_df)

        prob_cols = [c for c in pred_df.columns if c.endswith("_probability")]
        if prob_cols:
            return pred_df[prob_cols].values

        # Fallback: some setups emit only a single "<target>_prediction" as positive class prob
        alt_col = f"{self.target_col}_prediction"
        if alt_col in pred_df.columns:
            p_pos = pred_df[alt_col].values
            p_neg = 1.0 - p_pos
            return np.vstack([p_neg, p_pos]).T

        raise RuntimeError(
            f"No probability columns found in pytorch_tabular output. "
            f"Columns: {list(pred_df.columns)}"
        )


# =====================================================================
#              Pure PyTorch CTR Models: DeepFM / Wide&Deep / DCN / xDeepFM
# =====================================================================

class CrossNetwork(nn.Module):
    """
    DCN 的 Cross Network:
    x_{l+1} = x0 * (w_l^T x_l) + b_l + x_l
    """
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.ws = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1) * 0.01)
            for _ in range(num_layers)
        ])
        self.bs = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])

    def forward(self, x0, x):
        # x0, x: [B, D]
        for i in range(self.num_layers):
            xw = torch.matmul(x, self.ws[i])  # [B, 1]
            cross = x0 * xw + self.bs[i] + x  # [B, D]
            x = cross
        return x  # [B, D]


class CIN(nn.Module):
    """
    Compressed Interaction Network for xDeepFM.

    输入:
        X0: [B, F, D]，F = field_num，D = embedding_dim

    每一层 l:
        - 输入 Xk: [B, H_prev, D]
        - 与 X0 做外积得到 [B, F, H_prev, D]
        - 在 (F * H_prev) 维上用线性变换压缩到 H_l
        - 输出 X_{k+1}: [B, H_l, D]

    输出:
        concat 所有层在 embedding 维度上和后的向量: [B, sum(H_l)]
    """
    def __init__(self, field_num: int, embedding_dim: int, layer_sizes: List[int]):
        super().__init__()
        self.field_num = field_num
        self.embedding_dim = embedding_dim
        self.layer_sizes = layer_sizes

        self.W = nn.ParameterList()
        prev_field_size = field_num
        for h in layer_sizes:
            w = nn.Parameter(torch.randn(prev_field_size * field_num, h) * 0.01)
            self.W.append(w)
            prev_field_size = h

    def forward(self, X0: torch.Tensor) -> torch.Tensor:
        """
        X0: [B, F, D]
        返回: [B, sum(H_l)]
        """
        B, F, D = X0.size()
        Xk = X0
        outputs = []

        for l, h in enumerate(self.layer_sizes):
            B, H_prev, D = Xk.size()

            # 外积: X0: [B, F, D], Xk: [B, H_prev, D]
            # -> Z: [B, F, H_prev, D]
            Z = torch.einsum('bfd,bhd->bfhd', X0, Xk)
            # reshape: [B, F*H_prev, D]
            Z = Z.view(B, F * H_prev, D)

            W = self.W[l]  # [F*H_prev, H_l]

            # 压缩: 在 F*H_prev 上乘 W
            # Z^T: [B, D, F*H_prev] @ W: [F*H_prev, H_l] -> [B, D, H_l]
            ZW = torch.matmul(Z.transpose(1, 2), W)  # [B, D, H_l]

            # X_{k+1}: [B, H_l, D]
            Xk = ZW.permute(0, 2, 1)  # [B, H_l, D]
            outputs.append(Xk)

        # 每层在 embedding 维度 D 上求和: [B, H_l]
        outs = [x.sum(dim=2) for x in outputs]
        # concat 所有层: [B, sum(H_l)]
        cin_out = torch.cat(outs, dim=1)
        return cin_out


class CTRNet(nn.Module):
    """
    统一的 CTR 网络:
    - model_type = "deepfm":   wide + FM + deep
    - model_type = "widedeep": wide + deep
    - model_type = "dcn":      wide + cross + deep
    - model_type = "xdeepfm":  wide + FM + CIN + deep
    """
    def __init__(
        self,
        model_type: str,
        num_categorical: int,
        num_continuous: int,
        vocab_sizes: List[int],
        embedding_dim: int = 16,
        hidden_units: Optional[List[int]] = None,
        dropout: float = 0.0,
        cross_layers: int = 2,
        cin_layer_sizes: Optional[List[int]] = None,
    ):
        super().__init__()
        assert model_type in {"deepfm", "widedeep", "dcn", "xdeepfm"}
        self.model_type = model_type
        self.num_categorical = num_categorical
        self.num_continuous = num_continuous
        self.vocab_sizes = vocab_sizes
        self.embedding_dim = embedding_dim

        if hidden_units is None:
            hidden_units = [128, 64]
        if cin_layer_sizes is None:
            cin_layer_sizes = [16, 16]

        # Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(vs, embedding_dim) for vs in vocab_sizes
        ])

        # First-order (wide)
        self.first_order_embeddings = nn.ModuleList([
            nn.Embedding(vs, 1) for vs in vocab_sizes
        ])
        if num_continuous > 0:
            self.first_order_dense = nn.Linear(num_continuous, 1)
        else:
            self.first_order_dense = None

        # Deep MLP
        deep_input_dim = num_categorical * embedding_dim + num_continuous
        mlp_layers = []
        in_dim = deep_input_dim
        for h in hidden_units:
            mlp_layers.append(nn.Linear(in_dim, h))
            mlp_layers.append(nn.ReLU())
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
            in_dim = h
        self.deep_mlp = nn.Sequential(*mlp_layers)
        self.deep_out_dim = in_dim

        # Cross (DCN)
        if model_type == "dcn":
            self.cross_net = CrossNetwork(deep_input_dim, num_layers=cross_layers)
        else:
            self.cross_net = None

        # CIN (xDeepFM)
        if model_type == "xdeepfm":
            self.cin = CIN(field_num=num_categorical,
                           embedding_dim=embedding_dim,
                           layer_sizes=cin_layer_sizes)
            self.cin_out_layer = nn.Linear(sum(cin_layer_sizes), 1)
        else:
            self.cin = None
            self.cin_out_layer = None

        # 部分开关
        self.fm_enabled = (model_type in {"deepfm", "xdeepfm"})
        self.deep_enabled = True
        self.wide_enabled = True
        self.cross_enabled = (model_type == "dcn")
        self.cin_enabled = (model_type == "xdeepfm")

        # Deep 输出
        self.deep_out_layer = nn.Linear(self.deep_out_dim, 1)
        # Cross 输出
        if self.cross_enabled:
            self.cross_out_layer = nn.Linear(deep_input_dim, 1)
        else:
            self.cross_out_layer = None

    def forward(self, cat_inputs: torch.Tensor, cont_inputs: Optional[torch.Tensor] = None):
        """
        cat_inputs: [B, num_categorical]   (Long)
        cont_inputs: [B, num_continuous]   (Float) or None
        """
        device = cat_inputs.device
        batch_size = cat_inputs.size(0)

        # Embedding lookup
        embed_list = []
        first_order_list = []
        for i, emb in enumerate(self.embeddings):
            e = emb(cat_inputs[:, i])  # [B, k]
            embed_list.append(e)
        for i, emb1 in enumerate(self.first_order_embeddings):
            e1 = emb1(cat_inputs[:, i])  # [B, 1]
            first_order_list.append(e1)

        cat_embeddings = torch.stack(embed_list, dim=1)      # [B, F, k]
        first_order_cat = torch.cat(first_order_list, dim=1) # [B, F]

        if cont_inputs is not None and self.first_order_dense is not None:
            first_order_cont = self.first_order_dense(cont_inputs)  # [B, 1]
        else:
            first_order_cont = torch.zeros(batch_size, 1, device=device)

        # Wide
        wide_out = first_order_cat.sum(dim=1, keepdim=True) + first_order_cont  # [B, 1]

        # FM
        if self.fm_enabled:
            sum_emb = cat_embeddings.sum(dim=1)               # [B, k]
            sum_emb_square = sum_emb * sum_emb                # [B, k]
            square_emb = cat_embeddings * cat_embeddings      # [B, F, k]
            square_emb_sum = square_emb.sum(dim=1)            # [B, k]
            fm_part = 0.5 * (sum_emb_square - square_emb_sum) # [B, k]
            fm_out = fm_part.sum(dim=1, keepdim=True)         # [B, 1]
        else:
            fm_out = torch.zeros(batch_size, 1, device=device)

        # Deep input
        if cont_inputs is not None and self.num_continuous > 0:
            deep_input = torch.cat(
                [cat_embeddings.view(batch_size, -1), cont_inputs], dim=1
            )
        else:
            deep_input = cat_embeddings.view(batch_size, -1)

        deep_feat = self.deep_mlp(deep_input)        # [B, H]
        deep_out = self.deep_out_layer(deep_feat)    # [B, 1]

        # Cross (DCN)
        if self.cross_enabled and self.cross_net is not None:
            cross_feat = self.cross_net(deep_input, deep_input)
            cross_out = self.cross_out_layer(cross_feat)  # [B, 1]
        else:
            cross_out = torch.zeros(batch_size, 1, device=device)

        # CIN (xDeepFM)
        if self.cin_enabled and self.cin is not None:
            cin_feat = self.cin(cat_embeddings)            # [B, sum(H_l)]
            cin_out = self.cin_out_layer(cin_feat)         # [B, 1]
        else:
            cin_out = torch.zeros(batch_size, 1, device=device)

        logits = wide_out + fm_out + deep_out + cross_out + cin_out
        return logits.squeeze(-1)  # [B]


class TorchCTRClassifier(BaseEstimator):
    """
    sklearn 风格的 PyTorch CTR 模型封装器。

    支持:
    - model_type="deepfm"
    - model_type="widedeep"
    - model_type="dcn"
    - model_type="xdeepfm"
    """

    def __init__(
        self,
        model_type: str = "deepfm",
        categorical_cols: Optional[List[str]] = None,
        continuous_cols: Optional[List[str]] = None,
        embedding_dim: int = 16,
        hidden_units: Optional[List[int]] = None,
        dropout: float = 0.0,
        cross_layers: int = 2,
        cin_layer_sizes: Optional[List[int]] = None,
        lr: float = 1e-3,
        batch_size: int = 512,
        epochs: int = 5,
        weight_decay: float = 0.0,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        assert model_type in {"deepfm", "widedeep", "dcn", "xdeepfm"}
        self.model_type = model_type
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.cross_layers = cross_layers
        self.cin_layer_sizes = cin_layer_sizes
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.verbose = verbose

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_: Optional[CTRNet] = None
        self.cat_maps_: Dict[str, Dict[Any, int]] = {}
        self.cat_unknown_idx_: Dict[str, int] = {}
        self.cont_mean_: Dict[str, float] = {}
        self.cont_std_: Dict[str, float] = {}
        self.fitted_: bool = False

    # --------- 特征预处理 ---------
    def _ensure_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)

    def _infer_cols_if_needed(self, X: pd.DataFrame):
        # 如果用户没指定，就用简单 heuristic：
        # 非数值列 + 低基数整数列 -> 类别；其他 -> 数值
        if self.categorical_cols is None and self.continuous_cols is None:
            all_cols = X.columns.tolist()
            num_cols = X.select_dtypes(include=["number"]).columns.tolist()

            cat_cols = []
            cont_cols = []

            for col in all_cols:
                if col not in num_cols:
                    cat_cols.append(col)
                else:
                    if pd.api.types.is_integer_dtype(X[col]):
                        nunique = X[col].nunique(dropna=True)
                        if nunique <= 20:     # 阈值可以调整
                            cat_cols.append(col)
                        else:
                            cont_cols.append(col)
                    else:
                        cont_cols.append(col)

            self.categorical_cols = cat_cols
            self.continuous_cols = cont_cols
        elif self.categorical_cols is None:
            num_cols = X.select_dtypes(include=["number"]).columns.tolist()
            self.categorical_cols = [c for c in X.columns if c not in num_cols]
            if self.continuous_cols is None:
                self.continuous_cols = num_cols
        elif self.continuous_cols is None:
            self.continuous_cols = [c for c in X.columns if c not in self.categorical_cols]

    def _fit_encoders(self, X: pd.DataFrame):
        self._infer_cols_if_needed(X)

        # 类别编码
        self.cat_maps_ = {}
        self.cat_unknown_idx_ = {}
        for col in self.categorical_cols:
            vals = X[col].astype("category")
            categories = vals.cat.categories.tolist()
            mapping = {cat: i for i, cat in enumerate(categories)}
            unk_idx = len(categories)
            self.cat_maps_[col] = mapping
            self.cat_unknown_idx_[col] = unk_idx

        # 数值标准化
        self.cont_mean_ = {}
        self.cont_std_ = {}
        for col in self.continuous_cols:
            col_vals = X[col].astype(float)
            mean = col_vals.mean()
            std = col_vals.std()
            if std == 0 or np.isnan(std):
                std = 1.0
            self.cont_mean_[col] = float(mean)
            self.cont_std_[col] = float(std)

    def _transform_X(self, X: pd.DataFrame):
        X = self._ensure_dataframe(X)
        self._infer_cols_if_needed(X)

        # 类别 -> index
        cat_arrays = []
        for col in self.categorical_cols:
            mapping = self.cat_maps_[col]
            unk_idx = self.cat_unknown_idx_[col]
            vals = X[col].fillna("__NA__").tolist()
            idxs = [mapping.get(v, unk_idx) for v in vals]
            cat_arrays.append(np.array(idxs, dtype="int64"))
        if cat_arrays:
            cat_mat = np.vstack(cat_arrays).T  # [B, F_cat]
        else:
            cat_mat = np.zeros((len(X), 1), dtype="int64")

        # 数值 -> 标准化
        cont_arrays = []
        for col in self.continuous_cols:
            vals = X[col].astype(float).fillna(self.cont_mean_[col]).values
            vals = (vals - self.cont_mean_[col]) / self.cont_std_[col]
            cont_arrays.append(vals)
        if cont_arrays:
            cont_mat = np.vstack(cont_arrays).T.astype("float32")  # [B, F_cont]
        else:
            cont_mat = np.zeros((len(X), 0), dtype="float32")

        return cat_mat, cont_mat

    # --------- sklearn API ---------

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
    ):
        X_df = self._ensure_dataframe(X)
        y_arr = np.asarray(y).astype("float32").ravel()

        # 拟合编码器
        self._fit_encoders(X_df)

        # 转换
        cat_mat, cont_mat = self._transform_X(X_df)
        cat_tensor = torch.from_numpy(cat_mat).long()
        cont_tensor = torch.from_numpy(cont_mat).float()
        y_tensor = torch.from_numpy(y_arr).float()

        num_categorical = cat_tensor.shape[1]
        num_continuous = cont_tensor.shape[1]
        vocab_sizes = []
        for col in self.categorical_cols:
            vocab_sizes.append(len(self.cat_maps_[col]) + 1)

        model = CTRNet(
            model_type=self.model_type,
            num_categorical=num_categorical,
            num_continuous=num_continuous,
            vocab_sizes=vocab_sizes,
            embedding_dim=self.embedding_dim,
            hidden_units=self.hidden_units,
            dropout=self.dropout,
            cross_layers=self.cross_layers,
            cin_layer_sizes=self.cin_layer_sizes,
        ).to(self.device)

        self.model_ = model

        dataset = TensorDataset(cat_tensor, cont_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for cat_b, cont_b, y_b in loader:
                cat_b = cat_b.to(self.device)
                cont_b = cont_b.to(self.device)
                y_b = y_b.to(self.device)

                optimizer.zero_grad()
                logits = self.model_(cat_b, cont_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y_b)

            avg_loss = total_loss / len(dataset)
            if self.verbose:
                msg = f"[{self.model_type}] epoch {epoch+1}/{self.epochs}, loss={avg_loss:.4f}"
                if X_val is not None and y_val is not None:
                    val_loss = self._eval_loss(X_val, y_val, criterion)
                    msg += f", val_loss={val_loss:.4f}"
                print(msg)

        self.fitted_ = True
        return self

    def _eval_loss(self, X_val, y_val, criterion):
        self.model_.eval()
        with torch.no_grad():
            Xv = self._ensure_dataframe(X_val)
            yv = np.asarray(y_val).astype("float32").ravel()
            cat_mat, cont_mat = self._transform_X(Xv)
            cat_tensor = torch.from_numpy(cat_mat).long().to(self.device)
            cont_tensor = torch.from_numpy(cont_mat).float().to(self.device)
            y_tensor = torch.from_numpy(yv).float().to(self.device)
            logits = self.model_(cat_tensor, cont_tensor)
            loss = criterion(logits, y_tensor)
        self.model_.train()
        return loss.item()

    def _predict_logits(self, X):
        if not self.fitted_ or self.model_ is None:
            raise RuntimeError("TorchCTRClassifier is not fitted yet. Call `fit` first.")
        self.model_.eval()
        with torch.no_grad():
            X_df = self._ensure_dataframe(X)
            cat_mat, cont_mat = self._transform_X(X_df)
            cat_tensor = torch.from_numpy(cat_mat).long().to(self.device)
            cont_tensor = torch.from_numpy(cont_mat).float().to(self.device)
            logits = self.model_(cat_tensor, cont_tensor)
        self.model_.train()
        return logits.cpu().numpy()

    def predict_proba(self, X):
        logits = self._predict_logits(X)
        probs_pos = 1.0 / (1.0 + np.exp(-logits))
        probs_neg = 1.0 - probs_pos
        return np.vstack([probs_neg, probs_pos]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype("int64")
