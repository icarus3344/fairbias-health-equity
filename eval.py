"""
Evaluator Module

This module provides the Evaluator class for model training, prediction, and evaluation
of fairness and accuracy metrics. The class supports various classification models and
provides methods to calculate bias concentration (epsilon), fairness metrics, and
accuracy metrics.

Key features:
- Model initialization based on configuration parameters
- Support for different types of classifiers (scikit-learn, XGBoost, LightGBM, etc.)
- Training strategy with full dataset for first run and sampling for subsequent runs
- Calculation of various fairness metrics (BNC, BPC, CUAE, EOpp, etc.)
- Calculation of accuracy metrics (ACC, F1, Recall, Precision)
- Calculation of bias concentration metric (epsilon)

Usage:
    evaluator = Evaluator(label_O='SEX', label_Y='default payment next month',
                          cate_attrs=[], num_attrs=[])
    metrics = evaluator.evaluate(X_train, Y_train, O_train, X_test, Y_test, O_test)
    epsilon = evaluator.calculate_epsilon(X, Y, O, num_attrs, cate_attrs)
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

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

# Boosting
import xgboost as xgb
import lightgbm as lgb
import catboost

# Import custom classifiers
from classifiers import (
    TabularEstimator,
    TorchCTRClassifier,
    PYTORCH_TABULAR_AVAILABLE
)

# Import configuration parameters
from config import (
    SEED,
    VERBOSE,

    PARAMS_MAIN_CLASSIFIER,
    PARAMS_MAIN_TRAINING_RATE,

    PARAMS_EVAL_NUM,
    PARAMS_EVAL_CAT,
    PARAMS_EVAL_SUM,

    PARAMS_EVAL_H_ORDER,
    PARAMS_EVAL_SCALE,
    PARAMS_EVAL_NORM,
    PARAMS_EVAL_DIST_METRIC,

    PARAMS_EVAL_METRIC_FAIRNESS,
    PARAMS_EVAL_METRIC_ACCURACY,

    PARAMS_EVAL_MAX_COMPONENTS,
    PARAMS_EVAL_SLOPE_THRESHOLD
)

import warnings
warnings.filterwarnings("ignore")


class Evaluator:
    """
    Evaluator class for model training, prediction, and evaluation of metrics.
    
    This class provides functionality to train classification models, make predictions,
    and calculate various fairness and accuracy metrics. It also includes methods to
    compute bias concentration (epsilon) to measure the distribution of bias across features.
    """
    
    def __init__(self, label_O='SEX', label_Y='default payment next month', cate_attrs=[], num_attrs=[]):
        """
        Initialize the Evaluator class.
        
        Args:
            label_O (str or list): Protected attribute(s)
            label_Y (str): Target attribute
            cate_attrs (list): List of categorical attribute names
            num_attrs (list): List of numerical attribute names
        """
        self.label_O = label_O
        self.label_Y = label_Y
        self.cate_attrs = cate_attrs
        self.num_attrs = num_attrs

        self.h_order = PARAMS_EVAL_H_ORDER
        self.first_train = True
        self.random_state = SEED
        self.results_df = None
        self.metrics_results = {}

        self.model = self._create_model()
        

    def _create_model(self):
        """
        Create model based on PARAMS_MAIN_CLASSIFIER from config.
        
        Returns:
            model: Initialized classification model
        """
        if VERBOSE:
            print(f"Using classifier: {PARAMS_MAIN_CLASSIFIER}")

        if PARAMS_MAIN_CLASSIFIER == 'LR':
            return LogisticRegression(random_state=SEED)
        elif PARAMS_MAIN_CLASSIFIER == 'DT':
            return DecisionTreeClassifier(random_state=SEED)
        elif PARAMS_MAIN_CLASSIFIER == 'KNN':
            return KNeighborsClassifier()
        elif PARAMS_MAIN_CLASSIFIER == 'GBDT':
            return GradientBoostingClassifier(random_state=SEED)
        elif PARAMS_MAIN_CLASSIFIER == 'ADABoost':
            return AdaBoostClassifier(random_state=SEED)
        elif PARAMS_MAIN_CLASSIFIER == 'NB':
            return GaussianNB()
        elif PARAMS_MAIN_CLASSIFIER == 'SVM':
            return SVC(random_state=SEED, max_iter=100, probability=True)
        elif PARAMS_MAIN_CLASSIFIER == 'MLP':
            return MLPClassifier(random_state=SEED)
        elif PARAMS_MAIN_CLASSIFIER == 'XGBoost':
            return xgb.XGBClassifier(random_state=SEED)
        elif PARAMS_MAIN_CLASSIFIER == 'RF':
            return RandomForestClassifier(random_state=SEED)
        elif PARAMS_MAIN_CLASSIFIER == 'LGBM':
            return lgb.LGBMClassifier(random_state=SEED, force_col_wise=True, verbose=-1)
        elif PARAMS_MAIN_CLASSIFIER == 'CatBoost':
            return catboost.CatBoostClassifier(random_state=SEED, verbose=0)
        elif PARAMS_MAIN_CLASSIFIER == 'LDA':
            return LinearDiscriminantAnalysis()
        elif PARAMS_MAIN_CLASSIFIER == 'QDA':
            return QuadraticDiscriminantAnalysis(reg_param=0.01)
        elif PARAMS_MAIN_CLASSIFIER in [
            'TabNet', 'TabTransformer', 'CategoryEmbedding', 'GATE',
            'FTTransformer', 'AutoInt', 'DANet', 'GANDALF', 'NODE'
        ]:
            if not PYTORCH_TABULAR_AVAILABLE:
                raise ImportError(
                    "pytorch_tabular is not installed but a Tabular model was requested "
                    f"({PARAMS_MAIN_CLASSIFIER}). Please install pytorch_tabular."
                )
            name_map = {
                'TabNet': 'tabnet',
                'TabTransformer': 'tabtransformer',
                'CategoryEmbedding': 'category_embedding',
                'GATE': 'gate',
                'FTTransformer': 'ft_transformer',
                'AutoInt': 'autoint',
                'DANet': 'danet',
                'GANDALF': 'gandalf',
                'NODE': 'node',
            }
            model_name = name_map[PARAMS_MAIN_CLASSIFIER]
            return TabularEstimator(
                task="classification",
                model_name=model_name,
                target_col=self.label_Y,
                categorical_cols=self.cate_attrs if self.cate_attrs else None,
                continuous_cols=self.num_attrs if self.num_attrs else None,
            )
        elif PARAMS_MAIN_CLASSIFIER in ['DeepFM', 'WideDeep', 'DCN', 'xDeepFM']:
            model_type_map = {
                'DeepFM': 'deepfm',
                'WideDeep': 'widedeep',
                'DCN': 'dcn',
                'xDeepFM': 'xdeepfm',
            }
            model_type = model_type_map[PARAMS_MAIN_CLASSIFIER]
            return TorchCTRClassifier(
                model_type=model_type,
                categorical_cols=self.cate_attrs if self.cate_attrs else None,
                continuous_cols=self.num_attrs if self.num_attrs else None,
                embedding_dim=16,
                hidden_units=[128, 64],
                cin_layer_sizes=[16, 16],
                lr=1e-3,
                batch_size=512,
                epochs=5,
                device=None,
                verbose=VERBOSE,
            )
        else:
            raise ValueError(f"Unsupported classifier type: {PARAMS_MAIN_CLASSIFIER}")


    def fit(self, X, Y, O):
        """
        Train the model using separate X, Y, O data format.
        - First training always uses the full dataset
        - Subsequent trainings use a sample according to PARAMS_MAIN_TRAINING_RATE
        """
        if not isinstance(O, pd.DataFrame):
            O = pd.DataFrame(O)
        
        if PARAMS_EVAL_NORM == 'min-max':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))

            df_scaled_array = scaler.fit_transform(X)

            X = pd.DataFrame(
                df_scaled_array,
                columns=X.columns,
                index=X.index
            )
        elif PARAMS_EVAL_NORM == 'z-score':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

            df_scaled_array = scaler.fit_transform(X)

            X = pd.DataFrame(
                df_scaled_array,
                columns=X.columns,
                index=X.index
            )

        if self.first_train:
            if VERBOSE:
                print("First training, using full dataset")
            self.model.fit(X, Y)
            self.first_train = False
        else:
            if 0 < PARAMS_MAIN_TRAINING_RATE < 1:
                if VERBOSE:
                    print(f"Subsequent training, sampling at {PARAMS_MAIN_TRAINING_RATE} ratio")
                sample_indices = X.sample(frac=PARAMS_MAIN_TRAINING_RATE, random_state=SEED).index
                X_train = X.loc[sample_indices]
                if isinstance(Y, pd.Series):
                    Y_train = Y.loc[sample_indices]
                else:
                    Y_train = Y[sample_indices]
                self.model.fit(X_train, Y_train)
            else:
                if VERBOSE:
                    print("Subsequent training, using full dataset")
                self.model.fit(X, Y)

    def predict(self, X_test, Y_test, O_test):
        """
        Make predictions on test data using separate X, Y, O data format.
        """
        self.label_O = self.label_O if isinstance(self.label_O, list) else [self.label_O]
        if PARAMS_EVAL_NORM == 'min-max':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))

            df_scaled_array = scaler.fit_transform(X_test)

            X_test = pd.DataFrame(
                df_scaled_array,
                columns=X_test.columns,
                index=X_test.index
            )
        elif PARAMS_EVAL_NORM == 'z-score':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

            df_scaled_array = scaler.fit_transform(X_test)

            X_test = pd.DataFrame(
                df_scaled_array,
                columns=X_test.columns,
                index=X_test.index
            )
        
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_test)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                score_S = proba[:, 1]
            else:
                score_S = proba.ravel()
        else:
            print("Warning: model has no predict_proba, using hard predictions as score.")
            score_S = self.model.predict(X_test).astype(float)
        y_pred = self.model.predict(X_test)
        
        results_data = {
            'label_Y': Y_test,
            'pred_Y': y_pred,
            'score_S': score_S,
        }
        
        if isinstance(O_test, pd.DataFrame):
            for col in O_test.columns:
                results_data[col] = O_test[col]
        else:
            if isinstance(self.label_O, list) and len(self.label_O) > 0:
                results_data[self.label_O[0]] = O_test
            else:
                results_data[self.label_O] = O_test
        
        self.results_df = pd.DataFrame(results_data)
    
    def _binary_confusion_counts(self, df, positive_class):
        """
        Calculate binary confusion matrix metrics for a specific positive class (multi-class to binary)
        Args:
            df (pd.DataFrame): Filtered dataframe for a specific sensitive group
            positive_class: The class to treat as positive (others as negative)
        Returns:
            dict: Binary metrics including TPR, FPR, PPV, NPV, ACC, PPOS, FDR, FOR, FNR, FPR
        """
        if df.empty:
            return {
                'TPR': 0.0, 'FPR': 0.0, 'PPV': 0.0, 'NPV': 0.0,
                'ACC': 0.0, 'PPOS': 0.0, 'FDR': 0.0, 'FOR': 0.0,
                'FNR': 0.0, 'FPR': 0.0
            }
        
        # Convert to binary labels for the target positive class
        y_true = (df['label_Y'] == positive_class).astype(int)
        y_pred = (df['pred_Y'] == positive_class).astype(int)
        
        # Basic confusion matrix counts
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        # Calculate metrics with zero division handling
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # True Positive Rate (Recall)
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0  # False Positive Rate
        PPV = TP / (TP + FP) if (TP + FP) > 0 else 0.0  # Positive Predictive Value (Precision)
        NPV = TN / (TN + FN) if (TN + FN) > 0 else 0.0  # Negative Predictive Value
        ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0  # Accuracy
        PPOS = (TP + FP) / len(df) if len(df) > 0 else 0.0  # Positive Prediction Rate
        FDR = FP / (TP + FP) if (TP + FP) > 0 else 0.0  # False Discovery Rate
        FOR = FN / (TN + FN) if (TN + FN) > 0 else 0.0  # False Omission Rate
        FNR = FN / (TP + FN) if (TP + FN) > 0 else 0.0  # False Negative Rate
        
        return {
            'TPR': TPR, 'FPR': FPR, 'PPV': PPV, 'NPV': NPV,
            'ACC': ACC, 'PPOS': PPOS, 'FDR': FDR, 'FOR': FOR,
            'FNR': FNR, 'FPR': FPR
        }

    
    def _calculate_bnc(self, label_O, group_pairs):
        """
        Calculate Between Negative Classes (BNC) fairness metric for multi-class classification.
        """
        vals = []
        # Get all unique classes in true labels
        all_classes = pd.unique(self.results_df['label_Y'])
        
        for group_a, group_b in group_pairs:
            max_diff = 0.0
            
            for neg_class in all_classes:
                # Filter negative class samples for each group
                group_a_neg = self.results_df[
                    (self.results_df[label_O] == group_a) & 
                    (self.results_df['label_Y'] == neg_class)
                ]
                group_b_neg = self.results_df[
                    (self.results_df[label_O] == group_b) & 
                    (self.results_df['label_Y'] == neg_class)
                ]
                
                # Calculate mean score for negative class
                m_a = group_a_neg['score_S'].mean() if len(group_a_neg) > 0 else 0.0
                m_b = group_b_neg['score_S'].mean() if len(group_b_neg) > 0 else 0.0
                
                # Update max difference for this group pair
                current_diff = abs(m_a - m_b)
                if current_diff > max_diff:
                    max_diff = current_diff
            
            vals.append(max_diff)
        
        return sum(vals) / len(vals) if vals else 0.0


    def _calculate_bpc(self, label_O, group_pairs):
        """
        Calculate Between Positive Classes (BPC) fairness metric for multi-class classification.
        BPC measures the maximum difference in mean prediction scores of positive class samples 
        between different sensitive groups across all classes.
        """
        vals = []
        all_classes = pd.unique(self.results_df['label_Y'])
        
        for group_a, group_b in group_pairs:
            max_diff = 0.0
            
            for pos_class in all_classes:
                group_a_pos = self.results_df[
                    (self.results_df[label_O] == group_a) & 
                    (self.results_df['label_Y'] == pos_class)
                ]
                group_b_pos = self.results_df[
                    (self.results_df[label_O] == group_b) & 
                    (self.results_df['label_Y'] == pos_class)
                ]
                
                m_a = group_a_pos['score_S'].mean() if len(group_a_pos) > 0 else 0.0
                m_b = group_b_pos['score_S'].mean() if len(group_b_pos) > 0 else 0.0
                
                current_diff = abs(m_a - m_b)
                if current_diff > max_diff:
                    max_diff = current_diff
            
            vals.append(max_diff)

        return sum(vals) / len(vals) if vals else 0.0


    def _calculate_cuae(self, label_O, group_pairs):
        """
        Calculate Conditional Use Accuracy Equality (CUAE) fairness metric for multi-class classification.
        """
        vals = []
        all_classes = pd.unique(self.results_df['label_Y'])
        
        for group_a, group_b in group_pairs:
            max_diff = 0.0
            
            for pos_class in all_classes:
                # Calculate binary metrics for current positive class
                stats_a = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_a], 
                    pos_class
                )
                stats_b = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_b], 
                    pos_class
                )
                
                # Calculate max of PPV and NPV differences
                current_diff = max(
                    abs(stats_a['PPV'] - stats_b['PPV']),
                    abs(stats_a['NPV'] - stats_b['NPV'])
                )
                
                if current_diff > max_diff:
                    max_diff = current_diff
            
            vals.append(max_diff)
        
        return sum(vals) / len(vals) if vals else 0.0


    def _calculate_eopp(self, label_O, group_pairs):
        """
        Calculate Equal Opportunity (EOpp) fairness metric for multi-class classification.
        """
        vals = []
        all_classes = pd.unique(self.results_df['label_Y'])
        
        for group_a, group_b in group_pairs:
            max_diff = 0.0
            
            for pos_class in all_classes:
                stats_a = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_a], 
                    pos_class
                )
                stats_b = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_b], 
                    pos_class
                )
                
                current_diff = abs(stats_a['TPR'] - stats_b['TPR'])
                if current_diff > max_diff:
                    max_diff = current_diff
            
            vals.append(max_diff)
        
        return sum(vals) / len(vals) if vals else 0.0

    def _calculate_eo(self, label_O, group_pairs):
        """
        Calculate Equalized Odds (EO) fairness metric for multi-class classification.
        """
        vals = []
        all_classes = pd.unique(self.results_df['label_Y'])
        
        for group_a, group_b in group_pairs:
            max_diff = 0.0
            
            for pos_class in all_classes:
                stats_a = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_a], 
                    pos_class
                )
                stats_b = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_b], 
                    pos_class
                )
                
                current_diff = max(
                    abs(stats_a['TPR'] - stats_b['TPR']),
                    abs(stats_a['FPR'] - stats_b['FPR'])
                )
                
                if current_diff > max_diff:
                    max_diff = current_diff
            
            vals.append(max_diff)
        
        return sum(vals) / len(vals) if vals else 0.0

    def _calculate_fdrp(self, label_O, group_pairs):
        """
        Calculate False Discovery Rate Parity (FDRP) fairness metric for multi-class classification.
        """
        vals = []
        all_classes = pd.unique(self.results_df['label_Y'])
        
        for group_a, group_b in group_pairs:
            max_diff = 0.0
            
            for pos_class in all_classes:
                stats_a = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_a], 
                    pos_class
                )
                stats_b = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_b], 
                    pos_class
                )
                
                current_diff = abs(stats_a['FDR'] - stats_b['FDR'])
                if current_diff > max_diff:
                    max_diff = current_diff
            
            vals.append(max_diff)
        
        return sum(vals) / len(vals) if vals else 0.0

    def _calculate_forp(self, label_O, group_pairs):
        """
        Calculate False Omission Rate Parity (FORP) fairness metric for multi-class classification.
        """
        vals = []
        all_classes = pd.unique(self.results_df['label_Y'])
        
        for group_a, group_b in group_pairs:
            max_diff = 0.0
            
            for pos_class in all_classes:
                stats_a = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_a], 
                    pos_class
                )
                stats_b = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_b], 
                    pos_class
                )
                
                current_diff = abs(stats_a['FOR'] - stats_b['FOR'])
                if current_diff > max_diff:
                    max_diff = current_diff
            
            vals.append(max_diff)
        
        return sum(vals) / len(vals) if vals else 0.0

    def _calculate_fnrb(self, label_O, group_pairs):
        """
        Calculate False Negative Rate Balance (FNRB) fairness metric for multi-class classification.
        """
        vals = []
        all_classes = pd.unique(self.results_df['label_Y'])
        
        for group_a, group_b in group_pairs:
            max_diff = 0.0
            
            for pos_class in all_classes:
                stats_a = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_a], 
                    pos_class
                )
                stats_b = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_b], 
                    pos_class
                )
                
                current_diff = abs(stats_a['FNR'] - stats_b['FNR'])
                if current_diff > max_diff:
                    max_diff = current_diff
            
            vals.append(max_diff)
        
        return sum(vals) / len(vals) if vals else 0.0

    def _calculate_fprb(self, label_O, group_pairs):
        """
        Calculate False Positive Rate Balance (FPRB) fairness metric for multi-class classification.
        """
        vals = []
        all_classes = pd.unique(self.results_df['label_Y'])
        
        for group_a, group_b in group_pairs:
            max_diff = 0.0
            
            for pos_class in all_classes:
                stats_a = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_a], 
                    pos_class
                )
                stats_b = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_b], 
                    pos_class
                )
                
                current_diff = abs(stats_a['FPR'] - stats_b['FPR'])
                if current_diff > max_diff:
                    max_diff = current_diff
            
            vals.append(max_diff)
        
        return sum(vals) / len(vals) if vals else 0.0

    def _calculate_npvp(self, label_O, group_pairs):
        """
        Calculate Negative Predictive Value Parity (NPVP) fairness metric for multi-class classification.
        """
        vals = []
        all_classes = pd.unique(self.results_df['label_Y'])
        
        for group_a, group_b in group_pairs:
            max_diff = 0.0
            
            for pos_class in all_classes:
                stats_a = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_a], 
                    pos_class
                )
                stats_b = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_b], 
                    pos_class
                )
                
                current_diff = abs(stats_a['NPV'] - stats_b['NPV'])
                if current_diff > max_diff:
                    max_diff = current_diff
            
            vals.append(max_diff)
        
        return sum(vals) / len(vals) if vals else 0.0

    def _calculate_oae(self, label_O, group_pairs):
        """
        Calculate Overall Accuracy Equality (OAE) fairness metric for multi-class classification.
        """
        vals = []
        all_classes = pd.unique(self.results_df['label_Y'])
        
        for group_a, group_b in group_pairs:
            max_diff = 0.0
            
            for pos_class in all_classes:
                stats_a = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_a], 
                    pos_class
                )
                stats_b = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_b], 
                    pos_class
                )
                
                current_diff = abs(stats_a['ACC'] - stats_b['ACC'])
                if current_diff > max_diff:
                    max_diff = current_diff
            
            vals.append(max_diff)
        
        return sum(vals) / len(vals) if vals else 0.0

    def _calculate_ppvp(self, label_O, group_pairs):
        """
        Calculate Positive Predictive Value Parity (PPVP) fairness metric for multi-class classification.
        """
        vals = []
        all_classes = pd.unique(self.results_df['label_Y'])
        
        for group_a, group_b in group_pairs:
            max_diff = 0.0
            
            for pos_class in all_classes:
                stats_a = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_a], 
                    pos_class
                )
                stats_b = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_b], 
                    pos_class
                )
                
                current_diff = abs(stats_a['PPV'] - stats_b['PPV'])
                if current_diff > max_diff:
                    max_diff = current_diff
            
            vals.append(max_diff)
        
        return sum(vals) / len(vals) if vals else 0.0

    def _calculate_sp(self, label_O, group_pairs):
        """
        Calculate Statistical Parity (SP) fairness metric for multi-class classification.
        """
        vals = []
        all_classes = pd.unique(self.results_df['label_Y'])
        
        for group_a, group_b in group_pairs:
            max_diff = 0.0
            
            for pos_class in all_classes:
                stats_a = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_a], 
                    pos_class
                )
                stats_b = self._binary_confusion_counts(
                    self.results_df[self.results_df[label_O] == group_b], 
                    pos_class
                )
                
                current_diff = abs(stats_a['PPOS'] - stats_b['PPOS'])
                if current_diff > max_diff:
                    max_diff = current_diff
            
            vals.append(max_diff)
        
        return sum(vals) / len(vals) if vals else 0.0
    
    def _calculate_acc(self):
        """Calculate accuracy for multi-class classification tasks"""
        if self.results_df is None or self.results_df.empty:
            raise ValueError("results_df is empty, please generate prediction results first")
        
        y_true = self.results_df['label_Y']
        y_pred = self.results_df['pred_Y']
        
        acc = accuracy_score(y_true, y_pred)
        return acc
    
    def _calculate_f1(self, average='weighted'):
        """
        Calculate F1 score for multi-class classification tasks
        """
        if self.results_df is None or self.results_df.empty:
            raise ValueError("results_df is empty, please generate prediction results first")
        
        y_true = self.results_df['label_Y']
        y_pred = self.results_df['pred_Y']
        
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        return f1
    
    def _calculate_recall(self, average='weighted'):
        """
        Calculate recall for multi-class classification tasks
        """
        if self.results_df is None or self.results_df.empty:
            raise ValueError("results_df is empty, please generate prediction results first")
        
        y_true = self.results_df['label_Y']
        y_pred = self.results_df['pred_Y']
        
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        return recall
    
    def _calculate_precision(self, average='weighted'):
        """
        Calculate precision for multi-class classification tasks
        """
        if self.results_df is None or self.results_df.empty:
            raise ValueError("results_df is empty, please generate prediction results first")
        
        y_true = self.results_df['label_Y']
        y_pred = self.results_df['pred_Y']
        
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        return precision
    
    def calculate_metrics(self):
        """
        Calculate all requested fairness and accuracy metrics.
        """
        # Check if results_df exists
        if self.results_df is None:
            raise ValueError("No results available. Run predict first.")
        
        # Extract data from results_df
        exclude_cols = {'label_Y', 'pred_Y', 'score_S'}
        label_O_columns = [c for c in self.results_df.columns if c not in exclude_cols]
        
        # Calculate classifier metrics using encapsulated methods
        classifier_metrics = {}
        
        accuracy_metric_map = {
            'ACC': self._calculate_acc,
            'F1': self._calculate_f1,
            'Recall': self._calculate_recall,
            'Precision': self._calculate_precision
        }
        
        for metric_name in PARAMS_EVAL_METRIC_ACCURACY:
            if metric_name in accuracy_metric_map:
                classifier_metrics[metric_name] = accuracy_metric_map[metric_name]()
        
        # Calculate fairness metrics for each label_O
        metric_map = {
            'BNC'  : self._calculate_bnc,
            'BPC'  : self._calculate_bpc,
            'CUAE' : self._calculate_cuae,
            'EOpp' : self._calculate_eopp,
            'EO'   : self._calculate_eo,
            'FDRP' : self._calculate_fdrp,
            'FORP' : self._calculate_forp,
            'FNRB' : self._calculate_fnrb,
            'FPRB' : self._calculate_fprb,
            'NPVP' : self._calculate_npvp,
            'OAE'  : self._calculate_oae,
            'PPVP' : self._calculate_ppvp,
            'SP'   : self._calculate_sp
        }
        
        fairness_metrics = {name: {} for name in PARAMS_EVAL_METRIC_FAIRNESS if name in metric_map}
        
        for label_O in label_O_columns:
            groups = self.results_df[label_O].unique()
            if len(groups) < 2:
                print(f"Warning: Only one group found in {label_O}. Cannot calculate fairness metrics.")
                continue
            
            group_pairs = list(combinations(groups, 2))
            
            for name in fairness_metrics.keys():
                func = metric_map[name]
                fairness_metrics[name][label_O] = func(label_O, group_pairs)
        
        # Combine all metrics
        self.metrics_results = {**classifier_metrics, **fairness_metrics}
        
        if VERBOSE:
            print("Metrics calculated successfully")
        
        return self.metrics_results
    
    
    def evaluate(self, X_train, Y_train, O_train, X_test, Y_test, O_test):
        """
        Main evaluation pipeline using separate X, Y, O data format.
        """
        self.fit(X_train, Y_train, O_train)
        self.predict(X_test, Y_test, O_test)
        return self.calculate_metrics()
    

    def calculate_epsilon(self, X, O, cate_attrs, num_attrs):
        """
        Calculate epsilon metric for measuring bias concentration using separate X, Y, O data format.
        """
        # Create a single DataFrame from X, Y, O
        df_data = pd.concat([X.reset_index(drop=True), O.reset_index(drop=True)], axis=1)
        
        # Handle both single and multiple protected attributes
        label_O_list = [self.label_O] if isinstance(self.label_O, str) else self.label_O
        epsilon_results = {}
        
        # Set default h_order if not specified
        if self.h_order == 'default':
            self.h_order = len(X.columns) - 1
        
        # Create a copy of df_data with all label_O columns dropped
        all_label_O = label_O_list
        df_data_no_O = df_data.drop(all_label_O, axis=1) if len(all_label_O) > 0 else df_data.copy()
        
        # Calculate epsilon for each protected attribute
        for label_O in label_O_list:
            unique_values = df_data[label_O].unique()
            comb_label_arr = combinations(unique_values, 2)
            df_S_full = pd.DataFrame()
            
            # Process each combination of protected attribute values
            for p, n in comb_label_arr:
                mask = df_data[label_O].isin([p, n])
                df_filtered = df_data_no_O[mask].copy()
                df_filtered[label_O] = df_data[label_O][mask]
                
                # Normalize numerical attributes
                for col in num_attrs:
                    if col in df_filtered.columns:
                        min_val, max_val = df_filtered[col].min(), df_filtered[col].max()
                        if min_val != max_val:
                            df_filtered[col] = (df_filtered[col] - min_val) / (max_val - min_val)
                
                # Calculate differences for categorical attributes
                cat_diff = {}
                for attr in cate_attrs:
                    if attr in [label_O, self.label_Y] or attr not in df_filtered.columns:
                        continue
                    
                    p_data = df_filtered[df_filtered[label_O] == p][attr]
                    n_data = df_filtered[df_filtered[label_O] == n][attr]
                    
                    p_counts = p_data.value_counts()
                    n_counts = n_data.value_counts()
                    N_c1 = p_counts.sum()
                    N_c2 = n_counts.sum()
                    
                    combined = p_counts.reindex(pd.unique([*p_counts.index, *n_counts.index]), fill_value=0)
                    n_counts = n_counts.reindex(combined.index, fill_value=0)
                    if PARAMS_EVAL_CAT == 'cat-a':
                        K = len(combined)
                        proportions_p = combined / N_c1
                        proportions_n = n_counts / N_c2
                        cat_diff[attr] = (1 / K) * (proportions_p - proportions_n).abs().sum()
                    elif PARAMS_EVAL_CAT == 'cat-b':
                        numerator = (combined - n_counts) ** 2
                        denominator = combined + n_counts
                        denominator[denominator == 0] = 1
                        cat_diff[attr] = 0.5 * (numerator / denominator).sum()
                # Calculate differences for numerical attributes
                if PARAMS_EVAL_NUM != 'num-d':
                    num_diff = pd.Series(dtype='float64')
                    for col in num_attrs:
                        p_vals = df_filtered[df_filtered[label_O] == p][col].values
                        n_vals = df_filtered[df_filtered[label_O] == n][col].values
                        
                        if PARAMS_EVAL_NUM == 'num-a':
                            p_mean = p_vals.mean()
                            n_mean = n_vals.mean()
                            num_diff[col] = abs(p_mean - n_mean)
                        elif PARAMS_EVAL_NUM == 'num-b':
                            from scipy.stats import ks_2samp
                            ks_stat, _ = ks_2samp(p_vals, n_vals)
                            num_diff[col] = ks_stat
                        elif PARAMS_EVAL_NUM == 'num-c':
                            if len(p_vals) > 0 and len(n_vals) > 0:
                                diff_matrix = np.abs(p_vals[:, np.newaxis] - n_vals)
                                min_dist_p = np.min(diff_matrix, axis=1)
                                min_dist_n = np.min(diff_matrix.T, axis=1)
                                all_min_dist = np.concatenate([min_dist_p, min_dist_n])
                                N = len(p_vals) + len(n_vals)
                                num_diff[col] = np.sum(all_min_dist) / N
                            else:
                                num_diff[col] = 1

                    # Scale differences
                    if PARAMS_EVAL_SCALE == 'min':
                        if not num_diff.empty:
                            num_diff = num_diff / num_diff.min()
                        if cat_diff:
                            cat_diff_series = pd.Series(cat_diff)
                            cat_diff = (cat_diff_series / cat_diff_series.min()).to_dict()
                    elif PARAMS_EVAL_SCALE == 'mean':
                        if not num_diff.empty:
                            num_diff = num_diff / num_diff.mean()
                        if cat_diff:
                            cat_diff_series = pd.Series(cat_diff)
                            cat_diff = (cat_diff_series / cat_diff_series.mean()).to_dict()
                    elif PARAMS_EVAL_SCALE == 'sigma':
                        if not num_diff.empty:
                            num_diff = (num_diff - num_diff.mean()) / num_diff.std()
                            num_diff = num_diff - num_diff.min()
                        if cat_diff:
                            cat_diff_series = pd.Series(cat_diff)
                            cat_diff_series = (cat_diff_series - cat_diff_series.mean()) / cat_diff_series.std()
                            cat_diff_series = cat_diff_series - cat_diff_series.min()
                            cat_diff = cat_diff_series.to_dict()
                    comb_key = f"{p}_{n}"
                    df_S_full[comb_key] = pd.concat([num_diff, pd.Series(cat_diff)])
                
                elif PARAMS_EVAL_NUM == 'num-d':
                    comb_key = f"{p}_{n}"
                    df_S_full[comb_key] = pd.Series(cat_diff)
                
            # Prepare index array for distance matrix
            index_arr = list(df_S_full.index) + ['origin']
            distance_matrix = pd.DataFrame(np.nan, index=index_arr, columns=index_arr)
            np.fill_diagonal(distance_matrix.values, 0)
            df_S_sq = df_S_full ** 2
            temp_dict = {}
            
            def get_subsets(indexes, h_order):
                h_order = max(h_order, -1)
                subsets = []
                for i in range(len(indexes), h_order, -1):
                    subsets.extend(combinations(indexes, i))
                return [list(sub) for sub in subsets]
            
            sub_indexes = get_subsets(X.columns, self.h_order - 1)

            if PARAMS_EVAL_NUM != 'num-d':
                if PARAMS_EVAL_SUM == 'd1A':
                    for sub in sub_indexes:
                        temp_dict[tuple(sorted(sub))] = df_S_sq.loc[sub].mean() ** 0.5
                elif PARAMS_EVAL_SUM == 'd1B':
                    for sub in sub_indexes:
                        temp_dict[tuple(sorted(sub))] = df_S_sq.loc[sub].sum() ** 0.5
            elif PARAMS_EVAL_NUM == 'num-d':
                for sub in sub_indexes:
                    cate_sub = [s for s in sub if s in cate_attrs]
                    cate_temp = df_S_sq.loc[cate_sub].mean() ** 0.5
                    
                    num_sub = [s for s in sub if s in num_attrs]
                    num_temp = 0
                    if num_sub:
                        X_num = df_data[num_sub].values.astype(np.float32)
                        labels = df_data[self.label_O].values

                        dist_matrix = pairwise_distances(X_num, metric=PARAMS_EVAL_DIST_METRIC)

                        if PARAMS_EVAL_SCALE == 'mean':
                            dist_matrix = dist_matrix / np.mean(dist_matrix)
                        elif PARAMS_EVAL_SCALE == 'min':
                            dist_matrix = dist_matrix / np.mean(dist_matrix, axis=1).min()
                        elif PARAMS_EVAL_SCALE == 'zscore':
                            dist_matrix = (dist_matrix - dist_matrix.mean()) / dist_matrix.std()
                        
                        delta_matrix = (labels[:, np.newaxis] == labels).astype(np.float32)
                        indicator_matrix = 1 - delta_matrix

                        dist_masked = dist_matrix * indicator_matrix
                        dist_masked[dist_masked == 0] = np.inf
                        min_dist_per_i = np.min(dist_masked, axis=1)
                        min_dist_per_i[np.isinf(min_dist_per_i)] = 0
                        
                        num_temp = np.sum(min_dist_per_i) / N

                    temp_dict[tuple(sorted(sub))] = cate_temp + num_temp

            def calculate_distance(attr_1, attr_2):
                
                available_indexes = X.columns.tolist()
                if attr_1 != 'origin':
                    available_indexes.remove(attr_1)
                if attr_2 != 'origin':
                    available_indexes.remove(attr_2)
                
                adjusted_h_order = self.h_order-(len(X.columns.tolist()) - len(available_indexes))
                temp_index_list = get_subsets(available_indexes, adjusted_h_order)

                result_list = []
                for s in temp_index_list:
                    s1 = sorted(s + [attr_1]) if attr_1 != 'origin' else sorted(s)
                    s2 = sorted(s + [attr_2]) if attr_2 != 'origin' else sorted(s)

                    val1 = temp_dict.get(tuple(s1), 0)
                    val2 = temp_dict.get(tuple(s2), 0)

                    result_list.append(val1 - val2)
                
                scalar_results = []
                for item in result_list:
                    if hasattr(item, 'shape') and len(item.shape) > 0:
                        scalar_results.append(float(np.mean(item)))
                    else:
                        scalar_results.append(float(item))
                
                return np.mean(np.abs(scalar_results)) if scalar_results else 0
            
            for attr in X.columns:
                distance_matrix.loc[attr, attr] = 0
            
            for attr_1, attr_2 in combinations(index_arr, 2):
                dist = calculate_distance(attr_1, attr_2)
                distance_matrix.loc[attr_1, attr_2] = dist
                distance_matrix.loc[attr_2, attr_1] = dist
            
            print(distance_matrix)
            try:
                def find_optimal_mds_components(distance_matrix, max_components, slope_threshold):
                    n_samples = distance_matrix.shape[0]
                    max_components = min(max_components, n_samples - 1)
                    

                    stress_list = []
                    for n in range(1, max_components + 1):
                        mds_temp = MDS(
                            n_components=n,
                            dissimilarity='precomputed',
                            random_state=SEED,
                            n_init=1,
                            max_iter=20000,
                            eps=1e-10,
                            normalized_stress='auto'
                        )
                        mds_temp.fit(distance_matrix)
                        stress_list.append(mds_temp.stress_)

                    optimal_component = max_components
                    stress_list = (np.array(stress_list) - np.array(stress_list).min()) / (np.array(stress_list).max() - np.array(stress_list).min())
                    for i in range(1, len(stress_list)):
                        slope = abs(stress_list[i] - stress_list[i-1])
                        if slope < slope_threshold:
                            optimal_component = i + 1
                            break
                    
                    return optimal_component

                optimal_n = find_optimal_mds_components(
                    distance_matrix=distance_matrix,
                    max_components=PARAMS_EVAL_MAX_COMPONENTS,
                    slope_threshold=PARAMS_EVAL_SLOPE_THRESHOLD
                )

                mds = MDS(
                    n_components=optimal_n,
                    dissimilarity='precomputed',
                    random_state=SEED,
                    n_init=4,
                    max_iter=10000,
                    eps=1e-10,
                )
                pts = mds.fit_transform(distance_matrix)
                pts = pts - pts[-1]


                def nd_rotation(pts):
                    pts = np.asarray(pts, dtype=np.float64)
                    if pts.ndim != 2:
                        raise ValueError("The input data must be a two-dimensional array!!")
                    
                    n_samples, d = pts.shape
                    if d < 2:
                        raise ValueError("The input data dimension must be ≥21!!")
                    
                    pts_rotated = pts.copy()
                    farthest_indices = []

                    for k in range(d - 1):
                        subspace_pts = pts_rotated[:, k:].copy()
                        distances = np.linalg.norm(subspace_pts, axis=1)
                        
                        farthest_idx = np.argmax(distances)
                        farthest_indices.append(farthest_idx)
                        axis_direction = subspace_pts[farthest_idx]
                        
                        axis_norm = np.linalg.norm(axis_direction)
                        if axis_norm < 1e-10:
                            continue
                        
                        axis_direction_normalized = axis_direction / axis_norm
                        proj_len = np.dot(subspace_pts, axis_direction) / (axis_norm ** 2)
                        
                        proj_part = np.outer(proj_len, axis_direction)
                        orthogonal_part = subspace_pts - proj_part
                        
                        pts_rotated[:, k] = proj_len * axis_norm
                        if k + 1 < d:
                            pts_rotated[:, k+1:] = orthogonal_part[:, 1:]

                    last_distances = np.abs(pts_rotated[:, -1])
                    farthest_indices.append(np.argmax(last_distances))

                    return pts_rotated, farthest_indices
                
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D

                def universal_2d_3d_visualization(pts_original, pts_rotated, farthest_indices):
                    d_vis = min(pts_original.shape[1], 3)
                    pts_original_vis = pts_original[:, :d_vis]
                    pts_rotated_vis = pts_rotated[:, :d_vis]
                    
                    all_pts_vis = np.vstack((pts_original_vis, pts_rotated_vis))
                    x_min, x_max = all_pts_vis[:, 0].min() * 1.1, all_pts_vis[:, 0].max() * 1.1
                    y_min, y_max = all_pts_vis[:, 1].min() * 1.1, all_pts_vis[:, 1].max() * 1.1
                    z_min = z_max = None
                    if d_vis == 3:
                        z_min, z_max = all_pts_vis[:, 2].min() * 1.1, all_pts_vis[:, 2].max() * 1.1
                    
                    fig = plt.figure(figsize=(16, 8))
                    plot_titles = ["Before Rotation", "After Rotation"]
                    pts_data = [pts_original_vis, pts_rotated_vis]
                    point_colors = ['skyblue', 'lightgreen']
                    
                    for idx, (pts_data_i, title, color) in enumerate(zip(pts_data, plot_titles, point_colors)):
                        if d_vis == 2:
                            ax = fig.add_subplot(1, 2, idx + 1)
                        else:
                            ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
                        
                        if d_vis == 2:
                            ax.scatter(pts_data_i[:, 0], pts_data_i[:, 1], 
                                    c=color, alpha=0.7, label='Sample Points')
                        else:
                            ax.scatter(pts_data_i[:, 0], pts_data_i[:, 1], pts_data_i[:, 2],
                                    c=color, alpha=0.7, label='Sample Points')
                        
                        origin_coords = [0] * d_vis
                        if d_vis == 2:
                            ax.scatter(origin_coords[0], origin_coords[1], 
                                    c='red', s=150, marker='*', label='Origin (0,0)')
                        else:
                            ax.scatter(origin_coords[0], origin_coords[1], origin_coords[2],
                                    c='red', s=150, marker='*', label='Origin (0,0,0)')
                        
                        colors = ['darkorange', 'purple', 'darkgreen']
                        markers = ['P', 'X', 's']
                        for i, (farthest_idx, color_p, marker_p) in enumerate(zip(
                            farthest_indices[:d_vis], colors, markers
                        )):
                            point_coords = pts_data_i[farthest_idx]
                            if d_vis == 2:
                                ax.scatter(point_coords[0], point_coords[1],
                                        c=color_p, s=150, marker=marker_p,
                                        label=f'P{i+1} (Axis {chr(88+i)})')
                            else:
                                ax.scatter(point_coords[0], point_coords[1], point_coords[2],
                                        c=color_p, s=150, marker=marker_p,
                                        label=f'P{i+1} (Axis {chr(88+i)})')
                        
                        if idx == 1:
                            if d_vis == 2:
                                ax.axhline(y=0, color='darkorange', linewidth=2, alpha=0.8, label='X-axis (Main Axis 1)')
                            else:
                                ax.plot([x_min, x_max], [0, 0], [0, 0], 
                                        'darkorange', linewidth=2, alpha=0.8, label='X-axis (Main Axis 1)')
                                ax.plot([0, 0], [y_min, y_max], [0, 0], 
                                        'purple', linewidth=2, alpha=0.8, label='Y-axis (Main Axis 2)')
                        
                        ax.set_xlabel(f'{chr(88)} Coordinate')
                        ax.set_ylabel(f'{chr(89)} Coordinate')
                        if d_vis == 3:
                            ax.set_zlabel(f'{chr(90)} Coordinate')
                            ax.set_xlim(x_min, x_max)
                            ax.set_ylim(y_min, y_max)
                            ax.set_zlim(z_min, z_max)
                            ax.view_init(elev=25, azim=45)
                        else:
                            ax.set_xlim(x_min, x_max)
                            ax.set_ylim(y_min, y_max)
                        
                        ax.set_title(f'{title} ({d_vis}D) - Aligned Main Axes')
                        ax.grid(alpha=0.3)
                        ax.legend(loc='best', fontsize=8)
                    
                    plt.tight_layout()
                    plt.savefig('test02.png')
                
                pts_rotated, farthest_indices = nd_rotation(pts)

                epsilon_concentration = pd.DataFrame(pts_rotated, index=distance_matrix.index).T
                df_epsilon = pd.Series(np.sqrt((pts ** 2).sum(axis=1)), index=distance_matrix.index)
                epsilon_results[label_O] = df_epsilon.sort_values(ascending=False)
            except Exception as e:
                print(f"Error calculating MDS for {label_O}: {e}")
                epsilon_results[label_O] = pd.Series({attr: 0 for attr in X.columns})
        
        return epsilon_results
