### main.py
VERBOSE = True
# DATASET = {
#     # 'name': ''
#     'path': 'credit.xlsx',
#     'target': 'default payment next month',
#     'protected': 'SEX',
# }
DATASET = {
    # 'name': ''
    'path': 'data_COMPAS.csv',
    'label_Y': 'two_year_recid',
    'label_O': ['sex'],
}

DATASET_INFO = {
    "credit": {
        "data": 'data_Credit_Card.csv',
    },

    "compas": {
        "data": 'data_COMPAS.csv',
    }
}


SEED = 0
USE_BIAS_MITIGATION = True
USE_ACCURACY_ENHANCEMENT = False
PARAMS_MAIN_STEP = 'd3B'
# Supported classifier types: LR(Logistic Regression), DT(Decision Tree), KNN(K-Nearest Neighbors), GBDT(Gradient Boosting Decision Tree),
# ADABoost(AdaBoost), NB(Naive Bayes), SVM(Support Vector Machine), MLP(Multi-layer Perceptron),
# XGBoost(eXtreme Gradient Boosting), RF(Random Forest), LGBM(LightGBM),
# CatBoost, LDA(Linear Discriminant Analysis), QDA(Quadratic Discriminant Analysis)
PARAMS_MAIN_CLASSIFIER = 'LR'
PARAMS_MAIN_MAX_ITERATION = 5
PARAMS_MAIN_TRAINING_RATE = 0.5
PARAMS_MAIN_THRESHOLD_EPSILON = 0.5
PARAMS_MAIN_THRESHOLD_PHI = 100
PARAMS_MAIN_THRESHOLD_PHI_ADAPT = 'None'
PARAMS_MAIN_THRESHOLD_ACCURACY = 0.01
PARAMS_MAIN_AE_IMPORTANCE_MEASURE = 'a1'
PARAMS_MAIN_BM_REBIN_METHOD = 'r1'
PARAMS_MAIN_AE_REBIN_METHOD = 'r1'
PARAMS_MAIN_ALPHA_O = 0.8


### module_load.py
# Parameters for converting numerical protected attributes to categorical variables
PARAMS_NUM_TO_CAT_METHOD_O = 'median'  # 'median' or 'quartile'
PARAMS_NUM_TO_CAT_CUTS_O = 2  # Number of cuts when using custom binning
PARAMS_NUM_TO_CAT_METHOD_Y = 'median'  # 'median' or 'quartile'
PARAMS_NUM_TO_CAT_CUTS_Y = 2  # Number of cuts when using custom binning

# Parameters for distinguishing categorical variables from numerical variables
PARAMS_CAT_FROM_NUM_BINS = 20
PARAMS_CAT_FROM_NUM_RATIO = 0.1


### eval.py
PARAMS_EVAL_H_ORDER = 'default'  # 1- N
PARAMS_EVAL_SUM = 'd1B'  # d1A, d1B
PARAMS_EVAL_CAT = 'cat-a'  # a, b
PARAMS_EVAL_NUM = 'num-a'  # a, b, c, d
PARAMS_EVAL_SCALE = 'mean' # mean, min, zscore
PARAMS_EVAL_DIST_METRIC = 'euclidean'
"""
braycurtis, canberra, chebyshev, cityblock, correlation, cosine, dice, euclidean, hamming, jaccard, jensenshannon, kulczynski1, mahalanobis, matching, minkowski, rogerstanimoto, russellrao, seuclidean, sokalmichener, sokalsneath, sqeuclidean, yule
"""
# List of supported fairness metrics:
# BNC: Between Negative Classes
# BPC: Between Positive Classes
# CUAE: Conditional Use Accuracy Equality
# EOpp: Equal Opportunity
# EO: Equalized Odds
# FDRP: False Discovery Rate Parity
# FORP: False Omission Rate Parity
# FNRB: False Negative Rate Balance
# FPRB: False Positive Rate Balance
# NPVP: Negative Predictive Value Parity
# OAE: Overall Accuracy Equality
# PPVP: Positive Predictive Value Parity
# SP: Statistical Parity
PARAMS_EVAL_METRIC_FAIRNESS = ['BNC', 'BPC', 'CUAE', 'EOpp', 'EO', 'FDRP', 'FORP', 'FNRB', 'FPRB', 'NPVP', 'OAE', 'PPVP', 'SP']
# List of supported accuracy metrics:
# ACC: Accuracy
# F1: F1 Score - harmonic mean of precision and recall
# Recall: True Positive Rate - proportion of positive cases correctly identified
# Precision: Positive Predictive Value - proportion of predicted positives that are actual positives
PARAMS_EVAL_METRIC_ACCURACY = ['ACC', 'F1', 'Recall', 'Precision']

PARAMS_EVAL_NORM = 'min-max'
# PARAMS_EVAL_NORM = 'z-score'
PARAMS_EVAL_MAX_COMPONENTS = 15
PARAMS_EVAL_SLOPE_THRESHOLD = 0.01

### transform.py
PARAMS_TRANSFORM = 'poly'  # poly, log, arcsin
PARAMS_TRANSFORM_LOG_EPSILON = 1e-5
PARAMS_TRANSFORM_MULTI = 't1'  # t1, t2, t3
PARAMS_TRANSFORM_STREAM = 'd4A'  # d4A, d4B, E1-9
PARAMS_TRANSFORM_STREAM_CONFIG = {
    'p': 102,
    'q': 173,
    'emin': 1/2,
    'emax': 2,
    'order': 0,
    'length': 10
}
PARAMS_TRANSFORM_X_MAX = 1e+9
PARAMS_TRANSFORM_N_BINS = 10
