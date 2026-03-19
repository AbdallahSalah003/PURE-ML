from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "cleaned.csv"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


TARGET_COL     = "phishing"
TEST_SIZE      = 0.40      
CV_TEST_SIZE   = 0.50      
RANDOM_STATE   = 42
CV_FOLDS       = 5
SCORING        = "f1"
CORR_METHOD    = "spearman"
CORR_THRESHOLD = 0.90
PCA_VARIANCE   = 0.95
EARLY_STOPPING_ROUNDS = 30
CIEL_N_ESTIMATORS = 2000

LR_SVM_SCENARIOS = {
    "S1": (False, False, False),
    "S2": (False, False, True),
    "S3": (False, True,  False),
    "S4": (False, True,  True),
    "S5": (True,  False, False),
    "S6": (True,  False, True),
}

TREE_SCENARIOS = {
    "S1": False,
    "S2": True,
}

LR_GRID = [
    {
        "model__C":        [0.01, 0.1, 1, 10, 100], 
        "model__penalty": ["l1"],
        "model__solver": ["saga"],
        "model__max_iter": [2000],
    },
    {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__penalty": ["l2"],
        "model__solver": ["saga"],
        "model__max_iter": [2000],
    }
]
LINEAR_SVC_GRID = [
    {
        "model__estimator__C":        [0.01, 0.1, 1, 10, 100], 
        "model__estimator__penalty":  ["l2"],
        "model__estimator__loss":     ["squared_hinge"],
        "model__estimator__max_iter": [2000],
    },
    {
        "model__estimator__C":        [0.01, 0.1, 1, 10, 100],
        "model__estimator__penalty":  ["l1"],
        "model__estimator__loss":     ["squared_hinge"],
        "model__estimator__dual":     [False], # only needed for l1 regularizer
        "model__estimator__max_iter": [2000],
    },
]
SVM_GRID = {
    "model__C":      [0.01, 0.1, 1, 10, 100], # inv reg. -> control margin softness
    "model__kernel": ["rbf"], 
    "model__gamma":  ["scale", "auto"] # only affect rbf kernel 
    # control influence of a training point 
    # small gamma, large gamma -> smooth decision boundary, complex boundary
}

# RF_GRID = {
#     "model__n_estimators":      [300, 500], # number of trees
#     "model__max_depth":         [None, 10, 20], # deep trees overfit!
#     "model__min_samples_split": [2, 5, 10], # min samples required to split node
#      # 2 -> aggressive splitting, 5-> conservative splitting, higher values reeduce overfitting
#     "model__min_samples_leaf": [1, 2, 4], # this reduces overfitting
#     "model__max_features":      ["sqrt", "log2"] # features considered per split sqrt(n_features) or log2(n_features)
# }
RF_GRID = {
    "model__max_depth":         [8, 12, 16],
    "model__min_samples_split": [10, 20],
    "model__min_samples_leaf":  [4, 8],
    "model__n_estimators":      [300, 500],
    "model__max_features":      ["sqrt", "log2"],
}
# XGB_GRID = {
#     "model__n_estimators":  [300, 500], 
#     "model__max_depth":     [3, 6, 9], 
#     "model__learning_rate": [0.01, 0.05, 0.1], 
#     "model__subsample":     [0.8, 1.0], 
#     "model__device":        ["cuda"],
# }

XGB_GRID = {
    "model__max_depth":        [4, 6], # higher depth -> oveerfitting risk
    "model__min_child_weight": [5, 10], # min samples in a leaf beforee a split is allowed
    "model__gamma":            [0.1, 0.5], # min loss reduction required to make a split, reject noise splits
    "model__reg_lambda":       [5.0, 15.0], # L2 penalty on leaf weights, reeduce overconfident preedictions
    "model__learning_rate":    [0.05, 0.1], # step size of boosting
    "model__n_estimators":     [500], # number of boosting rounds, more rounds = better acc but slower
    "model__subsample":        [0.8], # fraction of training data used per tree
#     # 1.0 -> use all data, 0.8 -> random subset, subsampling reduce overfitting
    "model__device":           ["cuda"],
}