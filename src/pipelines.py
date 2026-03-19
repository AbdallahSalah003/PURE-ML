from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from src.config import RANDOM_STATE, PCA_VARIANCE, EARLY_STOPPING_ROUNDS, CIEL_N_ESTIMATORS

def build_lr_svm_pipeline(model, use_pca: bool, use_smote: bool) -> Pipeline: 
    steps = []
    if use_smote: 
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    if use_pca: 
        steps.append(("pca", PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)

def build_rf_xgb_pipline(model, use_smote: bool) -> Pipeline: 
    steps = []
    if use_smote: 
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)

def get_lr_svm_models(): 
    from src.config import LR_GRID, SVM_GRID, LINEAR_SVC_GRID
    return [
        ("LR", LogisticRegression(random_state=RANDOM_STATE), LR_GRID),
        # ("SVM", SVC(probability=True, random_state=RANDOM_STATE), SVM_GRID),
        ("LINEAR_SVC", CalibratedClassifierCV(LinearSVC(random_state=RANDOM_STATE),cv=3), LINEAR_SVC_GRID)
    ]

def get_rf_xgb_models(): 
    from src.config import RF_GRID, XGB_GRID
    return [
        ("RF", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), RF_GRID),
        ("XGB", XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            device="cuda",
            random_state=RANDOM_STATE), XGB_GRID)
    ]



def refit_xgb_with_early_stopping(grid, Xtr, y_train,
                                   Xcv, y_cv,
                                   random_state: int = RANDOM_STATE) -> XGBClassifier:
    """
    after GridSearchCV find best params refit XGBoost with
    early stopping using X_cv as the eval set
    """
    best = {
        k.replace("model__", ""): v
        for k, v in grid.best_params_.items()
    }
    best.pop("n_estimators", None)
    final_xgb = XGBClassifier(
        **best,
        n_estimators          = CIEL_N_ESTIMATORS, 
        early_stopping_rounds = EARLY_STOPPING_ROUNDS,
        eval_metric           = "logloss",
        random_state          = random_state,
    )

    final_xgb.fit(
        Xtr, y_train,
        eval_set = [(Xcv, y_cv)],
        verbose  = False,
    )
    n_trees = final_xgb.best_iteration + 1
    print(f"  Early stopping: best iteration = {n_trees} trees "
          f"(out of 2000 max)")

    return final_xgb

def refit_xgb_without_early_stopping(final_model, Xtr, y_train,
                                      random_state: int = RANDOM_STATE) -> XGBClassifier:
    """
    refit XGBoost using the optimal n_estimators found by early stopping
    but WITHOUT early stopping  so learning_curve() can call .fit()
    internally without needing an eval_set
    """
    optimal_n = final_model.best_iteration + 1
    params = final_model.get_params()
    params.pop("early_stopping_rounds", None)
    params.pop("callbacks", None)
    params["n_estimators"]  = optimal_n
    params["random_state"]  = random_state

    clean_xgb = XGBClassifier(**params)
    clean_xgb.fit(Xtr, y_train)

    print(f"  Clean model (no early stopping): "
          f"n_estimators={optimal_n}")
    return clean_xgb