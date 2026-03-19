from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from src.config import RANDOM_STATE, PCA_VARIANCE

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