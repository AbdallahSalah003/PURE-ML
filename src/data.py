import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import TARGET_COL, TEST_SIZE, CV_TEST_SIZE, RANDOM_STATE, CORR_METHOD, CORR_THRESHOLD
def load_splits(data_path: str): 
    df = pd.read_csv(data_path)
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    X_train, X_, y_train, Y_ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_, Y_, test_size=CV_TEST_SIZE, random_state=RANDOM_STATE, stratify=Y_
    )
    del X_, Y_ 
    print(f"Raw Data Split:\n Train: {X_train.shape}\n CV: {X_cv.shape} \n Test: {X_test.shape}\n")
    return X_train, X_cv, X_test, y_train, y_cv, y_test


def correlated_features_to_drop(X, y, method: str = CORR_METHOD, threshold: float = CORR_THRESHOLD):
    corr_mat = X.corr(method=method).abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    target_corr = X.join(y).corr(method=method)[TARGET_COL].abs()
    to_drop = []
    for col in upper.columns:
        for partner in upper.index[upper[col] > threshold].tolist():
            to_drop.append(partner if target_corr[col]>=target_corr[partner] else col)
    return list(set(to_drop))

def prepare_data_for_experiments(X_train, X_cv, X_test, y_train): 
    to_drop = correlated_features_to_drop(X_train, y_train)
    X_train_dcf = X_train.drop(columns=to_drop)
    X_cv_dcf = X_cv.drop(columns=to_drop)
    X_test_dcf = X_test.drop(columns=to_drop)

    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(X_train)
    Xcv_scaled = scaler.transform(X_cv)
    Xte_scaled = scaler.transform(X_test)

    scaler_dcf = StandardScaler()
    Xtr_dcf_scaled = scaler_dcf.fit_transform(X_train_dcf)
    Xcv_dcf_scaled = scaler_dcf.transform(X_cv_dcf)
    Xte_dcf_scaled = scaler_dcf.transform(X_test_dcf)

    return {
        "full_scaled": (Xtr_scaled, Xcv_scaled, Xte_scaled),
        "dcf_scaled": (Xtr_dcf_scaled, Xcv_dcf_scaled, Xte_dcf_scaled),
        "raw": (X_train, X_cv, X_test),
        "to_drop": to_drop
    }