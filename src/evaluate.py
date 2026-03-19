import json 
import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from src.config import RESULTS_DIR
import joblib

def find_best_threshold(model, X_cv, y_cv,
                        thresholds=np.arange(0.30, 0.71, 0.01)):
    best_threshold = 0.5
    best_f1        = 0.0

    proba_cv = model.predict_proba(X_cv)[:, 1]

    for t in thresholds:
        preds = (proba_cv >= t).astype(int)
        f1    = f1_score(y_cv, preds, zero_division=0)
        if f1 > best_f1:
            best_f1        = f1
            best_threshold = round(float(t), 2)

    print(f"  Best threshold: {best_threshold:.2f}  "
          f"(CV F1 at threshold: {best_f1:.4f})")
    return best_threshold, best_f1


def evaluate_at_threshold(model, X, y, threshold: float) -> dict:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    return {
        "threshold": threshold,
        "f1":        f1_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "recall":    recall_score(y, preds, zero_division=0),
        "accuracy":  accuracy_score(y, preds),
        "roc_auc":   roc_auc_score(y, proba),
        "cm":        confusion_matrix(y, preds).tolist(),
    }


def evaluate_fitted(pipeline, X_tr, y_tr, X_cv, y_cv,
                    X_te, y_te, threshold: float = 0.5) -> dict:
    results = {}
    for split, X, y in [("train", X_tr, y_tr),
                         ("cv",    X_cv, y_cv),
                         ("test",  X_te,  y_te)]:
        results[split] = evaluate_at_threshold(pipeline, X, y, threshold)
    return results

def select_best_model(all_results: dict) -> str:
    best_key    = None
    best_cv_f1  = 0.0

    for key, res in all_results.items():
        cv_f1 = res["cv"]["f1"]
        if cv_f1 > best_cv_f1:
            best_cv_f1 = cv_f1
            best_key   = key

    print(f"\n  Best model selected by CV F1: {best_key} "
          f"(CV F1 = {best_cv_f1:.4f})")
    return best_key

def build_summary(all_results: dict, best_params: dict,
                  best_thresholds: dict) -> pd.DataFrame:
    rows = []
    for key, res in all_results.items():
        r    = res["test"]
        rows.append({
            "experiment":       key,
            "cv_f1":            round(res["cv"]["f1"],   4),
            "test_f1_default":  round(res["test_default_f1"], 4),
            "test_f1_tuned":    round(r["f1"],            4),
            "threshold":        round(best_thresholds.get(key, 0.5), 2),
            "test_auc":         round(r["roc_auc"],       4),
            "precision":        round(r["precision"],     4),
            "recall":           round(r["recall"],        4),
            "accuracy":         round(r["accuracy"],      4),
            "best_params":      str(best_params[key]),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("cv_f1", ascending=False)  
        .reset_index(drop=True)
    )

def save_results(summary, best_params): 
    csv_path = RESULTS_DIR / "summary.csv"
    params_path = RESULTS_DIR / "best_params.json"
    summary.to_csv(csv_path, index=False)
    with open(params_path, "w") as f: 
        json.dump(best_params, f, indent=2)
    