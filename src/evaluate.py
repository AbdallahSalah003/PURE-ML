import json 
import pandas as pd 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from src.config import RESULTS_DIR
import joblib

def evaluate_fitted(pipeline, X_train, y_train, X_cv, y_cv, X_test, y_test): 
    results = {}
    data_list = [
        ("train", X_train, y_train), 
        ("cv", X_cv, y_cv), 
        ("test", X_test, y_test)
    ]
    for split, X, y in data_list:
        preds = pipeline.predict(X)
        proba = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, "predict_proba") else None 
        results[split] = {
            "accuracy": accuracy_score(y, preds),
            "f1": f1_score(y, preds),
            "precision": precision_score(y, preds),
            "recall": recall_score(y, preds),
            "roc_auc": roc_auc_score(y, proba) if proba is not None else None,
            "conf_mat": confusion_matrix(y, preds).tolist()
        }
    return results

def build_summary(results, best_params, grids):
    rows = []
    for key, res in results.items():
        r = res["test"]
        rows.append({
            "experiment": key,
            "cv_f1": round(grids[key].best_score_, 4),
            "test_f1": round(r["f1"],4),
            "test_roc_auc": round(r["roc_auc"], 4) if r["roc_auc"] else None,
            "prescision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "accuracy": round(r["accuracy"],4 ),
            "best_params": best_params[key]
        })
    return pd.DataFrame(rows).sort_values("test_f1", ascending=False).reset_index(drop=True)


def save_results(summary, best_params): 
    csv_path = RESULTS_DIR / "summary.csv"
    params_path = RESULTS_DIR / "best_params.json"
    summary.to_csv(csv_path, index=False)
    with open(params_path, "w") as f: 
        json.dump(best_params, f, indent=2)
    

def save_models(grid_objects): 
    MODELS_DIR = RESULTS_DIR.parent / "models"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for key, grid in grid_objects.items():
        path = MODELS_DIR / f"{key}.joblib"
        joblib.dump(grid.best_estimator_, path)
    print("\nAll models saved in models dir")