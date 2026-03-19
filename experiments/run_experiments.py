import warnings
import os
import sys
import joblib
import json 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.config import DATA_PATH, RANDOM_STATE, CV_FOLDS, SCORING, LR_SVM_SCENARIOS, TREE_SCENARIOS
from src.data import load_splits, prepare_data_for_experiments
from src.pipelines import build_lr_svm_pipeline, build_rf_xgb_pipline, get_lr_svm_models, get_rf_xgb_models, refit_xgb_with_early_stopping, refit_xgb_without_early_stopping
from src.evaluate import evaluate_fitted, build_summary, save_results, find_best_threshold, evaluate_at_threshold, select_best_model
from src.analysis import plot_learning_curves

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main(): 
    X_train, X_cv, X_test, y_train, y_cv, y_test = load_splits(DATA_PATH)
    arrays = prepare_data_for_experiments(X_train, X_cv, X_test, y_train)
    CV = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_results     = {}
    best_params     = {}
    best_thresholds = {}
    grid_objects    = {}
    for scenario, (use_pca, use_dcf, use_smote) in LR_SVM_SCENARIOS.items():

        Xtr, Xcv, Xte = (
            arrays["dcf_scaled"] if use_dcf else arrays["full_scaled"]
        )

        for model_name, base_model, param_grid in get_lr_svm_models():
            key = f"{model_name}__{scenario}"
            print(f"\n{'─'*55}")
            print(f" {key}  "
                  f"(pca={use_pca}, dcf={use_dcf}, smote={use_smote})")
            print(f"{'─'*55}")

            pipeline = build_lr_svm_pipeline(base_model, use_pca, use_smote)

            grid = GridSearchCV(
                estimator  = pipeline,
                param_grid = param_grid,
                cv         = CV,
                scoring    = SCORING,
                n_jobs     = -1,
                refit      = True,
                verbose    = 0,
            )
            grid.fit(Xtr, y_train)
            best_t, cv_f1_at_t = find_best_threshold(
                grid.best_estimator_, Xcv, y_cv
            )
            best_thresholds[key] = best_t
            default_res = evaluate_at_threshold(
                grid.best_estimator_, Xte, y_test, threshold=0.5
            )
            results = evaluate_fitted(
                grid.best_estimator_,
                Xtr, y_train,
                Xcv, y_cv,
                Xte, y_test,
                threshold = best_t      
            )
            results["test_default_f1"] = default_res["f1"]

            all_results[key]  = results
            best_params[key]  = grid.best_params_
            grid_objects[key] = grid

            joblib.dump(
                grid.best_estimator_,
                MODELS_DIR / f"{key}.joblib"
            )

            print(f"  Best params    : {grid.best_params_}")
            print(f"  Threshold      : {best_t:.2f}  "
                  f"(default=0.50)")
            print(f"  CV F1          : {results['cv']['f1']:.4f}")
            print(f"  Test F1 (0.50) : {default_res['f1']:.4f}")
            print(f"  Test F1 (tuned): {results['test']['f1']:.4f}  ")
            print(f"  Test AUC       : {results['test']['roc_auc']:.4f}")
            print(f"  Test Accuracy       : {results['test']['accuracy']:.4f}")

    for scenario, use_smote in TREE_SCENARIOS.items():

        Xtr, Xcv, Xte = arrays["raw"]
        for model_name, base_model, param_grid in get_rf_xgb_models():
            key = f"{model_name}__{scenario}"
            print(f"\n{'─'*55}")
            print(f" {key}  (smote={use_smote})")
            print(f"{'─'*55}")

            pipeline = build_rf_xgb_pipline(base_model, use_smote)

            grid = GridSearchCV(
                estimator  = pipeline,
                param_grid = param_grid,
                cv         = CV,
                scoring    = SCORING,
                n_jobs     = -1,
                refit      = True,
                verbose    = 0,
            )
            grid.fit(Xtr, y_train)
            if model_name == "XGB":
                print(f"  Refitting with early stopping...")
                final_model = refit_xgb_with_early_stopping(
                    grid         = grid,
                    Xtr          = Xtr,
                    y_train      = y_train,
                    Xcv          = Xcv,
                    y_cv         = y_cv,
                    random_state = RANDOM_STATE,
                )
                clean_model = refit_xgb_without_early_stopping(
                    final_model  = final_model,   
                    Xtr          = Xtr,
                    y_train      = y_train,
                    random_state = RANDOM_STATE,
                )
                joblib.dump(clean_model, MODELS_DIR / f"{key}_clean.joblib")
            else:
                final_model = grid.best_estimator_
            best_t, cv_f1_at_t = find_best_threshold(
                final_model, Xcv, y_cv
            )
            best_thresholds[key] = best_t
            default_res = evaluate_at_threshold(
                final_model, Xte, y_test, threshold=0.5
            )
            results = evaluate_fitted(
                final_model,
                Xtr, y_train,
                Xcv, y_cv,
                Xte, y_test,
                threshold = best_t
            )
            results["test_default_f1"] = default_res["f1"]

            all_results[key]  = results
            best_params[key]  = grid.best_params_
            grid_objects[key] = grid

            joblib.dump(final_model, MODELS_DIR / f"{key}.joblib")

            print(f"  Best params    : {grid.best_params_}")
            print(f"  Threshold      : {best_t:.2f}  "
                  f"(default=0.50)")
            print(f"  CV F1          : {results['cv']['f1']:.4f}")
            print(f"  Test F1 (0.50) : {default_res['f1']:.4f}")
            print(f"  Test F1 (tuned): {results['test']['f1']:.4f}  ")
            print(f"  Test AUC       : {results['test']['roc_auc']:.4f}")
            print(f"  Test Accuracy       : {results['test']['accuracy']:.4f}")

    print(f"\n{'═'*55}")
    print(" Model selection — using CV F1 only")
    print(f"{'═'*55}")
    best_key = select_best_model(all_results)

    print(f"\n Final test scores for selected model ({best_key}):")
    print(f"  Test F1 (default 0.50) : "
          f"{all_results[best_key]['test_default_f1']:.4f}")
    print(f"  Test F1 (tuned thresh) : "
          f"{all_results[best_key]['test']['f1']:.4f}")
    print(f"  Test AUC               : "
          f"{all_results[best_key]['test']['roc_auc']:.4f}")
    print(f"  Threshold used         : "
          f"{best_thresholds[best_key]:.2f}")
    summary_df = build_summary(
        all_results, best_params, grid_objects, best_thresholds
    )
    save_results(summary_df, best_params)
    best_model_info = {
        "key":              best_key,
        "threshold":        best_thresholds[best_key],
        "cv_f1":            all_results[best_key]["cv"]["f1"],
        "test_f1_default":  all_results[best_key]["test_default_f1"],
        "test_f1_tuned":    all_results[best_key]["test"]["f1"],
        "test_auc":         all_results[best_key]["test"]["roc_auc"],
    }
    with open("results/best_model.json", "w") as f:
        json.dump(best_model_info, f, indent=2)
    print(f"\n  Best model info saved → results/best_model.json")

    plot_learning_curves(summary_df, arrays, y_train, LR_SVM_SCENARIOS)

if __name__ == "__main__":
    main()

