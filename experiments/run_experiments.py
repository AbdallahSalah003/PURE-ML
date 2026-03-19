import warnings
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.config import DATA_PATH, RANDOM_STATE, CV_FOLDS, SCORING, LR_SVM_SCENARIOS, TREE_SCENARIOS
from src.data import load_splits, prepare_data_for_experiments
from src.pipelines import build_lr_svm_pipeline, build_rf_xgb_pipline, get_lr_svm_models, get_rf_xgb_models
from src.evaluate import evaluate_fitted, build_summary, save_results, save_models
from src.analysis import plot_learning_curves

def main(): 
    X_train, X_cv, X_test, y_train, y_cv, y_test = load_splits(DATA_PATH)
    experiments_data = prepare_data_for_experiments(X_train, X_cv, X_test, y_train)
    CV = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_results, best_params, grids = {}, {}, {}
    # for scenario, (use_pca, use_dcf, use_smote) in LR_SVM_SCENARIOS.items(): 
    #     Xtr, Xcv, Xte = experiments_data["dcf_scaled"] if use_dcf else experiments_data["full_scaled"]
    #     for model_name, base_model, param_grid in get_lr_svm_models():
    #         key = f"{model_name}__{scenario}"
    #         print(f"\n {key} (pca={use_pca}, dcf={use_dcf}, smote={use_smote})")
    #         grid = GridSearchCV(
    #             estimator= build_lr_svm_pipeline(base_model, use_pca, use_smote),
    #             param_grid = param_grid,
    #             cv = CV,
    #             scoring = SCORING,
    #             n_jobs = -1,
    #             refit = True,
    #             verbose = 0
    #         )
    #         grid.fit(Xtr, y_train)
    #         all_results[key] = evaluate_fitted(grid.best_estimator_, Xtr, y_train, Xcv, y_cv, Xte, y_test)
    #         best_params[key] = grid.best_params_
    #         grids[key] = grid 

    #         print("--------------------------------------------")
    #         print(f"Best Params: {grid.best_params_}")
    #         print(f"CV f1: {grid.best_score_}")
    #         print(f"Test f1: {all_results[key]['test']['f1']}")
    #         print(f"Test Precision: {all_results[key]['test']['precision']}")
    #         print(f"Test Recall: {all_results[key]['test']['recall']}")
    #         print(f"Test Accuracy: {all_results[key]['test']['accuracy']}")
    #         print(f"Test Roc-Auc: {all_results[key]['test']['roc_auc']}")
    for scenario, use_smote in TREE_SCENARIOS.items():
        Xtr, Xcv, Xte = experiments_data["raw"]
        for model_name, base_model, param_grid in get_rf_xgb_models():
            key = f"{model_name}__{scenario}"
            print(f"\n {key} (smote={use_smote})")

            grid = GridSearchCV(
                estimator=build_rf_xgb_pipline(base_model, use_smote),
                param_grid=param_grid,
                cv=CV,
                scoring=SCORING,
                n_jobs=-1,
                refit=True,
                verbose=0
            )
            grid.fit(Xtr, y_train)
            all_results[key] = evaluate_fitted(grid.best_estimator_, Xtr, y_train, Xcv, y_cv, Xte, y_test)
            best_params[key] = grid.best_params_
            grids[key] = grid 

            print("--------------------------------------------")
            print(f"Best Params: {grid.best_params_}")
            print(f"CV f1: {grid.best_score_}")
            print(f"Test f1: {all_results[key]['test']['f1']}")
            print(f"Test Precision: {all_results[key]['test']['precision']}")
            print(f"Test Recall: {all_results[key]['test']['recall']}")
            print(f"Test Accuracy: {all_results[key]['test']['accuracy']}")
            print(f"Test Roc-Auc: {all_results[key]['test']['roc_auc']}")
    
    summary_df = build_summary(all_results, best_params, grids)
    save_results(summary_df, best_params)
    save_models(grids)
    print("\nFinal Results sorted by test F1")

    plot_learning_curves(summary_df, experiments_data, y_train, LR_SVM_SCENARIOS)

if __name__ == "__main__":
    main()

