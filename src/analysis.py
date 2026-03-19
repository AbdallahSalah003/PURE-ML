import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold

from src.config import RANDOM_STATE, CV_FOLDS, RESULTS_DIR

MODELS_DIR  = RESULTS_DIR.parent / "models"
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def plot_learning_curves(summary_df, arrays: dict, y_train,
                          lr_svm_scenarios: dict,
                          train_sizes=np.linspace(0.1, 1.0, 8)):
    CV     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    top4 = summary_df.nlargest(4, "test_f1_tuned")["experiment"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for ax, key in zip(axes_flat, top4):
        model_prefix, scenario = key.split("__")

        if model_prefix in ("RF", "XGB"):
            Xtr = arrays["raw"][0]
        else:
            _, use_dcf, _ = lr_svm_scenarios[scenario]
            Xtr = arrays["dcf_scaled"][0] if use_dcf else arrays["full_scaled"][0]


        if model_prefix == "XGB":
            clean_path = MODELS_DIR / f"{key}_clean.joblib"
            if clean_path.exists():
                estimator = joblib.load(clean_path)
            else:
                estimator = joblib.load(MODELS_DIR / f"{key}.joblib")
                estimator.set_params(
                    early_stopping_rounds = None,
                    n_estimators          = estimator.best_iteration + 1
                )
        else:
            estimator = joblib.load(MODELS_DIR / f"{key}.joblib")

        train_sizes_abs, train_scores, cv_scores = learning_curve(
            estimator, Xtr, y_train,
            train_sizes  = train_sizes,
            cv           = CV,
            scoring      = "f1",
            n_jobs       = -1,
            shuffle      = True,
            random_state = RANDOM_STATE,
        )

        train_err     = 1 - train_scores.mean(axis=1)
        cv_err        = 1 - cv_scores.mean(axis=1)
        train_err_std = train_scores.std(axis=1)
        cv_err_std    = cv_scores.std(axis=1)
        gap           = cv_err[-1] - train_err[-1]

        ax.plot(train_sizes_abs, train_err, "o-", color="steelblue",
                label="Train error")
        ax.fill_between(train_sizes_abs,
                        train_err - train_err_std,
                        train_err + train_err_std,
                        alpha=0.15, color="steelblue")

        ax.plot(train_sizes_abs, cv_err, "o-", color="darkorange",
                label="CV error")
        ax.fill_between(train_sizes_abs,
                        cv_err - cv_err_std,
                        cv_err + cv_err_std,
                        alpha=0.15, color="darkorange")

        ax.set_title(f"{key}  |  gap={gap:.3f}", fontsize=11)
        ax.set_xlabel("Training samples")
        ax.set_ylabel("Error  (1 - F1)")
        ax.set_ylim(-0.01, 0.30)
        ax.legend(fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    plt.suptitle(
        "Learning curves — top 4 models\n"
        "Train error ~= bias  |  gap between curves ~= variance",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "learning_curves_top4.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {PLOTS_DIR / 'learning_curves_top4.png'}")


