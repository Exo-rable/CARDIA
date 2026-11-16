"""Stage 3: train, compare, tune, evaluate and save the model.

Protocol (no leakage):
  - stratified 80/20 split FIRST; the 20% test set is touched exactly once, at the end
  - imputer + scaler + encoder + model live in ONE Pipeline, so each CV fold learns them
    from its own training part only
  - 6 candidate models are compared with 5-fold x 3-repeat stratified CV on the training set
  - the top 2 are tuned with cross-validated search, the best tuned model is selected by CV
  - final evaluation on the untouched test set, with bootstrap confidence intervals
"""
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     RepeatedStratifiedKFold, cross_validate)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from common import (BACKGROUND, CATEGORICAL, CLASSIFICATION_THRESHOLD, FEATURES, FIG, META,
                    MODEL, NUMERIC, REPORTS, SEED, get_split, load_clean,
                    positive_class_probability)

warnings.filterwarnings("ignore")
BLUE, ORANGE, GREY = "#2a6f97", "#e07a1f", "#6b7280"


def make_pipeline(model) -> Pipeline:
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, eval_metric="logloss",
                                 random_state=SEED, n_jobs=1),
        "SVM (RBF)": SVC(probability=True, random_state=SEED),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-2,
                                              max_iter=3000, random_state=SEED),
    }


GRIDS = {
    "Logistic Regression": {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
    "Decision Tree": {"model__max_depth": [2, 3, 4, 5, 6],
                      "model__min_samples_leaf": [1, 3, 5, 10]},
    "Random Forest": {"model__n_estimators": [200, 400],
                      "model__max_depth": [3, 5, 8, None],
                      "model__min_samples_leaf": [1, 3, 5],
                      "model__max_features": ["sqrt", 0.5]},
    "XGBoost": {"model__n_estimators": [100, 200, 300],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__min_child_weight": [1, 3]},
    "SVM (RBF)": {"model__C": [0.3, 1, 3, 10], "model__gamma": ["scale", 0.01, 0.03, 0.1]},
    "Neural Network (MLP)": {"model__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
                             "model__alpha": [1e-3, 1e-2, 1e-1, 1.0]},
}


def grid_size(grid):
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


def metrics_at(y_true, proba, thr=CLASSIFICATION_THRESHOLD):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "recall_sensitivity": recall_score(y_true, pred),
        "specificity": tn / (tn + fp),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(y_true, proba, fn, n=2000):
    rng = np.random.default_rng(SEED)
    y_true = np.asarray(y_true)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(fn(y_true[idx], proba[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    X_train, X_test, y_train, y_test = get_split(df)
    print(f"Train {len(X_train)} rows | Test {len(X_test)} rows | "
          f"disease rate train {y_train.mean():.1%} / test {y_test.mean():.1%}")

    # ---------- 1. compare candidates with repeated stratified CV ----------
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    scoring = ["accuracy", "recall", "precision", "f1", "roc_auc"]
    rows = []
    for name, model in candidates().items():
        s = cross_validate(make_pipeline(model), X_train, y_train, cv=cv,
                           scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for m in scoring:
            row[f"{m}_mean"] = s[f"test_{m}"].mean()
            row[f"{m}_std"] = s[f"test_{m}"].std()
        rows.append(row)
        print(f"  {name:22s} AUC {row['roc_auc_mean']:.3f} +/- {row['roc_auc_std']:.3f} "
              f"| recall {row['recall_mean']:.3f} | acc {row['accuracy_mean']:.3f}")
    cv_table = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    cv_table.to_csv(REPORTS / "model_comparison_cv.csv", index=False)

    # ---------- 2. tune the top two (by CV ROC-AUC) ----------
    top2 = cv_table["model"].head(2).tolist()
    print("\nTuning:", top2)
    tune_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
    tuned = {}
    for name in top2:
        pipe = make_pipeline(candidates()[name])
        grid = GRIDS[name]
        if grid_size(grid) <= 60:
            search = GridSearchCV(pipe, grid, cv=tune_cv, scoring="roc_auc", n_jobs=-1)
        else:
            search = RandomizedSearchCV(pipe, grid, n_iter=60, cv=tune_cv,
                                        scoring="roc_auc", n_jobs=-1, random_state=SEED)
        search.fit(X_train, y_train)
        tuned[name] = search
        print(f"  {name:22s} best CV AUC {search.best_score_:.3f}  {search.best_params_}")

    best_name = max(tuned, key=lambda k: tuned[k].best_score_)
    best_pipe = tuned[best_name].best_estimator_  # already refit on the full training set
    print(f"\nSelected model: {best_name}")

    # ---------- 3. final, single evaluation on the untouched test set ----------
    p_test = positive_class_probability(best_pipe, X_test)
    p_train = positive_class_probability(best_pipe, X_train)
    test_m = metrics_at(y_test, p_test)
    train_m = metrics_at(y_train, p_train)
    test_m["roc_auc_ci95"] = bootstrap_ci(y_test, p_test, roc_auc_score)
    test_m["accuracy_ci95"] = bootstrap_ci(y_test, p_test,
                                           lambda a, b: accuracy_score(a, (b >= 0.5).astype(int)))
    test_m["recall_ci95"] = bootstrap_ci(y_test, p_test,
                                         lambda a, b: recall_score(a, (b >= 0.5).astype(int)))
    print("\nTEST SET:", {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in test_m.items() if k != "tn"})
    print(f"Train AUC {train_m['roc_auc']:.3f} vs test AUC {test_m['roc_auc']:.3f} "
          f"(gap {train_m['roc_auc'] - test_m['roc_auc']:.3f})")

    # Do not score alternative candidates on the held-out test set. Model comparison
    # belongs to training-only cross-validation; this test set is reserved for the
    # selected model's final evaluation above.

    # threshold table (healthcare: a lower threshold trades precision for recall)
    thr_rows = []
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        m = metrics_at(y_test, p_test, thr)
        thr_rows.append({"threshold": thr, **{k: round(m[k], 3) for k in
                         ["recall_sensitivity", "specificity", "precision", "accuracy"]},
                         "missed_patients_fn": m["fn"], "false_alarms_fp": m["fp"]})
    pd.DataFrame(thr_rows).to_csv(REPORTS / "threshold_analysis.csv", index=False)

    # ---------- 4. figures ----------
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    # model comparison (CV AUC with error bars)
    t = cv_table.sort_values("roc_auc_mean")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(t["model"], t["roc_auc_mean"], xerr=t["roc_auc_std"], color=BLUE, capsize=3)
    ax.set_xlim(0.7, 1.0)
    ax.set_xlabel("Cross-validated ROC-AUC (mean ± sd, 5-fold × 3 repeats)")
    ax.set_title("Model comparison on the training set")
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison_cv.png", dpi=160)
    plt.close(fig)

    # confusion matrix
    cm = np.array([[test_m["tn"], test_m["fp"]], [test_m["fn"], test_m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    ax.set_yticks([0, 1], ["No disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"Confusion matrix ({best_name}, test set)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # ROC and PR curves
    fpr, tpr, _ = roc_curve(y_test, p_test)
    prec, rec, _ = precision_recall_curve(y_test, p_test)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(fpr, tpr, color=BLUE, lw=2, label=f"AUC = {test_m['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color=GREY)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)",
                title="ROC curve (test set)")
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].plot(rec, prec, color=ORANGE, lw=2, label=f"AP = {test_m['pr_auc']:.3f}")
    axes[1].axhline(y_test.mean(), ls="--", color=GREY)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test set)")
    axes[1].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "roc_pr_curves.png", dpi=160)
    plt.close(fig)

    # calibration: is "40% risk" really about 40%?
    pt, pp = calibration_curve(y_test, p_test, n_bins=5, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect calibration")
    ax.plot(pp, pt, "o-", color=BLUE, label=best_name)
    ax.set(xlabel="Predicted probability", ylabel="Observed fraction with disease",
           title="Calibration curve (test set)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "calibration_curve.png", dpi=160)
    plt.close(fig)

    # ---------- 5. save model + metadata ----------
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(best_pipe, MODEL)
    bg = X_train.sample(n=min(100, len(X_train)), random_state=SEED)
    bg.to_csv(BACKGROUND, index=False)

    meta = {
        "selected_model": best_name,
        "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                        for k, v in tuned[best_name].best_params_.items()},
        "tuned_cv_auc": tuned[best_name].best_score_,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "features": FEATURES,
        "test_metrics": test_m, "train_metrics": train_m,
        "cv_table": cv_table.round(4).to_dict("records"),
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "seed": SEED,
    }
    META.write_text(json.dumps(meta, indent=2, default=float))
    print("\nSaved", MODEL.name, "and", META.name)


if __name__ == "__main__":
    main()
