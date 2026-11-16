"""Stage 4: explainability. Works for ANY model chosen in stage 3.

Methods:
  - SHAP (model-agnostic permutation explainer, in PROBABILITY space and on the original
    13 features, so a contribution reads "+12 percentage points of risk")
  - Permutation importance (a second, independent global view)
  - LIME (a local second opinion for one patient)
  - For linear models: the odds-ratio table from the coefficients
Also checks that SHAP and permutation importance agree on the feature ranking.
"""
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance

from common import (BACKGROUND, CATEGORICAL, DISPLAY, FEATURES, FIG, LABELS, MODEL, REPORTS,
                    SEED, SHORT, get_split, load_clean, positive_class_probability,
                    probabilities_in_class_order, value_label)

warnings.filterwarnings("ignore")
BLUE, ORANGE = "#2a6f97", "#e07a1f"


def predict_fn(pipe):
    def f(arr):
        return positive_class_probability(pipe, pd.DataFrame(np.asarray(arr), columns=FEATURES))
    return f


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    pipe = joblib.load(MODEL)
    X_train, X_test, y_train, y_test = get_split(load_clean())
    background = pd.read_csv(BACKGROUND)[FEATURES]
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

    # ---------------- SHAP (global) ----------------
    masker = shap.maskers.Independent(background.to_numpy(float), max_samples=100)
    explainer = shap.Explainer(predict_fn(pipe), masker, algorithm="permutation",
                               feature_names=[SHORT[c] for c in FEATURES], seed=SEED)
    exp = explainer(X_test.to_numpy(float), max_evals=2 * len(FEATURES) + 1 + 600)
    vals = exp.values  # (n_test, n_features), in probability points
    mean_abs = pd.Series(np.abs(vals).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    mean_abs.to_csv(REPORTS / "shap_mean_abs_importance.csv", header=["mean_abs_shap"])

    # additivity check: base value + sum(contributions) must equal the model output
    recon = exp.base_values + vals.sum(axis=1)
    err = np.abs(recon - predict_fn(pipe)(X_test.to_numpy(float))).max()
    print(f"SHAP additivity check, max error: {err:.6f}")
    assert err < 1e-6, "SHAP contributions do not add up to the prediction"

    fig, ax = plt.subplots(figsize=(7, 4.6))
    s = mean_abs.sort_values()
    ax.barh([SHORT[c] for c in s.index], s.values * 100, color=BLUE)
    ax.set_xlabel("Mean |SHAP value| (percentage points of risk)")
    ax.set_title("Global feature importance (SHAP, test set)")
    fig.tight_layout()
    fig.savefig(FIG / "shap_global_bar.png", dpi=160)
    plt.close(fig)

    # beeswarm
    plt.figure(figsize=(8, 5.4))
    shap.plots.beeswarm(shap.Explanation(values=vals * 100, base_values=exp.base_values * 100,
                                         data=X_test.to_numpy(float),
                                         feature_names=[SHORT[c] for c in FEATURES]),
                        max_display=13, show=False)
    plt.xlabel("SHAP value (percentage points of risk)")
    plt.title("Per-patient SHAP values (colour = numeric code; category codes are not ordered)",
              fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG / "shap_beeswarm.png", dpi=160)
    plt.close()

    # waterfall for the highest- and lowest-risk test patients
    probs = predict_fn(pipe)(X_test.to_numpy(float))
    for tag, idx in [("high_risk", int(np.argmax(probs))), ("low_risk", int(np.argmin(probs)))]:
        row = X_test.iloc[idx]
        e = shap.Explanation(values=vals[idx] * 100, base_values=float(exp.base_values[idx] * 100),
                             data=np.array([value_label(c, row[c]) if c in LABELS else f"{row[c]:g}"
                                            for c in FEATURES], dtype=object),
                             feature_names=[SHORT[c] for c in FEATURES])
        plt.figure(figsize=(7.5, 5))
        shap.plots.waterfall(e, max_display=10, show=False)
        plt.title(f"Example patient ({tag.replace('_', ' ')}): predicted risk {probs[idx]:.0%}",
                  fontsize=10)
        plt.tight_layout()
        plt.savefig(FIG / f"shap_waterfall_{tag}.png", dpi=160, bbox_inches="tight")
        plt.close()

    # ---------------- permutation importance ----------------
    pi = permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=50,
                                random_state=SEED)
    pi_s = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
    pi_s.to_csv(REPORTS / "permutation_importance.csv", header=["auc_drop"])
    fig, ax = plt.subplots(figsize=(7, 4.6))
    s = pi_s.sort_values()
    ax.barh([SHORT[c] for c in s.index], s.values, xerr=pi.importances_std[
        [FEATURES.index(c) for c in s.index]], color=ORANGE, capsize=2)
    ax.set_xlabel("Drop in test ROC-AUC when the feature is shuffled")
    ax.set_title("Permutation importance (test set)")
    fig.tight_layout()
    fig.savefig(FIG / "permutation_importance.png", dpi=160)
    plt.close(fig)

    # agreement between the two global methods
    rho = mean_abs.rank().corr(pi_s.reindex(mean_abs.index).rank(), method="spearman")
    top5_overlap = len(set(mean_abs.head(5).index) & set(pi_s.head(5).index))
    print(f"SHAP vs permutation importance: Spearman rho = {rho:.2f}, top-5 overlap = {top5_overlap}/5")

    # ---------------- LIME (local second opinion) ----------------
    fill = X_train.copy()
    for c in FEATURES:
        fill[c] = fill[c].fillna(fill[c].median() if c not in CATEGORICAL else fill[c].mode()[0])
    lime_exp = LimeTabularExplainer(
        fill.to_numpy(float), feature_names=[SHORT[c] for c in FEATURES],
        class_names=["No disease", "Disease"],
        categorical_features=[FEATURES.index(c) for c in CATEGORICAL],
        mode="classification", random_state=SEED, discretize_continuous=True)
    idx = int(np.argmax(probs))
    le = lime_exp.explain_instance(X_test.iloc[idx].fillna(0).to_numpy(float),
                                   lambda a: probabilities_in_class_order(
                                       pipe, pd.DataFrame(a, columns=FEATURES), [0, 1]
                                   ),
                                   num_features=8, num_samples=3000)
    fig = le.as_pyplot_figure()
    fig.set_size_inches(7.5, 4.2)
    plt.title("LIME explanation, same high-risk patient", fontsize=10)
    plt.tight_layout()
    fig.savefig(FIG / "lime_example.png", dpi=160)
    plt.close(fig)

    # ---------------- linear models: odds ratios ----------------
    model = pipe.named_steps["model"]
    if hasattr(model, "coef_") and model.coef_.shape[0] == 1:
        names = pipe.named_steps["prep"].get_feature_names_out()
        coefs = pd.DataFrame({"term": names, "coefficient": model.coef_[0],
                              "odds_ratio": np.exp(model.coef_[0])})
        coefs = coefs.reindex(coefs["coefficient"].abs().sort_values(ascending=False).index)
        coefs.to_csv(REPORTS / "logistic_coefficients.csv", index=False)
        print("\nLargest coefficients (standardised numerics, one-hot categories):")
        print(coefs.head(8).round(3).to_string(index=False))

    print("\nSHAP ranking (mean |contribution|, percentage points):")
    print((mean_abs * 100).round(2).head(8).to_string())
    (REPORTS / "explainability_summary.json").write_text(json.dumps({
        "shap_additivity_max_error": float(err),
        "spearman_shap_vs_permutation": float(rho),
        "top5_overlap": int(top5_overlap),
        "top_features_shap": [DISPLAY[c] for c in mean_abs.head(5).index],
    }, indent=2))


if __name__ == "__main__":
    main()
