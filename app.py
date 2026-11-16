"""Explainable AI-Based Healthcare Risk Prediction System: Streamlit app.

Run locally:   streamlit run app.py
The app only LOADS the model trained by src/03_train.py; it never retrains.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from common import (BACKGROUND, CATEGORICAL, CLEAN, DISPLAY, FEATURES, FIG, LABELS, META,  # noqa: E402
                    CLASSIFICATION_THRESHOLD, MODEL, NUMERIC, RANGES, REPORTS, SEED, SHORT,
                    positive_class_probability, value_label)

st.set_page_config(page_title="Explainable Heart Risk Predictor", page_icon="🫀", layout="wide")

ORANGE, BLUE = "#e07a1f", "#2a6f97"
PRESETS = {
    "Starting profile (median values)": dict(age=55, sex=1, cp=1, trestbps=130, chol=240, fbs=0,
                                         restecg=1, thalach=152, exang=0, oldpeak=0.8,
                                         slope=1, ca=0, thal=2),
    "Example: lower-risk profile": dict(age=45, sex=0, cp=2, trestbps=120, chol=200, fbs=0,
                                        restecg=1, thalach=170, exang=0, oldpeak=0.0,
                                        slope=2, ca=0, thal=2),
    "Example: higher-risk profile": dict(age=62, sex=1, cp=0, trestbps=150, chol=270, fbs=0,
                                         restecg=0, thalach=120, exang=1, oldpeak=3.0,
                                         slope=1, ca=2, thal=3),
}


# ----------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading model...")
def load_resources():
    pipe = joblib.load(MODEL)
    meta = json.loads(META.read_text())
    bg = pd.read_csv(BACKGROUND)[FEATURES]

    def predict(arr):
        return positive_class_probability(
            pipe, pd.DataFrame(np.asarray(arr, dtype=float), columns=FEATURES)
        )

    masker = shap.maskers.Independent(bg.to_numpy(float), max_samples=100)
    explainer = shap.Explainer(predict, masker, algorithm="permutation", seed=SEED)
    train_ranges = {}
    if CLEAN.exists():
        d = pd.read_csv(CLEAN)
        train_ranges = {c: (float(d[c].min()), float(d[c].max())) for c in NUMERIC}
    return pipe, meta, explainer, train_ranges


@st.cache_data(show_spinner=False)
def explain_patient(values: tuple):
    _, _, explainer, _ = load_resources()
    e = explainer(np.array([values], dtype=float), max_evals=2 * len(FEATURES) + 1 + 600)
    return e.values[0], float(e.base_values[0])


def show_image(path: Path, caption: str | None = None):
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


# ----------------------------------------------------------------- inputs
def apply_preset():
    for k, v in PRESETS[st.session_state["preset"]].items():
        st.session_state[f"w_{k}"] = float(v) if k == "oldpeak" else v


if "w_age" not in st.session_state:
    for k, v in PRESETS["Starting profile (median values)"].items():
        st.session_state[f"w_{k}"] = float(v) if k == "oldpeak" else v


def categorical_input(feature: str):
    options = list(LABELS[feature].keys())
    st.selectbox(DISPLAY[feature], options, key=f"w_{feature}",
                 format_func=lambda v, f=feature: LABELS[f][v])


def numeric_input(feature: str):
    lo, hi, step = RANGES[feature]
    if feature == "oldpeak":
        st.slider(DISPLAY[feature], float(lo), float(hi), step=float(step), key=f"w_{feature}")
    else:
        st.slider(DISPLAY[feature], int(lo), int(hi), step=int(step), key=f"w_{feature}")


try:
    pipe, meta, _explainer, train_ranges = load_resources()
except Exception as exc:  # missing or incompatible model file
    st.error("The trained model could not be loaded. Run `python src/run_all.py` first to "
             "create it, then restart the app.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("Patient data")
    st.selectbox("Load a profile", list(PRESETS), key="preset", on_change=apply_preset)
    st.caption("Change any value below; the estimate updates immediately.")
    with st.expander("Demographics and vitals", expanded=True):
        numeric_input("age")
        categorical_input("sex")
        numeric_input("trestbps")
        numeric_input("chol")
        categorical_input("fbs")
    with st.expander("Symptoms and resting tests", expanded=True):
        categorical_input("cp")
        categorical_input("restecg")
        categorical_input("exang")
    with st.expander("Exercise stress test", expanded=True):
        numeric_input("thalach")
        numeric_input("oldpeak")
        categorical_input("slope")
        categorical_input("ca")
        categorical_input("thal")

# ----------------------------------------------------------------- prediction
patient = {f: st.session_state[f"w_{f}"] for f in FEATURES}
x = pd.DataFrame([patient])[FEATURES]
prob = float(positive_class_probability(pipe, x)[0])
predicted_positive = prob >= CLASSIFICATION_THRESHOLD
class_probability = prob if predicted_positive else 1 - prob
contrib, base = explain_patient(tuple(float(patient[f]) for f in FEATURES))
contrib_pts = contrib * 100

st.title("Explainable Heart Disease Risk Predictor")
st.caption("AI Foundations (CSE 276) course project. Educational decision-support prototype, "
           "not a medical device and not a diagnosis.")

if meta["versions"]["scikit-learn"] != sklearn.__version__:
    st.warning(f"This model was trained with scikit-learn {meta['versions']['scikit-learn']} but "
               f"{sklearn.__version__} is installed. Re-run `python src/run_all.py` to retrain.")

tab_pred, tab_perf, tab_xai, tab_data, tab_about = st.tabs(
    ["AI assessment", "Model performance", "Explainability", "Data and cleaning", "About and limits"])

# ============================== TAB 1: risk assessment ==============================
with tab_pred:
    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.subheader("Your assessment")
        st.write(
            "The model found a pattern associated with the heart-disease dataset category."
            if predicted_positive else
            "The model did not identify the recorded heart-disease pattern in the information entered."
        )
        st.metric("Model-estimated probability for the predicted dataset class",
                  f"{class_probability:.1%}")
        st.caption(
            "This is a research-model output, not a clinically calibrated personal risk estimate, "
            "medical diagnosis, or treatment recommendation."
        )

        for f, (lo, hi) in train_ranges.items():
            if not (lo <= patient[f] <= hi):
                st.warning(f"{SHORT[f]} = {patient[f]:g} is outside the range the model saw in "
                           f"training ({lo:g} to {hi:g}); the model's output may be less reliable.")

        order = np.argsort(-np.abs(contrib_pts))
        up = [i for i in order if contrib_pts[i] > 0.5][:3]
        down = [i for i in order if contrib_pts[i] < -0.5][:3]

        def phrase(i):
            f = FEATURES[i]
            return f"**{SHORT[f]} = {value_label(f, patient[f])}** ({contrib_pts[i]:+.1f} points)"

        st.subheader("What influenced the AI result?")
        if up:
            st.markdown("Information moving the positive-class model output **up**: " + "; ".join(phrase(i) for i in up) + ".")
        if down:
            st.markdown("Information moving the positive-class model output **down**: " + "; ".join(phrase(i) for i in down) + ".")
        st.caption("These are associations learned by the model, not medical causes or individual diagnoses.")

    with right:
        st.subheader("Why this estimate?")
        top = list(order[:8])[::-1]
        labels = [f"{SHORT[FEATURES[i]]} = {value_label(FEATURES[i], patient[FEATURES[i]])}" for i in top]
        vals = [contrib_pts[i] for i in top]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.barh(labels, vals, color=[ORANGE if v > 0 else BLUE for v in vals])
        ax.axvline(0, color="#444", lw=0.8)
        span = max(abs(v) for v in vals) or 1
        for yi, v in enumerate(vals):
            ax.text(v + (0.04 * span if v >= 0 else -0.04 * span), yi, f"{v:+.1f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=9)
        ax.set_xlim(min(vals + [0]) - 0.22 * span, max(vals + [0]) + 0.22 * span)
        ax.set_xlabel("Change in positive-class model output (percentage points)")
        ax.set_title(f"Model output: {prob:.0%} for the positive dataset class", fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Orange bars move the positive-class model output upward; blue bars move it downward. "
                   "These SHAP contributions explain the model output and do not show causes.")
        others = [i for i in order[8:]]
        if others:
            st.caption(f"The remaining {len(others)} features contribute {sum(contrib_pts[i] for i in others):+.1f} points in total.")

    st.info("This tool supports learning about explainable AI. It cannot diagnose or rule out a condition, "
            "and its result should not guide treatment.")

# ============================== TAB 2: performance ==============================
with tab_perf:
    tm = meta["test_metrics"]
    st.subheader(f"Selected model: {meta['selected_model']}")
    st.write(
        f"Chosen from six candidate models by cross-validated ROC-AUC on the training set "
        f"({meta['n_train']} patients), then evaluated **once** on a held-out test set of "
        f"{meta['n_test']} patients that was never used for training or tuning.")
    c = st.columns(5)
    c[0].metric("ROC-AUC", f"{tm['roc_auc']:.3f}")
    c[1].metric("Accuracy", f"{tm['accuracy']:.1%}")
    c[2].metric("Recall (sensitivity)", f"{tm['recall_sensitivity']:.1%}")
    c[3].metric("Specificity", f"{tm['specificity']:.1%}")
    c[4].metric("Precision", f"{tm['precision']:.1%}")
    lo, hi = tm["roc_auc_ci95"]
    alo, ahi = tm["accuracy_ci95"]
    st.caption(f"95% bootstrap intervals on the test set: ROC-AUC {lo:.2f} to {hi:.2f}, accuracy "
               f"{alo:.0%} to {ahi:.0%}. With only {meta['n_test']} test patients the intervals are "
               f"wide: treat the point estimates as approximate.")
    st.caption(f"Train ROC-AUC {meta['train_metrics']['roc_auc']:.3f} vs test ROC-AUC "
               f"{tm['roc_auc']:.3f}: a small gap, so there is little sign of overfitting.")

    cmp_path = REPORTS / "model_comparison_test.csv"
    cv_path = REPORTS / "model_comparison_cv.csv"
    if cv_path.exists():
        st.subheader("All candidate models (cross-validation on training data)")
        cv = pd.read_csv(cv_path)
        show = pd.DataFrame({
            "Model": cv["model"],
            "ROC-AUC": cv["roc_auc_mean"].round(3).astype(str) + " ± " + cv["roc_auc_std"].round(3).astype(str),
            "Accuracy": cv["accuracy_mean"].round(3),
            "Recall": cv["recall_mean"].round(3),
            "Precision": cv["precision_mean"].round(3),
            "F1": cv["f1_mean"].round(3),
        })
        st.dataframe(show, hide_index=True, width="stretch")
    if cmp_path.exists():
        st.subheader("All candidate models (held-out test set)")
        st.dataframe(pd.read_csv(cmp_path).round(3), hide_index=True, width="stretch")
    thr_path = REPORTS / "threshold_analysis.csv"
    if thr_path.exists():
        st.subheader("Decision threshold trade-off")
        st.caption("Missing a sick patient is usually worse than a false alarm, so a lower threshold "
                   "may be preferred in screening. This app reports probabilities rather than a hard yes/no.")
        st.dataframe(pd.read_csv(thr_path), hide_index=True, width="stretch")

    a, b = st.columns(2)
    with a:
        show_image(FIG / "confusion_matrix.png")
        show_image(FIG / "calibration_curve.png")
    with b:
        show_image(FIG / "roc_pr_curves.png")
        show_image(FIG / "model_comparison_cv.png")

# ============================== TAB 3: explainability ==============================
with tab_xai:
    st.subheader("What the model relies on overall")
    st.write("**SHAP** splits each prediction into one contribution per feature, measured in "
             "percentage points of risk. Averaging the size of those contributions over the test "
             "patients gives the global ranking below.")
    a, b = st.columns(2)
    with a:
        show_image(FIG / "shap_global_bar.png")
    with b:
        show_image(FIG / "permutation_importance.png")
    st.write("A second, independent method (**permutation importance**) measures how much accuracy "
             "drops when a feature is shuffled. Agreement between methods increases confidence that "
             "the explanation reflects the model.")
    esum = REPORTS / "explainability_summary.json"
    if esum.exists():
        s = json.loads(esum.read_text())
        st.caption(f"Agreement check: rank correlation between SHAP and permutation importance = "
                   f"{s['spearman_shap_vs_permutation']:.2f}; {s['top5_overlap']} of the top 5 features "
                   f"are shared. SHAP contributions add up to the prediction (max error "
                   f"{s['shap_additivity_max_error']:.1e}).")
    show_image(FIG / "shap_beeswarm.png")
    a, b = st.columns(2)
    with a:
        show_image(FIG / "shap_waterfall_high_risk.png")
    with b:
        show_image(FIG / "shap_waterfall_low_risk.png")
    st.subheader("LIME: a second opinion for one patient")
    st.write("LIME fits a simple local model around one patient. It should broadly agree with SHAP "
             "for that patient; where it does not, treat the explanation with caution.")
    show_image(FIG / "lime_example.png")
    coef = REPORTS / "logistic_coefficients.csv"
    if coef.exists():
        st.subheader("Model coefficients (odds ratios)")
        st.caption("Because the selected model is logistic regression, its coefficients are directly "
                   "readable. An odds ratio above 1 raises the odds of disease; below 1 lowers them. "
                   "Numeric features are standardised, categories are one-hot.")
        st.dataframe(pd.read_csv(coef).round(3), hide_index=True, width="stretch")

# ============================== TAB 4: data ==============================
with tab_data:
    st.subheader("Dataset")
    st.write("Cleveland Clinic subset of the UCI Heart Disease data: 303 patients and 13 clinical "
             "features, with the label showing whether coronary disease was found on angiography.")
    log_path = REPORTS / "cleaning_log.json"
    if log_path.exists():
        lg = json.loads(log_path.read_text())
        st.markdown(
            f"**Cleaning summary:** {lg['raw_rows']} raw rows, {lg['duplicates_removed']} duplicate removed, "
            f"{lg['ca_code4_set_to_nan']} invalid `ca` codes and {lg['thal_code0_set_to_nan']} invalid `thal` codes "
            f"set to missing, target recoded so that 1 = disease present, leaving **{lg['clean_rows']} rows**. "
            f"Remaining missing values {lg['missing_after_cleaning']} are filled inside the training pipeline "
            "(median for numbers, most frequent for categories), after the train/test split, to avoid leakage.")
        ba = REPORTS / "cleaning_before_after.csv"
        if ba.exists():
            st.dataframe(pd.read_csv(ba), hide_index=True, width="stretch")
    a, b = st.columns(2)
    with a:
        show_image(FIG / "eda_class_balance.png")
        show_image(FIG / "eda_correlation.png")
    with b:
        show_image(FIG / "eda_categorical_disease_rate.png")
    show_image(FIG / "eda_numeric_distributions.png")
    show_image(FIG / "eda_boxplots.png")

# ============================== TAB 5: about ==============================
with tab_about:
    st.subheader("How the system works")
    st.markdown(
        """
1. **Data**: UCI Heart Disease (Cleveland), cleaned and validated by script.
2. **Pipeline**: imputation, scaling and one-hot encoding are part of one scikit-learn pipeline, fitted on training data only.
3. **Models**: logistic regression, decision tree, random forest, XGBoost, SVM and a small neural network are compared with repeated stratified cross-validation; the top two are tuned.
4. **Explainability**: SHAP (primary), permutation importance and LIME (cross-checks).
5. **Interface**: this app loads the saved pipeline and explains each prediction.
        """)
    st.subheader("Limitations and ethics")
    st.markdown(
        """
- **Small dataset** (about 300 patients from one hospital, collected decades ago). Results may not transfer to other populations; the test set has only 61 patients, so every metric has wide uncertainty.
- **Diagnosis-style labels, not forecasts.** The label records whether disease was found at angiography, so this estimates *current* disease probability from test results, not future events.
- **Fairness.** Sex is a model input and the data is 68% male. A model trained like this can perform differently across groups; this project has not audited subgroup performance in depth.
- **Explanations show associations, not causes.** A feature that raises the estimate is not necessarily the reason a patient is ill.
- **Privacy.** The app does not store anything you enter. Do not enter real patient data into a public deployment.
- **Human oversight.** Predictions are decision support only. Clinical decisions need a qualified clinician.
        """)
    st.subheader("Citation")
    st.markdown(
        "Janosi, A., Steinbrunn, W., Pfisterer, M., and Detrano, R. (1989). *Heart Disease* "
        "[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X")
