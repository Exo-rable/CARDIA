"""Shared paths, feature definitions and helpers used by every stage and by the app."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "heart_cleveland_raw.csv"
CLEAN = ROOT / "data" / "clean" / "heart_clean.csv"
MODEL = ROOT / "models" / "heart_model.joblib"
META = ROOT / "models" / "model_meta.json"
BACKGROUND = ROOT / "models" / "background.csv"
FIG = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"

SEED = 42
TEST_SIZE = 0.2

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL  # the one fixed column order used everywhere
TARGET = "disease"  # 1 = heart disease present, 0 = absent
POSITIVE_CLASS = 1
CLASSIFICATION_THRESHOLD = 0.5


def probabilities_in_class_order(estimator, features, requested_classes):
    """Return model probabilities ordered by explicit class labels."""
    classes = list(estimator.classes_)
    missing_classes = [label for label in requested_classes if label not in classes]
    if missing_classes:
        raise ValueError(f"Model classes {classes!r} do not include {missing_classes!r}.")
    probabilities = estimator.predict_proba(features)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Model probability output does not match its classes.")
    return probabilities[:, [classes.index(label) for label in requested_classes]]


def positive_class_probability(estimator, features):
    """Return P(TARGET == POSITIVE_CLASS) without assuming class-column order."""
    return probabilities_in_class_order(estimator, features, [POSITIVE_CLASS])[:, 0]

# Human-readable names, units and value labels.
# NOTE: the codes follow the 303-row Cleveland copy used in this project
# (see README, "Dataset" section). They were cross-checked against the
# category counts published for the original UCI Cleveland file.
DISPLAY = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG result",
    "thalach": "Maximum heart rate achieved (bpm)",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy",
    "thal": "Specialized heart-test category (thal)",
}

SHORT = {
    "age": "Age", "sex": "Sex", "cp": "Chest pain", "trestbps": "Resting BP",
    "chol": "Cholesterol", "fbs": "Fasting sugar", "restecg": "Resting ECG",
    "thalach": "Max heart rate", "exang": "Exercise angina",
    "oldpeak": "ST depression", "slope": "ST slope", "ca": "Vessels (ca)",
    "thal": "Specialized test category",
}

LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"},
    "fbs": {0: "No", 1: "Yes"},
    "restecg": {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    "ca": {0: "0", 1: "1", 2: "2", 3: "3"},
    "thal": {2: "Normal", 1: "Fixed defect", 3: "Reversible defect"},
}

RANGES = {  # (min, max, step) used for sliders and for the cleaning rules
    "age": (20, 90, 1),
    "trestbps": (80, 220, 1),
    "chol": (100, 600, 1),
    "thalach": (60, 220, 1),
    "oldpeak": (0.0, 7.0, 0.1),
}

def value_label(feature: str, value) -> str:
    """Readable text for one feature value."""
    if feature in LABELS:
        return LABELS[feature].get(int(value), str(value))
    return f"{value:g}"


def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN)


def get_split(df: pd.DataFrame | None = None):
    """Deterministic stratified split. Every stage calls this, so all stages agree."""
    if df is None:
        df = load_clean()
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
