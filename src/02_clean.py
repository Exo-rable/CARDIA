"""Stage 2: clean the raw data and PROVE the cleaning is correct.

Rules applied (and logged):
  1. standardise column names
  2. remove exact duplicate rows
  3. recode the target so that 1 = disease present (this copy stores it inverted)
  4. turn hidden-missing codes into real NaN (ca == 4, thal == 0)
  5. turn medically impossible values into NaN (range rules)

Deliberately NOT done here: imputation, scaling, encoding. Those learn from the data,
so they live inside the training pipeline (after the train/test split) to avoid leakage.
"""
import json

import numpy as np
import pandas as pd

from common import CLEAN, FEATURES, RAW, RANGES, REPORTS, TARGET

# Physiologically plausible limits (wider than the slider ranges on purpose).
PLAUSIBLE = {
    "age": (1, 120),
    "trestbps": (60, 260),
    "chol": (80, 700),
    "thalach": (40, 230),
    "oldpeak": (0, 10),
}


def main():
    raw = pd.read_csv(RAW)
    log = {"raw_rows": int(len(raw)), "raw_columns": list(raw.columns)}
    df = raw.copy()

    # 1. names
    df.columns = [c.strip().lower() for c in df.columns]

    # 2. duplicates
    log["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # 3. target recoding (verified: with disease = 1 - target, the known risk factors
    #    such as exercise angina, asymptomatic chest pain and blocked vessels all
    #    point the medically expected way; see README "Dataset").
    log["target_counts_original"] = {int(k): int(v) for k, v in df["target"].value_counts().items()}
    df[TARGET] = 1 - df["target"]
    df = df.drop(columns="target")

    # 4. hidden missing codes
    log["ca_code4_set_to_nan"] = int((df["ca"] == 4).sum())
    log["thal_code0_set_to_nan"] = int((df["thal"] == 0).sum())
    df.loc[df["ca"] == 4, "ca"] = np.nan
    df.loc[df["thal"] == 0, "thal"] = np.nan

    # 5. impossible values
    for col, (lo, hi) in PLAUSIBLE.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        log[f"{col}_out_of_range_set_to_nan"] = int(bad.sum())
        df.loc[bad, col] = np.nan

    df = df[FEATURES + [TARGET]]
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)

    # ---------------- validation: fail loudly if anything is off ----------------
    chk = pd.read_csv(CLEAN)
    assert chk.duplicated().sum() == 0, "duplicates remain"
    assert len(chk) == log["raw_rows"] - log["duplicates_removed"], "row accounting mismatch"
    assert list(chk.columns) == FEATURES + [TARGET], "unexpected columns/order"
    assert set(chk[TARGET].unique()) == {0, 1}, "target must be 0/1"
    assert chk[TARGET].isna().sum() == 0, "target has missing values"
    for col, (lo, hi) in PLAUSIBLE.items():
        assert chk[col].dropna().between(lo, hi).all(), f"{col} out of range"
    assert chk["ca"].dropna().isin([0, 1, 2, 3]).all(), "ca has invalid codes"
    assert chk["thal"].dropna().isin([1, 2, 3]).all(), "thal has invalid codes"
    assert chk["cp"].isin([0, 1, 2, 3]).all() and chk["sex"].isin([0, 1]).all()
    assert (chk.select_dtypes("number").shape[1] == chk.shape[1]), "non-numeric column"
    # cleaning must not distort the data: means may only move slightly
    for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        shift = abs(chk[col].mean() - raw[col].mean()) / raw[col].std()
        assert shift < 0.05, f"{col} mean shifted too much ({shift:.3f} sd)"
    # sanity: the target must relate to known risk factors in the expected direction
    assert chk.groupby("exang")[TARGET].mean()[1] > chk.groupby("exang")[TARGET].mean()[0]
    assert chk.groupby("cp")[TARGET].mean()[0] > chk.groupby("cp")[TARGET].mean()[2]

    log["clean_rows"] = int(len(chk))
    log["missing_after_cleaning"] = {k: int(v) for k, v in chk.isna().sum().items() if v}
    log["class_balance"] = {int(k): int(v) for k, v in chk[TARGET].value_counts().items()}
    log["validation"] = "all checks passed"

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "cleaning_log.json").write_text(json.dumps(log, indent=2))

    # before / after summary table for the report
    rows = []
    for col in FEATURES:
        rows.append({
            "feature": col,
            "raw_min": raw[col].min(), "raw_max": raw[col].max(),
            "clean_min": chk[col].min(), "clean_max": chk[col].max(),
            "raw_missing": int(raw[col].isna().sum()), "clean_missing": int(chk[col].isna().sum()),
        })
    pd.DataFrame(rows).to_csv(REPORTS / "cleaning_before_after.csv", index=False)

    print(json.dumps(log, indent=2))
    print("\nCleaning finished and validated ->", CLEAN)


if __name__ == "__main__":
    main()
