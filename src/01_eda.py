"""Stage 1: exploratory data analysis. Run it AFTER 02_clean.py so the plots show clean data.
Saves figures to reports/figures/ for the report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import DISPLAY, FIG, FEATURES, NUMERIC, SHORT, TARGET, load_clean

BLUE, ORANGE = "#2a6f97", "#e07a1f"  # colour-blind-safe pair: healthy / disease


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_clean()
    sns.set_theme(style="whitegrid", font_scale=0.95)

    # 1. class balance
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    counts = df[TARGET].value_counts().sort_index()
    ax.bar(["No disease", "Disease"], counts.values, color=[BLUE, ORANGE])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, f"{v} ({v / len(df):.0%})", ha="center")
    ax.set_ylabel("Patients")
    ax.set_title("Class balance")
    fig.tight_layout()
    fig.savefig(FIG / "eda_class_balance.png", dpi=160)
    plt.close(fig)

    # 2. numeric distributions by class
    fig, axes = plt.subplots(1, len(NUMERIC), figsize=(15, 3.2))
    for ax, col in zip(axes, NUMERIC):
        for val, colour, name in [(0, BLUE, "No disease"), (1, ORANGE, "Disease")]:
            sns.kdeplot(df.loc[df[TARGET] == val, col].dropna(), ax=ax, color=colour,
                        fill=True, alpha=0.35, label=name)
        ax.set_title(SHORT[col])
        ax.set_xlabel("")
        ax.set_ylabel("")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "eda_numeric_distributions.png", dpi=160)
    plt.close(fig)

    # 3. boxplots (outlier check)
    fig, axes = plt.subplots(1, len(NUMERIC), figsize=(15, 3.2))
    for ax, col in zip(axes, NUMERIC):
        sns.boxplot(y=df[col], ax=ax, color="#9db4c0")
        ax.set_title(SHORT[col])
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG / "eda_boxplots.png", dpi=160)
    plt.close(fig)

    # 4. disease rate for each categorical value
    cats = [c for c in FEATURES if c not in NUMERIC]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for ax, col in zip(axes.ravel(), cats):
        rate = df.groupby(col)[TARGET].mean()
        ax.bar([str(int(i)) for i in rate.index], rate.values, color=ORANGE)
        ax.set_ylim(0, 1)
        ax.set_title(SHORT[col])
        ax.set_ylabel("Disease rate")
    fig.suptitle("Share of patients with disease, by category code", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "eda_categorical_disease_rate.png", dpi=160)
    plt.close(fig)

    # 5. correlation heatmap
    corr = df.corr(numeric_only=True)
    corr.index = corr.columns = [SHORT.get(c, "Disease") for c in corr.columns]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(FIG / "eda_correlation.png", dpi=160)
    plt.close(fig)

    # 6. missing values
    miss = df.isna().sum()
    miss = miss[miss > 0]
    print("Missing values after cleaning (imputed later inside the pipeline):")
    print(miss.to_string() if len(miss) else "none")

    desc = df.describe().T.round(2)
    desc.to_csv(FIG.parent / "eda_summary_statistics.csv")
    print("\nSummary statistics:\n", desc.to_string())
    print("\nFigures saved to", FIG)


if __name__ == "__main__":
    main()
