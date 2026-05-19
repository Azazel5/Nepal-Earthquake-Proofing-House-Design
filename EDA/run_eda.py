#!/usr/bin/env python3
"""
Exploratory data analysis for Nepal earthquake building damage (training set only).

Reads:
  - data/driven_data/train_values.csv
  - data/driven_data/train_labels.csv

Writes figures to EDA/figures/ and a text summary to EDA/summary.txt
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from feature_glossary import (
    CATEGORICAL_DECODINGS,
    CATEGORICAL_FEATURES,
    DAMAGE_LABELS,
    FEATURE_GLOSSARY,
    NUMERIC_FEATURES,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "driven_data"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
SUMMARY_PATH = Path(__file__).resolve().parent / "summary.txt"

AGE_UNKNOWN = 995


def load_training_data() -> pd.DataFrame:
    values = pd.read_csv(DATA_DIR / "train_values.csv")
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    df = values.merge(labels, on="building_id", how="inner", validate="one_to_one")
    if len(df) != len(values):
        raise ValueError("Train values and labels row counts differ after merge.")
    return df


def binary_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def write_summary(df: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("NEPAL EARTHQUAKE DAMAGE — TRAINING SET EDA SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Rows: {len(df):,}  |  Columns (incl. target): {df.shape[1]}")
    lines.append(f"Missing values (all columns): {int(df.isna().sum().sum())}")
    lines.append("")

    lines.append("--- TARGET: damage_grade ---")
    counts = df["damage_grade"].value_counts().sort_index()
    total = len(df)
    for grade, n in counts.items():
        pct = 100.0 * n / total
        label = DAMAGE_LABELS.get(grade, str(grade))
        lines.append(f"  Grade {grade} ({label}): {n:>7,}  ({pct:5.2f}%)")
    imbalance = counts.max() / counts.min()
    lines.append(f"  Max/min class ratio: {imbalance:.2f}x  (moderate imbalance; grade 2 dominates)")
    lines.append("")

    lines.append("--- FEATURE TYPES & RANGES (numeric) ---")
    for col in NUMERIC_FEATURES:
        s = df[col]
        lines.append(f"  {col} [{FEATURE_GLOSSARY[col]}]")
        if col == "age":
            known = s[s != AGE_UNKNOWN]
            lines.append(
                f"    dtype=int  min={known.min()}  max={known.max()}  "
                f"mean={known.mean():.2f}  |  unknown ({AGE_UNKNOWN}): {(s == AGE_UNKNOWN).sum():,}"
            )
        else:
            lines.append(
                f"    dtype=int  min={s.min()}  max={s.max()}  "
                f"mean={s.mean():.2f}  median={s.median():.0f}  nunique={s.nunique()}"
            )
    lines.append("")

    lines.append("--- CATEGORICAL FEATURES (encoded letters) ---")
    for col in CATEGORICAL_FEATURES:
        lines.append(f"  {col} [{FEATURE_GLOSSARY[col]}]")
        decoding = CATEGORICAL_DECODINGS.get(col, {})
        vc = df[col].value_counts()
        for code, n in vc.items():
            meaning = decoding.get(code, "?")
            lines.append(f"    '{code}' ({meaning}): {n:,}  ({100 * n / total:.2f}%)")
    lines.append("")

    super_cols = binary_columns(df, "has_superstructure_")
    lines.append("--- BINARY: superstructure materials (multi-label) ---")
    for col in super_cols:
        rate = df[col].mean()
        lines.append(f"  {col}: {rate * 100:.2f}% present  — {FEATURE_GLOSSARY[col]}")
    lines.append(f"  Materials per building: mean={df[super_cols].sum(axis=1).mean():.2f}, max={df[super_cols].sum(axis=1).max()}")
    lines.append("")

    sec_cols = [c for c in binary_columns(df, "has_secondary_use") if c != "has_secondary_use"]
    lines.append("--- BINARY: secondary use flags ---")
    lines.append(f"  Any secondary use: {df['has_secondary_use'].mean() * 100:.2f}%")
    for col in sec_cols:
        if df[col].sum() > 0:
            lines.append(f"  {col}: {df[col].mean() * 100:.3f}%")
    lines.append("")

    lines.append("--- FULL FEATURE GLOSSARY ---")
    for name, desc in FEATURE_GLOSSARY.items():
        if name in df.columns or name == "damage_grade":
            lines.append(f"  {name}: {desc}")

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"


def save_fig(name: str) -> None:
    path = FIGURES_DIR / f"{name}.png"
    plt.savefig(path)
    plt.close()
    print(f"  saved {path.relative_to(ROOT)}")


def plot_target_distribution(df: pd.DataFrame) -> None:
    counts = df["damage_grade"].value_counts().sort_index()
    labels = [f"Grade {g}\n{DAMAGE_LABELS[g]}" for g in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = sns.color_palette("colorblind", n_colors=3)

    axes[0].bar(counts.index.astype(str), counts.values, color=colors)
    axes[0].set_xlabel("Damage grade")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Label distribution (imbalanced)")

    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
    )
    axes[1].set_title("Class proportions")
    plt.tight_layout()
    save_fig("01_target_distribution")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    plot_df = df[NUMERIC_FEATURES].copy()
    plot_df["age_known"] = plot_df["age"].where(plot_df["age"] != AGE_UNKNOWN)

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.ravel()

    hist_cols = [
        "count_floors_pre_eq",
        "age_known",
        "area_percentage",
        "height_percentage",
        "count_families",
        "geo_level_1_id",
        "geo_level_2_id",
        "geo_level_3_id",
    ]
    for ax, col in zip(axes, hist_cols):
        series = plot_df[col].dropna()
        ax.hist(series, bins=30, color=sns.color_palette()[0], edgecolor="white")
        title = "age (excluding unknown=995)" if col == "age_known" else col
        ax.set_title(title)
        ax.set_ylabel("Count")

    axes[-1].axis("off")
    fig.suptitle("Numeric feature distributions", y=1.02)
    plt.tight_layout()
    save_fig("02_numeric_distributions")


def plot_numeric_by_damage(df: pd.DataFrame) -> None:
    melt_cols = ["count_floors_pre_eq", "area_percentage", "height_percentage", "count_families"]
    long = df.melt(
        id_vars="damage_grade",
        value_vars=melt_cols,
        var_name="feature",
        value_name="value",
    )
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=long, x="feature", y="value", hue="damage_grade", hue_order=[1, 2, 3])
    plt.title("Selected numeric features vs. damage grade")
    plt.xlabel("")
    plt.legend(title="Damage", labels=[DAMAGE_LABELS[g] for g in [1, 2, 3]])
    save_fig("03_numeric_vs_damage")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    num = df[NUMERIC_FEATURES + ["damage_grade"]].copy()
    num["age"] = num["age"].replace(AGE_UNKNOWN, np.nan)
    corr = num.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
    plt.title("Correlation matrix (numeric features + target)")
    save_fig("04_correlation_heatmap")


def plot_categorical_vs_damage(df: pd.DataFrame, col: str, fig_idx: int) -> None:
    ct = pd.crosstab(df[col], df["damage_grade"], normalize="index") * 100
    ct = ct.reindex(columns=[1, 2, 3])
    decoding = CATEGORICAL_DECODINGS.get(col, {})
    index_labels = [f"{idx}\n({decoding.get(idx, '?')})" for idx in ct.index]

    ax = ct.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="viridis")
    ax.set_xticklabels(index_labels, rotation=0, ha="center")
    ax.set_ylabel("% within category")
    ax.set_xlabel(col)
    ax.set_title(f"{col} vs. damage grade (row-normalized %)")
    ax.legend(title="Grade", labels=[f"{g}" for g in ct.columns])
    save_fig(f"{fig_idx:02d}_{col}_vs_damage")


def plot_superstructure_prevalence(df: pd.DataFrame) -> None:
    cols = binary_columns(df, "has_superstructure_")
    rates = df[cols].mean().sort_values(ascending=True) * 100
    labels = [c.replace("has_superstructure_", "") for c in rates.index]

    plt.figure(figsize=(10, 7))
    plt.barh(labels, rates.values, color=sns.color_palette()[2])
    plt.xlabel("% of buildings with material")
    plt.title("Superstructure material prevalence")
    save_fig("13_superstructure_prevalence")


def plot_superstructure_count(df: pd.DataFrame) -> None:
    cols = binary_columns(df, "has_superstructure_")
    counts = df[cols].sum(axis=1)
    plt.figure(figsize=(8, 5))
    sns.countplot(x=counts, color=sns.color_palette()[0])
    plt.xlabel("Number of superstructure types flagged per building")
    plt.ylabel("Count")
    plt.title("Multi-label superstructure tags per building")
    save_fig("14_superstructure_count_per_building")


def plot_secondary_use(df: pd.DataFrame) -> None:
    cols = [c for c in binary_columns(df, "has_secondary_use") if c != "has_secondary_use"]
    rates = df[cols].mean().sort_values(ascending=True) * 100
    rates = rates[rates > 0]
    labels = [c.replace("has_secondary_use_", "") for c in rates.index]

    plt.figure(figsize=(10, 5))
    plt.barh(labels, rates.values, color=sns.color_palette()[4])
    plt.xlabel("% of buildings")
    plt.title("Secondary use flags (non-zero only)")
    save_fig("15_secondary_use_prevalence")


def plot_geo_levels(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]):
        vc = df[col].value_counts().head(15)
        ax.bar(range(len(vc)), vc.values, color=sns.color_palette()[0])
        ax.set_title(f"Top 15 {col} by frequency")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Count")
    plt.tight_layout()
    save_fig("16_geo_level_top15")


def plot_damage_by_age_bucket(df: pd.DataFrame) -> None:
    age = df["age"].copy()
    age_label = np.where(age == AGE_UNKNOWN, "unknown", pd.cut(age, bins=[0, 10, 20, 30, 50, 100, 200, 1000], right=True))
    tmp = df.assign(age_bucket=age_label)
    ct = pd.crosstab(tmp["age_bucket"], tmp["damage_grade"], normalize="index") * 100
    ct = ct.reindex(columns=[1, 2, 3])
    ct.plot(kind="bar", figsize=(12, 5), colormap="viridis")
    plt.ylabel("% within age bucket")
    plt.xlabel("Age bucket (years)")
    plt.title("Damage grade by building age bucket")
    plt.legend(title="Grade")
    plt.xticks(rotation=45, ha="right")
    save_fig("17_damage_by_age_bucket")


def main() -> None:
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    df = load_training_data()
    print(f"  {len(df):,} buildings, {df.shape[1]} columns")

    write_summary(df)

    print("\nGenerating figures...")
    plot_target_distribution(df)
    plot_numeric_distributions(df)
    plot_numeric_by_damage(df)
    plot_correlation_heatmap(df)

    fig_idx = 5
    for col in CATEGORICAL_FEATURES:
        plot_categorical_vs_damage(df, col, fig_idx)
        fig_idx += 1

    plot_superstructure_prevalence(df)
    plot_superstructure_count(df)
    plot_secondary_use(df)
    plot_geo_levels(df)
    plot_damage_by_age_bucket(df)

    print("\nDone. See EDA/summary.txt and EDA/figures/")


if __name__ == "__main__":
    main()
