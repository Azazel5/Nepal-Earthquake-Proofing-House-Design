#!/usr/bin/env python3
"""
Deep error analysis for run_001 vs run_002 using out-of-fold predictions.

Answers: where models fail, which buildings, which features, which regions —
and what to engineer next.
"""

from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

print = partial(print, flush=True)  # noqa: A001 — unbuffered console for long runs

# Project root on path for EDA.feature_glossary
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

from EDA.feature_glossary import CATEGORICAL_DECODINGS
from preprocess import (
    DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    binary_columns,
)
from run_manager import PROCESSED_DIR, ROOT, RunManager

EVAL_DIR = ROOT / "runs" / "evaluation"

# Set True to load cached OOF arrays and run sections 3–10 only (~2 min).
# Run once with False to generate and save cache under runs/evaluation/*.npy
SKIP_OOF = True

N_FOLDS = 5
EARLY_STOPPING_ROUNDS = 50
GRADES = [1, 2, 3]
HIGH_CONF_THRESHOLD = 0.80
CALIBRATION_SAMPLE = 25_000  # subsample for histogram speed on 200k+ rows

RUN_001 = "run_001"
RUN_002 = "run_002"

CATEGORICAL_FEATURES = list(CATEGORICAL_DECODINGS.keys())
GEO_COLS = ["geo_level_1_id", "geo_level_2_id", "geo_level_3_id"]
SUPERSTRUCTURE_COLS = [
    c
    for c in [
        "has_superstructure_adobe_mud",
        "has_superstructure_mud_mortar_stone",
        "has_superstructure_stone_flag",
        "has_superstructure_cement_mortar_stone",
        "has_superstructure_mud_mortar_brick",
        "has_superstructure_cement_mortar_brick",
        "has_superstructure_timber",
        "has_superstructure_bamboo",
        "has_superstructure_rc_non_engineered",
        "has_superstructure_rc_engineered",
        "has_superstructure_other",
    ]
]


def section_header(title: str) -> None:
    print("\n" + "═" * 55)
    print(title)
    print("═" * 55)


def load_train_frames() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Load processed X_train and align raw categorical/geo columns via the
    same stratified split used in preprocess.py (208,480 train rows).
    """
    X = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y = pd.read_csv(PROCESSED_DIR / "y_train_multiclass.csv")["damage_grade"].to_numpy()

    values = pd.read_csv(DATA_DIR / "train_values.csv")
    labels = pd.read_csv(DATA_DIR / "train_labels.csv")
    raw = values.merge(labels, on="building_id", how="inner")
    raw["survived"] = (raw["damage_grade"] <= 2).astype(int)

    from sklearn.model_selection import train_test_split

    train_raw, _ = train_test_split(
        raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=raw["survived"],
    )
    train_raw = train_raw.reset_index(drop=True)
    if len(train_raw) != len(X):
        raise ValueError(f"Train align mismatch: raw {len(train_raw)} vs X {len(X)}")

    meta_cols = ["building_id", "damage_grade"] + GEO_COLS + CATEGORICAL_FEATURES + binary_columns(raw)
    meta = train_raw[meta_cols].copy()
    return X, y, meta


def load_run_params(run_id: str) -> dict[str, Any]:
    with (ROOT / "runs" / run_id / "params.json").open(encoding="utf-8") as f:
        return json.load(f)


def generate_oof_predictions(
    X: pd.DataFrame, y: np.ndarray, params: dict[str, Any], run_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified K-fold OOF class predictions and probability matrix."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros((len(y), 3), dtype=float)

    print(f"  Generating OOF for {run_id} ({N_FOLDS} folds)...")
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        model = LGBMClassifier(**params)
        model.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        proba = model.predict_proba(X.iloc[va_idx])
        pred = model.classes_[np.argmax(proba, axis=1)].astype(int)
        oof_pred[va_idx] = pred
        # align proba columns to grades 1,2,3
        for i, cls in enumerate(model.classes_):
            col = int(cls) - 1
            oof_proba[va_idx, col] = proba[:, i]
        print(f"    fold {fold}/{N_FOLDS} done")

    return oof_pred, oof_proba


def _oof_cache_paths() -> dict[str, Path]:
    return {
        "oof_001": EVAL_DIR / "oof_001.npy",
        "oof_002": EVAL_DIR / "oof_002.npy",
        "proba_001": EVAL_DIR / "proba_001.npy",
        "proba_002": EVAL_DIR / "proba_002.npy",
    }


def oof_cache_exists() -> bool:
    """True if all four OOF cache files are present."""
    return all(p.exists() for p in _oof_cache_paths().values())


def save_oof_cache(
    oof_001: np.ndarray,
    proba_001: np.ndarray,
    oof_002: np.ndarray,
    proba_002: np.ndarray,
) -> None:
    """Persist OOF predictions so later runs can set SKIP_OOF=True."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    paths = _oof_cache_paths()
    np.save(paths["oof_001"], oof_001)
    np.save(paths["proba_001"], proba_001)
    np.save(paths["oof_002"], oof_002)
    np.save(paths["proba_002"], proba_002)
    print(f"  OOF cache saved to {EVAL_DIR.relative_to(ROOT)}/")


def load_oof_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached OOF predictions and probabilities."""
    paths = _oof_cache_paths()
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"OOF cache missing: {missing}. Run once with SKIP_OOF=False to generate."
        )
    print("  Loaded OOF cache (skipping fold training).")
    return (
        np.load(paths["oof_001"]),
        np.load(paths["proba_001"]),
        np.load(paths["oof_002"]),
        np.load(paths["proba_002"]),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Aggregate classification metrics for one model."""
    y_bin = label_binarize(y_true, classes=GRADES)
    try:
        auc_macro = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
    except ValueError:
        auc_macro = float("nan")

    report = classification_report(y_true, y_pred, labels=GRADES, output_dict=True, zero_division=0)
    m: dict[str, float] = {
        "micro_f1": f1_score(y_true, y_pred, average="micro", labels=GRADES),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=GRADES),
        "accuracy": accuracy_score(y_true, y_pred),
        "auc_macro_ovr": auc_macro,
    }
    for g in GRADES:
        key = str(g)
        m[f"grade{g}_f1"] = report[key]["f1-score"]
        m[f"grade{g}_precision"] = report[key]["precision"]
        m[f"grade{g}_recall"] = report[key]["recall"]
    return m


def print_metrics_table(m001: dict[str, float], m002: dict[str, float]) -> pd.DataFrame:
    """Side-by-side metrics with deltas."""
    rows = [
        ("Micro F1", "micro_f1"),
        ("Macro F1", "macro_f1"),
        ("Accuracy", "accuracy"),
        ("AUC (macro OvR)", "auc_macro_ovr"),
        ("Grade 1 F1", "grade1_f1"),
        ("Grade 2 F1", "grade2_f1"),
        ("Grade 3 F1", "grade3_f1"),
        ("Grade 1 Precision", "grade1_precision"),
        ("Grade 1 Recall", "grade1_recall"),
        ("Grade 2 Precision", "grade2_precision"),
        ("Grade 2 Recall", "grade2_recall"),
        ("Grade 3 Precision", "grade3_precision"),
        ("Grade 3 Recall", "grade3_recall"),
    ]
    print(f"\n{'Metric':<22} {'run_001':>10} {'run_002':>10} {'delta':>10}")
    print("─" * 56)
    summary_rows = []
    for label, key in rows:
        a, b = m001[key], m002[key]
        d = b - a
        print(f"{label:<22} {a:>10.4f} {b:>10.4f} {d:>+10.4f}")
        summary_rows.append({"metric": label, "run_001": a, "run_002": b, "delta": d})
    return pd.DataFrame(summary_rows)


def print_confusion_matrices(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> None:
    """Normalized, raw, delta, and critical cell analysis."""
    for run_id, pred in [(RUN_001, p1), (RUN_002, p2)]:
        cm = confusion_matrix(y, pred, labels=GRADES)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        print(f"\n{run_id} Confusion Matrix (row=actual, col=predicted, normalized by row):")
        print(f"{'':>10} {'Pred 1':>10} {'Pred 2':>10} {'Pred 3':>10}")
        for i, g in enumerate(GRADES):
            print(
                f"Actual {g}  "
                + " ".join(f"{cm_norm[i, j] * 100:>9.2f}%" for j in range(3))
            )
        print(f"\n{run_id} raw counts:")
        print(f"{'':>10} {'Pred 1':>10} {'Pred 2':>10} {'Pred 3':>10}")
        for i, g in enumerate(GRADES):
            print(f"Actual {g}  " + " ".join(f"{cm[i, j]:>10,}" for j in range(3)))

    cm1 = confusion_matrix(y, p1, labels=GRADES).astype(float)
    cm2 = confusion_matrix(y, p2, labels=GRADES).astype(float)
    n1 = cm1 / cm1.sum(axis=1, keepdims=True)
    n2 = cm2 / cm2.sum(axis=1, keepdims=True)
    delta = (n2 - n1) * 100
    print("\nDELTA matrix (run_002 - run_001, percentage points by row):")
    print(f"{'':>10} {'Pred 1':>10} {'Pred 2':>10} {'Pred 3':>10}")
    for i, g in enumerate(GRADES):
        print(f"Actual {g}  " + " ".join(f"{delta[i, j]:>+9.2f}pp" for j in range(3)))

    critical = [
        ("Actual 3 → Predicted 2 (dangerous miss)", 3, 2),
        ("Actual 3 → Predicted 1 (very dangerous miss)", 3, 1),
        ("Actual 1 → Predicted 3 (false alarm)", 1, 3),
    ]
    print("\nCritical cells:")
    for label, actual, pred in critical:
        for run_id, cm in [(RUN_001, cm1), (RUN_002, cm2)]:
            count = int(cm[actual - 1, pred - 1])
            row_total = cm[actual - 1].sum()
            pct = 100 * count / row_total if row_total else 0
            print(f"  {run_id} {label}: {count:,} ({pct:.2f}% of actual grade {actual})")


def plot_confusion_side_by_side(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, pred, title in zip(axes, [p1, p2, p2 - p1], [RUN_001, RUN_002, "delta counts"]):
        if title == "delta counts":
            data = confusion_matrix(y, p2, labels=GRADES) - confusion_matrix(y, p1, labels=GRADES)
            vmin, vmax = None, None
        else:
            data = confusion_matrix(y, pred, labels=GRADES)
            data = data.astype(float) / data.sum(axis=1, keepdims=True)
            vmin, vmax = 0, 1
        im = ax.imshow(data, cmap="Blues", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([0, 1, 2], labels=["Pred 1", "Pred 2", "Pred 3"])
        ax.set_yticks([0, 1, 2], labels=["Act 1", "Act 2", "Act 3"])
        for i in range(3):
            for j in range(3):
                val = data[i, j]
                txt = f"{val:.2f}" if title != "delta counts" else f"{int(val)}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Multiclass ECE using max-probability confidence."""
    conf = proba.max(axis=1)
    pred = np.array(GRADES)[np.argmax(proba, axis=1)]
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return ece


def class_calibration_ece(y_true: np.ndarray, proba: np.ndarray, grade: int, n_bins: int = 10) -> float:
    """ECE for one class (one-vs-rest)."""
    y_bin = (y_true == grade).astype(int)
    p = proba[:, grade - 1]
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p > bins[i]) & (p <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(y_true) * abs(y_bin[mask].mean() - p[mask].mean())
    return ece


def plot_calibration(y: np.ndarray, pred: np.ndarray, proba: np.ndarray, path: Path, run_id: str) -> dict[str, float]:
    eces = {}
    rng = np.random.RandomState(RANDOM_STATE)
    n_sub = min(len(y), CALIBRATION_SAMPLE)
    sub_idx = rng.choice(len(y), size=n_sub, replace=False)
    y_s, pred_s, proba_s = y[sub_idx], pred[sub_idx], proba[sub_idx]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, g in zip(axes, GRADES):
        mask_correct = (pred_s == y_s) & (y_s == g)
        mask_wrong = (pred_s != y_s) & (y_s == g)
        p = proba_s[:, g - 1]
        if mask_correct.any():
            ax.hist(p[mask_correct], bins=30, alpha=0.6, label="correct", density=True)
        if mask_wrong.any():
            ax.hist(p[mask_wrong], bins=30, alpha=0.6, label="incorrect", density=True)
        ax.set_title(f"Grade {g} probability")
        ax.legend(fontsize=8)
        eces[f"grade{g}"] = class_calibration_ece(y, proba, g)
    eces["overall"] = expected_calibration_error(y, proba)
    fig.suptitle(f"{run_id} calibration (ECE overall={eces['overall']:.4f})")
    plt.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return eces


def dangerous_miss_mask(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_true == 3) & (y_pred != 3)


def false_alarm_mask(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_true == 1) & (y_pred != 1)


def geo_error_table(
    meta: pd.DataFrame, y: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> pd.DataFrame:
    rows = []
    for geo_id, grp_idx in meta.groupby("geo_level_1_id").groups.items():
        idx = list(grp_idx)
        yt, ya = y[idx], meta["damage_grade"].values[idx]
        for run_id, pred in [(RUN_001, p1[idx]), (RUN_002, p2[idx])]:
            wrong = pred != yt
            g3 = yt == 3
            g1 = yt == 1
            rows.append(
                {
                    "geo_level_1_id": geo_id,
                    "run_id": run_id,
                    "n_buildings": len(idx),
                    "error_rate": wrong.mean(),
                    "dangerous_miss_rate": (
                        dangerous_miss_mask(yt, pred).sum() / g3.sum() if g3.sum() else np.nan
                    ),
                    "false_alarm_rate": (
                        false_alarm_mask(yt, pred).sum() / g1.sum() if g1.sum() else np.nan
                    ),
                }
            )
    df = pd.DataFrame(rows)
    # pivot comparison
    p = df.pivot_table(
        index="geo_level_1_id",
        columns="run_id",
        values=["error_rate", "dangerous_miss_rate", "false_alarm_rate"],
    )
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    p = p.reset_index()
    p["error_rate_delta"] = p.get("error_rate_run_002", 0) - p.get("error_rate_run_001", 0)
    p["n_buildings"] = df.groupby("geo_level_1_id")["n_buildings"].first().values
    return p.sort_values("dangerous_miss_rate_run_002", ascending=False, na_position="last")


def feature_slice_table(
    meta: pd.DataFrame, y: np.ndarray, p1: np.ndarray, p2: np.ndarray, feature: str
) -> pd.DataFrame:
    rows = []
    decoding = CATEGORICAL_DECODINGS.get(feature, {})
    for val, grp_idx in meta.groupby(feature).groups.items():
        idx = list(grp_idx)
        yt = y[idx]
        for run_id, pred in [(RUN_001, p1[idx]), (RUN_002, p2[idx])]:
            g3 = yt == 3
            rows.append(
                {
                    "feature": feature,
                    "value": val,
                    "label": decoding.get(val, str(val)),
                    "run_id": run_id,
                    "n": len(idx),
                    "error_rate": (pred != yt).mean(),
                    "dangerous_miss_rate": (
                        dangerous_miss_mask(yt, pred).sum() / g3.sum() if g3.sum() else np.nan
                    ),
                    "grade3_recall": ((pred == 3) & g3).sum() / g3.sum() if g3.sum() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def superstructure_analysis(
    meta: pd.DataFrame, y: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> pd.DataFrame:
    rows = []
    for col in SUPERSTRUCTURE_COLS:
        for present in (0, 1):
            mask = meta[col].values == present
            if mask.sum() == 0:
                continue
            yt = y[mask]
            for run_id, pred in [(RUN_001, p1[mask]), (RUN_002, p2[mask])]:
                g3 = yt == 3
                rows.append(
                    {
                        "material": col.replace("has_superstructure_", ""),
                        "present": bool(present),
                        "run_id": run_id,
                        "n": int(mask.sum()),
                        "error_rate": (pred != yt).mean(),
                        "dangerous_miss_rate": (
                            dangerous_miss_mask(yt, pred).sum() / g3.sum() if g3.sum() else np.nan
                        ),
                    }
                )
    # interaction: timber + mud_mortar_stone
    both = (meta["has_superstructure_timber"] == 1) & (
        meta["has_superstructure_mud_mortar_stone"] == 1
    )
    if both.sum() > 0:
        yt = y[both.values]
        for run_id, pred in [(RUN_001, p1[both.values]), (RUN_002, p2[both.values])]:
            g3 = yt == 3
            rows.append(
                {
                    "material": "timber+mud_mortar_stone",
                    "present": True,
                    "run_id": run_id,
                    "n": int(both.sum()),
                    "error_rate": (pred != yt).mean(),
                    "dangerous_miss_rate": (
                        dangerous_miss_mask(yt, pred).sum() / g3.sum() if g3.sum() else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def high_confidence_wrong(y: np.ndarray, pred: np.ndarray, proba: np.ndarray, meta: pd.DataFrame, run_id: str) -> None:
    conf = proba.max(axis=1)
    wrong = pred != y
    hcw = wrong & (conf > HIGH_CONF_THRESHOLD)
    n_err = wrong.sum()
    print(f"\n{run_id} high-confidence wrong (>{HIGH_CONF_THRESHOLD:.0%}):")
    print(f"  Count: {hcw.sum():,}  ({100 * hcw.sum() / len(y):.2f}% of all buildings)")
    if n_err:
        print(f"  % of all errors: {100 * hcw.sum() / n_err:.1f}%")
    if hcw.sum() == 0:
        return
    print("  Actual grade distribution in HCW:")
    for g in GRADES:
        n = ((y == g) & hcw).sum()
        print(f"    grade {g}: {n:,}")
  # top feature combos
    sub = meta.loc[hcw].copy()
    sub["combo"] = (
        sub["foundation_type"].astype(str)
        + "+"
        + sub["has_superstructure_mud_mortar_stone"].astype(str)
        + "+geo1="
        + sub["geo_level_1_id"].astype(str)
    )
    top = sub["combo"].value_counts().head(5)
    print("  Top 5 feature combos (foundation+mud_stone+geo1):")
    for combo, cnt in top.items():
        print(f"    {combo}: {cnt}")
    top_geo = sub["geo_level_1_id"].value_counts().head(5)
    print("  Top districts (geo_level_1_id):")
    for g, cnt in top_geo.items():
        print(f"    geo_1={g}: {cnt}")


def disagreement_analysis(
    y: np.ndarray, p1: np.ndarray, p2: np.ndarray, meta: pd.DataFrame
) -> None:
    disagree = p1 != p2
    n = disagree.sum()
    print(f"\nTotal disagreements: {n:,} ({100 * n / len(y):.2f}% of dataset)")
    if n == 0:
        return
    print("\nDisagreement confusion (rows=run_001 pred, cols=run_002 pred):")
    cm = confusion_matrix(p1[disagree], p2[disagree], labels=GRADES)
    print(f"{'':>10} " + " ".join(f"R2 pred {g}" for g in GRADES))
    for i, g in enumerate(GRADES):
        print(f"R1 pred {g} " + " ".join(f"{cm[i, j]:>10,}" for j in range(3)))
    print("\nGround truth on disagreement buildings:")
    for g in GRADES:
        c = (y[disagree] == g).sum()
        print(f"  grade {g}: {c:,} ({100 * c / n:.1f}%)")
    # who is right more often on disagreements
    r1_ok = (p1[disagree] == y[disagree]).mean()
    r2_ok = (p2[disagree] == y[disagree]).mean()
    print(f"  Accuracy on disagreements — run_001: {r1_ok:.3f}, run_002: {r2_ok:.3f}")
    sub = meta.loc[disagree]
    print("\nTop foundation_type in disagreements:")
    for v, c in sub["foundation_type"].value_counts().head(5).items():
        print(f"  {v}: {c}")
    print("Top geo_level_1_id in disagreements:")
    for v, c in sub["geo_level_1_id"].value_counts().head(5).items():
        print(f"  geo_1={v}: {c}")


def feature_importance_comparison() -> None:
    def load_fi(run_id: str) -> pd.DataFrame:
        path = ROOT / "runs" / run_id / "feature_importance.csv"
        if path.exists():
            return pd.read_csv(path).sort_values("importance_gain", ascending=False)
        model_path = ROOT / "runs" / run_id / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"No importance for {run_id}")
        model = joblib.load(model_path)
        names = pd.read_csv(PROCESSED_DIR / "X_train.csv").columns
        gain = model.booster_.feature_importance(importance_type="gain")
        return pd.DataFrame({"feature": names, "importance_gain": gain}).sort_values(
            "importance_gain", ascending=False
        )

    fi1 = load_fi(RUN_001)
    fi2 = load_fi(RUN_002)
    fi1["rank_001"] = range(1, len(fi1) + 1)
    fi2["rank_002"] = range(1, len(fi2) + 1)
    merged = fi1.merge(fi2, on="feature", suffixes=("_001", "_002"))
    merged["rank_shift"] = merged["rank_002"] - merged["rank_001"]
    merged = merged.sort_values("importance_gain_002", ascending=False)

    print(f"\n{'Rank':<5} {'Feature':<35} {'run_001 gain':>12} {'run_002 gain':>12} {'shift':>6}")
    print("─" * 75)
    for _, row in merged.head(20).iterrows():
        print(
            f"{int(row['rank_002']):<5} {row['feature']:<35} "
            f"{row['importance_gain_001']:>12.1f} {row['importance_gain_002']:>12.1f} "
            f"{int(row['rank_shift']):>+6}"
        )
    unstable = merged[merged["rank_shift"].abs() > 3].head(10)
    if len(unstable):
        print("\nFeatures with |rank shift| > 3:")
        for _, row in unstable.iterrows():
            print(f"  {row['feature']}: shift {int(row['rank_shift']):+d}")


def build_recommendations(
    geo_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    super_df: pd.DataFrame,
    m001: dict,
    m002: dict,
) -> str:
    lines = [
        "FEATURE ENGINEERING RECOMMENDATIONS",
        "═" * 60,
        "Generated from OOF error analysis (run_001 vs run_002).",
        "",
    ]

    # Priority 1: geo + foundation high dangerous miss
    if "dangerous_miss_rate_run_002" in geo_df.columns:
        bad_geo = geo_df[geo_df["dangerous_miss_rate_run_002"] > 0.25].head(5)
        if len(bad_geo):
            lines.append("Priority 1 (high dangerous miss by district):")
            for _, r in bad_geo.iterrows():
                gid = int(r["geo_level_1_id"])
                lines.append(
                    f"  → geo_level_1={gid} interaction features"
                )
                lines.append(
                    f"    Reason: dangerous_miss_rate={r['dangerous_miss_rate_run_002']:.1%}, "
                    f"error_rate={r.get('error_rate_run_002', float('nan')):.1%}"
                )
            lines.append("")

    high_dm = feat_df[feat_df["dangerous_miss_rate"] > 0.30]
    if len(high_dm):
        lines.append("Priority 2 (feature slices with dangerous_miss > 30%):")
        for _, r in high_dm.drop_duplicates(["feature", "value"]).head(8).iterrows():
            lines.append(
                f"  → {r['feature']}={r['value']} ({r['label']}) on {r['run_id']}: "
                f"dangerous_miss={r['dangerous_miss_rate']:.1%}, n={int(r['n'])}"
            )
        lines.append("")

    unstable = geo_df[geo_df["error_rate_delta"].abs() > 0.05] if "error_rate_delta" in geo_df.columns else pd.DataFrame()
    if len(unstable):
        lines.append("Priority 3 (unstable districts — models disagree >5% error rate):")
        for _, r in unstable.head(5).iterrows():
            lines.append(
                f"  → geo_level_1={int(r['geo_level_1_id'])}: "
                f"Δerror={r['error_rate_delta']:+.3f}"
            )
        lines.append("")

    lines.extend(
        [
            "Additional ideas:",
            "  → geo_1_enc × foundation_type one-hot interactions",
            "  → superstructure count + dominant material type",
            "  → age × height_percentage (tall old buildings)",
            f"  → grade-3 recall gap: run_001={m001.get('grade3_recall', 0):.3f} "
            f"run_002={m002.get('grade3_recall', 0):.3f}",
            "",
        ]
    )
    return "\n".join(lines)


def update_run_metadata(run_id: str, metrics: dict[str, float], ece: dict[str, float]) -> None:
    path = ROOT / "runs" / run_id / "metadata.json"
    with path.open(encoding="utf-8") as f:
        meta = json.load(f)
    meta["evaluated"] = True
    meta["evaluation_path"] = "runs/evaluation/"
    meta["oof_metrics"] = metrics
    meta["oof_ece"] = ece
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def run_sections_3_to_10(
    y: np.ndarray,
    meta: pd.DataFrame,
    oof_001: np.ndarray,
    proba_001: np.ndarray,
    oof_002: np.ndarray,
    proba_002: np.ndarray,
    m001: dict[str, float],
    m002: dict[str, float],
) -> None:
    """Sections 3–10: calibration through engineering recommendations."""
    section_header("SECTION 3: PER-CLASS PROBABILITY CALIBRATION")
    ece_001 = plot_calibration(y, oof_001, proba_001, EVAL_DIR / "calibration_run001.png", RUN_001)
    ece_002 = plot_calibration(y, oof_002, proba_002, EVAL_DIR / "calibration_run002.png", RUN_002)
    print(f"ECE run_001: overall={ece_001['overall']:.4f}  per-class={ {k: round(v,4) for k,v in ece_001.items() if k!='overall'} }")
    print(f"ECE run_002: overall={ece_002['overall']:.4f}  per-class={ {k: round(v,4) for k,v in ece_002.items() if k!='overall'} }")

    section_header("SECTION 4: GEOGRAPHIC ERROR ANALYSIS")
    geo_df = geo_error_table(meta, y, oof_001, oof_002)
    geo_df.to_csv(EVAL_DIR / "geo_error_analysis.csv", index=False)
    print("\nTop districts by dangerous_miss_rate (run_002):")
    show = geo_df.head(15)
    print(f"{'District':>10} {'N':>10} {'Err%':>8} {'DMiss%':>10} {'FAlarm%':>10} {'Δerr':>8}")
    for _, r in show.iterrows():
        print(
            f"geo_1={int(r['geo_level_1_id']):>3} {int(r['n_buildings']):>10,} "
            f"{100*r.get('error_rate_run_002', 0):>7.1f}% "
            f"{100*r.get('dangerous_miss_rate_run_002', 0):>9.1f}% "
            f"{100*r.get('false_alarm_rate_run_002', 0):>9.1f}% "
            f"{100*r.get('error_rate_delta', 0):>+7.1f}pp"
        )
    flag_geo = geo_df[
        (geo_df.get("dangerous_miss_rate_run_002", 0) > 0.25)
        | (geo_df.get("error_rate_delta", 0).abs() > 0.05)
    ]
    print(f"\nFlagged districts (DM>25% or |Δerror|>5pp): {len(flag_geo)}")

    section_header("SECTION 5: FEATURE SLICE ERROR ANALYSIS")
    feat_parts = []
    for feat in CATEGORICAL_FEATURES:
        print(f"\n{feat} error analysis (run_002, sorted by dangerous_miss_rate):")
        ft = feature_slice_table(meta, y, oof_001, oof_002, feat)
        feat_parts.append(ft)
        sub = ft[ft["run_id"] == RUN_002].sort_values("dangerous_miss_rate", ascending=False)
        print(f"{'Value':<6} {'Label':<22} {'N':>8} {'Err%':>7} {'DMiss%':>9} {'G3Rec%':>8}")
        for _, r in sub.iterrows():
            print(
                f"{r['value']:<6} {str(r['label'])[:22]:<22} {int(r['n']):>8,} "
                f"{100*r['error_rate']:>6.1f}% {100*r['dangerous_miss_rate']:>8.1f}% "
                f"{100*r['grade3_recall']:>7.1f}%"
            )
    feat_df = pd.concat(feat_parts, ignore_index=True)
    feat_df.to_csv(EVAL_DIR / "feature_slice_analysis.csv", index=False)
    high_dm = feat_df[feat_df["dangerous_miss_rate"] > 0.30]
    print(f"\nSlices with dangerous_miss > 30%: {len(high_dm)} rows")

    section_header("SECTION 6: SUPERSTRUCTURE MATERIAL ERROR ANALYSIS")
    super_df = superstructure_analysis(meta, y, oof_001, oof_002)
    print(f"\n{'Material':<28} {'Run':<8} {'Present':<8} {'N':>8} {'Err%':>7} {'DMiss%':>9}")
    for _, r in super_df.iterrows():
        print(
            f"{r['material']:<28} {r['run_id']:<8} {str(r['present']):<8} "
            f"{int(r['n']):>8,} {100*r['error_rate']:>6.1f}% {100*r['dangerous_miss_rate']:>8.1f}%"
        )
    print(f"\nPresent vs absent (run_002):")
    for mat in super_df["material"].unique():
        sub = super_df[(super_df["material"] == mat) & (super_df["run_id"] == RUN_002)]
        if len(sub) == 2:
            pr = sub[sub["present"]].iloc[0]
            ab = sub[~sub["present"]].iloc[0]
            delta_err = 100 * (pr["error_rate"] - ab["error_rate"])
            print(
                f"  {mat:<26} present err {100*pr['error_rate']:.1f}% | "
                f"absent err {100*ab['error_rate']:.1f}% | delta {delta_err:+.1f}pp | "
                f"DMiss(present) {100*pr['dangerous_miss_rate']:.1f}%"
            )

    section_header("SECTION 7: HIGH-CONFIDENCE WRONG PREDICTIONS")
    high_confidence_wrong(y, oof_001, proba_001, meta, RUN_001)
    high_confidence_wrong(y, oof_002, proba_002, meta, RUN_002)

    section_header("SECTION 8: WHERE run_001 AND run_002 DISAGREE")
    disagreement_analysis(y, oof_001, oof_002, meta)

    section_header("SECTION 9: FEATURE IMPORTANCE COMPARISON")
    feature_importance_comparison()

    section_header("SECTION 10: SUMMARY — FEATURE ENGINEERING RECOMMENDATIONS")
    rec_text = build_recommendations(geo_df, feat_df, super_df, m001, m002)
    print(rec_text)
    (EVAL_DIR / "engineering_recommendations.txt").write_text(rec_text, encoding="utf-8")

    update_run_metadata(RUN_001, m001, ece_001)
    update_run_metadata(RUN_002, m002, ece_002)

    section_header("UPDATED RUN COMPARISON")
    RunManager().print_all_runs()


def main() -> None:
    t0 = time.time()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    section_header("LOADING DATA & MODEL PARAMS")
    X, y, meta = load_train_frames()
    print(f"Train rows (processed split): {len(X):,}  features: {X.shape[1]}")
    print("Note: OOF uses the same 80% train split as model training (208,480 rows).")
    print(f"SKIP_OOF = {SKIP_OOF}")

    if SKIP_OOF:
        if not oof_cache_exists():
            raise FileNotFoundError(
                "SKIP_OOF=True but OOF cache not found in runs/evaluation/. "
                "Run once with SKIP_OOF=False to generate oof_*.npy and proba_*.npy."
            )
        section_header("LOADING CACHED OOF PREDICTIONS")
        oof_001, proba_001, oof_002, proba_002 = load_oof_cache()
        m001 = compute_metrics(y, oof_001, proba_001)
        m002 = compute_metrics(y, oof_002, proba_002)
        run_sections_3_to_10(y, meta, oof_001, proba_001, oof_002, proba_002, m001, m002)
    else:
        params_001 = load_run_params(RUN_001)
        params_002 = load_run_params(RUN_002)

        section_header("GENERATING OUT-OF-FOLD PREDICTIONS")
        oof_001, proba_001 = generate_oof_predictions(X, y, params_001, RUN_001)
        oof_002, proba_002 = generate_oof_predictions(X, y, params_002, RUN_002)
        save_oof_cache(oof_001, proba_001, oof_002, proba_002)

        m001 = compute_metrics(y, oof_001, proba_001)
        m002 = compute_metrics(y, oof_002, proba_002)

        section_header("SECTION 1: AGGREGATE METRICS")
        summary_df = print_metrics_table(m001, m002)
        summary_df.to_csv(EVAL_DIR / "error_summary.csv", index=False)

        section_header("SECTION 2: CONFUSION MATRICES")
        print_confusion_matrices(y, oof_001, oof_002)
        plot_confusion_side_by_side(y, oof_001, oof_002, EVAL_DIR / "confusion_matrices.png")

        run_sections_3_to_10(y, meta, oof_001, proba_001, oof_002, proba_002, m001, m002)

    print(f"\nAll outputs saved to {EVAL_DIR.relative_to(ROOT)}/")
    print(f"Total runtime: {time.time() - t0:.1f} seconds")


def save_oof_only() -> None:
    """Generate and cache OOF arrays only (no analysis sections)."""
    section_header("SAVE OOF CACHE ONLY")
    X, y, _ = load_train_frames()
    params_001 = load_run_params(RUN_001)
    params_002 = load_run_params(RUN_002)
    oof_001, proba_001 = generate_oof_predictions(X, y, params_001, RUN_001)
    oof_002, proba_002 = generate_oof_predictions(X, y, params_002, RUN_002)
    save_oof_cache(oof_001, proba_001, oof_002, proba_002)
    print("Done. Set SKIP_OOF=True and re-run for sections 3–10.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--save-oof":
        save_oof_only()
    else:
        main()
