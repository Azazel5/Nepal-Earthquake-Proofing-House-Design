#!/usr/bin/env python3
"""
run_010 feature engineering: four new blocks on top of run_003 embedded features.

No embedding retraining. Adds ~16 features:

  Block A — Cross-conditional geo rates (~6 features)
    P(grade | geo3, foundation_type)   — Laplace-smoothed, 5-fold CV on train
    P(grade | geo3, material_group)    — groups: rc / weak / other

  Block B — Neighbourhood aggregates per geo3 (~5 features)
    log1p(n_buildings), mean_age, prop_weak, prop_rc, prop_multistory
    Computed from input features only — no label leakage, no CV needed.

  Block C — Extended geo × numeric interactions (3 features)
    geo3_p_grade3 × age, × height_percentage, × count_floors_pre_eq

  Block D — Geo deviation (2 features)
    geo3_p_grade3 − geo2_p_grade3
    geo3_p_grade3 − geo1_p_grade3

Run from project root:
    python src/features_v2.py [--skip-retrain]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embed import (
    EMBED_PREFIX,
    GEO_RATE_COLS,
    GRADES,
    RANDOM_STATE,
    apply_age_sentinel,
    lightgbm_train_split,
    load_raw_data,
)
from run_manager import PROCESSED_DIR

print = partial(print, flush=True)

ALPHA_CROSS = 5        # smaller alpha — cross-groups are sparser than pure geo groups
CV_FOLDS = 5

_WEAK_COLS = [
    "has_superstructure_adobe_mud",
    "has_superstructure_mud_mortar_stone",
    "has_superstructure_bamboo",
]
_RC_COLS = [
    "has_superstructure_rc_engineered",
    "has_superstructure_rc_non_engineered",
]


# ── helpers ───────────────────────────────────────────────────────────────────


def _material_group(df: pd.DataFrame) -> pd.Series:
    """Assign each building to 'rc' / 'weak' / 'other'. rc takes priority."""
    weak_cols = [c for c in _WEAK_COLS if c in df.columns]
    rc_cols   = [c for c in _RC_COLS   if c in df.columns]
    is_weak = df[weak_cols].max(axis=1).astype(bool) if weak_cols else pd.Series(False, index=df.index)
    is_rc   = df[rc_cols].max(axis=1).astype(bool)   if rc_cols   else pd.Series(False, index=df.index)
    grp = pd.Series("other", index=df.index)
    grp[is_weak] = "weak"
    grp[is_rc]   = "rc"
    return grp


def _global_rates(df: pd.DataFrame) -> dict[int, float]:
    n = len(df)
    return {g: (df["damage_grade"] == g).sum() / n for g in GRADES}


def _cross_rate_maps(
    ref: pd.DataFrame,
    geo_col: str,
    cat: pd.Series,
    global_rates: dict[int, float],
) -> dict[int, dict]:
    """Laplace-smoothed P(grade | geo_id, cat_val) from ref rows only."""
    key = pd.Series(list(zip(ref[geo_col].values, cat.values)), index=ref.index)
    totals = key.value_counts()
    maps: dict[int, dict] = {}
    for g in GRADES:
        cnt = key[ref["damage_grade"] == g].value_counts()
        maps[g] = {
            k: (cnt.get(k, 0) + ALPHA_CROSS * global_rates[g]) / (n + ALPHA_CROSS)
            for k, n in totals.items()
        }
    return maps


def _apply_cross_rates(
    df: pd.DataFrame,
    geo_col: str,
    cat: pd.Series,
    maps: dict[int, dict],
    global_rates: dict[int, float],
    prefix: str,
) -> pd.DataFrame:
    key = list(zip(df[geo_col].values, cat.values))
    out = {}
    for g in GRADES:
        out[f"{prefix}_p_grade{g}"] = [maps[g].get(k, global_rates[g]) for k in key]
    return pd.DataFrame(out, index=df.index)


def _cv_cross_rates(
    train: pd.DataFrame,
    geo_col: str,
    cat: pd.Series,
    global_rates: dict[int, float],
    prefix: str,
) -> pd.DataFrame:
    """5-fold CV: each row's rate is computed from the other 4 folds only."""
    cols = [f"{prefix}_p_grade{g}" for g in GRADES]
    # initialise with global rates so any unfilled row has a sensible value
    out = pd.DataFrame(
        {f"{prefix}_p_grade{g}": global_rates[g] for g in GRADES},
        index=train.index,
        dtype=float,
    )
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for _, (tr_idx, val_idx) in enumerate(skf.split(train, train["damage_grade"])):
        maps = _cross_rate_maps(train.iloc[tr_idx], geo_col, cat.iloc[tr_idx], global_rates)
        assigned = _apply_cross_rates(
            train.iloc[val_idx], geo_col, cat.iloc[val_idx], maps, global_rates, prefix
        )
        for c in cols:
            out.iloc[val_idx, out.columns.get_loc(c)] = assigned[c].values
    return out


# ── Block B ───────────────────────────────────────────────────────────────────


def _neighbourhood_aggregates(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-geo3 statistics derived from input features only (no damage labels).
    train stats are used as the lookup table for both train and test.
    """
    geo = "geo_level_3_id"
    weak_cols = [c for c in _WEAK_COLS if c in train.columns]
    rc_cols   = [c for c in _RC_COLS   if c in train.columns]

    aux = train[[geo, "age", "count_floors_pre_eq"]].copy()
    aux["_weak"]       = train[weak_cols].max(axis=1).values if weak_cols else 0
    aux["_rc"]         = train[rc_cols].max(axis=1).values   if rc_cols   else 0
    aux["_multistory"] = (train["count_floors_pre_eq"] > 2).astype(int).values

    agg = aux.groupby(geo).agg(
        count=("age", "count"),
        mean_age=("age", "mean"),
        prop_weak=("_weak", "mean"),
        prop_rc=("_rc", "mean"),
        prop_multistory=("_multistory", "mean"),
    )
    agg["log1p_count"] = np.log1p(agg["count"])

    # global fallbacks for unseen geo3 IDs in test
    fb = {
        "log1p_count":    np.log1p(agg["count"].mean()),
        "mean_age":       train["age"].mean(),
        "prop_weak":      aux["_weak"].mean(),
        "prop_rc":        aux["_rc"].mean(),
        "prop_multistory": aux["_multistory"].mean(),
    }
    stat_cols = ["log1p_count", "mean_age", "prop_weak", "prop_rc", "prop_multistory"]

    def _lookup(df: pd.DataFrame) -> pd.DataFrame:
        rows = {}
        for s in stat_cols:
            rows[f"neigh_{s}"] = df[geo].map(agg[s]).fillna(fb[s]).values
        return pd.DataFrame(rows, index=df.index)

    return _lookup(train), _lookup(test)


# ── Blocks C & D ─────────────────────────────────────────────────────────────


def _extended_interactions(X: pd.DataFrame) -> pd.DataFrame:
    r = X["geo3_p_grade3"].values
    return pd.DataFrame({
        "geo3_rate_x_age":    r * X["age"].values,
        "geo3_rate_x_height": r * X["height_percentage"].values,
        "geo3_rate_x_floors": r * X["count_floors_pre_eq"].values,
    }, index=X.index)


def _geo_deviation(X: pd.DataFrame) -> pd.DataFrame:
    g3 = X["geo3_p_grade3"].values
    return pd.DataFrame({
        "geo3_minus_geo2_p3": g3 - X["geo2_p_grade3"].values,
        "geo3_minus_geo1_p3": g3 - X["geo1_p_grade3"].values,
    }, index=X.index)


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-retrain", action="store_true",
                   help="Build feature CSVs but don't call retrain.py")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # ── Load existing embedded features ──────────────────────────────────────
    print("── Loading existing embedded features ──")
    X_train_emb = pd.read_csv(PROCESSED_DIR / "X_train_embedded.csv")
    X_test_emb  = pd.read_csv(PROCESSED_DIR / "X_test_embedded.csv")
    y_train     = pd.read_csv(PROCESSED_DIR / "y_train_multiclass.csv")["damage_grade"].to_numpy()
    print(f"  X_train_embedded: {X_train_emb.shape}")
    print(f"  X_test_embedded:  {X_test_emb.shape}")

    # ── Load raw data (same split as embed.py) ────────────────────────────────
    print("\n── Loading raw data ──")
    train_full, test_raw = load_raw_data()
    train_full, test_raw = apply_age_sentinel(train_full, test_raw)
    train_df = lightgbm_train_split(train_full).reset_index(drop=True)
    print(f"  Raw train rows: {len(train_df)}")

    assert len(train_df) == len(X_train_emb), (
        f"Row mismatch: raw={len(train_df)}, embedded={len(X_train_emb)}"
    )
    assert (train_df["damage_grade"].values == y_train).all(), "Label alignment mismatch!"

    global_rates = _global_rates(train_df)
    mat_train = _material_group(train_df)
    mat_test  = _material_group(test_raw)

    # ── Block A: cross-conditional geo rates ──────────────────────────────────
    print("\n── Block A: cross-conditional geo rates ──")
    print("  geo3 × foundation_type (5-fold CV)...")
    fdn_train = train_df["foundation_type"].astype(str)
    fdn_test  = test_raw["foundation_type"].astype(str)

    a_fdn_train = _cv_cross_rates(train_df, "geo_level_3_id", fdn_train, global_rates, "geo3_x_fdn")
    a_fdn_maps  = _cross_rate_maps(train_df, "geo_level_3_id", fdn_train, global_rates)
    a_fdn_test  = _apply_cross_rates(test_raw, "geo_level_3_id", fdn_test, a_fdn_maps, global_rates, "geo3_x_fdn")

    print("  geo3 × material_group (5-fold CV)...")
    a_mat_train = _cv_cross_rates(train_df, "geo_level_3_id", mat_train, global_rates, "geo3_x_mat")
    a_mat_maps  = _cross_rate_maps(train_df, "geo_level_3_id", mat_train, global_rates)
    a_mat_test  = _apply_cross_rates(test_raw, "geo_level_3_id", mat_test, a_mat_maps, global_rates, "geo3_x_mat")

    print(f"  Block A: {a_fdn_train.shape[1] + a_mat_train.shape[1]} features")

    # ── Block B: neighbourhood aggregates ─────────────────────────────────────
    print("\n── Block B: neighbourhood aggregates ──")
    b_train, b_test = _neighbourhood_aggregates(train_df, test_raw)
    print(f"  Block B: {b_train.shape[1]} features")

    # ── Block C: extended interactions ────────────────────────────────────────
    print("\n── Block C: extended geo × numeric interactions ──")
    c_train = _extended_interactions(X_train_emb)
    c_test  = _extended_interactions(X_test_emb)
    print(f"  Block C: {c_train.shape[1]} features")

    # ── Block D: geo deviation ────────────────────────────────────────────────
    print("\n── Block D: geo deviation ──")
    d_train = _geo_deviation(X_train_emb)
    d_test  = _geo_deviation(X_test_emb)
    print(f"  Block D: {d_train.shape[1]} features")

    # ── Assemble ──────────────────────────────────────────────────────────────
    print("\n── Assembling final matrices ──")

    def _concat(base: pd.DataFrame, *blocks: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([base] + [b.reset_index(drop=True) for b in blocks], axis=1)

    X_train_v2 = _concat(X_train_emb, a_fdn_train, a_mat_train, b_train, c_train, d_train)
    X_test_v2  = _concat(X_test_emb,  a_fdn_test,  a_mat_test,  b_test,  c_test,  d_test)

    n_new = X_train_v2.shape[1] - X_train_emb.shape[1]
    print(f"  X_train_v2: {X_train_v2.shape}  (+{n_new} new features)")
    print(f"  X_test_v2:  {X_test_v2.shape}")

    new_cols = list(X_train_v2.columns[X_train_emb.shape[1]:])
    print(f"  New columns: {new_cols}")

    nan_train = X_train_v2.isna().sum().sum()
    nan_test  = X_test_v2.isna().sum().sum()
    if nan_train or nan_test:
        print(f"  WARNING: {nan_train} train NaNs, {nan_test} test NaNs — investigate before training")
        for c in X_train_v2.columns[X_train_v2.isna().any()]:
            print(f"    {c}: {X_train_v2[c].isna().sum()} NaN")
    else:
        print("  No NaN values.")

    out_train = PROCESSED_DIR / "X_train_v2.csv"
    out_test  = PROCESSED_DIR / "X_test_v2.csv"
    X_train_v2.to_csv(out_train, index=False)
    X_test_v2.to_csv(out_test,  index=False)
    print(f"\n  Saved {out_train.name} and {out_test.name}")
    print(f"  Feature engineering: {time.time() - t0:.1f}s")

    if args.skip_retrain:
        print("\n--skip-retrain set; done.")
        return

    # ── Retrain ───────────────────────────────────────────────────────────────
    from run_manager import RunManager
    run_id = RunManager().get_next_run_id()
    print(f"\n── Triggering retrain.py ({run_id}) ──")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src" / "retrain.py"),
            "--run-id",     run_id,
            "--x-train",    str(out_train),
            "--x-test",     str(out_test),
            "--description",
            "run_003 features + cross-geo rates + neighbourhood stats + extended interactions",
        ],
        check=True,
        cwd=str(ROOT),
    )
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
