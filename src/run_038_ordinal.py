#!/usr/bin/env python3
"""run_038: Ordinal LGBM via cumulative-link decomposition (Frank & Hall).

The target damage_grade is ordinal (1<2<3) and ALL errors are adjacent-class.
We have only ever trained NOMINAL multiclass. This trains two cumulative binary
LGBMs on the SAME run_019 PCA(embed,k=80) feature view:
    M1: P(y > 1)   (i.e. grade in {2,3})
    M2: P(y > 2)   (i.e. grade == 3)
Then reconstruct an ordinal posterior:
    P(y=1) = 1 - P(y>1)
    P(y=2) = P(y>1) - P(y>2)
    P(y=3) = P(y>2)
clipped to >=0 and renormalized. This respects ordinality and yields a
potentially decorrelated posterior vs the nominal run_026 blend.

Saves OOF/test proba. Then runs the FULL generalization gate vs run_026:
  (a) solo OOF; (b) blend with run_026; (c) per-fold deltas; (d) loss-corr;
  (e) sparse-cell leak check. Registers run_038 only after reporting.

Run:  env/bin/python src/run_038_ordinal.py [--quick]
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from functools import partial
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from lightgbm import LGBMClassifier, early_stopping
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retrain import EARLY_STOPPING_ROUNDS, TRIAL_66_PARAMS
from run_manager import PROCESSED_DIR, RunManager
print = partial(print, flush=True)

RANDOM_STATE = 42
CV_FOLDS = 5
PCA_K = 80
RUN_ID = "run_038"


def acc(y, proba):
    return float((proba.argmax(1) + 1 == y).mean())


def embed_cols(cols):
    return [c for c in cols if "_emb_" in c]


def non_embed_cols(cols):
    return [c for c in cols if "_emb_" not in c]


def transform_pca(X_tr, X_va, X_te, pca_cols, pass_cols, k):
    scaler = StandardScaler()
    pca = PCA(n_components=k, random_state=RANDOM_STATE, svd_solver="randomized")

    def sc_fit(df):
        a = scaler.fit_transform(df[pca_cols].to_numpy(dtype=np.float64))
        return np.clip(np.nan_to_num(a), -10, 10)

    def sc_app(df):
        a = scaler.transform(df[pca_cols].to_numpy(dtype=np.float64))
        return np.clip(np.nan_to_num(a), -10, 10)

    z_tr = pca.fit_transform(sc_fit(X_tr))
    z_va = pca.transform(sc_app(X_va))
    z_te = pca.transform(sc_app(X_te))
    names = [f"pca_{i}" for i in range(k)]
    if pass_cols:
        z_tr = np.hstack([z_tr, X_tr[pass_cols].to_numpy()])
        z_va = np.hstack([z_va, X_va[pass_cols].to_numpy()])
        z_te = np.hstack([z_te, X_te[pass_cols].to_numpy()])
        names = names + pass_cols
    return (pd.DataFrame(z_tr, columns=names),
            pd.DataFrame(z_va, columns=names),
            pd.DataFrame(z_te, columns=names))


def fit_binary(X_tr, z_tr, X_va, z_va):
    p = {k: v for k, v in TRIAL_66_PARAMS.items()
         if k not in ("objective", "num_class", "metric")}
    p.update(objective="binary", metric="binary_logloss")
    m = LGBMClassifier(**p)
    m.fit(X_tr, z_tr, eval_set=[(X_va, z_va)], eval_metric="binary_logloss",
          callbacks=[early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)])
    return m


def ordinal_posterior(p_gt1, p_gt2):
    """p_gt1=P(y>1), p_gt2=P(y>2) -> (N,3) posterior, clipped+renorm."""
    p1 = 1 - p_gt1
    p2 = p_gt1 - p_gt2
    p3 = p_gt2
    P = np.clip(np.stack([p1, p2, p3], axis=1), 1e-6, None)
    return P / P.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    X = pd.read_csv(PROCESSED_DIR / "X_train_run012.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test_run012.csv")
    y = pd.read_csv(PROCESSED_DIR / "y_train_full.csv")["damage_grade"].to_numpy()
    cols = list(X.columns)
    pca_cols, pass_cols = embed_cols(cols), non_embed_cols(cols)
    print(f"features {X.shape[1]} pca_cols {len(pca_cols)} pass {len(pass_cols)}")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(skf.split(X, y))
    n_folds = 1 if args.quick else CV_FOLDS
    oof = np.zeros((len(y), 3), dtype=np.float32)
    test_folds = []
    for fold, (tri, vai) in enumerate(splits[:n_folds], 1):
        tf = time.time()
        df_tr, df_va, df_te = transform_pca(X.iloc[tri], X.iloc[vai], X_test,
                                            pca_cols, pass_cols, PCA_K)
        z1_tr = (y[tri] > 1).astype(int)
        z2_tr = (y[tri] > 2).astype(int)
        z1_va = (y[vai] > 1).astype(int)
        z2_va = (y[vai] > 2).astype(int)
        m1 = fit_binary(df_tr, z1_tr, df_va, z1_va)
        m2 = fit_binary(df_tr, z2_tr, df_va, z2_va)
        pg1_va = m1.predict_proba(df_va)[:, 1]
        pg2_va = m2.predict_proba(df_va)[:, 1]
        oof[vai] = ordinal_posterior(pg1_va, pg2_va).astype(np.float32)
        pg1_te = m1.predict_proba(df_te)[:, 1]
        pg2_te = m2.predict_proba(df_te)[:, 1]
        test_folds.append(ordinal_posterior(pg1_te, pg2_te).astype(np.float32))
        print(f"  fold {fold}: solo acc={acc(y[vai], oof[vai]):.4f} ({time.time()-tf:.0f}s)")

    if args.quick:
        print(f"quick done {time.time()-t0:.0f}s, solo OOF(fold1)={acc(y[splits[0][1]], oof[splits[0][1]]):.4f}")
        return

    test_p = np.mean(test_folds, axis=0)
    solo = acc(y, oof)
    print(f"\nordinal solo OOF acc = {solo:.5f}")

    # ── Generalization analysis vs run_026 ──
    p26 = np.load(ROOT / "runs/run_026/oof_proba.npy").astype(np.float64)
    t26 = np.load(ROOT / "runs/run_026/test_proba.npy").astype(np.float64)
    f26 = acc(y, p26)
    eps = 1e-7
    idx = y - 1
    N = len(y)
    la = -np.log(oof[np.arange(N), idx] + eps)
    lb = -np.log(p26[np.arange(N), idx] + eps)
    loss_corr = float(np.corrcoef(la, lb)[0, 1])
    print(f"run_026 OOF {f26:.5f}  loss-corr(ordinal,run_026) = {loss_corr:.4f}")

    # blend (proba + logit grid)
    best = (f26, "none", 0.0)
    for space in ("proba", "logit"):
        for a in np.linspace(0, 1, 101):
            if space == "proba":
                b = a * oof + (1 - a) * p26
            else:
                b = a * np.log(oof + eps) + (1 - a) * np.log(p26 + eps)
            s = acc(y, b)
            if s > best[0]:
                best = (s, space, float(a))
    blend_acc, space, alpha = best
    print(f"best blend: {space} alpha(ordinal)={alpha:.3f} acc={blend_acc:.5f} "
          f"(Δ vs run_026 {blend_acc-f26:+.5f})")

    if space == "proba":
        oof_b = alpha * oof + (1 - alpha) * p26
        test_b = alpha * test_p + (1 - alpha) * t26
    elif space == "logit":
        oof_b = alpha * np.log(oof + eps) + (1 - alpha) * np.log(p26 + eps)
        test_b = alpha * np.log(test_p + eps) + (1 - alpha) * np.log(t26 + eps)
    else:
        oof_b, test_b = p26, t26

    # per-fold delta vs run_026
    fold_deltas = []
    for tri, vai in splits:
        fold_deltas.append(acc(y[vai], oof_b[vai]) - acc(y[vai], p26[vai]))
    n_pos = sum(d > 0 for d in fold_deltas)
    print(f"per-fold Δ: {[f'{d:+.4f}' for d in fold_deltas]} ({n_pos}/5 pos)")

    # sparse-cell leak check (foundation×geo3)
    tv = pd.read_csv(ROOT / "data/driven_data/train_values.csv",
                     usecols=["building_id", "geo_level_3_id", "foundation_type"])
    cell = list(zip(tv["geo_level_3_id"], tv["foundation_type"]))
    cell_n = pd.Series(cell).map(pd.Series(cell).value_counts()).to_numpy()
    corr_b = (p26.argmax(1) + 1 == y)
    corr_n = (oof_b.argmax(1) + 1 == y)
    print("gain by foundation×geo3 cell size:")
    dense_gain = 0.0
    for lo, hi, lab in [(0, 5, "<5"), (5, 50, "5-50"), (50, 200, "50-200"), (200, 10**9, "200+")]:
        m = (cell_n >= lo) & (cell_n < hi)
        if m.sum() == 0:
            continue
        g = float(corr_n[m].mean() - corr_b[m].mean())
        if lab == "200+":
            dense_gain = g
        print(f"  {lab:<7} n={m.sum():>7} gain={g:+.4f}")

    delta = blend_acc - f26
    passes = (delta > 0) and (dense_gain >= -0.0005) and (n_pos >= 4)
    print(f"\nGATE: {'PASS' if passes else 'FAIL'} "
          f"(Δ>0:{delta>0}, dense≥0:{dense_gain>=-0.0005}, folds≥4:{n_pos>=4})")

    rm = RunManager()
    rm.create_run(
        description="Ordinal LGBM (Frank-Hall cumulative-link) on run_019 PCA view",
        model_type="LightGBM_ordinal",
        feature_set="pca_embed_k80+passthrough (run_019 view), 2 cumulative binaries",
        params={"pca_k": PCA_K, "decomposition": "FrankHall_cumulative",
                "solo_oof": solo, "blend_space": space, "blend_alpha": alpha,
                "loss_corr_vs_run026": loss_corr},
        run_id=RUN_ID, objective="ordinal", n_features=PCA_K + len(pass_cols),
        cv_folds=CV_FOLDS, cv_metric="micro_f1",
        notes="Ordinal posterior to decorrelate from nominal run_026 blend.")
    rm.save_cv_scores(RUN_ID, [float(d) for d in fold_deltas], solo, 0.0)
    rd = rm.run_path(RUN_ID)
    np.save(rd / "oof_proba.npy", oof)
    np.save(rd / "test_proba.npy", test_p.astype(np.float32))
    np.save(rd / "oof_blend26.npy", oof_b.astype(np.float32))
    np.save(rd / "test_blend26.npy", test_b.astype(np.float32))
    with open(rd / "eval_report.json", "w") as f:
        json.dump({"solo_oof": solo, "run026_oof": f26, "blend_oof": blend_acc,
                   "delta": delta, "loss_corr": loss_corr, "blend_space": space,
                   "blend_alpha": alpha, "per_fold_delta": [float(d) for d in fold_deltas],
                   "dense_gain": dense_gain, "gate_pass": bool(passes)}, f, indent=2)
    print(f"\nRegistered {RUN_ID} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
