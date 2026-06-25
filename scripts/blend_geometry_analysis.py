#!/usr/bin/env python3
"""Read-only deep analysis of run_019/024/026 OOF posteriors.

No training, no writes to runs/. Goal: characterize the error structure and
test cheap alternative DECISION GEOMETRIES on existing probabilities to see if
any honest (per-fold) gain exists over run_026 argmax.

Covers:
  1. Confusion matrix + adjacent-error decomposition.
  2. Alternative blends of the EXISTING three models (proba/logit/rank/geomean),
     honest per-fold alpha selection.
  3. Whether 019+024 raw (not 026) can be re-blended better, per-fold.
  4. Calibration: are run_026 probabilities well-calibrated? (reliability by bin)
  5. Confidence-stratified accuracy: where do errors concentrate?
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
RS, K = 42, 5


def acc(y, P):
    return float((P.argmax(1) + 1 == y).mean())


def main():
    y = pd.read_csv(ROOT / "data/processed/y_train_full.csv")["damage_grade"].to_numpy()
    p19 = np.load(ROOT / "runs/run_019/oof_proba.npy").astype(np.float64)
    p24 = np.load(ROOT / "runs/run_024/oof_proba.npy").astype(np.float64)
    p26 = np.load(ROOT / "runs/run_026/oof_proba.npy").astype(np.float64)
    eps = 1e-7
    N = len(y)
    splits = list(StratifiedKFold(K, shuffle=True, random_state=RS).split(y, y))

    print("=" * 60)
    print(f"run_026 OOF acc {acc(y,p26):.5f}  (19={acc(y,p19):.5f} 24={acc(y,p24):.5f})")

    # 1. Confusion matrix
    pred = p26.argmax(1) + 1
    print("\nConfusion (rows=true, cols=pred):")
    cm = np.zeros((3, 3), int)
    for t in (1, 2, 3):
        for pp in (1, 2, 3):
            cm[t - 1, pp - 1] = ((y == t) & (pred == pp)).sum()
    print("        p1      p2      p3")
    for t in (1, 2, 3):
        print(f"  t{t}  " + "  ".join(f"{cm[t-1,c]:6d}" for c in range(3)))
    errs = (pred != y).sum()
    adj = (((y == 1) & (pred == 2)) | ((y == 2) & (pred == 1)) |
           ((y == 2) & (pred == 3)) | ((y == 3) & (pred == 2))).sum()
    far = (((y == 1) & (pred == 3)) | ((y == 3) & (pred == 1))).sum()
    print(f"  total errors {errs} | adjacent {adj} ({adj/errs:.1%}) | far 1<->3 {far} ({far/errs:.1%})")
    print(f"  dominant: true3->pred2 = {cm[2,1]}, true2->pred3 = {cm[1,2]}, "
          f"true2->pred1 = {cm[1,0]}, true1->pred2 = {cm[0,1]}")

    # 2. honest per-fold alternative geometries of existing 3 models
    def blend(a19, a24, a26, P19, P24, P26, space):
        if space == "proba":
            return a19 * P19 + a24 * P24 + a26 * P26
        if space == "logit":
            return (a19 * np.log(P19 + eps) + a24 * np.log(P24 + eps)
                    + a26 * np.log(P26 + eps))
        if space == "geom":
            return (P19 ** a19) * (P24 ** a24) * (P26 ** a26)

    def rank_avg(P19, P24, P26, w):
        from scipy.stats import rankdata
        R = np.zeros_like(P19)
        for P, wi in [(P19, w[0]), (P24, w[1]), (P26, w[2])]:
            for c in range(3):
                R[:, c] += wi * rankdata(P[:, c])
        return R

    # honest CV: choose weights on train folds, score val fold.
    print("\nHonest per-fold re-blend of {19,24,26}:")
    for space in ("proba", "logit", "geom"):
        oof = np.zeros_like(p26)
        # weight grid on simplex step 0.1
        ws = [(i/10, j/10, (10-i-j)/10) for i in range(11) for j in range(11-i)]
        for tri, vai in splits:
            best = (-1, None)
            for w in ws:
                s = acc(y[tri], blend(*w, p19[tri], p24[tri], p26[tri], space))
                if s > best[0]:
                    best = (s, w)
            w = best[1]
            oof[vai] = blend(*w, p19[vai], p24[vai], p26[vai], space)
        print(f"  {space:6s}: honest OOF {acc(y,oof):.5f}  (Δ {acc(y,oof)-acc(y,p26):+.5f})")

    # rank average honest
    oof = np.zeros_like(p26)
    ws = [(i/10, j/10, (10-i-j)/10) for i in range(11) for j in range(11-i)]
    for tri, vai in splits:
        best = (-1, None)
        for w in ws:
            R = rank_avg(p19[tri], p24[tri], p26[tri], w)
            s = acc(y[tri], R)
            if s > best[0]:
                best = (s, w)
        oof[vai] = rank_avg(p19[vai], p24[vai], p26[vai], best[1])
    print(f"  rank  : honest OOF {acc(y,oof):.5f}  (Δ {acc(y,oof)-acc(y,p26):+.5f})")

    # 3. calibration of run_026 (max-prob reliability)
    print("\nCalibration of run_026 (confidence bins, max-prob):")
    conf = p26.max(1)
    correct = (pred == y)
    for lo, hi in [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        print(f"  conf[{lo:.1f},{hi:.1f}) n={m.sum():>7} pred_conf={conf[m].mean():.3f} "
              f"actual_acc={correct[m].mean():.3f}")

    # 4. how many rows are "flippable" -- where 2nd choice is very close
    margin = np.sort(p26, 1)[:, -1] - np.sort(p26, 1)[:, -2]
    print(f"\nDecision margins: <0.05 margin: {(margin<0.05).mean():.1%} of rows, "
          f"acc there {correct[margin<0.05].mean():.3f}")
    print(f"  These {(margin<0.05).sum()} near-tie rows hold {((~correct)&(margin<0.05)).sum()} "
          f"errors ({((~correct)&(margin<0.05)).sum()/(~correct).sum():.1%} of all errors).")


if __name__ == "__main__":
    main()
