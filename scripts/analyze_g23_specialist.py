#!/usr/bin/env python3
"""Error analysis for run_030 G2|G3 specialist vs run_026."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g23_features import FOUNDATION_WEAK, load_g23_frames
from run_030 import _combine_probs

RUN_026 = ROOT / "runs" / "run_026"
RUN_030 = ROOT / "runs" / "run_030"


def main() -> None:
    y = pd.read_csv(ROOT / "data/processed/y_train_full.csv")["damage_grade"].to_numpy()
    p26 = np.load(RUN_026 / "oof_proba.npy")
    sp = np.load(RUN_030 / "oof_specialist_g3.npy")
    train, _ = load_g23_frames()

    pred26 = p26.argmax(1) + 1
    g3, g2 = y == 3, y == 2
    g23 = g2 | g3
    missed_g3 = g3 & (pred26 == 2)
    mass = p26[:, 1] + p26[:, 2]
    q26 = np.divide(p26[:, 2], mass, out=np.zeros_like(mass), where=mass > 1e-9)
    delta = sp - q26

    print("=" * 72)
    print("G2|G3 SPECIALIST ERROR ANALYSIS")
    print("=" * 72)

    print("\n── 1. The 0.8 recall metric vs what blending needs ──")
    sp50 = (sp >= 0.5).astype(int)
    print(f"Specialist recall@0.5 (OOF):     {recall_score(g3, sp50):.4f}")
    print(f"Precision@0.5 on G2|G3:          {precision_score(g3[g23], sp50[g23]):.4f}")
    print(f"AUC on G2|G3:                    {roc_auc_score(g3[g23], sp[g23]):.4f}")
    print(f"run_026 G3 recall:               {recall_score(g3, pred26 == 3):.4f}")
    print(f"run_026 G3→G2 errors:            {missed_g3.sum():,} ({missed_g3.mean() / g3.mean() * 100:.1f}% of G3)")
    for tau in (0.5, 0.55, 0.62, 0.7):
        n = (missed_g3 & (sp >= tau)).sum()
        print(f"  missed G3 with sp>={tau}:       {n:,} ({n / missed_g3.sum() * 100:.1f}%)")

    print("\n── 2. Why blend barely moves (upgrade funnel) ──")
    tau, boost = 0.62, 0.05
    mask = (pred26 == 2) & (mass > 0.5) & (sp >= tau)
    q_new = np.clip(q26[mask] + boost * (sp[mask] - q26[mask]), q26[mask], 1.0)
    flips = q_new > 0.5
    print(f"Upgrade candidates (pred G2, sp>={tau}): {mask.sum():,}")
    print(f"  true G3: {(mask & g3).sum():,}  true G2: {(mask & g2).sum():,}")
    print(f"  conditional flip @ boost={boost}: {flips.sum():,}  (G3: {(flips & g3[mask]).sum()}, G2: {(flips & g2[mask]).sum()})")

    print("\n── 3. Specialist is not orthogonal to run_026 ──")
    from scipy.stats import spearmanr

    rho, _ = spearmanr(sp[g23], p26[g23, 2])
    print(f"Spearman(sp, p26_P3) on G2|G3:   {rho:.3f}")
    for label, m in [
        ("G3→G2 errors", missed_g3),
        ("true G2, pred G2", g2 & (pred26 == 2)),
        ("true G3, pred G3", g3 & (pred26 == 3)),
    ]:
        print(f"  delta=sp-q26  {label:18s}  mean={delta[m].mean():+.3f}")

    print("\n── 4. Weak foundation — specialist underconfident on hardest misses ──")
    weak = train["foundation_type"].astype(str).isin(FOUNDATION_WEAK).to_numpy()
    for label, m in [("weak-foundation G3", g3 & weak), ("other G3", g3 & ~weak)]:
        miss = m & missed_g3
        print(f"{label}: miss_rate={miss.sum() / m.sum() * 100:.1f}%  sp|miss={sp[miss].mean():.3f}  sp|all={sp[m].mean():.3f}")

    print("\n── 5. Threshold sweep (pred G2 flips, net true G3) ──")
    for tau in np.arange(0.45, 0.71, 0.05):
        flip = (pred26 == 2) & (sp >= tau)
        net = (flip & g3).sum() - (flip & g2).sum()
        rec = recall_score(g3[g23], (sp[g23] >= tau).astype(int))
        print(f"  tau={tau:.2f}  rec_G23={rec:.3f}  flips={flip.sum():5d}  net_G3={net:+6d}")

    print("\n── 6. Oracle ceiling (perfect fixes on run_026 G2 preds) ──")
    pred_g2 = pred26 == 2
    oracle_rec = recall_score(g3, np.where(g3 & pred_g2, 3, pred26))
    base_rec = recall_score(g3, pred26 == 3)
    print(f"  G3 recall if all true G3 among pred-G2 flipped: {oracle_rec:.4f} (+{oracle_rec - base_rec:.4f})")
    print(f"  Specialist @0.5 catches ~{int(0.8 * (g3 & pred_g2).sum()):,} of {(g3 & pred_g2).sum():,} — but with massive G2 FP cost")


if __name__ == "__main__":
    main()
