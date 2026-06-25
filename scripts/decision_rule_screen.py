#!/usr/bin/env python3
"""Micro-F1-aware decision-rule screen on run_026 OOF probabilities.

Tests GLOBAL post-hoc adjustments to the argmax decision, validated honestly
per-fold (tune multipliers on 4 folds, apply to held-out fold) so we don't
fit the rule to the same data we score on. Three rules:
  (1) prior-matching: rescale class probs so predicted marginal ~= train prior
  (2) 3 per-class probability multipliers, grid+optimize on OOF accuracy
  (3) same multipliers but learned per-fold (out-of-fold) = honest estimate

GATE: a rule only counts if the honest per-fold (rule 3) version improves and
holds in >=4/5 folds. Rule 2 (tuned-on-all) is the OPTIMISTIC upper bound.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RANDOM_STATE = 42
CV_FOLDS = 5
PRIOR = np.array([0.09640792, 0.56891186, 0.33468022])


def acc(y, proba):
    # micro-F1 == accuracy for single-label multiclass; fast vectorized form
    return float((proba.argmax(1) + 1 == y).mean())


def apply_mult(proba, m):
    return proba * m  # argmax unaffected by normalization


def best_mult(proba, y, n_iter=3):
    """Grid + Nelder-Mead over 3 multipliers (m0 fixed=1 by scale invariance)."""
    def neg(m2):  # m for class2,3 ; class1 fixed at 1
        m = np.array([1.0, m2[0], m2[1]])
        return -acc(y, proba * m)
    best = (-neg([1.0, 1.0]), np.array([1.0, 1.0]))
    grid = np.linspace(0.6, 1.6, 21)
    for a in grid:
        for b in grid:
            s = -neg([a, b])
            if s > best[0]:
                best = (s, np.array([a, b]))
    res = minimize(neg, best[1], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    if -res.fun > best[0]:
        best = (-res.fun, res.x)
    return np.array([1.0, best[1][0], best[1][1]]), best[0]


def prior_match(proba, target_prior):
    """Iterative proportional rescale so predicted marginal matches target."""
    m = np.ones(3)
    for _ in range(50):
        pred_marg = np.bincount(((proba * m).argmax(1)), minlength=3) / len(proba)
        ratio = target_prior / (pred_marg + 1e-9)
        m = m * (ratio ** 0.5)
        m = m / m[0]
    return m


def main():
    for base in ["runs/run_026/oof_proba.npy", "runs/run_019/oof_proba.npy",
                 "runs/run_024/oof_proba.npy"]:
        run = Path(base).parent.name
        proba = np.load(ROOT / base).astype(np.float64)
        y = pd.read_csv(ROOT / "data/processed/y_train_full.csv")["damage_grade"].to_numpy()
        base_acc = acc(y, proba)
        pred_marg = np.bincount(proba.argmax(1), minlength=3) / len(proba)
        print(f"\n{'='*64}\n{run}: base OOF acc = {base_acc:.5f}")
        print(f"  pred marginal {pred_marg.round(4)}  vs train prior {PRIOR.round(4)}")

        # Rule 2: optimistic (tune on all OOF)
        m_all, sc_all = best_mult(proba, y)
        print(f"  [OPTIMISTIC] best global mult {m_all.round(3)} -> acc {sc_all:.5f} "
              f"(Δ {sc_all-base_acc:+.5f})")

        # Rule 1: prior-match (no target leak, marginal is structural)
        m_pm = prior_match(proba, PRIOR)
        sc_pm = acc(y, proba * m_pm)
        print(f"  [prior-match] mult {m_pm.round(3)} -> acc {sc_pm:.5f} (Δ {sc_pm-base_acc:+.5f})")

        # Rule 3: HONEST per-fold — learn mult on train folds, apply to val fold
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        oof_adj = proba.copy()
        fold_deltas = []
        for tri, vai in skf.split(proba, y):
            m_fold, _ = best_mult(proba[tri], y[tri])
            oof_adj[vai] = proba[vai] * m_fold
            d = acc(y[vai], proba[vai] * m_fold) - acc(y[vai], proba[vai])
            fold_deltas.append(d)
        sc_honest = acc(y, oof_adj)
        n_pos = sum(d > 0 for d in fold_deltas)
        print(f"  [HONEST per-fold] acc {sc_honest:.5f} (Δ {sc_honest-base_acc:+.5f})  "
              f"folds {[f'{d:+.4f}' for d in fold_deltas]} ({n_pos}/5 pos)")

        # Honest prior-match per fold
        oof_pm = proba.copy()
        pm_deltas = []
        for tri, vai in skf.split(proba, y):
            m_fold = prior_match(proba[tri], PRIOR)
            oof_pm[vai] = proba[vai] * m_fold
            pm_deltas.append(acc(y[vai], proba[vai] * m_fold) - acc(y[vai], proba[vai]))
        sc_pm_h = acc(y, oof_pm)
        print(f"  [HONEST prior-match] acc {sc_pm_h:.5f} (Δ {sc_pm_h-base_acc:+.5f})  "
              f"{sum(d>0 for d in pm_deltas)}/5 pos")


if __name__ == "__main__":
    main()
