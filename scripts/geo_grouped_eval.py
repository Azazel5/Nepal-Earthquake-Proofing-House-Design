#!/usr/bin/env python3
"""Read-only: evaluate existing OOF posteriors under a GEO-GROUPED partition.

Our canonical CV is StratifiedKFold (random). The test set contains buildings in
geo_level_2/3 units; some test geo units may be under-represented in train. If
StratifiedKFold is optimistic for the unseen-geo regime, model ordering could
differ when we score within geo groups. We can't retrain (other agent owns that),
but we CAN partition the EXISTING OOF predictions by geo group and ask:
  - Does run_026 still beat 019 and 024 within rare-geo strata?
  - Is the blend advantage uniform across geo density, or concentrated?

This reuses the already-computed OOF (each row predicted by a model that did NOT
see that row), so accuracy within any geo stratum is honest.
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore"); np.seterr(all="ignore")

ROOT = Path(__file__).resolve().parents[1]


def acc(y, P, m=None):
    pred = P.argmax(1) + 1
    if m is not None:
        return float((pred[m] == y[m]).mean())
    return float((pred == y).mean())


def main():
    y = pd.read_csv(ROOT / "data/processed/y_train_full.csv")["damage_grade"].to_numpy()
    p19 = np.load(ROOT / "runs/run_019/oof_proba.npy").astype(np.float64)
    p24 = np.load(ROOT / "runs/run_024/oof_proba.npy").astype(np.float64)
    p26 = np.load(ROOT / "runs/run_026/oof_proba.npy").astype(np.float64)
    tv = pd.read_csv(ROOT / "data/driven_data/train_values.csv",
                     usecols=["building_id", "geo_level_1_id", "geo_level_2_id", "geo_level_3_id"])

    for lvl in ["geo_level_2_id", "geo_level_3_id"]:
        g = tv[lvl].to_numpy()
        n = pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
        print(f"\n=== stratify by {lvl} density ===")
        for lo, hi, lab in [(0, 20, "<20"), (20, 100, "20-100"),
                            (100, 500, "100-500"), (500, 10**9, "500+")]:
            m = (n >= lo) & (n < hi)
            if m.sum() == 0:
                continue
            print(f"  {lab:<8} n={m.sum():>7}  "
                  f"19={acc(y,p19,m):.4f} 24={acc(y,p24,m):.4f} 26={acc(y,p26,m):.4f}  "
                  f"26-best19/24={acc(y,p26,m)-max(acc(y,p19,m),acc(y,p24,m)):+.4f}")

    # What fraction of TEST geo3 units are unseen / rare in train?
    te = pd.read_csv(ROOT / "data/driven_data/test_values.csv",
                     usecols=["geo_level_3_id", "geo_level_2_id"])
    for lvl in ["geo_level_2_id", "geo_level_3_id"]:
        tr_counts = tv[lvl].value_counts()
        te_units = te[lvl]
        seen = te_units.map(tr_counts).fillna(0)
        print(f"\nTEST rows by train-{lvl} support: "
              f"unseen={int((seen==0).sum())} ({(seen==0).mean():.1%}), "
              f"<20={(seen<20).mean():.1%}, <100={(seen<100).mean():.1%}")


if __name__ == "__main__":
    main()
