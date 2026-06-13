# run_033 G3 False-Negative Report

Base: **run_026** OOF F1=0.7546, G3 recall=0.6430

## Summary
- G3→G2 errors: **30,607** / 87,218 true G3 (35.1%)
- G3→G1 errors: 533
- Pred-G2 pool G3 rate: 19.4%
- q26 gap (G3 vs G2 in pool): 0.1113

## Recommendations
- ORDINAL: 533 G3→G1 skips — cumulative/ordinal loss worth testing
- BLEND: q26 gap exists — targeted upgrades on high-lift combos may work with F1 floor
- PSEUDO-LABEL: rare geo3 in test — check train vs test geo3 coverage before PL round
