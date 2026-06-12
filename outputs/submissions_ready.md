# Submissions ready for DrivenData upload

Upload each CSV manually at https://www.drivendata.org/competitions/57/nepal-earthquake/

After each upload, record the public score:

```bash
.venv/bin/python scripts/track_submissions.py --run-id run_018 --score YOUR_SCORE
```

| Run | CV micro F1 | Submission path | Notes |
|-----|-------------|-----------------|-------|
| run_018 | 0.7512 | `runs/run_018/submission.csv` | **Submitted — public 0.7477 (flop)** |
| run_019 | 0.7528 | `runs/run_019/submission.csv` | **Submitted — public 0.7520 (new SOTA)** |
| run_020 | 0.7446 | `runs/run_020/submission.csv` | AE latent 48-d + geo_rates + LGBM (underperformed) |
| run_021 | 0.6907 | `runs/run_021/submission.csv` | Fine-tuned AE classifier (skip for LB) |
| run_022 | 0.7480 | `runs/run_022/submission.csv` | Hybrid PCA numerics k=3 + cat AE 32 + LGBM (do not submit) |
| run_023 | 0.7588 | `runs/run_023/submission.csv` | **Blend run_012 + run_019** (α≈0.98 on run_012, logit space) |

List CV vs public gap after uploads:

```bash
.venv/bin/python scripts/track_submissions.py --list
```
