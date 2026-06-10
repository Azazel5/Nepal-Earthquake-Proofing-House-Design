# Submissions ready for DrivenData upload

Upload each CSV manually at https://www.drivendata.org/competitions/57/nepal-earthquake/

After each upload, record the public score:

```bash
.venv/bin/python scripts/track_submissions.py --run-id run_018 --score YOUR_SCORE
```

| Run | CV micro F1 | Submission path | Notes |
|-----|-------------|-----------------|-------|
| run_018 | 0.7512 | `runs/run_018/submission.csv` | MLP rebuild; blend gain +0.0008 (below 0.7604 threshold) |
| run_019 | 0.7528 | `runs/run_019/submission.csv` | PCA embed-only k=80 + LGBM |
| run_020 | 0.7446 | `runs/run_020/submission.csv` | AE latent 48-d + geo_rates + LGBM (underperformed) |
| run_021 | 0.6907 | `runs/run_021/submission.csv` | Fine-tuned AE classifier (skip for LB) |

List CV vs public gap after uploads:

```bash
.venv/bin/python scripts/track_submissions.py --list
```
