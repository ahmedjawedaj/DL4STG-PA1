# Dev variants, 29 September (not a submission candidate)

Goal: find a change with a clear reason to do better on a winter week than the first
submission's configuration (`abl_sm5_hmark_anchor_rep`, leaderboard RMSE 81.56).

All runs use the dev protocol: fit [0, 26304), early stop on [26304, 35064), score the
51 holdout weeks. Three seeds each. Base settings: known-horizon covariates, d_model 16,
d_ff 32, width-5 covariate embedding, horizon marks, replicate padding, no anchoring.
Each variant changes one setting (see `jobs_v2_*.txt`).

| Variant | RMSE (51 weeks) | Edge-16 RMSE (winter-like) | Winter paired diff vs attempt-1 config | Params |
| --- | ---: | ---: | --- | ---: |
| Attempt-1 config (anchored) | 69.34 | 81.97 | 0 | 11297 |
| Non-anchored base | 70.32 | 80.35 | -1.39 (sd 5.13, 8/16 better) | 11297 |
| Kernel 49 | 69.72 | 80.47 | -0.87 (sd 7.23, 10/16 better) | 11297 |
| Trailing means 6, 24, 72 | 70.94 | 80.61 | -0.03 (sd 9.42, 8/16 better) | 12897 |
| Dropout 0 | 68.15 | 81.44 | -0.06 (sd 3.73, 8/16 better) | 11297 |
| Early stop on winter weeks | 71.97 | 81.99 | +1.13 (sd 7.01, 5/16 better) | 11297 |

Scores are for the floored three-seed average. Full numbers: `variants_v2.json`
(regenerate with `python -m pa1q2.compare_variants`).

Contents: `runs/` holds each run's JSON record (config, curve, P, E) and its holdout
predictions (`.npz`). Dev-stage runs do not save weights, and no full-history forecast,
declaration or weights were produced, because no variant passed the bar below.

Decision: none is a clear improvement for a winter week. Paired changes are about zero
with a week-to-week spread of 4 to 9 RMSE. The active `submission/` folder is unchanged
(sha256 a5eba4fa48fc9f6c62ed17a623f06439ad2765b1f67620b58687cf3be308f4fd).
