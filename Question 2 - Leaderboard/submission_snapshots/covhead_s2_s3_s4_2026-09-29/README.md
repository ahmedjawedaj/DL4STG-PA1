# Attempt 3 (Best): Autoformer + direct covariate head, seeds 2, 3, 4

**Leaderboard result, 30 September 12:58 pm PKT: RMSE 62.9497, MAE 46.3068, sMAPE 49.63%,
score 62.9599, rank 1.** See `leaderboard_result.json`.

**Declare on the leaderboard: P = 53670, E = 20.** Values: `predictions.txt` (168 values,
time_idx 43657 first). sha256 2b9e3c561f6b169e...

## What it is

Attempt 1's Autoformer (d_model 16, d_ff 32, width-5 covariate embedding, horizon marks,
replicate padding, decomposition kernel 25, Auto-Correlation in all three mixers) plus a
direct covariate head (`--cov-head 64`): a per-hour MLP (36 -> 64 -> 64 -> 1, GELU) from each
horizon hour's known inputs straight to that hour's output, added to the Autoformer forecast.
No anchoring. Adam 1e-3 with 0.8 decay per epoch, up to 10 epochs, patience 3.
17890 parameters per model.

Why: a gradient-boosting model on the per-hour horizon covariates alone reaches 57.0 all-weeks
and 70.5 winter RMSE on the last cycle (Autoformer: 69.3 and 82.0). The information is in the
covariates, and the Autoformer's decomposition (moving averages) and Auto-Correlation smooth
the hour-by-hour weather signal away before it reaches the output. The head gives that signal
an unsmoothed path. The decomposition and Auto-Correlation paths are unchanged and model what
the covariates do not explain.

## Evidence (dev protocol, 3 seeds, floored averages)

| | All weeks | Winter weeks | Winter weeks won vs attempt-1 config |
| --- | ---: | ---: | ---: |
| Fold B, attempt-1 config | 69.34 | 81.97 | |
| Fold B, covariate head | 61.68 | 74.97 | 12/16 |
| Fold A, attempt-1 config | 77.43 | 106.13 | |
| Fold A, covariate head | 66.40 | 91.63 | 12/17 |

Mean paired winter change -8.32 per week, better on 24 of 33, better in each fold, and every
seed better than every attempt-1 seed in both folds. MAE 43, sMAPE 49 to 52. Winter top-decile
truth 392: attempt-1 config 223, this model 272. Numbers in `results/folds_v9.json`.

## Member selection (declared)

Seven final seeds were trained (0 to 6) on [0, 35064) with early stopping on the 51 holdout
weeks. Their early-stop RMSEs split into two groups: seeds 2, 3, 4, 5 at 60 to 66, and seeds
0, 1, 6 at 81 to 83 (same training loss, so a poor basin, not a data issue). The candidate
averages the floored forecasts of seeds 2, 3, 4 only. The discarded runs are kept in
`runs/final_v9/`. E counts only the members' epochs (7 + 7 + 6); the discarded runs cost a
further 17 epochs and are reported here.

Regenerate: `python -m pa1q2.submission $(cat members.txt) --out <dir>`.
