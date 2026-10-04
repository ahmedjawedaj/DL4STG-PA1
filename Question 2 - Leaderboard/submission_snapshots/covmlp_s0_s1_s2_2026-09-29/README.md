# Attempt 2: Autoformer with a nonlinear covariate encoder, 3 seeds (submitted 29 September, 10:18 pm PKT)

**Leaderboard result: RMSE 81.9270, MAE 62.7527, sMAPE 59.58%, score 81.9351.** Worse than
attempt 1 (81.5648) by 0.36. The leaderboard keeps the best attempt, so attempt 1 still counts.
See `leaderboard_result.json`.

**Declare on the leaderboard: P = 38979, E = 16.** Values: `predictions.txt` (168 values,
time_idx 43657 first). sha256 bc6cba0063ea66ee1944dda119dabc10556c7dd28c3de9481b9b2d0a31fceedb.

## What changed against attempt 1

Attempt 1's configuration (d_model 16, d_ff 32, width-5 covariate embedding, horizon marks,
anchoring, replicate padding) plus `--mark-mlp 16`: a per-step two-layer MLP
(34 covariate inputs -> 16 -> d_model, GELU) added to the covariate embedding in both the
encoder and the decoder. The linear embedding can only add covariates up; the MLP can respond
to combinations of them (for example humid and calm and a southerly wind), which is where
the pollution peaks come from. 12993 parameters per model, 1696 more than attempt 1's.

Members (`members.txt`, records and weights in `runs/`): seeds 0, 1, 2, each fitted on
[0, 35064) with early stopping on the 51 holdout weeks, exactly as attempt 1. Early-stop
RMSE 64.9, 72.2, 68.0 (attempt 1's members: 68.3, 79.0, 68.0). Forecasts floored at 7 per
model, then averaged. Regenerate with `python -m pa1q2.submission $(cat members.txt) --out <dir>`.

## Evidence (dev protocol, 3 seeds, floored averages)

| | All weeks | Winter weeks | Winter weeks won vs attempt-1 config |
| --- | ---: | ---: | ---: |
| Fold B (last cycle), attempt-1 config | 69.34 | 81.97 | |
| Fold B, covariate MLP | 64.79 | 76.80 | 8/16 |
| Fold A (cycle before), attempt-1 config | 77.43 | 106.13 | |
| Fold A, covariate MLP | 74.36 | 102.48 | 12/17 |

Mean paired winter change -3.19 per week, better on 20 of 33 winter weeks, better in each
fold. This is the only variant of 19 tested that passed the bar set beforehand (at least 3
better, 20 of 33 weeks, both folds). Profile on fold B: MAE 45.1, sMAPE 52.6 (attempt 1's
config: 50.4, 59.3). Top-decile winter truth 392: attempt-1 config predicts 223 on average,
this model 261. Full numbers: `results/folds_v7.json`.

The final forecast correlates 0.95 with attempt 1 (RMSE between them 23), so this is a
materially different forecast, mostly higher on days 2 and 3.
