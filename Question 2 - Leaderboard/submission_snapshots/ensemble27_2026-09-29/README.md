# Candidate: 27-model mixed ensemble (built 29 September, not submitted)

Equal-weight average of 27 Autoformer forecasts, each floored at the fit-region 1st
percentile before averaging (same rule as attempt 1). 9 configurations, 3 seeds each.
Every member fits [0, 35064) and early-stops on the 51 holdout weeks, like attempt 1.

| Configuration | Members | Notes |
| --- | ---: | --- |
| attempt-1 config (anchored, horizon marks, width-5 embedding, replicate padding) | 3 | the 3 models submitted as attempt 1 |
| same without anchoring | 3 | 2 existing finals + 1 new (seed 1) |
| decomposition kernel 49 | 3 | new |
| dropout 0 | 3 | new |
| trailing covariate means 6, 24, 72 | 3 | new |
| winter-weighted loss (x3), anchored | 3 | new |
| square-root target, anchored | 3 | new |
| width-5 embedding, no horizon marks | 3 | new |
| d_model 32, d_ff 64, no horizon marks | 3 | new |

**Declare on the leaderboard: P = 345435, E = 122.** These are the sums over all 27
members, as the handout requires for an ensemble (`declaration.json`). `predictions.txt`
holds the 168 values, time_idx 43657 first. Members and weights are in `runs/`, listed in
`members.txt`. Regenerate with:

    python -m pa1q2.submission $(cat members.txt) --out <dir>

## Evidence (dev protocol, fit to 26304, scored on the last cycle)

The same 27 configurations and seeds trained under the dev protocol, averaged the same way,
against the attempt-1 configuration's 3-seed dev average:

| | All 51 weeks | 16 winter-like weeks |
| --- | ---: | ---: |
| attempt-1 config (3 models) | 69.34 | 81.97 |
| 27-model ensemble | 68.12 | 79.38 |
| paired weekly difference | -1.44 (better on 32/51) | -2.37, sd 4.42 (better on 12/16) |

Leaving any one configuration out of the dev ensemble moves the winter RMSE only between
79.2 and 79.6, so the gain is from averaging many different models, not one lucky
configuration. On the earlier winter (fold A) only a 3-configuration subset could be tested
and it was flat, so this candidate did not pass the pre-set bar (3-point gain, 20 of 33
winter weeks, both folds). The final forecast correlates 0.997 with attempt 1 (RMSE between
them 6.1), so the realistic change is small either way.

Leaderboard penalty for the extra size, estimated from the four visible scores:
about alpha = 1.9e-9 per parameter and beta = 5.0e-4 per epoch, so roughly +0.06 on the
score. Negligible against the RMSE term.
