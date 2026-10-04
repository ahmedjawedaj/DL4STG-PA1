# Task 2 Variant Log

This log tracks saved candidates, historical experiments, and submitted results.

## Saved fallback: `ensemble_s0_s2_2026-09-28`

- Source runs:
  - `runs/final/future-log0-ph1-L168-d16-s0-final.json`
  - `runs/final/future-log0-ph1-L168-d16-s2-final_s2.json`
- Submission snapshot:
  - `submission_snapshots/ensemble_s0_s2_2026-09-28/`
- Model: Autoformer, future known covariates, horizon marks, mark kernel 5, replicate padding.
- Aggregation: floor each seed forecast at the training 1st percentile, then average seed 0 and seed 2.
- Declared resources: `P = 22594`, `E = 10`.
- Dev evidence: full holdout RMSE about `69.94`; edge/winter-like 16-origin RMSE about `79.32`.
- Status: current safe fallback; no leaderboard submission made from this thread.

## Next variants to test

- `d32_hmark_rep`: same feature setup as fallback, wider hidden size (`d_model=32`, `d_ff=64`), longer training patience.
- `d64_l2_hmark_rep`: larger Autoformer capacity (`d_model=64`, `d_ff=128`, two encoder layers), tested only if the cheaper wider run looks promising enough or finishes within time.

## 2026-09-28 checks

- `d32_hmark_rep`, seed 0, dev, stopped after epoch 2 because it was clearly behind:
  - epoch 1 ES RMSE: `85.08`
  - epoch 2 ES RMSE: `83.66`
  - status: rejected for now; the extra capacity costs much more time and did not catch the small model early.
- Existing completed dev ensembles, floored per seed before averaging:
  - `abl_sm5_hmark_anchor_rep`, seeds 0+1+2: full RMSE `69.34`, edge 16-origin RMSE `81.97`.
  - `abl_sm5_hmark_anchor_rep`, seeds 0+2: full RMSE `69.65`, edge 16-origin RMSE `83.10`.
  - saved fallback `abl_sm5_hmark_rep`, seeds 0+2: full RMSE `69.94`, edge 16-origin RMSE `79.32`.
- Decision: train a final `abl_sm5_hmark_anchor_rep` candidate because its full dev RMSE is the best so far. Keep the saved fallback unchanged until final candidate files are generated and compared.
- Final anchor runs completed and saved:
  - seed 0: `P=11297`, `E=6`, best ES RMSE `68.32`.
  - seed 1: `P=11297`, `E=3`, best ES RMSE `79.05`.
  - seed 2: `P=11297`, `E=6`, best ES RMSE `67.97`.
- Saved candidate snapshots:
  - `submission_snapshots/anchor_s0_s2_2026-09-28/`: `P=22594`, `E=12`.
  - `submission_snapshots/anchor_s0_s1_s2_2026-09-28/`: `P=33891`, `E=15`.
- Active `submission/` folder now contains `anchor_s0_s1_s2_2026-09-28`, because the matching dev ensemble had the best full-holdout RMSE (`69.34`). The older non-anchor fallback remains saved in `submission_snapshots/ensemble_s0_s2_2026-09-28/`.

## 2026-09-29: first submission and validation audit

- The user submitted the active three-seed ensemble. The supplied screenshot reports
  RMSE **81.5648**, MAE **60.9561**, sMAPE **57.81%**, score **81.5724**, and rank **4**.
  This is attempt **1/5**, with P=33891 and E=15. The leader in that screenshot has
  RMSE 69.1586; catching it would require about a 15.2% RMSE reduction.
- Exact submission files, run records, and weights are preserved in
  `submission_snapshots/attempt1_2026-09-29/`. Its result JSON records the prediction hash.
- Recomputed the forecasts from all three saved weight files: scaling, anchoring,
  ordering, and the 168 exported predictions reproduce the submitted ensemble.
- The 69.34 development RMSE was pooled over 51 historical weeks. Individual weekly
  RMSE ranged from 31.08 to 192.88, with median 50.43; 9/51 were above 81.5648.
  The edge-16 result was 81.97. None of these numbers establishes a hidden-test rank.
  The previous expectation of first place was too confident.
- The submitted models fit only the first 35064 observations and use the remainder
  for early stopping. That is a valid validation boundary, but leaves recent history
  unused in weight fitting. Test a fixed-epoch refit before changing architectures.
- Historical protocol: take each original dev run's selected epoch, reset its weights,
  fit on [0,35064), then evaluate the same 51 later weeks without early stopping on
  them. Compare anchored and unanchored configurations, each with seeds 0,1,2.
  Report the full pooled score, edge-16 score, per-seed results, and paired weekly
  differences. These weeks have been used for development before, so this is a
  retrospective comparison, not a fresh independent test.
- Deployment candidates use the same source-selected epoch counts and all 43656
  observed targets. Count source selection epochs plus refit epochs. Save both
  candidates independently; do not use hidden labels or spend another leaderboard
  attempt during these experiments.
- Reproduction: `python -m pa1q2.refit <source-run.json> --fit-end 35064`
  (historical) or `--fit-end 43656` (deployment). Historical comparison outputs go to
  `runs/refit_dev`, deployment outputs to `runs/refit_final` using `--out`.
  `python -m pa1q2.compare_refits` writes `results/refit_comparison.json`.

### Completed refit results

All historical rows use the same 51 weeks. Each refit configuration uses three
seeds; mixed rows average all three original anchored models and all three refits.

| Configuration | Historical RMSE | Edge-16 RMSE | Deployment P | Deployment E |
| --- | ---: | ---: | ---: | ---: |
| First submission's configuration | 69.344 | 81.972 | 33891 | 15 |
| Anchored refit | 69.679 | 82.270 | 33891 | 16 |
| Unanchored refit | 69.609 | 81.526 | 33891 | 14 |
| Original + anchored refit | 69.021 | 81.613 | 67782 | 31 |
| Original + unanchored refit | 68.971 | 81.236 | 67782 | 29 |

- Neither three-model refit beats the original configuration's historical pooled
  RMSE. Refitting is now supported and tested, but is not a demonstrated fix here.
- The mixed unanchored candidate has the lowest new historical RMSE, improving
  28/51 weeks. Its paired-week bootstrap interval for the pooled RMSE difference
  is [-1.614, +0.787], which includes no improvement. This is a descriptive
  resampling of historical weeks, not a prediction interval for the hidden week.
- Saved full-history forecasts, weights, declarations, and epoch-selection records:
  `refit_anchor_s0_s1_s2_2026-09-29/`, `refit_nonanchor_s0_s1_s2_2026-09-29/`,
  `mixed_anchor_2026-09-29/`, and `mixed_nonanchor_2026-09-29/`, all under
  `submission_snapshots/`. Historical predictions and curves remain in `runs/refit_dev/`.
- Output-floor check: using the training 5th or 10th percentile instead of the 1st
  changed the old ensembles' pooled RMSE by less than 0.06 and did not consistently
  improve edge-16 RMSE. Keep the original 1st-percentile rule. It gives floor 7 for
  the old fit region and 6 when fitted on the complete observed history.
- Decision: keep `submission/` identical to attempt 1. The best new result is too
  small and uncertain to justify recommending another scarce leaderboard attempt.
  Do not claim a future rank. No second submission was made.

## 2026-09-29: winter-slice audit and one-change variants

- Audit: data, windowing, scaling and the ensemble export are unchanged since the refit
  audit and still correct. The active `submission/` is byte-identical to attempt 1.
- Why 69 locally vs 81.56 on the leaderboard: the edge-16 weeks are the winter-like
  first 8 and last 8 holdout weeks, the same season as the hidden week. The attempt-1
  config scored 81.97 there, almost exactly the leaderboard value. The gap is seasonal,
  not a bug. Winter weekly RMSE ranged 34 to 143 (mean 76, sd 31).
- Main error: under-predicted peaks. Winter top-decile truth averages 392, forecasts 223.
  Forecast spread is 0.62 of the truth in winter (0.72 over the year). A linear
  rescaling fitted on 8 winter weeks and tested on the other 8 made RMSE worse both ways.
  Rejected.
- Baselines on edge-16: direct ridge with covariates 91.7, climatology 123.6. The
  Autoformer beats both on the relevant slice.
- New variants (3 seeds each, `runs/dev_v2`, snapshot
  `submission_snapshots/dev_variants_2026-09-29/`): kernel 49, trailing means 6/24/72,
  dropout 0, early stopping on winter weeks (new `--es-season winter` flag in train.py).
  Best pooled RMSE: dropout 0 (68.15), but its edge-16 is 81.44. Every variant's paired
  winter change vs the attempt-1 config is between -1.4 and +1.1, sd 4 to 9.
- Decision: no second submission. No candidate has a clear reason to do better on a
  winter week, and the handout scores the last submission, so a new one could replace
  81.56 with a worse number.

## 2026-09-29 (later): two-fold winter protocol and the 27-model ensemble

- Added a second dev fold (`--fold A`: fit to 17520, early stop to 26304, score the next
  cycle's 52 weeks) so winter weeks from two different years can be compared. Added
  `--winter-weight` (season-weighted loss) and `--sqrt-target` to `train.py`, and
  `compare_folds.py` for the paired two-fold comparison.
- Pre-set bar for a new submission: at least 3 RMSE better on the 33 winter weeks of both
  folds, better on at least 20, and better in each fold separately.
- Results (3 seeds, floored averages, vs attempt-1 config): winter-weighted loss +0.0 mean
  weekly change (17/33 better); sqrt target -0.4 (17/33). Neither passed. Full numbers in
  `results/folds_v3.json`.
- Averaging all 27 dev models across the 9 configurations gave winter RMSE 79.38 vs 81.97
  (12/16 weeks, mean weekly change -2.37, sd 4.42) and 68.12 vs 69.34 over all 51 weeks.
  Only a 3-config subset could be checked on fold A (flat), so this did not pass the bar
  either, but it was the best-supported option, and the user chose to build it.
- Built the matching 27-model final ensemble: `submission_snapshots/ensemble27_2026-09-29/`
  (P = 345435, E = 122). Not submitted. `submission/` is still attempt 1.

## 2026-09-29 (evening): stagnation / clear-out features (negative result)

- Motivation: the leaderboard gap to rank 1 is in peak height (rank 1 has RMSE 12 lower
  than the 81 cluster but MAE only 4.5 lower). Pollution accumulates over calm spells and
  clears with a strong wind from one direction (feature_H has the highest wind speeds and
  lowest target levels, feature_J the calmest and highest). Added `--accum 1`: hours since
  the last strong clearing-wind hour, calm share of the trailing 48 and 120 hours, and
  clearing-wind share of the trailing 72 hours, all from the covariate file only.
- Ridge check: the features correlate about 0.4 with log target and improved the ridge's
  winter RMSE (92.1 to 89.9) but not its full-year RMSE.
- Autoformer, attempt-1 config plus the features, 3 seeds, both folds: fold A all 79.13 vs
  77.43, winter 107.18 vs 106.13 (7/17 weeks better); fold B all 71.88 vs 69.34, winter
  81.34 vs 81.97 (7/16). Mean winter weekly change +2.1. Worse. Rejected.
  Runs in `runs/dev_v5/`, numbers in `results/folds_v5.json`.
- End state: `submission/` is attempt 1. The only saved candidate is the 27-model ensemble.

## 2026-09-29 (night): leaderboard read-out and the two-scale ensemble

- Leaderboard: ranks 2 to 5 sit at RMSE 81 to 83 with 6.6k to 446k parameters and 6 to
  105 epochs, so size and training length do not separate entries. Rank 1 (7,938 params,
  5 epochs) and rank 2 both have sMAPE about 50 (ours 57.8), which points to training on
  a relative (log-like) scale. Rank 1 additionally has RMSE 69, so it also got the peaks.
- Our log-target seeds have rank 1's MAE and sMAPE profile (48 to 52, 53 to 55) but worse
  RMSE. Raw and log models make different mistakes, so a two-scale ensemble was tested:
  raw attempt-1 config averaged with 3 log-target models (log outputs capped at the fit
  region's 99.9th percentile, since one seed predicted 1395 against an observed max of 671).
- Fold B (last cycle): 50/50 blend winter 78.80 vs 81.97 (mean weekly change -3.2, 8/16
  better), all weeks 67.45 vs 69.34, MAE 44.7, sMAPE 51.0.
- Fold A (earlier cycle, 3 new log runs in `runs/dev_v6`): log models much worse in winter
  (116 vs 106), 50/50 blend +2.4 (8/17), 30% log +0.7. Over both folds: mean -0.3 to -0.9,
  16/33 weeks better. Does not pass the bar. MAE and sMAPE improve in both folds, RMSE
  does not, so relative-scale training helps the metrics that are not ranked.
- Final log models (`runs/final_v6`, 3 seeds) were still trained so the blend can be
  assembled if wanted. Not submitted. `submission/` remains attempt 1.

## 2026-09-29 (late): three more directions, one passes

Three seeds, both folds, same bar, all on the attempt-1 configuration:
- Window-relative normalisation (`--window-norm 1`): worse in both folds (fold B all 77.4,
  winter 83.0; fold A 80.2 / 108.0). Rejected.
- d_model 8, d_ff 16 (4369 params): fold B 71.4 / 80.2, fold A 79.5 / 107.8. Rejected.
- Nonlinear covariate encoder (`--mark-mlp 16`, a per-step 2-layer MLP on the known
  inputs added to the linear embedding): fold B 64.79 / 76.80, fold A 74.36 / 102.48.
  Winter paired change -3.19, better 20/33, better in both folds. **Passes the bar.**
  Also lifts peak predictions (winter top decile 223 -> 261 against 392) and gives
  MAE 45 / sMAPE 53. Runs in `runs/dev_v7/`, numbers in `results/folds_v7.json`.
- Final 3-seed ensemble trained on all history with the same early-stopping rule:
  `submission_snapshots/covmlp_s0_s1_s2_2026-09-29/`, P = 38979, E = 16. Not submitted
  by me. `submission/` remains attempt 1.

## 2026-09-29, 22:18: attempt 2 result

- Submitted the covariate-encoder ensemble (P 38979, E 16). Leaderboard: RMSE 81.9270,
  MAE 62.7527, sMAPE 59.58%, score 81.9351. Worse than attempt 1 by 0.36 RMSE, and worse
  on MAE and sMAPE as well. The leaderboard keeps the best attempt, so attempt 1 (81.5648)
  is still the scored entry. Attempts used: 2 of 5.
- Reading: the change improved two full years of historical winter weeks (mean -3.2 per
  week, 20 of 33 weeks better, both years) but not the scored week. Two well-validated
  models now sit at 81.6 and 81.9 on this week, alongside three other teams at 81 to 83.
  The scored week differs from the history in a way this validation does not capture.
- Decision: no further submissions. Remaining attempts are kept in reserve.

## 2026-09-29, 23:00 to 00:10: ceiling test and the direct covariate head

- Ceiling test: HistGradientBoosting on per-hour horizon covariates (plus trailing means,
  accumulation features, phase, origin-level history summaries and steps-ahead) scores
  fold B all 56.97 / winter 70.48, fold A 67.11 / 83.57. A plain per-hour MLP on the same
  marks the Autoformer sees scores fold B 58.3 / 76.5. The Autoformer (69.3 / 82.0) is
  losing most of the covariate signal to its own smoothing.
- `--cov-head H`: per-hour MLP from the decoder marks straight to the output, added to the
  Autoformer forecast. First try (head 32, anchored, lr 3e-4 halving): fold B seed 1 holdout
  75.9, little gain, because the head must predict a change from an unseen anchor and the
  learning rate decays before it trains. Second try (head 64, no anchor, lr 1e-3 with 0.8
  decay, up to 10 epochs, patience 3): fold B all 61.68 / winter 74.97, fold A 66.40 / 91.63.
  Winter paired change -8.32, better 24/33, both folds. Passes the bar by a wide margin.
  `runs/dev_v9/`, `results/folds_v9.json`. Added `--lr-decay` to train.py.
- Finals: 7 seeds trained on [0, 35064). Early-stop RMSE bimodal: seeds 2, 3, 4, 5 at
  60 to 66, seeds 0, 1, 6 at 81 to 83 with the same training loss. Members selected on the
  early-stopping set: seeds 2, 3, 4 -> `submission_snapshots/covhead_s2_s3_s4_2026-09-29/`,
  P = 53670, E = 20 (discarded runs: 17 more epochs, documented). Seeds 5 and 6 finished on
  the laptop after the link dropped and are not in the snapshot.
- Not submitted by me. `submission/` remains attempt 1.

## 2026-09-30, 12:58: attempt 3 result

- Submitted the covariate-head ensemble (seeds 2, 3, 4; P 53670, E 20). Leaderboard:
  RMSE 62.9497, MAE 46.3068, sMAPE 49.63%, score 62.9599, rank 1 (previous leader 69.16).
  Tagged Best. Attempts used: 3 of 5. `submission/` now holds attempt 3.
- Reading: the ceiling test was right. The covariate file carries most of the signal for
  this series, and the Autoformer's smoothing paths were suppressing it. Giving the
  horizon covariates a direct per-hour path to the output cut RMSE by 23% against
  attempt 1 on the scored week, in line with the 8.3-point winter gain seen on two
  historical years.
- Decision: no further submissions.
