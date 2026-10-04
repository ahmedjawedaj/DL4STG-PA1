# Question 2: Autoformer leaderboard pipeline

From-scratch Autoformer (series decomposition + Auto-Correlation) for the 168-step forecast.
Adapted from the structure of Wu et al. (2021) and github.com/thuml/Autoformer, with the
delay scoring / aggregation convention from Question 1. See the report, Task 2.

## Layout
- `Data/` given files (unchanged)
- `pa1q2/data.py` loading, features, chronological splits, metrics
- `pa1q2/autoformer.py` the model
- `pa1q2/train.py` one configuration and seed, resumable (`--stage dev` or `--stage final`)
- `pa1q2/slice.py` runs a job list in time-limited slices (resumes between epochs)
- `pa1q2/baselines.py` climatology, persistence, direct ridge references
- `pa1q2/aggregate.py` seed-aggregated tables and paired per-block sign tests
- `pa1q2/figures.py` report figures
- `pa1q2/submission.py` writes the leaderboard string and the P / E declaration
- `pa1q2/refit.py` refits for an earlier validation-selected epoch count
- `pa1q2/compare_refits.py` compares three-seed refits on historical weeks
- `pa1q2/compare_variants.py` compares three-seed dev variants on all weeks and winter weeks
- `pa1q2/compare_folds.py` two-fold (two winters) paired comparison against the attempt-1 config
- `jobs_*.txt` the exact argument lists that were run
- `runs/dev/` every dev-stage run (JSON record + holdout predictions), `runs/explore/`
  early single-seed exploration (older code, not used in the tables)
- `runs/final/` the final model records and weights used for the ensemble
- `results/` tables and figures, `submission/` the pasted values and the declaration

## Reproduce
```bash
pip install "torch>=2.4" numpy pandas matplotlib
# dev ablation (3 seeds per configuration), e.g. the submitted configuration
python -m pa1q2.train --cov future --seed 0 --d-model 16 --d-ff 32 --mark-kernel 5 \
    --horizon-mark 1 --anchor 1 --pad replicate --tag abl_sm5_hmark_anchor_rep
python -m pa1q2.baselines
python -m pa1q2.aggregate runs/dev --reference future-abl_sm5_hmark_anchor_rep
python -m pa1q2.figures runs/dev
# final models used for the submission ensemble
python -m pa1q2.train --cov future --seed 0 --d-model 16 --d-ff 32 --mark-kernel 5 \
    --horizon-mark 1 --anchor 1 --pad replicate --stage final --out runs/final --tag final_anchor_rep
python -m pa1q2.train --cov future --seed 1 --d-model 16 --d-ff 32 --mark-kernel 5 \
    --horizon-mark 1 --anchor 1 --pad replicate --stage final --out runs/final --tag final_anchor_rep
python -m pa1q2.train --cov future --seed 2 --d-model 16 --d-ff 32 --mark-kernel 5 \
    --horizon-mark 1 --anchor 1 --pad replicate --stage final --out runs/final --tag final_anchor_rep
python -m pa1q2.submission runs/final/future-log0-ph1-L168-d16-s0-final_anchor_rep.json \
    runs/final/future-log0-ph1-L168-d16-s1-final_anchor_rep.json \
    runs/final/future-log0-ph1-L168-d16-s2-final_anchor_rep.json
```

## Submission
- Values: `submission/predictions.txt` (168 values, time_idx 43657 first, 43824 last)
- Trainable parameters P = 53670, epochs E = 20 (see `submission/declaration.json`)
- Output floor: values below the 1st percentile of the fit-region target (7.0) are raised
  to it per model before averaging.
- The submitted file averages the floored seed-0, seed-1 and seed-2 anchored final
  forecasts. On the dev holdout, the same three-seed average scored RMSE 69.34. The
  previous two-seed non-anchor fallback is saved in
  `submission_snapshots/ensemble_s0_s2_2026-09-28/`.

## Splits (0-based positions)
fit [0, 26304), early stop [26304, 35064), holdout [35064, 43656) = 51 blocks of 168.
Final model: fit [0, 35064), early stop on the 51 holdout blocks.

## Actual leaderboard result

Attempt 1 (29 September): RMSE **81.5648**, MAE **60.9561**, sMAPE **57.81%**,
score **81.5724**, rank **4** in the supplied screenshot. The exact predictions,
declaration, run records, and local weights are saved in
`submission_snapshots/attempt1_2026-09-29/`.

The historical 69.34 RMSE is pooled over 51 weeks using models trained from an
earlier cutoff. It is not the expected score of a particular hidden week or a
guarantee of rank. The edge-16 historical RMSE was 81.97.

## Refit checks

`refit.py` starts a fresh model using a completed run's settings and selected epoch
count. With `--fit-end 35064 --eval-end 43656 --out runs/refit_dev`, it evaluates on
later observed weeks without using them for early stopping. With `--fit-end 43656
--out runs/refit_final`, it forecasts from all available history. Both the source
selection epochs and the refit epochs count toward E.

Use `python -m pa1q2.submission <run.json> ... --out <candidate-directory>` to keep
new candidates separate from the submitted values. Each refit preserves its JSON
settings, training curve, model weights, and historical predictions or future
forecast. See `experiment_notes/variants.md` for results and decisions.

Checks: `python -m unittest discover -s tests -v`.

Completed checks: neither three-seed refit beat 69.34 historical RMSE. A six-model
average scored 68.97, but its small gain was uncertain across weeks. All four new
candidate bundles are preserved in `submission_snapshots/`; `submission/` remains
the exact first attempt. No further submission is recommended from these results.

## Winter-slice check (29 September)

The winter-like edge-16 weeks predicted the leaderboard well (81.97 locally vs 81.56).
Four one-change variants (kernel 49, 72-step trailing means, dropout 0, winter early
stopping) did not improve winter weeks beyond noise. See
`submission_snapshots/dev_variants_2026-09-29/README.md`. `submission/` is still attempt 1.

## 27-model ensemble candidate (29 September)

A second dev fold, two targeted variants (winter-weighted loss, square-root target) and a
27-model mixed ensemble were tested. Only the ensemble improved the winter weeks (79.4 vs
82.0, 12/16 weeks). Its full-history version is saved in
`submission_snapshots/ensemble27_2026-09-29/` with P = 345435 and E = 122. It has not been
submitted. See that folder's README and `experiment_notes/variants.md`.

## Covariate-encoder candidate (29 September, late)

A per-step MLP on the known-horizon covariates (`--mark-mlp 16`) was the only one of 19
variants to pass the pre-set two-winter bar (winter RMSE -3.2 per week on average, better
on 20 of 33 winter weeks, better in both years, and better on all-week RMSE in both years).
Its 3-seed full-history ensemble was submitted as attempt 2
(`submission_snapshots/covmlp_s0_s1_s2_2026-09-29/`, P = 38979, E = 16) and scored RMSE 81.9270,
worse than attempt 1 by 0.36. Attempt 1 remains the leaderboard entry. See that README. Other options added to `train.py` for the tests:
`--fold A`, `--winter-weight`, `--sqrt-target`, `--accum`, `--window-norm`, `--mark-mlp`.

## Attempt 3 (30 September): rank 1

`submission/` now holds attempt 3: the Autoformer with a direct covariate head, seeds 2, 3, 4
(`submission_snapshots/covhead_s2_s3_s4_2026-09-29/`). Leaderboard RMSE 62.9497, MAE 46.3068,
sMAPE 49.63%, score 62.9599, rank 1. Attempt 1 (81.5648) is kept in
`submission_snapshots/attempt1_2026-09-29/`. See `experiment_notes/variants.md` for the path.
