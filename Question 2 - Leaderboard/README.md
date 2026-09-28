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
    --horizon-mark 1 --pad replicate --tag abl_sm5_hmark_rep
python -m pa1q2.baselines
python -m pa1q2.aggregate runs/dev --reference future-abl_sm5_hmark_rep
python -m pa1q2.figures runs/dev
# final models used for the submission ensemble
python -m pa1q2.train --cov future --seed 0 --d-model 16 --d-ff 32 --mark-kernel 5 \
    --horizon-mark 1 --pad replicate --stage final --out runs/final --tag final
python -m pa1q2.train --cov future --seed 2 --d-model 16 --d-ff 32 --mark-kernel 5 \
    --horizon-mark 1 --pad replicate --stage final --out runs/final --tag final_s2
python -m pa1q2.submission runs/final/future-log0-ph1-L168-d16-s0-final.json \
    runs/final/future-log0-ph1-L168-d16-s2-final_s2.json
```

## Submission
- Values: `submission/predictions.txt` (168 values, time_idx 43657 first, 43824 last)
- Trainable parameters P = 22594, epochs E = 10 (see `submission/declaration.json`)
- Output floor: values below the 1st percentile of the fit-region target (7.0) are raised
  to it per model before averaging. Dev check, selected config, 3 seeds, holdout RMSE without / with floor:
  73.31 / 73.17, 75.51 / 75.38, 70.72 / 70.57.
- The submitted file averages the floored seed-0 and seed-2 final forecasts. On the dev
  holdout, the same two-seed average scored RMSE 69.94.

## Splits (0-based positions)
fit [0, 26304), early stop [26304, 35064), holdout [35064, 43656) = 51 blocks of 168.
Final model: fit [0, 35064), early stop on the 51 holdout blocks.
