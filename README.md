# DL4STG PA1

Programming Assignment 1 for AI651 Deep Learning for Space, Time and Graphs (LUMS, Fall 2026).
Ahmed Jawed, roll number 25280040.

Report: `report/main.pdf` (LaTeX source in `report/`, numbered figures in `report/figures/`).

## Contents

- `Question 1/Assignment1.ipynb`: executed `PA1_PRESET=full` notebook for Task 1.
- `Question 1/harness/`, `Question 1/requirements.txt`: supplied assignment harness.
- `Question 1/results/design/`: exported Task 1 tables and figures (Outputs 1.1 to 4.3).
- `Question 2 - Leaderboard/pa1q2/`: from-scratch Autoformer pipeline for Task 2
  (`autoformer.py`, `train.py`, `data.py`, `submission.py`, baselines, refit and comparison scripts).
- `Question 2 - Leaderboard/results/`: validation, ablation, fold comparisons and figure outputs.
- `Question 2 - Leaderboard/submission/`: the scored forecast (attempt 3) and its P/E declaration.
- `Question 2 - Leaderboard/submission_snapshots/`: every candidate that was built, including
  the three leaderboard attempts, with run records and leaderboard results.
- `Question 2 - Leaderboard/jobs_*.txt`, `run_queue.sh`: the exact argument lists of every run.
- `Question 2 - Leaderboard/tests/`: unit tests for the chronological boundaries and the
  submission writer.
- `report/`: LaTeX source, figures and the compiled report PDF.

Trained weights (`*.pt`) are not committed. Every run's arguments, best epoch, early-stopping
RMSE and forecast are in the matching `.json` record.

## Final submission (attempt 3)

Autoformer (`d_model` 16, `d_ff` 32, one encoder and one decoder layer, series decomposition
kernel 25 with replicate padding, Auto-Correlation in all three mixers, width-5 covariate
embedding, origin-relative horizon marks) plus a direct per-hour covariate head
(`--cov-head 64`). Three seeds averaged after flooring at the 1st percentile of the fit data.

Declared on the leaderboard: `P = 53670` (3 x 17890 trainable parameters) and `E = 20`
(7 + 7 + 6 epochs run by the three members). Leaderboard: RMSE 62.95, rank 1 at submission.

Reproduce from `Question 2 - Leaderboard/`:

```bash
pip install "torch>=2.4" numpy pandas matplotlib
for s in 2 3 4; do
python -m pa1q2.train --cov future --d-model 16 --d-ff 32 --mark-kernel 5 \
  --horizon-mark 1 --pad replicate --cov-head 64 --lr 1e-3 --lr-decay 0.8 \
  --max-epochs 10 --patience 3 --stage final --out runs/final_v9 \
  --seed $s --tag fin_head64
done
python -m pa1q2.submission runs/final_v9/future-log0-ph1-L168-d16-s2-fin_head64.json \
  runs/final_v9/future-log0-ph1-L168-d16-s3-fin_head64.json \
  runs/final_v9/future-log0-ph1-L168-d16-s4-fin_head64.json
```

`submission.py` rebuilds each member from its saved arguments, loads its weights, asserts
that the trainable parameter count matches the record, floors and averages the forecasts,
and writes `submission/predictions.txt` (168 values, `time_idx` 43657 first) and
`submission/declaration.json` (P and E summed over members).

Seven seeds (0 to 6) were trained for the final stage. Seeds 2, 3 and 4 were chosen on the
early-stopping set only. See the report, Section 2.7, and
`submission_snapshots/covhead_s2_s3_s4_2026-09-29/README.md`.

Attempt 1 (`P = 33891`, `E = 15`, RMSE 81.56) is preserved in
`submission_snapshots/attempt1_2026-09-29/` and attempt 2 in
`submission_snapshots/covmlp_s0_s1_s2_2026-09-29/`.

## Validation and ablation

Chronological split: fit on `[0, 26304)`, early stop on `[26304, 35064)`, compare designs on
51 non-overlapping 168-step blocks of the last year (fold B), plus a second fold one cycle
earlier (fold A). Every configuration uses 3 seeds. The required optional-data ablation
(none, past only, past plus horizon) is in `results/ablation_summary.csv` and the report.

```bash
python -m unittest discover -s tests   # from Question 2 - Leaderboard/
```
