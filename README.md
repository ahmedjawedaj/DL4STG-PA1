# DL4STG PA1

Programming Assignment 1 for AI651 / CS434 / EE5108.

## Contents

- `Question 1/Assignment1.ipynb`: executed full-preset notebook for Task 1.
- `Question 1/harness/`: supplied assignment harness.
- `Question 1/results/design/`: exported Task 1 tables and figures.
- `Question 2 - Leaderboard/pa1q2/`: from-scratch Autoformer pipeline for Task 2.
- `Question 2 - Leaderboard/results/`: validation, ablation, and figure outputs.
- `Question 2 - Leaderboard/submission/`: final 168 predictions and parameter/epoch declaration.
- `report/`: LaTeX source, figures, and compiled report PDF.

## Reproduce Task 2 Final Runs

From `Question 2 - Leaderboard/`:

```bash
pip install "torch>=2.4" numpy pandas matplotlib
python -m pa1q2.train --cov future --seed 0 --d-model 16 --d-ff 32 \
  --mark-kernel 5 --horizon-mark 1 --anchor 1 --pad replicate \
  --stage final --out runs/final --tag final_anchor_rep
python -m pa1q2.train --cov future --seed 1 --d-model 16 --d-ff 32 \
  --mark-kernel 5 --horizon-mark 1 --anchor 1 --pad replicate \
  --stage final --out runs/final --tag final_anchor_rep
python -m pa1q2.train --cov future --seed 2 --d-model 16 --d-ff 32 \
  --mark-kernel 5 --horizon-mark 1 --anchor 1 --pad replicate \
  --stage final --out runs/final --tag final_anchor_rep
python -m pa1q2.submission runs/final/future-log0-ph1-L168-d16-s0-final_anchor_rep.json \
  runs/final/future-log0-ph1-L168-d16-s1-final_anchor_rep.json \
  runs/final/future-log0-ph1-L168-d16-s2-final_anchor_rep.json
```

The leaderboard forecast in `Question 2 - Leaderboard/submission/predictions.txt` is the
average of the three per-model floored forecasts. Declaration: `P = 33891`, `E = 15`.
