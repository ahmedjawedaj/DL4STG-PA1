# Task 2 Variant Log

This log tracks candidate submissions before using any leaderboard attempt.

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
