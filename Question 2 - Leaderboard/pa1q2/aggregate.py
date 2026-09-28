"""Collect dev-stage runs into seed-aggregated tables.

    python -m pa1q2.aggregate RUN_DIR [RUN_DIR ...] --out results

Writes results/ablation_runs.csv (one row per run), results/ablation_summary.csv
(mean and std over seeds per configuration, all 51 holdout blocks and the 17 winter blocks),
and results/paired_blocks.csv (per-block paired comparisons against the chosen config).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

WINTER_DAYS = 59, 304  # holdout year 2014 starts at ES_END: Jan-Feb and Nov-Dec count as winter


def block_scores(pred, true):
    err = pred - true
    rmse = np.sqrt((err ** 2).mean(1))
    return rmse


def winter_mask(origins):
    day = (np.asarray(origins) + D.PRED_LEN // 2 - D.ES_END) // 24
    return (day < WINTER_DAYS[0]) | (day >= WINTER_DAYS[1])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", default="results")
    p.add_argument("--reference", default="future-abl")
    a = p.parse_args(argv)
    rows, blocks = [], {}
    for d in a.dirs:
        for f in sorted(Path(d).glob("*.json")):
            r = json.loads(f.read_text())
            if "holdout" not in r:
                continue
            args = r["args"]
            config = f"{args['cov']}-{args['tag'] or 'base'}"
            z = np.load(f.with_suffix(".npz"))
            pred, true, origins = z["pred"], z["true"], z["origins"]
            w = winter_mask(origins)
            win = D.metrics(pred[w], true[w])
            rows.append({"config": config, "cov": args["cov"], "seed": args["seed"],
                         "log_target": args["log_target"], "rolling": args["rolling"],
                         "d_model": args["d_model"], "mark_kernel": args.get("mark_kernel", 1),
                         "parameters": r["parameters"], "epochs_run": r["epochs_run"],
                         "best_epoch": r["best_epoch"], "es_RMSE": r["best_es_RMSE"],
                         **{f"holdout_{k}": v for k, v in r["holdout"].items()},
                         **{f"winter_{k}": v for k, v in win.items()}})
            blocks[(config, args["seed"])] = block_scores(pred, true)
    runs = pd.DataFrame(rows).sort_values(["config", "seed"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "ablation_runs.csv", index=False)
    metrics = ["es_RMSE", "holdout_RMSE", "holdout_MAE", "holdout_sMAPE", "holdout_mean_block_RMSE",
               "winter_RMSE", "winter_MAE", "winter_sMAPE", "epochs_run", "best_epoch"]
    g = runs.groupby("config")
    summary = g[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.insert(0, "seeds", g.size())
    summary.insert(1, "parameters", g["parameters"].first())
    summary = summary.sort_values("holdout_RMSE_mean")
    summary.to_csv(out / "ablation_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summary[["seeds", "parameters", "es_RMSE_mean", "es_RMSE_std", "holdout_RMSE_mean",
                       "holdout_RMSE_std", "holdout_MAE_mean", "holdout_sMAPE_mean",
                       "winter_RMSE_mean", "winter_RMSE_std", "epochs_run_mean"]].round(2))

    # paired per-block comparison: seed-averaged block RMSE of each config vs the reference
    def seed_avg(config):
        arr = [v for (c, s), v in blocks.items() if c == config]
        return np.mean(arr, 0) if arr else None
    ref = seed_avg(a.reference)
    paired = []
    if ref is not None:
        for config in runs["config"].unique():
            other = seed_avg(config)
            diff = other - ref
            paired.append({"config": config, "reference": a.reference,
                           "mean_block_RMSE_diff": float(diff.mean()),
                           "blocks_worse_than_reference": int((diff > 0).sum()),
                           "blocks": int(len(diff)),
                           "sign_test_p": float(_sign_test(diff))})
        paired = pd.DataFrame(paired).sort_values("mean_block_RMSE_diff")
        paired.to_csv(out / "paired_blocks.csv", index=False)
        print(paired.round(3).to_string(index=False))


def _sign_test(diff):
    """Two-sided exact sign test on non-zero paired differences."""
    from math import comb
    d = diff[diff != 0]
    n, k = len(d), int((d > 0).sum())
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


if __name__ == "__main__":
    main()
