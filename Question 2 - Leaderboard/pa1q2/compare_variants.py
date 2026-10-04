"""Compare three-seed dev variants on all 51 holdout weeks and on the 16 winter-like weeks.

    python -m pa1q2.compare_variants --out results/variants_v2.json

Each configuration is scored as the average of its per-seed forecasts, each floored at 7
(the fit-region 1st percentile), exactly as the submission averages seeds. "edge16" is the
first 8 and last 8 holdout weeks, the winter-like part of the final year, which is the same
season as the hidden week. Paired differences are against the first submission's config.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import data as D

REFERENCE = ("abl_sm5_hmark_anchor_rep", "runs/dev")
CONFIGS = [("abl_sm5_hmark_rep", "runs/dev"), REFERENCE,
           ("w_k49", "runs/dev_v2"), ("w_roll72", "runs/dev_v2"),
           ("w_do0", "runs/dev_v2"), ("w_eswinter", "runs/dev_v2")]
EDGE = np.r_[0:8, 43:51]
FLOOR = 7.0


def load(tag, folder):
    files = sorted(Path(folder).glob(f"*-{tag}.npz"))
    preds = [np.maximum(np.load(f)["pred"], FLOOR) for f in files]
    records = [json.loads(f.with_suffix(".json").read_text()) for f in files]
    return preds, records


def rmse(pred, true):
    return float(np.sqrt(((pred - true) ** 2).mean()))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/variants_v2.json")
    a = p.parse_args(argv)
    y, _ = D.load()
    origins = D.holdout_origins(D.PRED_LEN)
    true = np.stack([y[o:o + D.PRED_LEN] for o in origins])
    ref_preds, _ = load(*REFERENCE)
    ref_weeks = np.sqrt(((np.mean(ref_preds, 0) - true) ** 2).mean(1))
    rows = []
    for tag, folder in CONFIGS:
        preds, records = load(tag, folder)
        if len(preds) < 3:
            continue
        ens = np.mean(preds, 0)
        weeks = np.sqrt(((ens - true) ** 2).mean(1))
        diff = weeks[EDGE] - ref_weeks[EDGE]
        rows.append({
            "config": tag, "seeds": len(preds), "parameters": records[0]["parameters"],
            "mean_epochs": float(np.mean([r["epochs_run"] for r in records])),
            "mean_es_RMSE": float(np.mean([r["best_es_RMSE"] for r in records])),
            "ensemble_RMSE": rmse(ens, true), "ensemble_edge16_RMSE": rmse(ens[EDGE], true[EDGE]),
            "ensemble_last8_RMSE": rmse(ens[43:], true[43:]),
            "seed_RMSE": [rmse(q, true) for q in preds],
            "seed_edge16_RMSE": [rmse(q[EDGE], true[EDGE]) for q in preds],
            "winter_paired_diff_mean": float(diff.mean()),
            "winter_paired_diff_sd": float(diff.std()),
            "winter_weeks_better": int((diff < 0).sum()),
            "args": records[0]["args"]})
        r = rows[-1]
        print(f"{tag:26s} RMSE {r['ensemble_RMSE']:6.2f}  edge16 {r['ensemble_edge16_RMSE']:6.2f}  "
              f"winter diff {r['winter_paired_diff_mean']:+5.2f} (sd {r['winter_paired_diff_sd']:.2f}, "
              f"better {r['winter_weeks_better']}/16)  P {r['parameters']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
