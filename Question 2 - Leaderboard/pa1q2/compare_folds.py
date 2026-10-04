"""Two-fold winter comparison of three-seed variants against the first submission's config.

    python -m pa1q2.compare_folds CAND_TAG_PREFIX [...] --out results/folds_v3.json

Fold B: fit to 26304, early stop on the next cycle, score the last cycle (51 weeks).
Fold A: the same one cycle earlier (52 weeks). Winter-like weeks are those whose centre
falls in the first 59 or last 61 days of the assumed yearly cycle (16 + 17 = 33 weeks),
the season of the hidden week. Each configuration is the average of its floored seeds.

Submission bar (fixed before the results were seen): mean winter-week RMSE improvement of
at least 3, better on at least 20 of the 33 winter weeks, and better in both folds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import data as D

FLOOR = 7.0
REFERENCE = {"B": ("runs/dev", "abl_sm5_hmark_anchor_rep"), "A": ("runs/dev_v3", "v3_ref_A")}


def fold_truth(fold):
    _, _, start, end = D.fold_ranges(fold)
    origins = D.test_blocks(start, end)
    y, _ = D.load()
    return origins, np.stack([y[o:o + D.PRED_LEN] for o in origins])


def ensemble(folder, tag):
    files = sorted(Path(folder).glob(f"*-{tag}.npz"))
    if len(files) < 3:
        return None, []
    preds = [np.maximum(np.load(f)["pred"], FLOOR) for f in files]
    return np.mean(preds, 0), [json.loads(f.with_suffix(".json").read_text()) for f in files]


def week_rmse(pred, true):
    return np.sqrt(((pred - true) ** 2).mean(1))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("candidates", nargs="+", help="tag prefix, e.g. v3_ww3 (fold suffix added)")
    p.add_argument("--folder", default="runs/dev_v3")
    p.add_argument("--out", default="results/folds_v3.json")
    a = p.parse_args(argv)
    truth = {f: fold_truth(f) for f in "AB"}
    ref = {}
    for f in "AB":
        pred, _ = ensemble(*REFERENCE[f])
        ref[f] = week_rmse(pred, truth[f][1])
    rows = []
    for cand in ["reference"] + a.candidates:
        diffs, row = [], {"config": cand}
        for f in "AB":
            origins, true = truth[f]
            winter = D.is_winter(origins)
            if cand == "reference":
                pred, recs = ensemble(*REFERENCE[f])
            else:
                pred, recs = ensemble(a.folder, f"{cand}_{f}")
            if pred is None:
                break
            weeks = week_rmse(pred, true)
            d = weeks[winter] - ref[f][winter]
            diffs.append(d)
            row[f"fold{f}_all_RMSE"] = float(np.sqrt(((pred - true) ** 2).mean()))
            row[f"fold{f}_winter_RMSE"] = float(np.sqrt(((pred[winter] - true[winter]) ** 2).mean()))
            row[f"fold{f}_winter_mean_week_diff"] = float(d.mean())
            row[f"fold{f}_winter_weeks_better"] = f"{int((d < 0).sum())}/{len(d)}"
            row[f"fold{f}_epochs"] = [r["epochs_run"] for r in recs]
            row["parameters"] = recs[0]["parameters"]
        if len(diffs) < 2:
            print(f"{cand}: incomplete")
            continue
        d = np.concatenate(diffs)
        row.update(winter_mean_week_diff=float(d.mean()), winter_weeks_better=int((d < 0).sum()),
                   winter_weeks=len(d),
                   passes_bar=bool(d.mean() <= -3 and (d < 0).sum() >= 20
                                   and all(x.mean() < 0 for x in diffs)))
        rows.append(row)
        print(f"{cand:14s} A all {row['foldA_all_RMSE']:6.2f} win {row['foldA_winter_RMSE']:6.2f} "
              f"({row['foldA_winter_weeks_better']}) | B all {row['foldB_all_RMSE']:6.2f} "
              f"win {row['foldB_winter_RMSE']:6.2f} ({row['foldB_winter_weeks_better']}) | "
              f"winter week diff {row['winter_mean_week_diff']:+.2f}, better "
              f"{row['winter_weeks_better']}/{row['winter_weeks']}  pass={row['passes_bar']}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
