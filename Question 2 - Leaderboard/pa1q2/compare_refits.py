"""Compare fixed-epoch refits with the same seeds before refitting.

All scores are retrospective development results, not hidden-test estimates.
"""
import json
from pathlib import Path

import numpy as np

from . import data as D


def summarize(pred, true):
    block = np.sqrt(((pred - true) ** 2).mean(1))
    edge = np.r_[0:8, len(pred) - 8:len(pred)]
    return D.metrics(pred, true) | {
        "edge16_RMSE": D.metrics(pred[edge], true[edge])["RMSE"],
        "last8_RMSE": D.metrics(pred[-8:], true[-8:])["RMSE"],
        "block_RMSE_p10_p50_p90": np.percentile(block, [10, 50, 90]).tolist(),
        "per_block_RMSE": block.tolist(),
    }


def main():
    y, _ = D.load()
    rows = {}
    forecasts = {}
    for tag in ("abl_sm5_hmark_anchor_rep", "abl_sm5_hmark_rep"):
        original, refitted, per_seed = [], [], []
        for seed in (0, 1, 2):
            name = f"future-log0-ph1-L168-d16-s{seed}-{tag}"
            before = np.load(Path("runs/dev") / f"{name}.npz")
            path = Path("runs/refit_dev") / f"{name}-refit35064"
            after = np.load(path.with_suffix(".npz"))
            record = json.loads(path.with_suffix(".json").read_text())
            assert np.array_equal(before["origins"], after["origins"])
            assert np.array_equal(before["true"], after["true"])
            original.append(np.maximum(before["pred"], np.percentile(y[:D.FIT_END], 1)))
            refitted.append(np.maximum(after["pred"], record["output_floor"]))
            true = after["true"]
            per_seed.append({"seed": seed, "refit_epochs": record["refit_epochs"],
                             "before": summarize(original[-1], true),
                             "after": summarize(refitted[-1], true)})
        old = np.mean(original, axis=0)
        new = np.mean(refitted, axis=0)
        forecasts[tag] = (old, new)
        before_score, after_score = summarize(old, true), summarize(new, true)
        old_blocks = np.asarray(before_score["per_block_RMSE"])
        new_blocks = np.asarray(after_score["per_block_RMSE"])
        # Paired resampling describes variation across historical weeks. It is
        # not a confidence interval for the RMSE of the hidden week.
        rng = np.random.default_rng(0)
        samples = rng.integers(0, len(old_blocks), size=(10000, len(old_blocks)))
        delta = np.sqrt((new_blocks[samples] ** 2).mean(1)) - \
            np.sqrt((old_blocks[samples] ** 2).mean(1))
        rows[tag] = {"before": before_score, "after": after_score, "seeds": per_seed,
                     "weeks_improved": int((new_blocks < old_blocks).sum()),
                     "total_weeks": len(old_blocks),
                     "paired_week_bootstrap_delta_95pct": np.percentile(delta, [2.5, 97.5]).tolist()}
        print(tag, json.dumps({"before_RMSE": before_score["RMSE"],
                               "after_RMSE": after_score["RMSE"],
                               "before_edge16": before_score["edge16_RMSE"],
                               "after_edge16": after_score["edge16_RMSE"],
                               "weeks_improved": rows[tag]["weeks_improved"],
                               "delta_interval": rows[tag]["paired_week_bootstrap_delta_95pct"]}))
    baseline = forecasts["abl_sm5_hmark_anchor_rep"][0]
    baseline_blocks = np.sqrt(((baseline - true) ** 2).mean(1))
    for tag, (_, refitted) in forecasts.items():
        mixed = (baseline + refitted) / 2
        score = summarize(mixed, true)
        blocks = np.asarray(score["per_block_RMSE"])
        rng = np.random.default_rng(0)
        samples = rng.integers(0, len(blocks), size=(10000, len(blocks)))
        delta = np.sqrt((blocks[samples] ** 2).mean(1)) - \
            np.sqrt((baseline_blocks[samples] ** 2).mean(1))
        rows["mixed_" + tag] = {"metrics": score,
            "weeks_improved": int((blocks < baseline_blocks).sum()),
            "total_weeks": len(blocks),
            "paired_week_bootstrap_delta_95pct": np.percentile(delta, [2.5, 97.5]).tolist()}
        print("mixed_" + tag, json.dumps({"RMSE": score["RMSE"],
                                         "edge16_RMSE": score["edge16_RMSE"],
                                         "weeks_improved": rows["mixed_" + tag]["weeks_improved"]}))
    dest = Path("results/refit_comparison.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
