"""Turn a final-stage run into the leaderboard submission.

    python -m pa1q2.submission runs/final/<tag>.json

Writes submission/predictions.txt (168 comma-separated values, time_idx 43657 first) and
submission/declaration.json (P and E), after rebuilding the model from the saved arguments
to recount its trainable parameters.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .autoformer import Autoformer, count_parameters
from .train import build_arrays


def main(path):
    record = json.loads(Path(path).read_text())
    a = record["args"]
    assert a["stage"] == "final", "use a final-stage run"
    raw_forecast = np.asarray(record["forecast"], dtype=np.float64)
    # floor at the 1st percentile of the fit-region target: the target is a non-negative
    # physical quantity that is almost never near 0, and a linear output head can undershoot.
    # On the dev holdout this floor lowered RMSE for all 3 seeds (see README), by about 0.2.
    y, _ = D.load()
    floor = float(np.percentile(y[:D.ES_END], 1))
    forecast = np.maximum(raw_forecast, floor)
    assert forecast.shape == (D.PRED_LEN,) and np.isfinite(forecast).all() and (forecast >= 0).all()

    rolling = tuple(int(w) for w in a["rolling"].split(",") if w)
    _, _, values, marks = build_arrays(a["cov"], bool(a["log_target"]), D.ES_END, bool(a["phase"]),
                                       rolling)
    model = Autoformer(values.shape[1], marks.shape[1] + 2 * a.get("horizon_mark", 0), a["seq_len"], a["label_len"], D.PRED_LEN,
                       a["d_model"], a["heads"], a["d_ff"], a["e_layers"], a["d_layers"],
                       a["kernel"], a["dropout"], a["factor"], a["mark_kernel"],
                       a.get("pad", "circular"), a.get("trend_init", "mean"))
    state = torch.load(Path(path).with_suffix(".pt"))
    model.load_state_dict(state)
    params = count_parameters(model)
    assert params == record["parameters"], (params, record["parameters"])

    out = Path("submission")
    out.mkdir(exist_ok=True)
    text = ", ".join(f"{v:.4f}" for v in forecast)
    (out / "predictions.txt").write_text(text + "\n")
    time_idx = np.arange(D.N_HISTORY + 1, D.N_TOTAL + 1)
    (out / "predictions.csv").write_text(
        "time_idx,value\n" + "".join(f"{t},{v:.4f}\n" for t, v in zip(time_idx, forecast)))
    declaration = {"trainable_parameters_P": params, "epochs_E": record["epochs_run"],
                   "output_floor": floor, "values_raised_to_floor": int((raw_forecast < floor).sum()),
                   "best_epoch": record["best_epoch"], "run": Path(path).name,
                   "arguments": a, "first_time_idx": int(time_idx[0]),
                   "last_time_idx": int(time_idx[-1])}
    (out / "declaration.json").write_text(json.dumps(declaration, indent=1))
    print(f"P = {params}   E = {record['epochs_run']}   values = {len(forecast)}")
    print(f"first 5: {forecast[:5].round(2).tolist()}   last 5: {forecast[-5:].round(2).tolist()}")


if __name__ == "__main__":
    main(sys.argv[1])
