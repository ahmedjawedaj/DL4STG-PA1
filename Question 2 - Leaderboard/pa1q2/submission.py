"""Turn one or more final-stage runs into the leaderboard submission.

    python -m pa1q2.submission runs/final/<tag>.json [runs/final/<tag2>.json ...]

Writes submission/predictions.txt (168 comma-separated values, time_idx 43657 first) and
submission/declaration.json (P and E). With multiple runs, the submitted forecast is the
average of the per-run floored forecasts and P/E are summed as required by the handout.
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


def _load_final(path):
    path = Path(path)
    record = json.loads(path.read_text())
    args = record["args"]
    assert args["stage"] == "final", "use a final-stage run"
    return path, record, args


def _recount_parameters(path, args, expected):
    rolling = tuple(int(w) for w in args["rolling"].split(",") if w)
    _, _, values, marks = build_arrays(args["cov"], bool(args["log_target"]), D.ES_END,
                                       bool(args["phase"]), rolling)
    model = Autoformer(values.shape[1], marks.shape[1] + 2 * args.get("horizon_mark", 0),
                       args["seq_len"], args["label_len"], D.PRED_LEN,
                       args["d_model"], args["heads"], args["d_ff"], args["e_layers"],
                       args["d_layers"], args["kernel"], args["dropout"], args["factor"],
                       args["mark_kernel"], args.get("pad", "circular"),
                       args.get("trend_init", "mean"))
    state = torch.load(path.with_suffix(".pt"))
    model.load_state_dict(state)
    params = count_parameters(model)
    assert params == expected, (params, expected)
    return params


def main(paths):
    loaded = [_load_final(path) for path in paths]
    # floor at the 1st percentile of the fit-region target: the target is a non-negative
    # physical quantity that is almost never near 0, and a linear output head can undershoot.
    # On the dev holdout this floor lowered RMSE for all 3 seeds (see README), by about 0.2.
    y, _ = D.load()
    floor = float(np.percentile(y[:D.ES_END], 1))
    forecasts = []
    floored_counts = []
    parameters = []
    epochs = []
    for path, record, args in loaded:
        raw_forecast = np.asarray(record["forecast"], dtype=np.float64)
        forecasts.append(np.maximum(raw_forecast, floor))
        floored_counts.append(int((raw_forecast < floor).sum()))
        parameters.append(_recount_parameters(path, args, record["parameters"]))
        epochs.append(int(record["epochs_run"]))
    forecast = np.mean(forecasts, axis=0)
    assert forecast.shape == (D.PRED_LEN,) and np.isfinite(forecast).all() and (forecast >= 0).all()

    out = Path("submission")
    out.mkdir(exist_ok=True)
    text = ", ".join(f"{v:.4f}" for v in forecast)
    (out / "predictions.txt").write_text(text + "\n")
    time_idx = np.arange(D.N_HISTORY + 1, D.N_TOTAL + 1)
    (out / "predictions.csv").write_text(
        "time_idx,value\n" + "".join(f"{t},{v:.4f}\n" for t, v in zip(time_idx, forecast)))
    declaration = {"trainable_parameters_P": int(sum(parameters)), "epochs_E": int(sum(epochs)),
                   "ensemble": [path.name for path, _, _ in loaded],
                   "output_floor": floor,
                   "values_raised_to_floor_per_model": floored_counts,
                   "best_epochs": [record["best_epoch"] for _, record, _ in loaded],
                   "best_es_RMSE": [record["best_es_RMSE"] for _, record, _ in loaded],
                   "arguments": [args for _, _, args in loaded],
                   "first_time_idx": int(time_idx[0]), "last_time_idx": int(time_idx[-1]),
                   "aggregation": "average of per-model floored forecasts"}
    (out / "declaration.json").write_text(json.dumps(declaration, indent=1))
    print(f"P = {sum(parameters)}   E = {sum(epochs)}   values = {len(forecast)}")
    print(f"first 5: {forecast[:5].round(2).tolist()}   last 5: {forecast[-5:].round(2).tolist()}")


if __name__ == "__main__":
    main(sys.argv[1:])
