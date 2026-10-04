"""Refit an epoch-selected Autoformer on all history available at an origin.

Example (historical check, then the real forecast):
    python -m pa1q2.refit runs/dev/<run>.json --fit-end 35064 --eval-end 43656
    python -m pa1q2.refit runs/final/<run>.json --fit-end 43656

The selected epoch comes from the source run, never from the refit's evaluation
targets. Source training epochs and refit epochs both count in the declaration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import data as D
from .autoformer import Autoformer, count_parameters
from .train import Windows, build_arrays, evaluate, parser, predict


def run(source, fit_end, eval_end, out):
    source = Path(source)
    record = json.loads(source.read_text())
    a = SimpleNamespace(**(vars(parser().parse_args([])) | record["args"]))
    if a.stage not in ("dev", "final"):
        raise ValueError("Choose an original early-stopped run as the selection record")
    selection_end = D.ES_END if a.stage == "dev" else D.N_HISTORY
    if not selection_end <= fit_end <= D.N_HISTORY:
        raise ValueError("Refit must start after the source run's epoch-selection period")
    if not fit_end <= eval_end <= D.N_HISTORY:
        raise ValueError("Evaluation targets must follow training and be observed")
    epochs = int(record["best_epoch"])
    if not 1 <= epochs <= record["epochs_run"]:
        raise ValueError("Selected epoch must belong to the completed source run")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{source.stem}-refit{fit_end}"
    if dest.with_suffix(".json").exists():
        raise FileExistsError(f"Preserving completed run: {dest}")

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rolling = tuple(int(w) for w in a.rolling.split(",") if w)
    y, scaler, values, marks = build_arrays(a.cov, bool(a.log_target), fit_end,
                                           bool(a.phase), rolling,
                                           bool(getattr(a, "sqrt_target", 0)),
                                           bool(getattr(a, "accum", 0)))
    win = Windows(values, marks, a.seq_len, a.label_len, D.PRED_LEN,
                  bool(a.anchor), bool(a.horizon_mark), bool(getattr(a, "window_norm", 0)))
    model = Autoformer(values.shape[1], marks.shape[1] + 2 * a.horizon_mark,
                       a.seq_len, a.label_len, D.PRED_LEN, a.d_model, a.heads,
                       a.d_ff, a.e_layers, a.d_layers, a.kernel, a.dropout,
                       a.factor, a.mark_kernel, a.pad, a.trend_init,
                       getattr(a, "mark_mlp", 0), getattr(a, "cov_head", 0))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    gen = torch.Generator().manual_seed(a.seed)
    origins = D.origins(0, fit_end, a.seq_len, 1)
    curve = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for group in opt.param_groups:
            group["lr"] = a.lr * 0.5 ** (epoch - 1)
        perm = torch.as_tensor(origins)[torch.randperm(len(origins), generator=gen)]
        loss_sum = 0.0
        for idx in perm.split(a.batch):
            x, me, md, target, _ = win.batch(idx)
            loss = (model(x, me, md) - target).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * len(idx)
        row = {"epoch": epoch, "train_loss": loss_sum / len(origins),
               "epoch_seconds": time.time() - t0}
        curve.append(row)
        print(json.dumps({"run": dest.name, **row}), flush=True)
        torch.save(model.state_dict(), dest.with_suffix(".pt"))

    result = {"args": vars(a) | {"stage": "refit"}, "fit_end": fit_end,
              "selection_record": str(source),
              "selection_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "selection_epochs": record["epochs_run"], "refit_epochs": epochs,
              "epochs_run": record["epochs_run"] + epochs,
              "best_epoch": epochs, "parameters": count_parameters(model),
              "best_es_RMSE": record["best_es_RMSE"],
              "output_floor": float(np.percentile(y[:fit_end], 1)),
              "curve": curve, "train_seconds": time.time() - started}
    if eval_end > fit_end:
        # Align the final evaluation block with the end of the observed interval.
        blocks = (eval_end - fit_end) // D.PRED_LEN
        if not blocks:
            raise ValueError("Evaluation requires at least 168 observations")
        test = np.arange(eval_end - blocks * D.PRED_LEN,
                         eval_end - D.PRED_LEN + 1, D.PRED_LEN)
        pred, true, metrics = evaluate(model, win, test, y, scaler)
        result["holdout"] = metrics
        np.savez_compressed(dest.with_suffix(".npz"), pred=pred, true=true, origins=test)
    if fit_end == D.N_HISTORY:
        result["forecast"] = scaler.inverse(predict(model, win, np.array([fit_end])))[0].tolist()
    dest.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"run": dest.name, "epochs_total": result["epochs_run"],
                      "holdout": result.get("holdout")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("--fit-end", type=int, required=True)
    p.add_argument("--eval-end", type=int, default=D.N_HISTORY)
    p.add_argument("--out", default="runs/refit")
    args = p.parse_args()
    run(args.source, args.fit_end, args.eval_end, args.out)
