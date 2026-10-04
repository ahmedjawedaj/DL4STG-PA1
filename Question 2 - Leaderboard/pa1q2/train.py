"""Train one Autoformer configuration with one seed and write a JSON record.

Examples
    python -m pa1q2.train --cov none   --seed 0 --stage dev
    python -m pa1q2.train --cov future --seed 0 --stage final --out runs/final

stage=dev   : fit [0, 26304), early-stop on [26304, 35064), score 51 holdout blocks in 2014.
stage=final : fit [0, 35064), early-stop on the 51 holdout blocks, forecast time_idx 43657..43824.
Epoch = one full pass over every stride-1 training window of the fit region.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .autoformer import Autoformer, count_parameters


def build_arrays(cov: str, log_target: bool, fit_end: int, phase: bool, rolling=(),
                 sqrt_target: bool = False, accum: bool = False):
    y, ext = D.load()
    scaler = D.TargetScaler(log=log_target, sqrt=sqrt_target).fit(y[:fit_end])
    z = np.zeros(D.N_TOTAL, np.float32)
    z[:D.N_HISTORY] = scaler.transform(y)            # hidden block stays 0 and is never read
    covs = D.covariate_features(ext, stats_end=fit_end, rolling=rolling, accum=accum)
    ph = D.phase_features() if phase else np.zeros((D.N_TOTAL, 0), np.float32)
    values = z[:, None]
    marks = ph
    if cov == "past":        # covariates as encoder values: known only up to the origin
        values = np.concatenate([values, covs], 1)
    elif cov == "future":    # covariates as known inputs over encoder and decoder windows
        marks = np.concatenate([ph, covs], 1)
    elif cov == "both":
        values = np.concatenate([values, covs], 1)
        marks = np.concatenate([ph, covs], 1)
    elif cov != "none":
        raise ValueError(cov)
    return y, scaler, torch.from_numpy(values), torch.from_numpy(marks.astype(np.float32))


class Windows:
    """Vectorised window gathering; no window crosses its origin boundary."""

    def __init__(self, values, marks, seq_len, label_len, pred_len, anchor=False,
                 horizon_mark=False, window_norm=False):
        self.values, self.marks, self.anchor = values, marks, anchor
        self.window_norm = window_norm
        self.enc = torch.arange(-seq_len, 0)
        self.dec = torch.arange(-label_len, pred_len)
        # known-in-advance position relative to the forecast origin (Autoformer itself has no
        # positional encoding): scaled offset and an exp(-h/24) "closeness to origin" feature
        def rel(offsets):
            h = offsets.float()
            return torch.stack([h / pred_len, torch.exp(-h.clamp(min=0) / 24)], 1)
        self.enc_rel = rel(self.enc) if horizon_mark else None
        self.dec_rel = rel(self.dec) if horizon_mark else None
        self.fut = torch.arange(0, pred_len)

    def batch(self, origins):
        o = torch.as_tensor(origins).view(-1, 1)
        enc, dec = o + self.enc, o + self.dec
        x = self.values[enc]
        target = self.values[o + self.fut][..., 0]
        # anchoring (as in Question 1): subtract the last observed (scaled) target value from
        # the target channel and the target, so the network forecasts changes from it
        a = x[:, -1:, 0] if self.anchor else torch.zeros(len(o), 1)
        if self.anchor:
            x = x.clone()
            x[..., 0] = x[..., 0] - a
            target = target - a
        if self.window_norm:
            # window-relative normalisation (handout Q2.6): remove the encoder window's own
            # mean and spread from the target channel and the target, restore afterwards
            mu = x[:, :, 0].mean(1, keepdim=True)
            sd = x[:, :, 0].std(1, keepdim=True) + 0.1
            x = x.clone()
            x[..., 0] = (x[..., 0] - mu) / sd
            target = (target - mu) / sd
            a = torch.cat([a + mu, sd], 1)                  # [B, 2]: offset, scale
        me, md = self.marks[enc], self.marks[dec]
        if self.enc_rel is not None:
            n = len(o)
            me = torch.cat([me, self.enc_rel.expand(n, -1, -1)], -1)
            md = torch.cat([md, self.dec_rel.expand(n, -1, -1)], -1)
        return x, me, md, target, a


def restore(z, a):
    """Undo anchoring / window normalisation: a is [B,1] offset or [B,2] offset and scale."""
    if a.shape[1] == 2:
        return z * a[:, 1:2] + a[:, :1]
    return z + a


@torch.no_grad()
def predict(model, win, origins, batch=256):
    model.eval()
    out = []
    for i in range(0, len(origins), batch):
        x, me, md, _, a = win.batch(origins[i:i + batch])
        out.append(restore(model(x, me, md), a).numpy())
    return np.concatenate(out)


def evaluate(model, win, origins, y_raw, scaler):
    pred = scaler.inverse(predict(model, win, origins))
    true = np.stack([y_raw[o:o + D.PRED_LEN] for o in origins])
    return pred, true, D.metrics(pred, true)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--cov", default="future", choices=["none", "past", "future", "both"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", default="dev", choices=["dev", "final"])
    p.add_argument("--log-target", type=int, default=0)
    p.add_argument("--phase", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=168)
    p.add_argument("--label-len", type=int, default=48)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=64)
    p.add_argument("--e-layers", type=int, default=1)
    p.add_argument("--d-layers", type=int, default=1)
    p.add_argument("--kernel", type=int, default=25)
    p.add_argument("--factor", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=6)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--min-delta", type=float, default=0.0025,
                   help="relative early-stopping RMSE improvement that counts as progress")
    p.add_argument("--es-stride", type=int, default=24)
    p.add_argument("--threads", type=int, default=int(os.environ.get("T2_THREADS", "2")))
    p.add_argument("--out", default="runs/dev")
    p.add_argument("--tag", default="")
    p.add_argument("--rolling", default="6,24", help="comma list of trailing covariate windows")
    p.add_argument("--mark-kernel", type=int, default=1)
    p.add_argument("--anchor", type=int, default=0, help="1: forecast change from last value")
    p.add_argument("--horizon-mark", type=int, default=0, help="1: add origin-relative marks")
    p.add_argument("--trend-init", default="mean", choices=["mean", "last"])
    p.add_argument("--fold", default="B", choices=["A", "B"],
                   help="dev protocol: B fits to 26304 and scores the last cycle, A one cycle earlier")
    p.add_argument("--winter-weight", type=float, default=1.0,
                   help="loss weight for training windows whose target week is winter-like")
    p.add_argument("--sqrt-target", type=int, default=0, help="1: model the square root of y")
    p.add_argument("--accum", type=int, default=0, help="1: add stagnation / clear-out features")
    p.add_argument("--window-norm", type=int, default=0, help="1: window-relative normalisation")
    p.add_argument("--mark-mlp", type=int, default=0, help="hidden width of a nonlinear covariate encoder (0: linear)")
    p.add_argument("--lr-decay", type=float, default=0.5, help="per-epoch learning-rate factor")
    p.add_argument("--cov-head", type=int, default=0, help="hidden width of a direct per-hour covariate head (0: none)")
    p.add_argument("--es-season", default="all", choices=["all", "winter"],
                   help="winter: early-stop on winter-like weeks of the early-stopping year")
    p.add_argument("--pad", default="circular", choices=["circular", "replicate"],
                   help="padding of the value embedding and decoder trend projection")
    return p


def main(argv=None):
    a = parser().parse_args(argv)

    return run(a)


def run_tag(a):
    tag = f"{a.cov}-log{a.log_target}-ph{a.phase}-L{a.seq_len}-d{a.d_model}-s{a.seed}"
    return tag + (f"-{a.tag}" if a.tag else "")


def run(a, budget=None):
    """Train (resuming from runs/<out>/state/<tag>.pt if present). Returns True when finished.
    With budget (seconds), stops between epochs when another epoch would not fit."""
    call_start = time.time()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = run_tag(a)
    if (out / f"{tag}.json").exists():
        return True
    fold_fit, fold_es, test_start, test_end = D.fold_ranges(a.fold)
    fit_end = fold_fit if a.stage == "dev" else D.ES_END
    rolling = tuple(int(w) for w in a.rolling.split(",") if w)
    y_raw, scaler, values, marks = build_arrays(a.cov, bool(a.log_target), fit_end, bool(a.phase),
                                                rolling, bool(a.sqrt_target), bool(a.accum))
    win = Windows(values, marks, a.seq_len, a.label_len, D.PRED_LEN, bool(a.anchor),
                  bool(a.horizon_mark), bool(a.window_norm))

    train_origins = D.origins(0, fit_end, a.seq_len, 1)
    if a.stage == "dev":
        es_origins = D.origins(fold_fit, fold_es, a.seq_len, a.es_stride)
        test_origins = D.test_blocks(test_start, test_end)
        es_start = fold_fit
    else:
        es_origins = D.holdout_origins(a.seq_len)
        test_origins = None
        es_start = D.ES_END
    if a.es_season == "winter":
        # select the epoch on winter-like weeks only (first 59 and last 61 days of the
        # early-stopping year), because the forecast week falls at the end of a year
        day = (es_origins + D.PRED_LEN // 2 - es_start) // 24
        es_origins = es_origins[(day < 59) | (day >= 304)]

    model = Autoformer(values.shape[1], marks.shape[1] + 2 * a.horizon_mark, a.seq_len, a.label_len, D.PRED_LEN,
                       a.d_model, a.heads, a.d_ff, a.e_layers, a.d_layers, a.kernel,
                       a.dropout, a.factor, a.mark_kernel, a.pad, a.trend_init, a.mark_mlp,
                       a.cov_head)
    params = count_parameters(model)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    gen = torch.Generator().manual_seed(a.seed)
    st = dict(best=float("inf"), best_state=None, best_epoch=0, bad=0, curve=[], epoch=0,
              elapsed=0.0, stopped=False)
    state_path = out / "state" / f"{tag}.pt"
    if state_path.exists():
        saved = torch.load(state_path, weights_only=False)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        gen.set_state(saved["gen"])
        torch.set_rng_state(saved["torch_rng"])
        st = saved["st"]
    last_epoch_seconds = st["curve"][-1]["epoch_seconds"] if st["curve"] else 0.0
    while not st["stopped"] and st["epoch"] < a.max_epochs:
        if budget is not None and st["curve"] and \
                time.time() - call_start + 1.1 * last_epoch_seconds > budget:
            return False
        epoch = st["epoch"] + 1
        t0 = time.time()
        for g in opt.param_groups:                        # reference lradj "type1"
            g["lr"] = a.lr * a.lr_decay ** (epoch - 1)
        model.train()
        perm = torch.as_tensor(train_origins)[torch.randperm(len(train_origins), generator=gen)]
        total = 0.0
        for idx in perm.split(a.batch):
            x, me, md, target, _ = win.batch(idx)
            per_window = (model(x, me, md) - target).square().mean(1)
            if a.winter_weight != 1.0:
                # season-weighted loss: windows whose target week is winter-like count
                # winter_weight times as much (the forecast week is a winter week)
                w = torch.ones(len(idx))
                w[torch.as_tensor(D.is_winter(idx.numpy()))] = a.winter_weight
                loss = (w * per_window).sum() / w.sum()
            else:
                loss = per_window.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        _, _, es = evaluate(model, win, es_origins, y_raw, scaler)
        last_epoch_seconds = time.time() - t0
        st["elapsed"] += last_epoch_seconds
        st["epoch"] = epoch
        st["curve"].append({"epoch": epoch, "train_loss": total / len(train_origins), **es,
                            "epoch_seconds": last_epoch_seconds})
        print(json.dumps(st["curve"][-1]), flush=True)
        if es["RMSE"] < st["best"] * (1 - a.min_delta):
            st.update(best=es["RMSE"], best_epoch=epoch, bad=0,
                      best_state={k: v.clone() for k, v in model.state_dict().items()})
        else:
            st["bad"] += 1
            st["stopped"] = st["bad"] >= a.patience
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(),
                    "torch_rng": torch.get_rng_state(), "st": st}, state_path)
    model.load_state_dict(st["best_state"])

    record = {"args": vars(a), "parameters": params, "epochs_run": st["epoch"],
              "best_epoch": st["best_epoch"], "best_es_RMSE": st["best"], "curve": st["curve"],
              "train_seconds": st["elapsed"]}
    if a.stage == "dev":
        pred, true, m = evaluate(model, win, test_origins, y_raw, scaler)
        record["holdout"] = m
        record["holdout_blocks"] = int(len(test_origins))
        np.savez_compressed(out / f"{tag}.npz", pred=pred, true=true, origins=test_origins)
    else:
        origin = np.array([D.N_HISTORY])
        forecast = scaler.inverse(predict(model, win, origin))[0]
        record["forecast"] = forecast.tolist()
        torch.save(model.state_dict(), out / f"{tag}.pt")
    (out / f"{tag}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ("parameters", "epochs_run", "best_epoch",
                                            "best_es_RMSE")} | {"holdout": record.get("holdout")}))
    return True


if __name__ == "__main__":
    main()
