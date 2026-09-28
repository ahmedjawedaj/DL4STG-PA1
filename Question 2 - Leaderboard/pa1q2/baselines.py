"""Non-neural reference forecasts scored on the same 51 holdout blocks as the Autoformer runs.

These are context only (the submitted model is the Autoformer). Like the dev-stage Autoformer
runs, all are fitted on [0, 26304) only (2013 is not used, since they need no early stopping).
    persistence        repeat the last observed value
    seasonal_naive_24  repeat the last 24 observed values
    mean_last_168      the mean of the last week
    climatology        mean of the fit region per 24-step phase
    ridge_<log|raw>_no_cov      ridge on the last 168 values + phase, one output per horizon step
    ridge_<log|raw>_future_cov  the same plus the optional variables over the known horizon
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import data as D


def _ridge(X, Y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    Xs = np.concatenate([Xs, np.ones((len(Xs), 1))], 1)
    reg = lam * np.eye(Xs.shape[1])
    reg[-1, -1] = 0
    W = np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ Y)
    return lambda Z: np.concatenate([(Z - mu) / sd, np.ones((len(Z), 1))], 1) @ W


def main(out="runs/baselines.json"):
    y, ext = D.load()
    covs = D.covariate_features(ext, D.FIT_END)
    ph = D.phase_features()
    ly = np.log1p(y)
    test = D.holdout_origins(168)
    true = np.stack([y[o:o + 168] for o in test])
    res = {}
    res["persistence"] = np.stack([np.full(168, y[o - 1]) for o in test])
    res["seasonal_naive_24"] = np.stack([np.tile(y[o - 24:o], 7) for o in test])
    res["mean_last_168"] = np.stack([np.full(168, y[o - 168:o].mean()) for o in test])
    phase_mean = np.array([y[:D.FIT_END][np.arange(D.FIT_END) % 24 == k].mean() for k in range(24)])
    res["climatology"] = np.stack([phase_mean[(o + np.arange(168)) % 24] for o in test])

    # direct multi-output ridge (log1p or raw target); the covariate version appends the covariates
    # of each forecast step, so it reads the known horizon of the optional file
    train = D.origins(0, D.FIT_END, 168, 6)

    def design(origins, with_cov, base):
        rows = []
        for o in origins:
            parts = [base[o - 168:o], ph[o]]
            if with_cov:
                parts.append(covs[o:o + 168].reshape(-1))
            rows.append(np.concatenate(parts))
        return np.stack(rows)

    for space in ("log", "raw"):
        base = ly if space == "log" else y
        Ytr = np.stack([base[o:o + 168] for o in train])
        for name, with_cov in (("no_cov", False), ("future_cov", True)):
            f = _ridge(design(train, with_cov, base), Ytr, lam=10.0)
            pred = f(design(test, with_cov, base))
            res[f"ridge_{space}_{name}"] = np.clip(np.expm1(pred) if space == "log" else pred, 0, None)

    table = {k: D.metrics(v, true) for k, v in res.items()}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(table, indent=1))
    for k, v in table.items():
        print(f"{k:20s} " + "  ".join(f"{m}={x:7.2f}" for m, x in v.items()))


if __name__ == "__main__":
    main()
