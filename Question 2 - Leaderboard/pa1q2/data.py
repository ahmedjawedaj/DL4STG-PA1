"""Data loading, feature construction, window indexing and chronological splits for Task 2.

Layout of the series (0-based positions, time_idx = position + 1):
    history      [0, 43656)   observed target (student_train.csv)
    hidden block [43656, 43824)   168 positions to forecast (time_idx 43657..43824)

Chronological splits used everywhere (boundaries are 8760/8784-step blocks, i.e. one year
of the 24-step cycle recovered from the autocorrelation/spectrum):
    fit          [0, 26304)       model parameters
    early-stop   [26304, 35064)   choose the epoch (targets fully inside this range)
    holdout      [35064, 43656)   compare designs: 51 non-overlapping 168-step blocks
Final model: fit on [0, 35064), early-stop on the holdout blocks, forecast the hidden block.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
N_HISTORY = 43656
N_TOTAL = 43824
PRED_LEN = 168
FIT_END = 26304
ES_END = 35064
HOLDOUT_END = N_HISTORY
DAY = 24            # dominant short period recovered from the spectrum / ACF
YEAR = 8766         # slow period, ~365.25 * 24

CONTINUOUS = ["feature_A", "feature_B", "feature_C", "feature_D", "feature_E", "feature_F"]
BINARY = ["feature_G", "feature_H", "feature_I", "feature_J"]
HEAVY = ["feature_D", "feature_E", "feature_F"]  # cumulative counters: log1p first


def load(data_dir: Path = DATA_DIR):
    train = pd.read_csv(data_dir / "student_train.csv")
    test = pd.read_csv(data_dir / "student_test.csv")
    ext = pd.read_csv(data_dir / "optional_external_data.csv")
    assert len(train) == N_HISTORY and train["value"].notna().all()
    assert test["time_idx"].tolist() == list(range(N_HISTORY + 1, N_TOTAL + 1))
    assert ext["time_idx"].tolist() == list(range(1, N_TOTAL + 1))
    return train["value"].to_numpy(np.float64), ext


def phase_features(n: int = N_TOTAL) -> np.ndarray:
    """Periodic encodings derived from the integer index only (no calendar)."""
    t = np.arange(n, dtype=np.float64)
    cols = []
    for period in (DAY, YEAR):
        cols += [np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period)]
    return np.stack(cols, 1).astype(np.float32)


def accumulation_features(ext: pd.DataFrame) -> np.ndarray:
    """Four stagnation and clear-out features built from the covariate file alone.
    feature_H is the wind direction with high speeds and low target levels (a clearing
    wind), feature_J the calm one with the highest levels. Each feature at time t uses only
    covariate rows up to t, and the file covers the horizon, so they are known in advance.
        since_clear     hours since the last strong clearing-wind hour (H and D >= 20), cap 168
        calm48, calm120 share of the trailing 48 / 120 hours that were calm (J or D < 3)
        clear72         share of the trailing 72 hours with a strong clearing wind
    """
    n = len(ext)
    h, j, d = (ext[c].to_numpy(np.float64) for c in ("feature_H", "feature_J", "feature_D"))
    clear = (h == 1) & (d >= 20)
    calm = (j == 1) | (d < 3)
    since = np.empty(n)
    last = -PRED_LEN
    for t in range(n):
        if clear[t]:
            last = t
        since[t] = min(t - last, PRED_LEN)

    def trailing(x, w):
        c = np.cumsum(np.r_[0, x])
        i = np.arange(1, n + 1)
        return (c[i] - c[np.maximum(i - w, 0)]) / w

    return np.stack([since / PRED_LEN, trailing(calm, 48), trailing(calm, 120),
                     trailing(clear, 72)], 1).astype(np.float32)


def covariate_features(ext: pd.DataFrame, stats_end: int, rolling=(), accum: bool = False) -> np.ndarray:
    """Scale the ten optional variables using statistics from [0, stats_end) only.
    rolling: trailing-window lengths; each adds trailing means of all ten variables
    (computed from the covariate file alone, so no target information enters)."""
    cont = ext[CONTINUOUS].to_numpy(np.float64).copy()
    for j, name in enumerate(CONTINUOUS):
        if name in HEAVY:
            cont[:, j] = np.log1p(np.clip(cont[:, j], 0, None))
    binary = ext[BINARY].to_numpy(np.float64)
    base = np.concatenate([cont, binary], 1)
    blocks = [base]
    for w in rolling:
        c = np.cumsum(np.vstack([np.zeros((1, base.shape[1])), base]), 0)
        idx = np.arange(1, len(base) + 1)
        lo = np.maximum(idx - w, 0)
        blocks.append((c[idx] - c[lo]) / (idx - lo)[:, None])
    if accum:
        blocks.append(accumulation_features(ext))
    feats = np.concatenate(blocks, 1)
    mu, sd = feats[:stats_end].mean(0), feats[:stats_end].std(0) + 1e-8
    return ((feats - mu) / sd).astype(np.float32)


@dataclass
class TargetScaler:
    """Global (not per-window) scaling of the target, fitted on the fit region only.
    Optional variance-stabilising transform: log1p or square root."""
    log: bool
    mu: float = 0.0
    sd: float = 1.0
    sqrt: bool = False

    def _forward(self, y):
        if self.log:
            return np.log1p(y)
        return np.sqrt(np.clip(y, 0, None)) if self.sqrt else y

    def fit(self, y):
        z = self._forward(y)
        self.mu, self.sd = float(z.mean()), float(z.std())
        return self

    def transform(self, y):
        return ((self._forward(y) - self.mu) / self.sd).astype(np.float32)

    def inverse(self, z):
        z = np.asarray(z, np.float64) * self.sd + self.mu
        if self.log:
            out = np.expm1(z)
        elif self.sqrt:
            out = np.clip(z, 0, None) ** 2
        else:
            out = z
        return np.clip(out, 0, None)  # target is non-negative


def origins(start: int, end: int, seq_len: int, stride: int, pred_len: int = PRED_LEN):
    """Forecast origins o (first predicted position) with target [o, o+pred_len) in [start, end)."""
    first = max(start, seq_len)
    return np.arange(first, end - pred_len + 1, stride)


def season_day(origins):
    """Day within an assumed 365.25-day cycle (24-step days) of each target block's centre."""
    return ((np.asarray(origins) + PRED_LEN // 2) % YEAR) // 24


def is_winter(origins):
    """Winter-like: first 59 or last 61 days of the cycle, the season of the hidden week."""
    day = season_day(origins)
    return (day < 59) | (day >= 304)


def fold_ranges(fold: str):
    """(fit_end, es_end, test_start, test_end). Fold B is the main protocol. Fold A shifts
    everything one cycle earlier so that a second, independent winter can be scored."""
    if fold == "B":
        return FIT_END, ES_END, ES_END, HOLDOUT_END
    if fold == "A":
        return 17520, FIT_END, FIT_END, ES_END
    raise ValueError(fold)


def test_blocks(test_start: int, test_end: int):
    """Non-overlapping 168-step blocks tiling the end of [test_start, test_end)."""
    n = (test_end - test_start) // PRED_LEN
    return np.arange(test_end - n * PRED_LEN, test_end - PRED_LEN + 1, PRED_LEN)


def holdout_origins(seq_len: int):
    """51 non-overlapping 168-step blocks that tile the last 8568 steps of the history."""
    n_blocks = (HOLDOUT_END - ES_END) // PRED_LEN
    first = HOLDOUT_END - n_blocks * PRED_LEN
    return np.arange(first, HOLDOUT_END - PRED_LEN + 1, PRED_LEN)


def metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    denom = np.abs(true) + np.abs(pred)
    smape = np.where(denom > 0, 2 * np.abs(err) / np.where(denom > 0, denom, 1), 0.0)
    block_rmse = np.sqrt((err ** 2).reshape(len(err), -1).mean(1)) if err.ndim == 2 else None
    out = {"RMSE": float(np.sqrt((err ** 2).mean())), "MAE": float(np.abs(err).mean()),
           "sMAPE": float(100 * smape.mean())}
    if block_rmse is not None:
        out["mean_block_RMSE"] = float(block_rmse.mean())
    return out
