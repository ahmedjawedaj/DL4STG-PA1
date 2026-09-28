"""Report figures for Task 2.   python -m pa1q2.figures RUN_DIR [RUN_DIR ...]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data as D

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
INK, MUTED = "#1f1f1e", "#6b6a64"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c0", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe6",
                     "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.6})
OUT = Path("results/figures")

LABELS = {"none-abl": "no optional data", "past-abl": "past-only covariates",
          "future-abl": "known-horizon covariates (d32)",
          "future-abl_noroll": "  without trailing means",
          "future-abl_log": "  log1p target",
          "future-abl_mk5": "  + width-5 covariate embedding",
          "future-abl_small": "  d16",
          "future-abl_small_mk5": "  d16 + width-5 embedding",
          "future-abl_small_mk5_anchor": "  d16 + width-5 + last-value anchor",
          "future-abl_sm5_hmark_anchor_rep": "  d16 + width-5 + horizon marks + anchor",
          "future-abl_sm5_hmark_rep": "  d16 + width-5 + horizon marks (submitted)"}
SUBMITTED = "future-abl_sm5_hmark_rep"


def load_runs(dirs):
    runs = {}
    for d in dirs:
        for f in Path(d).glob("*.json"):
            r = json.loads(f.read_text())
            if "holdout" not in r:
                continue
            key = f"{r['args']['cov']}-{r['args']['tag']}"
            z = np.load(f.with_suffix(".npz"))
            runs.setdefault(key, []).append((r, z["pred"], z["true"], z["origins"]))
    return runs


def ablation_plot(runs, baselines):
    order = [k for k in LABELS if k in runs]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for i, key in enumerate(order):
        vals = [r["holdout"]["RMSE"] for r, *_ in runs[key]]
        colour = ORANGE if key == SUBMITTED else BLUE
        ax.scatter(vals, [i] * len(vals), s=22, color=colour, alpha=0.55, zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.scatter([np.mean(vals)], [i], s=70, marker="|", color=INK, zorder=4, linewidth=2)
    for name, x, ls in (("climatology", baselines["climatology"]["RMSE"], ":"),
                        ("direct ridge with known-horizon covariates", baselines["ridge_raw_future_cov"]["RMSE"], "--")):
        ax.axvline(x, color=GRAY, linestyle=ls, linewidth=1, label=f"{name} ({x:.1f})")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=7.5)
    ax.set_yticks(range(len(order)), [LABELS[k] for k in order])
    ax.invert_yaxis()
    ax.set_xlabel("holdout RMSE, 51 weekly blocks (dots are seeds, bar is the mean)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "q2_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def horizon_plot(runs):
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for key, colour, label in (("none-abl", GRAY, "no optional data"),
                               ("past-abl", AQUA, "past-only covariates"),
                               (SUBMITTED, ORANGE, "known-horizon covariates (submitted config)")):
        curves = [np.sqrt(((p - t) ** 2).mean(0)) for _, p, t, _ in runs[key]]
        ax.plot(np.arange(1, 169), np.mean(curves, 0), color=colour, label=label)
    ax.set_xlabel("forecast step")
    ax.set_ylabel("RMSE (seed mean)")
    ax.set_xticks(range(0, 169, 24))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "q2_horizon.pdf", bbox_inches="tight")
    plt.close(fig)


def example_plot(runs):
    _, _, true, origins = runs[SUBMITTED][0]
    winter = (origins - D.ES_END) // 24 >= 304
    i = int(np.flatnonzero(winter)[np.argmax(true[winter].std(1))])
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.plot(np.arange(168), true[i], color=INK, label="observed")
    for key, colour, label in (("none-abl", GRAY, "no optional data"),
                               (SUBMITTED, ORANGE, "known-horizon covariates")):
        pred = np.mean([p[i] for _, p, _, _ in runs[key]], 0)
        ax.plot(np.arange(168), pred, color=colour, label=f"{label} (seed mean)")
    ax.set_xlabel(f"step after origin (holdout block starting at time_idx {origins[i] + 1})")
    ax.set_ylabel("target")
    ax.set_xticks(range(0, 169, 24))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "q2_example_week.pdf", bbox_inches="tight")
    plt.close(fig)


def final_plot():
    y, _ = D.load()
    forecast = np.array([float(v) for v in Path("submission/predictions.txt").read_text().split(",")])
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    hist = np.arange(D.N_HISTORY - 336, D.N_HISTORY) + 1
    ax.plot(hist, y[-336:], color=BLUE, label="last two observed weeks")
    ax.plot(np.arange(D.N_HISTORY + 1, D.N_TOTAL + 1), forecast, color=ORANGE,
            label="submitted forecast (time_idx 43657 to 43824)")
    ax.axvline(D.N_HISTORY + 0.5, color=GRAY, linestyle=":", linewidth=1)
    ax.set_xlabel("time_idx")
    ax.set_ylabel("target")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "q2_final_forecast.pdf", bbox_inches="tight")
    plt.close(fig)


def main(dirs):
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs(dirs)
    baselines = json.loads(Path("results/baselines.json").read_text())
    ablation_plot(runs, baselines)
    horizon_plot(runs)
    example_plot(runs)
    final_plot()
    print(sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main(sys.argv[1:])
