# make_cost_accuracy_figure.py
# One figure: wall-clock time vs accuracy for every PINN run and for the classical solvers,
# for the three problems.  Put it in analysis/ and run from anywhere:
#     python analysis/make_cost_accuracy_figure.py
#
# Reads    results/kaggle/*/*.json                      (raw per-run PINN results)
#          results/baselines/classical_baselines.json   (classical solvers, timed on your machine)
# Writes   results/figures/cost_vs_accuracy.png
#
# Timing caveat (also printed on the figure): PINN seconds are per run on a Kaggle T4 GPU with four
# jobs sharing two GPUs; classical seconds are single-process CPU timings on the author's machine.
#
# Needs numpy, matplotlib.

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
base = json.load(open(ROOT / "results" / "baselines" / "classical_baselines.json"))


def load(pattern):
    out = []
    for f in sorted(glob.glob(str(ROOT / "results" / "kaggle" / pattern))):
        d = json.load(open(f))
        if "final" in d:
            d.setdefault("bc_weight", 1.0)
            out.append(d)
    return out


def runs_by(rows, keyfn, wanted):
    g = defaultdict(list)
    for r in rows:
        k = keyfn(r)
        if k in wanted:
            g[wanted[k]].append((r["seconds_adam"] + r["seconds_lbfgs"], r["final"]["l2"]))
    return g


heat = load("heat1d_*/*.json")
heat_g = runs_by(
    heat, lambda r: (r["precision"], r["constraint"], r["bc_weight"], "init" in r),
    {("float32", "soft", 1.0, True): "float32 soft", ("float32", "hard", 1.0, True): "float32 hard",
     ("float64", "soft", 1.0, False): "float64 soft", ("float64", "hard", 1.0, False): "float64 hard"})
burg = load("burgers_*/*.json")
burg_g = runs_by(
    burg, lambda r: (r["precision"], r["constraint"], r["bc_weight"]),
    {("float32", "soft", 1.0): "float32 soft", ("float32", "hard", 1.0): "float32 hard",
     ("float64", "hard", 1.0): "float64 hard"})
pois = load("poisson2d_*/*.json")
pois_g = runs_by(
    pois, lambda r: (r["constraint"], r["scale"]),
    {("soft", "raw"): "soft raw", ("soft", "rescaled"): "soft rescaled",
     ("hard", "raw"): "hard raw", ("hard", "rescaled"): "hard rescaled"})

panels = [
    ("1D heat (transient)", heat_g,
     [("explicit Euler", base["heat1d_explicit"], "nx", "s", "tab:gray"),
      ("Crank-Nicolson", base["heat1d_crank_nicolson"], "nx", "o", "black")]),
    ("Burgers (shock)", burg_g,
     [("central FD + BDF", base["burgers_central_fd_bdf"], "grid_points", "o", "black")]),
    ("2D Poisson", pois_g,
     [("direct sparse solve", base["poisson2d_direct"], "n", "o", "black")]),
]

# the same colour means the same thing in every panel: soft constraints red/orange, hard constraints green/blue
COLOR = {"float32 soft": "tab:red", "float64 soft": "tab:orange", "float32 hard": "tab:green", "float64 hard": "tab:blue",
         "soft raw": "tab:red", "soft rescaled": "tab:orange", "hard raw": "tab:green", "hard rescaled": "tab:blue"}
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
for ax, (title, groups, classical) in zip(axes, panels):
    for name, pts in groups.items():
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], s=26, alpha=0.75, color=COLOR[name], label=f"PINN {name}")
    for name, rows_, nkey, marker, color in classical:
        xs = [r["seconds"] for r in rows_]
        ys = [r["l2"] for r in rows_]
        ax.plot(xs, ys, marker=marker, color=color, lw=1.2, ms=5, label=f"classical: {name}")
        for r, x_, y_ in zip(rows_, xs, ys):
            ax.annotate(f"{r[nkey]:,}", (x_, y_), textcoords="offset points", xytext=(4, 4), fontsize=6.5, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock seconds per solve / per training run")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="best")
axes[0].set_ylabel("relative L2 error vs reference solution")
fig.suptitle("Accuracy versus cost: PINN runs (dots) and classical solvers (lines; labels = grid size)")
fig.text(0.5, 0.005,
         "PINN: per-run seconds on a Kaggle T4 GPU, four jobs sharing two GPUs.  Classical: single-process CPU timings on the author's machine.",
         ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.025, 1, 1))
(ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "results" / "figures" / "cost_vs_accuracy.png", dpi=150)
print("Saved results/figures/cost_vs_accuracy.png")
for title, groups, _ in panels:
    print(f"  {title}: " + ", ".join(f"{k} (n={len(v)})" for k, v in groups.items()))
