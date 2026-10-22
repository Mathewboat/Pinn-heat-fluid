# make_corrected_figures.py
# Regenerates every figure that compares PINNs with a finite-difference baseline,
# using (a) the corrected FDM numbers and (b) the PINN numbers already recorded
# by your earlier runs. No retraining is needed.
#
# Run from the repo root:   python make_corrected_figures.py
#
# Reads
#   results/fdm_baselines_corrected.txt              (from fdm_baselines_corrected.py)
#   01_heat_1d/results/data_efficiency_v2_results.txt (PINN section: N, L2 mean, L2 std)
#   02_heat_2d/results/heat2d_summary.txt             (PINN v1)
#   02_heat_2d/results/heat2d_summary_v2.txt          (PINN v2)
# Writes
#   01_heat_1d/results/fig_fdm_refinement_corrected.png
#   01_heat_1d/results/fig_efficiency_corrected.png
#   02_heat_2d/results/fig_2d_comparison_corrected.png

import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

FDM_TXT = "results/fdm_baselines_corrected.txt"
PINN1D_TXT = "01_heat_1d/results/data_efficiency_v2_results.txt"
HEAT2D_V1 = "02_heat_2d/results/heat2d_summary.txt"
HEAT2D_V2 = "02_heat_2d/results/heat2d_summary_v2.txt"


def need(path):
    if not os.path.exists(path):
        sys.exit(f"Missing input file: {path}\n"
                 f"Run this script from the repo root, and make sure the file exists.")


def to_float(tok):
    return float(tok.replace(",", ""))


def parse_fdm(path):
    need(path)
    fdm2d, fdm1d, section = [], [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "2D steady" in line:
                section = "2d"
                continue
            if "1D transient" in line:
                section = "1d"
                continue
            tok = line.split()
            if not tok or not re.fullmatch(r"\d+", tok[0]):
                continue
            try:
                vals = [to_float(t) for t in tok]
            except ValueError:
                continue
            if section == "2d" and len(vals) == 5:      # n, points, L2, Linf, time
                fdm2d.append((int(vals[0]), vals[1], vals[2]))
            elif section == "1d" and len(vals) == 4:    # nx, points, L2, time
                fdm1d.append((int(vals[0]), vals[1], vals[2]))
    if not fdm2d or not fdm1d:
        sys.exit(f"Could not parse FDM tables from {path}")
    return fdm2d, fdm1d


def parse_pinn1d(path):
    """Rows: (N_domain, total training points, L2 mean, L2 std)."""
    need(path)
    rows, inside = [], False
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("PINN"):
                inside = True
                continue
            if s.startswith("FDM"):
                break                      # ignore the old (flawed) FDM section
            if not inside:
                continue
            tok = s.split()
            if len(tok) == 4 and re.fullmatch(r"\d+", tok[0]):
                n = int(tok[0])
                n_bc_ic = max(50, n // 5)  # same rule as data_efficiency_study.py v2
                rows.append((n, n + 2 * n_bc_ic, float(tok[1]), float(tok[2])))
    if not rows:
        sys.exit(f"Could not parse PINN rows from {path}")
    return rows


def grab(path, pattern):
    need(path)
    with open(path, encoding="utf-8") as f:
        m = re.search(pattern, f.read())
    if not m:
        sys.exit(f"Pattern {pattern!r} not found in {path}")
    return float(m.group(1))


L2_PAT = r"L2 Relative Error\s*:\s*([0-9.]+[eE][+-]?\d+)"
PTS_PAT = r"Training points\s*:\s*(\d+)"

fdm2d, fdm1d = parse_fdm(FDM_TXT)
pinn1d = parse_pinn1d(PINN1D_TXT)

# ---------------------------------------------------------------------------
# Figure 1: 1D FDM refinement (observed order from a log-log fit)
# ---------------------------------------------------------------------------
nx = np.array([r[0] for r in fdm1d], float)
l2 = np.array([r[2] for r in fdm1d], float)
slope = np.polyfit(np.log(nx), np.log(l2), 1)[0]

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(nx, l2, "o-", lw=2, label="Explicit FDM (space-time L2)")
ax.loglog(nx, l2[0] * (nx / nx[0]) ** -2.0, "k--", lw=1, label="slope -2 reference")
ax.set_xlabel("grid points nx")
ax.set_ylabel("L2 relative error (all nodes, all time steps)")
ax.set_title(f"1D heat equation: FDM refinement\nobserved order = {-slope:.2f}")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
out1 = "01_heat_1d/results/fig_fdm_refinement_corrected.png"
plt.savefig(out1, dpi=150)
plt.close(fig)
print(f"Saved {out1}   (observed FDM order {-slope:.2f})")

# ---------------------------------------------------------------------------
# Figure 2: 1D efficiency, PINN (3-seed mean +/- std) vs FDM, same metric
# ---------------------------------------------------------------------------
fpts = np.array([r[1] for r in fdm1d], float)
fl2 = np.array([r[2] for r in fdm1d], float)
ppts = np.array([r[1] for r in pinn1d], float)
pmean = np.array([r[2] for r in pinn1d], float)
pstd = np.array([r[3] for r in pinn1d], float)

best = int(np.argmin(pmean))
j = int(np.argmin(np.abs(np.log(fl2 / pmean[best]))))   # FDM config with closest L2
ratio = fpts[j] / ppts[best]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.loglog(fpts, fl2, "s--", color="darkorange", lw=2, ms=8, label="FDM (explicit)")
ax.errorbar(ppts, pmean, yerr=pstd, fmt="o-", color="steelblue", lw=2, ms=7,
            capsize=4, label="PINN (3-seed mean ± 1 std)")
ax.annotate(
    f"PINN N={pinn1d[best][0]}: {int(ppts[best]):,} pts, L2={pmean[best]:.2e}\n"
    f"FDM nx={fdm1d[j][0]} (closest L2={fl2[j]:.2e}): {int(fpts[j]):,} pts\n"
    f"≈ {ratio:.0f}x fewer points (points are not compute cost)",
    xy=(ppts[best], pmean[best]), xytext=(1.5e2, 3e-5), fontsize=9,
    arrowprops=dict(arrowstyle="->", color="steelblue"),
    bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray"))
ax.set_xlabel("space-time points used")
ax.set_ylabel("L2 relative error (whole space-time domain)")
ax.set_xlim(8e1, 2e8)
ax.set_title("1D transient heat: PINN vs FDM on the same error metric")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper right")
plt.tight_layout()
out2 = "01_heat_1d/results/fig_efficiency_corrected.png"
plt.savefig(out2, dpi=150)
plt.close(fig)
print(f"Saved {out2}   (nearest-accuracy FDM nx={fdm1d[j][0]}, ratio {ratio:.1f}x)")

# ---------------------------------------------------------------------------
# Figure 3: 2D, direct-solve FDM vs PINN v1 / v2
# ---------------------------------------------------------------------------
p1_l2, p1_pts = grab(HEAT2D_V1, L2_PAT), grab(HEAT2D_V1, PTS_PAT)
p2_l2, p2_pts = grab(HEAT2D_V2, L2_PAT), grab(HEAT2D_V2, PTS_PAT)
gpts = np.array([r[1] for r in fdm2d], float)
gl2 = np.array([r[2] for r in fdm2d], float)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.loglog(gpts, gl2, "s--", color="darkorange", lw=2, ms=8,
          label="FDM (direct sparse solve)")
ax.loglog([p1_pts], [p1_l2], "o", color="lightsteelblue", ms=11,
          mec="k", label=f"PINN v1 (single seed)  L2={p1_l2:.2e}")
ax.loglog([p2_pts], [p2_l2], "o", color="steelblue", ms=11, mec="k",
          label=f"PINN v2 (single seed)  L2={p2_l2:.2e}")
ax.set_xlabel("grid / training points")
ax.set_ylabel("L2 relative error")
k = next((i for i, e in enumerate(gl2) if e <= p2_l2), len(gl2) - 1)
ax.set_title(f"2D Poisson on a square: direct-solve FDM reaches PINN v2's error\n"
             f"with {int(gpts[k]):,} points (PINN used {int(p2_pts):,})")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
plt.tight_layout()
out3 = "02_heat_2d/results/fig_2d_comparison_corrected.png"
plt.savefig(out3, dpi=150)
plt.close(fig)
print(f"Saved {out3}")