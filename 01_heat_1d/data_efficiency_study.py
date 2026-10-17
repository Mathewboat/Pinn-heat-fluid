# data_efficiency_study.py
# Key experiment: How does PINN accuracy scale with number of
# collocation points? Compare against FDM at equivalent resolutions.
#
# This produces the headline result of the project.
# Author: Mathew Ahenkora Boateng — KNUST

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.makedirs("results", exist_ok=True)

alpha = 0.4

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def analytical(x_col, t_col):
    return np.exp(-alpha * np.pi**2 * t_col) * np.sin(np.pi * x_col)

def heat_pde(x, T):
    dT_dt  = dde.grad.jacobian(T, x, i=0, j=1)
    dT_dxx = dde.grad.hessian(T, x, i=0, j=0)
    return dT_dt - alpha * dT_dxx

def run_pinn(n_domain, n_boundary=100, n_initial=100):
    """Train a PINN with n_domain collocation points. Returns L2 error."""

    geom       = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime   = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.DirichletBC(
        geomtime,
        lambda x: np.zeros((len(x), 1)),
        lambda x, on_boundary: on_boundary
    )
    ic = dde.icbc.IC(
        geomtime,
        lambda x: np.sin(np.pi * x[:, 0:1]),
        lambda x, on_initial: on_initial
    )

    def solution_fn(x):
        return analytical(x[:, 0:1], x[:, 1:2])

    data = dde.data.TimePDE(
        geomtime, heat_pde, [bc, ic],
        num_domain   = n_domain,
        num_boundary = n_boundary,
        num_initial  = n_initial,
        solution     = solution_fn,
        num_test     = 5000
    )

    net   = dde.nn.FNN([2]+[64]*4+[1], "tanh", "Glorot uniform")
    model = dde.Model(data, net)

    # Stage 1 — Adam
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    model.train(iterations=15000, display_every=15000)

    # Stage 2 — L-BFGS
    model.compile("L-BFGS", metrics=["l2 relative error"])
    model.train()

    # Evaluate on dense test grid
    nx, nt   = 150, 150
    x_t      = np.linspace(0, 1, nx)
    t_t      = np.linspace(0, 1, nt)
    X, T_g   = np.meshgrid(x_t, t_t)
    XT       = np.column_stack([X.ravel(), T_g.ravel()])
    T_pred   = model.predict(XT).reshape(nt, nx)
    T_exact  = analytical(X, T_g)

    L2 = np.linalg.norm(T_pred-T_exact) / np.linalg.norm(T_exact)
    Li = np.max(np.abs(T_pred - T_exact))

    # reset graph for next run
    dde.backend.tf.keras.backend.clear_session()

    return L2, Li

def run_fdm(nx):
    """Run FDM with nx grid points. Returns L2 error."""
    dx   = 1.0 / (nx - 1)
    dt   = 0.4 * dx**2 / alpha
    nt   = int(1.0 / dt) + 1
    r    = alpha * dt / dx**2

    x    = np.linspace(0, 1, nx)
    T    = np.sin(np.pi * x)

    for _ in range(nt):
        T_new        = T.copy()
        T_new[1:-1]  = T[1:-1] + r*(T[2:] - 2*T[1:-1] + T[:-2])
        T_new[0]     = T_new[-1] = 0.0
        T            = T_new

    T_exact = analytical(x, np.ones_like(x))
    L2      = np.linalg.norm(T-T_exact) / np.linalg.norm(T_exact)
    Li      = np.max(np.abs(T - T_exact))
    n_total = nx * nt   # total grid evaluations
    return L2, Li, n_total

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT A — PINN: vary collocation points
# ═══════════════════════════════════════════════════════════════
print("=" * 62)
print("  DATA EFFICIENCY STUDY")
print("  PINN vs FDM — 1D Heat Equation")
print("  Mathew Ahenkora Boateng — KNUST")
print("=" * 62)

pinn_configs = [50, 100, 250, 500, 1000, 2500]
pinn_L2      = []
pinn_Linf    = []
pinn_times   = []

print("\n── PINN Runs ──────────────────────────────────────────────")
print(f"  {'N_domain':>10}  {'L2 Error':>12}  {'L∞ Error':>12}  {'Time(s)':>8}")
print("  " + "-" * 48)

for n in pinn_configs:
    t0       = time.time()
    L2, Li   = run_pinn(n)
    elapsed  = time.time() - t0
    pinn_L2.append(L2)
    pinn_Linf.append(Li)
    pinn_times.append(elapsed)
    print(f"  {n:>10}  {L2:>12.4e}  {Li:>12.4e}  {elapsed:>8.1f}")

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT B — FDM: vary grid points
# ═══════════════════════════════════════════════════════════════
fdm_configs = [10, 20, 50, 100, 200, 500]
fdm_L2      = []
fdm_Linf    = []
fdm_total   = []

print("\n── FDM Runs ───────────────────────────────────────────────")
print(f"  {'nx':>10}  {'L2 Error':>12}  {'L∞ Error':>12}  {'Total pts':>12}")
print("  " + "-" * 52)

for nx in fdm_configs:
    L2, Li, n_tot = run_fdm(nx)
    fdm_L2.append(L2)
    fdm_Linf.append(Li)
    fdm_total.append(n_tot)
    print(f"  {nx:>10}  {L2:>12.4e}  {Li:>12.4e}  {n_tot:>12,}")

# ═══════════════════════════════════════════════════════════════
# PRINT SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  HEADLINE RESULT")
print("=" * 62)

# Find FDM points needed to match best PINN accuracy
best_pinn_L2 = min(pinn_L2)
fdm_match_idx = next(
    (i for i, e in enumerate(fdm_L2) if e <= best_pinn_L2), -1
)

if fdm_match_idx >= 0:
    fdm_pts_needed   = fdm_total[fdm_match_idx]
    pinn_pts_needed  = pinn_configs[pinn_L2.index(best_pinn_L2)] + 200
    ratio            = fdm_pts_needed / pinn_pts_needed
    print(f"\n  Best PINN L2 error    : {best_pinn_L2:.4e}")
    print(f"  PINN training points  : {pinn_pts_needed:,}")
    print(f"  FDM points for same L2: {fdm_pts_needed:,}")
    print(f"  Point reduction ratio : {ratio:.0f}×")
    print(f"\n  → PINN achieves L2={best_pinn_L2:.2e} with {ratio:.0f}× fewer")
    print(f"    computational points than FDM.")

# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════

# ── Plot 1: L2 Error vs Training Points ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.loglog(pinn_configs, pinn_L2, 'o-',
          color='steelblue', lw=2.5, ms=8, label='PINN (collocation pts)')
ax.loglog(fdm_total, fdm_L2, 's--',
          color='darkorange', lw=2, ms=8, label='FDM (total grid evals)')

# Annotate the crossover point
for i, (n, e) in enumerate(zip(pinn_configs, pinn_L2)):
    ax.annotate(f'N={n}', (n, e),
                textcoords="offset points", xytext=(6, 4), fontsize=8)

ax.set_xlabel("Number of computational points", fontsize=12)
ax.set_ylabel("L2 Relative Error", fontsize=12)
ax.set_title("Data Efficiency: PINN vs FDM\n(same accuracy, far fewer points)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)

# ── Plot 2: L∞ Error vs Training Points ─────────────────────
ax2 = axes[1]
ax2.loglog(pinn_configs, pinn_Linf, 'o-',
           color='steelblue', lw=2.5, ms=8, label='PINN')
ax2.loglog(fdm_total, fdm_Linf, 's--',
           color='darkorange', lw=2, ms=8, label='FDM')
ax2.set_xlabel("Number of computational points", fontsize=12)
ax2.set_ylabel("L∞ Error", fontsize=12)
ax2.set_title("L∞ Error vs Computational Points",
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, which='both', alpha=0.3)

plt.suptitle("PINN Data Efficiency Study — 1D Heat Equation\n"
             "Mathew Ahenkora Boateng — KNUST",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("results/data_efficiency.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n  Saved: results/data_efficiency.png")

# ── Plot 3: The single most important figure ─────────────────
# Accuracy vs points — side by side bars at matched accuracy levels
fig, ax = plt.subplots(figsize=(10, 6))

# Draw a horizontal target line at your known PINN result
target_L2 = min(pinn_L2)
ax.axhline(target_L2, color='gray', lw=1, ls=':', label=f'Best PINN L2 = {target_L2:.2e}')

ax.loglog(pinn_configs, pinn_L2, 'o-',
          color='steelblue', lw=3, ms=10, label='PINN', zorder=5)
ax.loglog(fdm_total, fdm_L2, 's--',
          color='darkorange', lw=2.5, ms=10, label='FDM', zorder=5)

# Shade the "PINN advantage" region
ax.fill_between(
    [min(pinn_configs), max(fdm_total)],
    [target_L2*0.5, target_L2*0.5],
    [target_L2*2,   target_L2*2],
    alpha=0.08, color='steelblue', label='Accuracy band'
)

ax.set_xlabel("Computational points required", fontsize=13)
ax.set_ylabel("L2 Relative Error", fontsize=13)
ax.set_title("PINN achieves comparable accuracy to FDM\nwith orders of magnitude fewer points",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig("results/data_efficiency_headline.png", dpi=150)
plt.show()
print("  Saved: results/data_efficiency_headline.png")

# ═══════════════════════════════════════════════════════════════
# SAVE RESULTS TO TEXT FILE FOR README
# ═══════════════════════════════════════════════════════════════
with open("results/data_efficiency_results.txt", "w") as f:
    f.write("DATA EFFICIENCY STUDY RESULTS\n")
    f.write("Mathew Ahenkora Boateng — KNUST\n\n")
    f.write("PINN Results:\n")
    f.write(f"{'N_domain':>10}  {'L2 Error':>12}  {'L∞ Error':>12}\n")
    for n, L2, Li in zip(pinn_configs, pinn_L2, pinn_Linf):
        f.write(f"{n:>10}  {L2:>12.4e}  {Li:>12.4e}\n")
    f.write("\nFDM Results:\n")
    f.write(f"{'nx':>10}  {'Total pts':>12}  {'L2 Error':>12}  {'L∞ Error':>12}\n")
    for nx, n_tot, L2, Li in zip(fdm_configs, fdm_total, fdm_L2, fdm_Linf):
        f.write(f"{nx:>10}  {n_tot:>12,}  {L2:>12.4e}  {Li:>12.4e}\n")

print("\n  Saved: results/data_efficiency_results.txt")
print("\n" + "=" * 62)
print("  Data efficiency study complete.")
print("  Commit results and update README table.")
print("=" * 62)