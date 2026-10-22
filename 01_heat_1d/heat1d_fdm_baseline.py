# heat1d_fdm_baseline.py
# Finite Difference Method (FDM) baseline for the 1D Heat Equation
# This is the CLASSICAL solution your PINN will be compared against
#
# PDE:  ∂T/∂t = α ∂²T/∂x²
# BC:   T(0,t) = 0,  T(1,t) = 0
# IC:   T(x,0) = sin(πx)
# Analytical: T(x,t) = exp(-α π² t) sin(πx)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

# ── 0. Output folder ────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════
alpha  = 0.4      # thermal diffusivity
L      = 1.0      # domain length
T_end  = 1.0      # total simulation time
nx     = 100      # spatial grid points
nt     = 10000    # time steps

dx = L / (nx - 1)
dt = T_end / nt
r  = alpha * dt / dx**2   # stability parameter — must be < 0.5

print("=" * 55)
print("  1D Heat Equation — Finite Difference Method (FDM)")
print("=" * 55)
print(f"  α (thermal diffusivity) = {alpha}")
print(f"  Grid points (nx)        = {nx}")
print(f"  Time steps  (nt)        = {nt}")
print(f"  dx                      = {dx:.5f}")
print(f"  dt                      = {dt:.6f}")
print(f"  Stability parameter r   = {r:.4f}  (must be < 0.5)")

if r >= 0.5:
    raise ValueError(f"UNSTABLE! r = {r:.4f} ≥ 0.5. Reduce dt or increase dx.")
else:
    print(f"  Stability check         ✓ STABLE\n")

# ═══════════════════════════════════════════════════════════════════════════
# 2. GRID SETUP
# ═══════════════════════════════════════════════════════════════════════════
x      = np.linspace(0, L, nx)
t_vals = np.linspace(0, T_end, nt + 1)

# ═══════════════════════════════════════════════════════════════════════════
# 3. INITIAL & BOUNDARY CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════
T = np.sin(np.pi * x)       # IC: T(x,0) = sin(πx)
T[0]  = 0.0                 # BC: T(0,t) = 0
T[-1] = 0.0                 # BC: T(1,t) = 0

# Store full solution history for plotting
# Storing every 100th step to save memory
store_every = 100
T_history   = [T.copy()]
t_stored    = [0.0]

# ═══════════════════════════════════════════════════════════════════════════
# 4. TIME MARCHING — EXPLICIT FDM
# ═══════════════════════════════════════════════════════════════════════════
# Scheme: T_new[i] = T[i] + r*(T[i+1] - 2*T[i] + T[i-1])
# This is the forward Euler + central difference discretization of:
# ∂T/∂t = α ∂²T/∂x²

print("  Running FDM time marching...")
for n in range(nt):
    T_new        = T.copy()
    T_new[1:-1]  = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
    T_new[0]     = 0.0      # enforce BC at every step
    T_new[-1]    = 0.0
    T            = T_new

    if (n + 1) % store_every == 0:
        T_history.append(T.copy())
        t_stored.append((n + 1) * dt)

T_history = np.array(T_history)   # shape: (n_stored, nx)
t_stored  = np.array(t_stored)
print(f"  Time marching complete. Stored {len(t_stored)} snapshots.\n")

# ═══════════════════════════════════════════════════════════════════════════
# 5. ANALYTICAL SOLUTION
# ═══════════════════════════════════════════════════════════════════════════
def analytical(x, t, alpha):
    """Exact solution: T(x,t) = exp(-α π² t) sin(πx)"""
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)

T_analytical_final = analytical(x, T_end, alpha)

# ═══════════════════════════════════════════════════════════════════════════
# 6. ERROR METRICS
# ═══════════════════════════════════════════════════════════════════════════
T_fdm_final = T_history[-1]

L2_error   = np.linalg.norm(T_fdm_final - T_analytical_final) / \
             np.linalg.norm(T_analytical_final)
Linf_error = np.max(np.abs(T_fdm_final - T_analytical_final))
RMSE       = np.sqrt(np.mean((T_fdm_final - T_analytical_final)**2))

print("─" * 55)
print("  ERROR METRICS at t = 1.0")
print("─" * 55)
print(f"  L2 Relative Error  : {L2_error:.6e}")
print(f"  L∞ Error           : {Linf_error:.6e}")
print(f"  RMSE               : {RMSE:.6e}")
print("─" * 55)
print()
print("  ← Save these numbers. Your PINN must beat or match them.")
print()

# ═══════════════════════════════════════════════════════════════════════════
# 7. PLOTS
# ═══════════════════════════════════════════════════════════════════════════

# ── Plot 1: Solution at multiple time slices ─────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
time_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
colors      = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

for t_target, color in zip(time_slices, colors):
    idx     = np.argmin(np.abs(t_stored - t_target))
    t_act   = t_stored[idx]
    T_fdm   = T_history[idx]
    T_exact = analytical(x, t_act, alpha)
    ax.plot(x, T_fdm,   color=color, lw=2,   label=f"FDM  t={t_act:.2f}")
    ax.plot(x, T_exact, color=color, lw=1.5, linestyle='--', alpha=0.7,
            label=f"Exact t={t_act:.2f}")

ax.set_xlabel("x", fontsize=13)
ax.set_ylabel("T(x, t)", fontsize=13)
ax.set_title("1D Heat Equation — FDM vs Analytical Solution", fontsize=14)
ax.legend(fontsize=8, ncol=2, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/fdm_time_slices.png", dpi=150)
plt.show()
print("  Saved: results/fdm_time_slices.png")

# ── Plot 2: 2D heatmap — T(x,t) ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# FDM solution
im0 = axes[0].imshow(T_history.T, origin='lower', aspect='auto',
                      extent=[0, T_end, 0, L], cmap='hot')
axes[0].set_title("FDM Solution T(x,t)", fontsize=12)
axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
plt.colorbar(im0, ax=axes[0], label="Temperature")

# Analytical solution — build full grid
T_exact_grid = np.array([analytical(x, t, alpha) for t in t_stored])
im1 = axes[1].imshow(T_exact_grid.T, origin='lower', aspect='auto',
                      extent=[0, T_end, 0, L], cmap='hot')
axes[1].set_title("Analytical Solution T(x,t)", fontsize=12)
axes[1].set_xlabel("t"); axes[1].set_ylabel("x")
plt.colorbar(im1, ax=axes[1], label="Temperature")

# Absolute error
error_grid = np.abs(T_history - T_exact_grid)
im2 = axes[2].imshow(error_grid.T, origin='lower', aspect='auto',
                      extent=[0, T_end, 0, L], cmap='Reds')
axes[2].set_title("Absolute Error |FDM - Exact|", fontsize=12)
axes[2].set_xlabel("t"); axes[2].set_ylabel("x")
plt.colorbar(im2, ax=axes[2], label="Error")

plt.suptitle("1D Heat Equation — FDM Baseline", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("results/fdm_heatmap.png", dpi=150)
plt.show()
print("  Saved: results/fdm_heatmap.png")

# ── Plot 3: Final time comparison + error ────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

ax1.plot(x, T_fdm_final,        lw=2.5, label="FDM solution",       color='steelblue')
ax1.plot(x, T_analytical_final, lw=2,   label="Analytical solution",
         color='darkorange', linestyle='--')
ax1.set_ylabel("T(x, 1.0)", fontsize=12)
ax1.set_title("Solution at t = 1.0", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2.plot(x, np.abs(T_fdm_final - T_analytical_final),
         color='crimson', lw=2)
ax2.set_xlabel("x", fontsize=12)
ax2.set_ylabel("|Error|", fontsize=12)
ax2.set_title(f"Absolute Error  (L2={L2_error:.2e}, L∞={Linf_error:.2e})",
              fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig("results/fdm_final_comparison.png", dpi=150)
plt.show()
print("  Saved: results/fdm_final_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# 8. GRID REFINEMENT STUDY
# ═══════════════════════════════════════════════════════════════════════════
# This shows how FDM error changes with grid size
# Your PINN comparison will use this data

print("\n" + "─" * 55)
print("  GRID REFINEMENT STUDY")
print("─" * 55)
print(f"  {'nx':>6}  {'nt':>8}  {'L2 Error':>12}  {'L∞ Error':>12}")
print("  " + "-" * 45)

nx_list    = [20, 50, 100, 200, 500]
L2_list    = []
Linf_list  = []

for nx_i in nx_list:
    dx_i = L / (nx_i - 1)
    dt_i = 0.4 * dx_i**2 / alpha    # auto-set dt for stability (r=0.4)
    nt_i = int(T_end / dt_i) + 1
    r_i  = alpha * dt_i / dx_i**2

    x_i  = np.linspace(0, L, nx_i)
    T_i  = np.sin(np.pi * x_i)

    for _ in range(nt_i):
        T_new       = T_i.copy()
        T_new[1:-1] = T_i[1:-1] + r_i * (T_i[2:] - 2*T_i[1:-1] + T_i[:-2])
        T_new[0]    = T_new[-1] = 0.0
        T_i         = T_new

    T_ex_i   = analytical(x_i, T_end, alpha)
    L2_i     = np.linalg.norm(T_i - T_ex_i) / np.linalg.norm(T_ex_i)
    Linf_i   = np.max(np.abs(T_i - T_ex_i))
    L2_list.append(L2_i)
    Linf_list.append(Linf_i)
    print(f"  {nx_i:>6}  {nt_i:>8}  {L2_i:>12.4e}  {Linf_i:>12.4e}")

# Plot refinement study
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(nx_list, L2_list,   'o-', lw=2, color='steelblue', label='L2 Error')
ax.loglog(nx_list, Linf_list, 's-', lw=2, color='crimson',   label='L∞ Error')
ax.set_xlabel("Number of grid points (nx)", fontsize=12)
ax.set_ylabel("Error", fontsize=12)
ax.set_title("FDM Grid Refinement Study\n",
             fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig("results/fdm_refinement.png", dpi=150)
plt.show()
print("\n  Saved: results/fdm_refinement.png")

print("\n" + "=" * 55)
print("  FDM BASELINE COMPLETE")
print("  These are your benchmark numbers for the PINN comparison:")
print(f"  nx=100  →  L2 = {L2_list[2]:.4e},  L∞ = {Linf_list[2]:.4e}")
print("=" * 55)
