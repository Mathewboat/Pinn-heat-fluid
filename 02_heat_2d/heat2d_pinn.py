# heat2d_pinn.py  — v2: corrected training strategy
# Fixes applied:
#   1. Exponential LR decay — prevents Adam oscillation after step 6000
#   2. 30,000 Adam iterations — better minimum before L-BFGS
#   3. Wider network [2,64x6,1] — more capacity for 2D surface
#   4. Output files use _v2 suffix — preserves v1 results
#   5. UTF-8 encoding on all file writes
#
# PDE:   d²T/dx² + d²T/dy² = -Q(x,y)    (x,y) in [0,1]²
# BC:    T = 0 on all four boundaries     (Dirichlet)
# Source: Q(x,y) = sin(pi*x)*sin(pi*y)
# Exact:  T(x,y) = sin(pi*x)*sin(pi*y) / (2*pi²)
#
# Author: Mathew Ahenkora Boateng — KNUST
# GitHub: https://github.com/Mathewboat/Pinn-heat-fluid

import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.makedirs("results", exist_ok=True)

# fixed seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
dde.config.set_random_seed(42)

print("=" * 62)
print("  Physics-Informed Neural Network")
print("  2D Steady-State Heat Conduction  [v2 — corrected]")
print("  Mathew Ahenkora Boateng — KNUST")
print("=" * 62)
print()
print("  PDE    : d2T/dx2 + d2T/dy2 = -sin(pi*x)sin(pi*y)")
print("  Domain : (x,y) in [0,1]^2")
print("  BC     : T = 0 on all boundaries (Dirichlet)")
print("  Exact  : T(x,y) = sin(pi*x)sin(pi*y) / (2*pi^2)")
print()
print("  v2 changes vs v1:")
print("  - LR schedule: exponential decay 1e-3 -> ~1e-4 over 30k steps")
print("  - Adam iterations: 30,000 (was 20,000)")
print("  - Network depth: 6 hidden layers (was 4)")
print("  - All output files saved with _v2 suffix")
print()

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — ANALYTICAL SOLUTION
# ═══════════════════════════════════════════════════════════════
def analytical(x):
    """
    Exact: T(x,y) = sin(pi*x)*sin(pi*y) / (2*pi^2)
    Input:  (N,2)  col0=x, col1=y
    Output: (N,1)
    """
    return (np.sin(np.pi * x[:, 0:1]) *
            np.sin(np.pi * x[:, 1:2])) / (2.0 * np.pi**2)

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — PDE RESIDUAL
# d2T/dx2 + d2T/dy2 + sin(pi*x)*sin(pi*y) = 0
# ═══════════════════════════════════════════════════════════════
def poisson_pde(x, T):
    """
    Residual of 2D Poisson equation.
    Both second derivatives computed by automatic differentiation.
    tf.sin used for source term — required inside TF graph.
    """
    dT_xx = dde.grad.hessian(T, x, i=0, j=0)   # d2T/dx2
    dT_yy = dde.grad.hessian(T, x, i=1, j=1)   # d2T/dy2
    Q     = tf.sin(np.pi * x[:, 0:1]) * tf.sin(np.pi * x[:, 1:2])
    return dT_xx + dT_yy + Q

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — GEOMETRY AND BC
# ═══════════════════════════════════════════════════════════════
geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[1, 1])

bc = dde.icbc.DirichletBC(
    geom,
    lambda x: np.zeros((len(x), 1)),
    lambda x, on_boundary: on_boundary
)

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — DATA
# ═══════════════════════════════════════════════════════════════
N_DOMAIN   = 2500
N_BOUNDARY = 400
N_TEST     = 10000

print(f"  Collocation points : {N_DOMAIN}")
print(f"  Boundary points    : {N_BOUNDARY}")
print(f"  Test points        : {N_TEST}")
print(f"  Total training pts : {N_DOMAIN + N_BOUNDARY}\n")

data = dde.data.PDE(
    geom,
    poisson_pde,
    bc,
    num_domain   = N_DOMAIN,
    num_boundary = N_BOUNDARY,
    solution     = analytical,
    num_test     = N_TEST
)

# ═══════════════════════════════════════════════════════════════
# SECTION 5 — NETWORK ARCHITECTURE
# Fix 3: 6 hidden layers instead of 4 — more capacity for 2D
# ═══════════════════════════════════════════════════════════════
LAYER_SIZE  = [2] + [64] * 6 + [1]   # deeper than v1
ACTIVATION  = "tanh"
INITIALIZER = "Glorot uniform"

net   = dde.nn.FNN(LAYER_SIZE, ACTIVATION, INITIALIZER)
model = dde.Model(data, net)

n_params = sum(
    (LAYER_SIZE[i] * LAYER_SIZE[i+1] + LAYER_SIZE[i+1])
    for i in range(len(LAYER_SIZE)-1)
)
print(f"  Architecture : {LAYER_SIZE}")
print(f"  Activation   : {ACTIVATION}")
print(f"  Parameters   : {n_params:,}  (v1 had 12,737)\n")

# ═══════════════════════════════════════════════════════════════
# SECTION 6 — TRAINING STAGE 1: ADAM WITH LR DECAY
# Fix 1: exponential decay prevents oscillation after step 6000
# Fix 2: 30,000 iterations instead of 20,000
# ═══════════════════════════════════════════════════════════════
print("─" * 62)
print("  STAGE 1 — Adam with Exponential LR Decay (30,000 iters)")
print("─" * 62)
print("  LR: 1e-3 decaying by 0.95 every 2000 steps -> ~2.1e-4")
print("  This prevents the oscillation seen in v1 after step 6000\n")

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 1e-3,
    decay_steps           = 2000,
    decay_rate            = 0.95,
    staircase             = True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer, metrics=["l2 relative error"])

t0 = time.time()
losshistory, train_state = model.train(
    iterations    = 30000,
    display_every = 3000
)
t_adam = time.time() - t0
print(f"\n  Adam time    : {t_adam:.1f}s")

# ═══════════════════════════════════════════════════════════════
# SECTION 7 — TRAINING STAGE 2: L-BFGS
# Now starts from a much better Adam minimum
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 62)
print("  STAGE 2 — L-BFGS Fine-tuning")
print("─" * 62)

model.compile("L-BFGS", metrics=["l2 relative error"])

t0 = time.time()
losshistory, train_state = model.train()
t_lbfgs = time.time() - t0

print(f"\n  L-BFGS time  : {t_lbfgs:.1f}s")
print(f"  Total time   : {t_adam + t_lbfgs:.1f}s")

# ═══════════════════════════════════════════════════════════════
# SECTION 8 — EVALUATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  RESULTS")
print("=" * 62)

n_eval   = 200
x_eval   = np.linspace(0, 1, n_eval)
y_eval   = np.linspace(0, 1, n_eval)
X, Y     = np.meshgrid(x_eval, y_eval)
XY       = np.column_stack([X.ravel(), Y.ravel()])

T_pred   = model.predict(XY).reshape(n_eval, n_eval)
T_exact  = (np.sin(np.pi*X) * np.sin(np.pi*Y)) / (2.0 * np.pi**2)

error    = T_pred - T_exact
L2_rel   = np.linalg.norm(error) / np.linalg.norm(T_exact)
Linf     = np.max(np.abs(error))
RMSE     = np.sqrt(np.mean(error**2))
R2       = 1 - np.sum(error**2) / np.sum(
           (T_exact - T_exact.mean())**2)

print(f"\n  L2 Relative Error : {L2_rel:.4e}")
print(f"  L-inf Error       : {Linf:.4e}")
print(f"  RMSE              : {RMSE:.4e}")
print(f"  R2 Score          : {R2:.6f}")
print(f"\n  Training points   : {N_DOMAIN + N_BOUNDARY:,}")
print(f"  FDM equivalent    : ~{n_eval**2:,} pts (nx=ny={n_eval})")
print(f"  Point reduction   : ~{n_eval**2//(N_DOMAIN+N_BOUNDARY)}x")

# ── v1 vs v2 comparison ──────────────────────────────────────
v1_L2 = 3.8018e-03
print(f"\n  v1 L2 error       : {v1_L2:.4e}")
print(f"  v2 L2 error       : {L2_rel:.4e}")
improvement = (v1_L2 - L2_rel) / v1_L2 * 100
if improvement > 0:
    print(f"  Improvement       : {improvement:.1f}% better than v1")
else:
    print(f"  Change            : {abs(improvement):.1f}% vs v1")

# ═══════════════════════════════════════════════════════════════
# SECTION 9 — FDM BASELINE
# Reuse v1 result — no need to re-run the slow Gauss-Seidel
# ═══════════════════════════════════════════════════════════════
L2_fdm  = 1.9493e-02
Li_fdm  = 9.8760e-04
fdm_pts = 10000
t_fdm   = 44.9

print(f"\n  FDM L2 (from v1)  : {L2_fdm:.4e}  (not re-run — same problem)")
print(f"  PINN vs FDM       : {L2_fdm/L2_rel:.1f}x more accurate")

# ═══════════════════════════════════════════════════════════════
# SECTION 10 — PLOTS  (all saved with _v2 suffix)
# ═══════════════════════════════════════════════════════════════

# ── Plot 1: Three-panel comparison ───────────────────────────
fig = plt.figure(figsize=(16, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

vmin = T_exact.min()
vmax = T_exact.max()

ax0 = fig.add_subplot(gs[0])
im0 = ax0.contourf(X, Y, T_pred, levels=50,
                   cmap='hot', vmin=vmin, vmax=vmax)
ax0.set_title("PINN Solution T(x,y)", fontsize=13,
              fontweight='bold')
ax0.set_xlabel("x"); ax0.set_ylabel("y")
ax0.set_aspect('equal')
plt.colorbar(im0, ax=ax0, label="Temperature")

ax1 = fig.add_subplot(gs[1])
im1 = ax1.contourf(X, Y, T_exact, levels=50,
                   cmap='hot', vmin=vmin, vmax=vmax)
ax1.set_title("Analytical Solution T(x,y)", fontsize=13,
              fontweight='bold')
ax1.set_xlabel("x"); ax1.set_ylabel("y")
ax1.set_aspect('equal')
plt.colorbar(im1, ax=ax1, label="Temperature")

ax2 = fig.add_subplot(gs[2])
im2 = ax2.contourf(X, Y, np.abs(error), levels=50, cmap='Reds')
ax2.set_title(f"|Error|   L2={L2_rel:.2e}", fontsize=13,
              fontweight='bold')
ax2.set_xlabel("x"); ax2.set_ylabel("y")
ax2.set_aspect('equal')
plt.colorbar(im2, ax=ax2, label="Absolute Error")

plt.suptitle(
    "PINN v2 — 2D Steady-State Heat Equation  |  "
    "Mathew Ahenkora Boateng — KNUST",
    fontsize=13, fontweight='bold', y=1.02
)
plt.savefig("results/heat2d_comparison_v2.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("\n  Saved: results/heat2d_comparison_v2.png")

# ── Plot 2: 3D surface ────────────────────────────────────────
fig = plt.figure(figsize=(14, 5))

ax3d_p = fig.add_subplot(121, projection='3d')
ax3d_p.plot_surface(X, Y, T_pred, cmap='hot',
                    linewidth=0, antialiased=True, alpha=0.9)
ax3d_p.set_title("PINN Solution v2 (3D)", fontsize=12,
                 fontweight='bold')
ax3d_p.set_xlabel("x"); ax3d_p.set_ylabel("y")
ax3d_p.set_zlabel("T")

ax3d_e = fig.add_subplot(122, projection='3d')
ax3d_e.plot_surface(X, Y, T_exact, cmap='hot',
                    linewidth=0, antialiased=True, alpha=0.9)
ax3d_e.set_title("Analytical Solution (3D)", fontsize=12,
                 fontweight='bold')
ax3d_e.set_xlabel("x"); ax3d_e.set_ylabel("y")
ax3d_e.set_zlabel("T")

plt.suptitle("2D Heat Equation v2 — 3D Surface Comparison",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("results/heat2d_3d_surface_v2.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: results/heat2d_3d_surface_v2.png")

# ── Plot 3: Cross-sections ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

y_mid = n_eval // 2
axes[0].plot(x_eval, T_pred[y_mid, :],
             lw=2.5, color='steelblue', label='PINN v2')
axes[0].plot(x_eval, T_exact[y_mid, :],
             lw=2, ls='--', color='darkorange',
             label='Analytical')
axes[0].set_xlabel("x", fontsize=12)
axes[0].set_ylabel("T(x, y=0.5)", fontsize=12)
axes[0].set_title("Cross-section at y = 0.5", fontsize=12,
                  fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

x_mid = n_eval // 2
axes[1].plot(y_eval, T_pred[:, x_mid],
             lw=2.5, color='steelblue', label='PINN v2')
axes[1].plot(y_eval, T_exact[:, x_mid],
             lw=2, ls='--', color='darkorange',
             label='Analytical')
axes[1].set_xlabel("y", fontsize=12)
axes[1].set_ylabel("T(x=0.5, y)", fontsize=12)
axes[1].set_title("Cross-section at x = 0.5", fontsize=12,
                  fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle(
    "2D Heat Equation v2 — Cross-Section Comparison\n"
    "PINN vs Analytical",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig("results/heat2d_cross_sections_v2.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: results/heat2d_cross_sections_v2.png")

# ── Plot 4: v1 vs v2 loss and error comparison ───────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Loss history
steps      = np.array(losshistory.steps)
loss_train = np.array(losshistory.loss_train)

ax = axes[0]
if loss_train.ndim > 1 and loss_train.shape[1] > 1:
    labels = ['PDE Residual', 'Boundary Condition']
    colors = ['steelblue', 'darkorange']
    for i, (lbl, col) in enumerate(
            zip(labels[:loss_train.shape[1]], colors)):
        ax.semilogy(steps, loss_train[:, i],
                    lw=2, label=lbl, color=col)
else:
    ax.semilogy(steps, loss_train.ravel(),
                lw=2, color='steelblue', label='Total Loss')

ax.set_xlabel("Iterations", fontsize=12)
ax.set_ylabel("Loss (log scale)", fontsize=12)
ax.set_title("v2 Training Loss\n(LR decay prevents oscillation)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)

# v1 vs v2 bar comparison
ax2 = axes[1]
methods  = ['FDM\n(nx=100)', 'PINN v1\n(20k Adam, 4 layers)',
            'PINN v2\n(30k Adam+LR, 6 layers)']
L2_vals  = [L2_fdm, v1_L2, L2_rel]
bar_cols = ['darkorange', 'lightsteelblue', 'steelblue']

bars = ax2.bar(methods, L2_vals, color=bar_cols,
               edgecolor='black', linewidth=0.8)
ax2.set_ylabel("L2 Relative Error", fontsize=12)
ax2.set_title("FDM vs PINN v1 vs PINN v2\nL2 Error Comparison",
              fontsize=12, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, which='both', alpha=0.3, axis='y')

for bar, val in zip(bars, L2_vals):
    ax2.text(bar.get_x() + bar.get_width()/2,
             val * 1.15,
             f'{val:.2e}',
             ha='center', va='bottom', fontsize=10,
             fontweight='bold')

plt.tight_layout()
plt.savefig("results/heat2d_v1_v2_comparison.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: results/heat2d_v1_v2_comparison.png")

# ── Plot 5: Error field detail ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im_err = axes[0].contourf(X, Y, np.abs(error),
                           levels=50, cmap='Reds')
axes[0].set_title(f"v2 Absolute Error\n"
                  f"L2={L2_rel:.2e}  Linf={Linf:.2e}",
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
axes[0].set_aspect('equal')
plt.colorbar(im_err, ax=axes[0])

# Error along diagonal x=y
diag_pts = np.linspace(0, 1, 200)
XY_diag  = np.column_stack([diag_pts, diag_pts])
T_diag_p = model.predict(XY_diag).ravel()
T_diag_e = (np.sin(np.pi*diag_pts) *
             np.sin(np.pi*diag_pts)) / (2*np.pi**2)
axes[1].plot(diag_pts, T_diag_p,
             lw=2.5, color='steelblue', label='PINN v2')
axes[1].plot(diag_pts, T_diag_e,
             lw=2, ls='--', color='darkorange', label='Exact')
axes[1].set_xlabel("x = y (diagonal)", fontsize=12)
axes[1].set_ylabel("T(x,x)", fontsize=12)
axes[1].set_title("Solution Along Diagonal x=y",
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle("2D Heat Equation v2 — Error Analysis",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("results/heat2d_error_analysis_v2.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: results/heat2d_error_analysis_v2.png")

# ═══════════════════════════════════════════════════════════════
# SECTION 11 — SAVE MODEL AND SUMMARY
# ═══════════════════════════════════════════════════════════════
model.save("results/heat2d_model_v2")

with open("results/heat2d_summary_v2.txt", "w",
          encoding="utf-8") as f:
    f.write("2D HEAT EQUATION — PINN v2 RESULTS\n")
    f.write("Mathew Ahenkora Boateng — KNUST\n")
    f.write("Fixes: LR decay, 30k Adam, 6 hidden layers\n\n")
    f.write(f"Architecture      : {LAYER_SIZE}\n")
    f.write(f"Parameters        : {n_params:,}\n")
    f.write(f"Adam iterations   : 30,000\n")
    f.write(f"LR schedule       : ExponentialDecay 1e-3->~2e-4\n\n")
    f.write(f"L2 Relative Error : {L2_rel:.4e}\n")
    f.write(f"L-inf Error       : {Linf:.4e}\n")
    f.write(f"RMSE              : {RMSE:.4e}\n")
    f.write(f"R2 Score          : {R2:.6f}\n\n")
    f.write(f"Training points   : {N_DOMAIN + N_BOUNDARY}\n")
    f.write(f"FDM points        : {fdm_pts}\n")
    f.write(f"FDM L2 error      : {L2_fdm:.4e}\n")
    f.write(f"PINN vs FDM       : {L2_fdm/L2_rel:.1f}x more accurate\n")
    f.write(f"Total train time  : {t_adam+t_lbfgs:.1f}s\n\n")
    f.write(f"v1 L2 error       : {v1_L2:.4e}\n")
    f.write(f"v2 L2 error       : {L2_rel:.4e}\n")
    improvement = (v1_L2 - L2_rel) / v1_L2 * 100
    f.write(f"Improvement       : {improvement:.1f}%\n")

print("\n  Saved: results/heat2d_summary_v2.txt")

print("\n" + "=" * 62)
print("  FINAL SUMMARY — 2D Heat Equation v2")
print("=" * 62)
print(f"\n  {'Metric':<28} {'PINN v2':>10}  "
      f"{'PINN v1':>10}  {'FDM':>10}")
print("  " + "-" * 62)
print(f"  {'L2 Relative Error':<28} {L2_rel:>10.4e}  "
      f"{v1_L2:>10.4e}  {L2_fdm:>10.4e}")
print(f"  {'L-inf Error':<28} {Linf:>10.4e}  "
      f"{'—':>10}  {Li_fdm:>10.4e}")
print(f"  {'R2 Score':<28} {R2:>10.6f}  "
      f"{'0.999959':>10}  {'N/A':>10}")
print(f"  {'Training points':<28} "
      f"{N_DOMAIN+N_BOUNDARY:>10,}  "
      f"{'2,900':>10}  {fdm_pts:>10,}")
print(f"  {'Train time (s)':<28} "
      f"{t_adam+t_lbfgs:>10.1f}  "
      f"{'1292.3':>10}  {t_fdm:>10.1f}")
print()

improvement = (v1_L2 - L2_rel) / v1_L2 * 100
if improvement > 0:
    print(f"  v2 is {improvement:.1f}% more accurate than v1")
print(f"  v2 is {L2_fdm/L2_rel:.1f}x more accurate than FDM")
print()
print("  Commit results and update README.")
print("  Next: Burgers equation (Days 19-24)")
print("=" * 62)