# heat1d_pinn.py
# Physics-Informed Neural Network — 1D Transient Heat Equation
#
# PDE:  ∂T/∂t = α ∂²T/∂x²
# BC:   T(0,t) = 0,  T(1,t) = 0
# IC:   T(x,0) = sin(πx)
# Exact: T(x,t) = exp(-α π² t) sin(πx)
#
# Author: Mathew Ahenkora Boateng
# Institution: KNUST
# Project: Physics-Informed Neural Networks for Heat & Fluid Flow
# GitHub: https://github.com/Mathewboat/Pinn-heat-fluid

import deepxde as dde
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time

# suppress TF info logs — keeps terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("  Physics-Informed Neural Network")
print("  1D Transient Heat Conduction")
print("  Mathew Ahenkora Boateng — KNUST")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — PHYSICS PARAMETERS
# ═══════════════════════════════════════════════════════════════
# Using real engineering material: Carbon Steel
# α = 1.17e-5 m²/s  (dimensional)
# Non-dimensionalized → α* = 0.4 for training stability
# See README for full dimensional analysis

alpha = 0.4    # non-dimensional thermal diffusivity

print(f"\n  Material    : Carbon Steel (non-dimensionalized)")
print(f"  α (nd)      : {alpha}")
print(f"  Domain      : x ∈ [0,1],  t ∈ [0,1]")
print(f"  BC          : T(0,t) = T(1,t) = 0  (Dirichlet)")
print(f"  IC          : T(x,0) = sin(πx)")
print(f"  Exact soln  : T(x,t) = exp(-α π² t) · sin(πx)\n")

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — ANALYTICAL SOLUTION (ground truth)
# ═══════════════════════════════════════════════════════════════
def analytical_solution(x):
    """
    Exact solution evaluated at points x = [x_coord, t_coord]
    Input shape: (N, 2) where col 0 = x, col 1 = t
    Output shape: (N, 1)
    """
    x_coord = x[:, 0:1]
    t_coord = x[:, 1:2]
    return np.exp(-alpha * np.pi**2 * t_coord) * np.sin(np.pi * x_coord)

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — PDE RESIDUAL
# This is the core of the PINN — what makes it physics-informed
# ═══════════════════════════════════════════════════════════════
def heat_pde(x, T):
    """
    Computes the PDE residual: ∂T/∂t - α ∂²T/∂x² = 0

    DeepXDE passes:
      x : input tensor of shape (N, 2) — [x_coord, t_coord]
      T : network output of shape (N, 1) — predicted temperature

    Both derivatives computed via automatic differentiation.
    Training drives this residual → 0 everywhere in the domain.
    """
    # ∂T/∂t  — first derivative w.r.t. time (column index 1)
    dT_dt = dde.grad.jacobian(T, x, i=0, j=1)

    # ∂²T/∂x² — second derivative w.r.t. space (column index 0)
    dT_dxx = dde.grad.hessian(T, x, i=0, j=0)

    # Residual: should be zero if PDE is satisfied
    return dT_dt - alpha * dT_dxx

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — GEOMETRY AND TIME DOMAIN
# ═══════════════════════════════════════════════════════════════
geom      = dde.geometry.Interval(0, 1)          # x ∈ [0, 1]
timedomain = dde.geometry.TimeDomain(0, 1)        # t ∈ [0, 1]
geomtime  = dde.geometry.GeometryXTime(geom, timedomain)

# ═══════════════════════════════════════════════════════════════
# SECTION 5 — BOUNDARY AND INITIAL CONDITIONS
# ═══════════════════════════════════════════════════════════════

# Dirichlet BC: T = 0 at x=0 and x=1 for all t
bc = dde.icbc.DirichletBC(
    geomtime,
    lambda x: np.zeros((len(x), 1)),        # T = 0
    lambda x, on_boundary: on_boundary      # applied on spatial boundaries
)

# Initial condition: T(x, 0) = sin(πx)
ic = dde.icbc.IC(
    geomtime,
    lambda x: np.sin(np.pi * x[:, 0:1]),   # T = sin(πx)
    lambda x, on_initial: on_initial        # applied at t = 0
)

# ═══════════════════════════════════════════════════════════════
# SECTION 6 — TRAINING DATA CONFIGURATION
# ═══════════════════════════════════════════════════════════════
# num_domain   : collocation points inside domain (PDE enforced here)
# num_boundary : points on spatial boundaries x=0, x=1
# num_initial  : points at t=0
# num_test     : points to evaluate error (not used in training)

N_DOMAIN   = 2500
N_BOUNDARY = 200
N_INITIAL  = 200
N_TEST     = 10000

data = dde.data.TimePDE(
    geomtime,
    heat_pde,
    [bc, ic],
    num_domain   = N_DOMAIN,
    num_boundary = N_BOUNDARY,
    num_initial  = N_INITIAL,
    solution     = analytical_solution,
    num_test     = N_TEST
)

print(f"  Collocation points (domain)   : {N_DOMAIN}")
print(f"  Boundary condition points     : {N_BOUNDARY}")
print(f"  Initial condition points      : {N_INITIAL}")
print(f"  Test points (validation)      : {N_TEST}")
print(f"  Total training points         : {N_DOMAIN + N_BOUNDARY + N_INITIAL}\n")

# ═══════════════════════════════════════════════════════════════
# SECTION 7 — NEURAL NETWORK ARCHITECTURE
# ═══════════════════════════════════════════════════════════════
# Input  : (x, t)       — 2 neurons
# Hidden : 4 layers × 64 neurons — tanh activation
# Output : T(x,t)       — 1 neuron
#
# Why tanh?
#   ReLU → zero second derivative almost everywhere → ∂²T/∂x² = 0
#   tanh → smooth, non-zero derivatives everywhere → PDE works

LAYER_SIZE  = [2] + [64] * 4 + [1]
ACTIVATION  = "tanh"
INITIALIZER = "Glorot uniform"

net = dde.nn.FNN(LAYER_SIZE, ACTIVATION, INITIALIZER)

print(f"  Architecture  : {LAYER_SIZE}")
print(f"  Activation    : {ACTIVATION}")
print(f"  Initializer   : {INITIALIZER}")
n_params = sum(
    (LAYER_SIZE[i] * LAYER_SIZE[i+1] + LAYER_SIZE[i+1])
    for i in range(len(LAYER_SIZE)-1)
)
print(f"  Total params  : {n_params:,}\n")

# ═══════════════════════════════════════════════════════════════
# SECTION 8 — MODEL ASSEMBLY
# ═══════════════════════════════════════════════════════════════
model = dde.Model(data, net)

# ═══════════════════════════════════════════════════════════════
# SECTION 9 — TRAINING STAGE 1: ADAM OPTIMIZER
# ═══════════════════════════════════════════════════════════════
print("─" * 60)
print("  STAGE 1 — Adam Optimizer (20,000 iterations)")
print("─" * 60)

model.compile(
    "adam",
    lr=1e-3,
    metrics=["l2 relative error"]
)

t_start = time.time()
losshistory, train_state = model.train(
    iterations=20000,
    display_every=2000    # print loss every 2000 steps
)
t_adam = time.time() - t_start
print(f"\n  Adam training time: {t_adam:.1f} seconds")

# ═══════════════════════════════════════════════════════════════
# SECTION 10 — TRAINING STAGE 2: L-BFGS FINE-TUNING
# ═══════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  STAGE 2 — L-BFGS Fine-tuning")
print("─" * 60)

model.compile(
    "L-BFGS",
    metrics=["l2 relative error"]
)

t_start = time.time()
losshistory, train_state = model.train()
t_lbfgs = time.time() - t_start
print(f"\n  L-BFGS training time: {t_lbfgs:.1f} seconds")
print(f"  Total training time : {t_adam + t_lbfgs:.1f} seconds")

# ═══════════════════════════════════════════════════════════════
# SECTION 11 — ERROR METRICS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)

# Build a dense test grid
nx_test = 200
nt_test = 200
x_test  = np.linspace(0, 1, nx_test)
t_test  = np.linspace(0, 1, nt_test)
X, T_grid = np.meshgrid(x_test, t_test)        # shape (nt, nx)
XT = np.column_stack([X.ravel(), T_grid.ravel()])  # shape (N, 2)

# PINN prediction
T_pred = model.predict(XT).reshape(nt_test, nx_test)

# Analytical solution on same grid
T_exact = np.exp(-alpha * np.pi**2 * T_grid) * np.sin(np.pi * X)

# Compute metrics
error       = T_pred - T_exact
L2_rel      = np.linalg.norm(error) / np.linalg.norm(T_exact)
Linf        = np.max(np.abs(error))
RMSE        = np.sqrt(np.mean(error**2))
R2          = 1 - np.sum(error**2) / np.sum((T_exact - T_exact.mean())**2)

print(f"\n  L2 Relative Error  : {L2_rel:.4e}")
print(f"  L∞ Error           : {Linf:.4e}")
print(f"  RMSE               : {RMSE:.4e}")
print(f"  R² Score           : {R2:.6f}")
print(f"\n  Training points    : {N_DOMAIN + N_BOUNDARY + N_INITIAL}")
print(f"  FDM equivalent     : ~1,000,000 grid evaluations")
print(f"  Point reduction    : ~{1_000_000 // (N_DOMAIN+N_BOUNDARY+N_INITIAL)}×")

# ═══════════════════════════════════════════════════════════════
# SECTION 12 — PLOTS
# ═══════════════════════════════════════════════════════════════

# ── Plot 1: Three-panel comparison ──────────────────────────────
fig = plt.figure(figsize=(16, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# PINN solution
ax0 = fig.add_subplot(gs[0])
im0 = ax0.imshow(T_pred, origin='lower', aspect='auto',
                 extent=[0,1,0,1], cmap='hot', vmin=0, vmax=1)
ax0.set_title("PINN Solution T(x,t)", fontsize=13, fontweight='bold')
ax0.set_xlabel("x"); ax0.set_ylabel("t")
plt.colorbar(im0, ax=ax0, label="Temperature")

# Exact solution
ax1 = fig.add_subplot(gs[1])
im1 = ax1.imshow(T_exact, origin='lower', aspect='auto',
                 extent=[0,1,0,1], cmap='hot', vmin=0, vmax=1)
ax1.set_title("Analytical Solution T(x,t)", fontsize=13, fontweight='bold')
ax1.set_xlabel("x"); ax1.set_ylabel("t")
plt.colorbar(im1, ax=ax1, label="Temperature")

# Absolute error
ax2 = fig.add_subplot(gs[2])
im2 = ax2.imshow(np.abs(error), origin='lower', aspect='auto',
                 extent=[0,1,0,1], cmap='Reds')
ax2.set_title(f"|Error|  (L2={L2_rel:.2e})", fontsize=13, fontweight='bold')
ax2.set_xlabel("x"); ax2.set_ylabel("t")
plt.colorbar(im2, ax=ax2, label="Absolute Error")

plt.suptitle("PINN — 1D Heat Equation  |  Mathew Ahenkora Boateng",
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig("results/pinn_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n  Saved: results/pinn_comparison.png")

# ── Plot 2: Solution slices ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
time_slices = [0.0, 0.25, 0.5, 0.75, 1.0]
colors      = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

for t_val, color in zip(time_slices, colors):
    t_idx    = np.argmin(np.abs(t_test - t_val))
    ax.plot(x_test, T_pred[t_idx],
            color=color, lw=2.5, label=f"PINN  t={t_val:.2f}")
    ax.plot(x_test, T_exact[t_idx],
            color=color, lw=1.5, ls='--', alpha=0.6,
            label=f"Exact t={t_val:.2f}")

ax.set_xlabel("x", fontsize=13)
ax.set_ylabel("T(x,t)", fontsize=13)
ax.set_title("PINN vs Analytical — Temperature at Multiple Time Slices",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=8, ncol=2, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/pinn_time_slices.png", dpi=150)
plt.show()
print("  Saved: results/pinn_time_slices.png")

# ── Plot 3: Loss history ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
loss_train = np.array(losshistory.loss_train)
steps      = np.array(losshistory.steps)

# Plot each loss component if multiple exist
if loss_train.ndim > 1 and loss_train.shape[1] > 1:
    labels = ['PDE Residual', 'Boundary Condition', 'Initial Condition']
    for i, label in enumerate(labels[:loss_train.shape[1]]):
        ax.semilogy(steps, loss_train[:, i], lw=2, label=label)
else:
    ax.semilogy(steps, loss_train.ravel(), lw=2,
                color='steelblue', label='Total Loss')

ax.set_xlabel("Training Iterations", fontsize=12)
ax.set_ylabel("Loss (log scale)", fontsize=12)
ax.set_title("PINN Training Loss History", fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig("results/pinn_loss_history.png", dpi=150)
plt.show()
print("  Saved: results/pinn_loss_history.png")

# ── Plot 4: PINN vs FDM final time comparison ────────────────────
# Load FDM result for direct comparison
# Re-run a quick FDM for comparison at same resolution
dx   = 1.0 / (nx_test - 1)
dt   = 0.4 * dx**2 / alpha
nt_f = int(1.0 / dt) + 1
x_f  = np.linspace(0, 1, nx_test)
T_f  = np.sin(np.pi * x_f)
for _ in range(nt_f):
    T_new       = T_f.copy()
    T_new[1:-1] = T_f[1:-1] + (alpha*dt/dx**2)*(T_f[2:]-2*T_f[1:-1]+T_f[:-2])
    T_new[0]    = T_new[-1] = 0.0
    T_f         = T_new

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1.plot(x_test, T_pred[-1],    lw=2.5, color='steelblue',  label='PINN')
ax1.plot(x_test, T_exact[-1],   lw=2,   color='darkorange',
         ls='--', label='Analytical')
ax1.plot(x_f,    T_f,           lw=1.5, color='green',
         ls=':',  label='FDM (nx=200)')
ax1.set_ylabel("T(x, 1.0)", fontsize=12)
ax1.set_title("Solution at t=1.0 — PINN vs FDM vs Analytical", fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

pinn_err = np.abs(T_pred[-1] - T_exact[-1])
fdm_err  = np.abs(T_f - T_exact[-1])
ax2.semilogy(x_test, pinn_err, lw=2,   color='steelblue',  label=f'PINN error')
ax2.semilogy(x_f,    fdm_err,  lw=1.5, color='green', ls=':', label=f'FDM error')
ax2.set_xlabel("x", fontsize=12)
ax2.set_ylabel("|Error| (log scale)", fontsize=12)
ax2.set_title("Absolute Error Comparison", fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig("results/pinn_vs_fdm.png", dpi=150)
plt.show()
print("  Saved: results/pinn_vs_fdm.png")

# ═══════════════════════════════════════════════════════════════
# SECTION 13 — SAVE MODEL AND SUMMARY
# ═══════════════════════════════════════════════════════════════
model.save("results/pinn_model")

print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Architecture     : {LAYER_SIZE}")
print(f"  Activation       : {ACTIVATION}")
print(f"  Domain points    : {N_DOMAIN}")
print(f"  L2 Relative Error: {L2_rel:.4e}")
print(f"  L∞ Error         : {Linf:.4e}")
print(f"  R² Score         : {R2:.6f}")
print(f"  Total train time : {t_adam + t_lbfgs:.1f} s")
print(f"  Model saved to   : results/pinn_model")
print("=" * 60)
print("\n  ✓ Day 9-13 complete. Commit results to GitHub.")
print("  Next: Data efficiency study (vary N_DOMAIN)")
