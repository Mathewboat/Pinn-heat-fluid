# fdm_baselines_corrected.py
# Corrected classical baselines for the PINN comparisons.
#
# Why this file exists
#   2D: the earlier Gauss-Seidel baseline stopped when the per-sweep update fell
#       below 1e-6. For a 100x100 grid the iteration converges so slowly
#       (spectral radius ~0.999) that the true error was still ~1e-3 absolute,
#       i.e. ~2% relative. That is iteration error, not discretisation error.
#       A direct sparse solve removes it.
#   1D: the earlier study compared FDM at a single final time (and sometimes
#       stepped slightly past t=1), while the PINN error was measured over the
#       whole space-time domain. Here FDM is measured over ALL time steps, and
#       dt is adjusted so the last step lands exactly on t=1.
#
# Run from the repo root:   python fdm_baselines_corrected.py
# Needs only numpy + scipy.

import os
import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.stdout.reconfigure(encoding="utf-8")
os.makedirs("results", exist_ok=True)

ALPHA = 0.4  # non-dimensional diffusivity used in the 1D PINN study


def fdm_2d_poisson(n):
    """-lap(T) = sin(pi x) sin(pi y) on the unit square, T=0 on the boundary.
    Second-order central differences, direct sparse solve."""
    h = 1.0 / (n - 1)
    xg = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xg, xg)
    Q = np.sin(np.pi * X) * np.sin(np.pi * Y)
    m = n - 2
    D = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(m, m))
    A = (sp.kron(sp.identity(m), D) + sp.kron(D, sp.identity(m))) / h**2
    t0 = time.time()
    sol = spla.spsolve(A.tocsc(), -Q[1:-1, 1:-1].ravel())
    elapsed = time.time() - t0
    T = np.zeros((n, n))
    T[1:-1, 1:-1] = sol.reshape(m, m)
    T_exact = Q / (2.0 * np.pi**2)
    l2 = np.linalg.norm(T - T_exact) / np.linalg.norm(T_exact)
    linf = np.max(np.abs(T - T_exact))
    return l2, linf, n * n, elapsed


def fdm_1d_heat(nx):
    """dT/dt = alpha d2T/dx2, T(0,t)=T(1,t)=0, T(x,0)=sin(pi x).
    Explicit scheme, dt chosen so that nt*dt == 1 exactly and r <= 0.4.
    Error is accumulated over every node and every time step."""
    dx = 1.0 / (nx - 1)
    nt = int(np.ceil(1.0 / (0.4 * dx**2 / ALPHA)))
    dt = 1.0 / nt
    r = ALPHA * dt / dx**2
    assert r <= 0.5
    xg = np.linspace(0.0, 1.0, nx)
    s = np.sin(np.pi * xg)
    T = s.copy()
    num = den = 0.0
    t0 = time.time()
    for k in range(1, nt + 1):
        Tn = T.copy()
        Tn[1:-1] = T[1:-1] + r * (T[2:] - 2.0 * T[1:-1] + T[:-2])
        T = Tn
        ex = np.exp(-ALPHA * np.pi**2 * k * dt) * s
        num += np.sum((T - ex) ** 2)
        den += np.sum(ex**2)
    elapsed = time.time() - t0
    return np.sqrt(num / den), nx * (nt + 1), elapsed


lines = []


def out(s=""):
    print(s)
    lines.append(s)


out("=" * 70)
out("  CORRECTED FDM BASELINES")
out("=" * 70)

out("\n  2D steady Poisson, direct sparse solve (no iteration error)")
out(f"  {'n':>5} {'points':>9} {'L2 rel':>12} {'Linf':>12} {'time (s)':>10}")
for n in (20, 50, 100, 200):
    l2, linf, pts, el = fdm_2d_poisson(n)
    out(f"  {n:>5} {pts:>9,} {l2:>12.3e} {linf:>12.3e} {el:>10.3f}")

out("\n  1D transient heat, explicit FDM, space-time L2 (same metric as PINN)")
out(f"  {'nx':>5} {'space-time pts':>15} {'L2 rel':>12} {'time (s)':>10}")
for nx in (20, 50, 100, 200, 500):
    l2, pts, el = fdm_1d_heat(nx)
    out(f"  {nx:>5} {pts:>15,} {l2:>12.3e} {el:>10.2f}")

out("\n  Compare these against the PINN numbers on the same metric.")
with open("results/fdm_baselines_corrected.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n  Saved: results/fdm_baselines_corrected.txt")