# classical_baselines.py
# Classical solvers used as baselines for the PINN experiments, measured on YOUR machine.
# Needs only numpy + scipy. Put it in baselines/ and run from anywhere:
#     python baselines/classical_baselines.py
#
# 1) 1D heat  u_t = 0.4 u_xx on [0,1]^2, u(0,t)=u(1,t)=0, u(x,0)=sin(pi x)
#      - explicit Euler (stability-limited dt ~ dx^2)   [naive baseline]
#      - Crank-Nicolson, dt = dx (second order, unconditionally stable)   [fair baseline]
#    error = relative L2 over ALL grid nodes and ALL time steps (same metric as the PINN study)
# 2) 2D Poisson  -lap(T) = sin(pi x) sin(pi y) on the unit square, T = 0 on the boundary
#      - second-order central differences, direct sparse solve
# 3) Burgers  u_t + u u_x = nu u_xx, nu = 0.01/pi, u(x,0) = -sin(pi x), u(+-1,t) = 0
#      - conservative central differences in space + BDF time integration (rtol 1e-8)
#      - error vs 03_burgers/burgers_shock.mat on all 256 x 100 points, and in the near-shock
#        region |x| <= 0.15, t >= 0.35
#    The reference file itself is verified against the exact Cole-Hopf solution.
#
# Writes results/baselines/classical_baselines.json and prints markdown tables.

import json
import time
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.4
NU = 0.01 / np.pi
out = {}


# ----------------------------------------------------------------------------
# 1) 1D heat
# ----------------------------------------------------------------------------
def heat1d_explicit(nx):
    dx = 1.0 / (nx - 1)
    nt = int(np.ceil(1.0 / (0.4 * dx**2 / ALPHA)))          # r <= 0.4, lands exactly on t = 1
    dt = 1.0 / nt
    r = ALPHA * dt / dx**2
    s = np.sin(np.pi * np.linspace(0, 1, nx))
    u = s.copy()
    num = den = 0.0
    t0 = time.perf_counter()
    for k in range(1, nt + 1):
        un = u.copy()
        un[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        u = un
        ex = np.exp(-ALPHA * np.pi**2 * k * dt) * s
        num += np.sum((u - ex) ** 2)
        den += np.sum(ex**2)
    return dict(nx=nx, steps=nt, points=nx * (nt + 1), l2=float(np.sqrt(num / den)),
                seconds=time.perf_counter() - t0)


def heat1d_cn(nx):
    nt = nx - 1                                              # dt = dx
    dx = 1.0 / (nx - 1)
    dt = 1.0 / nt
    m = nx - 2
    r = ALPHA * dt / dx**2
    L = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(m, m))
    t0 = time.perf_counter()
    lu = spla.splu((sp.identity(m) - 0.5 * r * L).tocsc())
    B = (sp.identity(m) + 0.5 * r * L).tocsr()
    s = np.sin(np.pi * np.linspace(0, 1, nx))[1:-1]
    u = s.copy()
    num = den = 0.0
    for k in range(1, nt + 1):
        u = lu.solve(B @ u)
        ex = np.exp(-ALPHA * np.pi**2 * k * dt) * s
        num += np.sum((u - ex) ** 2)
        den += np.sum(ex**2)
    return dict(nx=nx, steps=nt, points=nx * (nt + 1), l2=float(np.sqrt(num / den)),
                seconds=time.perf_counter() - t0)


out["heat1d_explicit"] = [heat1d_explicit(n) for n in (20, 50, 100, 200)]
out["heat1d_crank_nicolson"] = [heat1d_cn(n) for n in (21, 41, 81, 161, 321, 641)]


# ----------------------------------------------------------------------------
# 2) 2D Poisson, direct solve
# ----------------------------------------------------------------------------
def poisson2d(n):
    h = 1.0 / (n - 1)
    xg = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xg, xg)
    Q = np.sin(np.pi * X) * np.sin(np.pi * Y)
    m = n - 2
    D = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(m, m))
    A = (sp.kron(sp.identity(m), D) + sp.kron(D, sp.identity(m))) / h**2
    t0 = time.perf_counter()
    sol = spla.spsolve(A.tocsc(), -Q[1:-1, 1:-1].ravel())
    el = time.perf_counter() - t0
    T = np.zeros((n, n))
    T[1:-1, 1:-1] = sol.reshape(m, m)
    ex = Q / (2 * np.pi**2)
    return dict(n=n, points=n * n, l2=float(np.linalg.norm(T - ex) / np.linalg.norm(ex)), seconds=el)


out["poisson2d_direct"] = [poisson2d(n) for n in (20, 50, 100, 200)]


# ----------------------------------------------------------------------------
# 3) Burgers
# ----------------------------------------------------------------------------
data_path = ROOT / "03_burgers" / "burgers_shock.mat"
if data_path.exists():
    d = scipy.io.loadmat(data_path)
    t = d["t"].ravel()
    x = d["x"].ravel()
    U = np.real(d["usol"])                                  # (256, 100); x includes both endpoints
    assert np.allclose(np.diff(x), 2 / 255)
    near = (np.abs(x)[:, None] <= 0.15) & (t[None, :] >= 0.35)

    def burgers_fd(m):
        N = 255 * m                                         # reference nodes are every m-th node
        h = 2.0 / N
        xs = -1 + h * np.arange(N + 1)
        n = N - 1

        def rhs(_, u):
            up = np.zeros(n + 2)
            up[1:-1] = u
            return -(up[2:] ** 2 - up[:-2] ** 2) / (4 * h) + NU * (up[2:] - 2 * up[1:-1] + up[:-2]) / h**2

        S = sp.diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(n, n)).tocsr()
        t0 = time.perf_counter()
        sol = solve_ivp(rhs, (0, t[-1]), -np.sin(np.pi * xs[1:-1]), method="BDF", t_eval=t,
                        rtol=1e-8, atol=1e-10, jac_sparsity=S)
        el = time.perf_counter() - t0
        full = np.zeros((N + 1, len(t)))
        full[1:-1] = sol.y
        e = full[::m] - U
        return dict(grid_points=N + 1, cell_peclet=float(h / NU),
                    l2=float(np.linalg.norm(e) / np.linalg.norm(U)),
                    l2_near_shock=float(np.linalg.norm(e[near]) / np.linalg.norm(U[near])),
                    max_abs=float(np.abs(e).max()), seconds=el)

    out["burgers_central_fd_bdf"] = [burgers_fd(m) for m in (1, 2, 4, 8)]

    # verify the reference against the exact Cole-Hopf solution on a subset of time levels
    def cole_hopf(xv, tv, nq=40001):
        res = np.empty((len(xv), len(tv)))
        for j, tt in enumerate(tv):
            s = np.sqrt(2 * NU * tt)
            eta = np.linspace(-12 * s, 12 * s, nq)
            for i, xx in enumerate(xv):
                y = xx - eta
                lw = -np.cos(np.pi * y) / (2 * np.pi * NU) - eta**2 / (4 * NU * tt)
                w = np.exp(lw - lw.max())
                res[i, j] = -np.sum(np.sin(np.pi * y) * w) / np.sum(w)
        return res

    sel = np.arange(11, len(t), 11)                         # skips t = 0 (Cole-Hopf needs t > 0)
    ex = cole_hopf(x, t[sel])
    err = U[:, sel] - ex
    out["burgers_reference_check"] = dict(
        time_levels_checked=int(len(sel)),
        l2=float(np.linalg.norm(err) / np.linalg.norm(ex)), max_abs=float(np.abs(err).max()))
else:
    print(f"(skipping Burgers: {data_path} not found)")

# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
res_dir = ROOT / "results" / "baselines"
res_dir.mkdir(parents=True, exist_ok=True)
(res_dir / "classical_baselines.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def table(title, rows, cols):
    print(f"\n{title}")
    print("| " + " | ".join(h for h, _, _ in cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f(r[k]) for _, k, f in cols) + " |")


g_int = lambda v: f"{int(v):,}"
g_sci = lambda v: f"{v:.2e}"
g_sec = lambda v: f"{v:.3f}"
table("1D heat, explicit Euler (stability-limited)", out["heat1d_explicit"],
      [("nx", "nx", g_int), ("space-time points", "points", g_int), ("L2", "l2", g_sci), ("seconds", "seconds", g_sec)])
table("1D heat, Crank-Nicolson (dt = dx)", out["heat1d_crank_nicolson"],
      [("nx", "nx", g_int), ("space-time points", "points", g_int), ("L2", "l2", g_sci), ("seconds", "seconds", g_sec)])
table("2D Poisson, direct sparse solve", out["poisson2d_direct"],
      [("n", "n", g_int), ("points", "points", g_int), ("L2", "l2", g_sci), ("seconds", "seconds", g_sec)])
if "burgers_central_fd_bdf" in out:
    table("Burgers, central FD + BDF vs reference", out["burgers_central_fd_bdf"],
          [("grid points", "grid_points", g_int), ("cell Peclet", "cell_peclet", lambda v: f"{v:.2f}"),
           ("L2", "l2", g_sci), ("L2 near shock", "l2_near_shock", g_sci), ("seconds", "seconds", g_sec)])
    c = out["burgers_reference_check"]
    print(f"\nReference file vs exact Cole-Hopf solution ({c['time_levels_checked']} time levels): "
          f"relative L2 = {c['l2']:.2e}, max abs diff = {c['max_abs']:.2e}")
print(f"\nSaved {res_dir / 'classical_baselines.json'}")
