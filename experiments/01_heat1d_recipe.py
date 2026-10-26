# 01_heat1d_recipe.py
# 1D transient heat PINN: does numerical precision and exact (hard) enforcement of
# the boundary/initial conditions change accuracy? One run = one JSON file.
#
#   u_t = alpha u_xx,  x,t in [0,1],  alpha = 0.4
#   u(0,t) = u(1,t) = 0,   u(x,0) = sin(pi x)
#   exact: u = exp(-alpha pi^2 t) sin(pi x)
#
# soft  : BC/IC enforced through loss terms (200 + 200 points), as in the first study
# hard  : u = sin(pi x) + x(1-x) t N(x,t)  -> BC and IC hold exactly, no BC/IC loss
#
# Metrics (all against the exact solution)
#   l2, linf     : whole space-time domain, 200 x 200 grid
#   flux_l2      : wall heat flux du/dx at x=0 and x=1 over time (engineering quantity)
#   energy_l2    : stored energy  integral_0^1 u dx  over time
#
#   python experiments/01_heat1d_recipe.py --precision float64 --constraint hard --seed 0
# Writes <outdir>/<name>.json

import argparse
import json
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--precision", default="float32", choices=["float32", "float64"])
ap.add_argument("--constraint", default="soft", choices=["soft", "hard"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n-domain", type=int, default=2500)
ap.add_argument("--adam-iters", type=int, default=20000)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--lbfgs-maxiter", type=int, default=15000)
ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
ap.add_argument("--name", default=None)
ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "results"))
args = ap.parse_args()

os.environ["DDE_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
if args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
import deepxde as dde

dde.config.set_default_float(args.precision)
dde.config.set_random_seed(args.seed)
dde.optimizers.set_LBFGS_options(maxiter=args.lbfgs_maxiter)

ALPHA = 0.4
LAYERS = [2] + [64] * 4 + [1]


def exact(x, t):
    return np.exp(-ALPHA * np.pi**2 * t) * np.sin(np.pi * x)


def pde(x, u):
    return dde.grad.jacobian(u, x, i=0, j=1) - ALPHA * dde.grad.hessian(u, x, i=0, j=0)


geomtime = dde.geometry.GeometryXTime(dde.geometry.Interval(0, 1),
                                      dde.geometry.TimeDomain(0, 1))
sol = lambda x: exact(x[:, 0:1], x[:, 1:2])
if args.constraint == "soft":
    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]),
                     lambda _, on_initial: on_initial)
    data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=args.n_domain,
                            num_boundary=200, num_initial=200, solution=sol, num_test=5000)
else:
    data = dde.data.TimePDE(geomtime, pde, [], num_domain=args.n_domain,
                            num_boundary=0, num_initial=0, solution=sol, num_test=5000)

net = dde.nn.FNN(LAYERS, "tanh", "Glorot uniform")
if args.constraint == "hard":
    net.apply_output_transform(
        lambda x, y: dde.backend.sin(np.pi * x[:, 0:1])
        + x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * y)
model = dde.Model(data, net)


# ----------------------------------------------------------------------------
def evaluate(model):
    nx = nt = 200
    X, T = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, nt))   # (nt, nx)
    pred = np.asarray(model.predict(np.column_stack([X.ravel(), T.ravel()]))).reshape(nt, nx)
    ex = exact(X, T)
    err = pred - ex

    tt = np.linspace(0, 1, 201)
    d_dx = lambda x, y: dde.grad.jacobian(y, x, i=0, j=0)
    f0 = np.asarray(model.predict(np.column_stack([np.zeros_like(tt), tt]), operator=d_dx)).ravel()
    f1 = np.asarray(model.predict(np.column_stack([np.ones_like(tt), tt]), operator=d_dx)).ravel()
    fe0 = np.pi * np.exp(-ALPHA * np.pi**2 * tt)          # exact du/dx at x=0
    fe1 = -fe0                                            # exact du/dx at x=1
    flux = np.sqrt(np.sum((f0 - fe0) ** 2 + (f1 - fe1) ** 2) / np.sum(fe0**2 + fe1**2))

    xs = np.linspace(0, 1, nx)
    trap = getattr(np, "trapezoid", None) or np.trapz     # numpy 2.x renamed trapz
    E_pred = trap(pred, xs, axis=1)                       # one value per time level
    E_ex = trap(ex, xs, axis=1)
    return dict(l2=float(np.linalg.norm(err) / np.linalg.norm(ex)),
                linf=float(np.abs(err).max()),
                flux_l2=float(flux),
                energy_l2=float(np.linalg.norm(E_pred - E_ex) / np.linalg.norm(E_ex)))


model.compile("adam", lr=args.lr, metrics=["l2 relative error"])
t0 = time.perf_counter()
_, ts = model.train(iterations=args.adam_iters, display_every=max(1, args.adam_iters // 4))
t_adam = time.perf_counter() - t0
m_adam = evaluate(model)

model.compile("L-BFGS", metrics=["l2 relative error"])
step0 = ts.step
t0 = time.perf_counter()
lh, ts = model.train()
t_lbfgs = time.perf_counter() - t0
lbfgs_iters = int(ts.step - step0)
m_final = evaluate(model)

name = args.name or f"heat1d_{args.precision}_{args.constraint}_s{args.seed}"
result = dict(name=name, precision=args.precision, constraint=args.constraint,
              seed=args.seed, n_domain=args.n_domain, adam_iters=args.adam_iters,
              lr=args.lr, layers=LAYERS, lbfgs_maxiter=args.lbfgs_maxiter,
              lbfgs_iters_run=lbfgs_iters,
              lbfgs_stopped_early=bool(lbfgs_iters < args.lbfgs_maxiter),
              final_train_loss_terms=[float(v) for v in np.atleast_1d(lh.loss_train[-1])],
              after_adam=m_adam, final=m_final,
              seconds_adam=round(t_adam, 1), seconds_lbfgs=round(t_lbfgs, 1),
              deepxde=dde.__version__, tensorflow=tf.__version__)
out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)
(out / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"{name}: after Adam L2={m_adam['l2']:.2e} | final L2={m_final['l2']:.2e} "
      f"flux={m_final['flux_l2']:.2e} energy={m_final['energy_l2']:.2e} | "
      f"L-BFGS {lbfgs_iters} its | {t_adam:.0f}s + {t_lbfgs:.0f}s")