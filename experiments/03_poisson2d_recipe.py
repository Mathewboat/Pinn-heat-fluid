# 03_poisson2d_recipe.py
# 2D steady heat conduction with a source (Poisson equation) on the unit square.
# One run = one JSON file.
#
#   -(T_xx + T_yy) = sin(pi x) sin(pi y),   T = 0 on the boundary
#   exact: T = sin(pi x) sin(pi y) / (2 pi^2)        (peak value ~ 0.0507)
#
# The network learns u = SCALE * T, with SCALE = 1 ("raw") or 2 pi^2 ("rescaled", so that u is O(1)).
# The relative errors below do not depend on SCALE; only the training conditioning does.
#
# soft : T = 0 on the boundary through a loss term (weight --bc-weight)
# hard : u = x (1-x) y (1-y) N(x,y)   -> the boundary condition holds exactly.
#        This form does NOT contain the solution's shape (unlike the 1D heat problem).
#
# Metrics, all against the exact solution (200 x 200 grid, 201 points per edge)
#   l2, linf        : relative L2 / maximum absolute error of the field
#   flux_l2         : relative L2 error of the outward heat flux along all four edges
#   heat_total_err  : relative error of the total heat leaving the boundary (exact: 4/pi^2 for T)
#
#   python experiments/03_poisson2d_recipe.py --constraint hard --scale rescaled --seed 0
# Writes <outdir>/<name>.json  (and <name>.partial after the Adam stage)

import argparse
import json
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--constraint", default="soft", choices=["soft", "hard"])
ap.add_argument("--scale", default="raw", choices=["raw", "rescaled"])
ap.add_argument("--precision", default="float32", choices=["float32", "float64"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--depth", type=int, default=4, help="number of hidden layers")
ap.add_argument("--width", type=int, default=64, help="neurons per hidden layer")
ap.add_argument("--n-domain", type=int, default=2500)
ap.add_argument("--n-boundary", type=int, default=400, help="soft only")
ap.add_argument("--adam-iters", type=int, default=20000)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--lbfgs-maxiter", type=int, default=3000)
ap.add_argument("--bc-weight", type=float, default=1.0,
                help="weight on the boundary loss term (soft only; 1.0 = unweighted)")
ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
ap.add_argument("--xla", default="off", choices=["off", "on"])
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


class SeededGlorotUniform:
    """Glorot-uniform initializer driven by a numpy Generator (reproducible per seed, different per layer).
    DeepXDE's string initializer is not controlled by dde.config.set_random_seed with TF 2 / Keras 3."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape, dtype=None, **kwargs):
        limit = np.sqrt(6.0 / (shape[0] + shape[1]))
        return tf.constant(self.rng.uniform(-limit, limit, size=tuple(shape)),
                           dtype=dtype if dtype is not None else tf.keras.backend.floatx())


def boundary_metrics(g_left, g_right, g_bottom, g_top, s, scale):
    """g_* : outward normal derivative of u on each edge, sampled at tangential coordinate s.
    Exact outward normal derivative of u = scale*T is -scale*sin(pi s)/(2 pi) on every edge."""
    ge = -scale * np.sin(np.pi * s) / (2 * np.pi)
    gp = [g_left, g_right, g_bottom, g_top]
    flux = np.sqrt(sum(np.sum((g - ge) ** 2) for g in gp) / (4 * np.sum(ge**2)))
    trap = getattr(np, "trapezoid", None) or np.trapz               # numpy 2.x renamed trapz
    heat_pred = sum(trap(-g, s) for g in gp)                         # heat leaving = -integral of du/dn
    heat_exact = 4 * trap(-ge, s)                                    # = scale * 4/pi^2
    return float(flux), float(abs(heat_pred - heat_exact) / abs(heat_exact))


dde.config.set_default_float(args.precision)
dde.config.set_random_seed(args.seed)
dde.optimizers.set_LBFGS_options(maxiter=args.lbfgs_maxiter)
dde.config.enable_xla_jit(args.xla == "on")

SCALE = 2 * np.pi**2 if args.scale == "rescaled" else 1.0
LAYERS = [2] + [args.width] * args.depth + [1]


def pde(x, u):
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)
    return u_xx + u_yy + SCALE * dde.backend.sin(np.pi * x[:, 0:1]) * dde.backend.sin(np.pi * x[:, 1:2])


geom = dde.geometry.Rectangle([0, 0], [1, 1])
if args.constraint == "soft":
    bc = dde.icbc.DirichletBC(geom, lambda x: 0, lambda _, on_boundary: on_boundary)
    data = dde.data.PDE(geom, pde, [bc], num_domain=args.n_domain, num_boundary=args.n_boundary)
else:
    data = dde.data.PDE(geom, pde, [], num_domain=args.n_domain, num_boundary=0)

net = dde.nn.FNN(LAYERS, "tanh", SeededGlorotUniform(args.seed))
if args.constraint == "hard":
    net.apply_output_transform(
        lambda x, y: x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2]) * y)
model = dde.Model(data, net)


def evaluate(model):
    n = 200
    xg = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xg, xg)
    pred = np.asarray(model.predict(np.column_stack([X.ravel(), Y.ravel()]))).reshape(n, n)
    ex = SCALE * np.sin(np.pi * X) * np.sin(np.pi * Y) / (2 * np.pi**2)
    err = pred - ex

    s = np.linspace(0, 1, 201)
    z, o = np.zeros_like(s), np.ones_like(s)
    d_dx = lambda x, y: dde.grad.jacobian(y, x, i=0, j=0)
    d_dy = lambda x, y: dde.grad.jacobian(y, x, i=0, j=1)
    pr = lambda pts, op: np.asarray(model.predict(pts, operator=op)).ravel()
    g_left = -pr(np.column_stack([z, s]), d_dx)       # x = 0, outward normal = -x
    g_right = pr(np.column_stack([o, s]), d_dx)       # x = 1, outward normal = +x
    g_bottom = -pr(np.column_stack([s, z]), d_dy)     # y = 0, outward normal = -y
    g_top = pr(np.column_stack([s, o]), d_dy)         # y = 1, outward normal = +y
    flux, heat = boundary_metrics(g_left, g_right, g_bottom, g_top, s, SCALE)
    return dict(l2=float(np.linalg.norm(err) / np.linalg.norm(ex)), linf=float(np.abs(err).max()),
                flux_l2=flux, heat_total_err=heat)


# soft constraints have 2 loss terms [PDE, BC]; hard constraints only the PDE term
lw = [1.0, args.bc_weight] if (args.constraint == "soft" and args.bc_weight != 1.0) else None
wtag = f"_w{args.bc_weight:g}" if lw is not None else ""
name = args.name or f"poisson2d_{args.precision}_{args.constraint}_{args.scale}{wtag}_s{args.seed}"
out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)

model.compile("adam", lr=args.lr, loss_weights=lw)
t0 = time.perf_counter()
_, ts = model.train(iterations=args.adam_iters, display_every=max(1, args.adam_iters // 4))
t_adam = time.perf_counter() - t0
m_adam = evaluate(model)
(out / f"{name}.partial").write_text(json.dumps(dict(name=name, stage="after_adam", after_adam=m_adam,
                                                     seconds_adam=round(t_adam, 1))), encoding="utf-8")
print(f"{name}: Adam stage done, L2={m_adam['l2']:.2e} ({t_adam:.0f}s); starting L-BFGS", flush=True)

model.compile("L-BFGS", loss_weights=lw)
step0 = ts.step
t0 = time.perf_counter()
lh, ts = model.train()
t_lbfgs = time.perf_counter() - t0
lbfgs_iters = int(ts.step - step0)
m_final = evaluate(model)

result = dict(name=name, constraint=args.constraint, scale=args.scale, precision=args.precision,
              seed=args.seed, init="seeded_glorot_uniform", xla=args.xla,
              bc_weight=args.bc_weight if lw is not None else 1.0,
              layers=LAYERS, n_domain=args.n_domain,
              n_boundary=args.n_boundary if args.constraint == "soft" else 0,
              adam_iters=args.adam_iters, lr=args.lr, lbfgs_maxiter=args.lbfgs_maxiter,
              lbfgs_iters_run=lbfgs_iters, lbfgs_stopped_early=bool(lbfgs_iters < args.lbfgs_maxiter),
              final_train_loss_terms=[float(v) for v in np.atleast_1d(lh.loss_train[-1])],
              after_adam=m_adam, final=m_final,
              seconds_adam=round(t_adam, 1), seconds_lbfgs=round(t_lbfgs, 1),
              deepxde=dde.__version__, tensorflow=tf.__version__)
(out / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"{name}: after Adam L2={m_adam['l2']:.2e} | final L2={m_final['l2']:.2e} "
      f"flux={m_final['flux_l2']:.2e} heat={m_final['heat_total_err']:.2e} | "
      f"L-BFGS {lbfgs_iters} its | {t_adam:.0f}s + {t_lbfgs:.0f}s")
