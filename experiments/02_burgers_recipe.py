# 02_burgers_recipe.py
# PINN for the viscous Burgers equation in the shock-forming regime.
# One run = one JSON file.  Does exact (hard) enforcement of the initial/boundary
# conditions help on a problem with a shock?
#
#   u_t + u u_x = nu u_xx,   x in [-1, 1],  t in [0, 0.99],  nu = 0.01/pi
#   u(x,0) = -sin(pi x),     u(-1,t) = u(1,t) = 0
#
# Reference: 03_burgers/burgers_shock.mat (Raissi, Perdikaris & Karniadakis 2019), which
# baselines/classical_baselines.py verifies against the exact Cole-Hopf solution.
#
# soft : IC and BC enforced through loss terms (weight --bc-weight on both)
# hard : u = -sin(pi x) + t (1 - x^2) N(x,t)   -> IC and BC hold exactly, no IC/BC loss
#
# Metrics, all on the 256 x 100 reference grid
#   l2            : relative L2 error over the whole grid
#   l2_near_shock : same, restricted to |x| <= 0.15 and t >= 0.35 (where the shock lives)
#   linf          : maximum absolute error
#
#   python experiments/02_burgers_recipe.py --constraint hard --seed 0
# Writes <outdir>/<name>.json

import argparse
import json
import os
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--constraint", default="soft", choices=["soft", "hard"])
ap.add_argument("--precision", default="float32", choices=["float32", "float64"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--depth", type=int, default=8, help="number of hidden layers")
ap.add_argument("--width", type=int, default=20, help="neurons per hidden layer")
ap.add_argument("--n-domain", type=int, default=10000)
ap.add_argument("--n-boundary", type=int, default=200, help="soft only")
ap.add_argument("--n-initial", type=int, default=400, help="soft only")
ap.add_argument("--adam-iters", type=int, default=20000)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--lbfgs-maxiter", type=int, default=3000)
ap.add_argument("--bc-weight", type=float, default=1.0,
                help="weight on the IC and BC loss terms (soft only; 1.0 = unweighted)")
ap.add_argument("--resample", action="store_true", help="re-draw collocation points every 1000 Adam iterations")
ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
ap.add_argument("--xla", default="off", choices=["off", "on"])
ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "03_burgers" / "burgers_shock.mat"))
ap.add_argument("--name", default=None)
ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "results"))
args = ap.parse_args()

os.environ["DDE_BACKEND"] = "tensorflow"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
if args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import scipy.io
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


dde.config.set_default_float(args.precision)
dde.config.set_random_seed(args.seed)
dde.optimizers.set_LBFGS_options(maxiter=args.lbfgs_maxiter)
dde.config.enable_xla_jit(args.xla == "on")

NU = 0.01 / np.pi
T_MAX = 0.99
LAYERS = [2] + [args.width] * args.depth + [1]

# ---- reference data -------------------------------------------------------
d = scipy.io.loadmat(args.data)
t_ref, x_ref, U = d["t"].ravel(), d["x"].ravel(), np.real(d["usol"])      # U: (256, 100)
XX, TT = np.meshgrid(x_ref, t_ref)                                        # (100, 256)
X_ref = np.column_stack([XX.ravel(), TT.ravel()])
y_ref = U.T.reshape(-1, 1)
near = (np.abs(X_ref[:, 0]) <= 0.15) & (X_ref[:, 1] >= 0.35)


def pde(x, u):
    du_x = dde.grad.jacobian(u, x, i=0, j=0)
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    return du_t + u * du_x - NU * du_xx


geomtime = dde.geometry.GeometryXTime(dde.geometry.Interval(-1, 1), dde.geometry.TimeDomain(0, T_MAX))
if args.constraint == "soft":
    bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)
    data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=args.n_domain,
                            num_boundary=args.n_boundary, num_initial=args.n_initial)
else:
    data = dde.data.TimePDE(geomtime, pde, [], num_domain=args.n_domain, num_boundary=0, num_initial=0)

net = dde.nn.FNN(LAYERS, "tanh", SeededGlorotUniform(args.seed))
if args.constraint == "hard":
    net.apply_output_transform(
        lambda x, y: -dde.backend.sin(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1] ** 2) * y)
model = dde.Model(data, net)


def evaluate(model):
    err = np.asarray(model.predict(X_ref)).reshape(-1, 1) - y_ref
    return dict(l2=float(np.linalg.norm(err) / np.linalg.norm(y_ref)),
                l2_near_shock=float(np.linalg.norm(err[near]) / np.linalg.norm(y_ref[near])),
                linf=float(np.abs(err).max()))


# soft constraints have 3 loss terms [PDE, BC, IC]; hard constraints only the PDE term
lw = [1.0, args.bc_weight, args.bc_weight] if (args.constraint == "soft" and args.bc_weight != 1.0) else None
callbacks = [dde.callbacks.PDEPointResampler(period=1000)] if args.resample else []

model.compile("adam", lr=args.lr, loss_weights=lw)
t0 = time.perf_counter()
_, ts = model.train(iterations=args.adam_iters, display_every=max(1, args.adam_iters // 4), callbacks=callbacks)
t_adam = time.perf_counter() - t0
m_adam = evaluate(model)

model.compile("L-BFGS", loss_weights=lw)
step0 = ts.step
t0 = time.perf_counter()
lh, ts = model.train()
t_lbfgs = time.perf_counter() - t0
lbfgs_iters = int(ts.step - step0)
m_final = evaluate(model)

wtag = f"_w{args.bc_weight:g}" if lw is not None else ""
name = args.name or f"burgers_{args.precision}_{args.constraint}{wtag}_s{args.seed}"
result = dict(name=name, constraint=args.constraint, precision=args.precision, seed=args.seed,
              init="seeded_glorot_uniform", xla=args.xla,
              bc_weight=args.bc_weight if lw is not None else 1.0, resample=args.resample,
              layers=LAYERS, n_domain=args.n_domain,
              n_boundary=args.n_boundary if args.constraint == "soft" else 0,
              n_initial=args.n_initial if args.constraint == "soft" else 0,
              adam_iters=args.adam_iters, lr=args.lr, lbfgs_maxiter=args.lbfgs_maxiter,
              lbfgs_iters_run=lbfgs_iters, lbfgs_stopped_early=bool(lbfgs_iters < args.lbfgs_maxiter),
              final_train_loss_terms=[float(v) for v in np.atleast_1d(lh.loss_train[-1])],
              after_adam=m_adam, final=m_final,
              seconds_adam=round(t_adam, 1), seconds_lbfgs=round(t_lbfgs, 1),
              deepxde=dde.__version__, tensorflow=tf.__version__)
out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)
(out / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"{name}: after Adam L2={m_adam['l2']:.2e} | final L2={m_final['l2']:.2e} "
      f"near-shock={m_final['l2_near_shock']:.2e} Linf={m_final['linf']:.2e} | "
      f"L-BFGS {lbfgs_iters} its | {t_adam:.0f}s + {t_lbfgs:.0f}s")
