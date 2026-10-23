# 00_env_benchmark.py
# Environment check + speed benchmark for a 1D heat-equation PINN.
#
# Purpose: find out, with measurements, whether a GPU (and float64) actually helps
# for this kind of small PINN, before we commit to a plan.
#
# One configuration per process (backend, device and precision are fixed at import time):
#   python experiments/00_env_benchmark.py --backend pytorch --device gpu --float float32
#
# Writes <outdir>/bench_<backend>_<device>_<float>.json
#
# Problem: u_t = alpha u_xx, x,t in [0,1], alpha=0.4, u(0,t)=u(1,t)=0, u(x,0)=sin(pi x)
# Exact:   u = exp(-alpha pi^2 t) sin(pi x)

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="pytorch", choices=["pytorch", "tensorflow", "jax"])
ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
ap.add_argument("--float", dest="prec", default="float32", choices=["float32", "float64"])
ap.add_argument("--iters", type=int, default=2000, help="timed Adam iterations")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "results"))
args = ap.parse_args()

# These must be set BEFORE deepxde / the backend library is imported.
os.environ["DDE_BACKEND"] = args.backend
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import deepxde as dde

dde.config.set_default_float(args.prec)
dde.config.set_random_seed(args.seed)

# ----------------------------------------------------------------------------
# Environment report
# ----------------------------------------------------------------------------
def gpu_hardware():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip().replace("\n", " | ") or "none"
    except Exception:
        return "none"


info = dict(python=platform.python_version(), deepxde=dde.__version__,
            backend=dde.backend.backend_name, requested_device=args.device,
            precision=args.prec, gpu_hardware=gpu_hardware(), cpu_count=os.cpu_count())
if args.backend == "pytorch":
    import torch
    info["torch"] = torch.__version__
    info["backend_sees_gpu"] = bool(torch.cuda.is_available())
elif args.backend == "tensorflow":
    import tensorflow as tf
    info["tensorflow"] = tf.__version__
    info["backend_sees_gpu"] = len(tf.config.list_physical_devices("GPU")) > 0
else:
    import jax
    info["jax"] = jax.__version__
    info["backend_sees_gpu"] = any(d.platform == "gpu" for d in jax.devices())

# ----------------------------------------------------------------------------
# Problem (backend-neutral: only dde.grad and numpy are used)
# ----------------------------------------------------------------------------
ALPHA = 0.4


def pde(x, u):
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    return du_t - ALPHA * du_xx


def exact_np(x, t):
    return np.exp(-ALPHA * np.pi**2 * t) * np.sin(np.pi * x)


geomtime = dde.geometry.GeometryXTime(dde.geometry.Interval(0, 1),
                                      dde.geometry.TimeDomain(0, 1))
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda x: np.sin(np.pi * x[:, 0:1]),
                 lambda _, on_initial: on_initial)
data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=2500, num_boundary=200,
                        num_initial=200,
                        solution=lambda x: exact_np(x[:, 0:1], x[:, 1:2]), num_test=2000)
net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=1e-3, metrics=["l2 relative error"])

# Warm-up (graph tracing / kernel compilation / cuDNN autotune are excluded from timing)
model.train(iterations=50, display_every=10**9)

t0 = time.perf_counter()
losshistory, train_state = model.train(iterations=args.iters, display_every=args.iters)
elapsed = time.perf_counter() - t0

# Accuracy of the (briefly trained) network on a dense grid, as a sanity check
xg, tg = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
pred = np.asarray(model.predict(np.column_stack([xg.ravel(), tg.ravel()]))).ravel()
ex = exact_np(xg.ravel(), tg.ravel())
l2 = float(np.linalg.norm(pred - ex) / np.linalg.norm(ex))

result = dict(info, timed_iters=args.iters, seconds=round(elapsed, 3),
              ms_per_iter=round(1e3 * elapsed / args.iters, 3),
              final_train_loss=float(np.sum(losshistory.loss_train[-1])),
              l2_relative_error_after_run=l2)
out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)
fn = out / f"bench_{args.backend}_{args.device}_{args.prec}.json"
fn.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
print(f"\nSaved {fn}")