# run_grid.py
# Run a grid of experiment jobs as parallel subprocesses, spread over GPUs.
# Each job is one invocation of an experiment script that writes <outdir>/<name>.json.
# Jobs whose JSON already exists are skipped, so an interrupted run can simply be restarted.
#
# Example (Kaggle, 2 GPUs, 2 processes per GPU):
#   python experiments/run_grid.py --script experiments/01_heat1d_recipe.py \
#       --grid precision=float32,float64 constraint=soft,hard \
#       --seeds 0 1 2 3 4 --gpus 0 1 --procs-per-gpu 2 \
#       --name-template "heat1d_{precision}_{constraint}_s{seed}" \
#       --outdir /kaggle/working/results/heat1d_recipe
#
# Anything after --extra is passed to every job unchanged (e.g. --extra --adam-iters 500).

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--script", required=True)
ap.add_argument("--grid", nargs="+", default=[], help="key=v1,v2 ... (becomes --key value)")
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--gpus", type=int, nargs="+", default=[0])
ap.add_argument("--procs-per-gpu", type=int, default=1)
ap.add_argument("--name-template", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
args = ap.parse_args()

grid = {}
for item in args.grid:
    key, vals = item.split("=", 1)
    grid[key] = vals.split(",")

outdir = Path(args.outdir)
logdir = outdir / "logs"
logdir.mkdir(parents=True, exist_ok=True)

jobs = []
keys = list(grid)
for combo in itertools.product(*[grid[k] for k in keys]):
    params = dict(zip(keys, combo))
    for seed in args.seeds:
        name = args.name_template.format(seed=seed, **params)
        cmd = [sys.executable, args.script]
        for k, v in params.items():
            cmd += [f"--{k}", v]
        cmd += ["--seed", str(seed), "--name", name, "--outdir", str(outdir)] + args.extra
        jobs.append((name, cmd))

todo = [(n, c) for n, c in jobs if not (outdir / f"{n}.json").exists()]
print(f"{len(jobs)} jobs total, {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)

slots = [g for g in args.gpus for _ in range(args.procs_per_gpu)]
running = {}          # slot index -> (name, Popen, logfile handle, start time)
failed = []
t_start = time.time()
done_count = 0

while todo or running:
    for i in range(len(slots)):
        if i not in running and todo:
            name, cmd = todo.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slots[i]))
            fh = open(logdir / f"{name}.log", "w")
            running[i] = (name, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env),
                          fh, time.time())
            print(f"[{time.time() - t_start:6.0f}s] start  {name}  (gpu {slots[i]})", flush=True)
    for i in list(running):
        name, proc, fh, t0 = running[i]
        if proc.poll() is not None:
            fh.close()
            done_count += 1
            ok = proc.returncode == 0 and (outdir / f"{name}.json").exists()
            if not ok:
                failed.append(name)
            print(f"[{time.time() - t_start:6.0f}s] {'done ' if ok else 'FAILED'} {name}  "
                  f"({time.time() - t0:.0f}s)  [{done_count} finished]", flush=True)
            del running[i]
    time.sleep(2)

print(f"\nAll finished in {time.time() - t_start:.0f}s. Failed: {failed if failed else 'none'}")
if failed:
    print(f"See logs in {logdir}")
    sys.exit(1)