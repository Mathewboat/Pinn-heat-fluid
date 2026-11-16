# verify_data.py  — place in 03_burgers/ and run it
import scipy.io
import numpy as np

data = scipy.io.loadmat('03_burgers/burgers_shock.mat')

print("Keys in file:", list(data.keys()))

t    = data['t'].flatten()       # time points
x    = data['x'].flatten()       # spatial points
u    = np.real(data['usol'])     # solution field shape: (x, t)

print(f"t shape : {t.shape}  → {len(t)} time points,  t ∈ [{t.min():.2f}, {t.max():.2f}]")
print(f"x shape : {x.shape}  → {len(x)} spatial points, x ∈ [{x.min():.2f}, {x.max():.2f}]")
print(f"u shape : {u.shape}  → solution field (nx × nt)")
print(f"u range : [{u.min():.4f}, {u.max():.4f}]")
print("\n✓ Data loaded successfully — ready for Phase 3")