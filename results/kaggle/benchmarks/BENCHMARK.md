# Kaggle benchmark: 1D heat PINN training speed

Measured with `experiments/00_env_benchmark.py` (repo commit 6a7f335).

**Environment:** Kaggle notebook, 2 x Tesla T4 (15 GB each), NVIDIA driver 580.178.04
(CUDA 13.0 as reported by nvidia-smi), TensorFlow 2.20.0, tf_keras 2.20.1,
tensorflow-probability 0.25.0, Keras 3.13.2, NumPy 2.1.3, DeepXDE 1.15.0, TensorFlow backend.

**Task:** 1D heat equation PINN, 4x64 tanh network, 2,900 training points
(2,500 interior + 200 boundary + 200 initial). 50 warm-up iterations excluded,
then 2,000 timed Adam iterations (lr 1e-3, seed 0), one configuration per process.

| Device | Precision | XLA | ms / iteration |
|---|---|---|---|
| CPU | float32 | auto (off on CPU) | 26.8 |
| CPU | float64 | auto (off on CPU) | 67.5 |
| GPU (T4) | float32 | auto (on for GPU) | 1.9 |
| GPU (T4) | float32 | off | 4.3 |
| GPU (T4) | float64 | auto (on for GPU) | 9.6 |
| GPU (T4) | float64 | off | 12.7 |

**Limits:** only Adam iteration speed was timed (not L-BFGS), for one problem size.
The L2 errors the script printed after 2,000 iterations were only a sanity check and
are not a comparison between configurations. The raw JSON files were lost when the
Kaggle session ended; the numbers above are copied from the printed summary.