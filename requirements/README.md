# Fixed environments

The curriculum targets CPython 3.12.13. Create separate virtual environments;
do not install PyTorch and TensorFlow merely to solve NumPy-only problems.

| File | Intended machine | Notes |
|---|---|---|
| `core-cpu.txt` | Any supported CPU | NumPy, SciPy, and diagnostic plotting |
| `pytorch-cpu.txt` | CPU-only | PyTorch boundary/oracle problems |
| `pytorch-gpu-cu130.txt` | NVIDIA GPU | CUDA 13.0 wheel index; driver compatibility is an external prerequisite |
| `tensorflow-cpu.txt` | CPU-only | TensorFlow boundary/oracle problems |
| `tensorflow-gpu-linux.txt` | Linux/WSL2 NVIDIA GPU | Installs TensorFlow's `and-cuda` extra; host driver remains external |

GPU environments are listed for reproducibility, but Chapters 1--2 grade no
GPU speed. A GPU must not change numerical contracts or test interpretation.
