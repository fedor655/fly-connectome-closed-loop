"""Проба окружения: какие пакеты есть в текущем conda-env и виден ли GPU."""
import importlib
import sys

print("python", sys.version.split()[0])
print("executable", sys.executable)

MODULES = [
    "numpy", "pandas", "scipy", "torch", "mujoco", "flygym",
    "imageio", "imageio_ffmpeg", "PIL", "brian2", "gymnasium",
]

for name in MODULES:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:16s} OK    {getattr(mod, '__version__', '?')}")
    except Exception as exc:
        print(f"  {name:16s} MISS  {type(exc).__name__}")

try:
    import torch
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_device:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch probe failed:", exc)
