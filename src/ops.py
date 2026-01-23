# ops.py
import os
import torch
from torch.utils.cpp_extension import load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

lidar_ops = load(
    name="lidar_ops",
    sources=[os.path.join(_THIS_DIR, "lidar_ops.cpp")],
    extra_cflags=["-O3", "-std=c++17"],
    with_cuda=False,  # 你当前是纯 CPU
    verbose=False,
)

__all__ = ["lidar_ops"]