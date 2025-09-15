import torch
import numpy as np

class Quantizer(object):
    def __init__(self, step=0.01, critical_dist: float = 100):
        self.step = step
        self.critical_dist = critical_dist

    def quantize(self, value):
        # 自动选择 torch 或 numpy 操作
        if isinstance(value, torch.Tensor):
            clipped = torch.clamp(value, 0, self.critical_dist)
            quantized = torch.round(clipped / self.step)
        else:
            clipped = np.clip(value, 0, self.critical_dist)
            quantized = np.round(clipped / self.step)
        return quantized

    def dequantize(self, value):
        if isinstance(value, torch.Tensor):
            quantized = torch.round(value)
            return quantized * self.step
        else:
            quantized = np.round(value)
            return quantized * self.step

