import numpy as np
from mpmath.libmp import normalize


class Normalizer:
    def __init__(self, critical_dist: float = 100, quantize_step: float = 0.001):
        self.critical_dist = critical_dist
        self.quantize_step = quantize_step
        self.normalize_bound = critical_dist / quantize_step

    def normalize(self, data, quantize=True):
        divisor = self.normalize_bound if quantize else self.critical_dist
        return data / divisor

    def denormalize(self, data, quantize=True):
        multiplier = self.normalize_bound if quantize else self.critical_dist
        return data * multiplier
