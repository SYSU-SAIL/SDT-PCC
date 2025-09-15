from typing import Tuple

from torch import nn


def conv2d(in_channels, out_channels, kernel_size: Tuple[int, int], stride: Tuple[int, int]):
    kh, kw = kernel_size
    ph = (kh - 1) // 2
    pw = (kw - 1) // 2
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=(ph, pw))
