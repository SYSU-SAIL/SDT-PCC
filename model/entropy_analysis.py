import math
import torch
import torch.nn as nn
from compressai.layers import ResidualBlock

from model.layer import conv2d


class EntropyDecomposition(nn.Module):
    def __init__(self, in_channels, kernel_size=(3, 3), stride=(3, 3)):
        super(EntropyDecomposition, self).__init__()
        self.decomposition_conv = nn.Sequential(
            conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride),
            nn.ReLU()
        )

    def forward(self, x):
        B, C, H, W = x.size()
        decomposed_x = self.decomposition_conv(x) + 1e-8
        res_x = torch.div(x, decomposed_x)
        decomposed = torch.stack((decomposed_x, res_x), dim=2)
        result = decomposed.reshape(B, -1, H, W)
        return result


class EntropyFusion(nn.Module):
    def __init__(self, in_channels, kernel_size=(3, 3), stride=(3, 3)):
        super(EntropyFusion, self).__init__()
        # self.after_conv = nn.Sequential(
        #     conv2d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=kernel_size,
        #            stride=stride),
        #     nn.ReLU(),
        #     conv2d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=kernel_size,
        #            stride=stride),
        #     nn.ReLU()
        # )
        self.after_conv = ResidualBlock(in_channels // 2, in_channels // 2)

    def forward(self, x):
        # 将通道维度拆分为奇数通道和偶数通道
        even_channels = x[:, 0::2, :, :]  # 偶数通道
        odd_channels = x[:, 1::2, :, :]  # 奇数通道
        # 逐元素相乘
        output_tensor = even_channels * odd_channels
        output_tensor = self.after_conv(output_tensor)
        return output_tensor

class ChannelQuantizer(nn.Module):
    def __init__(self, out_channels=4, quantize_step: float = 0.01, critical_dist: float = 100.0):
        super(ChannelQuantizer, self).__init__()
        self.out_channels = out_channels
        self.quantize_step = quantize_step
        self.critical_dist = critical_dist

        base = critical_dist / quantize_step
        self.quantize_level = math.ceil(base ** (1 / out_channels))  # base for decomposition

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantizes the input tensor x into multi-channel representation.
        Args:
            x: (B, H, W) float tensor
        Returns:
            quantized: (B, out_channels, H, W) long tensor
        """
        B, H, W = x.shape
        x_clipped = torch.clamp(x, 0, self.critical_dist)
        x_int = torch.round(x_clipped / self.quantize_step).long()  # Convert to integers

        x_flat = x_int.view(B, -1)
        codes = []
        for _ in range(self.out_channels):
            codes.append((x_flat % self.quantize_level).view(B, 1, H, W))
            x_flat = x_flat // self.quantize_level

        quantized = torch.cat(codes[::-1], dim=1)  # (B, out_channels, H, W)
        return quantized

    def dequantize(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the original float tensor from quantized codes.
        Args:
            quantized: (B, out_channels, H, W)
        Returns:
            dequantized: (B, H, W) float tensor
        """
        B, out_ch, H, W = quantized.shape
        assert out_ch == self.out_channels

        val = torch.zeros((B, H, W), dtype=torch.long, device=quantized.device)
        for i in range(self.out_channels):
            val *= self.quantize_level
            val += quantized[:, i]

        return val.float() * self.quantize_step

    def max_per_channel(self):
        return self.quantize_level - 1


class ChannelNormalizer(nn.Module):
    def __init__(self, max_val):
        super(ChannelNormalizer, self).__init__()
        self.max_val = max_val
        self.half_val = math.ceil(max_val / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self.half_val) / self.half_val
        return normalized

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        denormalized = (x * self.half_val) + self.half_val
        return denormalized
