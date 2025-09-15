import collections
from typing import Tuple, Optional, Union

import math
import torch
from compressai.layers import ResidualBlock, ResidualBlockWithStride
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import multi_head_attention_forward
from torchvision.ops import DeformConv2d

from model.layer import conv2d


# B = batch_size
#
# N = num_heads
#
# H = hidden_size
#
# P = num_patches
#
# D = dim_per_head
#
# W = window_size
#
# L = local_patch

class SWDTWindowSplit(nn.Module):
    def __init__(self, in_channels: int, num_attention_heads: int, image_size: tuple[int, int],
                 patch_size: int, window_size: int, *args, **kwargs):
        """
        由于SWDT的windows分割
        :param in_channels: 输入通道数
        :param num_attention_heads: 头数
        :param image_size: 输入特征图大小
        :param patch_size: patch大小
        :param window_size: 局部窗口大小
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        if in_channels % num_attention_heads != 0:
            raise ValueError(
                f"请确保num_channels ({in_channels}) 可以被num_attention_heads ({num_attention_heads}) 整除。"
            )
        self.image_size = image_size
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_channels = in_channels
        self.num_attention_heads = num_attention_heads

        self.window_unfold = nn.Unfold(kernel_size=window_size, stride=patch_size,
                                       padding=(window_size - patch_size) // 2)

        self.window_fold = nn.Fold(
            output_size=image_size,
            kernel_size=window_size,
            stride=patch_size,
            padding=(window_size - patch_size) // 2
        )

        self.windows_num = None
        self.patch_num = None

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pixel_values.shape
        if H != self.image_size[0] or W != self.image_size[1]:
            raise ValueError(
                f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        if C != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels but got {C}."
            )
        windows = self.window_unfold(pixel_values)  # [B, C*W*W, P]
        if self.windows_num is None:
            self.windows_num = windows.shape[-1]
        windows = windows.reshape(B, C, self.window_size, self.window_size, self.windows_num)  # [B, C, W, W, P]
        return windows

    def inverse(self, windows: torch.Tensor) -> torch.Tensor:
        B = windows.shape[0]
        C = windows.shape[1]
        windows = windows.reshape(B, C * self.window_size * self.window_size, self.windows_num)
        image = self.window_fold(windows)  # [B, C, H, W]
        return image


class SWDTPatchEmbeddings(nn.Module):
    """
    这个类将形状为 `(batch_size, num_channels, height, width)` 的 `pixel_values` 转换为 Transformer 所需的
    初始 `hidden_states`（即 patch embeddings），其形状为 `(batch_size, seq_length, hidden_size)`。
    """

    def __init__(self, num_channels: int, hidden_size: int, image_size: Tuple[int, int], patch_size: int):
        super().__init__()
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size
        num_patches = (image_size[1] // patch_size) * (image_size[0] // patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)  # [B,P,H]
        return embeddings


class SWDTWindowValue(nn.Module):
    def __init__(self, in_channels: int, num_attention_heads: int, patch_size: int, local_stride: int):
        super().__init__()
        self.num_channels = in_channels
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.local_stride = local_stride
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=local_stride, padding=0)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, C, W, W, P = pixel_values.shape
        pixel_values = pixel_values.permute(0, 4, 1, 2, 3).reshape(B * P, C, W, W)  # [B*P,C,W,W]
        unfold_pixel = self.unfold(pixel_values)  # [B*P,?,L]
        L = unfold_pixel.shape[-1]
        output = unfold_pixel.permute(0, 2, 1).reshape(B, P, L, self.num_attention_heads, -1)  # [B,P,L,N,?]
        return output


class SWDTWindowEmbeddings(nn.Module):
    """
    这个类将形状为 `(batch_size, num_channels, height, width)` 的 `pixel_values` 转换为 Transformer 所需的
    初始 `hidden_states`（即 patch embeddings），其形状为 `(batch_size, seq_length, hidden_size)`。
    """

    def __init__(self, num_channels: int, hidden_size: int, patch_size: int, local_stride: int):
        super().__init__()
        patch_size = patch_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.local_stride = local_stride
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=local_stride)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, C, W, W, P = pixel_values.shape
        pixel_values = pixel_values.permute(0, 4, 1, 2, 3).reshape(B * P, C, W, W)
        embeddings = self.projection(pixel_values).flatten(2)  # [B*P,H,L]
        embeddings = embeddings.reshape(B, P, self.hidden_size, -1).permute(0, 1, 3, 2)  # [B,P,L,N]
        return embeddings


class SWDTMultiAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, patch_size: int, output_size, bias=True,
                 attention_probs_dropout_prob=0.0) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.patch_size = patch_size
        self.output_size = output_size
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=bias)
        # self.value = nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.output_fold = nn.Fold(output_size, kernel_size=patch_size, stride=patch_size)

    def query_transpose(self, query: torch.Tensor) -> torch.Tensor:
        """
        将输入query [B, P, H] 转换为 (batch_size,num_attention_heads, seq_length, attention_head_size)
        :param x:
        :return:
        """
        new_x_shape = query.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        query = query.view(new_x_shape)  # [B,P,N,D]
        return query.permute(0, 2, 1, 3)  # [B,N,P,D]

    def key_transpose(self, key: torch.Tensor) -> torch.Tensor:
        new_x_shape = key.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        key = key.view(new_x_shape)  # [B,P,L,N,D]
        return key.permute(0, 3, 1, 2, 4)  # [B,N,P,L,D]

    def forward(self, query, key, value) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # query [B,Hidden,P]
        # key [B, Hidden, P, W]

        key_layer = self.key_transpose(self.key(key))  # [B,N,P,L,D]
        query_layer = self.query_transpose(self.query(query))  # [B,N,P,D]
        query_layer = query_layer.unsqueeze(3)
        # 计算“query”和“key”之间的点积，以获得原始的注意力分数。
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [B, N, P, 1, L]
        attention_scores = attention_scores.squeeze(3)  # [B, N, P, L]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 将注意力分数归一化为概率值。
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs.permute(0, 2, 3, 1).unsqueeze(-1)  # [B,P,L,N,1]
        weight_value = value * attention_probs  # [B,P,L,N,Y]
        weight_value = weight_value.sum(dim=2)  # [B,P,N,Y]
        output_shape = weight_value.size()[:-2] + (-1,)
        output = weight_value.view(output_shape).permute(0, 2, 1)  # [B,?,P]
        output = self.output_fold(output)
        return output


class SWDTLayer(nn.Module):
    def __init__(self, in_channels, hidden_size: int, num_attention_heads: int, patch_size: int, window_size: int,
                 local_stride: int, conv_stride: int, conv_kernel: int, input_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.window_size = window_size
        self.local_stride = local_stride
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.input_size = input_size
        self.in_channels = in_channels
        self.window_split = SWDTWindowSplit(in_channels, num_attention_heads, input_size, patch_size, window_size)
        self.window_value = SWDTWindowValue(in_channels, num_attention_heads, patch_size, local_stride)
        self.window_embedding = SWDTWindowEmbeddings(in_channels, hidden_size, patch_size,local_stride)
        self.patch_embedding = SWDTPatchEmbeddings(in_channels, hidden_size, input_size, patch_size)
        self.multi_head_attention = SWDTMultiAttention(hidden_size, num_attention_heads, patch_size, input_size)
        self.res_blocks = nn.Sequential(
            ResidualBlock(in_channels, in_channels),
            nn.BatchNorm2d(in_channels),
            ResidualBlock(in_channels, in_channels),
            nn.BatchNorm2d(in_channels),
            ResidualBlock(in_channels, in_channels),
        )
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.offset_conv = nn.Sequential(
            conv2d(2 * in_channels, 2 * conv_kernel, (conv_kernel, conv_kernel), (1, 1)),
            conv2d(2 * conv_kernel, 2 * conv_kernel * conv_kernel, (conv_kernel, conv_kernel), (1, 1))
        )
        self.deform_conv = DeformConv2d(in_channels, in_channels, kernel_size=conv_kernel, padding=conv_kernel // 2)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        splited = self.window_split(memory)
        value = self.window_value(splited)
        key_embedding = self.window_embedding(splited)
        query_embedding = self.patch_embedding(tgt)
        attention_output = self.multi_head_attention(query_embedding, key_embedding, value)
        offset = self.offset_conv(torch.cat((tgt, attention_output), dim=1))
        deform_attention = self.deform_conv(attention_output, offset)
        layer_output = self.batch_norm(tgt + deform_attention)
        layer_output = self.res_blocks(layer_output)
        return layer_output
