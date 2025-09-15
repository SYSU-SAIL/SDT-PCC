import collections
from typing import Tuple, Optional, Union

import math
import torch
from compressai.layers import ResidualBlock
from torch import nn
import torch.nn.functional as F

from model.layer import conv2d


class ViTPatchEmbeddings(nn.Module):
    """
    这个类将形状为 `(batch_size, num_channels, height, width)` 的 `pixel_values` 转换为 Transformer 所需的
    初始 `hidden_states`（即 patch embeddings），其形状为 `(batch_size, seq_length, hidden_size)`。
    """

    def __init__(self, num_channels: int, hidden_size: int, image_size: Union[Tuple[int, int], int],
                 patch_size: Union[Tuple[int, int], int]):
        super().__init__()

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
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
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class ViTPatchValue(nn.Module):
    def __init__(self, num_channels: int, num_attention_heads: int, image_size: Union[Tuple[int, int], int],
                 patch_size: Union[Tuple[int, int], int], kernel_size: Union[Tuple[int, int], int],
                 stride: Union[Tuple[int, int], int], *args, **kwargs):
        """
        给定 x (batch_size, num_channels, height, width) 输入，将其转化为 value 可用的 (batch_size,num_attention_heads, seq_length, feature_len(kernel*kernel))格式
        :param num_channels: 通道数
        :param num_attention_heads: 多头注意力
        :param image_size: 特征图形状
        :param patch_size: patch形状
        :param kernel_size: 卷积核大小
        :param stride: 卷积步长
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        kernel_size = kernel_size if isinstance(kernel_size, collections.abc.Iterable) else (kernel_size, kernel_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        if num_channels % num_attention_heads != 0:
            raise ValueError(
                f"请确保num_channels ({num_channels}) 可以被num_attention_heads ({num_attention_heads}) 整除。"
            )
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.num_attention_heads = num_attention_heads
        self.projection = nn.Sequential(conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=stride),
                                        ResidualBlock(num_channels, num_channels))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model"
                f" ({self.image_size[0]}*{self.image_size[1]})."
            )

        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values)  # (batch_size, num_channels, height, width)
        embeddings = F.relu(embeddings)
        embeddings = F.unfold(embeddings, kernel_size=self.patch_size,
                              stride=self.patch_size)  # (batch_size, num_channels*p_s*p_s, seq_length)
        embeddings = embeddings.permute(0, 2, 1)  # (batch_size, seq_length, num_channels*p_s*p_s)
        new_embedding_shape = embeddings.size()[:-1] + (self.num_attention_heads, -1)
        embeddings = embeddings.view(
            new_embedding_shape)  # (batch_size, seq_length,num_attention_heads, num_channels*p_s*p_s/num_attention_heads)
        return embeddings.permute(0, 2, 1, 3)

    def inverse_pixel_values(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        将ViTMultiAttention输出的加权特征转换为特征图
        :param embeddings: 特征嵌入
        :return: 特征图
        """
        B, num_heads, N_patches, dim_per_head = embeddings.shape
        # Step 1: (B, num_heads, N_patches, dim_per_head) -> (B, N_patches, num_heads, dim_per_head)
        embeddings = embeddings.permute(0, 2, 1, 3)

        # Step 2: merge heads back -> (B, N_patches, C * patch_h * patch_w)
        embeddings = embeddings.contiguous().view(B, N_patches, -1)

        # Step 3: (B, N_patches, C * patch_h * patch_w) -> (B, C * patch_h * patch_w, N_patches)
        embeddings = embeddings.permute(0, 2, 1)

        # Step 4: Use F.fold to reverse unfold
        output = F.fold(
            embeddings, output_size=self.image_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        return output


class ViTMultiAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, bias=True, attention_probs_dropout_prob=0.0) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=bias)
        # self.value = nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入x (batch_size, seq_length, hidden_size) 转换为 (batch_size,num_attention_heads, seq_length, attention_head_size)
        :param x:
        :return:
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self, query, key, value, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        key_layer = self.transpose_for_scores(self.key(key))
        # value_layer = self.transpose_for_scores(self.value(value))
        value_layer = value
        query_layer = self.transpose_for_scores(self.query(query))
        # 计算“query”和“key”之间的点积，以获得原始的注意力分数。
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 将注意力分数归一化为概率值。
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        #      ┌──────────────────────────────────────────────────────────┐
        #      │  这实际上是丢弃了部分需要关注的整个 token，虽然看起来可能有点不寻常  │
        #      │  但这是源自原始 Transformer 论文的做法。                      │
        #      └──────────────────────────────────────────────────────────┘
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class ViTAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, bias=True, attention_probs_dropout_prob=0.0) -> None:
        super().__init__()
        self.attention = ViTMultiAttention(hidden_size, num_attention_heads, bias, attention_probs_dropout_prob)

    def forward(
            self,
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        attention_output = self.attention(query, key, value, head_mask, output_attentions)
        return attention_output


class ViTCrossLayer(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, image_size: Union[Tuple[int, int], int],
                 patch_size: Union[Tuple[int, int], int], num_attention_heads: int, bias=True,
                 attention_probs_dropout_prob=0.0, kernel_size: Union[Tuple[int, int], int] = (3, 3),
                 stride: Union[Tuple[int, int], int] = (1, 1)) -> None:
        """
        ViT panel 交出注意力层
        :param num_channels: 输入通道数量
        :param hidden_size: 嵌入层通道数量
        :param image_size: 图像大小
        :param patch_size: 分块大小
        :param num_attention_heads: 多头注意力
        :param bias: 是否需要偏置
        :param attention_probs_dropout_prob: dropout概率
        :param kernel_size: 卷积核大小
        :param stride: 卷积核步长
        """
        super().__init__()
        self.attention = ViTAttention(hidden_size, num_attention_heads, bias, attention_probs_dropout_prob)
        self.vit_patch_embedding = ViTPatchEmbeddings(num_channels, hidden_size, image_size, patch_size)
        self.vit_patch_value = ViTPatchValue(num_channels, num_attention_heads, image_size, patch_size, kernel_size,
                                             stride)
        self.conv_after = nn.Sequential(conv2d(num_channels, num_channels, kernel_size, stride),
                                        nn.ReLU(),
                                        conv2d(num_channels, num_channels, kernel_size, stride),
                                        nn.ReLU())
        self.norm_before = nn.LayerNorm(hidden_size)
        self.norm_after = nn.BatchNorm2d(num_channels)

    def forward(
            self,
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False) -> torch.Tensor:
        query_state = self.norm_before(self.vit_patch_embedding(query))
        key_state = self.norm_before(self.vit_patch_embedding(key))
        value_state = self.vit_patch_value(value)
        cross_attention_outputs = self.attention(
            self.norm_before(query_state),  # in ViT, layernorm is applied before self-attention
            self.norm_before(key_state),
            value_state,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        # first residual connection
        attention_output = self.vit_patch_value.inverse_pixel_values(attention_output)
        attention_output = self.conv_after(attention_output)
        hidden_states = attention_output + query
        # in ViT, layernorm is also applied after self-attention
        layer_output = self.norm_after(hidden_states)
        return layer_output


class ViTSelfLayer(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, image_size: Union[Tuple[int, int], int],
                 patch_size: Union[Tuple[int, int], int], num_attention_heads: int, bias=True,
                 attention_probs_dropout_prob=0.0, kernel_size: Union[Tuple[int, int], int] = (3, 3),
                 stride: Union[Tuple[int, int], int] = (1, 1), *args, **kwargs):
        """
        ViT panel 自注意力层
        :param num_channels: 输入通道数量
        :param hidden_size: 嵌入层通道数量
        :param image_size: 图像大小
        :param patch_size: 分块大小
        :param num_attention_heads: 多头注意力
        :param bias: 是否需要偏置
        :param attention_probs_dropout_prob: dropout概率
        :param kernel_size: 卷积核大小
        :param stride: 卷积核步长
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.attention = ViTAttention(hidden_size, num_attention_heads, bias, attention_probs_dropout_prob)
        self.vit_patch_embedding = ViTPatchEmbeddings(num_channels, hidden_size, image_size, patch_size)
        self.vit_patch_value = ViTPatchValue(num_channels, num_attention_heads, image_size, patch_size, kernel_size,
                                             stride)
        self.conv_after = nn.Sequential(nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding="same"),
                                        nn.ReLU(),
                                        nn.Conv2d(num_channels, num_channels, kernel_size, stride, padding="same"),
                                        nn.ReLU())
        self.norm_before = nn.LayerNorm(hidden_size)
        self.norm_after = nn.BatchNorm2d(num_channels)

    def forward(self, pixel_value: torch.Tensor, head_mask: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> torch.Tensor:
        query_state = self.norm_before(self.vit_patch_embedding(pixel_value))
        key_state = self.norm_before(self.vit_patch_embedding(pixel_value))
        value_state = self.vit_patch_value(pixel_value)
        cross_attention_outputs = self.attention(
            self.norm_before(query_state),  # in ViT, layernorm is applied before self-attention
            self.norm_before(key_state),
            value_state,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        # first residual connection
        attention_output = self.vit_patch_value.inverse_pixel_values(attention_output)
        attention_output = self.conv_after(attention_output)
        hidden_states = attention_output + pixel_value
        # in ViT, layernorm is also applied after self-attention
        layer_output = self.norm_after(hidden_states)
        return layer_output
