import numpy as np
import torch
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import HyperpriorLatentCodec, GaussianConditionalLatentCodec, HyperLatentCodec
from compressai.layers import ResidualBlockWithStride, ResidualBlockUpsample, ResidualBlock
from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
from compressai.ops import quantize_ste
from torch import nn
from torch.nn import init

from model.SWDT import SWDTLayer
from model.ViT import ViTSelfLayer
from model.entropy_analysis import ChannelQuantizer, ChannelNormalizer
from model.rec_coder import ResCoder
from model.vic_config import ViConfig


def init_all_weights(m):
    if hasattr(m, 'weight') and m.weight is not None:
        init.kaiming_normal_(m.weight)  # 或你想用的其他初始化方式
    if hasattr(m, 'bias') and m.bias is not None:
        init.zeros_(m.bias)


def tuple2int(arg):
    return arg[0] if isinstance(arg, tuple) else arg


class ViCModel(CompressionModel):
    def __init__(self, config: ViConfig):
        super().__init__()
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 整理配置参数 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.config = config
        self.pixel_size = [config.image_size]
        temp_image_size = config.image_size
        for i in range(len(config.encoder_channels) - 1):
            temp_image_size = [x // 2 for x in temp_image_size]
            self.pixel_size.append(temp_image_size)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 量化器和归一化器 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        quantizer_channel = config.encoder_channels[0]
        self.quantizer = ChannelQuantizer(out_channels=quantizer_channel, quantize_step=config.quantize_step,
                                          critical_dist=config.critical_dist)
        max_val = self.quantizer.max_per_channel()
        self.normalizer = ChannelNormalizer(max_val=max_val)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 编码网络 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.encoder_layer_num = config.layer_num + 1
        self.encoder_layer = nn.ModuleList()
        for i, value in enumerate(self.config.encoder_channels[:-1]):
            self.encoder_layer.append(ResidualBlockWithStride(in_ch=value, out_ch=self.config.encoder_channels[i + 1]))
            for j in range(config.layer_num):
                self.encoder_layer.append(
                    ResidualBlock(self.config.encoder_channels[i + 1], self.config.encoder_channels[i + 1]))
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 解码网络 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.decode_channels = self.config.encoder_channels[::-1]
        self.decode_patch_size = self.config.patch_size[::-1]
        self.decode_windows_size = self.config.window_size[::-1]
        self.decode_local_stride = self.config.local_stride[::-1]
        self.decode_pixel_size = self.pixel_size[::-1]
        self.decode_attention_heads = self.config.attention_heads[::-1]
        self.decoder_layer = nn.ModuleList()
        self.decoder_layer_num = config.layer_num * 3 + 1
        for i, value in enumerate(self.decode_channels[:-1]):
            for j in range(config.layer_num):
                self.decoder_layer.append(ViTSelfLayer(num_channels=value, hidden_size=config.attention_hidden_size,
                                                       image_size=self.decode_pixel_size[i],
                                                       patch_size=self.decode_patch_size[i],
                                                       kernel_size=config.conv_kernel,
                                                       stride=config.conv_stride,
                                                       num_attention_heads=self.decode_attention_heads[i]))
            for j in range(config.layer_num):
                self.decoder_layer.append(SWDTLayer(in_channels=value, hidden_size=config.attention_hidden_size,
                                                    num_attention_heads=self.decode_attention_heads[i],
                                                    patch_size=tuple2int(self.decode_patch_size[i]),
                                                    window_size=tuple2int(self.decode_windows_size[i]),
                                                    local_stride=tuple2int(self.decode_local_stride[i]),
                                                    conv_stride=tuple2int(config.conv_stride),
                                                    conv_kernel=tuple2int(config.conv_kernel),
                                                    input_size=self.decode_pixel_size[i]))
            self.decoder_layer.append(ResidualBlockUpsample(in_ch=2 * value, out_ch=self.decode_channels[i + 1]))
            for j in range(config.layer_num):
                self.decoder_layer.append(ResidualBlock(self.decode_channels[i + 1], self.decode_channels[i + 1]))
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 高斯编码 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        h_a = nn.Sequential(
            conv(self.config.encoder_channels[-1], self.config.entropy_channels, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(self.config.entropy_channels, self.config.entropy_channels, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(self.config.entropy_channels, self.config.entropy_channels, stride=2, kernel_size=5),
        )
        h_s = nn.Sequential(
            deconv(self.config.entropy_channels, self.config.encoder_channels[-1], stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(self.config.encoder_channels[-1], self.config.encoder_channels[-1] * 3 // 2, stride=2,
                   kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(self.config.encoder_channels[-1] * 3 // 2, self.config.encoder_channels[-1] * 2, stride=1,
                 kernel_size=3),
        )
        self.latent_codec = HyperpriorLatentCodec(
            # A HyperpriorLatentCodec is made of "hyper" and "y" latent codecs.
            latent_codec={
                # Side-information branch with entropy bottleneck for "z":
                "hyper": HyperLatentCodec(
                    h_a=h_a,
                    h_s=h_s,
                    entropy_bottleneck=EntropyBottleneck(self.config.entropy_channels),
                ),
                # Encode y using GaussianConditional:
                "y": GaussianConditionalLatentCodec(),
            })
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 熵编码 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if self.config.multi_entropy_channel:
            self.entropy_bottleneck = nn.ModuleList()
            for q_ch in range(quantizer_channel):
                self.entropy_bottleneck.append(EntropyBottleneck(1))
        else:
            self.entropy_bottleneck = EntropyBottleneck(quantizer_channel)
        self.debug = None
        # ──────────────────────────────────── 裁切设置 ────────────────────────────────────
        self.res_coder = ResCoder(config.rec_coder)

    def forward(self, image, refer_image):
        B, H, W = image.shape
        if self.config.image_size[0] != H or self.config.image_size[1] != W:
            raise RuntimeError("The input image shape is incorrect.")
        if image.shape != refer_image.shape:
            raise RuntimeError("The input image does not match the shape of the reference image.")
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 数据量化和归一化 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        quantized_image = self.quantizer(image)
        quantized_refer = self.quantizer(refer_image)
        label = quantized_image.clone().detach().float()
        normalized_image = self.normalizer(quantized_image).detach()
        normalized_refer = self.normalizer(quantized_refer).detach()
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 当前帧编码 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        image_fea = normalized_image
        for i in range(0, len(self.encoder_layer), self.encoder_layer_num):
            current_index = i
            image_fea = self.encoder_layer[current_index](image_fea)
            current_index += 1
            for j in range(self.config.layer_num):
                image_fea = self.encoder_layer[current_index](image_fea)
                current_index += 1
        image_fea_out = self.latent_codec(image_fea)
        image_fea_hat = image_fea_out["y_hat"]

        refer_image_fea_list = []
        refer_image_fea = normalized_refer
        for i in range(0, len(self.encoder_layer), self.encoder_layer_num):
            current_index = i
            refer_image_fea = self.encoder_layer[current_index](refer_image_fea)
            current_index += 1
            for j in range(self.config.layer_num):
                refer_image_fea = self.encoder_layer[current_index](refer_image_fea)
                current_index += 1
            refer_image_fea_list.insert(0, refer_image_fea)

        image_fea_record = []
        for i in range(0, len(self.decoder_layer), self.decoder_layer_num):
            refer_index = i // self.decoder_layer_num
            current_index = i
            for j in range(self.config.layer_num):
                image_fea_hat = self.decoder_layer[current_index](image_fea_hat)
                current_index += 1
            cross_fea_hat = image_fea_hat
            for j in range(self.config.layer_num):
                cross_fea_hat = self.decoder_layer[current_index](cross_fea_hat, refer_image_fea_list[refer_index])
                current_index += 1
            fuse_fea = torch.cat([image_fea_hat, cross_fea_hat], dim=1)
            image_fea_hat = self.decoder_layer[current_index](fuse_fea)
            current_index += 1
            for j in range(self.config.layer_num):
                image_fea_hat = self.decoder_layer[current_index](image_fea_hat)
                current_index += 1
            image_fea_record.append(image_fea_hat)

        image_hat = image_fea_hat
        quantized_hat = self.normalizer.denormalize(image_hat)
        residual = quantized_hat - label
        # residual = image_hat - label
        out = image_fea_out
        if self.config.multi_entropy_channel:
            for layer_idx, bottleneck_layer in enumerate(self.entropy_bottleneck):
                _, entropy_likelihood = bottleneck_layer(residual[:, layer_idx:layer_idx+1, :, :])
                out["likelihoods"][f"res{str(layer_idx)}"] = entropy_likelihood
        else:
            _, entropy_likelihood = self.entropy_bottleneck(residual)
        return out

    def compress(self, image, refer_image):
        B, H, W = image.shape
        if image.shape != refer_image.shape:
            raise RuntimeError("The input image does not match the shape of the reference image.")
        if not (self.config.image_size[0] <= H and self.config.image_size[1] <= W):
            pad_h = self.config.image_size[0]-H
            pad_w = self.config.image_size[1]-W
            image = F.pad(image, (0, pad_w, 0, pad_h))
            refer_image = F.pad(refer_image, (0, pad_w, 0, pad_h))
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 数据量化和归一化 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        quantized_image = self.quantizer(image)
        quantized_refer = self.quantizer(refer_image)
        label = quantized_image.clone().detach().float()
        normalized_image = self.normalizer(quantized_image).detach()
        normalized_refer = self.normalizer(quantized_refer).detach()
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 当前帧编码 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        image_fea = normalized_image
        for i in range(0, len(self.encoder_layer), self.encoder_layer_num):
            current_index = i
            image_fea = self.encoder_layer[current_index](image_fea)
            current_index += 1
            for j in range(self.config.layer_num):
                image_fea = self.encoder_layer[current_index](image_fea)
                current_index += 1
        latent_codec = self.latent_codec.compress(image_fea)
        image_fea_hat = latent_codec["y_hat"]

        refer_image_fea_list = []
        refer_image_fea = normalized_refer
        for i in range(0, len(self.encoder_layer), self.encoder_layer_num):
            current_index = i
            refer_image_fea = self.encoder_layer[current_index](refer_image_fea)
            current_index += 1
            for j in range(self.config.layer_num):
                refer_image_fea = self.encoder_layer[current_index](refer_image_fea)
                current_index += 1
            refer_image_fea_list.insert(0, refer_image_fea)

        image_fea_record = []
        for i in range(0, len(self.decoder_layer), self.decoder_layer_num):
            refer_index = i // self.decoder_layer_num
            current_index = i
            for j in range(self.config.layer_num):
                image_fea_hat = self.decoder_layer[current_index](image_fea_hat)
                current_index += 1
            cross_fea_hat = image_fea_hat
            for j in range(self.config.layer_num):
                cross_fea_hat = self.decoder_layer[current_index](cross_fea_hat, refer_image_fea_list[refer_index])
                current_index += 1
            fuse_fea = torch.cat([image_fea_hat, cross_fea_hat], dim=1)
            image_fea_hat = self.decoder_layer[current_index](fuse_fea)
            current_index += 1
            for j in range(self.config.layer_num):
                image_fea_hat = self.decoder_layer[current_index](image_fea_hat)
                current_index += 1
            image_fea_record.append(image_fea_hat)

        image_hat = image_fea_hat
        quantized_hat = self.normalizer.denormalize(image_hat)
        quantized_hat = quantize_ste(quantized_hat)
        residual = quantized_hat - label
        residual = residual[:,:,:H,:W]
        residual_arr = residual.detach().cpu().numpy()
        residual_arr = residual_arr.astype(np.int8)
        zlib_codec = {"strings": [], "shape": residual_arr.shape[-2:]}
        # 存入一个占用指示
        shoot_exist = image[0] != 0
        shoot_exist = shoot_exist.cpu().numpy()
        if not (self.config.image_size[0] <= H and self.config.image_size[1] <= W):
            shoot_exist = shoot_exist[:H, :W]
        byte_data = shoot_exist.tobytes()
        compress_byte = self.res_coder.compress(byte_data)
        zlib_codec["strings"].append(compress_byte)
        for i in range(residual_arr.shape[1]):
            arr = residual_arr[0, i, :, :]
            byte_data = arr[shoot_exist].tobytes()
            compress_byte = self.res_coder.compress(byte_data)
            zlib_codec["strings"].append(compress_byte)
        # 使用zlib压缩
        codec_output = {"latent": latent_codec, "zlib": zlib_codec}
        return codec_output

    def decompress(self, refer_image, codec_res):
        B, H, W = refer_image.shape
        if not (self.config.image_size[0] <= H and self.config.image_size[1] <= W):
            pad_h = self.config.image_size[0]-H
            pad_w = self.config.image_size[1]-W
            refer_image = F.pad(refer_image, (0, pad_w, 0, pad_h))
        quantized_refer = self.quantizer(refer_image)
        normalized_refer = self.normalizer(quantized_refer).detach()

        latent_codec = self.latent_codec.decompress(
            strings=codec_res["latent"]["strings"],
            shape=codec_res["latent"]["shape"]
        )
        image_fea_hat = latent_codec["y_hat"]

        refer_image_fea_list = []
        refer_image_fea = normalized_refer
        for i in range(0, len(self.encoder_layer), self.encoder_layer_num):
            current_index = i
            refer_image_fea = self.encoder_layer[current_index](refer_image_fea)
            current_index += 1
            for j in range(self.config.layer_num):
                refer_image_fea = self.encoder_layer[current_index](refer_image_fea)
                current_index += 1
            refer_image_fea_list.insert(0, refer_image_fea)

        image_fea_record = []
        for i in range(0, len(self.decoder_layer), self.decoder_layer_num):
            refer_index = i // self.decoder_layer_num
            current_index = i
            for j in range(self.config.layer_num):
                image_fea_hat = self.decoder_layer[current_index](image_fea_hat)
                current_index += 1
            cross_fea_hat = image_fea_hat
            for j in range(self.config.layer_num):
                cross_fea_hat = self.decoder_layer[current_index](cross_fea_hat, refer_image_fea_list[refer_index])
                current_index += 1
            fuse_fea = torch.cat([image_fea_hat, cross_fea_hat], dim=1)
            image_fea_hat = self.decoder_layer[current_index](fuse_fea)
            current_index += 1
            for j in range(self.config.layer_num):
                image_fea_hat = self.decoder_layer[current_index](image_fea_hat)
                current_index += 1
            image_fea_record.append(image_fea_hat)

        image_hat = image_fea_hat
        quantized_hat = self.normalizer.denormalize(image_hat)
        quantized_hat = quantize_ste(quantized_hat)
        # 新增裁剪边
        if not (self.config.image_size[0] <= H and self.config.image_size[1] <= W):
            quantized_hat = quantized_hat[:,:,:H, :W]
        quantized_hat = quantized_hat.detach().cpu().numpy()
        quantized_hat = quantized_hat.astype(np.int32)
        residual = np.zeros_like(quantized_hat, dtype=np.int8)
        bytes_data = self.res_coder.decompress(codec_res["zlib"]["strings"][0])
        shoot_exist = np.frombuffer(bytes_data, dtype=np.bool)
        shoot_exist = shoot_exist.reshape(codec_res["zlib"]["shape"])
        for i in range(residual.shape[1]):
            bytes_data = self.res_coder.decompress(codec_res["zlib"]["strings"][i+1])
            arr = np.frombuffer(bytes_data, dtype=np.int8)
            residual[0, i][shoot_exist] = arr
        residual = residual.astype(np.int32)
        y_hat = torch.tensor(quantized_hat - residual, dtype=torch.long)
        y_hat = self.quantizer.dequantize(y_hat)
        mask = torch.from_numpy(shoot_exist.copy()).to(y_hat.device)
        mask = mask.unsqueeze(0)  # [B, H, W]
        # 掩码为 False 的位置设置为 0
        y_hat = y_hat.masked_fill(~mask,0)
        codec_output = {"y_hat": y_hat}
        return codec_output
