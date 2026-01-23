import torch
import torch.nn as nn
import torch.nn.functional as F

class ImgCrop(nn.Module):
    def __init__(self, crop_factor: str = 'x1.4'):
        super().__init__()
        self.crop_factor = crop_factor
        self.h_factor = 1
        self.w_factor = 1
        if self.crop_factor == 'x1.4':
            self.h_factor = 1
            self.w_factor = 2
        elif self.crop_factor == 'x2':
            self.h_factor = 2
            self.w_factor = 2
        elif self.crop_factor == 'x2.5':
            self.h_factor = 2
            self.w_factor = 4
        else:
            print("⚠️ 非法裁切系数，默认不进行裁切")

    def image_crop(self, x):
        shape = x.shape
        if len(shape) == 3:
            B, H, W = shape
            x_down = x[:, ::self.h_factor, ::self.w_factor]
            h, w = x_down.shape[1:]
            pad_h = H - h
            pad_w = W - w
            x_padded = F.pad(x_down, (0, pad_w, 0, pad_h))
        elif len(shape) == 4:
            B, C, H, W = shape
            x_down = x[:, :, ::self.h_factor, ::self.w_factor]
            h, w = x_down.shape[2:]
            pad_h = H - h
            pad_w = W - w
            x_padded = F.pad(x_down, (0, pad_w, 0, pad_h))
        else:
            raise ValueError("只支持 [B,H,W] 或 [B,C,H,W] 维度的输入")
        return x_padded

    def image_depad(self, x):
        shape = x.shape
        if len(shape) == 3:
            B, H, W = shape
            h = H // self.h_factor
            w = W // self.w_factor
            x_depad = x[:, :h, :w]
        elif len(shape) == 4:
            B, C, H, W = shape
            h = H // self.h_factor
            w = W // self.w_factor
            x_depad = x[:, :, :h, :w]
        else:
            raise ValueError("只支持 [B,H,W] 或 [B,C,H,W] 维度的输入")
        return x_depad

    def image_decrop(self, x):
        shape = x.shape
        if len(shape) == 3:
            B, h, w = shape
            H, W = h * self.h_factor, w * self.w_factor
            x_decrop = torch.zeros((B, H, W), device=x.device, dtype=x.dtype)
            x_decrop[:, ::self.h_factor, ::self.w_factor] = x
        elif len(shape) == 4:
            B, C, h, w = shape
            H, W = h * self.h_factor, w * self.w_factor
            x_decrop = torch.zeros((B, C, H, W), device=x.device, dtype=x.dtype)
            x_decrop[:, :, ::self.h_factor, ::self.w_factor] = x
        else:
            raise ValueError("只支持 [B,H,W] 或 [B,C,H,W] 维度的输入")
        return x_decrop
