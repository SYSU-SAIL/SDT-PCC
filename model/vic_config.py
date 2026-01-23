from typing import Tuple, Union, Literal


class ViConfig:
    def __init__(self, encoder_channels=None,
                 conv_kernel: Union[Tuple[int, int], int] = (3, 3), conv_stride: Union[Tuple[int, int]] = (1, 1),
                 window_size=None, local_stride=None,
                 attention_hidden_size: int = 256,
                 image_size: Tuple[int, int] = (256, 512),
                 patch_size=None, attention_heads=None, layer_num: int = 3,
                 quantize_step=0.005,
                 critical_dist=100,
                 entropy_channels: int = 64,
                 multi_entropy_channel:bool = False,
                 rec_coder: Literal['bzip', 'zlib', 'lz']='zlib'):
        if attention_heads is None:
            attention_heads = [1, 2, 4, 8]
        if encoder_channels is None:
            encoder_channels = [3, 16, 32, 64]
        if patch_size is None:
            patch_size = [16, 16, 8, 8]
        if window_size is None:
            window_size = [32, 32, 16, 16]
        if local_stride is None:
            local_stride = [2, 2, 1, 1]
        self.encoder_channels = encoder_channels  # 编码通道数
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.window_size = window_size
        self.local_stride = local_stride
        self.attention_hidden_size = attention_hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.attention_heads = attention_heads
        self.layer_num = layer_num
        self.quantize_step = quantize_step
        self.critical_dist = critical_dist
        self.entropy_channels = entropy_channels
        self.multi_entropy_channel = multi_entropy_channel
        self.rec_coder = rec_coder