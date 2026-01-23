from dataset.kitti import KittiConfig
from model.vic_model import ViConfig


class Config:
    def __init__(self):
        self.kitti_train_config = KittiConfig("~/Dataset/data_odometry_velodyne/dataset",
                                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], quantize_step=0.005,
                                              is_normalize=False, is_quantize=False)
        self.kitti_val_config = KittiConfig("~/Dataset/data_odometry_velodyne/dataset", [11, 12, 13],
                                            quantize_step=0.005, is_normalize=False, is_quantize=False)
        # ──────────────────────────────────── Lava ────────────────────────────────────
        self.vic_config = ViConfig(encoder_channels=[2, 32, 64, 128, 128],
                                   conv_kernel=(3, 3),
                                   conv_stride=(1, 1),
                                   window_size=[32, 32, 16, 16],
                                   local_stride=[2, 2, 1, 1],
                                   attention_hidden_size=128,
                                   image_size=(256, 512),
                                   patch_size=[16, 16, 8, 8],
                                   attention_heads=[2, 8, 16, 32, 32],
                                   layer_num=6,
                                   quantize_step=0.005,
                                   critical_dist=100,
                                   entropy_channels=128,
                                   multi_entropy_channel=True)


config = Config()
