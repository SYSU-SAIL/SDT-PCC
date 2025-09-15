class SetConfig(object):
    def __init__(self, dataset_cfg='./dataset/lidar_cfg/Velodyne_HDL_64E.yaml', horizon_split=4, is_quantize=True,
                 quantize_step=0.01, critical_dist=100, is_normalize=True):
        self.dataset_cfg = dataset_cfg
        self.horizon_split = horizon_split
        self.is_quantize = is_quantize
        self.quantize_step = quantize_step
        self.critical_dist = critical_dist
        self.is_normalize = is_normalize

