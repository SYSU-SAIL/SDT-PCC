import os

import numpy as np
from torch.utils.data import Dataset

from dataset.set_config import SetConfig
from src.transformer import PCTransformer
from utils.normalizer import Normalizer
from utils.quantizer import Quantizer

EXTENSIONS_SCAN = ['.bin', '.npy', '.npz']


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


class KittiConfig(SetConfig):
    def __init__(self, root: str, sequences, **kwargs):
        super().__init__(**kwargs)
        self.root = root
        self.sequences = sequences


class KittiDataset(Dataset):
    def __init__(self, kitti_config: KittiConfig):
        self.kitti_config = kitti_config
        root_dir = os.path.expanduser(kitti_config.root)
        self.root = os.path.join(root_dir, "sequences")
        self.PCTransformer = PCTransformer(kitti_config.dataset_cfg)
        self.transform_map = self.PCTransformer.transform_map
        self.quantizer = Quantizer(step=kitti_config.quantize_step, critical_dist=kitti_config.critical_dist)
        self.normalizer = Normalizer(quantize_step=kitti_config.quantize_step, critical_dist=kitti_config.critical_dist)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 获得文件序列 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.scan_files = []
        for seq in kitti_config.sequences:
            # to string
            seq = '{0:02d}'.format(int(seq))
            # get paths for each
            scan_path = os.path.join(self.root, seq, "velodyne")
            if not os.path.exists(scan_path):
                raise RuntimeError('Path {} does not exist'.format(scan_path))
            # get files
            scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(scan_path)) for f in fn if is_scan(f)]

            # extend list
            self.scan_files.extend(scan_files)
        self.scan_files.sort()

    def _read_points(self, index: int):
        """
        读取点云数据
        :param index: 索引
        :return: 点云数据
        """
        scan_file = self.scan_files[index]
        # 测试文件名是否正确
        if not isinstance(scan_file, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(scan_file))))
        # 检查拓展名
        if not any(scan_file.endswith(ext) for ext in EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")
        # 读取点云文件
        scan = np.fromfile(scan_file, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        # 提取点云
        points = scan[:, 0:3]  # get xyz
        return points

    def _range_reshape(self, range_image: np.ndarray):
        # 沿着第二维度切割成4份
        splits = np.split(range_image, self.kitti_config.horizon_split, axis=1)
        # 沿着第一维度进行堆叠
        stacked_array = np.vstack(splits)
        return stacked_array

    def __len__(self):
        return len(self.scan_files) - 1

    def __getitem__(self, index):
        dir_name0 = os.path.dirname(self.scan_files[index])
        dir_name1 = os.path.dirname(self.scan_files[index + 1])
        if dir_name0 != dir_name1:
            index -= 2

        tag = ["refer_image", "image"]
        output = {}
        for tag_index, tag_name in enumerate(tag):
            points = self._read_points(index + tag_index)
            range_image = self.PCTransformer.point_cloud_to_range_image(points)
            range_image = self._range_reshape(range_image)
            if self.kitti_config.is_quantize:
                range_image = self.quantizer.quantize(range_image)
            if self.kitti_config.is_normalize:
                range_image = self.normalizer.normalize(range_image)
            output[tag_name] = range_image
        return output
