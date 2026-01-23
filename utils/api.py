import numpy as np
import torch

from dataset.set_config import SetConfig
from model.vic_config import ViConfig
from model.vic_model import ViCModel
from src.transformer import PCTransformer
import open3d as o3d


class PCC4JUN_API:
    def __init__(self, model_cfg: ViConfig, lidar_cfg: SetConfig,
                 checkpoint: str = "checkpoints/20250620_2128/model_best.pth", cuda: bool = True):
        self.model = ViCModel(model_cfg)
        self.pc_transformer = PCTransformer(lidar_cfg.dataset_cfg)
        self.cuda = cuda
        self.lidar_cfg = lidar_cfg
        checkpoint = torch.load(checkpoint, weights_only=True,map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.update()
        torch.use_deterministic_algorithms(True)  # 避免不确定性
        self.model.eval()

    def range_reshape(self, range: np.ndarray):
        # 沿着第二维度切割成4份
        splits = np.split(range, self.lidar_cfg.horizon_split, axis=1)
        # 沿着第一维度进行堆叠
        stacked_array = np.vstack(splits)
        return stacked_array

    def range_reshape_inverse(self, stacked_array: np.ndarray):
        # 获取原始高度（堆叠前单个split的高度）
        original_height = stacked_array.shape[0] // self.lidar_cfg.horizon_split
        # 沿着第一维度切割成 horizon_split 份
        splits = np.split(stacked_array, self.lidar_cfg.horizon_split, axis=0)
        # 沿着第二维度进行拼接（水平方向）
        original_array = np.concatenate(splits, axis=1)
        return original_array

    def get_range_image(self, points: np.ndarray, reshape=True):
        range_image = self.pc_transformer.point_cloud_to_range_image(points)
        if reshape:
            range_image = self.range_reshape(range_image)
        return range_image

    def get_points_from_range_image(self, range_image: np.ndarray, reshape=True):
        if reshape:
            range_image = self.range_reshape_inverse(range_image)
        points = self.pc_transformer.range_image_to_point_cloud(range_image)
        points = points.reshape((-1, 3))
        points = points[~(points == 0).all(axis=1)]
        return points

    def voxel_down_sample(self, points: np.ndarray, voxel_size=0.1):
        # 将 ndarray 转换为 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # 设置体素大小并执行降采样
        down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        # 将降采样后的点云转换回 numpy.ndarray
        downsampled_points = np.asarray(down_pcd.points)
        return downsampled_points

    def compress(self, points: np.ndarray, ref_points: np.ndarray, downsample_voxel=0.0, need_trans: bool = False):
        if downsample_voxel > 0:
            points = self.voxel_down_sample(points, downsample_voxel)
        range_image = self.get_range_image(points)
        refer_range_image = self.get_range_image(ref_points)
        range_image_torch = torch.from_numpy(range_image).to(self.device).unsqueeze(dim=0)
        refer_range_image_torch = torch.from_numpy(refer_range_image).to(self.device).unsqueeze(dim=0)
        with torch.no_grad():
            output = self.model.compress(range_image_torch, refer_range_image_torch)
        if need_trans:
            trans_points = self.get_points_from_range_image(range_image)
            return output, trans_points
        else:
            return output

    def decompress(self, codec_output: list, ref_points: np.ndarray):
        refer_range_image = self.get_range_image(ref_points)
        refer_range_image_torch = torch.from_numpy(refer_range_image).to(self.device).unsqueeze(dim=0)
        with torch.no_grad():
            decoded = self.model.decompress(refer_range_image_torch, codec_output)
            dec_img = decoded["y_hat"].squeeze(dim=0).cpu().numpy()
        dec_points = self.get_points_from_range_image(dec_img)
        return dec_points
