import numpy as np
import open3d as o3d

def _fix_ply_header(ply_path: str, has_normals: bool) -> None:
    """修正PLY文件头属性定义 (强制使用float类型)"""
    with open(ply_path, 'r') as f:
        lines = f.readlines()

    # 修改坐标属性定义
    lines[4] = 'property float x\n'
    lines[5] = 'property float y\n'
    lines[6] = 'property float z\n'

    # 修改法向量属性定义 (如果存在)
    if has_normals:
        lines[7] = 'property float nx\n'
        lines[8] = 'property float ny\n'
        lines[9] = 'property float nz\n'

    # 写回文件
    with open(ply_path, 'w') as f:
        f.writelines(lines)


def covert_point_ply(points: np.ndarray, file_path, normal=False):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if normal:
        point_cloud.estimate_normals()
    # 保存为 ply 格式
    o3d.io.write_point_cloud(file_path, point_cloud, write_ascii=True)
    # 修正PLY文件头 (参考你的原始方法)
    _fix_ply_header(file_path, has_normals=normal)