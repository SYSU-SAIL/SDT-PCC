import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from pcc_eval.nju import metric
from tqdm import tqdm

from dataset.set_config import SetConfig
from model.vic_config import ViConfig
from src.transformer import PCTransformer
from utils.api import PCC4JUN_API

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, required=True, help="输入点云的文件夹路径")
parser.add_argument("--checkpoint", type=str, default=None, help="还原检查点的路径")
parser.add_argument("-s", "--sample", action="store_true", help="是否对测试数据进行采样")
parser.add_argument("-f", "--front", type=int, default=0, help="从前开始采样n个样本")
parser.add_argument("-b", "--beside", type=int, default=0, help="从后开始采样n个样本")
parser.add_argument("-q", "--quantize_step", type=float, default=0.01, help="量化步长")
parser.add_argument("-p", "--prefix", type=str, default=None, help="保存文件的前缀")
parser.add_argument("-c", "--coder", type=str, default="zlib", help="使用的RES编码器")
args = parser.parse_args()


def read_points(file_path):
    """
    读取点云数据
    :param index: 索引
    :return: 点云数据
    """
    # 测试文件名是否正确
    if not isinstance(file_path, str):
        raise TypeError("Filename should be string type, "
                        "but was {type}".format(type=str(type(file_path))))
    # 检查拓展名
    if not any(file_path.endswith(ext) for ext in ['.bin', '.npy', '.npz']):
        raise RuntimeError("Filename extension is not valid scan file.")
    # 读取点云文件
    scan = np.fromfile(file_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    # 提取点云
    points = scan[:, 0:3]  # get xyz
    return points


def write_to_csv(point_file, eval_res, encode_time, dec_time, file_path):
    """批量写入 CSV"""
    file_exists = os.path.exists(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ['File', 'Bpp', 'D1_MSEF', 'D1_PSNR', 'D2_MSEF', 'D2_PSNR', 'Chamfer', 'Enc Time', 'Dec Time'])  # 写入标题行

        writer.writerow(
            [point_file, eval_res.bpp, eval_res.d1_msef, eval_res.d1_psnr, eval_res.d2_msef, eval_res.d2_psnr,
             eval_res.chamfer, encode_time, dec_time])


def main(args):
    csv_file_name = f"{datetime.now().strftime('%Y%m%d-%H%M')}.csv"
    if args.prefix:
        csv_file_name = f"{args.prefix}-{csv_file_name}"
    res_coder = args.coder
    model_config = ViConfig(encoder_channels=[2, 32, 64, 128, 128],
                            conv_kernel=(3, 3),
                            conv_stride=(1, 1),
                            window_size=[32, 32, 16, 16],
                            local_stride=[2, 2, 1, 1],
                            attention_hidden_size=128,
                            image_size=(256, 512),
                            patch_size=[16, 16, 8, 8],
                            attention_heads=[2, 8, 16, 32, 32],
                            layer_num=6,
                            quantize_step=args.quantize_step,
                            critical_dist=100,
                            entropy_channels=128,
                            rec_coder=res_coder)
    lidar_cfg = SetConfig(dataset_cfg="dataset/lidar_cfg/Velodyne_HDL_64E.yaml",
                          is_quantize=False,
                          quantize_step=args.quantize_step,
                          critical_dist=100,
                          is_normalize=False)
    eval_pc_transformer = PCTransformer(lidar_cfg.dataset_cfg)
    pcc_api = PCC4JUN_API(model_config, lidar_cfg, "checkpoints/20250620_2128/model_best.pth")
    points_dir = Path(os.path.expanduser(args.input_dir))
    files = list(points_dir.glob("*.bin")) + list(points_dir.glob("*.ply"))
    # 按照文件名（去掉后缀）转为整数排序
    files_sorted = sorted(files, key=lambda f: int(f.stem))
    if args.sample:
        if args.beside == 0 and args.front == 0:
            files_sorted = files_sorted[:(100 + 1)]
        elif args.front != 0:
            files_sorted = files_sorted[:(args.front + 1)]
        else:
            files_sorted = files_sorted[-(args.beside + 1):]

    pbar = tqdm(files_sorted, desc="Evaluating", dynamic_ncols=True)
    for i, point_file in enumerate(pbar):
        refer_index = i - 1
        if refer_index >= 0:
            refer_file = files_sorted[refer_index]
            refer_points = read_points(str(refer_file))
            points = read_points(str(point_file))
            encode_start_time = time.time()  # 记录开始时间
            output = pcc_api.compress(points, refer_points)
            encode_time = time.time() - encode_start_time  # 记录结束时间
            # 获取量化点云
            trans_img = eval_pc_transformer.point_cloud_to_range_image(points)
            trans_points = eval_pc_transformer.range_image_to_point_cloud(trans_img)
            trans_points = trans_points.reshape((-1, 3))
            trans_points = trans_points[~(trans_points == 0).all(axis=1)]
            # 计算BPP指标
            points_num = trans_points.shape[0]
            bytes = 0
            for strings in output["latent"]["strings"]:
                bytes += len(strings)
            for strings in output["zlib"]["strings"]:
                bytes += len(strings)
            bits = bytes * 8
            bpp = bits / points_num

            # 点云解码
            dec_start_time = time.time()
            dec_points = pcc_api.decompress(output, refer_points)
            dec_time = time.time() - dec_start_time
            eval_res = metric(trans_points, dec_points, normal=True)
            eval_res.bpp = bpp
            write_to_csv(point_file, eval_res, encode_time, dec_time, csv_file_name)
            pbar.set_postfix({
                "bpp": f"{bpp:.4f}",
                "D1-PSNR": f"{eval_res.d1_psnr:.4f}",
                "D2-PSNR": f"{eval_res.d2_psnr:.4f}",
            })


if __name__ == '__main__':
    main(args)
