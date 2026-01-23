import argparse
import os
import warnings
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

import resource

# ──────────────────────────────── 设置最大的打开文件数，比如 65535 ─────────────────────────────────
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
new_soft = 4096
resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard_limit))

parser = argparse.ArgumentParser()
parser.add_argument("--resume_path", type=str, default=None, help="还原检查点的路径")
parser.add_argument("--epoch", type=int, default=100, help="epoch的数量")
parser.add_argument("--batch_size", type=int, default=3, help="batch大小")
parser.add_argument("--num_workers", type=int, default=8, help="读取进程数量")
parser.add_argument("--gpu", nargs='+', help='Specify the GPU number to use', type=int)
args = parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))
    print(f'训练工作在设备: [{os.environ["CUDA_VISIBLE_DEVICES"]}]')

from config import config
from dataset.kitti import KittiDataset
from model.vic_model import ViCModel
from utils.ddp_trainer import DDPTrainer

if __name__ == '__main__':
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M")

    train_dataset = KittiDataset(config.kitti_train_config)
    val_dataset = KittiDataset(config.kitti_val_config)

    model = ViCModel(config.vic_config)
    trainer = DDPTrainer(model, train_dataset, val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                         save_dir=f"./checkpoints/{formatted_time}", log_dir=f"./logs/{formatted_time}",
                         resume_path=args.resume_path)
    trainer.train(args.epoch)
