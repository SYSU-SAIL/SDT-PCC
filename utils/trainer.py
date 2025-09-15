import os
from typing import Optional, Callable, Literal

import torch
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.criterion import RateLoss


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_dataset: torch.utils.data.Dataset,
            val_dataset: Optional[torch.utils.data.Dataset] = None,
            batch_size: int = 32,
            num_workers: int = 4,
            optimizer_conf=None,
            clip_max_norm=1.0,
            criterion: Callable = RateLoss(),
            log_interval: int = 10,
            save_dir: str = "./checkpoints",
            log_dir: str = "./logs",
            use_lr_scheduler: bool = False,
            scheduler_mode: Literal["min", "max"] = "min",  # "min"或"max"
            scheduler_factor: float = 0.1,
            scheduler_patience: int = 5,
            scheduler_threshold: float = 1e-4,
            scheduler_cooldown: int = 3,
            scheduler_min_lr: float = 1e-6,
            max_checkpoints: int = 3,  # 最大保留的检查点数量
            cuda=True
    ):
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 参数初始化 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.epochs = None
        self.save_interval = None
        if optimizer_conf is None:
            optimizer_conf = {"net": {"type": "Adam", "lr": 0.0001},
                              "aux": {"type": "Adam", "lr": 0.001}}
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.optimizer_conf = optimizer_conf
        self.clip_max_norm = clip_max_norm
        self.criterion = criterion
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.writer = None
        self.use_lr_scheduler = use_lr_scheduler
        self.scheduler_mode = scheduler_mode
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_threshold = scheduler_threshold
        self.scheduler_cooldown = scheduler_cooldown
        self.scheduler_min_lr = scheduler_min_lr
        self.max_checkpoints = max_checkpoints
        self.cuda = cuda
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 构建相关类 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

        # 初始化SummaryWriter
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.device_model = self.model.to(self.device)

        # 准备优化器
        optimizers = net_aux_optimizer(model, self.optimizer_conf)
        self.optimizer, self.aux_optimizer = optimizers["net"], optimizers["aux"]

        # 数据读取部分
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True
            )

            # 准备学习率调度器
        self.scheduler = None
        if self.use_lr_scheduler and self.val_dataset is not None:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=self.scheduler_mode,
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                threshold=self.scheduler_threshold,
                cooldown=self.scheduler_cooldown,
                min_lr=self.scheduler_min_lr,
                verbose=True  # 只在主进程显示信息
            )

    def _train_epoch(self, epoch: int):
        """训练单个epoch"""

        self.device_model.train()

        # 初始化进度条
        pbar = tqdm(
            total=len(self.train_loader),
            desc=f"Epoch {epoch}",
            leave=True,
            dynamic_ncols=True,
            postfix={
                "loss": "?",
                "bpp_0": "?",
                "bpp_1": "?",
                "bpp_2": "?",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                "GPU_mem": f"{torch.cuda.memory_allocated(self.device) / 1e9:.2f}GB"  # 显示GPU内存使用
            }
        )

        for batch_idx, data in enumerate(self.train_loader):
            refer_image, image = data["refer_image"].to(self.device), data["image"].to(self.device)

            self.optimizer.zero_grad()
            self.aux_optimizer.zero_grad()

            output = self.device_model(image, refer_image)
            loss, bpp_list = self.criterion(output, image)
            loss.backward()
            if self.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.device_model.parameters(), self.clip_max_norm)
            self.optimizer.step()

            aux_loss = self.device_model.aux_loss()
            aux_loss.backward()
            self.aux_optimizer.step()

            # 打印日志
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "bpp_0": f"{bpp_list[0].item():.4f}" if len(bpp_list) > 0 else "N/A",
                    "bpp_1": f"{bpp_list[1].item():.4f}" if len(bpp_list) > 1 else "N/A",
                    "bpp_2": f"{bpp_list[2].item():.4f}" if len(bpp_list) > 2 else "N/A",
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "GPU_mem": f"{torch.cuda.memory_allocated(self.device) / 1e9:.2f}GB"  # 显示GPU内存使用
                })
                pbar.update(self.log_interval)

        # 关闭主进程的进度条
        pbar.close()

    def _validate(self, epoch: int):
        """验证"""
        if self.val_loader is None:
            return None

        self.device_model.eval()
        bpp = 0.0

        with torch.no_grad():
            for data in self.val_loader:
                refer_image, image = data["refer_image"].to(self.device), data["image"].to(self.device)
                output = self.device_model(image, refer_image)
                batch_bpp, _ = self.criterion(output, image).item()
                bpp += batch_bpp

        avg_bpp = bpp / len(self.val_loader)

        # 获取当前学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        # 记录标量数据
        self.writer.add_scalar('LearningRate', current_lr, epoch)
        self.writer.add_scalar('BPP/val', avg_bpp, epoch)
        print(f"\nValidation set: Average bpp: {avg_bpp:.4f}")

        return avg_bpp

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""

        state = {
            'epoch': epoch,
            'state_dict': self.device_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler is not None else None,
            'optimizer': self.optimizer_conf["net"]["type"],
            'initial_lr': self.optimizer_conf["net"]["lr"]
        }

        filename = os.path.join(self.save_dir, f'checkpoint_epoch{epoch}.pth')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)

        if is_best:
            best_filename = os.path.join(self.save_dir, 'model_best.pth')
            torch.save(state, best_filename)

        # 清理过期的检查点
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """清理过期的检查点，保留最新的和最佳的"""
        # 获取所有检查点文件
        all_checkpoints = []
        for fname in os.listdir(self.save_dir):
            if fname.startswith('checkpoint_epoch') and fname.endswith('.pth'):
                epoch = int(fname.split('_')[1][5:].split('.')[0])
                all_checkpoints.append((epoch, fname))
        # 处理常规检查点 - 保留最新的max_checkpoints个
        if len(all_checkpoints) > self.max_checkpoints:
            # 按epoch排序，保留最新的
            all_checkpoints.sort()
            for i in range(len(all_checkpoints) - self.max_checkpoints):
                old_epoch, old_fname = all_checkpoints[i]
                old_path = os.path.join(self.save_dir, old_fname)
                if os.path.exists(old_path):
                    os.remove(old_path)

    def train(self, epochs: int, save_interval: int = 5):

        self.epochs = epochs
        self.save_interval = save_interval

        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            self._train_epoch(epoch)

            if self.val_loader is not None:
                val_loss = self._validate(epoch)

                # 更新学习率调度器
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)

            # 定期保存检查点
            if epoch % self.save_interval == 0:
                self._save_checkpoint(epoch)
