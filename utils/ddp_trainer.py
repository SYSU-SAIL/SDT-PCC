import os
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from compressai.optimizers import net_aux_optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Callable, Dict, Any, Literal

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.criterion import RateLoss


class DDPTrainer:
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
            gpu_ids: Optional[list[int]] = None,
            log_interval: int = 10,
            save_dir: str = "./checkpoints",
            log_dir: str = "./logs",
            use_lr_scheduler: bool = True,
            scheduler_mode: Literal["min", "max"] = "min",  # "min"或"max"
            scheduler_factor: float = 0.1,
            scheduler_patience: int = 5,
            scheduler_threshold: float = 1e-4,
            scheduler_cooldown: int = 3,
            scheduler_min_lr: float = 1e-6,
            max_checkpoints: int = 3,  # 最大保留的检查点数量
            resume_path: Optional[str] = None
    ):
        """
        DDP训练器初始化

        参数:
            model: 要训练的PyTorch模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集(可选)
            batch_size: 每个GPU的batch size
            num_workers: 每个GPU的数据加载worker数量
            optimizer_class: 优化器类
            optimizer_params: 优化器参数
            criterion: 损失函数
            gpu_ids: 使用的GPU ID列表(如[0,1,2,3]), None表示使用所有可用GPU
            use_amp: 是否使用自动混合精度
            log_interval: 日志打印间隔(批次数)
            save_dir: 模型保存目录
            log_dir: 记录文件位置
            use_lr_scheduler: 是否使用学习率调度器
            scheduler_mode: "min"(监控指标下降)或"max"(监控指标上升)
            scheduler_factor: 学习率衰减因子
            scheduler_patience: 等待多少个epoch没有改善
            scheduler_threshold: 改善的最小阈值
            scheduler_cooldown: 调整学习率后的冷却epoch数
            scheduler_min_lr: 学习率下限
            max_checkpoints: 最大保留的检查点数量(不包括最佳模型)
        """
        self.save_interval = None
        self.epochs = None
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
        self.gpu_ids = gpu_ids
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.resume_path = resume_path
        # 只在主进程初始化SummaryWriter
        self.writer = None
        # 学习率调度器相关参数
        self.use_lr_scheduler = use_lr_scheduler
        self.scheduler_mode = scheduler_mode
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_threshold = scheduler_threshold
        self.scheduler_cooldown = scheduler_cooldown
        self.scheduler_min_lr = scheduler_min_lr
        self.max_checkpoints = max_checkpoints

        # 自动检测可用GPU数量
        if self.gpu_ids is None:
            self.world_size = torch.cuda.device_count()
        else:
            self.world_size = len(self.gpu_ids)

    def _setup(self, rank: int):
        """设置单个进程的环境"""
        # 初始化进程组
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=self.world_size)

        # 设置设备
        torch.cuda.set_device(rank)

        # 准备模型
        model = self.model.to(rank)
        ddp_model = DDP(model, device_ids=[rank])

        # 准备优化器
        optimizers = net_aux_optimizer(model, self.optimizer_conf)
        optimizer, aux_optimizer = optimizers["net"], optimizers["aux"]

        # 准备学习率调度器
        scheduler = None
        if self.use_lr_scheduler and self.val_dataset is not None:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=self.scheduler_mode,
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                threshold=self.scheduler_threshold,
                cooldown=self.scheduler_cooldown,
                min_lr=self.scheduler_min_lr
            )

        # 准备数据加载器
        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=rank,
            shuffle=True
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = None
        if self.val_dataset is not None:
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=rank,
                shuffle=False
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                sampler=val_sampler,
                num_workers=self.num_workers,
                pin_memory=True
            )

        # 只在主进程初始化SummaryWriter
        if rank == 0:
            # 创建保存目录
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)

        if self.resume_path is not None and os.path.exists(self.resume_path):
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(self.resume_path, map_location=map_location)

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            if scheduler is not None and checkpoint.get('scheduler_state'):
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            if rank == 0:
                print(f"=> Resumed from checkpoint: {self.resume_path}")

        return {
            'rank': rank,
            'model': ddp_model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'aux_optimizer': aux_optimizer,
            'train_loader': train_loader,
            'train_sampler': train_sampler,
            'val_loader': val_loader,
        }

    def _train_epoch(self, epoch: int, setup_dict: dict):
        """训练单个epoch"""
        rank = setup_dict['rank']
        model = setup_dict['model']
        optimizer = setup_dict['optimizer']
        aux_optimizer = setup_dict['aux_optimizer']
        train_loader = setup_dict['train_loader']
        train_sampler = setup_dict['train_sampler']

        model.train()
        train_sampler.set_epoch(epoch)

        # 只在主进程初始化进度条
        if rank == 0:
            pbar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch}",
                leave=True,
                dynamic_ncols=True,
                postfix={
                    "loss": "?",
                    "bpp_0":"?",
                    "bpp_1":"?",
                    "bpp_2":"?",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "GPU_mem": f"{torch.cuda.memory_allocated(rank) / 1e9:.2f}GB"  # 显示GPU内存使用
                }
            )

        for batch_idx, data in enumerate(train_loader):
            refer_image, image = data["refer_image"].to(rank), data["image"].to(rank)

            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            output = model(image, refer_image)
            loss,bpp_list = self.criterion(output, image)
            loss.backward()
            if self.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_max_norm)
            optimizer.step()

            aux_loss = model.module.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            # 主进程打印日志
            if batch_idx % self.log_interval == 0 and rank == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "bpp_0": f"{bpp_list[0].item():.4f}" if len(bpp_list) > 0 else "N/A",
                    "bpp_1": f"{bpp_list[1].item():.4f}" if len(bpp_list) > 1 else "N/A",
                    "bpp_2": f"{bpp_list[2].item():.4f}" if len(bpp_list) > 2 else "N/A",
                    "lr": optimizer.param_groups[0]['lr'],
                    "GPU_mem": f"{torch.cuda.memory_allocated(rank) / 1e9:.2f}GB"  # 显示GPU内存使用
                })
                pbar.update(self.log_interval)

        # model.module.update(force=True)
        # 关闭主进程的进度条
        if rank == 0:
            pbar.close()

    def _validate(self, epoch: int, setup_dict: dict):
        """验证"""
        if setup_dict['val_loader'] is None:
            return None

        rank = setup_dict['rank']
        model = setup_dict['model']
        val_loader = setup_dict['val_loader']
        optimizer = setup_dict['optimizer']

        model.eval()
        total_bpp = 0.0
        valid_batches = 0
        invalid_batches = 0

        with torch.no_grad():
            for data in val_loader:
                refer_image, image = data["refer_image"].to(rank), data["image"].to(rank)
                output = model(image, refer_image)
                batch_bpp,_ = self.criterion(output, image)
                bpp_val = batch_bpp.item()
                if not torch.isnan(batch_bpp) and bpp_val < 32:
                    total_bpp += bpp_val
                    valid_batches += 1
                else:
                    invalid_batches += 1

        avg_bpp = total_bpp / max(1, valid_batches)

        if rank == 0:
            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            # 记录标量数据
            self.writer.add_scalar('LearningRate', current_lr, epoch)
            self.writer.add_scalar('BPP/val', avg_bpp, epoch)
            self.writer.add_scalar('BPP/invalid_batches', invalid_batches, epoch)
            print(f"\nValidation set: Average bpp: {avg_bpp:.4f}, Skipped {invalid_batches} invalid batches")

        return avg_bpp

    def _save_checkpoint(self, epoch: int, model: DDP, optimizer, scheduler, is_best: bool = False):
        """保存检查点"""
        if dist.get_rank() != 0:
            return
        # model.module.update() #保存CDF
        state = {
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
            'optimizer': self.optimizer_conf["net"]["type"],
            'initial_lr': self.optimizer_conf["net"]["lr"]
        }

        filename = os.path.join(self.save_dir, f'checkpoint_epoch{epoch}.pth')
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

    def _cleanup(self):
        """清理进程组"""
        dist.destroy_process_group()

    def _worker(self, rank: int):
        """单个工作进程的执行函数"""
        setup_dict = self._setup(rank)

        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            self._train_epoch(epoch, setup_dict)

            if setup_dict['val_loader'] is not None:
                val_loss = self._validate(epoch, setup_dict)

                # 更新学习率调度器
                if setup_dict['scheduler'] is not None:
                    setup_dict['scheduler'].step(val_loss)

                # 保存最佳模型
                if val_loss < best_val_loss and rank == 0:
                    best_val_loss = val_loss
                    self._save_checkpoint(
                        epoch,
                        setup_dict['model'],
                        setup_dict['optimizer'],
                        setup_dict['scheduler'],
                        is_best=True
                    )

            # 定期保存检查点
            if rank == 0 and epoch % self.save_interval == 0:
                self._save_checkpoint(
                    epoch,
                    setup_dict['model'],
                    setup_dict['optimizer'],
                    setup_dict['scheduler'],
                )

            # 确保所有进程等待检查点操作完成
            dist.barrier()
        self._cleanup()

    def train(self, epochs: int, save_interval: int = 5):
        """启动DDP训练

        参数:
            epochs: 训练的总epoch数
            save_interval: 保存检查点的间隔(epoch数)
        """
        self.epochs = epochs
        self.save_interval = save_interval

        # 设置GPU IDs
        if self.gpu_ids is None:
            self.gpu_ids = list(range(self.world_size))

        # 启动多进程训练
        mp.spawn(
            self._worker,
            args=(),
            nprocs=self.world_size,
            join=True
        )
