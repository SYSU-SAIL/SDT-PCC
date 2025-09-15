import math
import torch
import torch.nn as nn


class RateLoss(nn.Module):

    def forward(self, output, target):
        num_points = torch.count_nonzero(target)

        bpp_list = [
            torch.log(likelihoods).sum() / (-math.log(2) * num_points)
            for likelihoods in output["likelihoods"].values()
        ]
        bpp = sum(bpp_list)

        return bpp, bpp_list


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, sigma=0.1, reduction='mean'):
        """
        Soft Cross Entropy Loss using Gaussian kernel estimation.

        :param sigma: float, standard deviation for the Gaussian kernel
        :param reduction: str, 'mean' (default) or 'sum', determines how to reduce loss
        """
        super(SoftCrossEntropyLoss, self).__init__()
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, reconstructed, original):
        """
        Compute the soft cross entropy loss.

        :param reconstructed: (B, 1, H, W) predicted tensor
        :param original: (B, 1, H, W) ground-truth tensor
        :return: Scalar loss value
        """
        B, C, H, W = reconstructed.shape
        total_elements = B * C * H * W  # 总元素数量

        # 展平张量以便进行计算
        reconstructed = reconstructed.view(-1, 1)  # (N, 1)
        original = original.view(-1, 1)  # (N, 1)

        # 计算 reconstructed 的概率密度 q(x) (高斯核)
        diff = reconstructed - original  # (N, 1)
        gaussian_kernel = torch.exp(-0.5 * (diff / self.sigma) ** 2) / (
                self.sigma * torch.sqrt(torch.tensor(2 * torch.pi)))

        # 计算 soft 归一化概率
        # q_x = gaussian_kernel.mean(dim=1)  # (N,) 每个 reconstructed 点的概率
        q_x = gaussian_kernel

        # 计算交叉熵 H(p, q) = -sum(p(x) log q(x))
        loss = -torch.log(q_x + 1e-9)  # 避免 log(0)

        # 根据 reduction 方式处理 loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError("Invalid reduction mode. Choose from ['mean', 'sum'].")


if __name__ == '__main__':
    reconstructed = torch.randint(0, 10, (2, 1, 4, 4)).float()  # 预测值
    reconstructed.requires_grad_()
    original = torch.randint(0, 10, (2, 1, 4, 4)).float()  # 真实值
    loss_fn = SoftCrossEntropyLoss(sigma=0.5, reduction='mean')
    loss = loss_fn(reconstructed, original)
    loss.backward()  # 计算梯度
    pass
