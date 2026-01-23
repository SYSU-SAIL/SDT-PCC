import torch
import torch.nn.functional as F

def vectorized_smart_downsample_fill(x, stride_h=2, stride_w=4, diff_thresh=1.0):
    B, H, W = x.shape
    device = x.device

    # Step 1: 采样掩码（True = 被采样保留）
    sampling_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    sampling_mask[::stride_h, ::stride_w] = True
    sampling_mask = sampling_mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    # Step 2: 初始输出，先复制被采样点
    out = torch.zeros_like(x)
    out = out.masked_scatter(sampling_mask.expand(B, 1, H, W).squeeze(1),
                             x.masked_select(sampling_mask.expand(B, 1, H, W).squeeze(1)))

    # Step 3: 获取 unfold 3x3 patch，形状 [B, 9, H, W]
    x_unsqueezed = x.unsqueeze(1)  # [B,1,H,W]
    patches = F.unfold(x_unsqueezed, kernel_size=3, padding=1)  # [B, 9, H*W]
    patches = patches.view(B, 9, H, W)

    # Step 4: 中心值
    center_val = x.unsqueeze(1)  # [B,1,H,W]

    # Step 5: 对邻域为 0 的点屏蔽掉
    valid_mask = patches != 0
    diff = (patches - center_val).abs()
    diff_masked = diff.masked_fill(~valid_mask, float('nan'))  # 仅考虑有效点
    avg_diff = diff_masked.nanmean(dim=1)  # [B,H,W]

    # Step 6: 找出需补偿的位置：未采样 & 有效值 & 差异大
    x_valid_mask = x != 0
    sampling_mask_bool = sampling_mask.squeeze(0).squeeze(0)
    cond_mask = (~sampling_mask_bool) & x_valid_mask & (avg_diff > diff_thresh)  # [B,H,W]

    if cond_mask.sum() == 0:
        return out  # 无需补偿

    # Step 7: 屏蔽采样点 + 屏蔽值为0的点 -> 可用邻域候选点
    sampling_mask_exp = sampling_mask.expand(B, 1, H, W)
    sampling_mask_unfold = F.unfold(sampling_mask_exp.float(), kernel_size=3, padding=1).bool().view(B, 9, H, W)
    valid_patch_mask = (patches != 0) & (~sampling_mask_unfold)

    patch_vals = patches.masked_fill(~valid_patch_mask, float('nan'))  # 只保留未采样、非零点

    # 去掉中心（位置 4）用于均值计算
    patch_vals_excl_center = patch_vals.clone()
    patch_vals_excl_center[:, 4, :, :] = float('nan')
    neighbor_mean = patch_vals_excl_center.nanmean(dim=1)  # [B,H,W]

    # Step 8: 从邻域候选中选择最接近均值的点
    diff = (patch_vals - neighbor_mean.unsqueeze(1)).abs()
    diff = diff.masked_fill(~valid_patch_mask, float('inf'))
    best_idx = torch.argmin(diff, dim=1)  # [B,H,W]

    one_hot = F.one_hot(best_idx, num_classes=9).permute(0, 3, 1, 2).float()  # [B,9,H,W]
    selected_vals = (patches * one_hot).sum(dim=1)  # [B,H,W]

    # Step 9: 把补偿值 scatter 回输出图
    out = torch.where(cond_mask, selected_vals, out)

    return out