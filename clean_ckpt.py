import torch

# 加载 checkpoint
checkpoint_path = 'checkpoints/20250620_2128/model_best.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# 保留 state_dict，删除其他内容
state_dict = checkpoint.get('state_dict', {})
checkpoint = {'state_dict': state_dict}

# 删除 state_dict 中包含 'entropy_bottleneck' 的键
keys_to_delete = [k for k in state_dict.keys() if 'entropy_bottleneck' in k]
for k in keys_to_delete:
    del state_dict[k]

# 保存修改后的 checkpoint
torch.save(checkpoint, 'checkpoints/20250620_2128/model_best.pth')
