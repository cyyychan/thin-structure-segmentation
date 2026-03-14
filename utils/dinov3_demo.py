import os
import sys

# 项目根目录加入 path，支持直接运行 python utils/dinov3.py
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

from models.backbones.dino.dinov3 import DinoV3Backbone


# 1. 加载封装好的 DINOv3 backbone
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = DinoV3Backbone(
    model_name='dinov3_vits16',
    out_indices=(0,),
    frozen=True,
    pretrained=True,
    weights='checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
).to(device)
backbone.eval()

# 2. 图像预处理（标准 DINO 流程）
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

# 3. 推理 & 提取特征
img = Image.open("crack_datasets/Crack500/img_dir/test/20160222_081111_1_361.jpg").convert("RGB")
x = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

with torch.no_grad():
    # DinoV3Backbone.forward 返回 list[feat]，这里只取第一个
    feat_map = backbone(x)[0]  # [B, C, H_p, W_p]
    print('feat_map from backbone:', feat_map.shape)

    # 为了和原先逻辑一致，转成 [C,H_p,W_p]
    feat_map = feat_map[0]     # [C,H_p,W_p]
    C, H_p, W_p = feat_map.shape
    print('single-sample feat_map:', feat_map.shape)

    # 4.1 分别保存前 20 个通道的 feature map（三联图：原图 / 通道热力图 / 叠加）
    os.makedirs("dinov3_vis", exist_ok=True)
    img_np = np.array(img).astype(float) / 255.0  # [H_img, W_img, 3]

    num_show = min(20, C)
    for i in range(num_show):
        ch = feat_map[i]  # [H, W]
        # 单通道归一化到 [0,1]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-5)

        # 插值到原图大小
        ch_up = F.interpolate(
            ch.unsqueeze(0).unsqueeze(0),              # [1,1,H,W]
            size=img.size[::-1],                       # (H,W)
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()

        # 伪彩色热力图
        heat = plt.cm.viridis(ch_up)[..., :3]
        alpha = 0.5
        overlay = (1 - alpha) * img_np + alpha * heat

        plt.figure(figsize=(9, 3))

        plt.subplot(1, 3, 1)
        plt.title("Input")
        plt.imshow(img_np)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title(f"Feat ch {i}")
        plt.imshow(ch_up, cmap="viridis")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Overlay")
        plt.imshow(overlay)
        plt.axis("off")

        plt.tight_layout()
        ch_path = f"dinov3_vis/feat_ch_{i:02d}.png"
        plt.savefig(ch_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved channel {i} triplet to: {ch_path}")

