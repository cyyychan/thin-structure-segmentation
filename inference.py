from mmseg.apis import init_model, inference_model, show_result_pyplot
import cv2
import os
import glob
import numpy as np
import torch.nn.functional as F

config_path = 'configs/deeplabv3plus_r101-d8_4xb4-40k_crack500-512x512.py'
checkpoint_path = "seg_exp/exp_deeplabv3plus_crack500/work_dirs/deeplabv3plus_r101-d8_4xb4-40k_crack500-512x512/iter_40000.pth"

model = init_model(config_path, checkpoint_path, device='cuda:0')
print(model)

img_dir = '/dataset/siyuanchen/research/data/crack/test'
out_dir = 'output'
os.makedirs(out_dir, exist_ok=True)

# 支持的图片后缀
img_suffixes = ['*.jpg', '*.png', '*.jpeg', '*.bmp']

img_paths = []
for suf in img_suffixes:
    img_paths.extend(glob.glob(os.path.join(img_dir, suf)))

print(f'Found {len(img_paths)} images in {img_dir}')

for image_path in img_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f'Warning: failed to read {image_path}, skip.')
        continue

    base = os.path.basename(image_path)
    stem = os.path.splitext(base)[0]

    # 推理
    result = inference_model(model, image)

    # 1) 可视化叠加图（原图 + 预测）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # show_result_pyplot 需要 RGB
    overlay_file = os.path.join(out_dir, f'{stem}_overlay.png')
    _ = show_result_pyplot(
        model, image_rgb, result,
        opacity=0.5,
        show=False,
        out_file=overlay_file,
        draw_gt=False,
        draw_pred=True,
        with_labels=False,
    )

    # 2) 前景 softmax 概率图和按阈值二值结果
    logits = result.seg_logits.data  # [C, H, W] 或 [1, C, H, W]
    if logits.ndim == 4:
        logits = logits[0]  # -> [C, H, W]
    prob = F.softmax(logits, dim=0)  # 通道维 softmax
    fg_prob = prob[1].cpu().numpy()  # 取前景通道 [H, W]

    # 概率图（0-255 灰度）
    prob_vis = (fg_prob * 255).astype(np.uint8)
    prob_file = os.path.join(out_dir, f'{stem}_prob.png')
    cv2.imwrite(prob_file, prob_vis)

    print(f'Saved: overlay={overlay_file}, prob={prob_file}')