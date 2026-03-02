import os

import torch
import mmcv
from mmengine.config import Config
from mmengine.registry import DefaultScope
from mmseg.registry import DATASETS

# 注册自定义数据集（crack500Dataset）
import datasets  # noqa: F401


def main():
    # 加载 crack500 的数据配置（与训练完全一致）
    cfg = Config.fromfile('configs/datasets/crack500.py')
    train_dataset_cfg = cfg.train_dataloader['dataset']
    
    # 用和训练一致的 data_root
    train_dataset_cfg['data_root'] = '/dataset/siyuanchen/research/data/crack/Crack500'

    # 设置默认作用域为 mmseg，确保使用 mmseg 的 TRANSFORMS/模块
    DefaultScope.get_instance('mmseg', scope_name='mmseg')

    dataset = DATASETS.build(train_dataset_cfg)
    print(f'Dataset size: {len(dataset)} samples')

    save_dir = 'debug_crack500_train'
    os.makedirs(save_dir, exist_ok=True)

    # 保存前 N 张处理后的图像和标签
    num_to_save = min(50, len(dataset))
    for idx in range(num_to_save):
        data = dataset[idx]

        # 图像：C x H x W tensor -> H x W x C numpy (BGR)
        img = data['inputs']
        if isinstance(img, torch.Tensor):
            img_np = img.detach().cpu().numpy().transpose(1, 2, 0)
        else:
            img_np = img

        data_sample = data['data_samples']
        # 单样本模式：data_samples 是 SegDataSample
        gt = data_sample.gt_sem_seg.data.squeeze().cpu().numpy().astype('uint8')

        img_path = os.path.join(save_dir, f'{idx:04d}_img.png')
        gt_path = os.path.join(save_dir, f'{idx:04d}_gt.png')

        mmcv.imwrite(img_np, img_path)
        mmcv.imwrite(gt*255, gt_path)

    print(f'Saved {num_to_save} processed samples to {save_dir}')


if __name__ == '__main__':
    main()