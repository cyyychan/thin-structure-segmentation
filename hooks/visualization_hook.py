# Copyright (c) OpenMMLab. All rights reserved.
"""三图拼接可视化 Hook：输出 [原图 | GT | 预测] 水平拼接结果。"""
import os.path as osp
import warnings
from typing import List, Optional, Sequence

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.structures import PixelData
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer


def _draw_sem_seg_custom(image: np.ndarray,
                         sem_seg: PixelData,
                         classes: List,
                         palette: List,
                         alpha: float = 0.5,
                         draw_background: bool = False) -> np.ndarray:
    """绘制语义分割，支持自定义透明度和不绘制 background。"""
    num_classes = len(classes)
    sem_seg_data = sem_seg.cpu().data
    if isinstance(sem_seg_data, torch.Tensor):
        sem_seg_np = sem_seg_data.cpu().numpy()
    else:
        sem_seg_np = np.asarray(sem_seg_data)
    if sem_seg_np.ndim == 3:
        sem_seg_np = sem_seg_np[0]

    ids = np.unique(sem_seg_np)[::-1]
    ids = ids[(ids >= 0) & (ids < num_classes)]
    if not draw_background:
        ids = ids[ids != 0]
    labels = np.array(ids, dtype=np.int64)

    mask = np.zeros_like(image, dtype=np.uint8)
    fg_mask = np.zeros(image.shape[:2], dtype=bool)
    for label in labels:
        idx = sem_seg_np == label
        fg_mask |= idx
        mask[idx, :] = np.array(palette[label], dtype=np.uint8)

    # 仅在前景区域混合，背景保持原图不变暗
    color_seg = image.copy().astype(np.float32)
    color_seg[fg_mask] = (
        image[fg_mask].astype(np.float32) * (1 - alpha)
        + mask[fg_mask].astype(np.float32) * alpha)
    return color_seg.astype(np.uint8)


def _get_fg_prob(logit: torch.Tensor, fg_class: int = 1) -> np.ndarray:
    """从 seg_logits 提取前景概率，兼容单通道(sigmoid)和多通道(softmax)。"""
    if logit.ndim == 4:
        logit = logit[0]
    C = logit.shape[0]
    if C == 1:
        return torch.sigmoid(logit[0]).cpu().numpy()
    prob = F.softmax(logit, dim=0)
    if C <= fg_class:
        return np.zeros(logit.shape[1:], dtype=np.float32)
    return prob[fg_class].cpu().numpy()


def _draw_softmax_gray(image: np.ndarray,
                       data_sample: SegDataSample,
                       fg_class: int = 1,
                       alpha: float = 0.5,
                       thresh: Optional[float] = None,
                       palette: Optional[List] = None) -> np.ndarray:
    """使用 seg_logits 的 softmax 概率绘制热力图叠加.
    如果存在连续概率，根据预测概率的大小混合前景颜色。如果设置了阈值，则根据阈值二值化。
    """
    if not hasattr(data_sample, 'seg_logits'):
        return image

    logits = data_sample.seg_logits.data
    if isinstance(logits, torch.Tensor):
        logit = logits
    else:
        logit = torch.as_tensor(logits)

    fg_prob = _get_fg_prob(logit, fg_class)

    color = np.array(palette[fg_class], dtype=np.float32) if palette else np.array([255, 255, 255], dtype=np.float32)

    img = image.copy().astype(np.float32)
    
    if thresh is not None:
        # 二值化
        fg_mask = fg_prob > thresh
        if fg_mask.any():
            img[fg_mask] = img[fg_mask] * (1 - alpha) + color * alpha
    else:
        # 连续概率
        # 仅在概率 > 0 时进行混合
        fg_mask = fg_prob > 0
        if fg_mask.any():
            prob_mask = fg_prob[fg_mask, None] # [N, 1]
            # 用原图和用预测概率缩放过的颜色混合
            img[fg_mask] = img[fg_mask] * (1 - alpha) + (color * prob_mask) * alpha

    return img.astype(np.uint8)


@HOOKS.register_module()
class SegVisualizationHookConcat3(Hook):
    """三图拼接可视化 Hook：输出 [原图 | GT | 预测] 水平拼接。

    与 SegVisualizationHook 用法相同，但绘制结果为三列：原图、GT 叠加、预测叠加。
    支持 alpha 透明度和 draw_background 控制是否绘制背景。
    支持 draw_on_image 参数，当其为 False 时，GT和预测图将直接绘制在纯黑背景上，而不是叠加在原图上。

    Args:
        draw (bool): 是否绘制。Defaults to False.
        interval (int): 可视化间隔。Defaults to 50.
        show (bool): 是否显示窗口。Default to False.
        wait_time (float): 显示间隔(秒)。Defaults to 0.
        alpha (float): 叠加透明度，0~1，越大前景越不透明。Defaults to 0.5.
        draw_background (bool): 是否绘制背景类(class 0)。Defaults to False.
        draw_on_image (bool): 是否将结果叠加在原图上，如果为False，则在纯黑背景上绘制。Defaults to True.
        backend_args (dict, optional): 文件后端参数。
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 alpha: float = 0.5,
                 draw_background: bool = False,
                 draw_on_image: bool = True,
                 backend_args: Optional[dict] = None,
                 softmax_thresh: Optional[float] = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            self._visualizer._vis_backends = {}
            warnings.warn(
                'The show is True, vis_backends will be excluded.')
        self.wait_time = wait_time
        self.alpha = alpha
        self.draw_background = draw_background
        self.draw_on_image = draw_on_image
        self.backend_args = backend_args.copy() if backend_args else None
        # 若不为 None，则 softmax 可视化使用该阈值做二值化
        self.softmax_thresh = softmax_thresh
        self.draw = draw
        if not self.draw:
            warnings.warn(
                'The draw is False, the hook will not take effect.')
        self._test_index = 0

    def _draw_concat3(self,
                      image: np.ndarray,
                      data_sample: SegDataSample,
                      name: str,
                      step: int = 0) -> None:
        """绘制 [原图 | GT | 预测] 三图拼接。"""
        if not isinstance(self._visualizer, SegLocalVisualizer):
            # 回退到 add_datasample
            self._visualizer.add_datasample(
                name, image, data_sample=data_sample,
                show=self.show, wait_time=self.wait_time, step=step)
            return

        classes = self._visualizer.dataset_meta.get('classes', None)
        palette = self._visualizer.dataset_meta.get('palette', None)

        gt_img = None
        pred_img = None

        if 'gt_sem_seg' in data_sample and classes is not None:
            if self.draw_on_image:
                gt_img = _draw_sem_seg_custom(
                    image.copy(), data_sample.gt_sem_seg,
                    classes, palette,
                    alpha=self.alpha, draw_background=self.draw_background)
            else:
                sem_seg_np = data_sample.gt_sem_seg.data
                if isinstance(sem_seg_np, torch.Tensor):
                    sem_seg_np = sem_seg_np.cpu().numpy()
                if sem_seg_np.ndim == 3:
                    sem_seg_np = sem_seg_np[0]
                
                gt_img = np.zeros_like(image, dtype=np.uint8)
                # 直接将前景设为白色 (255, 255, 255)
                fg_mask = sem_seg_np == 1
                if fg_mask.any():
                    gt_img[fg_mask, :] = np.array([255, 255, 255], dtype=np.uint8)

        if 'seg_logits' in data_sample:
            if self.draw_on_image:
                # 使用 softmax 前景概率灰度/二值图叠加作为预测可视化
                pred_img = _draw_softmax_gray(
                    image.copy(), data_sample,
                    fg_class=1, alpha=self.alpha,
                    thresh=self.softmax_thresh,
                    palette=palette)
            else:
                # 直接绘制概率图
                logits = data_sample.seg_logits.data
                if isinstance(logits, torch.Tensor):
                    logit = logits
                else:
                    logit = torch.as_tensor(logits)
                if logit.ndim == 4:
                    logit = logit[0]
                C = logit.shape[0]
                if C == 1:
                    # 单通道：不用 sigmoid，直接 raw 归一化增强对比度（参考 SCSegamba）
                    raw = logit[0].cpu().numpy()
                    m = np.max(raw)
                    pred_gray = (255 * (raw / (m + 1e-8))).astype(np.uint8)
                else:
                    fg_prob = _get_fg_prob(logit, fg_class=1)
                    pred_gray = (fg_prob * 255).astype(np.uint8)
                pred_img = np.stack([pred_gray] * 3, axis=-1)

        # 拼接 [原图 | GT | 预测]
        imgs = [image]
        if gt_img is not None:
            imgs.append(gt_img)
        if pred_img is not None:
            imgs.append(pred_img)
        drawn_img = np.concatenate(imgs, axis=1)

        if self.show:
            self._visualizer.show(
                drawn_img, win_name=name, wait_time=self.wait_time)
        else:
            self._visualizer.add_image(name, drawn_img, step)

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[SegDataSample]) -> None:
        if not self.draw:
            return
        total_curr_iter = runner.iter + batch_idx
        img_path = outputs[0].img_path
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        window_name = f'val_{osp.basename(img_path)}'
        if total_curr_iter % self.interval == 0:
            self._draw_concat3(
                img, outputs[0], window_name, step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[SegDataSample]) -> None:
        if not self.draw:
            return
        for data_sample in outputs:
            self._test_index += 1
            img_path = data_sample.img_path
            window_name = f'test_{osp.basename(img_path)}'
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            self._draw_concat3(
                img, data_sample, window_name, step=self._test_index)
