from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmseg.registry import HOOKS

try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
    _HAS_SKIMAGE = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_SKIMAGE = False


def _get_statistics(pred: np.ndarray, gt: np.ndarray):
    """计算二值图的 TP/FP/FN"""
    assert pred.shape == gt.shape
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return tp, fp, fn


def _cal_OIS_metrics(pred_list, gt_list, thresh_step=0.01):
    """每张图各自扫阈值取最优 F1，再对图像平均（OIS）。"""
    final_F1_list = []
    for pred, gt in zip(pred_list, gt_list):
        F1_list = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = _get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            if tp + fn == 0:
                r_acc = 0
            else:
                r_acc = tp / (tp + fn)
            if p_acc + r_acc == 0:
                F1 = 0
            else:
                F1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            F1_list.append(F1)
        max_F1 = np.max(np.array(F1_list))
        final_F1_list.append(max_F1)
    final_F1 = float(np.sum(np.array(final_F1_list)) / len(final_F1_list))
    return final_F1


def _cal_ODS_metrics(pred_list, gt_list, thresh_step=0.01):
    """全数据共用一个阈值，扫阈值取平均 F1 最优（ODS）。"""
    final_ODS = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        ODS_list = []
        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = _get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            if tp + fn == 0:
                r_acc = 0
            else:
                r_acc = tp / (tp + fn)
            if p_acc + r_acc == 0:
                F1 = 0
            else:
                F1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            ODS_list.append(F1)
        ave_F1 = float(np.mean(np.array(ODS_list)))
        final_ODS.append(ave_F1)
    ODS = float(np.max(np.array(final_ODS)))
    return ODS


def _skeletonize_binary(mask: np.ndarray) -> np.ndarray:
    """对 0/1 二值图做细化骨架，返回 0/1 图."""
    if _HAS_SKIMAGE:
        skel = _skimage_skeletonize(mask.astype(bool))
        return skel.astype(np.uint8)

    # fallback: 简单细化近似（两次腐蚀-膨胀差分），效果不如真正 skeleton，但避免依赖
    from scipy.ndimage import binary_erosion  # type: ignore

    m = mask.astype(bool)
    eroded = binary_erosion(m)
    skel = m & ~eroded
    return skel.astype(np.uint8)


def _cal_skeleton_metrics(pred_list, gt_list, thresh: float):
    """Skeleton F1 和 Connectivity Score（按全数据像素统计）。

    简化定义：
        - 先把 pred/gt 二值化为 0/1，再做 skeletonize。
        - Skeleton F1: 以 skeleton 像素为正类，统计 TP/FP/FN 算 P/R/F1。
        - Connectivity Score: 对 GT skeleton 上的每个像素，若其 8 邻域中
          在 pred skeleton 中至少有一个正像素，则认为连通；否则视为断裂。
          得分 = 连通像素数 / GT skeleton 像素总数。
    """
    tp_sum = fp_sum = fn_sum = 0.0
    conn_hit = conn_total = 0.0

    for pred, gt in zip(pred_list, gt_list):
        # 先按给定阈值把 pred 二值化为 0/1，GT 仍按 127 阈值
        pred_bin = ((pred / 255.0) > thresh).astype(np.uint8)
        gt_bin = (gt > 127).astype(np.uint8)

        # 骨架提取
        skel_pred = _skeletonize_binary(pred_bin)
        skel_gt = _skeletonize_binary(gt_bin)

        # Skeleton F1 统计
        tp = np.sum((skel_pred == 1) & (skel_gt == 1))
        fp = np.sum((skel_pred == 1) & (skel_gt == 0))
        fn = np.sum((skel_pred == 0) & (skel_gt == 1))
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

        # Connectivity：GT skeleton 上每个像素，看 pred skeleton 是否在其 8 邻域连接上
        ys, xs = np.nonzero(skel_gt)
        conn_total += len(xs)
        if len(xs) == 0:
            continue
        h, w = skel_gt.shape
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 1), min(h, y + 2)
            x0, x1 = max(0, x - 1), min(w, x + 2)
            patch = skel_pred[y0:y1, x0:x1]
            if patch.any():
                conn_hit += 1

    # Skeleton F1
    if tp_sum == 0 and fp_sum == 0:
        P = 1.0
    else:
        P = float(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
    R = float(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
    if P + R == 0:
        F1 = 0.0
    else:
        F1 = float(2 * P * R / (P + R))

    # Connectivity Score
    if conn_total == 0:
        conn_score = 0.0
    else:
        conn_score = float(conn_hit / conn_total)

    return F1, conn_score


def _cal_IoU_metrics(pred_list, gt_list, thresh_step=0.01):
    """前景 IoU 的阈值扫描最优（按全像素统计）。

    同时返回该最佳 IoU 阈值及其对应的全局 Precision / Recall / F1。

    Returns:
        best_iou (float): 所有阈值中最大的前景 IoU。
        best_P (float): 在 best_iou 对应阈值下的全局 Precision。
        best_R (float): 在 best_iou 对应阈值下的全局 Recall。
        best_F1 (float): 在 best_iou 对应阈值下的全局 F1。
        best_thresh (float): 取得 best_iou 时所用的阈值。
    """
    best_iou = -1.0
    best_stats = (0.0, 0.0, 0.0)  # tp, fp, fn
    best_thresh = 0.0

    for thresh in np.arange(0.0, 1.0, thresh_step):
        tp_sum = fp_sum = fn_sum = 0.0

        for pred, gt in zip(pred_list, gt_list):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')

            TP = np.sum((pred_img == 1) & (gt_img == 1))
            FP = np.sum((pred_img == 1) & (gt_img == 0))
            FN = np.sum((pred_img == 0) & (gt_img == 1))

            tp_sum += TP
            fp_sum += FP
            fn_sum += FN

        denom = tp_sum + fp_sum + fn_sum
        if denom <= 0:
            iou_thresh = 0.0
        else:
            iou_thresh = float(tp_sum / denom)

        if iou_thresh > best_iou:
            best_iou = iou_thresh
            best_stats = (tp_sum, fp_sum, fn_sum)
            best_thresh = float(thresh)

    if best_iou < 0:
        # 没有有效样本
        return 0.0, 0.0, 0.0, 0.0, 0.0

    tp, fp, fn = best_stats
    # 全局 Precision / Recall / F1
    if tp == 0 and fp == 0:
        P = 1.0
    else:
        P = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    R = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    if P + R == 0:
        F1 = 0.0
    else:
        F1 = float(2 * P * R / (P + R))

    return float(best_iou), P, R, F1, best_thresh


def _cal_IoU_at_thresh(pred_list, gt_list, thresh: float):
    """在给定阈值下，用全像素统计前景 IoU / P / R / F1。"""
    tp_sum = fp_sum = fn_sum = 0.0

    for pred, gt in zip(pred_list, gt_list):
        gt_img = (gt / 255).astype('uint8')
        pred_img = (pred / 255 > thresh).astype('uint8')

        TP = np.sum((pred_img == 1) & (gt_img == 1))
        FP = np.sum((pred_img == 1) & (gt_img == 0))
        FN = np.sum((pred_img == 0) & (gt_img == 1))

        tp_sum += TP
        fp_sum += FP
        fn_sum += FN

    denom = tp_sum + fp_sum + fn_sum
    if denom <= 0:
        iou = 0.0
    else:
        iou = float(tp_sum / denom)

    if tp_sum == 0 and fp_sum == 0:
        P = 1.0
    else:
        P = float(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0

    R = float(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
    if P + R == 0:
        F1 = 0.0
    else:
        F1 = float(2 * P * R / (P + R))

    return iou, P, R, F1


@HOOKS.register_module()
class MetricsHook(Hook):
    """在验证阶段计算 OIS / ODS / IoU。

    Args:
        thresh_step (float): 阈值步长。
        fg_class (int): 作为前景的语义类别 ID（Crack 场景通常为 1）。
    """

    def __init__(self,
                 thresh_step: float = 0.01,
                 fg_class: int = 1,
                 fixed_thresh: float | None = None) -> None:
        self.thresh_step = thresh_step
        self.fg_class = fg_class
        # 若不为 None，则额外在该固定阈值下计算 IoU/P/R/F1
        self.fixed_thresh = fixed_thresh

        # 在一个验证 / 测试阶段中累积所有样本的预测与 GT
        self._pred_list: List[np.ndarray] = []
        self._gt_list: List[np.ndarray] = []

    def after_val_iter(self, runner: Runner, batch_idx: int,
                       data_batch: dict,
                       outputs: List) -> None:  # type: ignore[override]
        """每个验证 iter 后收集前景概率图和 GT 掩码."""
        for out, ds in zip(outputs, data_batch['data_samples']):
            # seg_logits: [C, H, W] 或 [1, C, H, W]
            logits = out.seg_logits.data
            if logits.ndim == 4:
                logits = logits[0]
            # 在通道维做 softmax
            prob = F.softmax(logits, dim=0)
            if prob.shape[0] <= self.fg_class:
                continue
            fg_prob = prob[self.fg_class].cpu().numpy()  # [H, W]
            pred_vis = (fg_prob * 255).astype(np.uint8)
            self._pred_list.append(pred_vis)

            # GT 掩码：[1, H, W] 或 [H, W]，取前景类 == fg_class
            gt = ds.gt_sem_seg.data
            if gt.ndim == 3:
                gt = gt[0]
            gt_np = gt.cpu().numpy()
            gt_bin = (gt_np == self.fg_class).astype(np.uint8) * 255
            self._gt_list.append(gt_bin)

    def after_val_epoch(self, runner: Runner,
                        metrics: dict = None) -> None:  # type: ignore[override]
        """验证结束后，用本次验证累计的结果计算 OIS / ODS / IoU。"""
        if not self._pred_list:
            runner.logger.warning(
                'MetricsHook: no valid predictions collected in this '
                'validation epoch.')
            return

        OIS = _cal_OIS_metrics(
            self._pred_list, self._gt_list, thresh_step=self.thresh_step)
        ODS = _cal_ODS_metrics(
            self._pred_list, self._gt_list, thresh_step=self.thresh_step)

        # 1) 计算 IoU/P/R/F1（扫描或固定阈值）
        if self.fixed_thresh is None:
            IoU, P, R, F1, best_t = _cal_IoU_metrics(
                self._pred_list, self._gt_list, thresh_step=self.thresh_step)
            skel_thresh = best_t
            prefix = (f'MetricsHook metrics (fg_class={self.fg_class}, '
                      f'mode=best, t={best_t:.2f})')
        else:
            IoU, P, R, F1 = _cal_IoU_at_thresh(
                self._pred_list, self._gt_list, self.fixed_thresh)
            skel_thresh = self.fixed_thresh
            prefix = (f'MetricsHook metrics (fg_class={self.fg_class}, '
                      f'mode=fixed, t={self.fixed_thresh:.2f})')

        # 2) Skeleton 与 Connectivity：使用同一阈值
        skel_F1, conn_score = _cal_skeleton_metrics(
            self._pred_list, self._gt_list, skel_thresh)

        runner.logger.info(
            f'{prefix}: '
            f'OIS={OIS:.4f}, ODS={ODS:.4f}, IoU={IoU:.4f}, '
            f'P={P:.4f}, R={R:.4f}, F1={F1:.4f}, '
            f'skel_F1={skel_F1:.4f}, Conn={conn_score:.4f}')

        if metrics is not None:
            metrics['metrics_OIS'] = float(OIS)
            metrics['metrics_ODS'] = float(ODS)
            metrics['metrics_IoU'] = float(IoU)
            metrics['metrics_P'] = float(P)
            metrics['metrics_R'] = float(R)
            metrics['metrics_F1'] = float(F1)
            metrics['metrics_skel_F1'] = float(skel_F1)
            metrics['metrics_connectivity'] = float(conn_score)

        # 本轮结束，清空缓存
        self._pred_list.clear()
        self._gt_list.clear()

    def after_test_iter(self, runner: Runner, batch_idx: int,
                        data_batch: dict,
                        outputs: List) -> None:  # type: ignore[override]
        """测试阶段每个 iter 后同样累积预测与 GT."""
        # 复用 after_val_iter 的逻辑
        self.after_val_iter(runner, batch_idx, data_batch, outputs)

    def after_test_epoch(self, runner: Runner,
                         metrics: dict = None) -> None:  # type: ignore[override]
        """测试结束后计算一次 OIS / ODS / IoU。"""
        if not self._pred_list:
            runner.logger.warning(
                'MetricsHook: no valid predictions collected in this '
                'test epoch.')
            return

        OIS = _cal_OIS_metrics(
            self._pred_list, self._gt_list, thresh_step=self.thresh_step)
        ODS = _cal_ODS_metrics(
            self._pred_list, self._gt_list, thresh_step=self.thresh_step)

        if self.fixed_thresh is None:
            IoU, P, R, F1, best_t = _cal_IoU_metrics(
                self._pred_list, self._gt_list, thresh_step=self.thresh_step)
            skel_thresh = best_t
            prefix = (f'[Test] MetricsHook metrics (fg_class={self.fg_class}, '
                      f'mode=best, t={best_t:.2f})')
        else:
            IoU, P, R, F1 = _cal_IoU_at_thresh(
                self._pred_list, self._gt_list, self.fixed_thresh)
            skel_thresh = self.fixed_thresh
            prefix = (f'[Test] MetricsHook metrics (fg_class={self.fg_class}, '
                      f'mode=fixed, t={self.fixed_thresh:.2f})')

        skel_F1, conn_score = _cal_skeleton_metrics(
            self._pred_list, self._gt_list, skel_thresh)
        runner.logger.info(
            f'{prefix}: '
            f'OIS={OIS:.4f}, ODS={ODS:.4f}, IoU={IoU:.4f}, '
            f'P={P:.4f}, R={R:.4f}, F1={F1:.4f}, '
            f'skel_F1={skel_F1:.4f}, Conn={conn_score:.4f}')

        if metrics is not None:
            metrics['metrics_OIS'] = float(OIS)
            metrics['metrics_ODS'] = float(ODS)
            metrics['metrics_IoU'] = float(IoU)
            metrics['metrics_P'] = float(P)
            metrics['metrics_R'] = float(R)
            metrics['metrics_F1'] = float(F1)
            metrics['metrics_skel_F1'] = float(skel_F1)
            metrics['metrics_connectivity'] = float(conn_score)

        self._pred_list.clear()
        self._gt_list.clear()

