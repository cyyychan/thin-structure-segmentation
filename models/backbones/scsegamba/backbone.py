from typing import Tuple, List

import torch
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class SCSegambaBackbone(BaseModule):
    """SAVSS backbone wrapper for mmsegmentation.

    直接复用 SCSegamba 中的 SAVSS(arch='Crack')，并将其多尺度特征输出给
    mmseg 的 decode head 使用。

    参数基本与原实现保持一致，只暴露常用的部分。
    """

    def __init__(
        self,
        arch: str = 'Crack',
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        drop_path_rate: float = 0.2,
        final_norm: bool = True,
        convert_syncbn: bool = True,
        img_size: int = 512,
        in_channels: int = 3,
        init_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        from .savss import SAVSS as _SAVSS
        self.backbone = _SAVSS(
            arch=arch,
            out_indices=out_indices,
            drop_path_rate=drop_path_rate,
            final_norm=final_norm,
            convert_syncbn=convert_syncbn,
            img_size=img_size,
            in_channels=in_channels,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward features.

        返回一个 list，对应 [c4, c3, c2, c1] 四个尺度特征，与原 SCSegamba
        中 Decoder/MFS 的期望输入一致。
        """
        feats = self.backbone(x)
        return feats

