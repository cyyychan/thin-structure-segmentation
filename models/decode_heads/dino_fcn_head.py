import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class DinoV2SegHead(BaseDecodeHead):
    """DINOv2 feature map -> segmentation mask (简单双线性上采样 + 卷积).

    设计目标：
    - 输入：单尺度 DINOv2 特征 [B, C, H_p, W_p]（patch grid）
    - 内部：逐步上采样 + 3×3 卷积，最后 1×1 分类
    - 输出：seg_logits [B, num_classes, H_dec, W_dec]
      最终由 EncoderDecoder 统一 resize 到原图大小。
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        decoder_channels=(256, 128, 64),
        **kwargs,
    ) -> None:
        # 单尺度输入，in_channels 对应 DINOv2 embedding 维度
        super().__init__(
            in_channels=in_channels,
            channels=decoder_channels[-1],
            num_classes=num_classes,
            input_transform=None,
            **kwargs,
        )

        c1, c2, c3 = decoder_channels
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),

            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),

            nn.Conv2d(c3, num_classes, 1)
        )

    def forward(self, inputs):
        """BaseDecodeHead 接口：inputs 可能是 tensor 或 list[tensor]."""
        x = self._transform_inputs(inputs)  # [B, C, H, W]
        x = self.head(x)
        
        # 将特征图还原到原图大小
        x = F.interpolate(
            x,
            scale_factor=14,
            mode='bilinear',
            align_corners=False
        )[:, :, :512, :512]
        return x

@MODELS.register_module()
class DinoV3SegHead(BaseDecodeHead):
    """DINOv3 feature map -> segmentation mask (简单双线性上采样 + 卷积).

    设计目标：
    - 输入：单尺度 DINOv3 特征 [B, C, H_p, W_p]（patch grid）
    - 内部：逐步上采样 + 3×3 卷积，最后 1×1 分类
    - 输出：seg_logits [B, num_classes, H_dec, W_dec]
      最终由 EncoderDecoder 统一 resize 到原图大小。
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        decoder_channels=(256, 128, 64),
        **kwargs,
    ) -> None:
        # 单尺度输入，in_channels 对应 DINOv3 embedding 维度
        super().__init__(
            in_channels=in_channels,
            channels=decoder_channels[-1],
            num_classes=num_classes,
            input_transform=None,
            **kwargs,
        )

        c1, c2, c3 = decoder_channels
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),

            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),

            nn.Conv2d(c3, num_classes, 1)
        )

    def forward(self, inputs):
        """BaseDecodeHead 接口：inputs 可能是 tensor 或 list[tensor]."""
        x = self._transform_inputs(inputs)  # [B, C, H, W]
        x = self.head(x)
        
        # 将特征图还原到原图大小
        x = F.interpolate(
            x,
            scale_factor=16,
            mode='bilinear',
            align_corners=False
        )
        return x