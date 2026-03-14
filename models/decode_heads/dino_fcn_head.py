import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class DinoV2ConcatHead(BaseDecodeHead):
    """DINOv2 多尺度特征 concat + MLP 融合 -> 分割预测.

    - 输入：多尺度 DINOv2 特征 [f0, f1, f2, f3]，每级 [B, C, H_p, W_p]
    - resize_concat 后 [B, 4*C, H_p, W_p] -> MLP -> seg_logits
    """

    def __init__(
        self,
        in_channels,  # list/tuple，每级通道数，如 (384, 384, 384, 384)
        num_classes: int = 2,
        mlp_channels: tuple = (512, 256),
        **kwargs,
    ) -> None:
        in_ch_list = list(in_channels)
        concat_channels = sum(in_ch_list)
        super().__init__(
            in_channels=in_ch_list,
            channels=mlp_channels[-1],
            num_classes=num_classes,
            input_transform="resize_concat",
            **kwargs,
        )
        # MLP: concat -> hidden -> output
        layers = []
        c_in = concat_channels
        for c_out in mlp_channels:
            layers += [
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # [B, sum(in_channels), H, W]
        x = self.mlp(x)
        x = self.conv_seg(x)  # [B, num_classes, H, W]
        x = F.interpolate(x, scale_factor=14, mode="bilinear", align_corners=False)
        return x[:, :, :512, :512]


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