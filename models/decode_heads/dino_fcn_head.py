import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from .siren import SirenNet, build_siren_input


@MODELS.register_module()
class DinoV2FCNHeadSiren(BaseDecodeHead):
    """DINOv2 多尺度特征 concat + 多次反卷积上采样 -> 分割预测.

    - 输入：多尺度 DINOv2 特征 [f0, f1, f2, f3]，每级 [B, C, H_p, W_p]
    - resize_concat 后 [B, 4*C, H_p, W_p] -> Conv -> 多次 Deconv(2x)+Conv -> seg_logits
    """

    def __init__(
        self,
        in_channels,  # list/tuple，每级通道数，如 (384, 384, 384, 384)
        num_classes: int = 1,
        decoder_channels: tuple = (256, 128, 64),
        mlp_channels: tuple = None,  # 兼容旧配置，等同于 decoder_channels
        num_deconv_layers: int = 4,
        with_edge_attn: bool = False,
        with_siren: bool = False,
        siren_coef: float = 0.5,
        **kwargs,
    ) -> None:
        if mlp_channels is not None:
            decoder_channels = mlp_channels
        in_ch_list = list(in_channels)
        concat_channels = sum(in_ch_list)
        super().__init__(
            in_channels=in_ch_list,
            channels=decoder_channels[-1],
            num_classes=num_classes,
            input_transform="resize_concat",
            **kwargs,
        )
        self.with_edge_attn = with_edge_attn
        if with_edge_attn:
            # 使用最浅层特征进行边缘注意力
            self.edge_head = nn.Sequential(
                nn.Conv2d(in_channels[0], 64, 3, padding=1),
                nn.SyncBatchNorm(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1)
            )
        
        # 初始卷积：concat -> decoder_channels[0]
        self.conv_in = nn.Sequential(
            nn.Conv2d(concat_channels, decoder_channels[0], 3, padding=1),
            nn.SyncBatchNorm(decoder_channels[0]),
            nn.ReLU(inplace=True),
        )
        # 多次反卷积：每次 2x 上采样 + Conv
        deconv_blocks = []
        ch_list = list(decoder_channels)
        for i in range(num_deconv_layers):
            c_in = ch_list[min(i, len(ch_list) - 1)]
            c_out = ch_list[min(i + 1, len(ch_list) - 1)]
            deconv_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2),
                    nn.SyncBatchNorm(c_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c_out, c_out, 3, padding=1),
                    nn.SyncBatchNorm(c_out),
                    nn.ReLU(inplace=True),
                )
            )
        self.deconv_blocks = nn.Sequential(*deconv_blocks)
        
        self.with_siren = with_siren
        if with_siren:
            self.siren_coef = siren_coef
            self.siren = SirenNet(in_dim=2 + num_classes)
            

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # [B, sum(in_channels), H, W]
        if self.with_edge_attn:
            # 原始浅层特征
            low_level_feat = inputs[0]
            edge = self.edge_head(low_level_feat)
            x = x * torch.sigmoid(edge)
        h, w = x.shape[2:]
        x = self.conv_in(x)
        x = self.deconv_blocks(x)
        x = self.conv_seg(x)  # [B, num_classes, H, W]
        x = F.interpolate(x, size=(w * 14, h * 14), mode="bilinear", align_corners=False)
        
        if self.with_siren:
            B, C, H, W = x.shape
            siren_in = build_siren_input(x)
            refined = self.siren(siren_in)
            refined = refined.view(B, H, W, C).permute(0,3,1,2)
            x = x + self.siren_coef * refined
        return x[:, :, :512, :512]
    
    
@MODELS.register_module()
class DinoV2FCNHead(BaseDecodeHead):
    """DINOv2 多尺度特征 concat + 多次反卷积上采样 -> 分割预测.

    - 输入：多尺度 DINOv2 特征 [f0, f1, f2, f3]，每级 [B, C, H_p, W_p]
    - resize_concat 后 [B, 4*C, H_p, W_p] -> Conv -> 多次 Deconv(2x)+Conv -> seg_logits
    """

    def __init__(
        self,
        in_channels,  # list/tuple，每级通道数，如 (384, 384, 384, 384)
        num_classes: int = 2,
        decoder_channels: tuple = (256, 128, 64),
        mlp_channels: tuple = None,  # 兼容旧配置，等同于 decoder_channels
        num_deconv_layers: int = 4,
        with_edge_attn: bool = False,
        dropout_ratio: float = 0.1,
        **kwargs,
    ) -> None:
        if mlp_channels is not None:
            decoder_channels = mlp_channels
        in_ch_list = list(in_channels)
        concat_channels = sum(in_ch_list)
        super().__init__(
            in_channels=in_ch_list,
            channels=decoder_channels[-1],
            num_classes=num_classes,
            input_transform="resize_concat",
            **kwargs,
        )
        self.with_edge_attn = with_edge_attn
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        if with_edge_attn:
            # 使用最浅层特征进行边缘注意力
            self.edge_head = nn.Sequential(
                nn.Conv2d(in_channels[0], 64, 3, padding=1),
                nn.SyncBatchNorm(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1)
            )
        
        # 初始卷积：concat -> decoder_channels[0]
        self.conv_in = nn.Sequential(
            nn.Conv2d(concat_channels, decoder_channels[0], 3, padding=1),
            nn.SyncBatchNorm(decoder_channels[0]),
            nn.ReLU(inplace=True),
        )
        # 多次反卷积：每次 2x 上采样 + Conv
        deconv_blocks = []
        ch_list = list(decoder_channels)
        for i in range(num_deconv_layers):
            c_in = ch_list[min(i, len(ch_list) - 1)]
            c_out = ch_list[min(i + 1, len(ch_list) - 1)]
            deconv_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2),
                    nn.SyncBatchNorm(c_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(c_out, c_out, 3, padding=1),
                    nn.SyncBatchNorm(c_out),
                    nn.ReLU(inplace=True),
                )
            )
        self.deconv_blocks = nn.Sequential(*deconv_blocks)
            

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # [B, sum(in_channels), H, W]
        if self.with_edge_attn:
            # 原始浅层特征
            low_level_feat = inputs[0]
            edge = self.edge_head(low_level_feat)
            x = x + 0.5 * x * torch.sigmoid(edge)
        h, w = x.shape[2:]
        x = self.conv_in(x)
        x = self.dropout(x)
        x = self.deconv_blocks(x)
        x = self.dropout(x)
        x = self.conv_seg(x)  # [B, num_classes, H, W]
        x = F.interpolate(x, size=(w * 14, h * 14), mode="bilinear", align_corners=False)
        return x[:, :, :512, :512]
    
    
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
                nn.SyncBatchNorm(c_out),
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
            nn.SyncBatchNorm(c1),
            nn.ReLU(),

            nn.Conv2d(c1, c2, 3, padding=1),
            nn.SyncBatchNorm(c2),
            nn.ReLU(),
            
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.SyncBatchNorm(c3),
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
            nn.SyncBatchNorm(c1),
            nn.ReLU(),

            nn.Conv2d(c1, c2, 3, padding=1),
            nn.SyncBatchNorm(c2),
            nn.ReLU(),
            
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.SyncBatchNorm(c3),
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