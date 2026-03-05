from typing import Tuple

import torch
import torch.nn as nn
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS

from models.backbones.scsegamba.gbc import GBC, BottConv


class DySample(nn.Module):
    """Dynamic upsampling module from SCSegamba (local copy)."""

    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        import torch.nn.functional as F  # local alias
        self.F = F

        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            nn.init.constant_(self.scope.weight, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange(
            (-self.scale + 1) / 2,
            (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(
            1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        F = self.F
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(
                                 x.dtype).to(x.device)
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype,
            device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(
            coords.view(B, -1, H, W),
            self.scale).view(
                B, 2, -1, self.scale * H,
                self.scale * W).permute(0, 2, 3, 4,
                                        1).contiguous().flatten(0, 1)
        return F.grid_sample(
            x.reshape(B * self.groups, -1, H, W),
            coords,
            mode='bilinear',
            align_corners=False,
            padding_mode="border").view(
                B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        F = self.F
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(
                self.offset(x_) * self.scope(x_).sigmoid(),
                self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(
                self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x)


class MFS(nn.Module):
    """Multi-scale feature selection module (local copy from SCSegamba)."""

    def __init__(self, embedding_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.linear_c4 = MLP(input_dim=128, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=64, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=32, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=16, embed_dim=embedding_dim)
        self.GBC_C = GBC(embedding_dim * 4)
        self.GBC_8 = GBC(8, norm_type='IN')
        self.GN_C = nn.GroupNorm(
            num_channels=embedding_dim * 4,
            num_groups=embedding_dim * 4 // 16)
        self.linear_fuse = BottConv(
            embedding_dim * 4,
            embedding_dim,
            embedding_dim // 8,
            kernel_size=1,
            padding=0,
            stride=1,
        )

        self.linear_pred = BottConv(embedding_dim, 1, 1, kernel_size=1)
        self.linear_pred_1 = nn.Conv2d(1, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)

        self.DySample_C_2 = DySample(embedding_dim, scale=2)
        self.DySample_C_4 = DySample(embedding_dim, scale=4)
        self.DySample_C_8 = DySample(embedding_dim, scale=8)

    def forward(self, inputs):
        c4, c3, c2, c1 = inputs

        b, c, h, w = c4.shape
        out_c4 = self.linear_c4(
            c4.reshape(b, c, h * w).permute(0, 2,
                                            1)).permute(0, 2, 1).reshape(
                                                b, self.embedding_dim, h, w)
        out_c4 = self.DySample_C_8(out_c4)

        b, c, h, w = c3.shape
        out_c3 = self.linear_c3(
            c3.reshape(b, c, h * w).permute(0, 2,
                                            1)).permute(0, 2, 1).reshape(
                                                b, self.embedding_dim, h, w)
        out_c3 = self.DySample_C_4(out_c3)

        b, c, h, w = c2.shape
        out_c2 = self.linear_c2(
            c2.reshape(b, c, h * w).permute(0, 2,
                                            1)).permute(0, 2, 1).reshape(
                                                b, self.embedding_dim, h, w)
        out_c2 = self.DySample_C_2(out_c2)

        b, c, h, w = c1.shape
        out_c1 = self.linear_c1(
            c1.reshape(b, c, h * w).permute(0, 2,
                                            1)).permute(0, 2, 1).reshape(
                                                b, self.embedding_dim, h, w)

        out_c = self.GBC_C(
            torch.cat([out_c4, out_c3, out_c2, out_c1], dim=1))
        out_c = self.linear_fuse(out_c)

        out_c = self.dropout(out_c)
        x = self.linear_pred_1(self.linear_pred(out_c))

        return x


@MODELS.register_module()
class MFSHead(BaseDecodeHead):
    """MFS decode head wrapping SCSegamba's MFS module.

    - 输入: 来自 SAVSSBackbone 的四个尺度特征 [c4, c3, c2, c1]。
    - 内部: 使用 MFS 将多尺度特征融合为单通道 logits（裂缝概率前的 logit）。
    - 输出: seg_logits (N, 1, H, W)，可配合 BCE+Dice loss 使用。
    """

    def __init__(
        self,
        embedding_dim: int = 8,
        in_channels: Tuple[int, int, int, int] = (128, 64, 32, 16),
        in_index: Tuple[int, int, int, int] = (0, 1, 2, 3),
        **kwargs,
    ) -> None:
        # 使用 multiple_select 直接拿到 4 个尺度特征
        super().__init__(
            in_channels=list(in_channels),
            in_index=list(in_index),
            input_transform='multiple_select',
            channels=embedding_dim,
            num_classes=1,  # 单通道二分类，配合 use_sigmoid=True
            **kwargs,
        )

        self.mfs = MFS(embedding_dim)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # inputs: backbone 输出的多尺度特征
        feats = self._transform_inputs(list(inputs))  # [c4, c3, c2, c1]
        assert len(feats) == 4, \
            f'MFSHead expects 4 feature maps, but got {len(feats)}'
        c4, c3, c2, c1 = feats
        logit = self.mfs([c4, c3, c2, c1])  # (N,1,H,W)
        return logit

