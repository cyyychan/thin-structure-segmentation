# Copyright (c) OpenMMLab. All rights reserved.
# VMamba backbone for mmsegmentation: re-implementation of VSSM/Backbone_VSSM.
#
# 结构说明:
#   - patch_embed: 将图像切块并投影到 embed_dim，支持 v1(单层卷积)/v2(重叠卷积)
#   - layers: 4 个 stage，每 stage 含若干 VSSBlock(SS2D+MLP) + 下采样
#   - 输出: 多尺度特征 [C1,C2,C3,C4]，供 UPerNet 等解码头使用

from collections import OrderedDict

import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_
from mmseg.registry import MODELS

from .block import VSSBlock
from .layers import LayerNorm2d, PatchMerging2D, Permute
from .ss2d import SS2D


@MODELS.register_module()
class VMamba(BaseModule):
    """VMamba backbone for mmsegmentation.

    Re-implementation of VMamba (VSSM) from
    https://github.com/MzeroMiko/VMamba.
    Outputs multi-scale features for decode heads (e.g. UPerNet).

    备注:
        - channel_first: norm_layer 为 'ln2d' 或 'bn' 时为 True，特征为 (B,C,H,W)
        - 预训练权重需与 depths/dims/ssm_ratio 等一致，否则用 strict=False 加载
        - forward_type 常用 v05_noz（与官方 vssm1 tiny 等配置一致）

    Args:
        out_indices: Indices of stages to output (0,1,2,3).
        patch_size: Patch size for patch_embed.
        in_chans: Input channels (3 for RGB).
        depths: Number of blocks per stage, e.g. (2, 2, 8, 2).
        dims: Base channel dim (int) or list per stage. If int, dims*2^i.
        ssm_d_state: SSM state dimension.
        ssm_ratio: SSM inner dimension ratio (d_inner = dim * ssm_ratio).
        ssm_dt_rank: SSM dt rank ('auto' or int).
        ssm_act_layer: SSM activation ('silu', 'gelu', ...).
        ssm_conv: Depthwise conv kernel size in SS2D.
        ssm_conv_bias: Whether to use bias in SS2D conv.
        ssm_drop_rate: Dropout in SS2D.
        ssm_init: SS2D init ('v0', 'v1', 'v2').
        forward_type: SS2D forward type ('v0', 'v2', 'v05_noz', ...).
        mlp_ratio: MLP hidden dim ratio (0 to disable MLP).
        mlp_act_layer: MLP activation.
        mlp_drop_rate: MLP dropout.
        gmlp: Use gMLP instead of MLP.
        drop_path_rate: Stochastic depth rate.
        patch_norm: Whether to use norm after patch embed.
        norm_layer: 'LN', 'ln2d', 'bn'.
        downsample_version: 'v1', 'v2', 'v3'.
        patchembed_version: 'v1', 'v2'.
        use_checkpoint: Use gradient checkpointing.
        posembed: Use learnable position embedding.
        init_cfg: Mmseg init_cfg (e.g. Pretrained checkpoint).
    """

    def __init__(
        self,
        out_indices=(0, 1, 2, 3),
        patch_size=4,
        in_chans=3,
        depths=(2, 2, 8, 2),
        dims=96,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank='auto',
        ssm_act_layer='silu',
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init='v0',
        forward_type='v2',
        mlp_ratio=4.0,
        mlp_act_layer='gelu',
        mlp_drop_rate=0.0,
        gmlp=False,
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer='ln2d',
        downsample_version='v2',
        patchembed_version='v1',
        use_checkpoint=False,
        posembed=False,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg=init_cfg)
        # 通道优先时使用 LayerNorm2d/BatchNorm2d，特征格式 (B,C,H,W)
        self.channel_first = (norm_layer.lower() in ['bn', 'ln2d'])
        self.out_indices = out_indices
        self.num_layers = len(depths)
        # dims 为 int 时各 stage 通道数: dims*2^0, dims*2^1, ...
        if isinstance(dims, int):
            dims = [int(dims * 2**i) for i in range(self.num_layers)]
        self.dims = dims
        self.num_features = dims[-1]

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )
        norm_layer_cls = _NORMLAYERS.get(norm_layer.lower(), LayerNorm2d)
        ssm_act = _ACTLAYERS.get(ssm_act_layer.lower(), nn.SiLU)
        mlp_act = _ACTLAYERS.get(mlp_act_layer.lower(), nn.GELU)

        # 随机深度：从 0 线性增加到 drop_path_rate，按 block 分配
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        # 可选位置编码（分类常用，分割通常不用）
        self.pos_embed = None
        if posembed:
            ph = pw = 224 // patch_size
            self.pos_embed = nn.Parameter(
                torch.zeros(1, dims[0] if isinstance(dims, list) else dims, ph,
                            pw))
            trunc_normal_(self.pos_embed, std=0.02)

        # Patch 嵌入：v1 单层 4x4 卷积，v2 重叠卷积（类似 Swin）
        _make_pe = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, self._make_patch_embed)
        self.patch_embed = _make_pe(
            in_chans,
            dims[0],
            patch_size,
            patch_norm,
            norm_layer_cls,
            channel_first=self.channel_first,
        )

        # Stage 间下采样：v1=PatchMerging2D, v2=Conv2d 2x2, v3=Conv2d 3x3 stride2
        _make_ds = dict(
            v1=self._make_downsample_v1,
            v2=self._make_downsample,
            v3=self._make_downsample_v3,
            none=(lambda *_, **__: nn.Identity()),
        ).get(downsample_version, self._make_downsample)

        self.layers = ModuleList()
        for i in range(self.num_layers):
            # 最后一 stage 不下采样
            downsample = (
                _make_ds(
                    dims[i],
                    dims[i + 1],
                    norm_layer=norm_layer_cls,
                    channel_first=self.channel_first,
                ) if (i < self.num_layers - 1) else nn.Identity())

            self.layers.append(
                self._make_layer(
                    dim=dims[i],
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    use_checkpoint=use_checkpoint,
                    norm_layer=norm_layer_cls,
                    downsample=downsample,
                    channel_first=self.channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                    _SS2D=SS2D,
                ))

        # 每个输出 stage 单独一层 norm，与 mmseg 多尺度输出约定一致
        for i in out_indices:
            layer = norm_layer_cls(dims[i])
            self.add_module(f'outnorm{i}', layer)

        self.apply(self._init_weights)

    @staticmethod
    def _make_patch_embed(in_chans, embed_dim, patch_size, patch_norm,
                          norm_layer, channel_first=False):
        """v1: 单层 Conv patch_size x patch_size，无重叠。"""
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                     stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans, embed_dim, patch_size, patch_norm,
                             norm_layer, channel_first=False):
        """v2: 两段重叠卷积（类似 Swin），stride=patch_size//2，重叠切块。"""
        stride = patch_size // 2
        k = stride + 1
        pad = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=k, stride=stride,
                     padding=pad),
            (nn.Identity() if (channel_first or not patch_norm) else Permute(
                0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or not patch_norm) else Permute(
                0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=k, stride=stride,
                      padding=pad),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample_v1(dim, out_dim, norm_layer=nn.LayerNorm,
                            channel_first=False):
        """v1: 2x2 窗口合并，通道 4->1 再线性映射到 out_dim。"""
        return PatchMerging2D(dim, out_dim, norm_layer, channel_first)

    @staticmethod
    def _make_downsample(dim, out_dim, norm_layer=nn.LayerNorm,
                         channel_first=False):
        """v2: 2x2 卷积 stride=2，空间减半。"""
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim, out_dim, norm_layer=nn.LayerNorm,
                            channel_first=False):
        """v3: 3x3 卷积 stride=2 padding=1，空间减半。"""
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim,
        drop_path,
        use_checkpoint=False,
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank='auto',
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init='v0',
        forward_type='v2',
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        _SS2D=SS2D,
        **kwargs,
    ):
        """构建一个 stage：若干 VSSBlock(SS2D+MLP) + 末尾 downsample。"""
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(
                VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[d],
                    norm_layer=norm_layer,
                    channel_first=channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                    use_checkpoint=use_checkpoint,
                    _SS2D=_SS2D,
                ))
        return nn.Sequential(
            OrderedDict(blocks=nn.Sequential(*blocks), downsample=downsample))

    def _init_weights(self, m):
        """Linear/LayerNorm 的默认初始化，Conv 等保持 PyTorch 默认。"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        """若 init_cfg 为 Pretrained，则加载 checkpoint（strict=False 以兼容 head 等）。
        官方 VMamba 分类 checkpoint 的权重在顶层 key 'model' 下，此处自动解包。
        """
        if self.init_cfg is None or self.init_cfg.get('type') != 'Pretrained':
            return
        ckpt = self.init_cfg.get('checkpoint')
        if not ckpt:
            return
        # 加载原始 checkpoint（支持 URL / 本地路径）
        if isinstance(ckpt, str):
            if ckpt.startswith(('http://', 'https://')):
                raw = torch.hub.load_state_dict_from_url(
                    ckpt, map_location='cpu')
            else:
                raw = torch.load(ckpt, map_location='cpu', weights_only=False)
        else:
            raw = ckpt
        # 官方 release 为 {'model': state_dict}，mmseg 常用 {'state_dict': state_dict}
        state_dict = raw.get('model', raw.get('state_dict', raw))
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        """输入 (B,C,H,W)，输出 list of tensors，对应 out_indices 的多尺度特征。"""
        def layer_forward(layer, x):
            x = layer.blocks(x)
            y = layer.downsample(x)  # 当前 stage 输出 o，下采样后作为下一 stage 输入
            return x, y

        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pe = self.pos_embed if self.channel_first else self.pos_embed.permute(
                0, 2, 3, 1)
            x = x + pe

        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                # 统一转为 (B,C,H,W) 供 decode head 使用
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x
        return outs
