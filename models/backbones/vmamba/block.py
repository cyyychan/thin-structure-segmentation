# Copyright (c) OpenMMLab. All rights reserved.
# VSSBlock: SS2D + norm + MLP + drop_path, ported from VMamba.

"""
VSSBlock: VMamba 的基础构建块
结构: SS2D(2D Selective Scan) 分支 + MLP 分支，均带残差连接和 drop_path
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath

from .layers import Mlp, gMlp
from .ss2d import SS2D


class VSSBlock(nn.Module):
    """VMamba block: optional SS2D branch + optional MLP branch + drop_path."""

    def __init__(
        self,
        hidden_dim: int = 0,           # 隐藏层维度（通道数）
        drop_path: float = 0,          # stochastic depth 概率，用于正则化
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,           # True: [B,C,H,W], False: [B,H,W,C]
        ssm_d_state: int = 16,         # SSM 状态维度
        ssm_ratio=2.0,                 # SSM 扩展比例，>0 则启用 SSM 分支
        ssm_dt_rank='auto',            # dt 投影秩
        ssm_act_layer=nn.SiLU,         # SSM 激活函数
        ssm_conv: int = 3,             # 局部卷积核大小
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init='v0',                 # SSM 初始化方式
        forward_type='v2',             # SS2D 前向实现版本
        mlp_ratio=4.0,                 # MLP 隐藏层扩展比例，>0 则启用 MLP 分支
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,                    # 是否使用 gated MLP (gMLP)
        use_checkpoint: bool = False,  # 是否用 gradient checkpoint 省显存
        post_norm: bool = False,       # True: PostNorm, False: PreNorm
        _SS2D: type = SS2D,            # 可替换的 SS2D 实现类
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        # SSM 分支: LayerNorm + 2D Selective Scan (十字扫描)
        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = _SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)  # 随机丢弃整个残差分支

        # MLP 分支: LayerNorm + FFN (可选 Mlp 或 gMlp)
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channels_first=channel_first,
            )

    def _forward(self, input: torch.Tensor):
        """前向传播: SSM 分支 -> MLP 分支，均为 x + DropPath(sub_block(x))"""
        x = input
        # SSM 分支: PreNorm 为 norm -> op -> drop_path; PostNorm 为 op -> norm -> drop_path
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        # MLP 分支: 同样支持 PreNorm / PostNorm
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        """入口: 启用 checkpoint 时用重计算换显存"""
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        return self._forward(input)
