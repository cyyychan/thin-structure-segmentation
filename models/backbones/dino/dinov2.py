from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class DinoV2Backbone(BaseModule):
    """DINOv2 ViT backbone wrapper for mmsegmentation.

    使用 torch.hub 加载 facebookresearch/dinov2 提供的 ViT backbone，并将
    patch token 特征还原为 2D feature map，作为分割模型的 backbone 输出。

    当前实现：
    - 只提供一个 stage（out_indices 只能包含 0），输出 [B, C, H_p, W_p]，
      其中 H_p×W_p = patch token 个数。
    - C 为 DINOv2 的 embedding 维度（如 vits14 为 384）。

    Args:
        model_name: torch.hub 中的模型名，例如 'dinov2_vits14'、'dinov2_vitb14'。
        out_indices: 输出的 stage 索引，当前仅支持 (0,)。
        frozen: 是否冻结 DINOv2 参数（默认 True，只做特征提取）。
        init_cfg: mmengine init_cfg（一般不用，使用 hub 的预训练即可）。
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        out_indices: Tuple[int, ...] = (0,),
        frozen: bool = True,
        init_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        if not set(out_indices).issubset({0}):
            raise ValueError(
                f"DinoV2Backbone currently only supports out_indices within {{0}}, "
                f"but got {out_indices}"
            )
        self.out_indices = out_indices

        # 通过 torch.hub 加载预训练 DINOv2 backbone
        # 参考: https://github.com/facebookresearch/dinov2
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True
        )

        # 可选：冻结参数，仅做特征提取
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 记录 embedding 维度，供 decode head 配置参考
        self.embed_dim = getattr(self.backbone, "embed_dim", None)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward.

        输入:
            x: [B, 3, H, W]，H/W 需为 patch_size 的整数倍（如 14）。

        输出:
            list，仅包含一个 feature map:
                feats[0]: [B, C, H_p, W_p]，C 为 DINO embedding_dim，
                          H_p×W_p = patch token 数。
        """
        # DINOv2 的 forward_features 接受 BCHW
        feats = self.backbone.forward_features(x)
        patch_tokens = feats["x_norm_patchtokens"]  # [B, N, C]

        B, N, C = patch_tokens.shape
        H_p = W_p = int(N ** 0.5)
        assert H_p * W_p == N, (
            f"Patch tokens number {N} is not a perfect square; "
            "please ensure input size是 patch_size 的整数倍。"
        )

        feat_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H_p, W_p)  # [B,C,H_p,W_p]

        outs: List[torch.Tensor] = []
        if 0 in self.out_indices:
            outs.append(feat_map.contiguous())
        return outs

