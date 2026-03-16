from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS

PATCH_SIZE = 14

@MODELS.register_module()
class DinoV2Backbone(BaseModule):
    """DINOv2 ViT backbone wrapper for mmsegmentation.

    使用 torch.hub 加载 facebookresearch/dinov2 提供的 ViT backbone，并将
    patch token 特征还原为 2D feature map，作为分割模型的 backbone 输出。

    当前实现：
    - 支持多 stage：out_indices 为 backbone 的 block 层索引（如 2,5,8,11），
      每级输出 [B, C, H_p, W_p]，其中 H_p×W_p = patch token 个数。
    - C 为 DINOv2 的 embedding 维度（如 vits14 为 384）。

    Args:
        model_name: torch.hub 中的模型名，例如 'dinov2_vits14'、'dinov2_vitb14'。
        out_indices: backbone block 的层索引，例如 (2,5,8,11) 输出 4 级特征。
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
        self.out_indices = tuple(sorted(out_indices))

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

    def init_weights(self) -> None:
        """Override to skip re-initialization of backbone.

        DINOv2 已通过 torch.hub.load(..., pretrained=True) 加载预训练权重，
        若在此处调用 backbone.init_weights() 会覆盖预训练权重（cls_token、pos_embed、
        attn、mlp 等），故此处不做任何操作。
        """
        pass

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward.

        输入:
            x: [B, 3, H, W]，H/W 需为 patch_size 的整数倍（如 14）。

        输出:
            list，按 out_indices 顺序的 feature map:
                feats[i]: [B, C, H_p, W_p]，C 为 DINO embedding_dim，
                          H_p×W_p = patch token 数。
        """
        B, _, H, W = x.shape
        pad_h = (PATCH_SIZE - H % PATCH_SIZE) % PATCH_SIZE
        pad_w = (PATCH_SIZE - W % PATCH_SIZE) % PATCH_SIZE
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)

        # 从 out_indices 指定的层提取特征
        features = self.backbone.get_intermediate_layers(x, n=list(self.out_indices))

        outs: List[torch.Tensor] = []
        for patch_tokens in features:
            N = patch_tokens.shape[1]
            C = patch_tokens.shape[2]
            H_p = W_p = int(N ** 0.5)
            assert H_p * W_p == N, (
                f"Patch tokens number {N} is not a perfect square; "
                "please ensure input size 是 patch_size 的整数倍。"
            )
            feat_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H_p, W_p)
            outs.append(feat_map.contiguous())
        return outs
