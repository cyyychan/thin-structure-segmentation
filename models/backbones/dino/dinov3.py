from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.registry import MODELS

# DINOv3 ViT 使用 patch_size=16（与 DINOv2 的 vits14 不同）
PATCH_SIZE = 16


@MODELS.register_module()
class DinoV3Backbone(BaseModule):
    """DINOv3 ViT backbone wrapper for mmsegmentation.

    使用 torch.hub 加载 facebookresearch/dinov3 提供的 ViT backbone，并将
    patch token 特征还原为 2D feature map，作为分割模型的 backbone 输出。

    当前实现：
    - 只提供一个 stage（out_indices 只能包含 0），输出 [B, C, H_p, W_p]，
      其中 H_p×W_p = patch token 个数。
    - C 为 DINOv3 的 embedding 维度（如 vits16 为 384）。
    - patch_size 固定为 16。

    Args:
        model_name: torch.hub 中的模型名，例如 'dinov3_vits16'、'dinov3_vitb16'、
            'dinov3_vits16plus'、'dinov3_vitl16'、'dinov3_vith16plus' 等。
        out_indices: 输出的 stage 索引，当前仅支持 (0,)。
        frozen: 是否冻结 DINOv3 参数（默认 True，只做特征提取）。
        pretrained: 是否加载预训练权重。
        weights: 预训练权重的本地路径（如 /path/to/dinov3_vits16_pretrain_lvd1689m.pth）。
            pretrained=True 时必须提供。
        init_cfg: mmengine init_cfg（一般不用，使用 hub 的预训练即可）。
    """

    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        out_indices: Tuple[int, ...] = (0,),
        frozen: bool = True,
        pretrained: bool = True,
        weights: Optional[str] = None,
        init_cfg=None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        if not set(out_indices).issubset({0}):
            raise ValueError(
                f"DinoV3Backbone currently only supports out_indices within {{0}}, "
                f"but got {out_indices}"
            )
        self.out_indices = out_indices

        if pretrained and not weights:
            raise ValueError("weights 必须提供本地路径，当 pretrained=True 时")

        # 延迟导入，避免 models/__init__ 链导致循环导入
        from dinov3.hub.backbones import dinov3_vits16
        # 通过 torch.hub 加载 DINOv3 backbone
        self.backbone = dinov3_vits16(weights=weights)

        # 可选：冻结参数，仅做特征提取
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 记录 embedding 维度，供 decode head 配置参考
        self.embed_dim = getattr(self.backbone, "embed_dim", None)

    def init_weights(self) -> None:
        """Override to skip re-initialization of backbone.

        DINOv3 已通过 torch.hub.load 加载预训练权重，
        若在此处调用 backbone.init_weights() 会覆盖预训练权重，故此处不做任何操作。
        """
        pass

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward.

        输入:
            x: [B, 3, H, W]，H/W 需为 patch_size 的整数倍（16）。

        输出:
            list，仅包含一个 feature map:
                feats[0]: [B, C, H_p, W_p]，C 为 DINO embedding_dim，
                          H_p×W_p = patch token 数。
        """
        # 确保输入是 patch_size 的整数倍
        B, _, H, W = x.shape
        pad_h = (PATCH_SIZE - H % PATCH_SIZE) % PATCH_SIZE
        pad_w = (PATCH_SIZE - W % PATCH_SIZE) % PATCH_SIZE
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)

        # 提取 patch token（DINOv3 与 DINOv2 相同，返回 x_norm_patchtokens）
        feats = self.backbone.forward_features(x)
        # 单张输入时 forward_features 返回 dict，多张时返回 list[dict]
        if isinstance(feats, list):
            feats = feats[0]
        patch_tokens = feats["x_norm_patchtokens"]  # [B, N, C]

        B, N, C = patch_tokens.shape
        H_p = W_p = int(N ** 0.5)
        assert H_p * W_p == N, (
            f"Patch tokens number {N} is not a perfect square; "
            "please ensure input size 是 patch_size 的整数倍。"
        )

        feat_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H_p, W_p)

        outs: List[torch.Tensor] = []
        if 0 in self.out_indices:
            outs.append(feat_map.contiguous())
        return outs
