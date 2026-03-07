import copy
from typing import Sequence, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from mmcv.cnn import build_norm_layer
from mmengine.model import ModuleList

from .patch_embed import ConvPatchEmbed
from .savss_layer import SAVSS_Layer
from .gbc import BottConv


def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    dst_weight = F.interpolate(
        src_weight, size=dst_shape, align_corners=False, mode=mode)
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)

    return torch.cat((extra_tokens, dst_weight), dim=1)


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class SAVSS(nn.Module):
    """Backbone from SCSegamba for Crack dataset.

    这是对 SCSegamba/MMCLS 中 SAVSS(arch='Crack') 的本地复现，只保留
    arch_zoo['Crack'] 分支，供分割任务使用。
    """

    arch_zoo = {
        'Crack': {
            'patch_size': 8,
            'embed_dims': 256,
            'num_layers': 4,
            'num_convs_patch_embed': 2,
            'layers_with_dwconv': [],
            'layer_cfgs': {
                'use_rms_norm': False,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2,
                    'conv_size': 7,
                    'dt_init': "random",
                    'conv_bias': True,
                    'bias': True,
                    'default_hw_shape': (512 // 8, 512 // 8),
                }
            }
        }
    }

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 arch='Crack',
                 patch_size=16,
                 embed_dims=192,
                 num_layers=20,
                 num_convs_patch_embed=1,
                 with_pos_embed=True,
                 out_indices=(-1,),
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 interpolate_mode='bicubic',
                 layer_cfgs=dict(),
                 layers_with_dwconv=[],
                 test_cfg=dict(),
                 convert_syncbn=False,
                 freeze_patch_embed=False,
                 **kwargs):
        super().__init__()

        self.test_cfg = test_cfg
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.convert_syncbn = convert_syncbn
        self.arch = arch

        assert self.arch in self.arch_zoo.keys()
        arch_cfg = self.arch_zoo[self.arch]
        self.embed_dims = arch_cfg['embed_dims']
        self.num_layers = arch_cfg['num_layers']
        self.patch_size = arch_cfg['patch_size']
        self.num_convs_patch_embed = arch_cfg['num_convs_patch_embed']
        self.layers_with_dwconv = arch_cfg['layers_with_dwconv']
        _layer_cfgs = arch_cfg['layer_cfgs']

        self.with_pos_embed = with_pos_embed
        self.interpolate_mode = interpolate_mode
        self.freeze_patch_embed = freeze_patch_embed

        self.patch_embed = ConvPatchEmbed(
            in_channels=in_channels,
            embed_dims=self.embed_dims,
            num_convs=self.num_convs_patch_embed,
            patch_size=self.patch_size,
            stride=self.patch_size,
            input_size=self.img_size,
        )
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        if with_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.embed_dims))
            trunc_normal_(self.pos_embed, std=0.02)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence)
        out_indices = list(out_indices)
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers
        self.out_indices = out_indices

        dpr = np.linspace(0, drop_path_rate, self.num_layers)
        self.drop_path_rate = drop_path_rate

        self.layer_cfgs = _layer_cfgs
        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [copy.deepcopy(_layer_cfgs) for _ in range(self.num_layers)]

        for i in range(self.num_layers):
            _layer_cfg_i = copy.deepcopy(layer_cfgs[i])
            _layer_cfg_i.update({
                "embed_dims": self.embed_dims,
                "drop_path_rate": dpr[i],
            })
            if i in self.layers_with_dwconv:
                _layer_cfg_i.update({"with_dwconv": True})
            else:
                _layer_cfg_i.update({"with_dwconv": False})
            # SAVSS_Layer expects (embed_dims, use_rms_norm, with_dwconv, layer_cfgs, drop_path_rate)
            # layer_cfgs must contain mamba_cfg; extract top-level args to avoid passing mamba_cfg as kwarg
            layer_cfgs_for_savss = {
                k: v for k, v in _layer_cfg_i.items()
                if k not in ("embed_dims", "use_rms_norm", "with_dwconv", "drop_path_rate")
            }
            self.layers.append(
                SAVSS_Layer(
                    embed_dims=_layer_cfg_i["embed_dims"],
                    use_rms_norm=_layer_cfg_i["use_rms_norm"],
                    with_dwconv=_layer_cfg_i["with_dwconv"],
                    layer_cfgs=layer_cfgs_for_savss,
                    drop_path_rate=_layer_cfg_i["drop_path_rate"],
                )
            )

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        for i in out_indices:
            if i != self.num_layers - 1:
                if norm_cfg is not None:
                    norm_layer = build_norm_layer(norm_cfg, self.embed_dims)[1]
                else:
                    norm_layer = nn.Identity()
                self.add_module(f'norm_layer{i}', norm_layer)

        # 4-stage conv projections to 128/64/32/16 with upsample
        self.conv256to128 = BottConv(
            in_channels=256, out_channels=128,
            mid_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv256to64 = BottConv(
            in_channels=256, out_channels=64,
            mid_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv256to32 = BottConv(
            in_channels=256, out_channels=32,
            mid_channels=8, kernel_size=1, stride=1, padding=0)
        self.conv256to16 = BottConv(
            in_channels=256, out_channels=16,
            mid_channels=4, kernel_size=1, stride=1, padding=0)
        self.gn128 = nn.GroupNorm(num_channels=128, num_groups=8)
        self.gn64 = nn.GroupNorm(num_channels=64, num_groups=4)
        self.gn32 = nn.GroupNorm(num_channels=32, num_groups=2)
        self.gn16 = nn.GroupNorm(num_channels=16, num_groups=2)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        if not hasattr(self, 'pos_embed'):
            return
        if not (hasattr(self, 'init_cfg')
                and isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            if self.with_pos_embed:
                trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x, patch_resolution = self.patch_embed(x)
        if self.with_pos_embed:
            pos_embed = resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=0
            )
            x = x + pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, hw_shape=patch_resolution)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x.reshape(B, *patch_resolution, C)
                if i != self.num_layers - 1:
                    norm_layer = getattr(self, f'norm_layer{i}')
                    patch_token = norm_layer(patch_token)
                patch_token = patch_token.permute(0, 3, 1, 2)

                if i == self.out_indices[0]:
                    feat = self.gn128(self.conv256to128(patch_token))
                    feat = nn.Upsample(
                        size=(64, 64), mode="bilinear")(feat)
                    outs.append(feat)
                elif i == self.out_indices[1]:
                    feat = self.gn64(self.conv256to64(patch_token))
                    feat = nn.Upsample(
                        size=(128, 128), mode="bilinear")(feat)
                    outs.append(feat)
                elif i == self.out_indices[2]:
                    feat = self.gn32(self.conv256to32(patch_token))
                    feat = nn.Upsample(
                        size=(256, 256), mode="bilinear")(feat)
                    outs.append(feat)
                elif i == self.out_indices[3]:
                    feat = self.gn16(self.conv256to16(patch_token))
                    feat = nn.Upsample(
                        size=(512, 512), mode="bilinear")(feat)
                    outs.append(feat)

        return outs

