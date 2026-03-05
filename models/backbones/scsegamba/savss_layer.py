import math

from einops import repeat
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_dropout
from timm.models.layers import trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layer_norm import RMSNorm

from .gbc import GBC, BottConv
from .paf import PAF


class SAVSS_2D(nn.Module):
    """2D selective scan block used inside SAVSS."""

    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_size=7,
            bias=False,
            conv_bias=False,
            init_layer_scale=None,
            default_hw_shape=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_hw_shape = default_hw_shape
        self.n_directions = 4

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(
                init_layer_scale * torch.ones((d_model)),
                requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert conv_size % 2 == 1
        self.conv2d = BottConv(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            mid_channels=self.d_inner // 16,
            kernel_size=3,
            padding=1,
            stride=1)
        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank,
            self.d_inner,
            bias=True,
        )

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.direction_Bs = nn.Parameter(
            torch.zeros(self.n_directions + 1, self.d_state))
        trunc_normal_(self.direction_Bs, std=0.02)

    def sass(self, hw_shape):
        H, W = hw_shape
        L = H * W
        o1, o2, o3, o4 = [], [], [], []
        d1, d2, d3, d4 = [], [], [], []
        o1_inverse = [-1 for _ in range(L)]
        o2_inverse = [-1 for _ in range(L)]
        o3_inverse = [-1 for _ in range(L)]
        o4_inverse = [-1 for _ in range(L)]

        if H % 2 == 1:
            i, j = H - 1, W - 1
            j_d = "left"
        else:
            i, j = H - 1, 0
            j_d = "right"

        while i > -1:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W - 1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "right"
        d1 = [0] + d1[:-1]

        i, j = 0, 0
        i_d = "down"
        while j < W:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d2.append(4)
                else:
                    j = j + 1
                    d2.append(1)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d2.append(3)
                else:
                    j = j + 1
                    d2.append(1)
                    i_d = "down"
        d2 = [0] + d2[:-1]

        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o1) - 1
                        d3.append(1 if j == diag else 4)
            else:
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o1) - 1
                        d3.append(4 if i == diag else 1)
        d3 = [0] + d3[:-1]

        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + (W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1
                        d4.append(1 if j == diag else 4)
            else:
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + (W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1
                        d4.append(4 if i == diag else 1)
        d4 = [0] + d4[:-1]

        return (tuple(o1), tuple(o2), tuple(o3), tuple(o4)), \
            (tuple(o1_inverse), tuple(o2_inverse), tuple(o3_inverse), tuple(o4_inverse)), \
            (tuple(d1), tuple(d2), tuple(d3), tuple(d4))

    def forward(self, x, hw_shape):
        batch_size, L, _ = x.shape
        H, W = hw_shape
        E = self.d_inner

        xz = self.in_proj(x)
        A = -torch.exp(self.A_log.float())

        x, z = xz.chunk(2, dim=-1)
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)

        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        dt = dt.permute(0, 2, 1).contiguous()
        B = B.permute(0, 2, 1).contiguous()
        C = C.permute(0, 2, 1).contiguous()

        orders, inverse_orders, directions = self.sass(hw_shape)
        direction_Bs = [self.direction_Bs[d, :] for d in directions]
        direction_Bs = [
            dB[None, :, :].expand(batch_size, -1, -1).permute(0, 2, 1).to(
                dtype=B.dtype) for dB in direction_Bs
        ]

        y_scan = [
            selective_scan_fn(
                x_conv[:, o, :].permute(0, 2, 1).contiguous(),
                dt,
                A,
                (B + dB).contiguous(),
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            ).permute(0, 2, 1)[:, inv_order, :]
            for o, inv_order, dB in zip(orders, inverse_orders, direction_Bs)
        ]

        y = sum(y_scan) * self.act(z)
        out = self.out_proj(y)
        if self.init_layer_scale is not None:
            out = out * self.gamma

        return out


class SAVSS_Layer(nn.Module):
    """A full SAVSS stage (Mamba + PAF + GBC) from SCSegamba."""

    def __init__(
            self,
            embed_dims,
            use_rms_norm,
            with_dwconv,
            layer_cfgs,
            drop_path_rate,
    ):
        super().__init__()
        d_model = embed_dims
        d_state = layer_cfgs['mamba_cfg']['d_state']
        expand = layer_cfgs['mamba_cfg']['expand']
        conv_size = layer_cfgs['mamba_cfg']['conv_size']
        default_hw_shape = layer_cfgs['mamba_cfg']['default_hw_shape']

        self.norm1 = RMSNorm(d_model) if use_rms_norm else nn.LayerNorm(
            d_model)
        self.mamba = SAVSS_2D(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            conv_size=conv_size,
            default_hw_shape=default_hw_shape,
        )
        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate))

        self.gbc = GBC(in_channels=256)
        self.paf = PAF(in_channels=256, mid_channels=64)

    def forward(self, x, hw_shape):
        # x: (B,L,C)
        shortcut = x
        x = self.norm1(x)
        x = self.mamba(x, hw_shape)
        x = shortcut + self.drop_path(x)
        # GBC/PAF 在 SAVSS 中用于后处理 256 通道 feature；
        # 这里只保留 Mamba 主干，GBC/PAF 由上层在 C=256 的特征上调用。
        return x

