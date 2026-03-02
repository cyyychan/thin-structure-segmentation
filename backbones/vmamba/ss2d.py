# Copyright (c) OpenMMLab. All rights reserved.
# SS2D (Selective Scan 2D) and mamba_init, ported from VMamba.
# Uses local kernels: csm_triton, csms6s, mamba2 (copied from VMamba repo).

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import cross_scan_fn, cross_merge_fn, selective_scan_fn
from .layers import Linear2d, LayerNorm2d, Permute, SoftmaxSpatial


# ============== mamba_init ==============
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init='random', dt_min=0.001,
                dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == 'constant':
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == 'random':
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) +
            math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(
            1, d_state + 1, dtype=torch.float32,
            device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min,
                   dt_max, dt_init_floor, k_group=4):
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max,
                        dt_init_floor) for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in dt_projs], dim=0))
        dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


# ============== SS2Dv0 ==============
class SS2Dv0:
    def __initv0__(self, d_model=96, d_state=16, ssm_ratio=2.0, dt_rank='auto',
                   dropout=0.0, seq=False, force_fp32=True, **kwargs):
        if kwargs.get('channel_first'):
            assert not kwargs['channel_first']
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == 'auto' else dt_rank
        self.forward = self.forwardv0
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.act = nn.SiLU()
        self.conv2d = nn.Conv2d(
            d_inner, d_inner, groups=d_inner, bias=True, kernel_size=3,
            padding=1)
        self.x_proj_weight = nn.Parameter(
            torch.stack([
                nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False).weight
                for _ in range(4)
            ],
                       dim=0))
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = (
            mamba_init.init_dt_A_D(
                d_state, dt_rank, d_inner, 1.0, 'random', 0.001, 0.1, 1e-4,
                k_group=4))
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forwardv0(self, x, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x)
        x = self.act(x)
        selective_scan = partial(selective_scan_fn, backend='mamba')

        B, D, H, W = x.shape
        K = 4
        N = self.A_logs.shape[1]
        R = self.dt_projs_weight.shape[2]
        L = H * W

        x_hwwh = torch.stack([
            x.view(B, -1, L),
            torch.transpose(x, 2, 3).contiguous().view(B, -1, L)
        ],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum('b k d l, k c d -> b k c l', xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum('b k r l, k d r -> b k d l', dts,
                           self.dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous()
        Cs = Cs.contiguous()
        As = -self.A_logs.float().exp()
        Ds = self.Ds.float()
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i],
                    dts.view(B, K, -1, L)[:, i],
                    As.view(K, -1, N)[i],
                    Bs[:, i].unsqueeze(1),
                    Cs[:, i].unsqueeze(1),
                    Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts, As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), 2,
                               3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), 2,
                                  3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(1, 2).contiguous()
        y = self.out_norm(y).view(B, H, W, -1)
        y = y * z
        return self.dropout(self.out_proj(y))


# ============== SS2Dv2 ==============
class SS2Dv2:
    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value

    @staticmethod
    def get_outnorm(forward_type='', d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        out_norm_none, forward_type = checkpostfix('_onnone', forward_type)
        out_norm_dwconv3, forward_type = checkpostfix('_ondwconv3', forward_type)
        out_norm_cnorm, forward_type = checkpostfix('_oncnorm', forward_type)
        out_norm_softmax, forward_type = checkpostfix('_onsoftmax', forward_type)
        out_norm_sigmoid, forward_type = checkpostfix('_onsigmoid', forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(
                    d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner,
                    bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(
                    d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner,
                    bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)
        return out_norm, forward_type

    def __initv2__(self, d_model=96, d_state=16, ssm_ratio=2.0, dt_rank='auto',
                  act_layer=nn.SiLU, d_conv=3, conv_bias=True, dropout=0.0,
                  bias=False, dt_min=0.001, dt_max=0.1, dt_init='random',
                  dt_scale=1.0, dt_init_floor=1e-4, initialize='v0',
                  forward_type='v2', channel_first=False, **kwargs):
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(
            math.ceil(self.d_model / 16) if dt_rank == 'auto' else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix('_no32', forward_type)
        self.oact, forward_type = checkpostfix('_oact', forward_type)
        self.disable_z, forward_type = checkpostfix('_noz', forward_type)
        self.disable_z_act, forward_type = checkpostfix('_nozact', forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner,
                                                     channel_first)

        FORWARD_TYPES = dict(
            v01=partial(
                self.forward_corev2,
                force_fp32=True,
                selective_scan_backend='mamba',
                scan_force_torch=True),
            v02=partial(
                self.forward_corev2,
                force_fp32=True,
                selective_scan_backend='mamba'),
            v05=partial(
                self.forward_corev2,
                force_fp32=False,
                no_einsum=True,
                selective_scan_backend='oflex'),
            v2=partial(
                self.forward_corev2,
                force_fp32=True,
                selective_scan_backend='core'),
            v3=partial(
                self.forward_corev2,
                force_fp32=False,
                selective_scan_backend='oflex'),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type)
        if self.forward_core is None:
            self.forward_core = partial(
                self.forward_corev2,
                force_fp32=(not getattr(self, 'disable_force32', False)),
                no_einsum=True,
                selective_scan_backend='oflex')

        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act = act_layer()
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                self.d_inner,
                self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
            )

        self.x_proj_weight = nn.Parameter(
            torch.stack([
                nn.Linear(self.d_inner,
                          (self.dt_rank + self.d_state * 2),
                          bias=False).weight for _ in range(self.k_group)
            ],
                       dim=0))
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize == 'v0':
            (self.A_logs, self.Ds, self.dt_projs_weight,
             self.dt_projs_bias) = mamba_init.init_dt_A_D(
                 self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init,
                 dt_min, dt_max, dt_init_floor, k_group=self.k_group)
        elif initialize == 'v1':
            self.Ds = nn.Parameter(torch.ones(self.k_group * self.d_inner))
            self.A_logs = nn.Parameter(
                torch.randn(self.k_group * self.d_inner, self.d_state))
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn(self.k_group, self.d_inner, self.dt_rank))
            self.dt_projs_bias = nn.Parameter(
                0.1 * torch.randn(self.k_group, self.d_inner))
        elif initialize == 'v2':
            self.Ds = nn.Parameter(torch.ones(self.k_group * self.d_inner))
            self.A_logs = nn.Parameter(
                torch.zeros(self.k_group * self.d_inner, self.d_state))
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.rand(self.k_group, self.d_inner, self.dt_rank))
            self.dt_projs_bias = nn.Parameter(
                0.1 * torch.rand(self.k_group, self.d_inner))

    def forward_corev2(self,
                       x,
                       force_fp32=False,
                       ssoflex=True,
                       no_einsum=False,
                       selective_scan_backend='oflex',
                       scan_mode='cross2d',
                       scan_force_torch=False,
                       **kwargs):
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(
            scan_mode, 0) if isinstance(scan_mode, str) else scan_mode
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, R = self.k_group, self.dt_rank
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None,
                          delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias,
                                    delta_softplus, ssoflex,
                                    backend=selective_scan_backend)

        if _scan_mode == -1:
            x_proj_bias = getattr(self, 'x_proj_bias', None)

            def scan_rowcol(x, proj_weight, proj_bias, dt_weight, dt_bias, _As,
                            _Ds, width=True):
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)
                if no_einsum:
                    x_dbl = F.conv1d(
                        xs.view(_B, -1, _L),
                        proj_weight.view(-1, _D, 1),
                        bias=proj_bias.view(-1) if proj_bias is not None else
                        None,
                        groups=2)
                    dts, Bs, Cs = torch.split(
                        x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(
                        dts.contiguous().view(_B, -1, _L),
                        dt_weight.view(2 * _D, -1, 1),
                        groups=2)
                else:
                    x_dbl = torch.einsum('b k d l, k c d -> b k c l', xs,
                                         proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum('b k r l, k d r -> b k d l', dts,
                                       dt_weight)
                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)
                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)
                ys = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias,
                                   delta_softplus).view(_B, 2, -1, _L)
                return ys

            As = -self.A_logs.to(torch.float).exp().view(4, -1, N)
            x = F.layer_norm(
                x.permute(0, 2, 3, 1), (int(x.shape[1]),)).permute(
                    0, 3, 1, 2).contiguous()
            x_proj_bias = getattr(self, 'x_proj_bias', None)
            y_row = scan_rowcol(
                x,
                self.x_proj_weight.view(4, -1, D)[:2].contiguous(),
                (x_proj_bias.view(4, -1)[:2].contiguous()
                 if x_proj_bias is not None else None),
                self.dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                (self.dt_projs_bias.view(4, -1)[:2].contiguous()
                 if getattr(self, 'dt_projs_bias', None) is not None else None),
                As[:2].contiguous().view(-1, N),
                self.Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3)
            y_row = F.layer_norm(
                y_row.permute(0, 2, 3, 1),
                (int(y_row.shape[1]),)).permute(0, 3, 1, 2).contiguous()
            y_col = scan_rowcol(
                y_row,
                self.x_proj_weight.view(4, -1, D)[2:].contiguous().to(
                    y_row.dtype),
                (x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype)
                 if x_proj_bias is not None else None),
                self.dt_projs_weight.view(4, D, -1)[2:].contiguous().to(
                    y_row.dtype),
                (self.dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype)
                 if getattr(self, 'dt_projs_bias', None) is not None else None),
                As[2:].contiguous().view(-1, N),
                self.Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            x_proj_bias = getattr(self, 'x_proj_bias', None)
            xs = cross_scan_fn(
                x,
                in_channel_first=True,
                out_channel_first=True,
                scans=_scan_mode,
                force_torch=scan_force_torch)
            if no_einsum:
                x_dbl = F.conv1d(
                    xs.view(B, -1, L),
                    self.x_proj_weight.view(-1, D, 1),
                    bias=(x_proj_bias.view(-1)
                          if x_proj_bias is not None else None),
                    groups=K)
                dts, Bs, Cs = torch.split(
                    x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                if hasattr(self, 'dt_projs_weight'):
                    dts = F.conv1d(
                        dts.contiguous().view(B, -1, L),
                        self.dt_projs_weight.view(K * D, -1, 1),
                        groups=K)
            else:
                x_dbl = torch.einsum('b k d l, k c d -> b k c l', xs,
                                     self.x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                if hasattr(self, 'dt_projs_weight'):
                    dts = torch.einsum('b k r l, k d r -> b k d l', dts,
                                       self.dt_projs_weight)

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp()
            Ds = self.Ds.to(torch.float)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)
            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys = selective_scan(xs, dts, As, Bs, Cs, Ds, delta_bias,
                               delta_softplus).view(B, K, -1, H, W)
            y = cross_merge_fn(
                ys,
                in_channel_first=True,
                out_channel_first=True,
                scans=_scan_mode,
                force_torch=scan_force_torch)

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(1, 2).contiguous().view(
                B, H, W, -1)
        y = out_norm(y)
        return y.to(x.dtype)

    def forwardv2(self, x, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        return self.dropout(self.out_proj(y))


# ============== SS2D ==============
class SS2D(nn.Module, SS2Dv0, SS2Dv2):
    def __init__(self,
                 d_model=96,
                 d_state=16,
                 ssm_ratio=2.0,
                 dt_rank='auto',
                 act_layer=nn.SiLU,
                 d_conv=3,
                 conv_bias=True,
                 dropout=0.0,
                 bias=False,
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init='random',
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 initialize='v0',
                 forward_type='v2',
                 channel_first=False,
                 **kwargs):
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            act_layer=act_layer,
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            bias=bias,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            initialize=initialize,
            forward_type=forward_type,
            channel_first=channel_first,
        )
        if forward_type in ['v0', 'v0seq']:
            self.__initv0__(seq=('seq' in forward_type), **kwargs)
        else:
            self.__initv2__(**kwargs)
