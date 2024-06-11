import math
import warnings

import torch
import torch.nn.functional as F

from math import ceil
import numpy as np
from typing import Optional

from torch import nn, einsum, Tensor
from einops import rearrange, reduce

from torch.nn.parameter import Parameter
from torch.overrides import has_torch_function, handle_torch_function
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear as _LinearWithBias

from .utils import exists, moore_penrose_iter_pinv, DropPath


__all__ = [
    "Attention",
    "NystromAttention",
    "PreNorm",
    "FeedForward",
    "Nystromformer",
    "multi_head_attention_forward",
    "MMAttention",
    "MMAttentionLayer",
    "MultiheadAttention",
    "RegionAttntion",
    "CrossRegionAttntion",
    "MITransformerLayer",
    "Mlp",
    "CASCOAttention",
]


class Mlp(nn.Module):
    # two mlp, fc-relu-drop-fc-relu-drop
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Attention from `"Dynamic Domain Generalization" <https://github.com/MetaVisionLab/DDG>`_.
    """

    def __init__(
        self,
        in_channels: int,
        out_features: int,
        squeeze=None,
        bias: bool = True
    ):
        super(Attention, self).__init__()
        self.squeeze = squeeze if squeeze else in_channels // 16
        assert self.squeeze > 0
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, self.squeeze, bias=bias)
        self.fc2 = nn.Linear(self.squeeze, out_features, bias=bias)
        self.sf = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.avg_pool(x).view(x.shape[:-2])
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        return self.sf(x)


# main attention class
class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.0,
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(
                padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = * \
            x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, "b n -> b () n")
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = "... (n l) d -> ... n d"
        q_landmarks = reduce(q, landmark_einops_eq, "sum", l=l)
        k_landmarks = reduce(k, landmark_einops_eq, "sum", l=l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, "... (n l) -> ... n", "sum", l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = "... i d, ... j d -> ... i j"
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(
                ~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(
                ~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(
                ~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(
            lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out


def region_partition(x, region_size):
    """
    Args:
        x: (B, H, W, C)
        region_size (int): region size
    Returns:
        regions: (num_regions*B, region_size, region_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // region_size, region_size,
               W // region_size, region_size, C)
    regions = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, region_size, region_size, C)
    return regions


def region_reverse(regions, region_size, H, W):
    """
    Args:
        regions: (num_regions*B, region_size, region_size, C)
        region_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(regions.shape[0] / (H * W / region_size / region_size))
    x = regions.view(B, H // region_size, W // region_size,
                     region_size, region_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class InnerAttention(nn.Module):
    def __init__(self, dim, head_dim=None, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., epeg=True, epeg_k=15, epeg_2d=False, epeg_bias=True, epeg_type='attn'):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.epeg_2d = epeg_2d
        self.epeg_type = epeg_type
        if epeg:
            padding = epeg_k // 2
            if epeg_2d:
                if epeg_type == 'attn':
                    self.pe = nn.Conv2d(
                        num_heads, num_heads, epeg_k, padding=padding, groups=num_heads, bias=epeg_bias)
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, epeg_k,
                                        padding=padding, groups=head_dim * num_heads, bias=epeg_bias)
            else:
                if epeg_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, (epeg_k, 1), padding=(
                        padding, 0), groups=num_heads, bias=epeg_bias)
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, (epeg_k, 1), padding=(
                        padding, 0), groups=head_dim * num_heads, bias=epeg_bias)
        else:
            self.pe = None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_regions*B, N, C)
        """
        B_, N, C = x.shape

        # x = self.pe(x)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.pe is not None and self.epeg_type == 'attn':
            pe = self.pe(attn)
            attn = attn+pe

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if self.pe is not None and self.epeg_type == 'value_bf':
            # B,H,N,C -> B,HC,N-0.5,N-0.5
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(
                B_, C, int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))))
            # pe = torch.einsum('ahbd->abhd',pe).flatten(-2,-1)
            v = v + pe.reshape(B_, self.num_heads,
                               self.head_dim, N).permute(0, 1, 3, 2)

        # print(v.size())

        x = (attn @ v).transpose(1, 2).reshape(B_,
                                               N, self.num_heads*self.head_dim)

        if self.pe is not None and self.epeg_type == 'value_af':
            # print(v.size())
            pe = self.pe(v.permute(0, 3, 1, 2).reshape(
                B_, C, int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))))
            # print(pe.size())
            # print(v.size())
            x = x + pe.reshape(B_, self.num_heads *
                               self.head_dim, N).transpose(-1, -2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, region_size={self.region_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 region with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class RegionAttntion(nn.Module):
    def __init__(self, dim, head_dim=None, num_heads=8, region_size=0, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., region_num=8, epeg=False, min_region_num=0, min_region_ratio=0., region_attn='native', **kawrgs):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio

        if region_attn == 'native':
            self.attn = InnerAttention(
                dim, head_dim=head_dim, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, epeg=epeg, **kawrgs)
        elif region_attn == 'ntrans':
            self.attn = NystromAttention(
                dim=dim,
                dim_head=head_dim,
                heads=num_heads,
                dropout=drop
            )

    def padding(self, x):
        B, L, C = x.shape
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H+_n, W+_n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H+_n, W+_n
            region_size = int(H // self.region_num)
            region_num = self.region_num

        add_length = H * W - L

        # if padding much，i will give up region attention. only for ablation
        if (add_length > L / (self.min_region_ratio+1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H+_n, W+_n
            add_length = H * W - L
            region_size = H
        if add_length > 0:
            x = torch.cat(
                [x, torch.zeros((B, add_length, C), device=x.device)], dim=1)

        return x, H, W, add_length, region_num, region_size

    def forward(self, x, return_attn=False):
        B, L, C = x.shape

        # padding
        x, H, W, add_length, region_num, region_size = self.padding(x)

        x = x.view(B, H, W, C)

        # partition regions
        # nW*B, region_size, region_size, C
        x_regions = region_partition(x, region_size)

        # nW*B, region_size*region_size, C
        x_regions = x_regions.view(-1, region_size * region_size, C)

        # R-MSA
        attn_regions = self.attn(x_regions)  # nW*B, region_size*region_size, C

        # merge regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)

        x = region_reverse(attn_regions, region_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        if add_length > 0:
            x = x[:, :-add_length]

        return x


class CrossRegionAttntion(nn.Module):
    def __init__(self, dim, head_dim=None, num_heads=8, region_size=0, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., region_num=8, epeg=False, min_region_num=0, min_region_ratio=0., crmsa_k=3, crmsa_mlp=False, region_attn='native', **kawrgs):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio

        self.attn = InnerAttention(
            dim, head_dim=head_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, epeg=epeg, **kawrgs)

        self.crmsa_mlp = crmsa_mlp
        if crmsa_mlp:
            self.phi = [nn.Linear(self.dim, self.dim // 4, bias=False)]
            self.phi += [nn.Tanh()]
            self.phi += [nn.Linear(self.dim // 4, crmsa_k, bias=False)]
            self.phi = nn.Sequential(*self.phi)
        else:
            self.phi = nn.Parameter(
                torch.empty(
                    (self.dim, crmsa_k),
                )
            )
        nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def padding(self, x):
        B, L, C = x.shape
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H+_n, W+_n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H+_n, W+_n
            region_size = int(H // self.region_num)
            region_num = self.region_num

        add_length = H * W - L

        # if padding much，i will give up region attention. only for ablation
        if (add_length > L / (self.min_region_ratio+1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H+_n, W+_n
            add_length = H * W - L
            region_size = H
        if add_length > 0:
            x = torch.cat(
                [x, torch.zeros((B, add_length, C), device=x.device)], dim=1)

        return x, H, W, add_length, region_num, region_size

    def forward(self, x, return_attn=False):
        B, L, C = x.shape

        # padding
        x, H, W, add_length, region_num, region_size = self.padding(x)

        x = x.view(B, H, W, C)

        # partition regions
        # nW*B, region_size, region_size, C
        x_regions = region_partition(x, region_size)

        # nW*B, region_size*region_size, C
        x_regions = x_regions.view(-1, region_size * region_size, C)

        # CR-MSA
        if self.crmsa_mlp:
            logits = self.phi(x_regions).transpose(
                1, 2)  # W*B, sW, region_size*region_size
        else:
            # nW*B, sW, region_size*region_size
            logits = torch.einsum("w p c, c n -> w p n",
                                  x_regions, self.phi).transpose(1, 2)

        combine_weights = logits.softmax(dim=-1)
        dispatch_weights = logits.softmax(dim=1)

        logits_min, _ = logits.min(dim=-1)
        logits_max, _ = logits.max(dim=-1)
        dispatch_weights_mm = (logits - logits_min.unsqueeze(-1)) / \
            (logits_max.unsqueeze(-1) - logits_min.unsqueeze(-1) + 1e-8)

        attn_regions = torch.einsum("w p c, w n p -> w n p c", x_regions,
                                    combine_weights).sum(dim=-2).transpose(0, 1)  # sW, nW, C

        if return_attn:
            attn_regions, _attn = self.attn(
                attn_regions, return_attn)  # sW, nW, C
            attn_regions = attn_regions.transpose(0, 1)  # nW, sW, C
        else:
            attn_regions = self.attn(attn_regions).transpose(0, 1)  # nW, sW, C

        # nW, sW, region_size*region_size, C
        attn_regions = torch.einsum(
            "w n c, w n p -> w n p c", attn_regions, dispatch_weights_mm)
        # nW, region_size*region_size, C
        attn_regions = torch.einsum(
            "w n p c, w n p -> w n p c", attn_regions, dispatch_weights).sum(dim=1)

        # merge regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)

        x = region_reverse(attn_regions, region_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        if add_length > 0:
            x = x[:, :-add_length]

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(self.norm(x))


class Nystromformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        attn_values_residual=True,
        attn_values_residual_conv_kernel=33,
        attn_dropout=0.0,
        ff_dropout=0.0
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            NystromAttention(
                                dim=dim,
                                dim_head=dim_head,
                                heads=heads,
                                num_landmarks=num_landmarks,
                                pinv_iterations=pinv_iterations,
                                residual=attn_values_residual,
                                residual_conv_kernel=attn_values_residual_conv_kernel,
                                dropout=attn_dropout,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim=dim, dropout=ff_dropout)),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    need_raw: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias,
                bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            need_raw=need_raw,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
        )
    tgt_len, bsz, embed_dim = query.size()
    # import pdb;pdb.set_trace()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = F.linear(query, in_proj_weight,
                               in_proj_bias).chunk(3, dim=-1)

        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt,
                         in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt,
                         in_proj_bias[embed_dim: (embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt,
                         in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert (
            attn_mask.dtype == torch.float32
            or attn_mask.dtype == torch.float64
            or attn_mask.dtype == torch.float16
            or attn_mask.dtype == torch.uint8
            or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError(
                    "The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError(
                    "The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError(
                "attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        # import pdb;pdb.set_trace()
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()
                      [2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()
                      [2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [
        bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(
            bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, tgt_len, src_len)

    attn_output_weights_raw = attn_output_weights
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(
        attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(
        0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        if need_raw:
            attn_output_weights_raw = attn_output_weights_raw.view(
                bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights_raw

            # attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            # return attn_output, attn_output_weights.sum(dim=1) / num_heads, attn_output_weights_raw, attn_output_weights_raw.sum(dim=1) / num_heads
        else:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class MMAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.,
        num_pathways=281,
    ):
        super().__init__()
        self.num_pathways = num_pathways
        self.eps = eps
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(
                padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, eps = *x.shape, self.heads, self.num_pathways, self.eps

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if mask != None:
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        # regular transformer scaling
        q = q * self.scale

        # extract the pathway/histology queries and keys
        # bs x head x num_pathways x dim
        q_pathways = q[:, :, :self.num_pathways, :]
        k_pathways = k[:, :, :self.num_pathways, :]

        # bs x head x num_patches x dim
        q_histology = q[:, :, self.num_pathways:, :]
        k_histology = k[:, :, self.num_pathways:, :]

        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        cross_attn_histology = einsum(einops_eq, q_histology, k_pathways)
        attn_pathways = einsum(einops_eq, q_pathways, k_pathways)
        cross_attn_pathways = einsum(einops_eq, q_pathways, k_histology)

        # softmax
        pre_softmax_cross_attn_histology = cross_attn_histology
        cross_attn_histology = cross_attn_histology.softmax(dim=-1)
        attn_pathways_histology = torch.cat(
            (attn_pathways, cross_attn_pathways), dim=-1).softmax(dim=-1)

        # compute output
        out_pathways = attn_pathways_histology @ v
        out_histology = cross_attn_histology @ v[:, :, :self.num_pathways]

        out = torch.cat((out_pathways, out_histology), dim=2)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)

        if return_attn:
            # return three matrices
            return out, attn_pathways.squeeze().detach().cpu(), cross_attn_pathways.squeeze().detach().cpu(), pre_softmax_cross_attn_histology.squeeze().detach().cpu()

        return out


class MMAttentionLayer(nn.Module):
    """
    Applies layer norm --> attention
    """

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dim_head=64,
        heads=6,
        residual=True,
        dropout=0.,
        num_pathways=281,
    ):

        super().__init__()
        self.norm = norm_layer(dim)
        self.num_pathways = num_pathways
        self.attn = MMAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            residual=residual,
            dropout=dropout,
            num_pathways=num_pathways
        )

    def forward(self, x=None, mask=None, return_attention=False):

        if return_attention:
            x, attn_pathways, cross_attn_pathways, cross_attn_histology = self.attn(
                x=self.norm(x), mask=mask, return_attn=True)
            return x, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            x = self.attn(x=self.norm(x), mask=mask)

        return x


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * \
            num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(
                torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, need_raw=True, attn_mask=None):
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a ByteTensor is provided, the non-zero positions will be ignored while the position
              with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
              positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
              while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
              is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                need_raw=need_raw,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )
        else:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                need_raw=need_raw,
                attn_mask=attn_mask,
            )


class MIAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super(MIAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, histology, pathways, global_feature, return_attn=False):
        B, N_histology, C = histology.shape
        _, N_pathways, _ = pathways.shape

        qkv_histology = self.qkv(histology).reshape(
            B, N_histology, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_histology, k_histology, v_histology = qkv_histology[0], qkv_histology[1], qkv_histology[2]

        qkv_pathways = self.qkv(pathways).reshape(
            B, N_pathways, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_pathways, k_pathways, v_pathways = qkv_pathways[0], qkv_pathways[1], qkv_pathways[2]

        qkv_global = self.qkv(global_feature).reshape(
            B, 1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_global, k_global, v_global = qkv_global[0], qkv_global[1], qkv_global[2]

        # self-attention for modality-specific features
        attn_histology = (
            q_histology @ k_histology.transpose(-2, -1)) * self.scale
        attn_pathways = (
            q_pathways @ k_pathways.transpose(-2, -1)) * self.scale

        # cross-attention for modality-common features
        attn_global = (q_global @ torch.cat((k_global, k_histology,
                       k_pathways), dim=2).transpose(-2, -1)) * self.scale

        attn_histology = attn_histology.softmax(dim=-1)
        attn_pathways = attn_pathways.softmax(dim=-1)
        attn_global = attn_global.softmax(dim=-1)

        attn_histology = self.attn_drop(attn_histology)
        attn_pathways = self.attn_drop(attn_pathways)
        attn_global = self.attn_drop(attn_global)

        attn_histology_x = (
            attn_histology @ v_histology).transpose(1, 2).reshape(B, N_histology, C)
        attn_pathways_x = (
            attn_pathways @ v_pathways).transpose(1, 2).reshape(B, N_pathways, C)
        attn_global_x = (attn_global @ torch.cat((v_global, v_histology,
                         v_pathways), dim=2)).transpose(1, 2).reshape(B, 1, C)

        attn_histology_x = self.proj(attn_histology_x)
        attn_pathways_x = self.proj(attn_pathways_x)
        attn_global_x = self.proj(attn_global_x)

        attn_histology_x = self.proj_drop(attn_histology_x)
        attn_pathways_x = self.proj_drop(attn_pathways_x)
        attn_global_x = self.proj_drop(attn_global_x)

        if return_attn:
            return [attn_histology_x, attn_pathways_x, attn_global_x], [attn_histology, attn_pathways]
        else:
            return [attn_histology_x, attn_pathways_x, attn_global_x]


class MITransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=1.,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super(MITransformerLayer, self).__init__()
        self.norm1 = norm_layer(dim)

        self.attn = MIAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       out_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_path)

    def forward(self, histology, pathways, global_feature, return_attn=False):
        if return_attn:
            [attn_histology_x, attn_pathways_x, attn_global_x], \
                [attn_histology, attn_pathways,] = self.attn(self.norm1(
                    histology), self.norm1(pathways), self.norm1(global_feature), return_attn)
        else:
            attn_histology_x, attn_pathways_x, attn_global_x = self.attn(self.norm1(
                histology), self.norm1(pathways), self.norm1(global_feature), return_attn)

        histology = histology + self.drop_path(attn_histology_x)
        pathways = pathways + self.drop_path(attn_pathways_x)
        global_feature = global_feature + self.drop_path(attn_global_x)

        histology = histology + self.drop_path(self.mlp(self.norm2(histology)))
        pathways = pathways + self.drop_path(self.mlp(self.norm2(pathways)))
        global_feature = global_feature + \
            self.drop_path(self.mlp(self.norm2(global_feature)))

        if return_attn:
            return histology, pathways,  global_feature, [attn_histology, attn_pathways]
        else:
            return histology, pathways, global_feature


class CASCOAttention(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads=8,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super(CASCOAttention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, top_k=512, return_attn=False):
        
        # B, num_patches, head_size*num_head
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        # B, num_head, num_patches, head_size
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        # B, num_head, num_patches, head_size
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # B, num_head, num_patches, num_patches
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        # B, num_head, num_patches(query), num_patches(key)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if return_attn else None
        attention_probs = self.attn_dropout(attention_probs)
        # import pdb;pdb.set_trace()

        # selection score\
        # import pdb;pdb.set_trace()
        if top_k > 0:
            # import pdb;pdb.set_trace()
            # top_k_a = torch.topk(attention_probs, top_k, dim=-1)
            # indices = top_k_a.indices
            # mask = torch.zeros_like(attention_probs, dtype=torch.bool)
            # mask = mask.expand_as(attention_probs)
            # mask.scatter_(-1, indices, True)
            # attention_probs = attention_probs.masked_fill(~mask, 0)
            select_attention = torch.mean(attention_probs, dim=1)
            mean_attention_probs = torch.mean(select_attention, dim=1)
            top_k_s = torch.topk(mean_attention_probs, top_k, dim=-1)
            k_idx = top_k_s.indices
            select_value = torch.gather(value, 1, k_idx.unsqueeze(-1).expand(-1, -1, value.shape[-1]))
        else:
            select_value = None

        # B, num_head, num_patches, head_size
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, select_value, weights
