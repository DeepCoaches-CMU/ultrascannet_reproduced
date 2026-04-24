"""
mobilefcmvitv3/models/mobilevitv3_block.py

MobileViTv3 building blocks re-implemented with sub-module names that match
the official cvnets checkpoint key structure exactly, enabling direct weight
loading from the pretrained MobileViTv3-S checkpoints.

Checkpoint key structure (from mobilevitv3_S_e300_7930):
  ConvLayer  : <name>.block.conv  /  <name>.block.norm
  InvResidual: <name>.block.exp_1x1.block.{conv,norm}
               <name>.block.conv_3x3.block.{conv,norm}
               <name>.block.red_1x1.block.{conv,norm}
  MobileViTv3Block:
    local_rep.conv_3x3.block.{conv,norm}
    local_rep.conv_1x1.block.conv          (no norm/act)
    global_rep.N.pre_norm_mha.0            LayerNorm
    global_rep.N.pre_norm_mha.1.qkv_proj   weight/bias
    global_rep.N.pre_norm_mha.1.out_proj   weight/bias
    global_rep.N.pre_norm_ffn.0            LayerNorm
    global_rep.N.pre_norm_ffn.1            Linear (in→ffn)
    global_rep.N.pre_norm_ffn.4            Linear (ffn→out)
    global_rep.<n_blocks>                  final LayerNorm
    conv_proj.block.{conv,norm}
    fusion.block.{conv,norm}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ── ConvLayer: wraps Conv2d + optional BN + optional activation ───────────────
# Produces keys: <name>.block.conv.weight  /  <name>.block.norm.*

class ConvLayer(nn.Module):
    """
    Mirrors cvnets ConvLayer: stores sub-modules under self.block as a
    Sequential with named children 'conv' and 'norm'.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 groups: int = 1, use_norm: bool = True, use_act: bool = True,
                 bias: bool = None):
        super().__init__()
        if bias is None:
            bias = not use_norm

        layers = nn.Sequential()
        layers.add_module('conv', nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=bias
        ))
        if use_norm:
            layers.add_module('norm', nn.BatchNorm2d(out_channels, eps=1e-4))
        if use_act:
            layers.add_module('act', nn.SiLU(inplace=True))
        self.block = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── InvertedResidual ──────────────────────────────────────────────────────────
# Produces keys: <name>.block.exp_1x1.*  /  .conv_3x3.*  /  .red_1x1.*

class _IRBlock(nn.Module):
    """Inner container that holds exp_1x1, conv_3x3, red_1x1 as named attrs."""
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden = int(in_ch * expand_ratio)
        self.exp_1x1 = ConvLayer(in_ch, hidden, kernel_size=1, padding=0,
                                  use_norm=True, use_act=True)
        self.conv_3x3 = ConvLayer(hidden, hidden, kernel_size=3, stride=stride,
                                   padding=1, groups=hidden,
                                   use_norm=True, use_act=True)
        self.red_1x1 = ConvLayer(hidden, out_ch, kernel_size=1, padding=0,
                                  use_norm=True, use_act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.exp_1x1(x)
        x = self.conv_3x3(x)
        x = self.red_1x1(x)
        return x


class InvertedResidual(nn.Module):
    """
    MobileNetV2-style inverted residual.
    Key structure: self.block.exp_1x1 / .conv_3x3 / .red_1x1
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 expand_ratio: int = 4):
        super().__init__()
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.block = _IRBlock(in_ch, out_ch, stride, expand_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_res else out


# ── TransformerEncoder ────────────────────────────────────────────────────────
# Produces keys:
#   pre_norm_mha.0          LayerNorm
#   pre_norm_mha.1.qkv_proj weight/bias
#   pre_norm_mha.1.out_proj  weight/bias
#   pre_norm_ffn.0          LayerNorm
#   pre_norm_ffn.1          Linear (in → ffn)
#   pre_norm_ffn.4          Linear (ffn → in)

class _MHA(nn.Module):
    """
    Multi-head attention with qkv_proj / out_proj named to match cvnets.
    cvnets uses a single Linear for qkv_proj (not split), so we do the same.
    """
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v,
                                            dropout_p=self.attn_drop.p
                                            if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.out_proj(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Single transformer block matching cvnets TransformerEncoder key structure:
      pre_norm_mha  = Sequential(LayerNorm, _MHA)   indices 0, 1
      pre_norm_ffn  = Sequential(LayerNorm, Linear, SiLU, Dropout, Linear, Dropout)
                      indices                0       1              4
    """
    def __init__(self, dim: int, num_heads: int, ffn_dim: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 ffn_drop: float = 0.0):
        super().__init__()
        # pre_norm_mha: index 0 = LayerNorm, index 1 = MHA
        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(dim),          # [0]
            _MHA(dim, num_heads, attn_drop, proj_drop),  # [1]
        )
        # pre_norm_ffn: index 0 = LN, 1 = Linear, 2 = SiLU, 3 = Dropout, 4 = Linear, 5 = Dropout
        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(dim),          # [0]
            nn.Linear(dim, ffn_dim),    # [1]
            nn.SiLU(),                  # [2]
            nn.Dropout(ffn_drop),       # [3]
            nn.Linear(ffn_dim, dim),    # [4]
            nn.Dropout(proj_drop),      # [5]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual
        norm_x = self.pre_norm_mha[0](x)
        x = x + self.pre_norm_mha[1](norm_x)
        # Pre-norm FFN with residual
        x = x + self.pre_norm_ffn(x)
        return x


# ── MobileViTv3Block ──────────────────────────────────────────────────────────

class MobileViTv3Block(nn.Module):
    """
    MobileViTv3 block with sub-module names matching the official cvnets
    checkpoint exactly.

    Key structure produced:
      local_rep.conv_3x3.*          depthwise 3x3
      local_rep.conv_1x1.block.conv pointwise (no norm/act)
      global_rep.0 ... N-1          TransformerEncoder blocks
      global_rep.N                  final LayerNorm
      conv_proj.*                   project global → in_channels
      fusion.*                      fuse local + global → in_channels
    """

    def __init__(self, in_channels: int, transformer_dim: int, ffn_dim: int,
                 n_transformer_blocks: int = 2, head_dim: int = 32,
                 patch_h: int = 2, patch_w: int = 2,
                 attn_dropout: float = 0.0, dropout: float = 0.0,
                 ffn_dropout: float = 0.0, conv_ksize: int = 3):
        super().__init__()

        assert transformer_dim % head_dim == 0, \
            f"transformer_dim {transformer_dim} must be divisible by head_dim {head_dim}"
        num_heads = transformer_dim // head_dim

        # local_rep: named conv_3x3 (DW) and conv_1x1 (PW, no norm/act)
        self.local_rep = nn.Sequential()
        self.local_rep.add_module('conv_3x3', ConvLayer(
            in_channels, in_channels,
            kernel_size=conv_ksize, padding=conv_ksize // 2,
            groups=in_channels, use_norm=True, use_act=True
        ))
        self.local_rep.add_module('conv_1x1', ConvLayer(
            in_channels, transformer_dim,
            kernel_size=1, padding=0,
            use_norm=False, use_act=False
        ))

        # global_rep: N TransformerEncoder blocks + final LayerNorm
        global_rep = nn.Sequential()
        for i in range(n_transformer_blocks):
            global_rep.add_module(str(i), TransformerEncoder(
                dim=transformer_dim, num_heads=num_heads, ffn_dim=ffn_dim,
                attn_drop=attn_dropout, proj_drop=dropout, ffn_drop=ffn_dropout
            ))
        global_rep.add_module(str(n_transformer_blocks), nn.LayerNorm(transformer_dim))
        self.global_rep = global_rep

        # conv_proj: transformer_dim → in_channels  (with norm+act)
        self.conv_proj = ConvLayer(
            transformer_dim, in_channels,
            kernel_size=1, padding=0, use_norm=True, use_act=True
        )

        # fusion: concat(local, global) → in_channels
        self.fusion = ConvLayer(
            transformer_dim + in_channels, in_channels,
            kernel_size=1, padding=0, use_norm=True, use_act=True
        )

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = patch_h * patch_w

    def _unfold(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, C, H, W = x.shape
        pH, pW = self.patch_h, self.patch_w
        new_H = math.ceil(H / pH) * pH
        new_W = math.ceil(W / pW) * pW
        interpolate = (new_H != H or new_W != W)
        if interpolate:
            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
        nH, nW = new_H // pH, new_W // pW
        N = nH * nW
        # [B, C, nH*pH, nW*pW] → [B*patch_area, N, C]
        x = x.reshape(B * C * nH, pH, nW, pW).transpose(1, 2)
        x = x.reshape(B, C, N, self.patch_area).transpose(1, 3)
        patches = x.reshape(B * self.patch_area, N, C)
        return patches, {'orig': (H, W), 'B': B, 'nH': nH, 'nW': nW,
                         'N': N, 'interp': interpolate}

    def _fold(self, patches: torch.Tensor, info: Dict) -> torch.Tensor:
        B, nH, nW = info['B'], info['nH'], info['nW']
        C = patches.shape[-1]
        x = patches.reshape(B, self.patch_area, info['N'], C).transpose(1, 3)
        x = x.reshape(B * C * nH, nW, self.patch_h, self.patch_w).transpose(1, 2)
        x = x.reshape(B, C, nH * self.patch_h, nW * self.patch_w)
        if info['interp']:
            x = F.interpolate(x, size=info['orig'], mode='bilinear', align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        local_feat = self.local_rep(x)              # (B, transformer_dim, H, W)
        patches, info = self._unfold(local_feat)
        patches = self.global_rep(patches)
        global_feat = self._fold(patches, info)      # (B, transformer_dim, H, W)
        global_feat = self.conv_proj(global_feat)    # (B, in_channels, H, W)
        # MobileViTv3 fusion: concat(local, global)
        out = self.fusion(torch.cat([local_feat, global_feat], dim=1))
        return out + res                             # residual skip
