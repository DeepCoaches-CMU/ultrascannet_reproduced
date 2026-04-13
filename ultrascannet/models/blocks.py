import torch.serialization
import argparse
import torch
import torch.nn as nn
from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg
try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from .registry import register_pip_model
from pathlib import Path



class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x
    


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale=1e-6, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm2d(dim, eps=1e-5)
        self.pwconv = nn.Sequential(
            nn.Conv2d(dim, 4 * dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4 * dim, dim, kernel_size=1),
        )
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim)) if layer_scale else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x
        return shortcut + self.drop_path(x)


class SimpleMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return x + self.ffn(self.norm(x))


class ResMambaBlock(nn.Module):
    def __init__(self, dim, drop_path, layer_scale):
        super().__init__()
        self.conv = ConvBlock(dim, drop_path, layer_scale)
        self.mamba = SimpleMambaBlock(dim)

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x_flat = self.mamba(x_flat)
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        return x
    



class GatedConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale=None, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim * 2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.norm1 = nn.BatchNorm2d(dim * 2, eps=1e-5)

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.layer_scale = layer_scale
        if layer_scale is not None and isinstance(layer_scale, (int, float)):
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.norm1(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * torch.sigmoid(x2)

        x = self.conv2(x)
        x = self.norm2(x)

        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = shortcut + self.drop_path(x)
        return x

    
class ResMambaBlockGated(nn.Module):
    def __init__(self, dim, drop_path, layer_scale):
        super().__init__()
        self.conv = GatedConvBlock(dim, drop_path, layer_scale)
        self.mamba = SimpleMambaBlock(dim)

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x_flat = self.mamba(x_flat)
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        return x
    

class LocalGlobalBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale=None):
        super().__init__()
        self.conv_local = ConvBlock(dim, drop_path=drop_path)
        
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        
        self.conv_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        x_local = self.conv_local(x)

        # Prep for MHSA
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_attn, _ = self.attn(self.attn_norm(x_flat), x_flat, x_flat)
        x_attn = x_attn.reshape(B, H, W, C).permute(0, 3, 1, 2)

        x_out = x_local + self.drop_path(self.conv_proj(x_attn))
        return x_out
    

class MobileViTBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale=None, num_layers=1, use_spatial_gating=False):
        super().__init__()
        self.local_rep = ConvBlock(dim, drop_path=drop_path)
        
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=dim * 2, batch_first=True),
            num_layers=num_layers,
        )

        # Spatial Gating Unit (SGU)
        self.use_spatial_gating = use_spatial_gating
        self.sgu_gate = nn.Conv2d(dim, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.fuse = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

       

    def forward(self, x):
        B, C, H, W = x.shape
        x_local = self.local_rep(x)

        x_patch = self.proj(x_local)  # [B, C, H, W]
        x_patch = x_patch.flatten(2).transpose(1, 2)  # [B, HW, C]


        x_patch = self.transformer(self.norm(x_patch))
        x_patch = x_patch.transpose(1, 2).reshape(B, C, H, W)

        if self.use_spatial_gating:
            gate = self.sigmoid(self.sgu_gate(x_patch))
            x_patch = x_patch * gate

        return x + self.drop_path(self.fuse(x_patch))
    


class SEBlock(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.pool(x)  # [B, C, 1, 1]
        scale = self.fc(scale)
        return x * scale