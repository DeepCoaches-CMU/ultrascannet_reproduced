#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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


class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x
    



class PatchEmbedInv(nn.Module):
    """
    Patch embedding block using inverted residual style.
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels (e.g., 3).
            in_dim: intermediate channel dimension (e.g., 64).
            dim: output embedding dimension (e.g., 96).
        """
        super().__init__()
        self.proj = nn.Identity()

        self.conv_down = nn.Sequential(
            # 1x1 pointwise conv (project to in_dim)
            nn.Conv2d(in_chans, in_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(inplace=True),

            # 3x3 depthwise conv with stride 2
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(inplace=True),

            # 1x1 pointwise conv to final dim
            nn.Conv2d(in_dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU(inplace=True),

            # Optional: another 3x3 depthwise to mimic original's 2-stage downsample
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=2):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)

        self.conv = nn.Sequential(
            # Pointwise
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),

            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),

            # Pointwise linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class HybridPatchEmbed(nn.Module):
    """
    Patch embedding block (hybrid - compatible with PatchEmbed and PatchEmbedInv).
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.proj = nn.Identity()

        self.conv_down = nn.Sequential(
            # Initial 1x1 conv
            nn.Conv2d(in_chans, in_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(inplace=True),

            # First IRB with stride 2
            InvertedResidual(in_dim, in_dim, stride=2, expand_ratio=2),
            nn.ReLU(inplace=True),

            # Project to final dim
            nn.Conv2d(in_dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU(inplace=True),

            # Optional: another IRB with stride 2
            InvertedResidual(dim, dim, stride=2, expand_ratio=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x
    

import torch.nn as nn

class PatchEmbedHybridConvNeXt(nn.Module):
    """
    Hybrid Patch Embedding block: CNN + ConvNeXt-style attention via depthwise conv.
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.proj = nn.Identity()

        self.conv_down = nn.Sequential(
            # Stage 1: Conv downsample
            nn.Conv2d(in_chans, in_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.GELU(),

            nn.Conv2d(in_dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.GELU(),

            # Stage 2: ConvNeXt-style context
            nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=True),  # DW conv
            nn.BatchNorm2d(dim, eps=1e-6),
            nn.GELU(),

            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


import torch.nn as nn

class PatchEmbedHybridDropout(nn.Module):
    """
    Hybrid Patch embedding block with dropout to improve generalization.
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96, dropout_rate=0.1):
        """
        Args:
            in_chans: Number of input channels (e.g., 3).
            in_dim: Intermediate channel dimension (e.g., 64).
            dim: Output embedding dimension (e.g., 96).
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__()
        self.proj = nn.Identity()

        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),  # 🔸 Dropout after first conv + BN + ReLU

            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),  # 🔸 Dropout after depthwise conv

            nn.Conv2d(in_dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),  # 🔸 Dropout after final pointwise conv

            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU(inplace=True)
            # Optional final dropout here, if needed
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x

class ShallowAttention(nn.Module):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B x (HW) x C
        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).reshape(B, C, H, W)  # Back to B x C x H x W
        return x


class PatchEmbedWithAttention(nn.Module):
    def __init__(self, in_chans=3, in_dim=64, dim=96, num_heads=2):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, in_dim, dim)
        self.attn = ShallowAttention(dim, num_heads=num_heads)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.attn(x)
        return x
    

class PatchEmbed1StageWithPosEmbed(nn.Module):
    """
    Patch embedding using a single Conv2d followed by a positional embedding.
    Compatible with (in_chans, in_dim, dim) constructor.
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96, img_size=224, patch_size=4):
        """
        Args:
            in_chans (int): Number of input channels.
            in_dim (int): Intermediate dim, kept for compatibility (not used here).
            dim (int): Output embedding dimension.
            img_size (int): Input image size.
            patch_size (int): Patch size (conv stride).
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

        # Calculate the number of patches
        num_patches = (img_size // patch_size) ** 2

        self.pos_embed = nn.Parameter(torch.zeros(1, dim, img_size // patch_size, img_size // patch_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.proj(x)              # (B, dim, H', W')
        x = x + self.pos_embed        # Add positional embedding
        return x
    

class PatchEmbedLearnedPos(nn.Module):
    """
    Patch embedding block with learned 2D positional embedding.
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96, img_size=224):
        """
        Args:
            in_chans: number of input channels (usually 3).
            in_dim: intermediate feature dimension after first conv.
            dim: final output embedding dimension.
            img_size: input image size (assumed square).
        """
        super().__init__()

        self.grid_size = img_size // 4  # assuming 2x2 stride (2 convs)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, dim, self.grid_size, self.grid_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.proj = nn.Identity()

        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        x = x + self.pos_embed
        return x
    

class PatchEmbedLearnedPosAttn(nn.Module):
    """
    Patch embedding with learned 2D positional embedding + shallow attention.
    """
    def __init__(self, in_chans=3, in_dim=64, dim=96, img_size=224, attn_heads=2):
        super().__init__()

        self.grid_size = img_size // 4  # assumes 2x2 downsampling
        self.pos_embed = nn.Parameter(
            torch.zeros(1, dim, self.grid_size, self.grid_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.proj = nn.Identity()

        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=attn_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        x = x + self.pos_embed  # [B, C, H, W]

        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x_attn, _ = self.attn(x_flat, x_flat, x_flat)
        x_attn = self.attn_norm(x_attn)
        x = x_attn.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]

        return x

class PatchEmbedLearnedPosConvNeXtAttn(nn.Module):
    """
    Patch embedding with learned 2D positional embedding and ConvNeXt-style shallow attention.
    """
    def __init__(self, in_chans=3, in_dim=64, dim=96, img_size=224):
        super().__init__()

        self.grid_size = img_size // 4  # assuming 2 convs with stride=2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, dim, self.grid_size, self.grid_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.proj = nn.Identity()

        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU(inplace=True)
        )

        # ConvNeXt-style shallow attention block
        self.attn_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),  # depthwise
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, kernel_size=1),  # pointwise
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        x = x + self.pos_embed
        x = x + self.attn_block(x)  # ConvNeXt-style refinement
        return x
    
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
        # x: [B, N, C]
        return x + self.ffn(self.norm(x))
    

class PatchEmbedLearnedPosMambaAttn(nn.Module):
    """
    Patch embedding with learned 2D positional embedding + shallow Mamba block(s).
    """
    def __init__(self, in_chans=3, in_dim=64, dim=96, img_size=224, mamba_blocks=2):
        super().__init__()
        self.grid_size = img_size // 4
        self.dim = dim
        self.num_tokens = self.grid_size * self.grid_size

        self.pos_embed = nn.Parameter(
            torch.zeros(1, dim, self.grid_size, self.grid_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU(inplace=True)
        )

        # Stack of shallow Mamba blocks
        self.mamba_blocks = nn.Sequential(*[
            SimpleMambaBlock(dim) for _ in range(mamba_blocks)
        ])

    def forward(self, x):
        x = self.conv_down(x)                      # [B, C, H, W]
        x = x + self.pos_embed                     # [B, C, H, W]
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)           # [B, N, C]
        x = self.mamba_blocks(x)                   # [B, N, C]
        x = x.transpose(1, 2).view(B, C, H, W)      # [B, C, H, W]

        return x
    


patch_embed_registry = {
    'inv': PatchEmbedInv,
    'hybrid': HybridPatchEmbed,
    'hybrid_convnext': PatchEmbedHybridConvNeXt,
    'hybrid_dropout': lambda **kwargs: PatchEmbedHybridDropout(**kwargs, dropout_rate=0.1),
    'shallow_attn': PatchEmbedWithAttention,
    'posemb_patch1stage': PatchEmbed1StageWithPosEmbed,
    'learned_pos': PatchEmbedLearnedPos,
    'learned_pos_attn': PatchEmbedLearnedPosAttn,
    'convnextattn': PatchEmbedLearnedPosConvNeXtAttn,
    'mamba_attn': PatchEmbedLearnedPosMambaAttn,
    'default': PatchEmbed

}