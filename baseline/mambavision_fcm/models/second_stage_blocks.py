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

from .blocks import Downsample, ConvBlock, ConvNeXtBlock, SimpleMambaBlock, ResMambaBlock, \
    GatedConvBlock, LocalGlobalBlock, MobileViTBlock, SEBlock


    

class ConvBlocks(nn.Module):

    def __init__(self,
                 dim,
                 depth=2,
                 drop_path=0.0,
                 layer_scale=None,
                 downsample=True,
                 **kwargs
    ):
        super().__init__()
        
        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            dp_rates = drop_path[:depth] if len(drop_path) >= depth else drop_path + [0.0] * (depth - len(drop_path))
        else:
            dp_rates = [drop_path] * depth
            
        self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                               drop_path=dp_rates[i],
                                               layer_scale=layer_scale)
                                    for i in range(depth)])
        
        if downsample:
            self.downsample = Downsample(dim=dim)
        else:
            self.downsample = nn.Identity()
    
    def forward(self, x):
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return self.downsample(x)

    

class ConvNeXtStage1(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.0, layer_scale=1e-6, downsample=True, **kwargs):
        super().__init__()
        
        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            dp_rates = drop_path[:depth] if len(drop_path) >= depth else drop_path + [0.0] * (depth - len(drop_path))
        else:
            dp_rates = [drop_path] * depth
            
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=dim,
                drop_path=dp_rates[i],
                layer_scale=layer_scale
            )
            for i in range(depth)
        ])

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dim * 2, eps=1e-5),
                nn.GELU()
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x
    

    


class ResMambaStage1(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.0, layer_scale=None, downsample=True, **kwargs):
        super().__init__()
        
        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            dp_rates = drop_path[:depth] if len(drop_path) >= depth else drop_path + [0.0] * (depth - len(drop_path))
        else:
            dp_rates = [drop_path] * depth
            
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(ResMambaBlock(dim, drop_path=dp_rates[i], layer_scale=layer_scale))

        if downsample:
            self.downsample = Downsample(dim)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x
    
    

class MambaHybridLayer(nn.Module):
    def __init__(self, dim, depth=3, drop_path=0.0, layer_scale=1e-6, downsample=True, **kwargs):
        super().__init__()
        
        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            # Ensure we have at least 3 values for the 3 blocks
            dp_rates = drop_path[:3] if len(drop_path) >= 3 else drop_path + [0.0] * (3 - len(drop_path))
        else:
            dp_rates = [drop_path] * 3
        
        # Fixed structure with 3 specific blocks
        self.blocks = nn.ModuleList([
            ConvBlock(dim=dim,
                     drop_path=dp_rates[0],
                     layer_scale=layer_scale),
            ConvBlock(dim=dim,
                     drop_path=dp_rates[1],
                     layer_scale=layer_scale),
            MobileViTBlock(dim=dim,
                          drop_path=dp_rates[2],
                          layer_scale=layer_scale, 
                          num_layers=2, 
                          use_spatial_gating=True),
        ])
        
        if downsample:
            self.downsample = Downsample(dim=dim)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x


# Additional stage classes for more flexibility

class MambaStage1(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.0, layer_scale=None, downsample=True, **kwargs):
        super().__init__()
        
        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            dp_rate = drop_path[0] if drop_path else 0.0
        else:
            dp_rate = drop_path
            
        self.blocks = nn.ModuleList([
            SimpleMambaBlock(dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        
        if downsample:
            self.downsample = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] → [B, N, C]

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = x.transpose(1, 2).view(B, C, H, W)  # [B, N, C] → [B, C, H, W]
        x = self.downsample(x)
        return x


class SEConvStage1(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.0, layer_scale=None, downsample=True, **kwargs):
        super().__init__()
        
        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            dp_rates = drop_path[:depth] if len(drop_path) >= depth else drop_path + [0.0] * (depth - len(drop_path))
        else:
            dp_rates = [drop_path] * depth
            
        self.blocks = nn.ModuleList([
            ConvBlock(dim=dim, drop_path=dp_rates[i], layer_scale=layer_scale)
            for i in range(depth)
        ])
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        
        if downsample:
            self.downsample = Downsample(dim=dim)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = x * self.se(x)  # Apply SE attention
        x = self.downsample(x)
        return x


class GatedConvStage1(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.0, layer_scale=None, downsample=True, **kwargs):
        super().__init__()
        
        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            dp_rates = drop_path[:depth] if len(drop_path) >= depth else drop_path + [0.0] * (depth - len(drop_path))
        else:
            dp_rates = [drop_path] * depth
            
        self.blocks = nn.ModuleList([
            GatedConvBlock(dim=dim, drop_path=dp_rates[i], layer_scale=layer_scale)
            for i in range(depth)
        ])
        
        if downsample:
            self.downsample = Downsample(dim=dim)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x


class LocalGlobalStage1(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.0, layer_scale=None, downsample=True, **kwargs):
        super().__init__()
        
        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            dp_rates = drop_path[:depth] if len(drop_path) >= depth else drop_path + [0.0] * (depth - len(drop_path))
        else:
            dp_rates = [drop_path] * depth
            
        self.blocks = nn.ModuleList([
            LocalGlobalBlock(dim=dim, drop_path=dp_rates[i], layer_scale=layer_scale)
            for i in range(depth)
        ])
        
        if downsample:
            self.downsample = Downsample(dim=dim)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x


second_stage_block_registry = {
    'default': ConvBlocks,
    'convnext': ConvNeXtStage1,
    'resmamba': ResMambaStage1,
    'hybrid': MambaHybridLayer,
    'mamba': MambaStage1,
    'se_conv': SEConvStage1,
    'gated_conv': GatedConvStage1,
    'local_global': LocalGlobalStage1,
}