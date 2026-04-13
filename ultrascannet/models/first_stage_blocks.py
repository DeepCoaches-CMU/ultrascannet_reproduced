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
                 kernel_size=3,
                 **kwargs):
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

        if isinstance(drop_path, list):
            dp_rate = drop_path[0] if drop_path else 0.0
        else:
            dp_rate = drop_path
            
        self.drop_path = DropPath(dp_rate) if dp_rate > 0. else nn.Identity()

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
    def __init__(self, dim, drop_path=0., layer_scale=None, kernel_size=7, **kwargs):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim, eps=1e-5)
        self.pwconv = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(dim * 4, dim, kernel_size=1)
        )
        self.layer_scale = layer_scale
        if isinstance(layer_scale, (int, float)):
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = residual + self.drop_path(x)
        return x
    

class SimpleMambaBlock(nn.Module):
    def __init__(self, dim, **kwargs):
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
    

class MambaStage(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.0, layer_scale=None, **kwargs):
        super().__init__()
        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            drop_path = drop_path[0] if drop_path else 0.0
        
        self.blocks = nn.Sequential(*[
            SimpleMambaBlock(dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.downsample = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)     # [B, C, H, W] → [B, N, C]

        x = self.blocks(x)
        x = self.norm(x)

        x = x.transpose(1, 2).view(B, C, H, W)  # [B, N, C] → [B, C, H, W]
        x = self.downsample(x)

        return x
    

class MambaHybridStage0(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.0, layer_scale=None, **kwargs):
        super().__init__()
        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            drop_path = drop_path[0] if drop_path else 0.0
        
        self.dim = dim

        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.mamba_blocks = nn.Sequential(*[
            SimpleMambaBlock(dim) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

        self.downsample = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.local_conv(x)  # [B, C, H, W]

        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.mamba_blocks(x)
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]

        x = self.downsample(x)
        return x
    

class ConvNeXtStage0(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.0, layer_scale=None, **kwargs):
        super().__init__()

        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            dp_rates = drop_path[:depth] if len(drop_path) >= depth else drop_path + [0.0] * (depth - len(drop_path))
        else:
            dp_rates = [drop_path for _ in range(depth)]

        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(
                dim=dim,
                drop_path=dp_rates[i],
                layer_scale=layer_scale
            )
            for i in range(depth)
        ])

        self.downsample = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim * 2, eps=1e-5),
            nn.GELU()
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.downsample(x)
        return x
    

class Convx2NeXtStage0(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.0, layer_scale=None, **kwargs):
        super().__init__()

        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            dp_rate = drop_path[0] if drop_path else 0.0
        else:
            dp_rate = drop_path

        self.block1 = ConvBlock(dim, drop_path=dp_rate, layer_scale=layer_scale)
        self.block2 = ConvNeXtBlock(dim, drop_path=dp_rate, layer_scale=layer_scale)

        # Baseline-style downsampling
        self.downsample = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.downsample(x)
        return x
    

class ResidualConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale=None, kernel_size=3, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)

        self.layer_scale = layer_scale
        if isinstance(layer_scale, (int, float)):
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)

        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)

        x = residual + self.drop_path(x)
        return x
    
    
    
class SEConv(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.0, layer_scale=None, **kwargs):
        super().__init__()

        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            dp = drop_path[:depth] if len(drop_path) >= depth else drop_path + [0.0] * (depth - len(drop_path))
        else:
            dp = [drop_path] * depth

        self.blocks = nn.Sequential(*[
            ResidualConvBlock(
                dim=dim,
                drop_path=dp[i],
                layer_scale=layer_scale
            )
            for i in range(depth)
        ])
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        self.downsample = Downsample(dim=dim)

    def forward(self, x):
        x = self.blocks(x)
        x = x * self.se(x)  # Fixed: apply SE attention
        x = self.downsample(x)
        return x


class CoordConvStage0(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.0, layer_scale=None, **kwargs):
        super().__init__()

        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            dp_rate = drop_path[0] if drop_path else 0.0
        else:
            dp_rate = drop_path

        # One-time projection to match model dim
        self.input_proj = nn.Conv2d(dim + 2, dim, kernel_size=1)

        self.blocks = nn.Sequential(*[
            ConvBlock(dim, drop_path=dp_rate, layer_scale=layer_scale) for _ in range(depth)
        ])

        self.downsample = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)

    def add_coords(self, x):
        B, C, H, W = x.shape
        xx = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        yy = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x = torch.cat([x, xx, yy], dim=1) 
        return x

    def forward(self, x):
        x = self.add_coords(x)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.downsample(x)
        return x
    


class ConvMixerBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, drop_path=0.0, **kwargs):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.pwconv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        self.norm = nn.BatchNorm2d(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.dwconv(x))
        x = self.norm(x)
        x = self.pwconv(x)
        return x
    
    
class ConvMixerStage0(nn.Module):
    def __init__(self, dim, depth=2, kernel_size=7, drop_path=0.0, layer_scale=None, **kwargs):
        super().__init__()

        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            dp_rate = drop_path[0] if drop_path else 0.0
        else:
            dp_rate = drop_path

        self.blocks = nn.Sequential(*[
            ConvMixerBlock(dim, kernel_size=kernel_size, drop_path=dp_rate)
            for _ in range(depth)
        ])
        self.downsample = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.GELU()
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.downsample(x)
        return x


class ConvBlockWithPosEncStage0(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.0, layer_scale=None, input_size=56, **kwargs):
        super().__init__()

        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            dp_rate = drop_path[0] if drop_path else 0.0
        else:
            dp_rate = drop_path

        # Positional embedding: (1, dim, H, W)
        self.grid_size = input_size
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, self.grid_size, self.grid_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(*[
            ConvBlock(dim=dim, drop_path=dp_rate, layer_scale=layer_scale)
            for _ in range(depth)
        ])

        self.downsample = Downsample(dim=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.pos_embed[:, :, :H, :W]
        x = self.blocks(x)
        x = self.downsample(x)
        return x
    


class ConvBlockLN(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale=None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.ln1 = nn.LayerNorm(dim, eps=1e-5)

        self.act1 = nn.GELU(approximate="tanh")

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.ln2 = nn.LayerNorm(dim, eps=1e-5)

        self.layer_scale = layer_scale
        if isinstance(layer_scale, (int, float)):
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x

        B, C, H, W = x.shape
        x = self.conv1(x)
        x = self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LayerNorm over C

        x = self.act1(x)

        x = self.conv2(x)
        x = self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)

        x = residual + self.drop_path(x)
        return x


class ConvBlockWithLayerNormStage0(nn.Module):
    def __init__(self, dim, depth=2, drop_path=0.0, layer_scale=1e-6, input_size=56, **kwargs):
        super().__init__()
        self.grid_size = input_size
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, self.grid_size, self.grid_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Handle drop_path as list or single value
        if isinstance(drop_path, list):
            dp_rates = drop_path[:depth] if len(drop_path) >= depth else drop_path + [0.0] * (depth - len(drop_path))
        else:
            dp_rates = [drop_path] * depth

        self.blocks = nn.Sequential(*[
            ConvBlockLN(dim, drop_path=dp_rates[i], layer_scale=layer_scale)
            for i in range(depth)
        ])

        self.downsample = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.pos_embed[:, :, :H, :W]
        x = self.blocks(x)
        x = self.downsample(x)
        return x
    


first_stage_block_registry = {
    'mamba_simple': MambaStage,
    'mamba_hybrid': MambaHybridStage0,
    'convnext': ConvNeXtStage0,
    'convblock_convnext': Convx2NeXtStage0,
    'se_conv': SEConv,
    'coordconv': CoordConvStage0,
    'convmixer': ConvMixerStage0,
    'convblock_posenc': ConvBlockWithPosEncStage0,
    'convblock_ln_posenc': ConvBlockWithLayerNormStage0,
    'default': ConvBlock
}