"""
mobilefcmvitv3/models/mobilefcmvitv3_net.py

MobileFCMViTv3 — MobileViTv3-S backbone with fully differentiable
feature-space soft clustering.

Top-level module names match the official cvnets checkpoint exactly so that
pretrained weights load without any key remapping:
  conv_1        ← checkpoint: conv_1.block.*
  layer_1..5    ← checkpoint: layer_1..5.*
  conv_1x1_exp  ← checkpoint: conv_1x1_exp.block.*

New modules (randomly initialised, not in checkpoint):
  fcm_proj      — project 640-ch features before clustering
  clustering    — SoftClusteringLayer (learnable centers)
  fusion        — attention or concat fusion of membership maps
  classifier    — classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pathlib import Path
from timm.models.registry import register_model

from .mobilevitv3_block import ConvLayer, InvertedResidual, MobileViTv3Block
from .soft_clustering import SoftClusteringLayer


# ── MobileViTv3-S config (mode='small') ──────────────────────────────────────
# Matches cvnets config/mobilevit.py mode='small' exactly.
_S_CONFIG = {
    'layer_1': {'out': 32,  'blocks': 1, 'stride': 1, 'expand': 4},
    'layer_2': {'out': 64,  'blocks': 3, 'stride': 2, 'expand': 4},
    'layer_3': {'out': 96,  'blocks': 1, 'stride': 2, 'expand': 4,
                'tr_dim': 144, 'ffn_dim': 288, 'tr_blocks': 2, 'head_dim': 16,
                'patch_h': 2, 'patch_w': 2},
    'layer_4': {'out': 128, 'blocks': 1, 'stride': 2, 'expand': 4,
                'tr_dim': 192, 'ffn_dim': 384, 'tr_blocks': 4, 'head_dim': 32,
                'patch_h': 2, 'patch_w': 2},
    'layer_5': {'out': 160, 'blocks': 1, 'stride': 2, 'expand': 4,
                'tr_dim': 240, 'ffn_dim': 480, 'tr_blocks': 3, 'head_dim': 16,
                'patch_h': 2, 'patch_w': 2},
    'exp_factor': 4,   # 160 * 4 = 640
}


# ── Fusion modules ─────────────────────────────────────────────────────────────

class ConcatFusion(nn.Module):
    """(B, C+K, H, W) → (B, C, H, W) via Conv1x1."""
    def __init__(self, feat_ch: int, n_clusters: int):
        super().__init__()
        self.conv = ConvLayer(feat_ch + n_clusters, feat_ch,
                              kernel_size=1, padding=0)

    def forward(self, features: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([features, membership], dim=1))


class AttentionFusion(nn.Module):
    """
    Project membership (B, K, H, W) → per-channel attention, apply:
        enhanced = features * (1 + sigmoid(proj(membership)))
    Zero-initialised so training starts from the pretrained backbone.
    """
    def __init__(self, feat_ch: int, n_clusters: int):
        super().__init__()
        self.proj = nn.Conv2d(n_clusters, feat_ch, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, features: torch.Tensor, membership: torch.Tensor) -> torch.Tensor:
        return features * (1.0 + torch.sigmoid(self.proj(membership)))


# ── Main model ─────────────────────────────────────────────────────────────────

class MobileFCMViTv3(nn.Module):
    """
    MobileViTv3-S with fully differentiable feature-space soft clustering.

    Backbone module names match the official cvnets checkpoint so that
    pretrained weights load directly via load_pretrained().

    Args:
        num_classes:    output classes (3 for BUSI)
        in_chans:       input channels (3 — standard RGB)
        fcm_k:          number of cluster prototypes (K)
        fcm_proj_dim:   project 640-ch encoder output before clustering
        tau:            softmax temperature
        fcm_m:          fuzziness exponent for FCM-style membership
        membership:     'softmax' | 'fcm'
        normalize_feat: L2-normalise features before clustering
        fusion_type:    'attention' | 'concat'
        dropout:        classifier dropout
        attn_dropout:   transformer attention dropout
        ffn_dropout:    transformer FFN dropout
        cfg:            backbone channel config (defaults to MobileViTv3-S)
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_chans: int = 3,
        fcm_k: int = 3,
        fcm_proj_dim: int = 32,
        tau: float = 1.0,
        fcm_m: float = 2.0,
        membership: str = 'softmax',
        normalize_feat: bool = True,
        fusion_type: str = 'attention',
        drop_path_rate: float = 0.1,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        cfg: dict = None,
    ):
        super().__init__()
        cfg = cfg or _S_CONFIG

        # ── Backbone (names match cvnets checkpoint) ──────────────────────────
        self.conv_1 = ConvLayer(in_chans, 16, kernel_size=3, stride=2, padding=1)

        self.layer_1 = self._make_mv2_stage(16,  cfg['layer_1'])
        self.layer_2 = self._make_mv2_stage(32,  cfg['layer_2'])
        self.layer_3 = self._make_mit_stage(64,  cfg['layer_3'],
                                            dropout, attn_dropout, ffn_dropout)
        self.layer_4 = self._make_mit_stage(96,  cfg['layer_4'],
                                            dropout, attn_dropout, ffn_dropout)
        self.layer_5 = self._make_mit_stage(128, cfg['layer_5'],
                                            dropout, attn_dropout, ffn_dropout)

        exp_ch = min(cfg['exp_factor'] * 160, 960)   # 640
        self.conv_1x1_exp = ConvLayer(160, exp_ch, kernel_size=1, padding=0)

        # ── FCM projection ────────────────────────────────────────────────────
        clust_ch = fcm_proj_dim if fcm_proj_dim > 0 else exp_ch
        self.fcm_proj = (nn.Conv2d(exp_ch, fcm_proj_dim, kernel_size=1, bias=False)
                         if fcm_proj_dim > 0 else nn.Identity())

        # ── Differentiable soft clustering ────────────────────────────────────
        self.clustering = SoftClusteringLayer(
            in_channels=clust_ch,
            n_clusters=fcm_k,
            tau=tau,
            m=fcm_m,
            membership=membership,
            normalize=normalize_feat,
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        if fusion_type == 'attention':
            self.fcm_fusion = AttentionFusion(exp_ch, fcm_k)
        elif fusion_type == 'concat':
            self.fcm_fusion = ConcatFusion(exp_ch, fcm_k)
        else:
            raise ValueError(f"fusion_type must be 'attention' or 'concat', got '{fusion_type}'")

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(exp_ch, num_classes),
        )

        self._init_weights()

    # ── Stage builders ────────────────────────────────────────────────────────

    @staticmethod
    def _make_mv2_stage(in_ch: int, cfg: dict) -> nn.Sequential:
        layers, out_ch = [], cfg['out']
        for i in range(cfg['blocks']):
            stride = cfg['stride'] if i == 0 else 1
            layers.append(InvertedResidual(in_ch, out_ch, stride=stride,
                                           expand_ratio=cfg['expand']))
            in_ch = out_ch
        return nn.Sequential(*layers)

    @staticmethod
    def _make_mit_stage(in_ch: int, cfg: dict,
                        dropout, attn_dropout, ffn_dropout) -> nn.Sequential:
        out_ch = cfg['out']
        return nn.Sequential(
            InvertedResidual(in_ch, out_ch, stride=cfg['stride'],
                             expand_ratio=cfg['expand']),
            MobileViTv3Block(
                in_channels=out_ch,
                transformer_dim=cfg['tr_dim'],
                ffn_dim=cfg['ffn_dim'],
                n_transformer_blocks=cfg['tr_blocks'],
                head_dim=cfg['head_dim'],
                patch_h=cfg['patch_h'],
                patch_w=cfg['patch_w'],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
            ),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            logits:           (B, num_classes)
            compactness_loss: scalar
            entropy_loss:     scalar
        """
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        features = self.conv_1x1_exp(x)                    # (B, 640, h, w)

        proj = self.fcm_proj(features)                     # (B, proj_dim, h, w)
        membership, c_loss, e_loss = self.clustering(proj) # (B, K, h, w)

        if membership.shape[-2:] != features.shape[-2:]:
            membership = F.interpolate(membership, size=features.shape[-2:],
                                       mode='bilinear', align_corners=False)

        fused = self.fcm_fusion(features, membership)      # (B, 640, h, w)
        logits = self.classifier(fused)
        return logits, c_loss, e_loss

    # ── Pretrained weight loading ─────────────────────────────────────────────

    def load_pretrained(self, ckpt_path: str):
        """
        Load MobileViTv3-S pretrained weights from the official checkpoint.

        Backbone modules (conv_1, layer_1..5, conv_1x1_exp) are named to
        match the checkpoint keys exactly. FCM projection, clustering, fusion,
        and classifier are randomly initialised.

        Handles both:
          - Classification checkpoints (flat keys: conv_1.block.conv.weight)
          - Segmentation checkpoints   (encoder. prefix stripped automatically)
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_file():
            print(f"[WARN] Checkpoint not found: {ckpt_path}")
            return

        raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = raw.get('state_dict', raw.get('model', raw))

        # Strip encoder. prefix (segmentation checkpoints) and module. prefix
        state = {k.replace('encoder.', '').replace('module.', ''): v
                 for k, v in state.items()}

        current = self.state_dict()
        new_state = OrderedDict(
            (k, v) for k, v in state.items()
            if k in current and v.shape == current[k].shape
        )
        missing = set(current.keys()) - set(new_state.keys())
        self.load_state_dict(new_state, strict=False)

        backbone_keys = [k for k in new_state if not k.startswith(
            ('fcm_proj', 'clustering', 'fcm_fusion', 'classifier'))]
        print(f"Loaded {len(new_state)}/{len(current)} params "
              f"({len(backbone_keys)} backbone, "
              f"{len(missing)} randomly initialised)")


@register_model
def mobilefcmvitv3_s(pretrained=False, **kwargs):
    model = MobileFCMViTv3(**kwargs)
    return model
