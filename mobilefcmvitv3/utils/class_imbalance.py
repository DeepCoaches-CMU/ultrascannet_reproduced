"""
mobilefcmvitv3/utils/class_imbalance.py

Utilities for handling BUSI class imbalance.

BUSI class distribution (full dataset):
  benign:    437  (56.0%)
  malignant: 210  (26.9%)
  normal:    133  (17.1%)

Inverse-frequency weights: w_c = N / (C * n_c)
  benign:    780 / (3 * 437) = 0.595
  malignant: 780 / (3 * 210) = 1.238
  normal:    780 / (3 * 133) = 1.955
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

# BUSI inverse-frequency class weights (benign=0, malignant=1, normal=2)
BUSI_CLASS_WEIGHTS = torch.tensor([0.595, 1.238, 1.955])


class WeightedSoftTargetCrossEntropy(nn.Module):
    """
    Cross-entropy loss for soft (mixup) targets with per-class weights.
    Compatible with timm's Mixup which produces soft label tensors.
    """

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer('weight', weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        weighted = -(target * log_probs * self.weight.unsqueeze(0))
        return weighted.sum(dim=-1).mean()


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Focuses training on hard, misclassified examples.
    Supports per-class weights and label smoothing.
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer('weight', weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(x, target, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss_fn(smoothing: float = 0.1,
                  mixup_active: bool = False,
                  use_class_weights: bool = True,
                  num_classes: int = 3,
                  focal_gamma: float = 0.0,
                  device: str = 'cuda') -> nn.Module:
    """
    Build the appropriate loss function based on training configuration.

    Args:
        smoothing:      label smoothing epsilon
        mixup_active:   whether mixup/cutmix is active (soft targets)
        use_class_weights: apply BUSI inverse-frequency class weights
        num_classes:    number of output classes
        focal_gamma:    focal loss gamma (0 = standard cross-entropy)
        device:         device string
    """
    weights = BUSI_CLASS_WEIGHTS.to(device) if (use_class_weights and num_classes == 3) else None

    if mixup_active:
        if weights is not None:
            loss_fn = WeightedSoftTargetCrossEntropy(weight=weights)
        else:
            loss_fn = SoftTargetCrossEntropy()
    elif focal_gamma > 0:
        loss_fn = FocalLoss(gamma=focal_gamma, weight=weights, label_smoothing=smoothing)
    elif smoothing > 0:
        if weights is not None:
            loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)
        else:
            loss_fn = LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weights) if weights is not None \
            else nn.CrossEntropyLoss()

    if weights is not None:
        print(f"Using class-weighted loss: {weights.tolist()}")

    return loss_fn.to(device)
