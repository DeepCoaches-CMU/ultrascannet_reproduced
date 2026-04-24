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


def build_loss_fn(smoothing: float = 0.1,
                  mixup_active: bool = False,
                  use_class_weights: bool = True,
                  num_classes: int = 3,
                  device: str = 'cuda') -> nn.Module:
    """
    Build the appropriate loss function based on training configuration.

    Args:
        smoothing:          label smoothing epsilon
        mixup_active:       whether mixup/cutmix is active (soft targets)
        use_class_weights:  apply BUSI inverse-frequency class weights
        num_classes:        number of output classes
        device:             device string

    Returns:
        nn.Module loss function
    """
    weights = BUSI_CLASS_WEIGHTS.to(device) if (use_class_weights and num_classes == 3) else None

    if mixup_active:
        if weights is not None:
            loss_fn = WeightedSoftTargetCrossEntropy(weight=weights)
        else:
            loss_fn = SoftTargetCrossEntropy()
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
