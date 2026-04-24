"""
mobilefcmvitv3/models/soft_clustering.py

Fully differentiable soft clustering layer for feature-space deep clustering.

Replaces the classical (non-differentiable) Fuzzy C-Means with a learnable
module whose cluster centers are nn.Parameters updated by gradient descent.

Two membership modes:
  'softmax'  — U = softmax(-dist / tau, dim=-1)
               Simple, fast, well-behaved gradients.
  'fcm'      — U_ij = 1 / Σ_k (d_ij / d_ik)^(2/(m-1))
               Approximates classical FCM membership; numerically stabilised.

Two auxiliary losses (both optional via lambda weights):
  compactness  — mean(U * dist²)   : pull features toward their centers
  entropy      — -mean(U * log U)  : encourage confident, non-uniform assignments

Total loss passed back to the caller:
  loss = task_loss + λ1 * compactness_loss + λ2 * entropy_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftClusteringLayer(nn.Module):
    """
    Differentiable soft clustering operating on spatial feature maps.

    Args:
        in_channels:    C — feature channels entering the layer
        n_clusters:     K — number of cluster prototypes
        tau:            temperature for softmax membership (lower = sharper)
        m:              fuzziness exponent for FCM-style membership (>1)
        membership:     'softmax' | 'fcm'
        normalize:      L2-normalise features before distance computation
    """

    def __init__(
        self,
        in_channels: int,
        n_clusters: int = 3,
        tau: float = 1.0,
        m: float = 2.0,
        membership: str = 'softmax',
        normalize: bool = True,
    ):
        super().__init__()
        assert membership in ('softmax', 'fcm'), \
            f"membership must be 'softmax' or 'fcm', got '{membership}'"

        self.K = n_clusters
        self.tau = tau
        self.m = m
        self.membership_mode = membership
        self.normalize = normalize

        # Learnable cluster centers — shape (K, C)
        # Initialised with small random values; updated via backprop.
        self.centers = nn.Parameter(torch.randn(n_clusters, in_channels) * 0.02)

    # ── Membership computation ────────────────────────────────────────────────

    def _softmax_membership(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: (N, K) squared Euclidean distances
        Returns:
            U:    (N, K) soft memberships summing to 1 along K
        """
        return F.softmax(-dist / self.tau, dim=-1)

    def _fcm_membership(self, dist: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        FCM-style membership:
            U_ij = 1 / Σ_k ( (d_ij / d_ik)^(2/(m-1)) )

        Numerically stabilised: distances clamped away from zero.

        Args:
            dist: (N, K) squared Euclidean distances
        Returns:
            U:    (N, K) soft memberships summing to 1 along K
        """
        dist = dist.clamp(min=eps)                          # (N, K)
        exp = 2.0 / (self.m - 1)
        # ratio[i, j, k] = (d_ij / d_ik)^exp
        ratio = (dist.unsqueeze(2) / dist.unsqueeze(1)).pow(exp)  # (N, K, K)
        U = 1.0 / ratio.sum(dim=-1).clamp(min=eps)         # (N, K)
        return U

    # ── Losses ────────────────────────────────────────────────────────────────

    @staticmethod
    def compactness_loss(U: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        """
        Encourage features to be close to their assigned centers.
            L = mean(U * dist²)

        Args:
            U:    (N, K) memberships
            dist: (N, K) squared distances
        Returns:
            scalar
        """
        return (U * dist).mean()

    @staticmethod
    def entropy_loss(U: torch.Tensor) -> torch.Tensor:
        """
        Entropy regularisation — encourages confident, non-uniform assignments.
        Maximising entropy (negative sign) prevents cluster collapse.
            L = -mean(U * log(U + ε))

        Args:
            U: (N, K) memberships
        Returns:
            scalar
        """
        return -(U * torch.log(U + 1e-8)).mean()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: (B, C, H, W)

        Returns:
            membership:       (B, K, H, W)  soft cluster maps in (0, 1)
            compactness_loss: scalar
            entropy_loss:     scalar
        """
        B, C, H, W = features.shape
        N = B * H * W

        # Flatten to (N, C)
        x = features.permute(0, 2, 3, 1).reshape(N, C)

        # Optional L2 normalisation — stabilises distance geometry
        if self.normalize:
            x = F.normalize(x, dim=-1)
            centers = F.normalize(self.centers, dim=-1)
        else:
            centers = self.centers

        # Squared Euclidean distances: (N, K)
        # torch.cdist returns Euclidean; square for FCM convention
        dist = torch.cdist(x, centers).pow(2)   # (N, K)

        # Membership
        if self.membership_mode == 'softmax':
            U = self._softmax_membership(dist)   # (N, K)
        else:
            U = self._fcm_membership(dist)       # (N, K)

        # Auxiliary losses
        c_loss = self.compactness_loss(U, dist)
        e_loss = self.entropy_loss(U)

        # Reshape membership back to spatial: (B, K, H, W)
        membership = U.reshape(B, H, W, self.K).permute(0, 3, 1, 2)

        return membership, c_loss, e_loss
