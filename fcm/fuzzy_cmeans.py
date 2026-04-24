"""
fcm/fuzzy_cmeans.py

Pixel-space Fuzzy C-Means for precomputing lesion-probability map PNGs.
Used only by fcm/precompute.py — not used inside the model.

For the differentiable in-model clustering, see:
  mobilefcmvitv3/models/soft_clustering.py
"""

import numpy as np
from pathlib import Path
from PIL import Image


def fuzzy_cmeans(pixels: np.ndarray, n_clusters: int = 3,
                 m: float = 2.0, max_iter: int = 100,
                 eps: float = 1e-4, seed: int = 42) -> tuple:
    """
    Fuzzy C-Means clustering.

    Args:
        pixels:     (N, 1) float32 array of pixel intensities
        n_clusters: number of clusters
        m:          fuzziness exponent (>1)
        max_iter:   maximum iterations
        eps:        convergence threshold
        seed:       random seed

    Returns:
        U:       (C, N) membership matrix
        centers: (C,)  cluster centroids
    """
    N = len(pixels)
    rng = np.random.default_rng(seed)
    U = rng.dirichlet(np.ones(n_clusters), size=N).T  # (C, N)

    for _ in range(max_iter):
        Um = U ** m
        centers = (Um @ pixels) / Um.sum(axis=1, keepdims=True)  # (C, 1)
        dist = np.abs(pixels[None, :, 0] - centers[:, 0:1])      # (C, N)
        dist = np.maximum(dist, 1e-10)
        inv = (1.0 / dist) ** (2.0 / (m - 1))
        U_new = inv / inv.sum(axis=0, keepdims=True)
        if np.max(np.abs(U_new - U)) < eps:
            U = U_new
            break
        U = U_new

    return U, centers.flatten()


# ── Pixel-space FCM (used by precompute.py) ───────────────────────────────────

def compute_fcm_map(image_path, n_clusters: int = 3) -> np.ndarray:
    """
    Compute a lesion-probability map for a single image.

    The lesion cluster is identified as the highest-intensity centroid,
    consistent with ultrasound imaging where lesions appear brighter.

    Args:
        image_path: path to image file (str or Path)
        n_clusters: number of FCM clusters

    Returns:
        (H, W) uint8 array in [0, 255]
    """
    img = Image.open(image_path).convert('L')
    arr = np.array(img, dtype=np.float32)
    H, W = arr.shape
    pixels = arr.flatten()[:, None]  # (N, 1)

    U, centers = fuzzy_cmeans(pixels, n_clusters=n_clusters)
    lesion_idx = int(np.argmax(centers))
    membership = U[lesion_idx].reshape(H, W)
    return (membership * 255).clip(0, 255).astype(np.uint8)
