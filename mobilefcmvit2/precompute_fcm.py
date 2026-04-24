#!/usr/bin/env python3
"""
precompute_fcm.py
Generates FCM lesion-probability maps for all images in the BUSI split and
saves them as grayscale PNGs into mobilefcmvit2/fcm/.

Usage (from repo root):
    python3 mobilefcmvit2/precompute_fcm.py

The output directory mirrors the flat structure expected by ImageFolderWithFCM:
    mobilefcmvit2/fcm/<image_basename>.png
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def fuzzy_cmeans(pixels, n_clusters=3, m=2.0, max_iter=100, eps=1e-4):
    """Minimal FCM implementation — no external skfuzzy dependency required."""
    n = len(pixels)
    rng = np.random.default_rng(42)
    # Random initialisation of membership matrix
    U = rng.dirichlet(np.ones(n_clusters), size=n).T  # (C, N)

    for _ in range(max_iter):
        Um = U ** m
        # Cluster centres
        centers = (Um @ pixels) / Um.sum(axis=1, keepdims=True)  # (C, 1)
        # Distances
        dist = np.abs(pixels[None, :, 0] - centers[:, 0:1])  # (C, N)
        dist = np.maximum(dist, 1e-10)
        # Update membership
        inv = (1.0 / dist) ** (2.0 / (m - 1))
        U_new = inv / inv.sum(axis=0, keepdims=True)
        if np.max(np.abs(U_new - U)) < eps:
            U = U_new
            break
        U = U_new

    return U, centers.flatten()


def compute_fcm_map(image_path: Path, n_clusters: int = 3) -> np.ndarray:
    """Return a (H, W) uint8 lesion-probability map in [0, 255]."""
    img = Image.open(image_path).convert('L')
    arr = np.array(img, dtype=np.float32)
    H, W = arr.shape
    pixels = arr.flatten()[:, None]  # (N, 1)

    U, centers = fuzzy_cmeans(pixels, n_clusters=n_clusters)
    # Lesion cluster = highest-intensity centroid
    lesion_idx = int(np.argmax(centers))
    membership = U[lesion_idx].reshape(H, W)  # (H, W) in [0, 1]
    return (membership * 255).clip(0, 255).astype(np.uint8)


def find_images(root: Path):
    exts = {'.png', '.jpg', '.jpeg'}
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() in exts:
            yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=str(REPO_ROOT / 'datasets' / 'BUSI_split'),
                        help='Root of the BUSI split (contains train/ and val/ subfolders)')
    parser.add_argument('--out_dir', type=str,
                        default=str(REPO_ROOT / 'mobilefcmvit2' / 'fcm'),
                        help='Output directory for FCM PNG maps')
    parser.add_argument('--n_clusters', type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(find_images(data_dir))
    if not images:
        print(f"No images found under {data_dir}. Check --data_dir.")
        sys.exit(1)

    print(f"Found {len(images)} images. Writing FCM maps to {out_dir} ...")
    skipped = 0
    for i, img_path in enumerate(images):
        out_path = out_dir / (img_path.stem + '.png')
        if out_path.exists():
            skipped += 1
            continue
        try:
            fcm_map = compute_fcm_map(img_path, n_clusters=args.n_clusters)
            Image.fromarray(fcm_map).save(out_path)
        except Exception as e:
            print(f"  [WARN] Failed {img_path.name}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(images)} done ...")

    print(f"Done. {len(images) - skipped} maps written, {skipped} already existed.")


if __name__ == '__main__':
    main()
