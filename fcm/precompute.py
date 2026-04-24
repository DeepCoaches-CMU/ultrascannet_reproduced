"""
fcm/precompute.py

Precompute FCM maps for all images in the BUSI split and save as PNGs.

Usage:
    python fcm/precompute.py --data_dir datasets/BUSI_split --out_dir fcm/maps
"""

import sys
import argparse
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from fcm.fuzzy_cmeans import compute_fcm_map


def find_images(root: Path):
    exts = {'.png', '.jpg', '.jpeg'}
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() in exts:
            yield p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root of BUSI split (train/validation subdirs)')
    parser.add_argument('--out_dir', type=str, default='fcm/maps',
                        help='Output directory for FCM PNGs')
    parser.add_argument('--n_clusters', type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(find_images(data_dir))
    if not images:
        print(f"No images found under {data_dir}")
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
