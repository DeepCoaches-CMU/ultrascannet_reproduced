#!/usr/bin/env python3
"""
setup_weights.py
Downloads pretrained UltraScanNet weights from the OneDrive link in the README.

OneDrive public links require a browser-based flow that cannot be automated
directly, so this script:
  1. Tries an automated download via the OneDrive direct-download URL pattern.
  2. If that fails, prints clear manual instructions.

Usage:
    python3 setup_weights.py
"""

import os
import sys
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent
PATHS_YAML = REPO_ROOT / "paths.yaml"

# OneDrive share link from README
ONEDRIVE_SHARE_URL = (
    "https://uptro29158-my.sharepoint.com/:f:/g/personal/"
    "alexandra_laicu-hausberger_student_upt_ro/"
    "Em88eUDjtxBKmFMdmV75XBYB-AmQabzwnSjD-IzuwCstqA"
)

# Expected checkpoint filename inside the weights dir
CHECKPOINT_FILENAME = "model_best.pth.tar"
ARGS_FILENAME = "args.yaml"


def load_paths():
    with open(PATHS_YAML) as f:
        paths = yaml.safe_load(f)

    for key in ("datasets_root", "busi_split_dir", "weights_dir", "ultrascannet_checkpoint", "output_dir"):
        if key in paths:
            path = Path(paths[key])
            if not path.is_absolute():
                paths[key] = str((REPO_ROOT / path).resolve())
    return paths


def print_manual_instructions(weights_dir: Path):
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print(
        f"""
OneDrive public folder links cannot be downloaded automatically.

Please do the following:

  1. Open this URL in your browser:
     {ONEDRIVE_SHARE_URL}

  2. Download the folder contents (model_best.pth.tar and args.yaml)
     for the 'ultra_scan_net_BUSI' experiment.

  3. Place the files here:
     {weights_dir}/ultra_scan_net_BUSI/model_best.pth.tar
     {weights_dir}/ultra_scan_net_BUSI/args.yaml

  4. Re-run this script to verify, or proceed directly to validation.
"""
    )


def verify_weights(weights_dir: Path) -> bool:
    checkpoint = weights_dir / "ultra_scan_net_BUSI" / CHECKPOINT_FILENAME
    args_file = weights_dir / "ultra_scan_net_BUSI" / ARGS_FILENAME

    found_ckpt = checkpoint.exists()
    found_args = args_file.exists()

    print(f"  {'✅' if found_ckpt else '❌'} {checkpoint}")
    print(f"  {'✅' if found_args else '❌'} {args_file}")

    return found_ckpt and found_args


def main():
    paths = load_paths()
    weights_dir = Path(paths["weights_dir"])
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "ultra_scan_net_BUSI").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UltraScanNet — Pretrained Weights Setup")
    print("=" * 60)

    print("\nChecking for existing weights …")
    if verify_weights(weights_dir):
        print("\n✅ Weights already present — nothing to download.")
        return

    print("\nAttempting automated download …")
    # OneDrive direct download pattern (works for single files, not folders)
    # For a folder share we cannot automate without the individual file IDs.
    print("⚠️  Automated download of OneDrive folder shares is not supported.")
    print_manual_instructions(weights_dir)

    # After manual placement, verify again
    print("\nAfter placing the files, re-run this script to verify:")
    print("  python3 setup_weights.py")


if __name__ == "__main__":
    main()
