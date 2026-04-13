#!/usr/bin/env python3
"""
setup_dataset.py
Downloads the BUSI dataset from Kaggle and organizes it into the
train/validation split defined in data/BUSI_split.json.

Usage:
    python3 setup_dataset.py

Requirements:
    - Kaggle API credentials at ~/.kaggle/kaggle.json
      Get them at: https://www.kaggle.com/settings → API → Create New Token
    - pip install kaggle
"""

import json
import os
import shutil
import zipfile
from pathlib import Path

import yaml

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent
PATHS_YAML = REPO_ROOT / "paths.yaml"
SPLIT_JSON = REPO_ROOT / "data" / "BUSI_split.json"

# Kaggle dataset slug (Dataset for Breast Ultrasound Images)
KAGGLE_DATASET = "aryashah2k/breast-ultrasound-images-dataset"

# Class folder names in the raw Kaggle download
# The Kaggle zip contains: Dataset_BUSI_with_GT/{benign,malignant,normal}/
KAGGLE_SUBDIR = "Dataset_BUSI_with_GT"
CLASSES = ["benign", "malignant", "normal"]
# ──────────────────────────────────────────────────────────────────────────────


def load_paths():
    with open(PATHS_YAML) as f:
        paths = yaml.safe_load(f)

    for key in ("datasets_root", "busi_split_dir", "weights_dir", "ultrascannet_checkpoint", "output_dir"):
        if key in paths:
            path = Path(paths[key])
            if not path.is_absolute():
                paths[key] = str((REPO_ROOT / path).resolve())
    return paths


def download_busi(raw_dir: Path):
    """Download and unzip BUSI from Kaggle into raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "busi.zip"

    if (raw_dir / KAGGLE_SUBDIR).exists():
        print(f"✅ Raw dataset already present at {raw_dir / KAGGLE_SUBDIR}")
        return

    print(f"⬇️  Downloading {KAGGLE_DATASET} from Kaggle …")
    ret = os.system(
        f"kaggle datasets download -d {KAGGLE_DATASET} -p {raw_dir} --unzip"
    )
    if ret != 0:
        # Fallback: download zip then unzip manually
        ret = os.system(
            f"kaggle datasets download -d {KAGGLE_DATASET} -p {raw_dir}"
        )
        if ret != 0:
            raise RuntimeError(
                "Kaggle download failed.\n"
                "Make sure ~/.kaggle/kaggle.json exists with valid credentials.\n"
                "Get them at: https://www.kaggle.com/settings → API → Create New Token"
            )
        zips = list(raw_dir.glob("*.zip"))
        if not zips:
            raise RuntimeError("No zip file found after download.")
        with zipfile.ZipFile(zips[0], "r") as zf:
            zf.extractall(raw_dir)
        zips[0].unlink()

    print(f"✅ Download complete → {raw_dir}")


def organize_split(raw_dir: Path, split_dir: Path, split_json: Path):
    """
    Copy images from raw_dir into split_dir/{train,validation}/{class}/
    according to the split JSON.

    The split JSON uses 'val' as the key; timm's create_dataset expects
    the folder to be named 'validation'.
    """
    with open(split_json) as f:
        split = json.load(f)

    src_root = raw_dir / KAGGLE_SUBDIR

    # Verify source exists
    if not src_root.exists():
        # Some Kaggle downloads don't nest under KAGGLE_SUBDIR
        candidates = [d for d in raw_dir.iterdir() if d.is_dir()]
        if len(candidates) == 1:
            src_root = candidates[0]
            print(f"ℹ️  Using source root: {src_root}")
        else:
            raise FileNotFoundError(
                f"Expected {src_root} but found: {candidates}"
            )

    split_key_map = {"train": "train", "val": "validation"}
    total_copied = 0
    total_skipped = 0

    for json_split, folder_name in split_key_map.items():
        for cls in CLASSES:
            dst = split_dir / folder_name / cls
            dst.mkdir(parents=True, exist_ok=True)

            filenames = split.get(json_split, {}).get(cls, [])
            for fname in filenames:
                src = src_root / cls / fname
                dst_file = dst / fname
                if dst_file.exists():
                    total_skipped += 1
                    continue
                if not src.exists():
                    # BUSI images sometimes have mask siblings; skip masks
                    print(f"  ⚠️  Missing: {src}")
                    continue
                shutil.copy2(src, dst_file)
                total_copied += 1

    print(f"✅ Organized split → {split_dir}")
    print(f"   Copied: {total_copied}  |  Already present: {total_skipped}")


def verify_split(split_dir: Path, split_json: Path):
    """Print a count summary and warn about any missing files."""
    with open(split_json) as f:
        split = json.load(f)

    split_key_map = {"train": "train", "val": "validation"}
    all_ok = True
    for json_split, folder_name in split_key_map.items():
        for cls in CLASSES:
            expected = split.get(json_split, {}).get(cls, [])
            dst = split_dir / folder_name / cls
            found = len(list(dst.glob("*.png"))) if dst.exists() else 0
            status = "✅" if found == len(expected) else "⚠️ "
            if found != len(expected):
                all_ok = False
            print(f"  {status} {folder_name}/{cls}: {found}/{len(expected)}")

    if all_ok:
        print("\n✅ All files present — dataset ready.")
    else:
        print("\n⚠️  Some files are missing. Check warnings above.")


def main():
    paths = load_paths()
    datasets_root = Path(paths["datasets_root"])
    split_dir = Path(paths["busi_split_dir"])
    raw_dir = datasets_root / "BUSI_raw"

    print("=" * 60)
    print("UltraScanNet — BUSI Dataset Setup")
    print("=" * 60)

    download_busi(raw_dir)
    organize_split(raw_dir, split_dir, SPLIT_JSON)

    print("\nVerifying split …")
    verify_split(split_dir, SPLIT_JSON)


if __name__ == "__main__":
    main()
