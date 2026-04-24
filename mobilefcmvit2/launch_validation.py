#!/usr/bin/env python3
"""
launch_validation.py
Validates trained models on BUSI. All paths are read from paths.yaml at the repo root.

Expects checkpoints at:
    <weights_dir>/<model>_<dataset>/model_best.pth.tar
    <weights_dir>/<model>_<dataset>/args.yaml
"""

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
PATHS_YAML = REPO_ROOT / "paths.yaml"


def resolve_repo_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def load_paths():
    with open(PATHS_YAML) as f:
        paths = yaml.safe_load(f)

    for key in ("datasets_root", "busi_split_dir", "weights_dir", "mobilefcmvit2_checkpoint", "output_dir"):
        if key in paths:
            paths[key] = resolve_repo_path(paths[key])
    return paths


def main():
    paths = load_paths()
    busi_dir = paths["busi_split_dir"]
    output_dir = Path(paths["output_dir"])
    weights_dir = Path(paths["weights_dir"])

    model_list = [
        "mobilefcmvit2_net",
        # "mamba_vision_T2_baseline",
        # "resnet50",
        # "mobilenetv2_100",
        # "densenet121",
        # "vit_small_patch16_224",
        # "efficientnet_b0",
        # "convnext_tiny",
        # "swin_tiny_patch4_window7_224",
        # "deit_tiny_patch16_224",
        # "maxvit_tiny_rw_224",
    ]

    datasets = [
        {"name": "BUSI", "path": busi_dir},
    ]

    for model in model_list:
        for ds in datasets:
            experiment_name = f"{model}_{ds['name']}"
            checkpoint_roots = []
            for root in (output_dir, weights_dir):
                if root not in checkpoint_roots:
                    checkpoint_roots.append(root)

            ckpt_dir = None
            args_path = None
            checkpoint_path = None
            for root in checkpoint_roots:
                candidate_dir = root / experiment_name
                candidate_args = candidate_dir / "args.yaml"
                candidate_checkpoint = candidate_dir / "model_best.pth.tar"
                if candidate_args.exists() and candidate_checkpoint.exists():
                    ckpt_dir = candidate_dir
                    args_path = candidate_args
                    checkpoint_path = candidate_checkpoint
                    break

            if checkpoint_path is None or args_path is None:
                searched = ", ".join(str(root / experiment_name) for root in checkpoint_roots)
                print(f"\n⚠️  Checkpoint bundle not found for {experiment_name}. Searched: {searched}")
                continue

            cmd = [
                sys.executable, "val_simple.py",
                "-c", str(args_path),
                "--loadcheckpoint", str(checkpoint_path),
                "--data_dir", ds["path"],
                "--metrics-json", str(ckpt_dir / "evaluation_metrics.json"),
            ]
            print(f"\n✅ Validating: {experiment_name}\n")
            subprocess.run(cmd, cwd=Path(__file__).parent, check=True)


if __name__ == "__main__":
    main()
