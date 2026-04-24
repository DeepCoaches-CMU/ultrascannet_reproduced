#!/usr/bin/env python3
"""
launch_experiments.py
Trains MobileFCMVit2 on BUSI. All paths are read from paths.yaml at the repo root.
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
    output_dir = paths["output_dir"]
    wandb_group = paths["wandb_group"]

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
        {"name": "BUSI", "path": busi_dir, "data_len": 624},
        # {"name": "BUSBRA", "path": "<path>", "data_len": 1499},
    ]

    for model in model_list:
        for ds in datasets:
            experiment_name = f"{model}_{ds['name']}"
            cmd = [
                sys.executable, "train.py",
                "-c", "./configs/experiments/mambavision_tiny2_1k_run_exp.yaml",
                f"--group={wandb_group}",
                f"--model={model}",
                f"--experiment={experiment_name}",
                f"--data_dir={ds['path']}",
                f"--data_len={ds['data_len']}",
                f"--output={output_dir}",
                "--log-wandb",
            ]
            print(f"\n🚀 Running: {experiment_name}\n")
            subprocess.run(cmd, cwd=Path(__file__).parent, check=True)


if __name__ == "__main__":
    main()
