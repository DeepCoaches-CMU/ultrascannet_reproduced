#!/usr/bin/env python3
"""
launch_experiments_ablation.py
Ablation study over patch_embed / first_layer / second_layer variants.
All paths are read from paths.yaml at the repo root.
"""

import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent.parent
PATHS_YAML = REPO_ROOT / "paths.yaml"


def resolve_repo_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def load_paths():
    with open(PATHS_YAML) as f:
        paths = yaml.safe_load(f)

    for key in ("datasets_root", "busi_split_dir", "weights_dir", "mambavision_fcm_checkpoint", "output_dir"):
        if key in paths:
            paths[key] = resolve_repo_path(paths[key])
    return paths


def main():
    paths = load_paths()
    busi_dir = paths["busi_split_dir"]
    output_dir = paths["output_dir"]
    wandb_group = paths["wandb_group"]

    model = "mambavision_fcm_net"
    data_len = 624

    patch_embed_keys = [
        "learned_pos",
        # "inv", "hybrid", "hybrid_convnext", "hybrid_dropout",
        # "shallow_attn", "posemb_patch1stage", "learned_pos_attn",
        # "convnextattn", "mamba_attn", "default",
    ]

    first_stage_block_keys = [
        "convblock_posenc",
        # "mamba_simple", "mamba_hybrid", "convnext", "convblock_convnext",
        # "se_conv", "coordconv", "convmixer", "convblock_ln_posenc", "default",
    ]

    second_stage_block_keys = [
        "hybrid",
        # "default", "convnext", "resmamba", "mamba", "se_conv",
        # "gated_conv", "local_global",
    ]

    for patch_embed in patch_embed_keys:
        for first_layer in first_stage_block_keys:
            for second_layer in second_stage_block_keys:
                experiment_name = f"{patch_embed}_{first_layer}_{second_layer}"
                cmd = [
                    "python3", "train.py",
                    "-c", "./configs/experiments/mambavision_tiny2_1k_run_exp.yaml",
                    f"--group=ablation_{wandb_group}",
                    f"--model={model}",
                    f"--experiment={experiment_name}",
                    f"--data_dir={busi_dir}",
                    f"--data_len={data_len}",
                    f"--output={output_dir}",
                    f"--patch_embed={patch_embed}",
                    f"--first_layer={first_layer}",
                    f"--second_layer={second_layer}",
                    "--log-wandb",
                ]
                print(f"\n🚀 Running ablation: {experiment_name}\n")
                subprocess.run(cmd, cwd=Path(__file__).parent, check=True)


if __name__ == "__main__":
    main()
