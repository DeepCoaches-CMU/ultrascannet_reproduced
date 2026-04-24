#!/bin/bash
#SBATCH --job-name=setup_venv
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --qos=normal
#SBATCH --output=/home/jgibson2/projects/ultrascannet_reproduced/slurm_logs/setup_venv_%j.out
#SBATCH --error=/home/jgibson2/projects/ultrascannet_reproduced/slurm_logs/setup_venv_%j.err

set -euo pipefail

VENV=/home/jgibson2/projects/ultrascannet_reproduced/venv
PROJECT_DIR=/home/jgibson2/projects/ultrascannet_reproduced

echo "==== MobileFCMVit2 venv setup ===="
echo "Job started on $(hostname) at $(date)"
echo "CUDA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"

# Detect CUDA version available on this node
CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
echo "CUDA major version: $CUDA_VER"

# Create fresh venv
python3 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip wheel setuptools

# Install PyTorch matching the node's CUDA — use cu128 which works with driver 525+
# cu128 is compatible with CUDA 12.x drivers (12.0–12.9)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Verify CUDA is available
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, device: {torch.cuda.get_device_name(0)}')"

# Install mamba_ssm (requires CUDA to be available for compilation)
pip install mamba_ssm --no-build-isolation

# Install remaining requirements
pip install -r "$PROJECT_DIR/requirements.txt"
pip install torchinfo wandb einops timm

echo "==== venv setup complete ===="
echo "Activate with: source $VENV/bin/activate"
echo "Job finished at $(date)"
