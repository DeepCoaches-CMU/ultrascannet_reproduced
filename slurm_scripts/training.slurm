#!/bin/bash
#SBATCH --job-name=training
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=92G
#SBATCH --time=4:00:00
#SBATCH --qos=normal
#SBATCH --output=/home/jgibson2/projects/UltraScanNet/slurm_logs/%x_%j.out
#SBATCH --error=/home/jgibson2/projects/UltraScanNet/slurm_logs/%x_%j.err
#SBATCH --signal=B:SIGTERM@120

set -euo pipefail

resolve_project_dir() {
	local candidates=()
	if [[ -n "${PROJECT_DIR:-}" ]]; then
		candidates+=("$PROJECT_DIR")
	fi
	if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
		candidates+=("$SLURM_SUBMIT_DIR" "$SLURM_SUBMIT_DIR/..")
	fi
	candidates+=("$PWD" "$PWD/.." "/home/jgibson2/projects/UltraScanNet")

	for candidate in "${candidates[@]}"; do
		if [[ -f "$candidate/paths.yaml" && -f "$candidate/ultrascannet/train.py" ]]; then
			cd "$candidate" >/dev/null 2>&1 && pwd
			return 0
		fi
	done

	return 1
}

PROJECT_DIR=$(resolve_project_dir) || {
	echo "Unable to resolve project root. Set PROJECT_DIR before submitting." >&2
	exit 1
}


read_path_from_yaml() {
	local key=$1
	python3 - "$PROJECT_DIR/paths.yaml" "$PROJECT_DIR" "$key" <<'PY'
from pathlib import Path
import sys
import yaml

yaml_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
key = sys.argv[3]
with open(yaml_path) as handle:
	data = yaml.safe_load(handle)
value = Path(data[key])
if not value.is_absolute():
	value = (repo_root / value).resolve()
print(value)
PY
}


echo "==== UltraScanNet Training ===="

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"


# ---- Load environment ----
cd "$PROJECT_DIR"
if [[ -f "$PROJECT_DIR/venv/bin/activate" ]]; then
	source "$PROJECT_DIR/venv/bin/activate"
else
	echo "Virtual environment not found at $PROJECT_DIR/venv/bin/activate" >&2
	echo "Set PROJECT_DIR correctly or create the venv before submitting." >&2
	exit 1
fi


SLURM_LOG_DIR=$PROJECT_DIR/slurm_logs
mkdir -p $SLURM_LOG_DIR


export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_PORT=${MASTER_PORT:-29500}

BUSI_DIR=$(read_path_from_yaml busi_split_dir)
OUTPUT_DIR=$(read_path_from_yaml output_dir)
WANDB_GROUP=$(python3 - "$PROJECT_DIR/paths.yaml" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as handle:
	data = yaml.safe_load(handle)
print(data["wandb_group"])
PY
)

CONFIG_PATH=${1:-$PROJECT_DIR/ultrascannet/configs/experiments/mambavision_tiny2_1k_run_exp.yaml}
MODEL_NAME=${2:-ultra_scan_net}
EXPERIMENT_NAME=${3:-${MODEL_NAME}_BUSI}
DATA_LEN=${4:-624}

mkdir -p "$OUTPUT_DIR"

# ---- Environment preflight ----
python3 - <<'PY'
import sys

issues = []

try:
	import torch
	cuda_available = torch.cuda.is_available()
	print(f"[preflight] torch={torch.__version__} cuda_build={torch.version.cuda} cuda_available={cuda_available}")
	if not cuda_available:
		issues.append(
			"CUDA is not available from this environment. "
			"Your PyTorch build and NVIDIA driver/runtime are likely incompatible."
		)
except Exception as exc:
	issues.append(f"Failed to import torch: {exc}")

try:
	import mamba_ssm  # noqa: F401
	print("[preflight] mamba_ssm import: OK")
except Exception as exc:
	issues.append(f"Failed to import mamba_ssm: {exc}")

if issues:
	print("[preflight] Environment check failed:")
	for issue in issues:
		print(f"  - {issue}")
	sys.exit(1)

print("[preflight] Environment check passed")
PY


# ---- Run training ----
echo "Config      : $CONFIG_PATH"
echo "Model       : $MODEL_NAME"
echo "Experiment  : $EXPERIMENT_NAME"
echo "Data dir    : $BUSI_DIR"
echo "Output dir  : $OUTPUT_DIR"
echo "WandB group : $WANDB_GROUP"

PYTHONPATH=$PROJECT_DIR torchrun \
	--standalone \
	--nnodes=1 \
	--nproc_per_node=${SLURM_GPUS_ON_NODE:-1} \
	ultrascannet/train.py \
	-c "$CONFIG_PATH" \
	--group="$WANDB_GROUP" \
	--model="$MODEL_NAME" \
	--experiment="$EXPERIMENT_NAME" \
	--data_dir="$BUSI_DIR" \
	--data_len="$DATA_LEN" \
	--output="$OUTPUT_DIR"

echo "Job finished at $(date)"
