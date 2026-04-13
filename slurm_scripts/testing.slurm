#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/home/jgibson2/projects/UltraScanNet/slurm_logs/%x_%j.out
#SBATCH --error=/home/jgibson2/projects/UltraScanNet/slurm_logs/%x_%j.err
#SBATCH --signal=B:SIGTERM@120

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)


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


find_args_for_checkpoint() {
    local checkpoint_path=$1
    local checkpoint_dir
    checkpoint_dir=$(dirname "$checkpoint_path")
    echo "$checkpoint_dir/args.yaml"
}

echo "==== UltraScanNet Testing ===="
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

cd "$PROJECT_DIR"
source venv/bin/activate

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p $PROJECT_DIR/slurm_logs

DATA_DIR=${1:-$(read_path_from_yaml busi_split_dir)}
CHECKPOINT=${2:-$(read_path_from_yaml ultrascannet_checkpoint)}
ARGS_PATH=${3:-$(find_args_for_checkpoint "$CHECKPOINT")}
EXPERIMENT_NAME=${4:-$(basename "$(dirname "$CHECKPOINT")")}

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

if [[ ! -f "$ARGS_PATH" ]]; then
    echo "args.yaml not found: $ARGS_PATH" >&2
    exit 1
fi

echo "Data dir   : $DATA_DIR"
echo "Checkpoint : $CHECKPOINT"
echo "Args file  : $ARGS_PATH"
echo "Experiment : $EXPERIMENT_NAME"

PYTHONPATH=$PROJECT_DIR python3 ultrascannet/val_simple.py \
    -c "$ARGS_PATH" \
    --loadcheckpoint "$CHECKPOINT" \
    --data_dir "$DATA_DIR" \
    --experiment "$EXPERIMENT_NAME" \
    --validate_only

echo "Job finished at $(date)"
