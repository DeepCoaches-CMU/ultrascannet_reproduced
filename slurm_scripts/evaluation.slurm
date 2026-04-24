#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/home/jgibson2/projects/mobilefcmvit2/slurm_logs/%x_%j.out
#SBATCH --error=/home/jgibson2/projects/mobilefcmvit2/slurm_logs/%x_%j.err
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
	candidates+=("$PWD" "$PWD/.." "/home/jgibson2/projects/mobilefcmvit2")

	for candidate in "${candidates[@]}"; do
		if [[ -f "$candidate/paths.yaml" && -f "$candidate/mobilefcmvit2/launch_validation.py" ]]; then
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

echo "==== MobileFCMVit2 Evaluation ===="
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

cd "$PROJECT_DIR"
if [[ -f "$PROJECT_DIR/venv/bin/activate" ]]; then
	source "$PROJECT_DIR/venv/bin/activate"
else
	echo "Virtual environment not found at $PROJECT_DIR/venv/bin/activate" >&2
	exit 1
fi

mkdir -p $PROJECT_DIR/slurm_logs

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Running mobilefcmvit2/launch_validation.py using paths.yaml"
PYTHONPATH="$PROJECT_DIR" python mobilefcmvit2/launch_validation.py

echo "Job finished at $(date)"
