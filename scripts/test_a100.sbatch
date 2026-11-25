#!/bin/bash
# How to test: build the toy megakernel on an A100, run it, and verify the
# task log with the independent checker (tools/checker.py).
#
# From the repo root:
#   mkdir -p logs && sbatch scripts/test_a100.sbatch
#
# Then read logs/toy_mk_test-<jobid>.out — the last line is PASS or FAIL.
# The checker needs a Python with numpy; override the default with:
#   PY_ENV=/path/to/conda/env sbatch scripts/test_a100.sbatch
#
#SBATCH --job-name=toy_mk_test
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-preempt-highvram,gpu-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
PY_ENV="${PY_ENV:-/work/pi_arunaj_umass_edu/abhijit/envs/mpk}"
PYTHON="$PY_ENV/bin/python"
BUILD="$REPO/build/a100"
RUN="$BUILD/run-${SLURM_JOB_ID:-local}"

echo "host: $(hostname)  date: $(date)  commit: $(git -C "$REPO" rev-parse --short HEAD)"
module load cuda/12.6
module list 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" --version

# 1. Checker unit tests (synthetic logs, no GPU needed)
(cd "$REPO/tools" && "$PYTHON" test_checker.py)

# 2. Build with per-task logging on
cmake -S "$REPO" -B "$BUILD" -DCMAKE_CUDA_ARCHITECTURES=80 -DENABLE_TASK_LOG=ON
cmake --build "$BUILD" -j "${SLURM_CPUS_PER_TASK:-1}"

# 3. Run the toy megakernel (writes task_log.bin + task_graph.json to CWD)
mkdir -p "$RUN"
cd "$RUN"
"$BUILD/toy_megakernel"

# 4. Verify the run with the independent checker
"$PYTHON" -I "$REPO/tools/checker.py" task_log.bin task_graph.json
