#!/bin/bash
# Verify MPK batched decode gives correct tokens for every request (sm80 fix).
# Submit from the repo root:  mkdir -p logs && sbatch profiling/check_batch_outputs.sbatch
#SBATCH --job-name=c2_batch_check
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-preempt-highvram,gpu-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
MIRAGE="${MIRAGE:-$HOME/mirage}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
BATCHES="${BATCHES-2 8}"
SEQ="${SEQ:-256}"
OUT="$REPO/results/batch-check-${SLURM_JOB_ID:-local}"

echo "host: $(hostname)  date: $(date)"
module load cuda/12.8 conda/latest
source "$HOME/.cargo/env"
conda activate mirage
module list 2>&1
which python && python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "mirage: $(git -C "$MIRAGE" branch --show-current) $(git -C "$MIRAGE" rev-parse --short HEAD)"
git -C "$MIRAGE" diff --quiet || echo "WARNING: uncommitted changes in $MIRAGE"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HUB_OFFLINE=1

status=0
for R in $BATCHES; do
    T=$(( R > 8 ? R : 8 ))
    d="$OUT/R$R"; mkdir -p "$d"
    echo; echo "===== batch $R"
    ( cd "$d" && python "$REPO/profiling/check_batch_outputs.py" \
        "$MIRAGE/tests/ci-tests/run_batch_perf.py" --model "$MODEL" \
        --output-dir "$d/build" --max-num-batched-tokens "$T" \
        --max-num-batched-requests "$R" --max-seq-length "$SEQ" --ignore-eos ) \
        > "$d/stdout.txt" 2> "$d/stderr.txt" || { status=1; echo "FAILED (see $d/stderr.txt)"; tail -n 5 "$d/stderr.txt"; }
    grep -E "^\[check\]|^        (MPK|HF|text)" "$d/stdout.txt" || true
done
echo; echo "date: $(date)"
exit $status
