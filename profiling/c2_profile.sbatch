#!/bin/bash
# C2: profile MPK decoding of Qwen3-1.7B on one A100.
#   A. one profiled iteration per tokens-per-iteration T (MPK profiling builds
#      stop after one iteration) -> per-task CSV via profiling/mpk_profiled_run.py
#      (needs the c2-profiler-fix branch of ~/mirage)
#   B. unprofiled decode timing per batch size R with MPK's run_batch_perf.py
# Submit from the repo root:  mkdir -p logs && sbatch profiling/c2_profile.sbatch
# Analyze:                   python profiling/analyze_c2.py results/c2-<jobid>
#SBATCH --job-name=c2_mpk_profile
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu-preempt-highvram,gpu-highvram
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
MIRAGE="${MIRAGE:-$HOME/mirage}"
DEMO="$MIRAGE/demo/qwen3/demo.py"
BATCH_PERF="$MIRAGE/tests/ci-tests/run_batch_perf.py"
WRAPPER="$REPO/profiling/mpk_profiled_run.py"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
PROFILE_TOKENS="${PROFILE_TOKENS-1 8 16}"  # set to "" to skip part A
BATCHES="${BATCHES-1 2 4 8}"              # set to "" to skip part B
SEQ_SHORT="${SEQ_SHORT:-256}"
SEQ_LONG="${SEQ_LONG:-512}"
OUT="$REPO/results/c2-${SLURM_JOB_ID:-local}"

echo "host: $(hostname)  date: $(date)"
module load cuda/12.8 conda/latest
source "$HOME/.cargo/env"
conda activate mirage
module list 2>&1
which python && python --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "mirage: $(git -C "$MIRAGE" branch --show-current) $(git -C "$MIRAGE" rev-parse --short HEAD)"
git -C "$MIRAGE" diff --quiet HEAD~0 || echo "WARNING: uncommitted changes in $MIRAGE"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-1}"
# Compute nodes may have no internet: use the model already in the HF cache.
export HF_HUB_OFFLINE=1

mkdir -p "$OUT"
cat > "$OUT/config.txt" <<EOF
model=$MODEL
profile_tokens=$PROFILE_TOKENS
batches=$BATCHES
seq_short=$SEQ_SHORT
seq_long=$SEQ_LONG
mirage_commit=$(git -C "$MIRAGE" rev-parse HEAD)
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)
EOF

run_in() {  # $1 = run dir, rest = command; runs in its own dir (MPK writes to CWD)
    local dir="$1"; shift
    mkdir -p "$dir"
    echo; echo "===== $(basename "$dir")"
    ( cd "$dir" && /usr/bin/time -v "$@" ) > "$dir/stdout.txt" 2> "$dir/stderr.txt" \
        || { echo "FAILED (see $dir/stderr.txt)"; grep -v "^\s" "$dir/stderr.txt" | tail -n 8; return 0; }
    grep -E "^\[c2\]|per-token latency|total time|throughput" "$dir/stdout.txt" || true
}

# ---- A. profiled single iteration -----------------------------------------
for T in $PROFILE_TOKENS; do
    d="$OUT/profile_T$T"
    run_in "$d" python "$WRAPPER" "$d/profile_T$T" "$DEMO" --use-mirage \
        --model "$MODEL" --output-dir "$d/build" --profiling \
        --max-num-batched-tokens "$T" --max-num-batched-requests 1 \
        --max-seq-length "$SEQ_SHORT"
done

# ---- B. unprofiled decode timing (1-token prompt, --ignore-eos) -----------
for R in $BATCHES; do
    T=$(( R > 8 ? R : 8 ))
    for L in "$SEQ_SHORT" "$SEQ_LONG"; do
        d="$OUT/timing_R${R}_L$L"
        run_in "$d" python "$BATCH_PERF" --model "$MODEL" --output-dir "$d/build" \
            --max-num-batched-tokens "$T" --max-num-batched-requests "$R" \
            --max-seq-length "$L" --ignore-eos
    done
done

echo; echo "date: $(date)"
echo "Results in $OUT"
echo "Next: python profiling/analyze_c2.py $OUT"
