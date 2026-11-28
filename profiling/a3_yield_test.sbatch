#!/bin/bash
# A3 validation on A100: MPK yield/resume mid-decode, outputs must match an
# uninterrupted run. Uses the A3 MPK tree (default ~/mirage-a3).
# Submit from the repo root:  mkdir -p logs && sbatch profiling/a3_yield_test.sbatch
#SBATCH --job-name=a3_yield_test
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
MPK_A3="${MPK_A3:-$HOME/mirage-a3}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
SEQ="${SEQ:-512}"
OUT="$REPO/results/a3-${SLURM_JOB_ID:-local}"
TEST="$REPO/profiling/a3_yield_test.py"
PERF="$MPK_A3/tests/ci-tests/run_batch_perf.py"

echo "host: $(hostname)  date: $(date)"
module load cuda/12.8 conda/latest
source "$HOME/.cargo/env"
conda activate mirage
export PYTHONPATH="$MPK_A3/python"
module list 2>&1
which python && python --version
python -c "import mirage; print('mirage from', mirage.__file__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "mpk-a3 base: $(git -C "$MPK_A3" rev-parse --short HEAD) (+ uncommitted A3 patch)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HUB_OFFLINE=1
mkdir -p "$OUT"

one() {  # name R yields gap_ms pause_ms mode(task|drain) [--random-gap]
    local name=$1 R=$2 Y=$3 G=$4 P=$5 M=$6 X=${7:-} d="$OUT/$1"
    local T=$(( R > 8 ? R : 8 ))
    mkdir -p "$d"
    echo; echo "===== $name (R=$R yields=$Y gap=${G}ms pause=${P}ms mode=$M $X)"
    ( cd "$d" && MPK_YIELD_MODE=$M python "$TEST" run "$d/$name" --yields "$Y" --gap-ms "$G" \
        --pause-ms "$P" $X -- \
        "$PERF" --model "$MODEL" --output-dir "$d/build" --max-num-batched-tokens "$T" \
        --max-num-batched-requests "$R" --max-seq-length "$SEQ" --ignore-eos ) \
        > "$d/stdout.txt" 2> "$d/stderr.txt" \
        || { echo "FAILED (see $d/stderr.txt)"; grep -v '^\s' "$d/stderr.txt" | tail -n 8; return 0; }
    grep -E "^MPK yield mode|total time|per-token latency" "$d/stdout.txt" || true
    grep -E "^\[a3\] \{" "$d/stdout.txt" | head -1 || true
}

# R=1: reference x2, then task-boundary vs drain yields (same random schedule)
one base1_R1     1 0  0   0  task
one base2_R1     1 0  0   0  task
one task5_R1     1 5  250 20 task
one task50_R1    1 50 20  2  task  --random-gap
one drain50_R1   1 50 20  2  drain --random-gap
# R=8
one base1_R8     8 0  0   0  task
one task5_R8     8 5  250 20 task
one task50_R8    8 50 20  2  task  --random-gap
one drain50_R8   8 50 20  2  drain --random-gap

echo
cmp() { local ref=$1; shift; local args=(); for s in "$@"; do [ -f "$OUT/$s/$s.json" ] && args+=("$OUT/$s/$s"); done
        [ -f "$OUT/$ref/$ref.json" ] && python "$TEST" compare "$OUT/$ref/$ref" "${args[@]}" || echo "compare $ref: missing runs"; }
cmp base1_R1 base2_R1 task5_R1 task50_R1 drain50_R1 || true
cmp base1_R8 task5_R8 task50_R8 drain50_R8 || true
echo; echo "date: $(date)"
