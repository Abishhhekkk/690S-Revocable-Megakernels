# C2: MPK profiling

Qwen3-1.7B on one A100, MPK with `mpk/patches/0002`–`0003`. Raw tables are in [data/](data/).

## Key numbers

| | value |
|---|---|
| decode step, batch 1 / 8 | 3.75 / 3.89 ms (SXM4-80GB); 4.1 ms (PCIe-40GB) |
| longest task (`lm_head`, 96 tasks) | ~450 µs, at the end of each step |
| all other tasks | < 50 µs (7,996 of 8,092) |
| tasks per step | ~8,100, plus 1,837 scheduler events |
| worker utilization, batch 1 | ~52% |

So a task-boundary yield should wait under 50 µs typically and about 450 µs at worst, compared with up to one full step (~4 ms) when waiting for the iteration boundary.

## How MPK runs on A100

- Two persistent kernels: 96 worker blocks (one per SM) and 12 scheduler blocks with 48 schedulers.
- In offline mode one `mpk()` launch runs prefill and all decode steps. Batch changes only happen inside the kernel, in `prepare_next_batch()`.
- Prefill is chunked into the same steps (`--max-num-batched-tokens`).
- Profiling builds stop after one step.

## Task durations (one step, batch 1)

| task | count | mean µs | max µs | share of time |
|---|---:|---:|---:|---:|
| LINEAR | 4,576 | 35 | 452 (`lm_head`) | 81% |
| LINEAR_WITH_RESIDUAL | 1,792 | 18 | 29 | 17% |
| PAGED_ATTENTION | 224 | 10 | 11 | 1% |
| everything else | | ≤ 7 | | < 1% |

Attention grows with tokens per step: 10 µs at T=1, 45 µs at T=8, 86 µs at T=16. Linear tasks barely change.

## Issues found

1. **Profiler broken with split kernels.** Worker and scheduler kernels overwrite each other's region. Fixed in patch 0003; decode with [mpk_profiled_run.py](mpk_profiled_run.py).
2. **Batch > 1 crashed on A100.** A paged-attention param layout mismatch made the KV page stride garbage. Fixed in patch 0002.
3. **MPK's A100 output doesn't match Hugging Face**, even at batch 1, and at batch > 1 the requests disagree with each other. This is an upstream bug and still open. MPK's output is deterministic, so use MPK's uninterrupted run as the reference ([check_batch_outputs.py](check_batch_outputs.py)).

## Reproduce

```bash
sbatch profiling/c2_profile.sbatch                 # profile + batch timing (~15 min)
python profiling/analyze_c2.py results/c2-<jobid>
sbatch profiling/check_batch_outputs.sbatch        # output check vs Hugging Face
```
