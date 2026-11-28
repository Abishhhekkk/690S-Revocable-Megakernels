# C2: MPK execution and profiling report

Qwen3-1.7B on one A100-PCIE-40GB with MPK at commit `d2189e50` plus a profiler fix (`c2-profiler-fix` branch, see below). Collected on Nov 25, 2026.
Raw tables: [data/c2_summary_profile.md](data/c2_summary_profile.md), [data/c2_summary_timing.md](data/c2_summary_timing.md) (batch 1, PCIe 40 GB), [data/c2_summary_batch_timing.md](data/c2_summary_batch_timing.md) (batch 1–8, SXM4 80 GB, with the sm80 fix), [data/c2_task_types.csv](data/c2_task_types.csv).

> **Correctness caveat.** MPK's A100 output does not match Hugging Face greedy decoding even at batch 1 (it agrees for 6 tokens, then repeats `1. 1. 1.`), and at batch > 1 requests 1+ disagree with request 0 from the first token. Timing numbers are still meaningful (each step does the same work), but **use MPK's own uninterrupted output, not HF, as the reference** for yield/resume correctness (B1/B2/A4). See problem 4.

## Key numbers

| Quantity | Value | Why it matters |
|---|---|---|
| Decode iteration, batch 1 | **4.1–4.2 ms** (~240 tokens/s) | Worst-case wait for **iteration-boundary admission** (MPK as-is) |
| Longest task (`lm_head` linear) | **~450 µs** | Worst-case wait for a **task-boundary yield** |
| All other tasks | **< 50 µs** (7,996 of 8,092 tasks) | Typical wait for a task-boundary yield |
| Tasks per decode iteration | ~8,100 + 1,837 scheduler events | Size of the state a yield must preserve |
| Worker utilization in one iteration | ~52% at T=1 | Workers sit idle waiting on dependencies |
| Decode iteration vs batch (SXM4-80GB) | 3.75 / 3.75 / 3.79 / 3.89 ms at R = 1/2/4/8 | Batching to 8 costs ~4% per step for ~7.7× throughput (C4 background workload) |

So a task-boundary yield cuts the worst-case wait by about **9×** compared with waiting for the iteration to end (about 450 µs vs about 4.1 ms). It cuts the typical wait by about **80×** (under 50 µs vs about 4.1 ms). These are waits only; relaunch cost (A5) still has to be added.

## How MPK executes (offline mode, as used by the demo)

- **Two persistent kernels on the A100.** `worker_kernel` uses 96 blocks, one per SM, and `scheduler_kernel` uses 12 blocks with 4 scheduler warps each (48 schedulers). Both run concurrently. Released SMs for partial yield (B3) must come from the 96 worker SMs.
- **One launch runs the whole request.** A single `mpk()` call runs prefill and every decode step until EOS or `max_seq_length`. There is no host-visible boundary between iterations.
- **Iteration boundaries live inside the kernel.** `prepare_next_batch()` runs on the scheduler side between iterations. It finalizes tokens, retires finished requests and admits new ones. This is the only place MPK changes the batch.
- **Prefill is chunked into the same iterations.** Each iteration processes up to `--max-num-batched-tokens` prompt tokens. Measured spans of one prefill iteration: 4.0 ms (T=1), 5.1 ms (T=8), 6.5 ms (T=16). A 39-token prompt at T=8 is 5 iterations, about 26 ms of prefill before the first decode token.
- **Profiling builds run exactly one iteration.** Under `MPK_ENABLE_PROFILING`, `prepare_next_batch()` marks the request done after the first iteration. Task durations therefore come from one profiled iteration, and step latency comes from separate unprofiled runs.

## Task durations (one iteration, T=1)

| Task type | Count | Mean µs | p50 µs | Max µs | Share of task time |
|---|---:|---:|---:|---:|---:|
| TASK_LINEAR (all) | 4,576 | 35.2 | 33.8 | 451.6 | 81.4% |
|   of which `lm_head` | 96 | ~450 | | 451.6 | ~11% of the iteration |
| TASK_LINEAR_WITH_RESIDUAL | 1,792 | 18.4 | 25.6 | 28.7 | 16.7% |
| TASK_PAGED_ATTENTION_1 | 224 | 10.2 | 10.2 | 11.3 | 1.2% |
| TASK_SILU_MUL, RMS_NORM, ARGMAX, ... | | ≤ 7 | | | < 1% |

`lm_head` (2048 × 153,600, about 630 MB in bf16) is split into 96 tasks, one per worker. They all start at about 3.52 ms and finish at the end of the iteration. A yield request that arrives just after they start waits about 450 µs; any other time it waits under 50 µs.

Attention grows with tokens per iteration: 10 µs at T=1, 45 µs at T=8, 86 µs at T=16. Linear tasks barely change, because they are bound by reading weights.

## Implications for other parts

- **D3 (yield policy):** use about 4.1 ms as the cost of waiting for the iteration boundary. Use under 50 µs as the typical and about 450 µs as the worst-case cost of reaching a task boundary. Policy thresholds should be recalibrated with A5's relaunch cost.
- **B6 (task size):** `lm_head` is the obvious task to split. Splitting it 4× (384 tasks) should bring the worst case down to about 115 µs. Nothing else is above 50 µs at batch 1.
- **B3 (partial yield):** about half of worker time is idle at batch 1. Releasing k worker SMs may cost background throughput less than k/96. This should be measured.
- **D2 (memory):** these runs used the 40 GB A100. Note that the cluster also has 80 GB nodes, so pin the GPU type with `--constraint`.

## Problems found

1. **MPK profiler is broken when worker and scheduler kernels are separate (fixed locally).** Both kernels write the profiler header at `buffer[0]` and use overlapping slot layouts. The header ends up saying 12 blocks instead of 96, scheduler events interleave with worker events, and MPK's export crashes (`KeyError`). Fix: `~/mirage` branch `c2-profiler-fix` gives the scheduler kernel its own region (`MPK_PROFILER_SCHED_OFFSET`). [mpk_profiled_run.py](mpk_profiled_run.py) decodes both regions. The profiler macro also doesn't parenthesize its buffer argument, which is worth noting if anyone else patches it.
2. **Batch size > 1 crashed on A100 (fixed locally).** The Python side emits the 14-param paged-attention form, but the sm80 C++ parser reads a fixed 10-param form, so `params[9]` (the qk-norm eps bits of `1e-6f`, 897,988,541) became `PAGE_STRIDE_ROWS`. Request 0 sits on page 0, so `0 × garbage` hid it; any second request addressed far outside the KV cache (`cudaErrorIllegalAddress`). Fix: `~/mirage/python/mirage/mpk/persistent_kernel.py` sends the 10-param form when `target_cc == 80` (stride now 8,192 rows). Batches 2/4/8 run (job 2270526).
3. **Profiler buffer has no bounds check.** Each block writes into a fixed set of slots, and a run that is too long writes past the end of the buffer silently. That's fine for one-iteration profiling builds, but don't enlarge the profiled region without resizing the buffer.

4. **MPK's A100 decode output is wrong (open, upstream).** [check_batch_outputs.py](check_batch_outputs.py) (job 2272300) runs `run_batch_perf.py` with identical prompts and compares tokens. Request 0 diverges from HF greedy after 6 tokens; this was already true before the fix (job 2254718). At R = 2 and 8, requests 1+ get a different first token and then fluent but different text, which points at the first-iteration KV write/read for requests on pages > 0. Not debugged. Request 0's first 32 tokens were identical in all 10 runs we have, so MPK's own uninterrupted output is a usable reference; B1's "run twice, confirm bit-identical" step should still confirm this on full outputs.

## Reproduce

```bash
# needs ~/mirage on branch c2-profiler-fix, the `mirage` conda env, Qwen/Qwen3-1.7B in the HF cache
mkdir -p logs && sbatch profiling/c2_profile.sbatch          # ~15 min on one A100
PROFILE_TOKENS="" sbatch --time=00:30:00 profiling/c2_profile.sbatch   # batch timing only (~10 min)
sbatch profiling/check_batch_outputs.sbatch                  # output correctness check (~5 min)
python profiling/analyze_c2.py results/c2-<jobid>            # writes c2_summary.md + c2_task_types.csv
```
