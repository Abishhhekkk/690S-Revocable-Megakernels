# A3 on A100: MPK yield/resume results

Job 2343944, A100-SXM4-80GB, Qwen3-1.7B, 512-token decode (`run_batch_perf.py --ignore-eos`). The MPK code was upstream `ed771bda` plus Abhishek's A3 patch and our sm80 and profiler fixes (`~/mirage-a3`, built with `mpk/setup_mpk.sh`). Script: [a3_yield_test.py](a3_yield_test.py) / [a3_yield_test.sbatch](a3_yield_test.sbatch). Raw data: [data/a3-2343944/](data/a3-2343944/).

## Correctness: pass

| run | yields | checkpoint every time | tokens vs uninterrupted |
|---|---:|---|---|
| batch 1, second baseline | 0 | – | identical |
| batch 1, every 250 ms, 20 ms pause | 5 | yes | **identical** |
| batch 1, every 50 ms, 5 ms pause | 20 | yes | **identical** |
| batch 8, second baseline | 0 | – | identical |
| batch 8, every 250 ms, 20 ms pause | 5 | yes | **identical** |

## Cost

| | batch 1 | batch 8 |
|---|---|---|
| decode step (no yields) | 3.43 ms | 3.61 ms |
| yield latency (request → GPU free), p50 / max | 1.3–1.8 / 3.1 ms | 2.2 / 2.4 ms |
| `resume()` call (async relaunch) | ~0.05 ms | ~0.06 ms |
| extra run time beyond the pauses | ~0.2–0.6 ms per yield | ~0 per yield |

## Yield latency was about one decode step, not one task (drain mode)

Across the 20 yields, latency climbs from 0.3 ms to 3.1 ms and then wraps around, over a period equal to the 3.43 ms decode step. Cause: MPK's scheduler hands workers tasks before their dependencies are ready, and workers wait for those dependencies inside the worker loop (`persistent_kernel.cuh`, dependent-event wait). A3 made workers drain their whole queue before exiting, so a yield waited for the rest of the dependency chain, which is roughly the rest of the iteration.

## Task-boundary yield (fix, job 2345773)

[mpk/patches/0004-task-boundary-yield.patch](../mpk/patches/0004-task-boundary-yield.patch), about 40 lines on top of A3: before each task, and while waiting on a task's dependency, a worker checks for a revocation request. If there is one, it saves `consumed_task_pos` with that task not yet run, and exits. Resume reloads the queue from there, using A3's existing checkpoint. `MPK_YIELD_MODE=drain` keeps A3's original behaviour for comparison. A100-SXM4-80GB (uri-gpu002); 50 yields at random gaps (10–30 ms, 2 ms pause), so every point in a decode step is sampled. The same random schedule was used in both modes.

| | batch 1, drain | batch 1, **task** | batch 8, drain | batch 8, **task** |
|---|---:|---:|---:|---:|
| yield latency, mean | 1.88 ms | **0.092 ms** | 1.95 ms | **0.104 ms** |
| p50 | 2.16 ms | **0.076 ms** | 1.97 ms | **0.080 ms** |
| p90 | 3.11 ms | **0.144 ms** | 3.57 ms | **0.212 ms** |
| max | 3.58 ms | **0.279 ms** | 3.63 ms | **0.392 ms** |
| extra run time per yield, beyond the pauses | 0.15 ms | 0.21 ms | 0.16 ms | 0.32 ms |
| tokens vs uninterrupted | identical | identical | identical | identical |

- **Task-boundary yield is about 20× faster on average and 9–13× faster at worst** than draining, which behaves like iteration-boundary waiting. The decode step is 3.68 ms (batch 1) and 3.86 ms (batch 8) on this node.
- These latencies are measured on the host (request → both kernel streams synced). They include a host floor of a few tens of µs for the flag copy and the stream syncs, so the GPU-side latency is lower.
- Task-boundary yield costs about 0.1 ms more per yield than drain. A likely cause is that the pipeline restarts mid-dependency chain after resume.
- Correctness: 210 yields across both modes and batch sizes, all with valid checkpoints. Every output was bit-identical to the uninterrupted run.

## Bug found

After a run that finishes normally without `wait()`, `PersistentKernel.finalize()` still calls `request_revocation()` and then raises "kernel stopped without a resumable checkpoint". It should only revoke if the kernel is still running.
