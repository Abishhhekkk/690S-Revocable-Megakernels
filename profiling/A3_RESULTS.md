# A3: MPK yield/resume on A100

Qwen3-1.7B, 512-token decode, A100-SXM4-80GB, MPK + `mpk/patches/0001`–`0004`. Test: [a3_yield_test.sbatch](a3_yield_test.sbatch). Raw data: [data/](data/).

## Results (job 2345773, 50 yields at random points per run)

| yield latency | batch 1, drain | batch 1, **task** | batch 8, drain | batch 8, **task** |
|---|---:|---:|---:|---:|
| mean | 1.88 ms | **0.09 ms** | 1.95 ms | **0.10 ms** |
| p90 | 3.11 ms | **0.14 ms** | 3.57 ms | **0.21 ms** |
| max | 3.58 ms | **0.28 ms** | 3.63 ms | **0.39 ms** |
| extra run time per yield | 0.15 ms | 0.21 ms | 0.16 ms | 0.32 ms |
| tokens vs uninterrupted | identical | identical | identical | identical |

- **drain** (patch 0001 as written): workers finish their whole queue before exiting. MPK queues tasks ahead of their dependencies, so a yield waits for about the rest of the decode step (3.7 ms).
- **task** (patch 0004): workers stop at the next task boundary, including while waiting on a dependency. That's about **20× faster** on average and 9–13× faster at worst.
- **Correctness:** 210 yields across both modes and batch sizes. Every run was bit-identical to the uninterrupted one.
- Latency is measured on the host (request until both kernel streams have stopped), so it includes a few tens of µs of host overhead.

## Known bug

`PersistentKernel.finalize()` raises "no resumable checkpoint" if the kernel finished normally without `wait()`.
