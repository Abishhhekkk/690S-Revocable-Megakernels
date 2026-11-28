# C2 profiling summary: c2-2254718

```
model=Qwen/Qwen3-1.7B
profile_tokens=1 8 16
batches=1 2 4 8
seq_short=256
seq_long=512
mirage_commit=b58d5de9b7e0004a68bb190ba97a5159f3af1798
gpu=NVIDIA A100-PCIE-40GB
```

## Task durations (one profiled iteration)

### profile_T1

- iteration span: **3982.3 us** (9929 tasks on 96 worker blocks)
- worker utilization: 51.7% (sum of task time / span x workers)
- tasks per worker: min 95, max 113
- longest single task: **451.6 us** (TASK_LINEAR) -> worst-case wait for a task-boundary yield

| task type | count | mean us | p50 us | p99 us | max us | share of task time |
|---|---:|---:|---:|---:|---:|---:|
| TASK_LINEAR | 4576 | 35.2 | 33.8 | 441.3 | 451.6 | 81.4% |
| TASK_LINEAR_WITH_RESIDUAL | 1792 | 18.4 | 25.6 | 27.6 | 28.7 | 16.7% |
| TASK_PAGED_ATTENTION_1 | 224 | 10.2 | 10.2 | 11.3 | 11.3 | 1.2% |
| TASK_SCHD_EVENTS | 1837 | 0.4 | 0.0 | 1.0 | 2.0 | 0.4% |
| TASK_ARGMAX_PARTIAL | 96 | 2.8 | 3.1 | 4.1 | 4.1 | 0.1% |
| TASK_SILU_MUL | 1344 | 0.1 | 0.0 | 1.0 | 1.0 | 0.1% |
| TASK_RMS_NORM | 57 | 2.1 | 2.0 | 2.0 | 3.1 | 0.1% |
| TASK_EMBEDDING | 1 | 7.2 | 7.2 | 7.2 | 7.2 | 0.0% |
| (sched) TASK_SCHD_PREPARE_BATCH | 1 | 3.1 | 3.1 | 3.1 | 3.1 | (scheduler) |
| TASK_ARGMAX_REDUCE | 1 | 2.0 | 2.0 | 2.0 | 2.0 | 0.0% |
| TASK_BEGIN_TASK_GRAPH | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0% |

### profile_T16

- iteration span: **6518.8 us** (10784 tasks on 96 worker blocks)
- worker utilization: 37.7% (sum of task time / span x workers)
- tasks per worker: min 99, max 132
- longest single task: **474.1 us** (TASK_LINEAR) -> worst-case wait for a task-boundary yield

| task type | count | mean us | p50 us | p99 us | max us | share of task time |
|---|---:|---:|---:|---:|---:|---:|
| TASK_LINEAR | 4576 | 36.9 | 35.8 | 468.0 | 474.1 | 71.5% |
| TASK_LINEAR_WITH_RESIDUAL | 1792 | 19.6 | 26.6 | 29.7 | 31.7 | 14.9% |
| TASK_PAGED_ATTENTION_1 | 224 | 85.6 | 85.0 | 89.1 | 90.1 | 8.1% |
| TASK_SILU_MUL | 1344 | 4.7 | 5.1 | 5.1 | 6.1 | 2.7% |
| TASK_ARGMAX_PARTIAL | 96 | 38.9 | 38.9 | 42.0 | 43.0 | 1.6% |
| TASK_RMS_NORM | 912 | 2.0 | 2.0 | 2.0 | 2.0 | 0.8% |
| TASK_SCHD_EVENTS | 1837 | 0.5 | 0.0 | 1.0 | 3.1 | 0.4% |
| TASK_EMBEDDING | 1 | 116.7 | 116.7 | 116.7 | 116.7 | 0.0% |
| TASK_ARGMAX_REDUCE | 1 | 19.5 | 19.5 | 19.5 | 19.5 | 0.0% |
| (sched) TASK_SCHD_PREPARE_BATCH | 2 | 6.1 | 5.1 | 7.2 | 7.2 | (scheduler) |
| TASK_BEGIN_TASK_GRAPH | 1 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0% |

### profile_T8

- iteration span: **5112.8 us** (10328 tasks on 96 worker blocks)
- worker utilization: 43.5% (sum of task time / span x workers)
- tasks per worker: min 96, max 119
- longest single task: **462.8 us** (TASK_LINEAR) -> worst-case wait for a task-boundary yield

| task type | count | mean us | p50 us | p99 us | max us | share of task time |
|---|---:|---:|---:|---:|---:|---:|
| TASK_LINEAR | 4576 | 35.9 | 34.8 | 455.7 | 462.8 | 76.8% |
| TASK_LINEAR_WITH_RESIDUAL | 1792 | 18.5 | 25.6 | 27.6 | 28.7 | 15.5% |
| TASK_PAGED_ATTENTION_1 | 224 | 45.1 | 45.1 | 46.1 | 46.1 | 4.7% |
| TASK_SILU_MUL | 1344 | 2.1 | 2.0 | 3.1 | 3.1 | 1.3% |
| TASK_ARGMAX_PARTIAL | 96 | 18.3 | 18.4 | 20.5 | 20.5 | 0.8% |
| TASK_RMS_NORM | 456 | 2.0 | 2.0 | 2.0 | 2.0 | 0.4% |
| TASK_SCHD_EVENTS | 1837 | 0.4 | 0.0 | 1.0 | 3.1 | 0.4% |
| TASK_EMBEDDING | 1 | 57.3 | 57.3 | 57.3 | 57.3 | 0.0% |
| (sched) TASK_SCHD_PREPARE_BATCH | 2 | 5.6 | 4.1 | 7.2 | 7.2 | (scheduler) |
| TASK_ARGMAX_REDUCE | 1 | 9.2 | 9.2 | 9.2 | 9.2 | 0.0% |
| TASK_BEGIN_TASK_GRAPH | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% |

## Decode iteration latency (unprofiled)

| batch R | max seq | prompt | generated/request | total ms | per-token ms | tokens/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 1 | 255 | 1022.0 | 3.992 | 250 |
| 1 | 512 | 1 | 511 | 2095.1 | 4.092 | 244 |
| timing_R2_L256 | | | | FAILED | |
| timing_R2_L512 | | | | FAILED | |
| timing_R4_L256 | | | | FAILED | |
| timing_R4_L512 | | | | FAILED | |
| timing_R8_L256 | | | | FAILED | |
| timing_R8_L512 | | | | FAILED | |

Per decode iteration = (total_long - total_short) / (gen_long - gen_short), which cancels launch, prefill and setup:

- R=1: **4.192 ms per decode iteration** (~239 tokens/s for the batch)

