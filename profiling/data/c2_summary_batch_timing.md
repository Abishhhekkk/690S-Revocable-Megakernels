# C2 profiling summary: c2-2270526

```
model=Qwen/Qwen3-1.7B
profile_tokens=
batches=1 2 4 8
seq_short=256
seq_long=512
mirage_commit=b58d5de9b7e0004a68bb190ba97a5159f3af1798
gpu=NVIDIA A100-SXM4-80GB
```

## Task durations (one profiled iteration)

## Decode iteration latency (unprofiled)

| batch R | max seq | prompt | generated/request | total ms | per-token ms | tokens/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 1 | 255 | 907.4 | 3.545 | 281 |
| 1 | 512 | 1 | 511 | 1867.3 | 3.647 | 274 |
| 2 | 256 | 1 | 255 | 912.4 | 3.564 | 559 |
| 2 | 512 | 1 | 511 | 1873.1 | 3.658 | 546 |
| 4 | 256 | 1 | 255 | 921.5 | 3.599 | 1107 |
| 4 | 512 | 1 | 511 | 1892.4 | 3.696 | 1080 |
| 8 | 256 | 1 | 255 | 938.1 | 3.664 | 2175 |
| 8 | 512 | 1 | 511 | 1933.8 | 3.777 | 2114 |

Per decode iteration = (total_long - total_short) / (gen_long - gen_short), which cancels launch, prefill and setup:

- R=1: **3.750 ms per decode iteration** (~267 tokens/s for the batch)
- R=2: **3.753 ms per decode iteration** (~533 tokens/s for the batch)
- R=4: **3.793 ms per decode iteration** (~1055 tokens/s for the batch)
- R=8: **3.889 ms per decode iteration** (~2057 tokens/s for the batch)

