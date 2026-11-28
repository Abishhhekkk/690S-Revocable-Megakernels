## Decode iteration latency (unprofiled)

| batch R | max seq | prompt | generated/request | total ms | per-token ms | tokens/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 1 | 255 | 1085.2 | 4.239 | 235 |
| 1 | 512 | 1 | 511 | 2135.1 | 4.170 | 239 |
| timing_R2_L256 | | | | FAILED | |
| timing_R2_L512 | | | | FAILED | |
| timing_R4_L256 | | | | FAILED | |
| timing_R4_L512 | | | | FAILED | |
| timing_R8_L256 | | | | FAILED | |
| timing_R8_L512 | | | | FAILED | |

Per decode iteration = (total_long - total_short) / (gen_long - gen_short), which cancels launch, prefill and setup:

- R=1: **4.101 ms per decode iteration** (~244 tokens/s for the batch)

