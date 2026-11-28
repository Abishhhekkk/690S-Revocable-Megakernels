# C3: workload generator

`gen.py` turns a JSON config into a reproducible request schedule (JSONL). It uses only the standard library, so it runs on a login node.

```bash
python workload/gen.py --list-presets
python workload/gen.py poisson -o results/wl/poisson.jsonl          # preset name or a path
python workload/gen.py bursty --seed 3 --duration 30 -o results/wl/bursty_s3.jsonl
python workload/test_gen.py                                         # 10 tests
```

Each output line is one request:

```json
{"id": 17, "cls": "urgent", "arrival_s": 1.2345, "prompt_len": 34, "output_len": 16, "deadline_ms": 50, "priority": 0}
```

The first line, `{"_meta": ...}`, stores the seed and the full config. Read a file back with `from gen import load_workload`.

## Arrival processes

| type | parameters | use |
|---|---|---|
| `poisson` | `rate` | steady arrivals |
| `bursty` | `rate_on`, `rate_off`, `mean_on_s`, `mean_off_s` | on/off bursts (2-state MMPP) |
| `gamma` | `rate`, `cv` | burstiness as a single knob (`cv` = 1 is Poisson) |
| `random` | `count`, `min_gap_s` | random urgent arrivals with a fixed count (C4 step 1) |
| `periodic` | `interval_s`, `jitter_s` | controlled sweeps |
| `batch` | `count`, `at_s` | large-batch / steady background decode |
| `trace` | `path`, `speedup`, `offset_s`, column names | public-trace replay |

Lengths: `fixed`, `uniform`, `lognormal`, or `choice`, each with optional `lo`/`hi` clamps. `prompt_len + output_len` is clamped to `max_seq_len`. Each class has its own RNG stream, so adding an urgent class does not change the background arrivals.

## Presets (`configs/`)

| preset | background | urgent |
|---|---|---|
| `poisson` | Poisson 2/s, prompt about 128, output 64–256 | Poisson 0.5/s, 16–64 → 16 tokens, 50 ms deadline |
| `bursty` | same as `poisson` | bursts of about 5/s for about 1 s every about 10 s |
| `steady_batch_random_urgent` | 8 requests at t=0, 480 output tokens | 20 random arrivals, at least 0.5 s apart |
| `long_prompt_background` | prompts of 1–3k tokens (prefill-heavy) | Poisson 0.5/s |
| `azure_trace` | Azure LLM inference trace (conversation) | 30 random arrivals |

The 50 ms deadline is a placeholder. From C2, one decode step is about 3.75–4.2 ms, so waiting for the iteration boundary fits easily. Make deadlines tighter for the yield-latency experiments, or loosen them for the queueing experiments, together with D3.

## Public trace

`azure_trace` expects `workload/traces/AzureLLMInferenceTrace_conv.csv`, which is in the Azure Public Dataset on GitHub (`AzurePublicDatasetV2`, LLM inference 2023). Its columns are `TIMESTAMP, ContextTokens, GeneratedTokens`. The trace is not checked in. Most trace prompts exceed what we run on MPK, so clamp them with `max_seq_len`, or add `prompt_len`/`output_len` to the class to keep only the trace's timing.
