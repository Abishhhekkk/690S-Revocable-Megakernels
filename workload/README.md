# C3: workload generator

Writes a reproducible request schedule (JSONL) from a JSON config. Standard library only.

```bash
python workload/gen.py poisson -o results/wl/poisson.jsonl    # preset name or config path
python workload/gen.py bursty --seed 3 --duration 30 -o out.jsonl
python workload/test_gen.py
```

One request per line (the first line holds the seed and config):

```json
{"id": 17, "cls": "urgent", "arrival_s": 1.23, "prompt_len": 34, "output_len": 16, "deadline_ms": 50, "priority": 0}
```

**Arrivals:** `poisson`, `bursty` (on/off), `gamma` (set `cv`), `random` (fixed count), `periodic`, `batch` (all at once), `trace` (CSV replay, Azure format).
**Lengths:** `fixed`, `uniform`, `lognormal`, `choice`.

**Presets** (`configs/`): `poisson`, `bursty`, `steady_batch_random_urgent`, `long_prompt_background`, `azure_trace`. The Azure trace isn't included; put the CSV in `workload/traces/`.

The 50 ms deadline is a placeholder; set it with D3.
