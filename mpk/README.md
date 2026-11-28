# MPK changes

Our changes to [MPK](https://github.com/mirage-project/mirage), kept as patches in `patches/` against upstream commit `ed771bda` (branch `mpk`), applied in order.

## Setup

```bash
mpk/setup_mpk.sh            # -> ~/mirage-a3, reusing the built ~/mirage
export PYTHONPATH=~/mirage-a3/python
```

`~/mirage` must be a built checkout of upstream mirage (`pip install -e .` in the `mirage` conda env). Building it is a heavy source build, so do that in a job, not on a login node.

## Tests

- `profiling/a3_yield_test.sbatch`: yield/resume during decode on an A100. Outputs must be bit-identical to an uninterrupted run. Results are in [profiling/A3_RESULTS.md](../profiling/A3_RESULTS.md).
- `tests/python/test_llm_engine_yield.py` (in the patched tree): CPU-only tests of the engine's yield/resume control flow.
