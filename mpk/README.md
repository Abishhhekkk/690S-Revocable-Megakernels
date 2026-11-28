# MPK changes

Our changes to [MPK](https://github.com/mirage-project/mirage), kept as patches against upstream commit `ed771bda` (branch `mpk`), applied in order:

| patch | what | by |
|---|---|---|
| `0001-a3-yield-resume.patch` | A3: revocable persistent kernel. Scheduler and worker checkpoints, `request_revocation()` / `wait()` / `resume()`, and yield/resume in `LLMEngine` and `launch_server` | Abhishek |
| `0002-sm80-paged-attention-params.patch` | A100 fix: send the 10-param paged-attention layout the sm80 code generator expects (batch > 1 crashed) | Abhijit |
| `0003-profiler-scheduler-region.patch` | Profiler fix: give `scheduler_kernel` its own profiler-buffer region | Abhijit |
| `0004-task-boundary-yield.patch` | Workers stop at the next task boundary, including while waiting on a dependency, instead of draining their queue. `MPK_YIELD_MODE=drain` restores the 0001 behaviour | Abhijit |

## Setup

```bash
mpk/setup_mpk.sh            # -> ~/mirage-a3, reusing the built ~/mirage
export PYTHONPATH=~/mirage-a3/python
```

`~/mirage` must be a built checkout of upstream mirage (`pip install -e .` in the `mirage` conda env). Building it is a heavy source build, so do that in a job, not on a login node.

## Tests

- `profiling/a3_yield_test.sbatch`: yield/resume during decode on an A100. Outputs must be bit-identical to an uninterrupted run. Results are in [profiling/A3_RESULTS.md](../profiling/A3_RESULTS.md).
- `tests/python/test_llm_engine_yield.py` (in the patched tree): CPU-only tests of the engine's yield/resume control flow.
