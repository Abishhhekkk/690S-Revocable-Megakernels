#!/usr/bin/env python3
"""A3 on A100: yield/resume MPK mid-decode and check the output is unchanged.

Run (inside a GPU job, with PYTHONPATH pointing at the A3 MPK tree):
    python profiling/a3_yield_test.py run <out_stem> --yields N --gap-ms G --pause-ms P \
        -- <run_batch_perf.py> [its args...]
    python profiling/a3_yield_test.py compare <ref_stem> <stem> [<stem> ...]

`run` executes MPK's run_batch_perf.py. After MPK launches, it waits G ms,
calls request_revocation() + wait() (yield), keeps the GPU free for P ms,
then resume(), N times. It saves <out_stem>.tokens.npy and <out_stem>.json
(per-yield latency, resume time, total time). N=0 is the uninterrupted
reference.

`compare` checks every run's tokens are bit-identical to the reference.
"""
import json
import os
import runpy
import sys
import time

import numpy as np


def run(stem, yields, gap_ms, pause_ms, script, script_args, random_gap=False):
    import random
    import mirage.mpk.persistent_kernel as pkm
    rng = random.Random(0)

    events = []
    orig_call = pkm.PersistentKernel.__call__

    def call_with_yields(self, **kw):
        t0 = time.perf_counter()
        orig_call(self, **kw)
        for i in range(yields):
            time.sleep((gap_ms * (0.5 + rng.random()) if random_gap else gap_ms) / 1e3)
            ev = {"i": i}
            a = time.perf_counter()
            try:
                self.request_revocation()
                self.wait()
            except RuntimeError as e:   # kernel finished before the yield
                ev["error"] = str(e)
                events.append(ev)
                break
            b = time.perf_counter()
            ev["yield_ms"] = (b - a) * 1e3
            ev["checkpoint"] = bool(self.checkpoint_available)
            time.sleep(pause_ms / 1e3)  # GPU is free here (urgent work would run)
            c = time.perf_counter()
            self.resume()
            d = time.perf_counter()
            ev["resume_call_ms"] = (d - c) * 1e3
            ev["t_since_launch_ms"] = (a - t0) * 1e3
            events.append(ev)
        self.wait()                     # natural completion
        events.append({"done_ms": (time.perf_counter() - t0) * 1e3})

    pkm.PersistentKernel.__call__ = call_with_yields

    captured = {}

    def grab(frame, event, _arg):
        if event == "return" and frame.f_code.co_name == "main" and "tokens" in frame.f_locals:
            captured.update({k: frame.f_locals[k] for k in ("tokens", "step")})

    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    sys.argv = [script] + script_args
    sys.setprofile(grab)
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        sys.setprofile(None)

    tokens = captured["tokens"].cpu().numpy()
    steps = captured["step"].cpu().numpy()
    np.save(stem + ".tokens.npy", tokens)
    ok = [e for e in events if "yield_ms" in e]
    summary = {"yield_mode": os.environ.get("MPK_YIELD_MODE", "task"),
               "yields_requested": yields, "yields_done": len(ok),
               "gap_ms": gap_ms, "pause_ms": pause_ms,
               "steps": steps.tolist(), "events": events}
    if ok:
        ys = sorted(e["yield_ms"] for e in ok)
        rs = sorted(e["resume_call_ms"] for e in ok)
        summary.update(yield_ms_p50=ys[len(ys) // 2], yield_ms_max=ys[-1],
                       yield_ms_mean=sum(ys) / len(ys),
                       yield_ms_p90=ys[min(len(ys) - 1, int(0.9 * len(ys)))],
                       resume_call_ms_p50=rs[len(rs) // 2], resume_call_ms_max=rs[-1],
                       all_checkpoints=all(e["checkpoint"] for e in ok))
    with open(stem + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    print("[a3] " + json.dumps({k: v for k, v in summary.items() if k != "events"}))
    for e in events:
        print("[a3]   " + json.dumps(e))


def compare(ref, stems):
    r = np.load(ref + ".tokens.npy")
    rsteps = json.load(open(ref + ".json"))["steps"]
    ok = True
    for s in stems:
        t = np.load(s + ".tokens.npy")
        steps = json.load(open(s + ".json"))["steps"]
        same = t.shape == r.shape and np.array_equal(t, r) and steps == rsteps
        ok &= same
        detail = ""
        if not same and t.shape == r.shape:
            diff = np.argwhere(t != r)
            if len(diff):
                req, pos = diff[0]
                detail = f" first mismatch: request {req} position {pos}, {int((t != r).sum())} tokens differ"
            elif steps != rsteps:
                detail = f" steps differ: {steps} vs {rsteps}"
        print(f"[a3] {os.path.basename(s)} vs {os.path.basename(ref)}: "
              f"{'IDENTICAL' if same else 'DIFFERENT'}{detail}")
    print(f"[a3] COMPARE {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    if sys.argv[1] == "compare":
        sys.exit(0 if compare(sys.argv[2], sys.argv[3:]) else 1)
    if sys.argv[1] != "run" or "--" not in sys.argv:
        sys.exit(__doc__)
    i = sys.argv.index("--")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--yields", type=int, default=0)
    ap.add_argument("--gap-ms", type=float, default=250)
    ap.add_argument("--pause-ms", type=float, default=20)
    ap.add_argument("--random-gap", action="store_true",
                    help="gap uniform in [0.5, 1.5] x gap-ms (samples every phase of a step)")
    a = ap.parse_args(sys.argv[2:i])
    run(a.stem, a.yields, a.gap_ms, a.pause_ms, sys.argv[i + 1], sys.argv[i + 2:], a.random_gap)


if __name__ == "__main__":
    main()
