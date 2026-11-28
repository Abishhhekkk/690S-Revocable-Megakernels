#!/usr/bin/env python3
"""Run an MPK demo with profiling, using our own profiler-buffer export.

Usage:
    python profiling/mpk_profiled_run.py <stem> <demo.py> [demo args...]

MPK's built-in export (export_to_perfetto_trace / export_to_csv) assumes one
kernel owns the profiler buffer. On GPUs where MPK launches worker_kernel and
scheduler_kernel separately (e.g. A100: 96 workers + 12 scheduler blocks),
both kernels write a header at buffer[0] and interleave their events, so the
export crashes (KeyError) or mixes events.

This needs the companion MPK patch (branch c2-profiler-fix in ~/mirage) that
moves the scheduler kernel's events to buffer[MPK_PROFILER_SCHED_OFFSET:].
We then decode the two regions separately and write:
    <stem>.raw.npy   the raw uint64 profiler buffer
    <stem>.csv       one row per task/scheduler event, with a `kernel` column
"""
import csv
import os
import runpy
import sys

import numpy as np

SCHED_OFFSET = 192000  # must match MPK_PROFILER_SCHED_OFFSET in the MPK patch


def decode_region(buf, kernel, names):
    """Pair BEGIN/END events of one kernel's region. Returns (rows, dangling)."""
    if len(buf) < 2:
        return [], 0
    nblocks, ngroups = buf[:1].view(np.uint32)
    nblocks, ngroups = int(nblocks), int(ngroups)
    if nblocks == 0 or ngroups == 0:
        return [], 0  # region never initialized (e.g. fused single-kernel mode)

    entries = buf[1:]
    entries = entries[entries != 0]
    pairs = entries.view(np.uint32).reshape(-1, 2)  # (tag, timestamp)
    pending, rows = {}, []
    for tag, ts in pairs:
        tag, ts = int(tag), int(ts)
        event_no = tag >> 19
        bg = (tag >> 11) & 0xFF
        block, group = bg // ngroups, bg % ngroups
        event_idx = (tag >> 2) & 0x1FF
        event_type = tag & 0x3
        name = names.get(event_idx, f"UNKNOWN_{event_idx}")
        key = (block, group, event_idx)
        if event_type == 0:  # begin
            pending[key] = (event_no, ts)
        elif event_type == 1:  # end
            if key in pending:
                no, b = pending.pop(key)
                rows.append((kernel, event_idx, name, block, group, no, b, ts,
                             (ts - b) & 0xFFFFFFFF))
        elif event_type == 2:  # instant
            rows.append((kernel, event_idx, name, block, group, event_no, ts, ts, 0))
    return rows, len(pending)


def install_export(stem):
    import mirage.mpk.profiler_persistent as pp

    def export_all(profiler_tensor, _file_name):
        # torch.uint64 has no direct .numpy(); reinterpret the bits via int64.
        import torch
        buf = profiler_tensor.detach().cpu().view(torch.int64).numpy().view(np.uint64)
        np.save(stem + ".raw.npy", buf)
        rows, dangling = [], 0
        for kernel, region in (("worker", buf[:SCHED_OFFSET]),
                               ("scheduler", buf[SCHED_OFFSET:])):
            r, d = decode_region(region, kernel, pp.event_name_list)
            rows += r
            dangling += d
        with open(stem + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kernel", "task_type_id", "task_type_name", "block_idx",
                        "group_idx", "event_no", "begin_ts", "end_ts", "duration_ns"])
            w.writerows(rows)
        print(f"[c2] wrote {stem}.csv ({len(rows)} events, {dangling} unpaired BEGINs)")
        if dangling:
            print("[c2] WARNING: unpaired BEGIN events -> profiler region may have overflowed")

    # persistent_kernel.__call__ imports these names from the module at call time.
    pp.export_to_perfetto_trace = export_all
    pp.export_to_csv = lambda *a, **k: None


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    stem, demo = sys.argv[1], os.path.abspath(sys.argv[2])
    install_export(stem)
    sys.path.insert(0, os.path.dirname(demo))  # demo imports its local `models` package
    sys.argv = [demo] + sys.argv[3:]
    runpy.run_path(demo, run_name="__main__")


if __name__ == "__main__":
    main()
