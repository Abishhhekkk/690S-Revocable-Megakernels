#!/usr/bin/env python3
"""Independent correctness checker for the revocable megakernel (Part B1).

Reads the per-task records dumped by the kernel (task_log.bin) and the expected
task graph (task_graph.json) and checks, without trusting the kernel's own
counters:

  1. exactly-once   every (step, task_id) appears exactly once across ALL launches
  2. dependencies   start(task) >= end(dep) for every edge (ties allowed, see below)
  3. cross-step     step s+1 sources start after step s sinks end (if enabled)
  4. sanity         unknown ids/steps, unfinished tasks (end==0), end < start,
                    log overflow, and one worker running two tasks at once

Tie rule: %globaltimer is coarse, so end(dep) == start(task) is ACCEPTED. A
violation is start(task) + tolerance_ns < end(dep). Ties are only counted and
reported as info. The kernel writes end_ns before releasing dependents, so a
correct run satisfies end(dep) <= start(dependent) by construction.

Usage:
    python3 checker.py task_log.bin task_graph.json [--tolerance-ns N]
Exit code 0 = PASS, 1 = FAIL.
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np

# Must match struct TaskRecord in include/task_record.cuh (40 bytes).
TASK_RECORD_DTYPE = np.dtype([
    ("start_ns", "<u8"),
    ("end_ns", "<u8"),
    ("task_id", "<u4"),
    ("step", "<u4"),
    ("launch_id", "<u4"),
    ("sm_id", "<u4"),
    ("worker_id", "<u4"),
    ("reserved", "<u4"),
])
assert TASK_RECORD_DTYPE.itemsize == 40


def load_records(path):
    return np.fromfile(path, dtype=TASK_RECORD_DTYPE)


def load_graph(path):
    with open(path) as f:
        return json.load(f)


class Report:
    def __init__(self):
        self.errors = []   # (category, message)
        self.info = []

    def err(self, cat, msg):
        self.errors.append((cat, msg))

    def note(self, msg):
        self.info.append(msg)

    @property
    def ok(self):
        return not self.errors

    def categories(self):
        return sorted({c for c, _ in self.errors})

    def print(self, max_per_cat=10, file=sys.stdout):
        by_cat = defaultdict(list)
        for c, m in self.errors:
            by_cat[c].append(m)
        for c in sorted(by_cat):
            print(f"[FAIL] {c}: {len(by_cat[c])} issue(s)", file=file)
            for m in by_cat[c][:max_per_cat]:
                print(f"       {m}", file=file)
            if len(by_cat[c]) > max_per_cat:
                print(f"       ... {len(by_cat[c]) - max_per_cat} more", file=file)
        for m in self.info:
            print(f"[info] {m}", file=file)
        print("RESULT: " + ("PASS" if self.ok else "FAIL"), file=file)


def check(records, graph, tolerance_ns=0):
    """Return a Report. `records` is a structured numpy array (TASK_RECORD_DTYPE)."""
    rep = Report()

    tasks = graph["tasks"]
    num_steps = int(graph.get("num_steps", 1))
    cross_step = graph.get("cross_step", "none")
    ids = [t["id"] for t in tasks]
    id_set = set(ids)
    deps = {t["id"]: list(t.get("deps", [])) for t in tasks}

    # ---- log overflow --------------------------------------------------
    if "log_count" in graph and "log_capacity" in graph:
        if graph["log_count"] > graph["log_capacity"]:
            rep.err("overflow",
                    f"log overflowed: {graph['log_count']} records attempted, "
                    f"capacity {graph['log_capacity']}; checker results unreliable")
        if graph["log_count"] != len(records) and graph["log_count"] <= graph["log_capacity"]:
            rep.err("overflow",
                    f"header says {graph['log_count']} records, file has {len(records)}")

    # ---- group records by (step, task_id) ------------------------------
    by_key = defaultdict(list)
    for i, r in enumerate(records):
        key = (int(r["step"]), int(r["task_id"]))
        by_key[key].append(i)

    # ---- sanity: unknown ids / steps -----------------------------------
    for (step, tid), idx in by_key.items():
        if tid not in id_set:
            rep.err("unknown_task", f"record for unknown task_id {tid} (step {step})")
        if step >= num_steps:
            rep.err("unknown_task", f"record for step {step} but graph has {num_steps} step(s)")

    # ---- 1. exactly-once -----------------------------------------------
    for step in range(num_steps):
        for tid in ids:
            n = len(by_key.get((step, tid), []))
            if n == 0:
                rep.err("missing_task", f"step {step} task {tid} never executed")
            elif n > 1:
                launches = sorted({int(records[i]["launch_id"]) for i in by_key[(step, tid)]})
                rep.err("duplicate_task",
                        f"step {step} task {tid} executed {n} times (launches {launches})")

    # ---- sanity: per-record timing -------------------------------------
    for i, r in enumerate(records):
        if r["end_ns"] == 0:
            rep.err("unfinished_task",
                    f"step {r['step']} task {r['task_id']} started but never finished "
                    f"(launch {r['launch_id']}, worker {r['worker_id']})")
        elif r["end_ns"] < r["start_ns"]:
            rep.err("bad_timestamps",
                    f"step {r['step']} task {r['task_id']}: end < start")

    # ---- 2. dependency order (within a step) ---------------------------
    # Use the (single) record per key; skip keys that are missing/duplicated
    # (already reported above) so one bug does not produce a cascade of noise.
    def single(step, tid):
        idx = by_key.get((step, tid), [])
        return records[idx[0]] if len(idx) == 1 else None

    ties = 0
    for step in range(num_steps):
        for tid in ids:
            t = single(step, tid)
            if t is None:
                continue
            for d in deps[tid]:
                dr = single(step, d)
                if dr is None or dr["end_ns"] == 0:
                    continue
                s, e = int(t["start_ns"]), int(dr["end_ns"])
                if s == e:
                    ties += 1
                elif s + tolerance_ns < e:
                    rep.err("dependency_violation",
                            f"step {step} task {tid} started {e - s} ns before "
                            f"dependency {d} ended")
    if ties:
        rep.note(f"{ties} dependency edge(s) with start == end (accepted tie)")

    # ---- 3. cross-step ordering ----------------------------------------
    if cross_step == "barrier" and num_steps > 1:
        has_dependent = {d for t in tasks for d in t.get("deps", [])}
        sinks = [i for i in ids if i not in has_dependent]
        sources = [i for i in ids if not deps[i]]
        for step in range(1, num_steps):
            for tid in sources:
                t = single(step, tid)
                if t is None:
                    continue
                for sk in sinks:
                    pr = single(step - 1, sk)
                    if pr is None or pr["end_ns"] == 0:
                        continue
                    s, e = int(t["start_ns"]), int(pr["end_ns"])
                    if s + tolerance_ns < e:
                        rep.err("cross_step_violation",
                                f"step {step} task {tid} started {e - s} ns before "
                                f"step {step - 1} task {sk} ended")

    # ---- 4. one worker cannot run two tasks at the same time -----------
    per_worker = defaultdict(list)
    for r in records:
        if r["end_ns"] != 0:
            per_worker[(int(r["launch_id"]), int(r["worker_id"]))].append(
                (int(r["start_ns"]), int(r["end_ns"]), int(r["step"]), int(r["task_id"])))
    for (launch, worker), spans in per_worker.items():
        spans.sort()
        for a, b in zip(spans, spans[1:]):
            if b[0] + tolerance_ns < a[1]:
                rep.err("worker_overlap",
                        f"launch {launch} worker {worker}: step {a[2]} task {a[3]} and "
                        f"step {b[2]} task {b[3]} overlap")

    # ---- summary info ---------------------------------------------------
    launches = sorted({int(x) for x in records["launch_id"]}) if len(records) else []
    rep.note(f"{len(records)} records, {num_steps} step(s), {len(ids)} tasks/step, "
             f"launches seen: {launches}")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="task_log.bin")
    ap.add_argument("graph", help="task_graph.json")
    ap.add_argument("--tolerance-ns", type=int, default=0,
                    help="slack for dependency/overlap comparisons (default 0)")
    args = ap.parse_args()

    rep = check(load_records(args.log), load_graph(args.graph), args.tolerance_ns)
    rep.print()
    sys.exit(0 if rep.ok else 1)


if __name__ == "__main__":
    main()
