"""Synthetic task logs for testing the checker without a GPU (Part B1).

`simulate()` list-schedules a task graph on W workers and emits records in the
exact on-disk format of the kernel (TASK_RECORD_DTYPE). The `plant_*` helpers
corrupt a clean log in one specific way; they are the seed of the B2 bug suite.
"""
from collections import deque

import numpy as np

from checker import TASK_RECORD_DTYPE

# The 6-task diamond graph hard-coded in src/main.cu
TOY_GRAPH = {
    "num_steps": 1,
    "cross_step": "none",
    "tasks": [
        {"id": 0, "deps": []},
        {"id": 1, "deps": []},
        {"id": 2, "deps": [0, 1]},
        {"id": 3, "deps": [2]},
        {"id": 4, "deps": [2]},
        {"id": 5, "deps": [3, 4]},
    ],
}


def simulate(graph, num_steps=1, workers=4, duration_ns=1000,
             launch_boundaries=(), base_ns=1_000_000):
    """Return a structured array of records for a correct execution.

    launch_boundaries: sorted timestamps; a task that STARTS at/after the k-th
    boundary gets launch_id k (models yield/resume splitting the run).
    Steps run back to back (barrier between steps), as in autoregressive decode.
    """
    tasks = graph["tasks"]
    deps = {t["id"]: list(t.get("deps", [])) for t in tasks}
    dependents = {t["id"]: [] for t in tasks}
    for tid, ds in deps.items():
        for d in ds:
            dependents[d].append(tid)

    rows = []
    step_start = base_ns
    for step in range(num_steps):
        remaining = {tid: len(ds) for tid, ds in deps.items()}
        ready = deque((step_start, tid) for tid, ds in deps.items() if not ds)
        free = [step_start] * workers
        end_of = {}
        while ready:
            ready_time, tid = ready.popleft()
            w = min(range(workers), key=lambda i: free[i])
            start = max(free[w], ready_time)
            end = start + duration_ns
            free[w] = end
            end_of[tid] = end
            launch = sum(1 for b in launch_boundaries if start >= b)
            rows.append((start, end, tid, step, launch, w, w, 0))
            for dep_id in dependents[tid]:
                remaining[dep_id] -= 1
                if remaining[dep_id] == 0:
                    ready.append((max(end_of[d] for d in deps[dep_id]), dep_id))
        step_start = max(end_of.values())

    arr = np.array(rows, dtype=TASK_RECORD_DTYPE)
    return arr


def graph_with_steps(graph, num_steps, cross_step="barrier"):
    g = dict(graph)
    g["num_steps"] = num_steps
    g["cross_step"] = cross_step
    return g


# ---------------------------------------------------------------------------
# Planted bugs. Each returns a modified COPY.
# ---------------------------------------------------------------------------
def _find(arr, step, tid):
    idx = np.nonzero((arr["step"] == step) & (arr["task_id"] == tid))[0]
    assert len(idx) == 1
    return int(idx[0])


def plant_duplicate(arr, step=0, tid=3):
    i = _find(arr, step, tid)
    dup = arr[i:i + 1].copy()
    dup["launch_id"] += 1
    dup["start_ns"] += 50_000
    dup["end_ns"] += 50_000
    dup["worker_id"] = 7
    return np.concatenate([arr, dup])


def plant_missing(arr, step=0, tid=4):
    return np.delete(arr, _find(arr, step, tid))


def plant_dependency_violation(arr, step=0, tid=3, dep=2):
    out = arr.copy()
    i, d = _find(out, step, tid), _find(out, step, dep)
    shift = int(out[i]["start_ns"]) - int(out[d]["end_ns"]) + 100  # start 100 ns early
    out[i]["start_ns"] -= shift
    out[i]["end_ns"] -= shift
    return out


def plant_unfinished(arr, step=0, tid=5):
    out = arr.copy()
    out[_find(out, step, tid)]["end_ns"] = 0
    return out


def plant_worker_overlap(arr, step=0, a=0, b=1):
    """Force two concurrent tasks onto the same worker."""
    out = arr.copy()
    ia, ib = _find(out, step, a), _find(out, step, b)
    out[ib]["worker_id"] = out[ia]["worker_id"]
    out[ib]["launch_id"] = out[ia]["launch_id"]
    return out


def plant_cross_step_violation(arr, step=1, tid=0):
    out = arr.copy()
    out[_find(out, step, tid)]["start_ns"] -= 10_000
    return out


def plant_unknown_task(arr):
    extra = arr[:1].copy()
    extra["task_id"] = 99
    return np.concatenate([arr, extra])
