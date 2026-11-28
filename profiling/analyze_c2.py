#!/usr/bin/env python3
"""Summarize a C2 profiling run (see profiling/c2_profile.sbatch).

Usage:
    python profiling/analyze_c2.py results/c2-<jobid>

Reads
  profile_T*/profile_T*.csv   MPK per-task CSV from one profiled iteration
  timing_R*_L*/stdout.txt     unprofiled decode runs (MPK run_batch_perf.py output)
Writes
  <run>/c2_task_types.csv     per task type: count, mean/p50/p99/max duration
  <run>/c2_summary.md         the tables below, ready for the C2 report

Timestamps are MPK's 32-bit %globaltimer_lo values. Durations are already
computed modulo 2^32 by MPK; iteration spans are unwrapped relative to the
earliest begin, which is valid because one iteration is far below 4.3 s.
"""
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCHED_MIN_ID = 200  # fallback: task_type_id >= 200 are scheduler events


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0
    k = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[k]


def load_profile(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "kernel": r.get("kernel") or (
                    "scheduler" if int(r["task_type_id"]) >= SCHED_MIN_ID else "worker"),
                "type_id": int(r["task_type_id"]),
                "type": r["task_type_name"],
                "block": int(r["block_idx"]),
                "group": int(r["group_idx"]),
                "begin": int(r["begin_ts"]),
                "end": int(r["end_ts"]),
                "dur": int(r["duration_ns"]),
            })
    return rows


def summarize_profile(name, rows):
    tasks = [r for r in rows if r["kernel"] == "worker"]
    sched = [r for r in rows if r["kernel"] == "scheduler"]
    out = []
    out.append(f"### {name}\n")
    if not tasks:
        out.append("_no task events recorded_\n")
        return out, []

    # Wrap-aware earliest begin: offsets relative to an arbitrary event,
    # interpreted as signed 32-bit, so a timer wrap mid-iteration is handled.
    ref = tasks[0]["begin"]
    base = ref + min(((r["begin"] - ref + 2**31) & 0xFFFFFFFF) - 2**31 for r in tasks)
    span = max(((r["end"] - base) & 0xFFFFFFFF) for r in tasks)
    busy = sum(r["dur"] for r in tasks)
    workers = {r["block"] for r in tasks}
    per_worker = defaultdict(int)
    for r in tasks:
        per_worker[r["block"]] += 1
    longest = max(tasks, key=lambda r: r["dur"])

    out.append(f"- iteration span: **{span / 1e3:.1f} us** "
               f"({len(tasks)} tasks on {len(workers)} worker blocks)")
    out.append(f"- worker utilization: {100 * busy / (span * len(workers)):.1f}% "
               f"(sum of task time / span x workers)")
    out.append(f"- tasks per worker: min {min(per_worker.values())}, "
               f"max {max(per_worker.values())}")
    out.append(f"- longest single task: **{longest['dur'] / 1e3:.1f} us** "
               f"({longest['type']}) -> worst-case wait for a task-boundary yield\n")

    by_type = defaultdict(list)
    for r in tasks:
        by_type[r["type"]].append(r["dur"])
    for r in sched:
        by_type["(sched) " + r["type"]].append(r["dur"])
    table = []
    out.append("| task type | count | mean us | p50 us | p99 us | max us | share of task time |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for t, ds in sorted(by_type.items(), key=lambda kv: -sum(kv[1])):
        is_task = not t.startswith("(sched)")
        share = f"{100 * sum(ds) / busy:.1f}%" if is_task else "(scheduler)"
        row = (t, len(ds), statistics.mean(ds) / 1e3, pct(ds, 50) / 1e3,
               pct(ds, 99) / 1e3, max(ds) / 1e3)
        table.append((name,) + row)
        out.append(f"| {t} | {row[1]} | {row[2]:.1f} | {row[3]:.1f} | "
                   f"{row[4]:.1f} | {row[5]:.1f} | {share} |")
    out.append("")
    return out, table


def load_timing(stdout_path):
    """Parse MPK run_batch_perf.py's summary block."""
    text = Path(stdout_path).read_text(errors="replace")
    def num(label):
        m = re.search(rf"{label}:\s+([\d.]+)", text)
        return float(m.group(1)) if m else None
    total, per_req = num("total time"), None
    m = re.search(r"generated \(per request\):\s+\[?(\d+)", text)
    if m:
        per_req = int(m.group(1))
    if total is None or per_req is None:
        return None
    return {"prompt": int(num("prompt length") or 0), "gen": per_req,
            "total_ms": total, "per_tok_ms": num("per-token latency") or 0.0,
            "tput": num("throughput") or 0.0}


def summarize_timing(run):
    out = ["## Decode iteration latency (unprofiled)\n",
           "| batch R | max seq | prompt | generated/request | total ms | per-token ms | tokens/s |",
           "|---:|---:|---:|---:|---:|---:|---:|"]
    runs = defaultdict(dict)
    for d in sorted(run.glob("timing_R*_L*")):
        m = re.match(r"timing_R(\d+)_L(\d+)", d.name)
        t = load_timing(d / "stdout.txt")
        if not m or not t:
            out.append(f"| {d.name} | | | | FAILED | |")
            continue
        R, L = int(m.group(1)), int(m.group(2))
        runs[R][L] = t
        out.append(f"| {R} | {L} | {t['prompt']} | {t['gen']} | "
                   f"{t['total_ms']:.1f} | {t['per_tok_ms']:.3f} | {t['tput']:.0f} |")
    out.append("")
    out.append("Per decode iteration = (total_long - total_short) / "
               "(gen_long - gen_short), which cancels launch, prefill and setup:\n")
    for R, by_len in sorted(runs.items()):
        if len(by_len) >= 2:
            (ls, s), (ll, l) = sorted(by_len.items())[0], sorted(by_len.items())[-1]
            if l["gen"] > s["gen"] and l["total_ms"] > s["total_ms"]:
                it = (l["total_ms"] - s["total_ms"]) / (l["gen"] - s["gen"])
                out.append(f"- R={R}: **{it:.3f} ms per decode iteration** "
                           f"(~{1000 / it * R:.0f} tokens/s for the batch)")
    out.append("")
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run = Path(sys.argv[1])
    md = [f"# C2 profiling summary: {run.name}\n"]
    cfg = run / "config.txt"
    if cfg.exists():
        md += ["```", cfg.read_text().strip(), "```", ""]

    md.append("## Task durations (one profiled iteration)\n")
    all_rows = []
    for d in sorted(run.glob("profile_T*")):
        csvs = sorted(d.glob("*.csv"))
        if not csvs:
            md.append(f"### {d.name}\n\n_no CSV (run failed?)_\n")
            continue
        text, table = summarize_profile(d.name, load_profile(csvs[0]))
        md += text
        all_rows += table

    md += summarize_timing(run)

    with open(run / "c2_task_types.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "task_type", "count", "mean_us", "p50_us", "p99_us", "max_us"])
        w.writerows(all_rows)
    (run / "c2_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {run / 'c2_summary.md'} and {run / 'c2_task_types.csv'}")


if __name__ == "__main__":
    main()
