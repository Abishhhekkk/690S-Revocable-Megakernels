#!/usr/bin/env python3
"""C3: configurable workload generator for urgent + background LLM requests.

Usage:
    python workload/gen.py workload/configs/poisson.json -o results/wl/poisson.jsonl
    python workload/gen.py workload/configs/bursty.json --seed 3 --duration 30 -o out.jsonl
    python workload/gen.py --list-presets

Output: one JSON object per line, sorted by arrival time:
    {"id": 17, "cls": "urgent", "arrival_s": 1.2345, "prompt_len": 64,
     "output_len": 16, "deadline_ms": 50.0, "priority": 0}
A first line {"_meta": {...}} records the config and seed so every sweep is
reproducible from the file alone. Read files back with load_workload().

Config (JSON):
{
  "seed": 0,
  "duration_s": 60,            # arrivals are generated in [0, duration_s)
  "max_seq_len": 512,          # prompt_len + output_len is clamped to this
  "classes": {
    "<name>": {
      "priority": 0,           # lower = more urgent (0 = urgent)
      "deadline_ms": 50,       # TTFT target; null for best-effort background
      "arrival":    {...},     # see ARRIVALS below
      "prompt_len": {...},     # see LENGTHS below
      "output_len": {...}
    }, ...
  }
}

ARRIVALS
  {"type": "poisson", "rate": 2.0}                      rate in requests/s
  {"type": "bursty", "rate_on": 20, "rate_off": 0.5,    on/off Markov-modulated
   "mean_on_s": 1.0, "mean_off_s": 4.0}                 Poisson (MMPP-2)
  {"type": "gamma", "rate": 2.0, "cv": 3.0}             renewal process; cv > 1 is bursty
  {"type": "random", "count": 20, "min_gap_s": 0.0}     `count` arrivals uniform in the window
  {"type": "periodic", "interval_s": 0.5, "jitter_s": 0}
  {"type": "batch", "count": 8, "at_s": 0.0}            `count` requests at once (steady
                                                        background decode / large batch)
  {"type": "trace", "path": "trace.csv",                replay a public trace (CSV);
   "time_col": "TIMESTAMP", "speedup": 1.0,             Azure LLM inference trace columns
   "prompt_col": "ContextTokens",                       by default; timestamps may be
   "output_col": "GeneratedTokens",                     seconds or ISO-8601 datetimes
   "offset_s": 0, "max_requests": null}

LENGTHS (tokens; a trace class may omit them to use the trace's columns)
  {"dist": "fixed", "value": 128}
  {"dist": "uniform", "lo": 32, "hi": 256}              inclusive
  {"dist": "lognormal", "median": 128, "sigma": 0.8, "lo": 1, "hi": 4096}
  {"dist": "choice", "values": [64, 256, 1024], "weights": [0.5, 0.3, 0.2]}

Only the standard library is used, so this runs anywhere (login node is fine).
"""
import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

PRESET_DIR = Path(__file__).resolve().parent / "configs"


# ---------------------------------------------------------------- arrivals --
def arr_poisson(rng, spec, T):
    rate, t, out = float(spec["rate"]), 0.0, []
    if rate <= 0:
        return out
    while True:
        t += rng.expovariate(rate)
        if t >= T:
            return out
        out.append(t)


def arr_bursty(rng, spec, T):
    """Two-state MMPP: exponential on/off sojourns, Poisson within each."""
    r_on, r_off = float(spec["rate_on"]), float(spec.get("rate_off", 0.0))
    m_on, m_off = float(spec["mean_on_s"]), float(spec["mean_off_s"])
    on = spec.get("start_on", rng.random() < m_on / (m_on + m_off))
    t, out = 0.0, []
    while t < T:
        end = min(T, t + rng.expovariate(1.0 / (m_on if on else m_off)))
        rate = r_on if on else r_off
        s = t
        while rate > 0:
            s += rng.expovariate(rate)
            if s >= end:
                break
            out.append(s)
        t, on = end, not on
    return out


def arr_gamma(rng, spec, T):
    """Gamma renewal process with mean 1/rate and coefficient of variation cv."""
    rate, cv = float(spec["rate"]), float(spec.get("cv", 1.0))
    k = 1.0 / (cv * cv)               # shape
    theta = 1.0 / (rate * k)          # scale, so mean gap = k * theta = 1/rate
    t, out = 0.0, []
    while True:
        t += rng.gammavariate(k, theta)
        if t >= T:
            return out
        out.append(t)


def arr_random(rng, spec, T):
    n, gap = int(spec["count"]), float(spec.get("min_gap_s", 0.0))
    if gap <= 0:
        return sorted(rng.uniform(0, T) for _ in range(n))
    # uniform order statistics on the reduced window, then re-insert gaps
    span = T - gap * (n - 1)
    if span <= 0:
        raise ValueError(f"random: {n} arrivals with min_gap_s={gap} do not fit in {T}s")
    base = sorted(rng.uniform(0, span) for _ in range(n))
    return [b + i * gap for i, b in enumerate(base)]


def arr_periodic(rng, spec, T):
    dt, jit = float(spec["interval_s"]), float(spec.get("jitter_s", 0.0))
    t0 = float(spec.get("start_s", 0.0))
    out, i = [], 0
    while t0 + i * dt < T:
        t = t0 + i * dt + (rng.uniform(-jit, jit) if jit else 0.0)
        if 0 <= t < T:
            out.append(t)
        i += 1
    return sorted(out)


def arr_batch(rng, spec, T):
    return [float(spec.get("at_s", 0.0))] * int(spec["count"])


def _parse_time(v):
    try:
        return float(v)
    except ValueError:
        v = v.strip().replace("Z", "+00:00")
        # Azure traces have 7 fractional digits; datetime takes at most 6
        if "." in v:
            head, frac = v.split(".", 1)
            digits = "".join(ch for ch in frac if ch.isdigit())
            rest = frac[len(digits):]
            v = f"{head}.{digits[:6]}{rest}"
        return datetime.fromisoformat(v).timestamp()


def arr_trace(rng, spec, T):
    """Returns (times, [(prompt, output) or None])."""
    path = Path(spec["path"]).expanduser()
    if not path.is_absolute() and not path.exists():
        path = PRESET_DIR.parent.parent / path   # relative to the repo root
    tcol = spec.get("time_col", "TIMESTAMP")
    pcol = spec.get("prompt_col", "ContextTokens")
    ocol = spec.get("output_col", "GeneratedTokens")
    speed = float(spec.get("speedup", 1.0))
    offset = float(spec.get("offset_s", 0.0))
    cap = spec.get("max_requests")
    raw = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            t = _parse_time(row[tcol])
            lens = None
            if pcol in row and ocol in row and row[pcol] and row[ocol]:
                lens = (int(float(row[pcol])), int(float(row[ocol])))
            raw.append((t, lens))
    if not raw:
        return [], []
    raw.sort(key=lambda x: x[0])
    t0 = raw[0][0] + offset
    times, lens = [], []
    for t, l in raw:
        rel = (t - t0) / speed
        if rel < 0:
            continue
        if rel >= T or (cap is not None and len(times) >= cap):
            break
        times.append(rel)
        lens.append(l)
    return times, lens


ARRIVALS = {"poisson": arr_poisson, "bursty": arr_bursty, "gamma": arr_gamma,
            "random": arr_random, "periodic": arr_periodic, "batch": arr_batch}


# ----------------------------------------------------------------- lengths --
def sample_len(rng, spec):
    d = spec.get("dist", "fixed")
    if d == "fixed":
        v = spec["value"]
    elif d == "uniform":
        v = rng.randint(int(spec["lo"]), int(spec["hi"]))
    elif d == "lognormal":
        v = round(rng.lognormvariate(math.log(spec["median"]), spec["sigma"]))
    elif d == "choice":
        v = rng.choices(spec["values"], weights=spec.get("weights"))[0]
    else:
        raise ValueError(f"unknown length dist {d!r}")
    lo, hi = spec.get("lo", 1), spec.get("hi")
    v = max(int(lo), int(v))
    return min(int(hi), v) if hi is not None else v


# ---------------------------------------------------------------- generate --
def generate(cfg, seed=None, duration=None):
    seed = cfg.get("seed", 0) if seed is None else seed
    T = float(cfg.get("duration_s", 60) if duration is None else duration)
    max_seq = cfg.get("max_seq_len")
    reqs = []
    # one independent RNG stream per class, so adding a class does not
    # perturb the arrivals of the others
    for ci, (name, c) in enumerate(sorted(cfg["classes"].items())):
        rng = random.Random(f"{seed}:{name}")
        a = c["arrival"]
        if a["type"] == "trace":
            times, tlens = arr_trace(rng, a, T)
        else:
            times = ARRIVALS[a["type"]](rng, a, T)
            tlens = [None] * len(times)
        for t, tl in zip(times, tlens):
            if "prompt_len" in c or tl is None:
                p = sample_len(rng, c.get("prompt_len", {"dist": "fixed", "value": 128}))
                o = sample_len(rng, c.get("output_len", {"dist": "fixed", "value": 128}))
            else:
                p, o = tl
            p, o = max(1, p), max(1, o)
            if max_seq:
                p = min(p, max_seq - 1)
                o = min(o, max_seq - p)
            reqs.append({"cls": name, "arrival_s": round(t, 6), "prompt_len": p,
                         "output_len": o, "deadline_ms": c.get("deadline_ms"),
                         "priority": c.get("priority", ci)})
    reqs.sort(key=lambda r: (r["arrival_s"], r["priority"], r["cls"]))
    for i, r in enumerate(reqs):
        r["id"] = i
    meta = {"seed": seed, "duration_s": T, "config": cfg,
            "generated": datetime.now().isoformat(timespec="seconds")}
    return meta, reqs


def write_workload(path, meta, reqs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for r in reqs:
            f.write(json.dumps({k: r[k] for k in ("id", "cls", "arrival_s", "prompt_len",
                                                  "output_len", "deadline_ms", "priority")})
                    + "\n")


def load_workload(path):
    """Returns (meta, requests) from a file written by write_workload."""
    meta, reqs = {}, []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "_meta" in d:
                meta = d["_meta"]
            else:
                reqs.append(d)
    return meta, reqs


def summarize(meta, reqs):
    T = meta["duration_s"]
    lines = [f"seed={meta['seed']} duration={T}s total={len(reqs)} requests"]
    for cls in sorted({r["cls"] for r in reqs}):
        rs = [r for r in reqs if r["cls"] == cls]
        ts = [r["arrival_s"] for r in rs]
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        cv = "n/a"
        if len(gaps) > 1:
            m = sum(gaps) / len(gaps)
            if m > 0:
                sd = math.sqrt(sum((g - m) ** 2 for g in gaps) / (len(gaps) - 1))
                cv = f"{sd / m:.2f}"
        pl = sorted(r["prompt_len"] for r in rs)
        ol = sorted(r["output_len"] for r in rs)
        lines.append(
            f"  {cls:<12} n={len(rs):<5} rate={len(rs) / T:.2f}/s gap_cv={cv:<5} "
            f"prompt p50={pl[len(pl) // 2]} max={pl[-1]}  "
            f"output p50={ol[len(ol) // 2]} max={ol[-1]}  "
            f"deadline_ms={rs[0]['deadline_ms']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", nargs="?", help="JSON config, or a preset name from workload/configs")
    ap.add_argument("-o", "--out", help="output JSONL (default: print summary only)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--list-presets", action="store_true")
    a = ap.parse_args(argv)
    if a.list_presets:
        for p in sorted(PRESET_DIR.glob("*.json")):
            print(p.stem)
        return
    if not a.config:
        ap.error("config is required")
    path = Path(a.config)
    if not path.exists():
        path = PRESET_DIR / f"{a.config}.json"
    cfg = json.loads(path.read_text())
    meta, reqs = generate(cfg, a.seed, a.duration)
    meta["config_path"] = str(path)
    if a.out:
        write_workload(a.out, meta, reqs)
        print(f"wrote {len(reqs)} requests to {a.out}")
    print(summarize(meta, reqs))


if __name__ == "__main__":
    main()
