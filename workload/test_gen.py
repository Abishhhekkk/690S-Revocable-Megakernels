#!/usr/bin/env python3
"""Tests for workload/gen.py (stdlib only):  python workload/test_gen.py"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen  # noqa: E402


def cls_cfg(arrival, **kw):
    c = {"arrival": arrival, "prompt_len": {"dist": "fixed", "value": 10},
         "output_len": {"dist": "fixed", "value": 5}}
    c.update(kw)
    return c


def run(classes, T=1000.0, seed=1, **kw):
    cfg = {"seed": seed, "duration_s": T, "classes": classes, **kw}
    return gen.generate(cfg)[1]


def cv(ts):
    g = [b - a for a, b in zip(ts, ts[1:])]
    m = sum(g) / len(g)
    return math.sqrt(sum((x - m) ** 2 for x in g) / (len(g) - 1)) / m


def test_poisson_rate_and_cv():
    r = run({"bg": cls_cfg({"type": "poisson", "rate": 5.0})})
    ts = [x["arrival_s"] for x in r]
    assert abs(len(r) / 1000 - 5.0) < 0.3, len(r)
    assert 0.9 < cv(ts) < 1.1, cv(ts)
    assert all(0 <= t < 1000 for t in ts)


def test_bursty_is_burstier_than_poisson():
    r = run({"u": cls_cfg({"type": "bursty", "rate_on": 20, "rate_off": 0.1,
                           "mean_on_s": 1, "mean_off_s": 9})})
    ts = [x["arrival_s"] for x in r]
    # long-run rate = (20*1 + 0.1*9) / 10 = 2.09/s
    assert 1.6 < len(r) / 1000 < 2.6, len(r)
    assert cv(ts) > 1.5, cv(ts)


def test_gamma_cv():
    r = run({"u": cls_cfg({"type": "gamma", "rate": 4, "cv": 3})}, T=3000)
    ts = [x["arrival_s"] for x in r]
    assert abs(len(r) / 3000 - 4) < 0.6, len(r)
    assert 2.3 < cv(ts) < 3.7, cv(ts)


def test_random_count_and_gap():
    r = run({"u": cls_cfg({"type": "random", "count": 50, "min_gap_s": 2.0})}, T=200)
    ts = [x["arrival_s"] for x in r]
    assert len(r) == 50
    assert min(b - a for a, b in zip(ts, ts[1:])) >= 2.0 - 1e-6
    assert ts[-1] < 200


def test_batch_and_periodic():
    r = run({"bg": cls_cfg({"type": "batch", "count": 8}),
             "p": cls_cfg({"type": "periodic", "interval_s": 0.5})}, T=10)
    assert sum(x["cls"] == "bg" and x["arrival_s"] == 0 for x in r) == 8
    assert sum(x["cls"] == "p" for x in r) == 20


def test_deterministic_and_class_independent():
    a = run({"bg": cls_cfg({"type": "poisson", "rate": 3})}, seed=7)
    b = run({"bg": cls_cfg({"type": "poisson", "rate": 3})}, seed=7)
    c = run({"bg": cls_cfg({"type": "poisson", "rate": 3}),
             "u": cls_cfg({"type": "poisson", "rate": 1})}, seed=7)
    d = run({"bg": cls_cfg({"type": "poisson", "rate": 3})}, seed=8)
    key = lambda rs: [x["arrival_s"] for x in rs if x["cls"] == "bg"]
    assert key(a) == key(b) == key(c)
    assert key(a) != key(d)


def test_lengths_and_clamp():
    r = run({"bg": {"arrival": {"type": "batch", "count": 500},
                    "prompt_len": {"dist": "lognormal", "median": 300, "sigma": 1, "lo": 4, "hi": 1000},
                    "output_len": {"dist": "uniform", "lo": 100, "hi": 400}}},
            T=1, max_seq_len=512)
    assert all(x["prompt_len"] + x["output_len"] <= 512 for x in r)
    assert all(x["prompt_len"] >= 4 and x["output_len"] >= 1 for x in r)
    ch = run({"bg": {"arrival": {"type": "batch", "count": 300},
                     "prompt_len": {"dist": "choice", "values": [1, 2], "weights": [1, 0]}}}, T=1)
    assert {x["prompt_len"] for x in ch} == {1}


def test_ordering_ids_priority():
    r = run({"bg": cls_cfg({"type": "batch", "count": 2}, priority=1),
             "urgent": cls_cfg({"type": "batch", "count": 1}, priority=0, deadline_ms=50)}, T=1)
    assert [x["id"] for x in r] == [0, 1, 2]
    assert r[0]["cls"] == "urgent" and r[0]["deadline_ms"] == 50  # ties: urgent first


def test_trace_replay_and_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        tr = Path(d) / "t.csv"
        tr.write_text("TIMESTAMP,ContextTokens,GeneratedTokens\n"
                      "2023-11-16 18:15:46.6805900,374,44\n"
                      "2023-11-16 18:15:50.9951690,396,109\n"
                      "2023-11-16 18:15:47.0000000,879,7\n"
                      "2023-11-16 18:17:00.0000000,10,10\n")
        r = run({"bg": {"arrival": {"type": "trace", "path": str(tr), "speedup": 2.0}}}, T=10)
        assert [x["prompt_len"] for x in r] == [374, 879, 396], r   # sorted, 4th is past T
        assert r[0]["arrival_s"] == 0.0
        assert abs(r[2]["arrival_s"] - (50.995169 - 46.68059) / 2) < 1e-3
        meta, reqs = gen.generate({"seed": 0, "duration_s": 10, "classes": {
            "bg": {"arrival": {"type": "trace", "path": str(tr)}}}})
        out = Path(d) / "w.jsonl"
        gen.write_workload(out, meta, reqs)
        m2, r2 = gen.load_workload(out)
        assert m2["seed"] == 0 and [x["id"] for x in r2] == [x["id"] for x in reqs]


def test_presets_load():
    for p in sorted(gen.PRESET_DIR.glob("*.json")):
        cfg = json.loads(p.read_text())
        if any(c["arrival"]["type"] == "trace" for c in cfg["classes"].values()):
            continue  # needs the downloaded trace
        meta, reqs = gen.generate(cfg)
        assert reqs, p.name
        assert {"urgent", "background"} <= {x["cls"] for x in reqs}, p.name


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
