#!/usr/bin/env python3
"""Tests for tools/checker.py. Run:  python3 -I -m unittest -v   (from tools/)
or:  python3 test_checker.py
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import checker
import synth
from synth import TOY_GRAPH, simulate, graph_with_steps


def cats(rep):
    return set(rep.categories())


class CleanRuns(unittest.TestCase):
    def test_single_step(self):
        rep = checker.check(simulate(TOY_GRAPH), TOY_GRAPH)
        self.assertTrue(rep.ok, rep.errors)

    def test_multi_step_barrier(self):
        g = graph_with_steps(TOY_GRAPH, 5)
        rep = checker.check(simulate(g, num_steps=5), g)
        self.assertTrue(rep.ok, rep.errors)

    def test_multi_launch_yield_resume(self):
        g = graph_with_steps(TOY_GRAPH, 5)
        recs = simulate(g, num_steps=5, launch_boundaries=(1_004_000, 1_009_000))
        self.assertGreater(len(set(recs["launch_id"].tolist())), 1)
        rep = checker.check(recs, g)
        self.assertTrue(rep.ok, rep.errors)

    def test_ties_are_accepted(self):
        recs = simulate(TOY_GRAPH, workers=1)  # back-to-back on one worker => start == end ties
        rep = checker.check(recs, TOY_GRAPH)
        self.assertTrue(rep.ok, rep.errors)
        self.assertTrue(any("tie" in m for m in rep.info))


class PlantedBugs(unittest.TestCase):
    def setUp(self):
        self.g = graph_with_steps(TOY_GRAPH, 3)
        self.clean = simulate(self.g, num_steps=3)

    def test_duplicate(self):
        rep = checker.check(synth.plant_duplicate(self.clean), self.g)
        self.assertIn("duplicate_task", cats(rep))

    def test_missing(self):
        rep = checker.check(synth.plant_missing(self.clean), self.g)
        self.assertEqual(cats(rep), {"missing_task"})

    def test_dependency_violation(self):
        rep = checker.check(synth.plant_dependency_violation(self.clean, step=1), self.g)
        self.assertIn("dependency_violation", cats(rep))

    def test_unfinished(self):
        rep = checker.check(synth.plant_unfinished(self.clean), self.g)
        self.assertEqual(cats(rep), {"unfinished_task"})

    def test_worker_overlap(self):
        rep = checker.check(synth.plant_worker_overlap(self.clean), self.g)
        self.assertIn("worker_overlap", cats(rep))

    def test_cross_step_violation(self):
        rep = checker.check(synth.plant_cross_step_violation(self.clean), self.g)
        self.assertIn("cross_step_violation", cats(rep))

    def test_unknown_task(self):
        rep = checker.check(synth.plant_unknown_task(self.clean), self.g)
        self.assertIn("unknown_task", cats(rep))

    def test_log_overflow(self):
        g = copy.deepcopy(self.g)
        g["log_count"], g["log_capacity"] = 5000, 4096
        rep = checker.check(self.clean, g)
        self.assertIn("overflow", cats(rep))

    def test_tolerance_absorbs_small_violation(self):
        bad = synth.plant_dependency_violation(self.clean)  # 100 ns early
        self.assertFalse(checker.check(bad, self.g).ok)
        self.assertTrue(checker.check(bad, self.g, tolerance_ns=200).ok)


class FileRoundTrip(unittest.TestCase):
    def test_binary_format_and_cli(self):
        with tempfile.TemporaryDirectory() as d:
            log, graph = os.path.join(d, "task_log.bin"), os.path.join(d, "task_graph.json")
            synth.simulate(TOY_GRAPH).tofile(log)
            with open(graph, "w") as f:
                json.dump(dict(TOY_GRAPH, log_count=6, log_capacity=4096), f)
            here = os.path.dirname(os.path.abspath(__file__))
            ok = subprocess.run([sys.executable, os.path.join(here, "checker.py"), log, graph],
                                capture_output=True, text=True)
            self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
            synth.plant_missing(synth.simulate(TOY_GRAPH)).tofile(log)
            bad = subprocess.run([sys.executable, os.path.join(here, "checker.py"), log, graph],
                                 capture_output=True, text=True)
            self.assertEqual(bad.returncode, 1)
            self.assertIn("missing_task", bad.stdout)


class CompareOutputs(unittest.TestCase):
    def run_cmp(self, a, b):
        here = os.path.dirname(os.path.abspath(__file__))
        with tempfile.TemporaryDirectory() as d:
            pa, pb = os.path.join(d, "a.bin"), os.path.join(d, "b.bin")
            a.tofile(pa); b.tofile(pb)
            return subprocess.run([sys.executable, os.path.join(here, "compare_outputs.py"), pa, pb],
                                  capture_output=True, text=True)

    def test_identical(self):
        x = np.arange(100, dtype=np.float32)
        self.assertEqual(self.run_cmp(x, x.copy()).returncode, 0)

    def test_one_ulp_difference_is_caught(self):
        x = np.arange(100, dtype=np.float32)
        y = x.copy()
        y.view(np.uint32)[10] += 1
        r = self.run_cmp(x, y)
        self.assertEqual(r.returncode, 1)
        self.assertIn("DIFFERENT", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
