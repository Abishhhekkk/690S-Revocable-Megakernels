#!/usr/bin/env python3
"""Bit-for-bit output comparison (Part B1).

Step 1 (do this BEFORE any yield testing): dump the output of an uninterrupted
decode twice and compare run1 vs run2 with this script. If they differ, the
decode is not deterministic and exact comparison is meaningless until you add a
deterministic mode (fixed reduction order, no atomic-order-dependent
accumulation) or agree on a tolerance.

Step 2: compare the yielded/resumed run against the uninterrupted reference.

Usage:
    python3 compare_outputs.py a.bin b.bin
    python3 compare_outputs.py a.bin b.bin --dtype float32 --atol 1e-5   # only if you must
Exit code 0 = identical, 1 = different.
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--dtype", default=None,
                    help="interpret files as this numpy dtype for diagnostics (e.g. float32)")
    ap.add_argument("--atol", type=float, default=None,
                    help="also report whether outputs match within this absolute tolerance")
    args = ap.parse_args()

    a = np.fromfile(args.a, dtype=np.uint8)
    b = np.fromfile(args.b, dtype=np.uint8)

    if a.size != b.size:
        print(f"DIFFERENT: sizes differ ({a.size} vs {b.size} bytes)")
        sys.exit(1)

    diff = np.nonzero(a != b)[0]
    if diff.size == 0:
        print(f"IDENTICAL: {a.size} bytes bit-for-bit equal")
        sys.exit(0)

    print(f"DIFFERENT: {diff.size} of {a.size} bytes differ; first at byte offset {diff[0]}")
    if args.dtype:
        dt = np.dtype(args.dtype)
        n = a.size // dt.itemsize
        x = np.frombuffer(a[: n * dt.itemsize].tobytes(), dtype=dt)
        y = np.frombuffer(b[: n * dt.itemsize].tobytes(), dtype=dt)
        nd = np.nonzero(x.view(np.uint8).reshape(n, -1) != y.view(np.uint8).reshape(n, -1))[0]
        nd = np.unique(nd)
        print(f"  {nd.size} of {n} {args.dtype} elements differ; first at element {nd[0]}")
        if np.issubdtype(dt, np.floating):
            err = np.abs(x.astype(np.float64) - y.astype(np.float64))
            print(f"  max abs diff = {np.nanmax(err):.3e}")
            if args.atol is not None:
                print(f"  within atol={args.atol}: {bool(np.all(err <= args.atol))}")
    sys.exit(1)


if __name__ == "__main__":
    main()
