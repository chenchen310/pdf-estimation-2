#!/usr/bin/env python
"""Generate GDS-like simulated D2DB test data (see sim/simulate_gds.py)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.simulate_gds import GdsSimParams, generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-layouts", type=int, default=16)
    ap.add_argument("--dies", type=int, default=4,
                    help="dies (frames) per layout, with per-die process drift")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    p = GdsSimParams(n_layouts=args.n_layouts, dies_per_layout=args.dies,
                     seed=args.seed)
    generate(args.out, p)


if __name__ == "__main__":
    main()
