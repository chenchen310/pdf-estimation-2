"""Generate BBP-like test data: DID_{index}_def.npy / DID_{index}_ref1.npy."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.simulate import SimParams, generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=650)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    p = SimParams(n_pairs=args.n, seed=args.seed)
    generate(args.out, p)


if __name__ == "__main__":
    main()
