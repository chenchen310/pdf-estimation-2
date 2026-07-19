#!/usr/bin/env python
"""Calibrate the Stage-0 GDS -> BBP model on a data directory.

Works identically on simulated and real (converted) fab data; truth/ is only
used for the train/test split bookkeeping (layout ids), never for fitting.

Run-dir contract: config.json, model.npz, perframe.csv, split.json,
run_summary.json, cache/ (region densities, safe to delete).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from d2db import io_utils  # noqa: E402
from d2db.calibrate import calibrate  # noqa: E402
from d2db.config import D2DBConfig  # noqa: E402


def load_meta(data_dir):
    path = os.path.join(data_dir, "truth", "meta.csv")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["frame_id"]: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", help="D2DBConfig JSON (default: built-ins)")
    ap.add_argument("--layers", nargs="*",
                    help="layer names (default: auto-discover from files)")
    ap.add_argument("--split", choices=("die", "layout", "frame"),
                    default="die",
                    help="holdout grouping; 'layout' needs truth/meta.csv")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-fit-frames", type=int)
    args = ap.parse_args()

    cfg = D2DBConfig.from_json(args.config) if args.config else D2DBConfig()
    if args.max_fit_frames:
        cfg.fit_frames_max = args.max_fit_frames
    if args.layers:
        cfg.layer_names = tuple(args.layers)
    else:
        found = io_utils.discover_layer_names(args.data_dir)
        if not found:
            raise SystemExit(f"no frames found in {args.data_dir}")
        cfg.layer_names = tuple(found)
    print(f"layers: {cfg.layer_names}")

    frames = io_utils.find_frames(args.data_dir, cfg.layer_names)
    if not frames:
        raise SystemExit("no complete frames (bbp + all layers) found")
    print(f"{len(frames)} frames discovered")

    meta = load_meta(args.data_dir)
    if args.split == "layout" and not meta:
        raise SystemExit("--split layout needs truth/meta.csv with layout_id")

    def group_key(fid):
        if args.split == "layout":
            return meta[fid]["layout_id"]
        if args.split == "die" and fid in meta:
            return meta[fid].get("die_id", fid)
        return fid

    groups = sorted({group_key(f.frame_id) for f in frames})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(groups)
    n_train = max(1, int(round(args.train_frac * len(groups))))
    train_groups = set(groups[:n_train])
    train = [f for f in frames if group_key(f.frame_id) in train_groups]
    test = [f for f in frames if group_key(f.frame_id) not in train_groups]
    print(f"split '{args.split}': {len(train)} train / {len(test)} test frames")

    os.makedirs(args.out_dir, exist_ok=True)
    cfg.to_json(os.path.join(args.out_dir, "config.json"))
    with open(os.path.join(args.out_dir, "split.json"), "w") as f:
        json.dump({"split": args.split, "seed": args.seed,
                   "train": [x.frame_id for x in train],
                   "test": [x.frame_id for x in test]}, f, indent=2)

    summary = calibrate(train, cfg, args.out_dir)
    summary["split"] = args.split
    with open(os.path.join(args.out_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"run dir written: {args.out_dir}")


if __name__ == "__main__":
    main()
