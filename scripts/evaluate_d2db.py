#!/usr/bin/env python
"""Evaluate a D2DB run: held-out frames scored against the frozen model.

Reports the production-relevant numbers (residual z, nuisance rate vs
threshold) from the data alone; when simulator truth is present it also
separates model error from sensor noise (sigma_model / sigma_noise) and
cross-checks the fitted registration against the injected offsets.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from d2db import io_utils  # noqa: E402
from d2db.config import D2DBConfig  # noqa: E402
from d2db.evald2db import aggregate, evaluate_frames, make_figures  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--set", choices=("test", "train", "all"), default="test")
    args = ap.parse_args()

    cfg = D2DBConfig.from_json(os.path.join(args.run_dir, "config.json"))
    with open(os.path.join(args.run_dir, "split.json")) as f:
        split = json.load(f)
    wanted = set(split["train"] if args.set == "train" else
                 split["test"] if args.set == "test" else
                 split["train"] + split["test"])
    frames = [f for f in io_utils.find_frames(args.data_dir, cfg.layer_names)
              if f.frame_id in wanted]
    if not frames:
        raise SystemExit(f"no frames for set '{args.set}'")
    print(f"evaluating {len(frames)} '{args.set}' frames "
          f"(split: {split['split']})")

    truth_dir = os.path.join(args.data_dir, "truth")
    truth_dir = truth_dir if os.path.isdir(truth_dir) else None
    rows, extras = evaluate_frames(frames, args.run_dir, cfg,
                                   truth_dir=truth_dir)
    agg = aggregate(rows, cfg)

    # registration vs injected truth, counting only axes the pattern actually
    # constrains (CC peak curvature): errors along invariant directions of
    # line/space layouts are arbitrary and harmless by construction
    reg_check = None
    meta_path = os.path.join(truth_dir, "meta.csv") if truth_dir else None
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = {r["frame_id"]: r for r in csv.DictReader(f)}
        errs = []
        for r in rows:
            if r["frame_id"] not in meta:
                continue
            m = meta[r["frame_id"]]
            if r["reg_curv_y"] > 0.05:
                errs.append(abs(r["dy_px"] - float(m["expected_dy_px"])))
            if r["reg_curv_x"] > 0.05:
                errs.append(abs(r["dx_px"] - float(m["expected_dx_px"])))
        if errs:
            reg_check = {"n_constrained_axes": len(errs),
                         "median_reg_err_px": float(np.median(errs)),
                         "p90_reg_err_px": float(np.percentile(errs, 90))}

    out = {"set": args.set, "aggregate": agg,
           "registration_check": reg_check, "frames": rows}
    with open(os.path.join(args.run_dir, f"evaluation_{args.set}.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    make_figures(os.path.join(args.run_dir, "figs"), cfg, extras, rows)

    print(f"median robust RMS   : {agg['robust_rms_adu_median']:.2f} ADU")
    print(f"median NRMSE        : {agg['nrmse_pct_median']:.2f} %")
    print(f"median |z|          : {agg['median_abs_z_median']:.3f}")
    for tau in cfg.nuisance_taus:
        print(f"nuisance @ |z|>{tau:g}   : "
              f"{agg[f'nuisance_at_{tau:g}_median']:.2e} (median frame)")
    if "frac_model_below_noise_median" in agg:
        print(f"sigma_model<noise   : "
              f"{100 * agg['frac_model_below_noise_median']:.1f} % of area "
              f"(median frame)")
    if reg_check:
        print(f"registration error  : {reg_check['median_reg_err_px']:.3f} px "
              f"(median vs truth, {reg_check['n_constrained_axes']} "
              f"constrained axes)")
    print(f"figures + evaluation_{args.set}.json in {args.run_dir}")


if __name__ == "__main__":
    main()
