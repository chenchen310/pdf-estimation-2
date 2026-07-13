"""Generate synthetic-defect U-Net training data from a completed pipeline run.

Usage:
  python scripts/run_synth.py --run-dir runs/sim01 --out-dir data/synth --n 2000
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psfest.config import PipelineConfig  # noqa: E402
from psfest import io_utils  # noqa: E402
from psfest.epsf import load_epsf  # noqa: E402
from psfest.noise import NoiseModel  # noqa: E402
from psfest.synth import synthesize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--negative-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = PipelineConfig.from_json(os.path.join(args.run_dir, "config.json"))
    h, s, R = load_epsf(os.path.join(args.run_dir, "psf.npz"))
    nm = NoiseModel.load(os.path.join(args.run_dir, "noise_model.npz"))
    with open(os.path.join(args.run_dir, "run_summary.json")) as f:
        summary = json.load(f)
    hot_npz = np.load(os.path.join(args.run_dir, "hot_map.npz"))
    hot_by_id = {k: hot_npz[k] for k in hot_npz.files}

    pairs = {p.pair_id: p for p in io_utils.find_pairs(summary["data_dir"])}
    clean_ids = [pid for pid in summary["clean_ids"] if pid in pairs]
    if not clean_ids:
        raise RuntimeError("no clean pairs available for synthesis")

    rows = synthesize(pairs, clean_ids, hot_by_id, h, nm,
                      summary["strength_dist"], cfg, args.out_dir, args.n,
                      negative_frac=args.negative_frac, seed=args.seed,
                      load_pair_fn=io_utils.load_pair)
    n_pos = sum(r["has_defect"] for r in rows)
    print(f"wrote {len(rows)} samples ({n_pos} with defects) to {args.out_dir}")

    # visual QA: synthetic sample gallery + real-vs-synthetic comparison
    from psfest import report
    from psfest.diffimg import compute_diff
    import csv as _csv
    R = cfg.window_radius()
    report.save_synth_gallery(args.out_dir, rows, R,
                              os.path.join(args.out_dir, "synth_gallery.png"))
    real_crops = []
    with open(os.path.join(args.run_dir, "events.csv")) as f:
        evs = sorted((r for r in _csv.DictReader(f) if r.get("fit_kept") == "1"),
                     key=lambda r: abs(float(r["fit_a"])))
    rng = np.random.default_rng(1)
    for r in [evs[i] for i in
              sorted(rng.choice(len(evs), size=min(8, len(evs)),
                                replace=False))]:
        p = pairs[r["pair_id"]]
        d, rf = io_utils.load_pair(p)
        dres = compute_diff(d, rf, cfg)
        y, x = int(r["y"]), int(r["x"])
        real_crops.append(dres.diff[y - R: y + R + 1, x - R: x + R + 1])
    # match each shown real event to the synthetic sample closest in |a|
    real_amps = [abs(float(r["fit_a"])) for r in
                 [evs[i] for i in sorted(rng.choice(len(evs),
                  size=min(8, len(evs)), replace=False))]]
    cand = sorted((r for r in rows if r["has_defect"]),
                  key=lambda r: abs(r["a"]))
    synth_crops = []
    used = set()
    for ra in sorted(real_amps):
        pick = min((r for r in cand if r["index"] not in used),
                   key=lambda r: abs(abs(r["a"]) - ra))
        used.add(pick["index"])
        dif = np.load(os.path.join(args.out_dir,
                                   f"SYN_{pick['index']:05d}_diff.npy"))
        y, x = int(round(pick["y"])), int(round(pick["x"]))
        synth_crops.append(dif[y - R: y + R + 1, x - R: x + R + 1])
    real_crops.sort(key=lambda c: abs(c.min()))
    report.save_real_vs_synth(real_crops, synth_crops,
                              os.path.join(args.out_dir, "real_vs_synth.png"))


if __name__ == "__main__":
    main()
