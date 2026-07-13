"""End-to-end impulse-response estimation pipeline.

Usage:
  python scripts/run_pipeline.py --data-dir data/sim --out-dir runs/sim01 [--kfold]
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psfest.config import PipelineConfig  # noqa: E402
from psfest import io_utils, noise, detect, report  # noqa: E402
from psfest.diffimg import compute_diff  # noqa: E402
from psfest.epsf import EventCrop, run_epsf, save_epsf  # noqa: E402
from psfest.synth import fit_strength_distribution  # noqa: E402
from psfest.validate import validation_metrics, kfold_validation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--kfold", action="store_true")
    ap.add_argument("--min-events", type=int, default=None)
    args = ap.parse_args()

    cfg = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    if args.min_events:
        cfg.min_events = args.min_events
    os.makedirs(args.out_dir, exist_ok=True)
    cfg.to_json(os.path.join(args.out_dir, "config.json"))

    pairs = io_utils.find_pairs(args.data_dir)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"[1/6] {len(pairs)} pairs found", flush=True)

    # ---- pass A: signed diff + QC -------------------------------------------
    t0 = time.time()
    diffs, rms, qc_rows = {}, {}, []
    for k, p in enumerate(pairs):
        d, r = io_utils.load_pair(p)
        res = compute_diff(d, r, cfg)
        diffs[p.pair_id] = res.diff.astype(np.float32)
        rms[p.pair_id] = res.ref_matched.astype(np.float32)
        qc_rows.append({"pair_id": p.pair_id, "alpha": res.alpha,
                        "beta": res.beta, "shift_dy": res.shift_yx[0],
                        "shift_dx": res.shift_yx[1],
                        "reregistered": int(res.reregistered)})
        if (k + 1) % 100 == 0:
            print(f"  diff {k + 1}/{len(pairs)}", flush=True)
    n_rereg = sum(r["reregistered"] for r in qc_rows)
    print(f"[2/6] diffs done in {time.time() - t0:.0f}s, "
          f"{n_rereg} pairs re-registered", flush=True)

    # ---- noise model ---------------------------------------------------------
    rng = np.random.default_rng(0)
    sel_pairs = list(diffs.keys())
    rng.shuffle(sel_pairs)
    sel_pairs = sel_pairs[: cfg.noise_pairs_max]
    si, sg, sd = [], [], []
    for pid in sel_pairs:
        rm = rms[pid].astype(np.float64)
        df = diffs[pid].astype(np.float64)
        g = noise.grad_mag(rm)
        n = rm.size
        idx = rng.choice(n, size=int(n * cfg.noise_pixel_frac), replace=False)
        si.append(rm.ravel()[idx])
        sg.append(g.ravel()[idx])
        sd.append(df.ravel()[idx])
    si, sg, sd = map(np.concatenate, (si, sg, sd))
    pedestal = float(np.percentile(si, cfg.pedestal_percentile))
    nm = noise.build_noise_model(si, sg, sd, cfg, pedestal)
    nm.save(os.path.join(args.out_dir, "noise_model.npz"))
    print(f"[3/6] noise model built (pedestal ~ {pedestal:.0f} ADU)", flush=True)

    # ---- detection + screening ----------------------------------------------
    R = cfg.window_radius()
    all_events, crops, clean_ids = [], [], []
    hot_map = {}
    reject_gallery = []
    for k, p in enumerate(pairs):
        pid = p.pair_id
        diff = diffs[pid].astype(np.float64)
        rm = rms[pid].astype(np.float64)
        def_img = np.load(p.def_path).astype(np.float64)
        sigma = nm.sigma_map(rm)
        events, is_clean, hot = detect.detect_pair(pid, diff, rm, def_img,
                                                   sigma, cfg)
        all_events.extend(events)
        if is_clean:
            clean_ids.append(pid)
            hot_map[pid] = hot
        for ev in events:
            if ev["is_psf_candidate"]:
                y, x = ev["y"], ev["x"]
                crops.append(EventCrop(
                    pair_id=pid, y=y, x=x,
                    data=diff[y - R: y + R + 1, x - R: x + R + 1].copy(),
                    sigma=sigma[y - R: y + R + 1, x - R: x + R + 1].copy(),
                    sign=ev["sign"], i_local=ev["i_local"]))
            elif ev["reject_reason"] not in ("below_tau_psf", "wrong_polarity") \
                    and len(reject_gallery) < 16:
                y, x = ev["y"], ev["x"]
                reject_gallery.append(
                    (diff[y - R: y + R + 1, x - R: x + R + 1].copy(),
                     ev["reject_reason"]))
        if (k + 1) % 100 == 0:
            print(f"  detect {k + 1}/{len(pairs)}", flush=True)
    n_det = len(all_events)
    print(f"[4/6] {n_det} detections, {len(crops)} PSF candidates, "
          f"{len(clean_ids)} clean pairs", flush=True)

    # ---- ePSF ---------------------------------------------------------------
    t0 = time.time()
    res = run_epsf(crops, cfg)
    print(f"[5/6] ePSF converged in {time.time() - t0:.0f}s: "
          f"{res.history[-1]}", flush=True)
    save_epsf(os.path.join(args.out_dir, "psf.npz"), res, cfg)

    # merge fit results into the event table
    fitmap = {(ev.pair_id, ev.y, ev.x): ev for ev in res.events}
    for row in all_events:
        key = (row["pair_id"], row["y"], row["x"])
        if key in fitmap:
            ev = fitmap[key]
            row.update({"fit_a": ev.a, "fit_b": ev.b,
                        "fit_cy": ev.cy, "fit_cx": ev.cx,
                        "fit_chi2": ev.chi2, "fit_rho": ev.rho,
                        "fit_kept": int(ev.kept),
                        "fit_drop_reason": ev.drop_reason})

    # ---- validation + report -------------------------------------------------
    metrics = validation_metrics(res.h, cfg, res.events, h_raw=res.h_raw)
    metrics.update({"n_pairs": len(pairs), "n_detections": n_det,
                    "n_psf_candidates": len(crops),
                    "n_clean_pairs": len(clean_ids),
                    "n_reregistered": n_rereg,
                    "epsf_history": res.history})
    if args.kfold:
        print("  running k-fold validation ...", flush=True)
        metrics["kfold"] = kfold_validation(res.events, cfg, k=5)

    strength_dist = fit_strength_distribution(res.events, nm.pedestal)
    with open(os.path.join(args.out_dir, "run_summary.json"), "w") as f:
        json.dump({"clean_ids": clean_ids, "strength_dist": strength_dist,
                   "data_dir": os.path.abspath(args.data_dir)}, f, indent=2)
    np.savez(os.path.join(args.out_dir, "hot_map.npz"),
             **{pid: hot_map[pid] for pid in clean_ids})

    keys = sorted({k for row in all_events for k in row})
    with open(os.path.join(args.out_dir, "events.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_events)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    fig_dir = os.path.join(args.out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    report.save_noise_plot(nm, os.path.join(fig_dir, "noise_model.png"))
    report.save_qc_plot(qc_rows, os.path.join(fig_dir, "diff_qc.png"))
    report.save_psf_figs(res.h, cfg, fig_dir)
    report.save_gallery(res.events, res.h, cfg,
                        os.path.join(fig_dir, "event_gallery.png"))
    report.save_reject_gallery(reject_gallery,
                               os.path.join(fig_dir, "reject_gallery.png"))
    report.save_amplitude_plot(res.events,
                               os.path.join(fig_dir, "amplitude_vs_bg.png"))
    report.save_chi2_rho_plot(res.events,
                              os.path.join(fig_dir, "chi2_rho.png"))
    print(f"[6/6] outputs in {args.out_dir}")
    print(json.dumps({k: v for k, v in metrics.items()
                      if not isinstance(v, (list, dict))}, indent=2))


if __name__ == "__main__":
    main()
