"""Score a pipeline run against simulator ground truth (honest end-to-end test:
the pipeline never sees any of these files).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from psfest.epsf import load_epsf, _fourier_shift  # noqa: E402
from psfest.noise import NoiseModel  # noqa: E402
from psfest.validate import fwhm_px, radial_profile  # noqa: E402
from psfest.config import PipelineConfig  # noqa: E402


def align_and_compare(h_est: np.ndarray, h_gt: np.ndarray):
    """Fit global (shift, scale) of GT to estimate; return NRMSE and shift."""
    def objective(delta):
        hg = _fourier_shift(h_gt, delta[0], delta[1])
        beta = (h_est * hg).sum() / max((hg * hg).sum(), 1e-12)
        return float(((h_est - beta * hg) ** 2).sum())

    res = minimize(objective, np.zeros(2), method="Nelder-Mead",
                   options={"xatol": 1e-3})
    hg = _fourier_shift(h_gt, res.x[0], res.x[1])
    beta = (h_est * hg).sum() / (hg * hg).sum()
    nrmse = float(np.linalg.norm(h_est - beta * hg) / np.linalg.norm(beta * hg))
    return nrmse, res.x, beta, hg * beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    truth_dir = os.path.join(args.data_dir, "truth")
    out = {}

    cfg = PipelineConfig.from_json(os.path.join(args.run_dir, "config.json"))
    h_est, s, R = load_epsf(os.path.join(args.run_dir, "psf.npz"))
    gt = np.load(os.path.join(truth_dir, "psf_gt.npz"))
    h_gt = gt["h_eff"]

    # --- PSF recovery ---------------------------------------------------------
    nrmse, shift, beta, h_gt_al = align_and_compare(h_est, h_gt)
    out["psf_nrmse"] = nrmse
    out["psf_align_shift_oscells"] = list(map(float, shift))
    out["fwhm_est_px"] = fwhm_px(h_est, s)
    out["fwhm_gt_px"] = fwhm_px(h_gt, s)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    v = 1.0
    axes[0].imshow(h_est, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[0].set_title("estimated h")
    axes[1].imshow(h_gt_al, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[1].set_title("ground truth (aligned)")
    im = axes[2].imshow(h_est - h_gt_al, cmap="RdBu_r", vmin=-0.05, vmax=0.05)
    axes[2].set_title(f"difference (NRMSE={nrmse:.3f})")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(args.run_dir, "figs", "eval_psf_vs_gt.png"), dpi=130)
    plt.close(fig)

    r1, p1 = radial_profile(h_est, s)
    r2, p2 = radial_profile(h_gt_al, s)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(r1, p1 / p1.max(), label="estimated")
    ax.plot(r2, p2 / p2.max(), "--", label="ground truth")
    ax.set_xlabel("radius (px)")
    ax.legend()
    ax.set_title("radial profile: estimate vs truth")
    fig.tight_layout()
    fig.savefig(os.path.join(args.run_dir, "figs", "eval_profiles.png"), dpi=130)
    plt.close(fig)

    # --- detection / event-level scores ----------------------------------------
    with open(os.path.join(truth_dir, "defects.csv")) as f:
        gt_rows = [r for r in csv.DictReader(f)]
    with open(os.path.join(args.run_dir, "events.csv")) as f:
        ev_rows = [r for r in csv.DictReader(f)]
    nm = NoiseModel.load(os.path.join(args.run_dir, "noise_model.npz"))

    sim_params = json.load(open(os.path.join(truth_dir, "sim_params.json")))
    scale, ped = sim_params["counts_scale"], sim_params["pedestal"]

    det_by_pair = {}
    for r in ev_rows:
        det_by_pair.setdefault(r["pair_id"], []).append(r)

    point_like = ("point", "two_points")
    matches, recalls = [], []
    for g in gt_rows:
        if g["type"] == "none":
            continue
        gy, gx = float(g["y_px"]), float(g["x_px"])
        peak = abs(float(g["peak_dadu"]))
        i_counts = ped + float(g["i_loc_opt"]) * scale
        snr = peak / max(float(nm.sigma_diff_at(i_counts)), 1e-6)
        best = None
        for r in det_by_pair.get(g["pair_id"], []):
            d = np.hypot(float(r["y"]) - gy, float(r["x"]) - gx)
            if d <= 4.0 and (best is None or d < best[0]):
                best = (d, r)
        recalls.append({"type": "point" if g["type"] in point_like
                        else g["type"], "snr": snr,
                        "detected": int(best is not None)})
        if best and g["type"] in point_like:
            r = best[1]
            row = {"gy": gy, "gx": gx, "peak_gt": float(g["peak_dadu"]),
                   "strength_gt": float(g["strength"]),
                   "y_est": float(r["y"]), "x_est": float(r["x"])}
            if r.get("fit_kept") == "1":
                row.update({"a_est": float(r["fit_a"]),
                            "cy": float(r["fit_cy"]), "cx": float(r["fit_cx"]),
                            "i_local": float(r["i_local"])})
            matches.append(row)

    # recall vs SNR
    bins = [(3, 5), (5, 8), (8, 12), (12, 20), (20, 1e9)]
    rec_curve = []
    for lo, hi in bins:
        sel = [m for m in recalls if m["type"] == "point" and lo <= m["snr"] < hi]
        if sel:
            rec_curve.append({"snr_bin": f"{lo}-{hi if hi < 1e9 else 'inf'}",
                              "n": len(sel),
                              "recall": float(np.mean([m["detected"]
                                                       for m in sel]))})
    out["recall_vs_snr_point"] = rec_curve
    ext = [m for m in recalls if m["type"] == "extended"]
    out["extended_detected_frac"] = float(np.mean([m["detected"] for m in ext])) \
        if ext else None

    # false positives: detections >4px from every GT defect of the pair
    gt_by_pair = {}
    for g in gt_rows:
        if g["type"] != "none":
            gt_by_pair.setdefault(g["pair_id"], []).append(
                (float(g["y_px"]), float(g["x_px"])))
    n_fp = 0
    for pid, rows in det_by_pair.items():
        for r in rows:
            pts = gt_by_pair.get(pid, [])
            if all(np.hypot(float(r["y"]) - gy, float(r["x"]) - gx) > 4.0
                   for gy, gx in pts):
                n_fp += 1
    out["false_positives"] = n_fp
    out["false_positives_per_1000_pairs"] = 1000.0 * n_fp / max(
        len({r['pair_id'] for r in ev_rows}), 1)

    # purity of the ePSF event set: every kept event must be a true point defect
    kept_rows = [r for r in ev_rows if r.get("fit_kept") == "1"]
    gt_pts_by_pair = {}
    for g in gt_rows:
        if g["type"] in point_like:
            gt_pts_by_pair.setdefault(g["pair_id"], []).append(
                (float(g["y_px"]), float(g["x_px"])))
    n_pure = sum(
        1 for r in kept_rows
        if any(np.hypot(float(r["y"]) - gy, float(r["x"]) - gx) <= 2.0
               for gy, gx in gt_pts_by_pair.get(r["pair_id"], [])))
    out["epsf_events_kept"] = len(kept_rows)
    out["epsf_event_purity"] = float(n_pure / max(len(kept_rows), 1))

    # amplitude / position / strength recovery on kept fitted events
    fit = [m for m in matches if "a_est" in m]
    if fit:
        a_est = np.array([m["a_est"] for m in fit])
        a_gt = np.array([m["peak_gt"] for m in fit])
        pos_err = np.array([np.hypot(m["y_est"] + m["cy"] - m["gy"],
                                     m["x_est"] + m["cx"] - m["gx"])
                            for m in fit])
        st_est = -a_est / np.maximum(
            np.array([m["i_local"] for m in fit]) - nm.pedestal, 1.0)
        st_gt = np.array([m["strength_gt"] for m in fit])
        out["n_fitted_matches"] = len(fit)
        out["amplitude_slope"] = float(np.sum(a_est * a_gt) / np.sum(a_gt ** 2))
        out["amplitude_scatter"] = float(np.std(a_est / a_gt))
        out["position_rmse_px"] = float(np.sqrt((pos_err ** 2).mean()))
        out["strength_slope"] = float(np.sum(st_est * st_gt) / np.sum(st_gt ** 2))

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        axes[0].scatter(a_gt, a_est, s=8, alpha=0.6)
        lim = [min(a_gt.min(), a_est.min()), 0]
        axes[0].plot(lim, lim, "k--", lw=0.8)
        axes[0].set_xlabel("GT peak (ADU)")
        axes[0].set_ylabel("fitted a (ADU)")
        axes[0].set_title("amplitude recovery")
        axes[1].hist(pos_err, bins=30)
        axes[1].set_xlabel("|position error| (px)")
        axes[1].set_title(f"position RMSE = {out['position_rmse_px']:.3f} px")
        fig.tight_layout()
        fig.savefig(os.path.join(args.run_dir, "figs", "eval_events.png"),
                    dpi=130)
        plt.close(fig)

    with open(os.path.join(args.run_dir, "eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
