"""D2DB evaluation: rendered reference vs real frame.

Mirrors the production die-to-database procedure: the calibrated model (w,
sigma_extra) is frozen; per frame only gain/offset and registration are fitted
(exactly what a D2DB runtime would do). The detection statistic is

    z = (Y - g*render - b) / sigma_noise

with a per-pixel noise map measured from the frame's own out-of-band spectrum:
beyond the optical cutoff (0.3 cyc/px * guard) a BBP image contains only
sensor noise, so the high-pass band gives an honest, pattern-free noise floor.

With simulator truth available (noiseless clean counts + noise params), model
error is separated from noise: sigma_model = local RMS of (render - clean),
reported as the area fraction where sigma_model < sigma_noise -- the Stage
go/no-go number.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import ndimage

from . import io_utils, regions, render
from .calibrate import (_cc_shift, _robust_gain_offset_masked,
                        prepare_frame, open_cache)


def load_model(run_dir: str):
    m = np.load(os.path.join(run_dir, "model.npz"), allow_pickle=False)
    return {"w": m["w"], "kept": m["kept"], "sigma_extra": float(m["sigma_extra"]),
            "names": [str(s) for s in m["names"]]}


def noise_sigma_map(Y: np.ndarray, fmin_cyc_px: float, win: int = 11) -> np.ndarray:
    """Per-pixel sensor-noise sigma from the out-of-band spectrum."""
    n = Y.shape[0]
    f = np.fft.fftfreq(n)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    mask = np.hypot(fy, fx) > fmin_cyc_px
    hp = np.fft.ifft2(np.fft.fft2(Y) * mask).real
    med = np.median(hp)
    mad = 1.4826 * np.median(np.abs(hp - med)) + 1e-9
    hpc = np.clip(hp, med - 6 * mad, med + 6 * mad)  # defects must not inflate
    var_loc = ndimage.uniform_filter(hpc ** 2, win)
    return np.sqrt(np.maximum(var_loc, 1e-12) / mask.mean())


def fit_frame(Y, D_kept, model, cfg):
    """Runtime per-frame fit: (g, b, t) only; returns the final render."""
    n = Y.shape[0]
    mb = cfg.border_margin_px
    border = np.zeros((n, n), dtype=bool)
    border[mb:-mb, mb:-mb] = True
    t = np.zeros(2)
    g, b = 1.0, 0.0
    for it in range(3):
        Dm = render.px_ops_stack(D_kept, model["sigma_extra"], t[0], t[1])
        R = render.render(model["w"], Dm)
        g, b = _robust_gain_offset_masked(Y, R, border, cfg.clip_sigma)
        yn = (Y - b) / g
        radius = cfg.max_shift_px if it == 0 else 1
        if radius > 0:
            t += _cc_shift(yn, R, radius)
    Dm = render.px_ops_stack(D_kept, model["sigma_extra"], t[0], t[1])
    R = render.render(model["w"], Dm)
    g, b = _robust_gain_offset_masked(Y, R, border, cfg.clip_sigma)
    _, _, curv_y, curv_x = _cc_shift((Y - b) / g, R, 1, return_curv=True)
    return g, b, t, g * R + b, border, (curv_y, curv_x)


def radial_power(img: np.ndarray, nbins: int = 64):
    n = img.shape[0]
    f = np.fft.fftfreq(n)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    fr = np.hypot(fy, fx).ravel()
    p = (np.abs(np.fft.fft2(img - img.mean())) ** 2).ravel() / img.size
    edges = np.linspace(0, 0.5 * np.sqrt(2), nbins + 1)
    idx = np.digitize(fr, edges) - 1
    prof = np.zeros(nbins)
    for k in range(nbins):
        sel = idx == k
        prof[k] = p[sel].mean() if sel.any() else np.nan
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, prof


def evaluate_frames(frame_paths, run_dir, cfg, truth_dir=None, log=print):
    """Per-frame D2DB metrics; returns (rows, extras for figures)."""
    model = load_model(run_dir)
    cache_dir = os.path.join(run_dir, "cache")
    open_cache(cache_dir, cfg)
    otf_half = render.nominal_otf_hr(cfg)
    rids = regions.region_ids(len(cfg.layer_names), cfg.use_orientation)
    kept = model["kept"]
    dropped = np.setdiff1d(np.arange(len(rids)), kept)

    sim_params = None
    if truth_dir and os.path.exists(os.path.join(truth_dir, "sim_params.json")):
        with open(os.path.join(truth_dir, "sim_params.json")) as f:
            sim_params = json.load(f)

    rows, all_z, spec_resid, spec_img = [], [], [], []
    worst, fc = None, None
    for i, fp in enumerate(frame_paths):
        D = prepare_frame(fp, cfg, otf_half, cache_dir)
        Y = io_utils.load_bbp(fp, cfg.frame_px)
        g, b, t, R, border, curv = fit_frame(Y, D[kept], model, cfg)
        rare = D[dropped].sum(axis=0) if dropped.size else 0.0
        valid = border & (rare < cfg.rare_mask_thresh)

        resid = Y - R
        sigma = noise_sigma_map(Y, cfg.noise_annulus_min)
        z = resid / sigma
        zv = z[valid]
        med = np.median(resid[valid])
        mad = 1.4826 * np.median(np.abs(resid[valid] - med))
        rng = np.percentile(Y[valid], 99) - np.percentile(Y[valid], 1)
        row = {"frame_id": fp.frame_id, "g": g, "b": b,
               "dy_px": float(t[0]), "dx_px": float(t[1]),
               "reg_curv_y": float(curv[0]), "reg_curv_x": float(curv[1]),
               "robust_rms_adu": float(mad),
               "nrmse_pct": float(100.0 * np.sqrt((resid[valid] ** 2).mean())
                                  / max(rng, 1e-9)),
               "median_abs_z": float(np.median(np.abs(zv))),
               "valid_frac": float(valid.mean())}
        for tau in cfg.nuisance_taus:
            row[f"nuisance_at_{tau:g}"] = float((np.abs(zv) > tau).mean())

        if sim_params is not None:
            clean = np.load(os.path.join(truth_dir, f"{fp.frame_id}_clean.npy"))
            ped = sim_params["pedestal"]
            sig_true = np.sqrt(np.maximum(clean - ped, 0.0)
                               / sim_params["gain_e_per_adu"]
                               + sim_params["read_adu"] ** 2)
            merr = R - clean
            sig_model = np.sqrt(ndimage.uniform_filter(merr ** 2, 9))
            row["sigma_model_med_adu"] = float(np.median(sig_model[valid]))
            row["sigma_noise_med_adu"] = float(np.median(sig_true[valid]))
            row["frac_model_below_noise"] = float(
                (sig_model[valid] < sig_true[valid]).mean())

        rows.append(row)
        all_z.append(zv[:: max(1, zv.size // 20000)])
        fc, pr = radial_power(np.where(valid, resid, 0.0))
        _, pi = radial_power(np.where(valid, Y, Y[valid].mean()))
        spec_resid.append(pr)
        spec_img.append(pi)
        key = row[f"nuisance_at_{cfg.nuisance_taus[0]:g}"]
        if worst is None or key > worst[0]:
            worst = (key, fp.frame_id, Y.copy(), R.copy(), z * valid)
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(frame_paths)} frames evaluated")

    extras = {"z_sample": np.concatenate(all_z) if all_z else np.zeros(0),
              "spec_freq": fc,
              "spec_resid": np.nanmean(spec_resid, axis=0) if rows else None,
              "spec_img": np.nanmean(spec_img, axis=0) if rows else None,
              "worst": worst}
    return rows, extras


def aggregate(rows, cfg) -> dict:
    if not rows:
        return {}
    agg = {"n_frames": len(rows)}
    keys = ["robust_rms_adu", "nrmse_pct", "median_abs_z", "g", "valid_frac"]
    keys += [f"nuisance_at_{tau:g}" for tau in cfg.nuisance_taus]
    if "frac_model_below_noise" in rows[0]:
        keys += ["sigma_model_med_adu", "sigma_noise_med_adu",
                 "frac_model_below_noise"]
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        agg[f"{k}_median"] = float(np.median(vals))
        agg[f"{k}_p90"] = float(np.percentile(vals, 90))
    return agg


def make_figures(fig_dir, cfg, extras, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    z = extras["z_sample"]
    if z.size:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bins = np.linspace(-10, 10, 121)
        ax.hist(z, bins=bins, density=True, alpha=0.7, label="residual z")
        xs = np.linspace(-10, 10, 400)
        ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "k--",
                label="N(0,1)")
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1)
        ax.set_xlabel("z")
        ax.legend()
        ax.set_title("D2DB residual z (held-out)")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "z_hist.png"), dpi=130)
        plt.close(fig)

    if extras["spec_freq"] is not None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.semilogy(extras["spec_freq"], extras["spec_img"], label="image")
        ax.semilogy(extras["spec_freq"], extras["spec_resid"], label="residual")
        ax.axvline(cfg.optics.cutoff_cyc_per_px, color="k", ls=":",
                   label="optical cutoff")
        ax.set_xlabel("cyc/px")
        ax.set_ylabel("radial power")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "resid_spectrum.png"), dpi=130)
        plt.close(fig)

    taus = np.array(cfg.nuisance_taus)
    med = [np.median([r[f"nuisance_at_{tau:g}"] for r in rows]) for tau in taus]
    p90 = [np.percentile([r[f"nuisance_at_{tau:g}"] for r in rows], 90)
           for tau in taus]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogy(taus, np.maximum(med, 1e-8), "o-", label="median frame")
    ax.semilogy(taus, np.maximum(p90, 1e-8), "s--", label="p90 frame")
    ax.set_xlabel("|z| threshold")
    ax.set_ylabel("nuisance pixel fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "nuisance.png"), dpi=130)
    plt.close(fig)

    if extras["worst"] is not None:
        _, fid, Y, R, zmap = extras["worst"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        v0, v1 = np.percentile(Y, [1, 99])
        axes[0].imshow(Y, vmin=v0, vmax=v1, cmap="gray")
        axes[0].set_title(f"{fid} real")
        axes[1].imshow(R, vmin=v0, vmax=v1, cmap="gray")
        axes[1].set_title("rendered reference")
        im = axes[2].imshow(zmap, vmin=-8, vmax=8, cmap="coolwarm")
        axes[2].set_title("z = resid / sigma")
        fig.colorbar(im, ax=axes[2], shrink=0.8)
        for a in axes:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "worst_frame.png"), dpi=130)
        plt.close(fig)
