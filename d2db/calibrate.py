"""Stage-0 calibration: alternating linear least squares.

Per-frame model:  Y_i ~= g_i * sum_k w_k D_ik(t_i, sigma_x) + b_i
- w        shared effective region weights (the physics: |r_eff|^2 gray levels)
- g_i, b_i per-frame illumination gain/offset (robust, sigma-clipped -- real
           frames may contain defects/particles which must not bias the fit)
- t_i      per-frame design->image registration: integer cross-correlation
           search once, then Lucas-Kanade refinement each outer iteration
- sigma_x  one shared Gaussian kernel inflation absorbing nominal-optics
           mismatch (aberrations, TDI MTF, source shape)

The expensive hi-res blur of region coverages happens once per frame and is
cached to disk; everything in the alternation runs on the 256^2 pixel grid
where Fourier shift/blur of the band-limited densities is exact.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict

import numpy as np
from scipy import ndimage, optimize

from . import io_utils, regions, render


# --- frame preparation & cache ------------------------------------------------

def _region_param_key(cfg) -> str:
    d = {"layer_names": list(cfg.layer_names), "s_gds": cfg.s_gds,
         "frame_px": cfg.frame_px, "use_orientation": cfg.use_orientation,
         "orient": [cfg.orient_grad_sigma, cfg.orient_window,
                    cfg.orient_coherence_min, cfg.orient_energy_pct,
                    cfg.orient_energy_frac],
         "n_lambda": cfg.n_lambda, "optics": asdict(cfg.optics)}
    return json.dumps(d, sort_keys=True)


def open_cache(cache_dir: str, cfg) -> None:
    """Create/validate the density cache dir (invalidated on config change)."""
    os.makedirs(cache_dir, exist_ok=True)
    key_path = os.path.join(cache_dir, "cache_key.json")
    key = _region_param_key(cfg)
    if os.path.exists(key_path):
        with open(key_path) as f:
            if f.read() == key:
                return
        for p in os.listdir(cache_dir):
            if p.endswith("_D.npz"):
                os.remove(os.path.join(cache_dir, p))
    with open(key_path, "w") as f:
        f.write(key)


def prepare_frame(fp, cfg, otf_half, cache_dir: str) -> np.ndarray:
    """Pixel-grid density stack (K, n, n) float32 for one frame (cached)."""
    cpath = os.path.join(cache_dir, f"{fp.frame_id}_D.npz")
    if os.path.exists(cpath):
        return np.load(cpath)["D"].astype(np.float32)
    layers, conv = io_utils.load_layers(fp, cfg.layer_names, cfg.hr_n())
    stack = [render.density_to_pixel(cov, otf_half, cfg.s_gds)
             for _, cov in regions.iter_region_coverages(layers, cfg)]
    D = np.stack(stack)
    np.savez_compressed(cpath, D=D.astype(np.float16), convention=conv)
    return D


# --- small robust helpers -------------------------------------------------------

def _robust_gain_offset_masked(y, r, mask, clip=3.0, iters=3):
    m = mask.copy()
    alpha, beta = 1.0, float(np.median(y[m]) - np.median(r[m]))
    for _ in range(iters):
        xm, ym = r[m], y[m]
        vx = xm.var()
        if vx <= 0:
            alpha, beta = 1.0, float(ym.mean() - xm.mean())
            break
        alpha = float(((xm * ym).mean() - xm.mean() * ym.mean()) / vx)
        beta = float(ym.mean() - alpha * xm.mean())
        resid = y - (alpha * r + beta)
        med = np.median(resid[mask])
        sig = 1.4826 * np.median(np.abs(resid[mask] - med)) + 1e-9
        m = mask & (np.abs(resid - med) < clip * sig)
    return alpha, beta


def _highpass(img, sigma=8.0):
    return img - ndimage.gaussian_filter(img, sigma)


def _cc_shift(a, b, radius: int, return_curv: bool = False):
    """Sub-pixel (dy, dx) to apply to `b` so it best matches `a`.

    High-passed cross-correlation: integer argmax within +-radius, refined by
    parabolic peak interpolation (exact-enough for the smooth CC peak of
    band-limited images). Gradient-based LK is the wrong tool here: with all
    pattern sub-resolution, gradient energy lives on sparse macro-boundary
    pixels and falls below its noise eigenvalue floor. Pattern-invariant
    directions give a flat CC ridge -> ~zero curvature -> no refinement,
    which is the correct (harmless) behavior.
    """
    fa = np.fft.rfft2(_highpass(a))
    fb = np.fft.rfft2(_highpass(b))
    cc = np.fft.irfft2(fa * np.conj(fb), s=a.shape)
    n0, n1 = a.shape
    best, arg = -np.inf, (0, 0)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            v = cc[dy % n0, dx % n1]
            if v > best:
                best, arg = v, (dy, dx)
    dy, dx = arg
    c0 = cc[dy % n0, dx % n1]
    prom = max(c0 - float(np.median(cc)), 1e-12)

    def _parabolic(cm, cp):
        den = cm - 2.0 * c0 + cp
        curv = max(0.0, -den) / prom  # ~0 on pattern-invariant flat ridges
        if den >= -1e-12 * max(abs(c0), 1.0):
            return 0.0, curv
        return float(np.clip(0.5 * (cm - cp) / den, -0.75, 0.75)), curv

    ddy, curv_y = _parabolic(cc[(dy - 1) % n0, dx % n1],
                             cc[(dy + 1) % n0, dx % n1])
    ddx, curv_x = _parabolic(cc[dy % n0, (dx - 1) % n1],
                             cc[dy % n0, (dx + 1) % n1])
    if return_curv:
        return dy + ddy, dx + ddx, curv_y, curv_x
    return dy + ddy, dx + ddx


# --- main calibration -----------------------------------------------------------

def calibrate(frame_paths, cfg, out_dir: str, log=print) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "cache")
    open_cache(cache_dir, cfg)
    otf_half = render.nominal_otf_hr(cfg)

    rids = regions.region_ids(len(cfg.layer_names), cfg.use_orientation)
    names = [regions.region_name(r, cfg.layer_names) for r in rids]
    n_px = cfg.frame_px
    mb = cfg.border_margin_px
    border = np.zeros((n_px, n_px), dtype=bool)
    border[mb:-mb, mb:-mb] = True

    frame_paths = list(frame_paths)[: cfg.fit_frames_max]
    log(f"preparing {len(frame_paths)} frames "
        f"({len(rids)} candidate regions) ...")
    Ys, Ds, ids = [], [], []
    area = np.zeros(len(rids))
    for i, fp in enumerate(frame_paths):
        D = prepare_frame(fp, cfg, otf_half, cache_dir)
        Y = io_utils.load_bbp(fp, n_px)
        area += D[:, border].mean(axis=1)
        Ys.append(Y)
        Ds.append(D)
        ids.append(fp.frame_id)
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(frame_paths)} frames prepared")
    area /= len(frame_paths)

    kept = np.where(area >= cfg.min_region_area)[0]
    if kept.size < 2:
        raise RuntimeError("fewer than 2 regions above min_region_area")
    dropped = np.setdiff1d(np.arange(len(rids)), kept)
    log("regions kept: " + ", ".join(
        f"{names[k]}({area[k]:.3f})" for k in kept))
    if dropped.size:
        log("regions rare/dropped: " + ", ".join(
            f"{names[k]}({area[k]:.4f})" for k in dropped
            if area[k] > 0) or "(none with support)")

    # per-frame fixed masks: border AND not dominated by rare regions
    masks = []
    for D in Ds:
        rare = D[dropped].sum(axis=0) if dropped.size else np.zeros((n_px, n_px))
        masks.append(border & (rare < cfg.rare_mask_thresh))
    Dk = [D[kept] for D in Ds]
    K = kept.size

    F = len(Ds)
    g = np.ones(F)
    b = np.zeros(F)
    t = np.zeros((F, 2))
    clip_masks = [m.copy() for m in masks]
    w = np.zeros(K)
    sigma_x = 0.0

    def modified(i):
        return render.px_ops_stack(Dk[i], sigma_x, t[i, 0], t[i, 1])

    def solve_w(sig):
        nonlocal sigma_x
        old, sigma_x = sigma_x, sig
        G = np.zeros((K, K))
        h = np.zeros(K)
        for i in range(F):
            X = modified(i).reshape(K, -1)[:, clip_masks[i].ravel()]
            yv = (Ys[i].ravel()[clip_masks[i].ravel()] - b[i]) / g[i]
            G += X @ X.T
            h += X @ yv
        G += cfg.ridge * np.trace(G) / K * np.eye(K)
        sigma_x = old
        return np.linalg.solve(G, h)

    def total_loss(sig, wv):
        nonlocal sigma_x
        old, sigma_x = sigma_x, sig
        sse, npx = 0.0, 0
        for i in range(F):
            R = render.render(wv, modified(i))
            resid = (Ys[i] - (g[i] * R + b[i]))[clip_masks[i]]
            sse += float((resid ** 2).sum())
            npx += resid.size
        sigma_x = old
        return sse / max(npx, 1)

    for it in range(cfg.outer_iters):
        if it > 0:
            for i in range(F):
                R = render.render(w, modified(i))
                g[i], b[i] = _robust_gain_offset_masked(
                    Ys[i], R, masks[i], cfg.clip_sigma)
                yn = (Ys[i] - b[i]) / g[i]
                radius = cfg.max_shift_px if it == 1 else 1
                if radius > 0:
                    t[i] += _cc_shift(yn, R, radius)
                R = render.render(w, modified(i))
                resid = Ys[i] - (g[i] * R + b[i])
                med = np.median(resid[masks[i]])
                sig = 1.4826 * np.median(np.abs(resid[masks[i]] - med)) + 1e-9
                clip_masks[i] = masks[i] & (
                    np.abs(resid - med) < cfg.clip_sigma * sig)
            s = g.mean()
            g /= s
            w *= s
            c = b.mean()
            b -= c
            w += c  # region densities partition unity, so a constant moves freely
        if cfg.fit_sigma_extra and it in (2, cfg.outer_iters - 1):
            res = optimize.minimize_scalar(
                lambda s: total_loss(s, solve_w(s)),
                bounds=(0.0, cfg.sigma_extra_max_px), method="bounded",
                options={"xatol": 0.02})
            sigma_x = float(res.x)
            w = solve_w(sigma_x)
            log(f"  iter {it}: sigma_extra = {sigma_x:.3f} px")
        else:
            w = solve_w(sigma_x)
        log(f"  iter {it}: loss = {total_loss(sigma_x, w):.2f} ADU^2")

    # final per-frame metrics
    rows = []
    for i in range(F):
        R = render.render(w, modified(i))
        resid = Ys[i] - (g[i] * R + b[i])
        med = np.median(resid[masks[i]])
        mad = 1.4826 * np.median(np.abs(resid[masks[i]] - med))
        yc = Ys[i][masks[i]] - Ys[i][masks[i]].mean()
        rc = R[masks[i]] - R[masks[i]].mean()
        pear = float((yc * rc).sum() /
                     (np.sqrt((yc ** 2).sum() * (rc ** 2).sum()) + 1e-12))
        rows.append({"frame_id": ids[i], "g": g[i], "b": b[i],
                     "dy_px": t[i, 0], "dx_px": t[i, 1],
                     "robust_rms_adu": mad, "pearson_r": pear,
                     "clip_frac": 1.0 - clip_masks[i].sum() / masks[i].sum()})

    np.savez(os.path.join(out_dir, "model.npz"),
             w=w, kept=kept,
             combo_mask=np.array([rids[k][0] for k in kept]),
             orient_class=np.array([rids[k][1] for k in kept]),
             names=np.array([names[k] for k in kept]),
             area=area[kept], sigma_extra=sigma_x,
             layer_names=np.array(cfg.layer_names))
    with open(os.path.join(out_dir, "perframe.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    med_rms = float(np.median([r["robust_rms_adu"] for r in rows]))
    med_r = float(np.median([r["pearson_r"] for r in rows]))
    summary = {"n_fit_frames": F, "n_regions_kept": int(K),
               "region_names": [names[k] for k in kept],
               "w": [float(v) for v in w], "sigma_extra_px": sigma_x,
               "median_robust_rms_adu": med_rms, "median_pearson_r": med_r}
    log(f"calibration done: median robust RMS = {med_rms:.2f} ADU, "
        f"median r = {med_r:.4f}")
    return summary
