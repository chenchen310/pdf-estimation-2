"""Synthetic BBP-like inspection data with known ground truth.

Deliberately independent of the estimator: imaging happens on a 6x-oversampled
grid (5 nm pitch) with a broadband aberrated pupil model, box pixel
integration, per-frame illumination drift, small residual reference shifts,
Poisson shot noise, read noise and 12-bit quantization. Defects are absorbing
point scatterers whose amplitude scales with the local background (cross-term
behaviour), plus a few extended defects and bright nuisances to exercise the
screening gates.

Outputs: DID_{index:05d}_def.npy / _ref1.npy (uint16), plus truth/ tables and
the ground-truth PSF for evaluation only.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import ndimage, signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psfest.optics import _zernike_phase, band_weights  # noqa: E402


@dataclass
class SimParams:
    n_pairs: int = 650
    seed: int = 0
    patch: int = 256
    s_hr: int = 6                      # high-res oversampling (5 nm pitch)
    pixel_nm: float = 30.0
    na: float = 0.95
    lambda_min_nm: float = 190.0
    lambda_max_nm: float = 260.0
    n_lambda: int = 8
    zernike_waves: dict = field(default_factory=lambda: {4: 0.12, 6: 0.08, 8: 0.05})
    # signal chain
    counts_scale: float = 2900.0
    pedestal: float = 100.0
    gain_e_per_adu: float = 5.0
    read_adu: float = 2.0
    # population fractions
    frac_none: float = 0.25
    frac_extended: float = 0.05
    frac_bright: float = 0.03
    frac_two_points: float = 0.04
    # dark point defect strength (absorption fraction of local background)
    strength_ln_med: float = 0.12
    strength_ln_sig: float = 0.7
    strength_clip: tuple = (0.02, 0.5)
    # frame-to-frame nuisances
    ref_shift_sigma_px: float = 0.03
    ref_shift_bad_p: float = 0.03
    ref_shift_bad_range: tuple = (0.12, 0.2)
    illum_grad: float = 0.015
    ref_gain_sigma: float = 0.004
    margin_px: int = 24


# --- optics ---------------------------------------------------------------

def build_optics(p: SimParams):
    """Broadband OTF on the high-res field grid + insertion kernel + GT ePSF."""
    n_hr = p.patch * p.s_hr
    pitch = p.pixel_nm / p.s_hr
    lambdas = np.linspace(p.lambda_min_nm, p.lambda_max_nm, p.n_lambda)
    weights = band_weights(lambdas)
    lam_ref = 0.5 * (p.lambda_min_nm + p.lambda_max_nm)

    f = np.fft.fftfreq(n_hr, d=pitch)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    fr = np.hypot(fy, fx)
    theta = np.arctan2(fy, fx)

    otf = np.zeros((n_hr, n_hr), dtype=complex)
    for lam, w in zip(lambdas, weights):
        rho = fr * lam / p.na
        sup = rho <= 1.0
        opd = _zernike_phase(np.where(sup, rho, 0.0), theta, p.zernike_waves)
        pupil = np.where(sup, np.exp(2j * np.pi * opd * (lam_ref / lam)), 0.0)
        asf = np.fft.ifft2(pupil)
        psf = np.abs(asf) ** 2
        o = np.fft.fft2(psf)
        otf += w * o / o[0, 0]

    psf_bb = np.fft.fftshift(np.fft.ifft2(otf).real)
    c = n_hr // 2
    half = p.patch // 2
    kernel = psf_bb[c - half: c + half, c - half: c + half].copy()
    kernel /= kernel.max()

    # ground-truth effective PSF on the estimator's convention:
    # pixel-aperture integration (trapezoid = centered box of s_hr cells),
    # sampled at pixel/3 pitch on a 73x73 window (matches oversample=3, R=12).
    ap = np.ones(p.s_hr + 1)
    ap[0] = ap[-1] = 0.5
    ap /= ap.sum()
    boxed = ndimage.convolve1d(psf_bb, ap, axis=0, mode="nearest")
    boxed = ndimage.convolve1d(boxed, ap, axis=1, mode="nearest")
    step = p.s_hr // 3                       # 2 hr cells = pixel/3
    R_est, s_est = 12, 3
    m = R_est * s_est                        # 36 -> 73 grid
    idx = c + step * np.arange(-m, m + 1)
    h_eff = boxed[np.ix_(idx, idx)].copy()
    h_eff /= h_eff.max()
    return otf, kernel, h_eff


# --- backgrounds (reflectance on the high-res grid) -----------------------

def _coords_nm(n_hr: int, pitch: float):
    v = (np.arange(n_hr) - n_hr / 2) * pitch
    return np.meshgrid(v, v, indexing="ij")


def _lines(rng, n_hr, pitch):
    yy, xx = _coords_nm(n_hr, pitch)
    ang = rng.choice([0.0, np.pi / 2]) if rng.random() < 0.7 \
        else rng.uniform(0, np.pi)
    coord = xx * np.cos(ang) + yy * np.sin(ang)
    period = rng.uniform(150, 600)
    duty = rng.uniform(0.35, 0.65)
    hi, lo = rng.uniform(0.7, 1.0), rng.uniform(0.05, 0.25)
    bars = ((coord / period + rng.random()) % 1.0) < duty
    return np.where(bars, hi, lo)


def _blocks(rng, n_hr, pitch):
    yy, xx = _coords_nm(n_hr, pitch)
    per_y = rng.uniform(200, 700)
    per_x = rng.uniform(200, 700)
    hi, lo = rng.uniform(0.7, 1.0), rng.uniform(0.05, 0.25)
    by = ((yy / per_y + rng.random()) % 1.0) < rng.uniform(0.4, 0.6)
    bx = ((xx / per_x + rng.random()) % 1.0) < rng.uniform(0.4, 0.6)
    return np.where(by & bx, hi, lo)


def _maze(rng, n_hr, pitch):
    img = np.full((n_hr, n_hr), rng.uniform(0.05, 0.25))
    hi = rng.uniform(0.65, 1.0)
    for _ in range(int(rng.integers(20, 60))):
        w = int(rng.uniform(100, 1500) / pitch)
        h = int(rng.uniform(100, 1500) / pitch)
        y = int(rng.integers(0, n_hr - 1))
        x = int(rng.integers(0, n_hr - 1))
        img[y: y + h, x: x + w] = hi
    return img


def _flat(rng, n_hr, pitch):
    return np.full((n_hr, n_hr), rng.uniform(0.08, 0.9))


def _split(rng, n_hr, pitch):
    a, _ = _make_background(rng, n_hr, pitch, no_split=True)
    b, _ = _make_background(rng, n_hr, pitch, no_split=True)
    yy, xx = _coords_nm(n_hr, pitch)
    ang = rng.uniform(0, np.pi)
    off = rng.uniform(-0.25, 0.25) * n_hr * pitch
    side = (xx * np.cos(ang) + yy * np.sin(ang)) > off
    return np.where(side, a, b)


def _make_background(rng, n_hr, pitch, no_split=False):
    kinds = [(_lines, 0.30), (_blocks, 0.15), (_maze, 0.25), (_flat, 0.08),
             (_split, 0.22)]
    if no_split:
        kinds = kinds[:-1]
    fns, ws = zip(*kinds)
    ws = np.array(ws) / sum(ws)
    fn = fns[int(rng.choice(len(fns), p=ws))]
    return fn(rng, n_hr, pitch), fn.__name__.lstrip("_")


# --- frame rendering -------------------------------------------------------

def _fourier_shift(img, dy, dx):
    f0 = np.fft.fftfreq(img.shape[0])
    f1 = np.fft.fftfreq(img.shape[1])
    ramp = np.exp(-2j * np.pi * (f0[:, None] * dy + f1[None, :] * dx))
    return np.fft.ifft2(np.fft.fft2(img) * ramp).real


def _downsample(field_hr: np.ndarray, s: int) -> np.ndarray:
    n = field_hr.shape[0] // s
    return field_hr.reshape(n, s, n, s).mean(axis=(1, 3))


def _illum(rng, n, grad):
    v = np.linspace(-0.5, 0.5, n)
    yy, xx = np.meshgrid(v, v, indexing="ij")
    return (1.0 + rng.uniform(-grad, grad) * yy + rng.uniform(-grad, grad) * xx
            + rng.uniform(-grad, grad) * yy * xx)


def _to_counts(i_px, illum, p: SimParams):
    return p.pedestal + i_px * p.counts_scale * illum


def _add_noise(counts, rng, p: SimParams):
    photons = np.maximum(counts - p.pedestal, 0.0)
    e = rng.poisson(photons * p.gain_e_per_adu) / p.gain_e_per_adu
    out = e + p.pedestal + rng.normal(0.0, p.read_adu, counts.shape)
    return np.clip(np.rint(out), 0, 4095).astype(np.uint16)


# --- defect insertion -------------------------------------------------------

def _hr_pos(c_px: float, s: int) -> tuple[int, float]:
    """Pixel coordinate -> (integer hr index, fractional remainder)."""
    u = c_px * s + (s - 1) / 2.0   # px center i maps to hr center i*s+(s-1)/2
    ui = int(np.rint(u))
    return ui, float(u - ui)


def _insert_point(field, kernel, y_px, x_px, amp, s):
    n_hr = field.shape[0]
    half = kernel.shape[0] // 2
    uy, fy = _hr_pos(y_px, s)
    ux, fx = _hr_pos(x_px, s)
    k = _fourier_shift(kernel, fy, fx)
    sl = (slice(uy - half, uy + half), slice(ux - half, ux + half))
    if (sl[0].start < 0 or sl[1].start < 0 or sl[0].stop > n_hr
            or sl[1].stop > n_hr):
        raise ValueError("defect too close to border")
    field[sl] += amp * k


def _insert_extended(rng, field, kernel_sum1, y_px, x_px, strength, i_loc, s,
                     pitch):
    n_hr = field.shape[0]
    L = rng.uniform(60, 400) / pitch
    W = rng.uniform(30, 90) / pitch
    ang = rng.uniform(0, np.pi)
    uy, _ = _hr_pos(y_px, s)
    ux, _ = _hr_pos(x_px, s)
    box = 512
    lo_y, lo_x = uy - box // 2, ux - box // 2
    yy, xx = np.mgrid[0:box, 0:box].astype(float)
    yy -= box // 2
    xx -= box // 2
    t = yy * np.sin(ang) + xx * np.cos(ang)
    u = -yy * np.cos(ang) + xx * np.sin(ang)
    A = ((np.abs(t) <= L / 2) & (np.abs(u) <= W / 2)).astype(float)
    blur = signal.fftconvolve(A, kernel_sum1, mode="same")
    field[lo_y: lo_y + box, lo_x: lo_x + box] -= strength * i_loc * blur


# --- main ------------------------------------------------------------------

def generate(out_dir: str, p: SimParams) -> None:
    os.makedirs(out_dir, exist_ok=True)
    truth_dir = os.path.join(out_dir, "truth")
    os.makedirs(truth_dir, exist_ok=True)

    print("building broadband optics model ...", flush=True)
    otf, kernel, h_eff = build_optics(p)
    kernel_sum1 = kernel / kernel.sum()
    np.savez(os.path.join(truth_dir, "psf_gt.npz"),
             kernel_hr=kernel, h_eff=h_eff, s_hr=p.s_hr,
             pixel_nm=p.pixel_nm, na=p.na,
             lambda_min_nm=p.lambda_min_nm, lambda_max_nm=p.lambda_max_nm)

    n_hr = p.patch * p.s_hr
    pitch = p.pixel_nm / p.s_hr
    defect_rows, pair_rows = [], []

    for i in range(p.n_pairs):
        rng = np.random.default_rng(p.seed * 1_000_003 + i)
        bg, bg_type = _make_background(rng, n_hr, pitch)
        img_opt = np.fft.ifft2(np.fft.fft2(bg) * otf).real
        img_opt = np.clip(img_opt, 0.0, None)

        # choose defect content
        u = rng.random()
        if u < p.frac_none:
            kind = "none"
        elif u < p.frac_none + p.frac_extended:
            kind = "extended"
        elif u < p.frac_none + p.frac_extended + p.frac_bright:
            kind = "bright"
        elif u < (p.frac_none + p.frac_extended + p.frac_bright
                  + p.frac_two_points):
            kind = "two_points"
        else:
            kind = "point"

        def_opt = img_opt.copy()
        placed = []
        if kind != "none":
            n_pts = 2 if kind == "two_points" else 1
            # extended defects need extra margin for their 512-cell hr work box
            mar = 44 if kind == "extended" else p.margin_px
            for j in range(n_pts):
                for _ in range(100):
                    y = rng.uniform(mar, p.patch - mar)
                    x = rng.uniform(mar, p.patch - mar)
                    if all(np.hypot(y - py, x - px_) > 30 for py, px_, *_ in placed):
                        break
                uy, _ = _hr_pos(y, p.s_hr)
                ux, _ = _hr_pos(x, p.s_hr)
                i_loc = float(img_opt[uy, ux])
                if kind == "bright":
                    strength = float(rng.uniform(0.05, 0.30))
                    _insert_point(def_opt, kernel, y, x, +strength, p.s_hr)
                elif kind == "extended":
                    strength = float(rng.uniform(0.05, 0.35))
                    _insert_extended(rng, def_opt, kernel_sum1, y, x,
                                     strength, i_loc, p.s_hr, pitch)
                else:
                    strength = float(np.exp(rng.normal(
                        np.log(p.strength_ln_med), p.strength_ln_sig)))
                    strength = float(np.clip(strength, *p.strength_clip))
                    _insert_point(def_opt, kernel, y, x,
                                  -strength * i_loc, p.s_hr)
                placed.append((y, x, strength, i_loc))

        # frame nuisances
        if rng.random() < p.ref_shift_bad_p:
            mag = rng.uniform(*p.ref_shift_bad_range)
            ang = rng.uniform(0, 2 * np.pi)
            ref_dy, ref_dx = mag * np.sin(ang), mag * np.cos(ang)
        else:
            ref_dy = rng.normal(0, p.ref_shift_sigma_px)
            ref_dx = rng.normal(0, p.ref_shift_sigma_px)
        ref_opt = _fourier_shift(img_opt, ref_dy * p.s_hr, ref_dx * p.s_hr)
        ref_opt = np.clip(ref_opt, 0.0, None)

        illum = _illum(rng, p.patch, p.illum_grad)
        gain_eps = rng.normal(0, p.ref_gain_sigma)
        def_px = _downsample(def_opt, p.s_hr)
        ref_px = _downsample(ref_opt, p.s_hr)
        clean_px = _downsample(img_opt, p.s_hr)

        def_counts = _to_counts(def_px, illum, p)
        ref_counts = _to_counts(ref_px, illum * (1.0 + gain_eps), p)
        clean_counts = _to_counts(clean_px, illum, p)

        pair_id = f"DID_{i:05d}"
        np.save(os.path.join(out_dir, f"{pair_id}_def.npy"),
                _add_noise(def_counts, rng, p))
        np.save(os.path.join(out_dir, f"{pair_id}_ref1.npy"),
                _add_noise(ref_counts, rng, p))

        # exact noiseless defect signal for truth
        sig = def_counts - clean_counts
        if kind == "none":
            defect_rows.append({"pair_id": pair_id, "type": "none",
                                "y_px": -1, "x_px": -1, "strength": 0,
                                "i_loc_opt": 0, "peak_dadu": 0, "flux_dadu": 0})
        else:
            for (y, x, strength, i_loc) in placed:
                yi, xi = int(round(y)), int(round(x))
                w = sig[max(yi - 14, 0): yi + 15, max(xi - 14, 0): xi + 15]
                peak = float(w.min()) if kind != "bright" else float(w.max())
                defect_rows.append({
                    "pair_id": pair_id, "type": kind, "y_px": y, "x_px": x,
                    "strength": strength, "i_loc_opt": i_loc,
                    "peak_dadu": peak, "flux_dadu": float(w.sum())})
        pair_rows.append({"pair_id": pair_id, "bg_type": bg_type,
                          "kind": kind, "ref_dy": ref_dy, "ref_dx": ref_dx,
                          "ref_gain_eps": gain_eps})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{p.n_pairs} pairs", flush=True)

    for name, rows in [("defects.csv", defect_rows), ("pairs.csv", pair_rows)]:
        with open(os.path.join(truth_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(truth_dir, "sim_params.json"), "w") as f:
        json.dump(asdict(p), f, indent=2, default=list)
    print(f"done: {p.n_pairs} pairs in {out_dir}", flush=True)
