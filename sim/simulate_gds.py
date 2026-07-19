"""GDS-like synthetic BBP data with known ground truth for the D2DB pipeline.

Same philosophy as sim/simulate.py: the simulator is deliberately *richer*
than the estimator's model so the estimator cannot cheat.

- Imaging is partially coherent: Abbe sum over an ECP-like annular source
  (the real ECP pupil geometry is unavailable; this is a stand-in with the
  right character) and the broadband spectrum, of a *complex* reflectance map
  on the 3.75 nm raster grid. The Stage-0 estimator is linear-incoherent.
- Layer geometry gets CD bias, line-end pullback and corner rounding relative
  to the *design* rasters that are saved for the estimator (design != wafer).
- Dense sub-lambda line arrays get a polarization-anisotropy surrogate
  (VN illumination: v-lines vs h-lines differ in effective amplitude).
- Each die drifts in per-material film amplitude/phase and focus; the frame
  window is offset by a random sub-pixel amount vs the design raster.

Outputs per frame (contract shared with real converted fab data):
  FID_{i:05d}_bbp.npy    uint16 256x256 12-bit frame
  FID_{i:05d}_OD.npy     uint8 2048x2048 design coverage (0..255)
  FID_{i:05d}_POLY.npy   uint8 2048x2048 design coverage (0..255)
truth/ (simulation only, the pipeline never reads it):
  FID_{i:05d}_clean.npy  float32 noiseless counts
  meta.csv, sim_params.json, reflectance.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psfest.optics import _zernike_phase, band_weights  # noqa: E402
from sim.simulate import _illum, _to_counts, _add_noise, _downsample  # noqa: E402


@dataclass
class GdsSimParams:
    n_layouts: int = 16
    dies_per_layout: int = 4
    seed: int = 0
    patch: int = 256
    s_hr: int = 8                       # 3.75 nm raster cells
    pixel_nm: float = 30.0
    pad_px: int = 32                    # canvas margin (kills FFT wrap in truth)
    # optics
    na: float = 0.95
    lambda_min_nm: float = 190.0
    lambda_max_nm: float = 260.0
    n_lambda: int = 4
    src_sigmas: tuple = (0.85, 0.62)    # ECP-like annulus (stand-in geometry)
    src_counts: tuple = (6, 3)
    src_inner_weight: float = 0.7
    zernike_waves: dict = field(default_factory=lambda: {4: 0.10, 6: 0.06, 8: 0.04})
    focus_jitter_waves: float = 0.05    # per-die Z4 jitter (wafer flatness)
    n_red: int = 1280                   # reduced field grid (7.5 nm), see _crop
    # materials: complex effective reflectance (amp, phase) -- ground truth
    r_field: tuple = (0.55, 0.0)
    r_a: tuple = (0.80, 0.45)           # "OD" material
    r_b_on_field: tuple = (0.42, -0.50) # "POLY" over field
    r_b_on_a: tuple = (0.30, -0.75)
    aniso_a: float = 0.08               # VN surrogate: v-lines amp*(1+x/2),
    aniso_b: float = 0.05               #               h-lines amp*(1-x/2)
    die_amp_sig: float = 0.02           # per-die per-material film drift
    die_phase_sig: float = 0.04
    # process bias (actual wafer vs design raster)
    bias_a_nm: float = 1.5
    bias_b_nm: float = -2.0
    pullback_b_nm: float = 8.0
    round_sigma_cells: float = 1.0
    # frame window offset vs design raster (in 7.5 nm reduced cells; /4 = px)
    offset_max_red: int = 12
    # signal chain (same convention as sim/simulate.py)
    counts_scale: float = 2900.0
    pedestal: float = 100.0
    gain_e_per_adu: float = 5.0
    read_adu: float = 2.0
    illum_grad: float = 0.015

    def cell_nm(self) -> float:
        return self.pixel_nm / self.s_hr

    def canvas_cells(self) -> int:
        return (self.patch + 2 * self.pad_px) * self.s_hr


# --- anti-aliased drawing ----------------------------------------------------

def _add_rect(canvas, y0, y1, x0, x1):
    """Max-composite an axis-aligned rect with exact area coverage."""
    n = canvas.shape[0]
    y0, y1 = max(y0, 0.0), min(y1, float(n))
    x0, x1 = max(x0, 0.0), min(x1, float(n))
    if y1 <= y0 or x1 <= x0:
        return
    iy0, iy1 = int(np.floor(y0)), int(np.ceil(y1))
    ix0, ix1 = int(np.floor(x0)), int(np.ceil(x1))
    yy = np.arange(iy0, iy1)
    xx = np.arange(ix0, ix1)
    cy = np.clip(np.minimum(y1, yy + 1.0) - np.maximum(y0, yy), 0.0, 1.0)
    cx = np.clip(np.minimum(x1, xx + 1.0) - np.maximum(x0, xx), 0.0, 1.0)
    sl = (slice(iy0, iy1), slice(ix0, ix1))
    canvas[sl] = np.maximum(canvas[sl], cy[:, None] * cx[None, :])


def _draw_lines(canvas, rect, pitch_c, width_c, horiz, phase01,
                seg_seed=None, seg_range_c=None, gap_range_c=None,
                pullback_c=0.0):
    """Line array in `rect` (cells); optional segmented lines (line ends).

    seg_seed makes the segment layout reproducible so design and biased
    "actual" rasters share geometry and differ only by CD/pullback.
    """
    y0, y1, x0, x1 = rect
    lo, hi = (y0, y1) if horiz else (x0, x1)
    run0, run1 = (x0, x1) if horiz else (y0, y1)
    first = int(np.floor((lo - phase01 * pitch_c) / pitch_c)) - 1
    last = int(np.ceil((hi - phase01 * pitch_c) / pitch_c)) + 1
    for k in range(first, last + 1):
        c = phase01 * pitch_c + k * pitch_c
        a0, a1 = c - width_c / 2, c + width_c / 2
        if a1 <= lo or a0 >= hi:
            continue
        a0, a1 = max(a0, lo), min(a1, hi)
        if seg_seed is None:
            runs = [(run0, run1)]
        else:
            rs = np.random.default_rng(seg_seed + k + 1_000_000)
            runs, tpos = [], run0 - rs.uniform(0, seg_range_c[0])
            while tpos < run1:
                seg = rs.uniform(*seg_range_c)
                gap = rs.uniform(*gap_range_c)
                runs.append((tpos + pullback_c, tpos + seg - pullback_c))
                tpos += seg + gap
        for r0, r1 in runs:
            r0, r1 = max(r0, run0), min(r1, run1)
            if r1 <= r0:
                continue
            if horiz:
                _add_rect(canvas, a0, a1, r0, r1)
            else:
                _add_rect(canvas, r0, r1, a0, a1)


def _draw_pads(canvas, rect, w_c, h_c, gx_c, gy_c, phase, bias_c=0.0):
    y0, y1, x0, x1 = rect
    py, px_ = h_c + gy_c, w_c + gx_c
    ky0 = int(np.floor((y0 - phase[0] * py) / py)) - 1
    kx0 = int(np.floor((x0 - phase[1] * px_) / px_)) - 1
    for ky in range(ky0, int(np.ceil(y1 / py)) + 1):
        cy = phase[0] * py + ky * py
        for kx in range(kx0, int(np.ceil(x1 / px_)) + 1):
            cx = phase[1] * px_ + kx * px_
            _add_rect(canvas,
                      max(cy - bias_c, y0), min(cy + h_c + bias_c, y1),
                      max(cx - bias_c, x0), min(cx + w_c + bias_c, x1))


# --- layout description --------------------------------------------------------

_PITCHES_A = (24.0, 30.0, 36.0, 48.0, 64.0)
_PITCHES_B = (45.0, 60.0, 90.0)


def _split_blocks(rng, n, n_min):
    rects = [(0.0, float(n), 0.0, float(n))]
    for _ in range(int(rng.integers(2, 5))):
        idx = int(rng.integers(len(rects)))
        y0, y1, x0, x1 = rects[idx]
        if rng.random() < 0.5 and (y1 - y0) > 2 * n_min:
            cut = y0 + (y1 - y0) * rng.uniform(0.3, 0.7)
            rects[idx: idx + 1] = [(y0, cut, x0, x1), (cut, y1, x0, x1)]
        elif (x1 - x0) > 2 * n_min:
            cut = x0 + (x1 - x0) * rng.uniform(0.3, 0.7)
            rects[idx: idx + 1] = [(y0, y1, x0, cut), (y0, y1, cut, x1)]
    return rects


def make_layout(rng, p: GdsSimParams) -> list:
    """Blocks with per-layer drawing recipes (shared by design & actual)."""
    n = p.canvas_cells()
    cell = p.cell_nm()
    blocks = []
    styles = ["field", "a_lines", "logic", "sram", "pads", "b_only"]
    weights = np.array([0.08, 0.20, 0.34, 0.14, 0.12, 0.12])
    for rect in _split_blocks(rng, n, n_min=int(0.6e3 / cell)):
        style = styles[int(rng.choice(len(styles), p=weights / weights.sum()))]
        ents = []
        if style in ("a_lines", "logic", "sram"):
            horiz = bool(rng.random() < 0.5)
            ents.append({"layer": "OD", "kind": "lines",
                         "pitch_nm": float(rng.choice(_PITCHES_A)),
                         "duty": float(rng.uniform(0.42, 0.58)),
                         "horiz": horiz, "phase01": float(rng.random()),
                         "seg": None})
        if style == "logic":
            ents.append({"layer": "POLY", "kind": "lines",
                         "pitch_nm": float(rng.choice(_PITCHES_B)),
                         "duty": float(rng.uniform(0.35, 0.50)),
                         "horiz": not ents[0]["horiz"],
                         "phase01": float(rng.random()),
                         "seg": [float(rng.uniform(150, 800)),
                                 float(rng.uniform(40, 140)),
                                 int(rng.integers(0, 2 ** 31))]})
        if style == "sram":
            ents.append({"layer": "POLY", "kind": "lines",
                         "pitch_nm": float(rng.choice((36.0, 48.0))),
                         "duty": float(rng.uniform(0.40, 0.55)),
                         "horiz": not ents[0]["horiz"],
                         "phase01": float(rng.random()), "seg": None})
        if style == "pads":
            ents.append({"layer": "OD", "kind": "pads",
                         "w_nm": float(rng.uniform(120, 400)),
                         "h_nm": float(rng.uniform(120, 400)),
                         "gx_nm": float(rng.uniform(80, 300)),
                         "gy_nm": float(rng.uniform(80, 300)),
                         "phase": [float(rng.random()), float(rng.random())]})
            if rng.random() < 0.5:
                ents.append({"layer": "POLY", "kind": "lines",
                             "pitch_nm": float(rng.choice(_PITCHES_B)),
                             "duty": float(rng.uniform(0.35, 0.50)),
                             "horiz": bool(rng.random() < 0.5),
                             "phase01": float(rng.random()), "seg": None})
        if style == "b_only":
            ents.append({"layer": "POLY", "kind": "lines",
                         "pitch_nm": float(rng.choice(_PITCHES_B)),
                         "duty": float(rng.uniform(0.35, 0.55)),
                         "horiz": bool(rng.random() < 0.5),
                         "phase01": float(rng.random()), "seg": None})
        blocks.append({"rect": rect, "style": style, "ents": ents})
    return blocks


def rasterize(blocks, p: GdsSimParams, actual: bool):
    """Coverage rasters per layer + anisotropy factor maps (actual only)."""
    n = p.canvas_cells()
    cell = p.cell_nm()
    cov = {"OD": np.zeros((n, n), np.float32),
           "POLY": np.zeros((n, n), np.float32)}
    fac = {"OD": np.ones((n, n), np.float32),
           "POLY": np.ones((n, n), np.float32)}
    bias = {"OD": p.bias_a_nm, "POLY": p.bias_b_nm}
    aniso = {"OD": p.aniso_a, "POLY": p.aniso_b}
    for blk in blocks:
        rect = blk["rect"]
        for e in blk["ents"]:
            lay = e["layer"]
            if e["kind"] == "lines":
                pitch_c = e["pitch_nm"] / cell
                width_nm = e["duty"] * e["pitch_nm"] + (bias[lay] if actual else 0.0)
                seg = e["seg"]
                _draw_lines(cov[lay], rect, pitch_c, width_nm / cell,
                            e["horiz"], e["phase01"],
                            seg_seed=None if seg is None else seg[2],
                            seg_range_c=None if seg is None else
                            (seg[0] / cell, 2 * seg[0] / cell),
                            gap_range_c=None if seg is None else
                            (seg[1] / cell, 2 * seg[1] / cell),
                            pullback_c=(p.pullback_b_nm / cell
                                        if actual and seg is not None else 0.0))
                f = 1.0 + (0.5 if not e["horiz"] else -0.5) * aniso[lay]
                y0, y1, x0, x1 = (int(round(v)) for v in rect)
                fac[lay][y0:y1, x0:x1] = f
            else:
                _draw_pads(cov[lay], rect, e["w_nm"] / cell, e["h_nm"] / cell,
                           e["gx_nm"] / cell, e["gy_nm"] / cell, e["phase"],
                           bias_c=(0.5 * bias[lay] / cell if actual else 0.0))
    if actual and p.round_sigma_cells > 0:
        for lay in cov:
            cov[lay] = np.clip(ndimage.gaussian_filter(
                cov[lay], p.round_sigma_cells), 0.0, 1.0)
    return cov, fac


# --- partially coherent imaging -------------------------------------------------

def _crop_spectrum(F, n_red):
    n = F.shape[0]
    c, h = n // 2, n_red // 2
    Fs = np.fft.fftshift(F)
    return np.fft.ifftshift(Fs[c - h: c + h, c - h: c + h]).copy()


def _material_spectra(cov, fac, p: GdsSimParams):
    """Cropped spectra of the 4 material coverage maps (complex64)."""
    a, b = cov["OD"], cov["POLY"]
    fa, fb = fac["OD"], fac["POLY"]
    maps = {"field": (1 - a) * (1 - b),
            "a": a * (1 - b) * fa,
            "b_on_field": (1 - a) * b * fb,
            "b_on_a": a * b * fa * fb}
    out = {}
    for k, m in maps.items():
        F = sfft.fft2(m.astype(np.complex64), workers=-1)
        out[k] = _crop_spectrum(F, p.n_red)
    return out


def _source_points(p: GdsSimParams):
    pts = []
    for sigma, count, wgt in zip(p.src_sigmas, p.src_counts,
                                 (1.0, p.src_inner_weight)):
        for k in range(count):
            ang = 2 * np.pi * (k + 0.5 * (sigma == p.src_sigmas[1])) / count
            pts.append((sigma * np.cos(ang), sigma * np.sin(ang), wgt))
    return pts


def _pupils(p: GdsSimParams, zern):
    """Per-lambda pupils with aberrations on the reduced grid (fft order)."""
    f = np.fft.fftfreq(p.n_red, d=2.0 * p.cell_nm())  # 7.5 nm reduced cells
    fy, fx = np.meshgrid(f, f, indexing="ij")
    fr = np.hypot(fy, fx)
    theta = np.arctan2(fy, fx)
    lambdas = np.linspace(p.lambda_min_nm, p.lambda_max_nm, p.n_lambda)
    weights = band_weights(lambdas)
    lam_ref = 0.5 * (p.lambda_min_nm + p.lambda_max_nm)
    pupils = []
    for lam, w in zip(lambdas, weights):
        rho = fr * lam / p.na
        sup = rho <= 1.0
        opd = _zernike_phase(np.where(sup, rho, 0.0), theta, zern)
        pup = np.where(sup, np.exp(2j * np.pi * opd * (lam_ref / lam)), 0.0)
        pupils.append((lam, w, pup.astype(np.complex64)))
    return pupils


def _abbe_intensity(mat_spectra, coeffs, pupils, src_pts, p: GdsSimParams):
    """Partially coherent |field|^2 on the reduced grid."""
    n_can = p.canvas_cells()
    FU = np.zeros_like(mat_spectra["field"])
    for k, c in coeffs.items():
        FU += np.complex64(c) * mat_spectra[k]
    scale = (p.n_red / n_can) ** 2
    bin_nm = 1.0 / (p.n_red * 2.0 * p.cell_nm())  # cyc/nm per bin
    I = np.zeros((p.n_red, p.n_red), np.float32)
    wsum = 0.0
    for sx, sy, sw in src_pts:
        for lam, lw, pup in pupils:
            by = int(round(sy * p.na / lam / bin_nm))
            bx = int(round(sx * p.na / lam / bin_nm))
            spec = np.roll(FU, (by, bx), axis=(0, 1)) * pup
            fld = sfft.ifft2(spec, workers=-1)
            I += (sw * lw) * (np.abs(fld) ** 2).astype(np.float32)
            wsum += sw * lw
    return I * (scale ** 2 / wsum)


# --- main ---------------------------------------------------------------------

def generate(out_dir: str, p: GdsSimParams) -> None:
    os.makedirs(out_dir, exist_ok=True)
    truth_dir = os.path.join(out_dir, "truth")
    os.makedirs(truth_dir, exist_ok=True)
    cell = p.cell_nm()
    n_can = p.canvas_cells()
    hr_n = p.patch * p.s_hr
    c0 = (n_can - hr_n) // 2

    mats = {"field": p.r_field, "a": p.r_a,
            "b_on_field": p.r_b_on_field, "b_on_a": p.r_b_on_a}
    with open(os.path.join(truth_dir, "reflectance.json"), "w") as f:
        json.dump({k: {"amp": v[0], "phase": v[1]} for k, v in mats.items()},
                  f, indent=2)

    src_pts = _source_points(p)
    meta_rows = []
    fid = 0
    for lid in range(p.n_layouts):
        rng_lay = np.random.default_rng(p.seed * 7919 + lid)
        blocks = make_layout(rng_lay, p)
        cov_design, _ = rasterize(blocks, p, actual=False)
        cov_actual, fac = rasterize(blocks, p, actual=True)
        design_crop = {lay: np.rint(255 * cov_design[lay][
            c0: c0 + hr_n, c0: c0 + hr_n]).astype(np.uint8)
            for lay in ("OD", "POLY")}
        mat_spectra = _material_spectra(cov_actual, fac, p)

        for die in range(p.dies_per_layout):
            rng = np.random.default_rng(p.seed * 1_000_003 + fid)
            coeffs = {}
            drift = {}
            for k, (amp, ph) in mats.items():
                ea = rng.normal(0.0, p.die_amp_sig)
                ep = rng.normal(0.0, p.die_phase_sig)
                coeffs[k] = amp * (1 + ea) * np.exp(1j * (ph + ep))
                drift[k] = (ea, ep)
            zern = dict(p.zernike_waves)
            z4 = zern.get(4, 0.0) + rng.normal(0.0, p.focus_jitter_waves)
            zern[4] = z4
            pupils = _pupils(p, zern)
            I_red = _abbe_intensity(mat_spectra, coeffs, pupils, src_pts, p)

            dy = int(rng.integers(-p.offset_max_red, p.offset_max_red + 1))
            dx = int(rng.integers(-p.offset_max_red, p.offset_max_red + 1))
            n_win = p.patch * 4                       # reduced cells per frame
            r0 = (p.n_red - n_win) // 2
            win = I_red[r0 + dy: r0 + dy + n_win, r0 + dx: r0 + dx + n_win]
            i_px = _downsample(win.astype(np.float64), 4)

            illum = _illum(rng, p.patch, p.illum_grad)
            clean = _to_counts(i_px, illum, p)
            frame = _add_noise(clean, rng, p)

            frame_id = f"FID_{fid:05d}"
            np.save(os.path.join(out_dir, f"{frame_id}_bbp.npy"), frame)
            for lay in ("OD", "POLY"):
                np.save(os.path.join(out_dir, f"{frame_id}_{lay}.npy"),
                        design_crop[lay])
            np.save(os.path.join(truth_dir, f"{frame_id}_clean.npy"),
                    clean.astype(np.float32))
            meta_rows.append({
                "frame_id": frame_id, "layout_id": lid, "die_id": fid,
                "expected_dy_px": -dy / 4.0, "expected_dx_px": -dx / 4.0,
                "z4_waves": z4,
                "drift_amp_a": drift["a"][0], "drift_phase_a": drift["a"][1]})
            fid += 1
        print(f"layout {lid + 1}/{p.n_layouts} done ({fid} frames)", flush=True)

    with open(os.path.join(truth_dir, "meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        w.writeheader()
        w.writerows(meta_rows)
    with open(os.path.join(truth_dir, "sim_params.json"), "w") as f:
        json.dump(asdict(p), f, indent=2, default=list)
    print(f"done: {fid} frames in {out_dir}", flush=True)
