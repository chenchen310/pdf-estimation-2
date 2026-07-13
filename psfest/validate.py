"""Stage 4: validation of the estimated impulse response.

- K-fold: h estimated on train folds must explain held-out events down to the
  noise floor.
- Physics: FWHM against 0.5*lambda_c/NA, and spectral energy beyond the optical
  cutoff 2NA/lambda_min (must be ~0 for a real optical response).
"""
from __future__ import annotations

import copy

import numpy as np

from .epsf import EPSFModel, EventCrop, fit_event, run_epsf


def radial_profile(h: np.ndarray, oversample: int):
    """Azimuthal average; bin width matches the grid pitch so no bin is empty."""
    M = h.shape[0]
    c = M // 2
    yy, xx = np.mgrid[0:M, 0:M]
    rr = np.hypot(yy - c, xx - c) / oversample  # in pixels
    rmax = rr.max() * 0.95
    width = 1.0 / oversample
    nbins = int(rmax / width)
    edges = np.arange(nbins + 1) * width
    idx = np.clip(np.digitize(rr.ravel(), edges) - 1, 0, nbins - 1)
    centers, prof = [], []
    hv = h.ravel()
    rv = rr.ravel()
    for i in range(nbins):
        sel = (idx == i) & (rv < rmax)
        if sel.any():
            centers.append(rv[sel].mean())
            prof.append(hv[sel].mean())
    return np.array(centers), np.array(prof)


def fwhm_px(h: np.ndarray, oversample: int) -> float:
    r, p = radial_profile(h, oversample)
    p = p / p[0] if p[0] != 0 else p
    below = np.nonzero(p < 0.5)[0]
    if len(below) == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return float("nan")
    # linear interpolation across the half crossing
    r_half = r[i - 1] + (0.5 - p[i - 1]) * (r[i] - r[i - 1]) / (p[i] - p[i - 1])
    return float(2.0 * r_half)


def mtf_curve(h: np.ndarray, oversample: int):
    """Azimuthally averaged |H(f)|, f in cycles/pixel."""
    M = h.shape[0]
    H = np.abs(np.fft.fftshift(np.fft.fft2(h)))
    f = np.fft.fftshift(np.fft.fftfreq(M, d=1.0 / oversample))  # cyc/px
    fy, fx = np.meshgrid(f, f, indexing="ij")
    fr = np.hypot(fy, fx)
    width = oversample / M
    nb = int(fr.max() / width)
    edges = np.arange(nb + 1) * width
    idx = np.clip(np.digitize(fr.ravel(), edges) - 1, 0, nb - 1)
    centers, prof = [], []
    Hv = H.ravel()
    rv = fr.ravel()
    for i in range(nb):
        sel = idx == i
        if sel.any():
            centers.append(rv[sel].mean())
            prof.append(Hv[sel].mean())
    prof = np.array(prof)
    return np.array(centers), prof / prof[0]


def energy_beyond_cutoff(h: np.ndarray, oversample: int, cutoff_cyc_px: float,
                         guard: float = 1.0) -> float:
    M = h.shape[0]
    H2 = np.abs(np.fft.fft2(h)) ** 2
    f = np.fft.fftfreq(M, d=1.0 / oversample)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    out = (fy ** 2 + fx ** 2) > (cutoff_cyc_px * guard) ** 2
    return float(H2[out].sum() / H2.sum())


def kfold_validation(events: list, cfg, k: int = 5, seed: int = 0) -> dict:
    """Held-out normalized residual energy and chi2 per fold."""
    rng = np.random.default_rng(seed)
    kept = [ev for ev in events if ev.kept]
    if len(kept) < cfg.min_events * 2:
        return {"skipped": True, "reason": f"too few events ({len(kept)})"}
    order = rng.permutation(len(kept))
    folds = np.array_split(order, k)
    chi2s, nres = [], []
    for fi in range(k):
        test_idx = set(folds[fi].tolist())
        train = [copy.deepcopy(kept[i]) for i in range(len(kept))
                 if i not in test_idx]
        test = [copy.deepcopy(kept[i]) for i in range(len(kept))
                if i in test_idx]
        for ev in train + test:
            ev.kept = True
            ev.drop_reason = ""
        try:
            res = run_epsf(train, cfg)
        except RuntimeError:
            continue
        model = EPSFModel(res.h, cfg.oversample, cfg.window_radius())
        for ev in test:
            fit_event(model, ev, cfg)
            chi2s.append(ev.chi2)
            m = model.render(ev.cy, ev.cx)
            resid = ev.data - ev.a * m - ev.b
            sig = (ev.data - ev.b)
            nres.append(float((resid ** 2).sum() / max((sig ** 2).sum(), 1e-9)))
    return {
        "skipped": False,
        "n_test_events": len(chi2s),
        "chi2_median": float(np.median(chi2s)),
        "chi2_p90": float(np.percentile(chi2s, 90)),
        "resid_energy_median": float(np.median(nres)),
        "resid_energy_p90": float(np.percentile(nres, 90)),
    }


def validation_metrics(h: np.ndarray, cfg, events: list,
                       h_raw: np.ndarray | None = None) -> dict:
    opt = cfg.optics
    fw = fwhm_px(h, cfg.oversample)
    kept = [ev for ev in events if ev.kept]
    amps = np.array([ev.a for ev in kept])
    # the projected h is band-limited by construction; the meaningful check is
    # how much energy the *raw* accumulation put beyond the optical cutoff
    blv = energy_beyond_cutoff(h_raw, cfg.oversample, opt.cutoff_cyc_per_px,
                               guard=1.05) if h_raw is not None else None
    return {
        "fwhm_px": fw,
        "fwhm_nm": fw * opt.pixel_nm,
        "fwhm_theory_nm": 0.5 * opt.lambda_c_nm / opt.na,
        "cutoff_cyc_per_px": opt.cutoff_cyc_per_px,
        "bandlimit_violation_raw": blv,
        "n_events_kept": len(kept),
        "n_events_total": len(events),
        "n_dark": int((amps < 0).sum()),
        "n_bright": int((amps > 0).sum()),
        "chi2_median_kept": float(np.median([ev.chi2 for ev in kept]))
        if kept else float("nan"),
        "rho_median_kept": float(np.median([ev.rho for ev in kept]))
        if kept else float("nan"),
    }
