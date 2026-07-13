"""Stage 1: robust photometric matching + signed difference with alignment QC.

diff = defect - (alpha * reference + beta), where (alpha, beta) absorb frame-to-
frame illumination/gain drift. Residual sub-pixel misalignment is measured with
upsampled phase correlation; if it exceeds the QC gate the reference is
re-shifted before subtraction (pattern residue is the dominant systematic error
on the impulse-response tails).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class DiffResult:
    diff: np.ndarray          # signed difference, defect-image photometric scale
    ref_matched: np.ndarray   # alpha*ref+beta (shifted if re-registered)
    alpha: float
    beta: float
    shift_yx: tuple           # measured residual shift (ref -> def), pixels
    reregistered: bool


def robust_gain_offset(d: np.ndarray, r: np.ndarray, clip: float = 3.0,
                       iters: int = 3) -> tuple[float, float]:
    """Fit d ~= alpha*r + beta with sigma-clipping (robust to sparse defects)."""
    x = r.ravel()
    y = d.ravel()
    mask = np.ones(x.size, dtype=bool)
    alpha, beta = 1.0, 0.0
    for _ in range(iters):
        xm, ym = x[mask], y[mask]
        vx = xm.var()
        if vx <= 0:
            alpha, beta = 1.0, ym.mean() - xm.mean()
            break
        alpha = ((xm * ym).mean() - xm.mean() * ym.mean()) / vx
        beta = ym.mean() - alpha * xm.mean()
        resid = y - (alpha * x + beta)
        med = np.median(resid)
        sigma = 1.4826 * np.median(np.abs(resid - med)) + 1e-9
        mask = np.abs(resid - med) < clip * sigma
    return float(alpha), float(beta)


def measure_shift(a: np.ndarray, b: np.ndarray, n_iter: int = 3,
                  huber_c: float = 3.0) -> tuple:
    """Sub-pixel shift t such that shifting `b` by t best matches `a`.

    Robust gradient-based (Lucas-Kanade style) estimation with Fourier
    re-warping. This is the right estimator for residual misalignment of
    tool-aligned frames: phase correlation is ambiguous on periodic patterns
    (aliases at pattern periods) and unconstrained along the invariant
    direction of line/space patterns, whereas the structure tensor here simply
    reports ~0 for components that produce no difference residue -- which is
    exactly the component we would not need to correct anyway. Valid for
    |t| < ~0.5 px; tool-aligned data satisfies this by construction.
    """
    t = np.zeros(2)
    bw = b
    sl = (slice(4, -4), slice(4, -4))
    for _ in range(n_iter):
        gy, gx = np.gradient(bw)
        d = (a - bw)[sl].ravel()
        gy = gy[sl].ravel()
        gx = gx[sl].ravel()
        w = np.ones_like(d)
        s = np.zeros(2)
        sig_d = 1.4826 * np.median(np.abs(d - np.median(d))) + 1e-9
        # structure-tensor eigenvalue floor: a direction whose eigenvalue is
        # consistent with pure noise is unconstrained by the pattern (flat
        # background, or along line/space bars). It also produces no diff
        # residue, so report 0 for that component instead of noise-fit junk.
        eig_floor = 8.0 * d.size * sig_d ** 2 / 4.0
        for _ in range(3):  # IRLS: defects/outliers must not bias the fit
            A = np.array([[np.sum(w * gy * gy), np.sum(w * gy * gx)],
                          [np.sum(w * gy * gx), np.sum(w * gx * gx)]])
            rhs = np.array([np.sum(w * gy * d), np.sum(w * gx * d)])
            evals, evecs = np.linalg.eigh(A)
            keep = evals > max(eig_floor, 1e-4 * evals.max())
            inv = np.where(keep, 1.0 / np.maximum(evals, 1e-12), 0.0)
            s = evecs @ (inv * (evecs.T @ rhs))
            r = d - gy * s[0] - gx * s[1]
            mad = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
            w = np.minimum(1.0, huber_c / np.maximum(np.abs(r) / mad, 1e-9))
        # d ~= -grad . t  =>  t_increment = -s
        t -= s
        if np.hypot(*s) < 1e-4:
            break
        bw = fourier_shift_image(b, t[0], t[1])
    return (float(t[0]), float(t[1]))


def fourier_shift_image(img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    return np.fft.ifft2(ndimage.fourier_shift(np.fft.fft2(img), (dy, dx))).real


def compute_diff(def_img: np.ndarray, ref_img: np.ndarray, cfg) -> DiffResult:
    alpha, beta = robust_gain_offset(def_img, ref_img,
                                     cfg.gain_clip_sigma, cfg.gain_iters)
    rm = alpha * ref_img + beta
    shift = measure_shift(def_img, rm)
    rereg = False
    if np.hypot(*shift) > cfg.max_resid_shift_px and cfg.reregister:
        rm = fourier_shift_image(rm, *shift)
        rereg = True
    return DiffResult(def_img - rm, rm, alpha, beta, shift, rereg)
