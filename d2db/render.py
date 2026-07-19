"""Forward rendering: region coverages -> band-limited pixel densities -> image.

The optical kernel is the nominal broadband incoherent OTF (from
D2DBConfig.optics, no aberrations) applied on the GDS raster grid, followed by
exact s x s box pixel integration (reshape-mean, the simulator's convention).
Real-tool deviations (aberrations, partial coherence, TDI MTF) are absorbed at
Stage 0 by the fitted region weights plus one fitted isotropic Gaussian
inflation applied on the pixel grid -- valid there because pixel densities are
band-limited well below Nyquist, where Fourier-domain Gaussian blur and
sub-pixel shifts are exact.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft

from psfest.optics import band_weights


def nominal_otf_hr(cfg) -> np.ndarray:
    """Broadband incoherent OTF on the hi-res raster grid (rfft2 half-plane).

    Real and normalized to 1 at DC; multiplying an rfft2 spectrum by it and
    inverting gives the aberration-free broadband blur.
    """
    n = cfg.hr_n()
    pitch = cfg.pitch_gds_nm()
    opt = cfg.optics
    lambdas = np.linspace(opt.lambda_min_nm, opt.lambda_max_nm, cfg.n_lambda)
    weights = band_weights(lambdas)
    f = np.fft.fftfreq(n, d=pitch)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    fr = np.hypot(fy, fx)
    otf = np.zeros((n, n), dtype=np.float64)
    for lam, w in zip(lambdas, weights):
        pupil = (fr * lam / opt.na <= 1.0).astype(np.complex64)
        psf = np.abs(sfft.ifft2(pupil, workers=-1)) ** 2
        o = sfft.fft2(psf.astype(np.complex64), workers=-1).real
        otf += w * o / o[0, 0]
    return otf[:, : n // 2 + 1].astype(np.float32)


def density_to_pixel(cov_hr: np.ndarray, otf_half: np.ndarray, s: int) -> np.ndarray:
    """Blur a coverage raster with the nominal OTF, then box-integrate s x s."""
    spec = sfft.rfft2(cov_hr, workers=-1)
    blurred = sfft.irfft2(spec * otf_half, s=cov_hr.shape, workers=-1)
    n_px = cov_hr.shape[0] // s
    return blurred.reshape(n_px, s, n_px, s).mean(axis=(1, 3)).astype(np.float32)


def _px_transfer(n: int, sigma_px: float, dy: float, dx: float) -> np.ndarray:
    f = np.fft.fftfreq(n)
    fy, fx = np.meshgrid(f, f, indexing="ij")
    h = np.exp(-2.0 * np.pi ** 2 * sigma_px ** 2 * (fy ** 2 + fx ** 2))
    return h * np.exp(-2j * np.pi * (fy * dy + fx * dx))


def px_ops_stack(D: np.ndarray, sigma_px: float, dy: float, dx: float) -> np.ndarray:
    """Gaussian inflation + sub-pixel shift of a (K, n, n) density stack."""
    if sigma_px == 0.0 and dy == 0.0 and dx == 0.0:
        return D
    h = _px_transfer(D.shape[1], sigma_px, dy, dx)
    out = np.empty_like(D)
    for k in range(D.shape[0]):
        out[k] = sfft.ifft2(sfft.fft2(D[k], workers=-1) * h, workers=-1).real
    return out


def render(w: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Sum_k w_k * D_k."""
    return np.tensordot(w, D, axes=(0, 0))
