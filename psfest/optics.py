"""Optical helpers: band-limit projection and a broadband scalar PSF model.

The band-limit projection is the physics-based regularizer used by the ePSF
estimator: any incoherently imaged intensity distribution has zero spectral
content beyond 2*NA/lambda_min, so the estimated impulse response is projected
onto that support at every iteration.

The broadband PSF model is used by the simulator (with aberrations) and for
theory sanity checks. The estimator itself never assumes this model.
"""
from __future__ import annotations

import numpy as np


def freq_grid(n: int, pitch_nm: float):
    """Return (fy, fx) in cycles/nm for an n x n grid, FFT ordering."""
    f = np.fft.fftfreq(n, d=pitch_nm)
    return np.meshgrid(f, f, indexing="ij")


def bandlimit_mask(n: int, pitch_nm: float, na: float, lambda_min_nm: float,
                   guard: float = 1.0) -> np.ndarray:
    fy, fx = freq_grid(n, pitch_nm)
    fc = 2.0 * na / lambda_min_nm * guard
    return (fy * fy + fx * fx) <= fc * fc


def project_bandlimit(img: np.ndarray, pitch_nm: float, na: float,
                      lambda_min_nm: float, guard: float = 1.0) -> np.ndarray:
    mask = bandlimit_mask(img.shape[0], pitch_nm, na, lambda_min_nm, guard)
    return np.fft.ifft2(np.fft.fft2(img) * mask).real


# --- Zernike aberrations (Noll indexing, unit-variance-free simple forms) -----

def _zernike_phase(rho: np.ndarray, theta: np.ndarray, coeffs: dict) -> np.ndarray:
    """OPD in waves (at reference lambda). Simple non-normalized polynomials."""
    w = np.zeros_like(rho)
    for j, c in coeffs.items():
        if c == 0.0:
            continue
        if j == 4:      # defocus
            w += c * (2 * rho ** 2 - 1)
        elif j == 5:    # oblique astigmatism
            w += c * (rho ** 2) * np.sin(2 * theta)
        elif j == 6:    # vertical astigmatism
            w += c * (rho ** 2) * np.cos(2 * theta)
        elif j == 7:    # vertical coma
            w += c * (3 * rho ** 3 - 2 * rho) * np.sin(theta)
        elif j == 8:    # horizontal coma
            w += c * (3 * rho ** 3 - 2 * rho) * np.cos(theta)
        elif j == 11:   # spherical
            w += c * (6 * rho ** 4 - 6 * rho ** 2 + 1)
        else:
            raise ValueError(f"unsupported Zernike index {j}")
    return w


def band_weights(lambdas: np.ndarray, peak_nm: float = 215.0, width_nm: float = 45.0):
    """Smooth plasma-like spectral weighting across the band."""
    w = np.exp(-((lambdas - peak_nm) / width_nm) ** 2)
    return w / w.sum()


def broadband_psf(n: int, pitch_nm: float, na: float,
                  lambdas: np.ndarray, weights: np.ndarray,
                  zernike_waves: dict | None = None,
                  lambda_ref_nm: float | None = None) -> np.ndarray:
    """Incoherent broadband PSF on an n x n grid with the given pitch.

    zernike_waves: {noll_index: OPD amplitude in waves at lambda_ref_nm}.
    Returns the PSF centered at (n//2, n//2), normalized to unit sum.
    """
    zernike_waves = zernike_waves or {}
    if lambda_ref_nm is None:
        lambda_ref_nm = float(np.mean(lambdas))
    fy, fx = freq_grid(n, pitch_nm)
    fr = np.hypot(fy, fx)
    theta = np.arctan2(fy, fx)
    psf = np.zeros((n, n))
    for lam, w in zip(lambdas, weights):
        rho = fr * lam / na
        sup = rho <= 1.0
        opd_waves = _zernike_phase(np.where(sup, rho, 0.0), theta, zernike_waves)
        # OPD specified in waves at lambda_ref -> phase = 2*pi*OPD*lambda_ref/lam
        phase = 2.0 * np.pi * opd_waves * (lambda_ref_nm / lam)
        pupil = np.where(sup, np.exp(1j * phase), 0.0)
        asf = np.fft.ifft2(pupil)
        p = np.abs(asf) ** 2
        psf += w * p / p.sum()
    psf = np.fft.fftshift(psf)
    return psf / psf.sum()
