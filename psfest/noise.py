"""Stage 0: data-driven noise model for the signed difference image.

sigma_diff is modeled as a function of (reference intensity, reference gradient
magnitude): shot noise depends on intensity; residual-misregistration noise
concentrates on pattern edges, hence the gradient axis. Estimates use the MAD
so sparse real defects do not bias the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def grad_mag(img: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(img)
    return np.hypot(gy, gx)


def _mad_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


@dataclass
class NoiseModel:
    i_centers: np.ndarray      # intensity bin centers
    g_centers: np.ndarray      # gradient bin centers
    sigma_grid: np.ndarray     # (n_gbins, n_ibins) sigma_diff
    sigma_i_lowg: np.ndarray   # sigma_diff(I) using low-gradient pixels only
    pedestal: float            # crude dark-level estimate (counts)

    def _interp(self):
        return RegularGridInterpolator(
            (self.g_centers, self.i_centers), self.sigma_grid,
            bounds_error=False, fill_value=None)

    def sigma_map(self, ref_matched: np.ndarray) -> np.ndarray:
        gi = self._interp()
        i = np.clip(ref_matched, self.i_centers[0], self.i_centers[-1])
        g = np.clip(grad_mag(ref_matched), self.g_centers[0], self.g_centers[-1])
        return gi(np.stack([g.ravel(), i.ravel()], axis=1)).reshape(ref_matched.shape)

    def sigma_diff_at(self, intensity) -> np.ndarray:
        """Low-gradient (sensor-only) diff noise at the given intensity."""
        return np.interp(np.asarray(intensity, dtype=float),
                         self.i_centers, self.sigma_i_lowg)

    def sigma_image_at(self, intensity) -> np.ndarray:
        """Single-image sensor noise (diff noise / sqrt(2))."""
        return self.sigma_diff_at(intensity) / np.sqrt(2.0)

    def save(self, path: str) -> None:
        np.savez(path, i_centers=self.i_centers, g_centers=self.g_centers,
                 sigma_grid=self.sigma_grid, sigma_i_lowg=self.sigma_i_lowg,
                 pedestal=self.pedestal)

    @classmethod
    def load(cls, path: str) -> "NoiseModel":
        z = np.load(path)
        return cls(z["i_centers"], z["g_centers"], z["sigma_grid"],
                   z["sigma_i_lowg"], float(z["pedestal"]))


def build_noise_model(samples_i: np.ndarray, samples_g: np.ndarray,
                      samples_d: np.ndarray, cfg,
                      pedestal: float) -> NoiseModel:
    """Build the model from pooled (I, |grad I|, diff) samples."""
    ni, ng = cfg.noise_n_ibins, cfg.noise_n_gbins
    i_edges = np.quantile(samples_i, np.linspace(0, 1, ni + 1))
    g_edges = np.quantile(samples_g, np.linspace(0, 1, ng + 1))
    # guard against duplicate edges on flat backgrounds
    i_edges = np.unique(i_edges)
    g_edges = np.unique(g_edges)
    ni, ng = len(i_edges) - 1, len(g_edges) - 1
    i_idx = np.clip(np.searchsorted(i_edges, samples_i, side="right") - 1, 0, ni - 1)
    g_idx = np.clip(np.searchsorted(g_edges, samples_g, side="right") - 1, 0, ng - 1)

    sigma_grid = np.zeros((ng, ni))
    counts = np.zeros((ng, ni), dtype=int)
    i_centers = np.zeros(ni)
    g_centers = np.zeros(ng)
    for gi in range(ng):
        for ii in range(ni):
            sel = (g_idx == gi) & (i_idx == ii)
            counts[gi, ii] = sel.sum()
            if counts[gi, ii] >= cfg.noise_min_bin_count:
                sigma_grid[gi, ii] = _mad_sigma(samples_d[sel])
    for ii in range(ni):
        sel = i_idx == ii
        i_centers[ii] = np.median(samples_i[sel]) if sel.any() else 0.5 * (
            i_edges[ii] + i_edges[ii + 1])
    for gi in range(ng):
        sel = g_idx == gi
        g_centers[gi] = np.median(samples_g[sel]) if sel.any() else 0.5 * (
            g_edges[gi] + g_edges[gi + 1])

    # marginal sigma(I) over all gradients, used to fill sparse bins
    sigma_i_all = np.zeros(ni)
    for ii in range(ni):
        sel = i_idx == ii
        sigma_i_all[ii] = _mad_sigma(samples_d[sel]) if sel.sum() > 50 else np.nan
    sigma_i_all = _fill_nan_interp(i_centers, sigma_i_all)
    for gi in range(ng):
        empty = sigma_grid[gi] == 0
        sigma_grid[gi, empty] = sigma_i_all[empty]

    # low-gradient marginal: sensor-dominated noise curve for synthesis
    n_low = max(1, ng // 3)
    low = g_idx < n_low
    sigma_i_lowg = np.zeros(ni)
    for ii in range(ni):
        sel = low & (i_idx == ii)
        sigma_i_lowg[ii] = _mad_sigma(samples_d[sel]) if sel.sum() > 50 else np.nan
    sigma_i_lowg = _fill_nan_interp(i_centers, sigma_i_lowg)

    return NoiseModel(i_centers, g_centers, sigma_grid, sigma_i_lowg, pedestal)


def _fill_nan_interp(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = y.copy()
    bad = ~np.isfinite(y)
    if bad.all():
        raise ValueError("noise model: no bin had enough samples")
    y[bad] = np.interp(x[bad], x[~bad], y[~bad])
    return y
