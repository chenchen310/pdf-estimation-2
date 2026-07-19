"""Layer rasters -> region decomposition (Boolean combo x orientation class).

Every drawn feature is far below the optical resolution limit, so the image
only carries locally averaged effective reflectance. Regions are the units
that get one fitted effective weight each:

- combo: Boolean combination of the configured layers (bit j of the combo id
  = layer j present). Coverage rasters are area fractions, so combo
  membership is the product over layers of (cov or 1-cov) -- exact unless
  boundaries of different layers cross the same raster cell.
- orientation class: iso / h-lines / v-lines from the summed structure tensor
  of all layers. Needed because VN (linearly polarized) illumination makes
  the effective reflectance of dense sub-lambda gratings anisotropic
  (form birefringence); h and v arrays of the same combo differ.

Region coverages partition unity cell-by-cell, so the fitted weights live on
an absolute intensity scale.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

ORIENT_NAMES = ("iso", "h", "v")  # h = horizontal lines (gradients along y)


def combo_name(mask: int, layer_names) -> str:
    if mask == 0:
        return "field"
    return "+".join(n for j, n in enumerate(layer_names) if (mask >> j) & 1)


def region_ids(n_layers: int, use_orientation: bool) -> list:
    """All (combo_mask, orient_class) pairs in fixed order."""
    n_or = len(ORIENT_NAMES) if use_orientation else 1
    return [(m, o) for m in range(2 ** n_layers) for o in range(n_or)]


def region_name(rid, layer_names) -> str:
    m, o = rid
    base = combo_name(m, layer_names)
    return base if o == 0 else f"{base}:{ORIENT_NAMES[o]}"


def orientation_map(layers, cfg) -> np.ndarray:
    """Per-cell orientation class (uint8: 0 iso, 1 h-lines, 2 v-lines).

    Structure tensors of all layers are summed: crossed dense gratings
    (SRAM-like) lose coherence and correctly classify isotropic.
    """
    n = layers[0].shape[0]
    jyy = np.zeros((n, n), dtype=np.float32)
    jxx = np.zeros((n, n), dtype=np.float32)
    jxy = np.zeros((n, n), dtype=np.float32)
    for lay in layers:
        g = ndimage.gaussian_filter(lay, cfg.orient_grad_sigma)
        gy, gx = np.gradient(g)
        w = int(cfg.orient_window)
        jyy += ndimage.uniform_filter(gy * gy, w)
        jxx += ndimage.uniform_filter(gx * gx, w)
        jxy += ndimage.uniform_filter(gy * gx, w)
    energy = jxx + jyy
    aniso = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    coherence = aniso / (energy + 1e-12)
    floor = cfg.orient_energy_frac * np.percentile(energy, cfg.orient_energy_pct)
    oriented = (coherence > cfg.orient_coherence_min) & (energy > max(floor, 1e-12))
    cls = np.where(oriented, np.where(jyy > jxx, 1, 2), 0).astype(np.uint8)
    return cls


def iter_region_coverages(layers, cfg):
    """Yield (rid, float32 coverage raster), one region at a time (memory)."""
    ocls = orientation_map(layers, cfg) if cfg.use_orientation else None
    for rid in region_ids(len(layers), cfg.use_orientation):
        m, o = rid
        cov = np.ones_like(layers[0])
        for j, lay in enumerate(layers):
            cov = cov * (lay if (m >> j) & 1 else (1.0 - lay))
        if ocls is not None:
            cov = cov * (ocls == o)
        yield rid, cov
