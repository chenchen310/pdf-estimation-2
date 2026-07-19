"""Configuration for the GDS -> BBP rendering (die-to-database) pipeline.

Same philosophy as psfest.config: optics-derived quantities come from
OpticsConfig, and a different process step / layer set / optical mode is a new
config JSON, never edited constants. The layer list is deliberately open-ended
(other stations will bring other GDS layers).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from psfest.config import OpticsConfig


@dataclass
class D2DBConfig:
    optics: OpticsConfig = field(default_factory=OpticsConfig)

    # --- GDS raster contract -------------------------------------------------
    # Layer rasters are area-coverage images at pixel_nm / s_gds per cell
    # (3.75 nm for 30 nm pixels, s_gds=8), one file per layer per frame:
    # FID_{index:05d}_{layer}.npy alongside FID_{index:05d}_bbp.npy.
    layer_names: tuple = ("OD", "POLY")   # order defines the combo bit order
    s_gds: int = 8                        # raster cells per BBP pixel
    frame_px: int = 256

    # --- region model ---------------------------------------------------------
    use_orientation: bool = True          # VN polarization: dense sub-lambda
    orient_grad_sigma: float = 2.0        # gratings get orientation-dependent
    orient_window: int = 12               # effective reflectance
    orient_coherence_min: float = 0.5
    orient_energy_pct: float = 90.0
    orient_energy_frac: float = 0.10
    min_region_area: float = 0.002        # smaller combos -> "rare", masked out
    rare_mask_thresh: float = 0.02        # pixel excluded if rare density above

    # --- Stage-0 fit ------------------------------------------------------------
    n_lambda: int = 6                     # spectral samples for the nominal OTF
    fit_frames_max: int = 150             # memory cap for the calibration set
    outer_iters: int = 5
    clip_sigma: float = 3.5               # sigma-clip (defects must not bias w)
    max_shift_px: int = 4                 # integer design->image search radius
    fit_sigma_extra: bool = True          # Gaussian inflation of the nominal PSF
    sigma_extra_max_px: float = 2.0
    ridge: float = 1e-6                   # relative Tikhonov on the Gram matrix
    border_margin_px: int = 12            # FFT wrap + kernel tails exclusion

    # --- evaluation --------------------------------------------------------------
    noise_annulus_min: float = 0.42       # cyc/px; beyond any optical content
    nuisance_taus: tuple = (4.0, 5.0, 6.0)

    def hr_n(self) -> int:
        return self.frame_px * self.s_gds

    def pitch_gds_nm(self) -> float:
        return self.optics.pixel_nm / self.s_gds

    # --- serialization -----------------------------------------------------------
    def to_json(self, path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=list)

    @classmethod
    def from_json(cls, path) -> "D2DBConfig":
        with open(path) as f:
            d = json.load(f)
        opt = OpticsConfig(**d.pop("optics"))
        cfg = cls(optics=opt,
                  **{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cfg.layer_names = tuple(cfg.layer_names)
        cfg.nuisance_taus = tuple(cfg.nuisance_taus)
        return cfg
