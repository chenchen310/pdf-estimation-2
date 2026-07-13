"""Pipeline configuration.

All optics-dependent quantities are derived from OpticsConfig so that a future
optical-mode change only requires a new config, not code changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


@dataclass
class OpticsConfig:
    pixel_nm: float = 30.0
    na: float = 0.95
    lambda_min_nm: float = 190.0
    lambda_max_nm: float = 260.0

    @property
    def lambda_c_nm(self) -> float:
        return 0.5 * (self.lambda_min_nm + self.lambda_max_nm)

    @property
    def cutoff_cyc_per_px(self) -> float:
        """Incoherent optical cutoff (2NA/lambda_min) in cycles/pixel."""
        return 2.0 * self.na / self.lambda_min_nm * self.pixel_nm

    @property
    def fwhm_px(self) -> float:
        """Rough diffraction-limited PSF FWHM in pixels (0.5 * lambda_c / NA)."""
        return 0.5 * self.lambda_c_nm / self.na / self.pixel_nm


@dataclass
class PipelineConfig:
    optics: OpticsConfig = field(default_factory=OpticsConfig)

    # --- Stage 1: signed diff -------------------------------------------------
    gain_clip_sigma: float = 3.0
    gain_iters: int = 3
    max_resid_shift_px: float = 0.1   # QC gate on residual def/ref misalignment
    reregister: bool = True           # Fourier re-shift reference when gate trips

    # --- Stage 0: noise model -------------------------------------------------
    noise_pairs_max: int = 200
    noise_pixel_frac: float = 0.25
    noise_n_ibins: int = 12
    noise_n_gbins: int = 6
    noise_min_bin_count: int = 400
    pedestal_percentile: float = 0.2  # crude dark-level estimate from ref images

    # --- Stage 2: detection & screening ---------------------------------------
    tau_detect: float = 6.0           # |z| threshold for cataloged detections
    tau_psf: float = 10.0             # |z| threshold for PSF-estimation candidates
    clean_tau: float = 5.5            # pair is "clean" if no |z| peak above this
    hot_tau: float = 4.5              # exclusion zones for synthetic insertion
    psf_polarity: int = -1            # -1 dark, +1 bright, 0 both (defects are dark)
    min_sep_factor: float = 2.0       # peak min separation, in units of FWHM
    conc_r_in_factor: float = 1.5     # inner radius of concentration gate (x FWHM)
    conc_r_out_factor: float = 3.0    # outer radius (x FWHM)
    conc_min: float = 0.45            # min energy fraction inside r_in
    max_rms_radius_factor: float = 1.15  # max RMS radius of event weight map (x FWHM)
    crowd_ratio: float = 0.5          # second peak above this fraction => crowded
    saturation_level: float = 4090.0

    # --- Stage 3: ePSF estimation ----------------------------------------------
    window_radius_px: int | None = None  # default: round(3 * FWHM)
    oversample: int = 3
    epsf_iters: int = 8
    huber_c: float = 2.5
    chi2_prune: float = 3.0
    prune_start_iter: int = 3
    epsf_tol: float = 1e-4
    max_center_shift_px: float = 2.0
    min_events: int = 25
    bandlimit_guard: float = 1.08     # cutoff mask radius safety factor
    apodize_start: float = 0.85       # radial taper start (fraction of window radius)

    # --- Stage 5: synthesis -----------------------------------------------------
    synth_margin_px: int = 18
    synth_hot_dist_px: int = 16
    synth_weak_frac: float = 0.35     # fraction of samples pushed below detection SNR
    synth_weak_scale: tuple = (0.15, 1.0)
    synth_mask_k: float = 1.0         # mask = |a*h| > k * sigma_diff
    synth_mask_min_r: float = 1.5

    def window_radius(self) -> int:
        if self.window_radius_px is not None:
            return int(self.window_radius_px)
        return max(8, int(round(3.0 * self.optics.fwhm_px)))

    def min_sep_px(self) -> int:
        s = int(round(self.min_sep_factor * self.optics.fwhm_px))
        return max(5, s | 1)  # odd

    # --- serialization ----------------------------------------------------------
    def to_json(self, path) -> None:
        d = asdict(self)
        with open(path, "w") as f:
            json.dump(d, f, indent=2, default=list)

    @classmethod
    def from_json(cls, path) -> "PipelineConfig":
        with open(path) as f:
            d = json.load(f)
        opt = OpticsConfig(**d.pop("optics"))
        cfg = cls(optics=opt, **{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if isinstance(cfg.synth_weak_scale, list):
            cfg.synth_weak_scale = tuple(cfg.synth_weak_scale)
        return cfg
