"""Stage 2: event detection on the normalized difference and point-source
screening. There are no trusted labels, so every gate is recorded per event and
the pipeline emits galleries for human spot checks.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def z_map(diff: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return diff / np.maximum(sigma, 1e-6)


def find_peaks(z: np.ndarray, tau: float, min_sep: int, margin: int):
    """Local extrema of both polarities with |z| >= tau, away from borders."""
    peaks = []
    az = np.abs(z)
    mf = ndimage.maximum_filter(az, size=min_sep, mode="nearest")
    cand = (az == mf) & (az >= tau)
    cand[:margin, :] = cand[-margin:, :] = False
    cand[:, :margin] = cand[:, -margin:] = False
    ys, xs = np.nonzero(cand)
    order = np.argsort(-az[ys, xs])
    for y, x in zip(ys[order], xs[order]):
        peaks.append((int(y), int(x), float(z[y, x])))
    return peaks


def _crop(img: np.ndarray, y: int, x: int, r: int) -> np.ndarray:
    return img[y - r: y + r + 1, x - r: x + r + 1]


def screen_event(zc: np.ndarray, defc: np.ndarray, refc: np.ndarray,
                 cfg) -> tuple[bool, str, dict]:
    """Point-source gates on a candidate crop (centered on the peak)."""
    r = zc.shape[0] // 2
    yy, xx = np.mgrid[-r: r + 1, -r: r + 1]
    rr = np.hypot(yy, xx)
    fwhm = cfg.optics.fwhm_px

    feats: dict = {}
    if defc.max() >= cfg.saturation_level or refc.max() >= cfg.saturation_level:
        return False, "saturated", feats

    w = np.maximum(np.abs(zc) - 2.0, 0.0) ** 2
    w_in = w[rr <= cfg.conc_r_in_factor * fwhm].sum()
    w_out = w[rr <= cfg.conc_r_out_factor * fwhm].sum()
    conc = w_in / (w_out + 1e-9)
    feats["concentration"] = conc
    if conc < cfg.conc_min:
        return False, "not_concentrated", feats

    wt = w.sum()
    if wt <= 0:
        return False, "empty", feats
    cy = (w * yy).sum() / wt
    cx = (w * xx).sum() / wt
    rms = np.sqrt((w * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / wt)
    feats["rms_radius"] = rms
    if rms > cfg.max_rms_radius_factor * fwhm:
        return False, "extended", feats

    # single significant connected component
    sig = np.abs(zc) >= max(4.0, 0.6 * cfg.tau_detect)
    lab, nlab = ndimage.label(sig)
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, nlab + 1))
    n_sig = int((sizes >= 3).sum())
    feats["n_components"] = n_sig
    if n_sig > 1:
        return False, "multi_blob", feats

    # crowding: a second local peak comparable to the main one
    az = np.abs(zc).copy()
    az[rr <= 1.5] = 0.0
    if az.max() >= cfg.crowd_ratio * abs(zc[r, r]):
        mfy, mfx = np.unravel_index(np.argmax(az), az.shape)
        if np.hypot(mfy - r, mfx - r) > 2.0 * fwhm:
            return False, "crowded", feats

    return True, "", feats


def detect_pair(pair_id: str, diff: np.ndarray, ref_matched: np.ndarray,
                def_img: np.ndarray, sigma: np.ndarray, cfg):
    """Returns (events, is_clean, hot_yx) for one pair."""
    z = z_map(diff, sigma)
    r = cfg.window_radius()
    peaks = find_peaks(z, cfg.tau_detect, cfg.min_sep_px(), margin=r + 2)
    i_local_map = ndimage.uniform_filter(ref_matched, size=5)

    events = []
    for (y, x, zval) in peaks:
        sign = int(np.sign(zval))
        ev = {
            "pair_id": pair_id, "y": y, "x": x, "sign": sign,
            "peak_z": zval, "peak_diff": float(diff[y, x]),
            "i_local": float(i_local_map[y, x]),
            "is_psf_candidate": False, "reject_reason": "",
        }
        if abs(zval) >= cfg.tau_psf and (cfg.psf_polarity == 0
                                         or sign == cfg.psf_polarity):
            ok, reason, feats = screen_event(
                _crop(z, y, x, r), _crop(def_img, y, x, r),
                _crop(ref_matched, y, x, r), cfg)
            ev.update({f"gate_{k}": v for k, v in feats.items()})
            ev["is_psf_candidate"] = bool(ok)
            ev["reject_reason"] = reason
        else:
            ev["reject_reason"] = "below_tau_psf" if abs(zval) < cfg.tau_psf \
                else "wrong_polarity"
        events.append(ev)

    # clean pair: nothing anywhere above clean_tau (border included)
    is_clean = bool(np.abs(z).max() < cfg.clean_tau)
    hot_yx = np.argwhere(np.abs(z) > cfg.hot_tau)
    return events, is_clean, hot_yx
