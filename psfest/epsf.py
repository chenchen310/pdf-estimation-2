"""Stage 3: effective impulse response (ePSF) estimation.

Model per event i (signed diff crop D_i, P x P pixels, P = 2R+1):

    D_i(x) = a_i * h(x - c_i) + b_i + noise,   Var = sigma_i(x)^2

h lives on an oversample-by-s grid (M = 2*R*s + 1) and is normalized to unit
peak; the sign of the defect lives in a_i (dark defects: a_i < 0). Estimation
alternates per-event weighted fits of (a_i, b_i, c_i) with a weighted
accumulation update of h (Anderson & King style), followed by a radial
apodization and a projection onto the optical band limit (|f| <= 2NA/lambda_min
-- physics, not a tuning knob). Events with poor chi^2 are pruned, which doubles
as data cleaning since there are no trusted labels.

Because h is band-limited and sampled above Nyquist, Fourier shifting is an
exact interpolator: rendering h at any sub-pixel center is
ifft(fft(h) * phase_ramp) decimated by s. No smoothing bias is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.signal.windows import tukey

from .optics import project_bandlimit


@dataclass
class EventCrop:
    pair_id: str
    y: int
    x: int
    data: np.ndarray     # signed diff crop, P x P
    sigma: np.ndarray    # noise sigma crop, P x P
    sign: int
    i_local: float = 0.0
    # fit state
    a: float = 0.0
    b: float = 0.0
    cy: float = 0.0
    cx: float = 0.0
    chi2: float = np.inf
    rho: float = 0.0
    kept: bool = True
    drop_reason: str = ""


@dataclass
class EPSFResult:
    h: np.ndarray
    oversample: int
    window_radius: int
    events: list
    history: list = field(default_factory=list)
    h_raw: np.ndarray | None = None   # apodized but NOT band-limit projected


class EPSFModel:
    """Oversampled impulse response with exact band-limited rendering."""

    def __init__(self, h: np.ndarray, oversample: int, window_radius: int):
        self.h = h
        self.s = oversample
        self.R = window_radius
        M = h.shape[0]
        assert M == 2 * self.R * self.s + 1
        f = np.fft.fftfreq(M)
        self._fy = f[:, None]
        self._fx = f[None, :]
        self._H = np.fft.fft2(h)

    def refresh(self, h: np.ndarray) -> None:
        self.h = h
        self._H = np.fft.fft2(h)

    def render(self, cy: float, cx: float) -> np.ndarray:
        """Sample h(x - c) on the P x P pixel grid (P = 2R+1).

        The ramp shifts array content by +c*s cells, so cell u holds
        h(u - c*s); decimating at the pixel lattice u = center + k*s yields
        h(k - c) as required. (Sign checked against simulator ground truth:
        the opposite sign renders h(x + c) and mirrors every fitted center.)
        """
        ramp = np.exp(-2j * np.pi * (self._fy * (cy * self.s)
                                     + self._fx * (cx * self.s)))
        shifted = np.fft.ifft2(self._H * ramp).real
        return shifted[:: self.s, :: self.s]


def _weighted_ab(model_px: np.ndarray, D: np.ndarray, w: np.ndarray):
    """Closed-form weighted LS for D ~= a*model + b."""
    sw = w.sum()
    sh = (w * model_px).sum()
    shh = (w * model_px * model_px).sum()
    sd = (w * D).sum()
    shd = (w * model_px * D).sum()
    denom = sw * shh - sh * sh
    if abs(denom) < 1e-12:
        return 0.0, float(sd / max(sw, 1e-12))
    a = (sw * shd - sh * sd) / denom
    b = (sd - a * sh) / sw
    return float(a), float(b)


def _fit_ab_chi2(model: EPSFModel, ev: EventCrop, cy: float, cx: float,
                 huber_c: float):
    """IRLS (Huber) fit of (a, b) at fixed center; returns (a, b, chi2red, w)."""
    m = model.render(cy, cx)
    w = 1.0 / np.maximum(ev.sigma, 1e-6) ** 2
    a = b = 0.0
    for _ in range(3):
        a, b = _weighted_ab(m, ev.data, w)
        u = (ev.data - a * m - b) / np.maximum(ev.sigma, 1e-6)
        hub = np.minimum(1.0, huber_c / np.maximum(np.abs(u), 1e-9))
        w = hub / np.maximum(ev.sigma, 1e-6) ** 2
    u = (ev.data - a * m - b) / np.maximum(ev.sigma, 1e-6)
    hub = np.minimum(1.0, huber_c / np.maximum(np.abs(u), 1e-9))
    chi2 = float((hub * u * u).sum() / max(ev.data.size - 4, 1))
    return a, b, chi2, hub


def fit_event(model: EPSFModel, ev: EventCrop, cfg) -> None:
    """Optimize (cy, cx) with nested linear (a, b); updates the event in place."""
    def objective(c):
        _, _, chi2, _ = _fit_ab_chi2(model, ev, c[0], c[1], cfg.huber_c)
        return chi2

    lim = cfg.max_center_shift_px
    x0 = np.clip(np.array([ev.cy, ev.cx]), -lim + 1e-3, lim - 1e-3)
    res = minimize(objective, x0, method="Nelder-Mead",
                   bounds=[(-lim, lim), (-lim, lim)],
                   options={"xatol": 5e-3, "fatol": 1e-4, "maxiter": 80})
    ev.cy, ev.cx = float(res.x[0]), float(res.x[1])
    ev.a, ev.b, ev.chi2, _ = _fit_ab_chi2(model, ev, ev.cy, ev.cx, cfg.huber_c)
    m = model.render(ev.cy, ev.cx)
    num = ((ev.data - ev.b) * (ev.a * m)).sum()
    den = np.sqrt(((ev.data - ev.b) ** 2).sum() * ((ev.a * m) ** 2).sum())
    ev.rho = float(num / max(den, 1e-12))


# --- h construction -----------------------------------------------------------

def _fft_upsample(x: np.ndarray, s: int) -> np.ndarray:
    """Zero-padded FFT upsampling (exact for band-limited x). Odd sizes only."""
    n = x.shape[0]
    m = n * s
    F = np.fft.fftshift(np.fft.fft2(x))
    G = np.zeros((m, m), dtype=complex)
    lo = (m - n) // 2
    G[lo: lo + n, lo: lo + n] = F
    return np.fft.ifft2(np.fft.ifftshift(G)).real * (s * s)


def _fourier_shift(x: np.ndarray, dy: float, dx: float) -> np.ndarray:
    f = np.fft.fftfreq(x.shape[0])
    ramp = np.exp(-2j * np.pi * (f[:, None] * dy + f[None, :] * dx))
    return np.fft.ifft2(np.fft.fft2(x) * ramp).real


def _radial_taper(M: int, cfg) -> np.ndarray:
    s = cfg.oversample
    R = cfg.window_radius()
    c = M // 2
    yy, xx = np.mgrid[0:M, 0:M]
    rr = np.hypot(yy - c, xx - c)
    r0 = cfg.apodize_start * R * s
    r1 = R * s
    taper = np.ones((M, M))
    band = (rr > r0) & (rr <= r1)
    taper[band] = np.cos(0.5 * np.pi * (rr[band] - r0) / (r1 - r0)) ** 2
    taper[rr > r1] = 0.0
    return taper


def _postprocess_h(h: np.ndarray, cfg) -> np.ndarray:
    """Apodize, project on the optical band limit, recenter, normalize."""
    s = cfg.oversample
    M = h.shape[0]
    c = M // 2
    yy, xx = np.mgrid[0:M, 0:M]
    rr = np.hypot(yy - c, xx - c)
    h = h * _radial_taper(M, cfg)

    h = project_bandlimit(h, cfg.optics.pixel_nm / s, cfg.optics.na,
                          cfg.optics.lambda_min_nm, cfg.bandlimit_guard)

    # recenter on the (sub-cell) peak of |h| within the core region
    core = rr <= 2.5 * cfg.optics.fwhm_px * s
    ah = np.where(core, np.abs(h), 0.0)
    py, px = np.unravel_index(np.argmax(ah), ah.shape)

    def _para(fm, f0, fp):
        d = fm - 2 * f0 + fp
        return 0.0 if abs(d) < 1e-12 else 0.5 * (fm - fp) / d

    dy = _para(ah[py - 1, px], ah[py, px], ah[py + 1, px])
    dx = _para(ah[py, px - 1], ah[py, px], ah[py, px + 1])
    off_y = py + dy - c
    off_x = px + dx - c
    if np.hypot(off_y, off_x) > 0.3:
        h = _fourier_shift(h, -off_y, -off_x)

    peak = h.flat[np.argmax(np.where(core, np.abs(h), 0.0))]
    if abs(peak) < 1e-12:
        raise RuntimeError("ePSF collapsed to zero")
    return h / peak


def initial_h(events: list, cfg) -> np.ndarray:
    """Shift-and-add initialization from tapered, FFT-upsampled crops."""
    s = cfg.oversample
    R = cfg.window_radius()
    P = 2 * R + 1
    M = 2 * R * s + 1
    win = tukey(P, 0.25)[:, None] * tukey(P, 0.25)[None, :]
    num = np.zeros((M, M))
    den = 0.0
    for ev in events:
        q = (ev.data - ev.b) / ev.a
        up = _fft_upsample(q * win, s)                    # (P*s, P*s)
        up = _fourier_shift(up, -ev.cy * s, -ev.cx * s)
        lo = (P * s - M) // 2
        w = (ev.a / np.median(ev.sigma)) ** 2
        num += w * up[lo: lo + M, lo: lo + M]
        den += w
    return _postprocess_h(num / den, cfg)


def init_event_params(ev: EventCrop, cfg) -> bool:
    """Local background, amplitude and centroid initialization."""
    P = ev.data.shape[0]
    r = P // 2
    yy, xx = np.mgrid[-r: r + 1, -r: r + 1]
    rr = np.hypot(yy, xx)
    ring = rr > 0.8 * r
    ev.b = float(np.median(ev.data[ring]))
    q = ev.data - ev.b
    ev.a = float(q[r, r])
    if abs(ev.a) < 1e-6:
        return False
    w = np.maximum(np.abs(q) / np.maximum(ev.sigma, 1e-6) - 3.0, 0.0) ** 2
    w[rr > 2.0 * cfg.optics.fwhm_px] = 0.0
    if w.sum() <= 0:
        return False
    ev.cy = float((w * yy).sum() / w.sum())
    ev.cx = float((w * xx).sum() / w.sum())
    return True


def accumulate_h(events: list, model: EPSFModel, cfg) -> np.ndarray:
    """Anderson-King weighted accumulation of (D-b)/a onto the oversampled grid."""
    s, R = cfg.oversample, cfg.window_radius()
    M = 2 * R * s + 1
    P = 2 * R + 1
    yy, xx = np.mgrid[-R: R + 1, -R: R + 1].astype(float)
    num = np.zeros(M * M)
    den = np.zeros(M * M)
    for ev in events:
        if not ev.kept:
            continue
        m = model.render(ev.cy, ev.cx)
        u = (ev.data - ev.a * m - ev.b) / np.maximum(ev.sigma, 1e-6)
        hub = np.minimum(1.0, cfg.huber_c / np.maximum(np.abs(u), 1e-9))
        w = hub * (ev.a / np.maximum(ev.sigma, 1e-6)) ** 2
        q = (ev.data - ev.b) / ev.a
        iy = np.rint((yy - ev.cy) * s).astype(int) + R * s
        ix = np.rint((xx - ev.cx) * s).astype(int) + R * s
        ok = (iy >= 0) & (iy < M) & (ix >= 0) & (ix < M)
        flat = (iy * M + ix)[ok]
        np.add.at(num, flat, (w * q)[ok])
        np.add.at(den, flat, w[ok])
    h_new = model.h.copy().ravel()
    good = den > 0
    h_new[good] = num[good] / den[good]
    return h_new.reshape(M, M)


def run_epsf(events: list, cfg) -> EPSFResult:
    events = [ev for ev in events if init_event_params(ev, cfg)]
    if len(events) < cfg.min_events:
        raise RuntimeError(f"only {len(events)} usable events "
                           f"(min {cfg.min_events})")
    h = initial_h(events, cfg)
    model = EPSFModel(h, cfg.oversample, cfg.window_radius())
    history = []
    for it in range(cfg.epsf_iters):
        for ev in events:
            if ev.kept:
                fit_event(model, ev, cfg)
                if np.hypot(ev.cy, ev.cx) > cfg.max_center_shift_px - 1e-6:
                    ev.kept = False
                    ev.drop_reason = "center_runaway"
        if it >= cfg.prune_start_iter:
            kept = [ev for ev in events if ev.kept]
            for ev in kept:
                if ev.chi2 > cfg.chi2_prune:
                    if sum(e.kept for e in events) > cfg.min_events:
                        ev.kept = False
                        ev.drop_reason = "chi2"
        h_new = _postprocess_h(accumulate_h(events, model, cfg), cfg)
        delta = float(np.linalg.norm(h_new - model.h) / np.linalg.norm(model.h))
        model.refresh(h_new)
        kept_evs = [ev for ev in events if ev.kept]
        history.append({
            "iter": it, "n_kept": len(kept_evs),
            "median_chi2": float(np.median([ev.chi2 for ev in kept_evs])),
            "delta_h": delta,
        })
        if delta < cfg.epsf_tol and it >= 3:
            break
    # diagnostic: raw accumulation without the band-limit projection, so the
    # validation stage can report how much energy the data placed out of band
    raw = accumulate_h(events, model, cfg) * _radial_taper(model.h.shape[0], cfg)
    peak = raw.flat[np.argmax(np.abs(raw))]
    h_raw = raw / peak if abs(peak) > 1e-12 else raw

    # final consistent fit of every event (including previously dropped)
    for ev in events:
        fit_event(model, ev, cfg)
    return EPSFResult(model.h, cfg.oversample, cfg.window_radius(),
                      events, history, h_raw=h_raw)


def save_epsf(path: str, res: EPSFResult, cfg) -> None:
    np.savez(path, h=res.h, oversample=res.oversample,
             window_radius=res.window_radius,
             pixel_nm=cfg.optics.pixel_nm, na=cfg.optics.na,
             lambda_min_nm=cfg.optics.lambda_min_nm,
             lambda_max_nm=cfg.optics.lambda_max_nm)


def load_epsf(path: str):
    z = np.load(path)
    return z["h"], int(z["oversample"]), int(z["window_radius"])
