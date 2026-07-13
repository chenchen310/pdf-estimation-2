"""Figures and human-spot-check galleries (matplotlib, Agg backend)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .epsf import EPSFModel
from .validate import mtf_curve, radial_profile


def save_noise_plot(nm, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for gi, g in enumerate(nm.g_centers):
        ax.plot(nm.i_centers, nm.sigma_grid[gi], marker="o", ms=3,
                label=f"|grad| ~ {g:.0f}")
    ax.plot(nm.i_centers, nm.sigma_i_lowg, "k--", lw=2, label="low-grad (sensor)")
    ax.set_xlabel("reference intensity (ADU)")
    ax.set_ylabel(r"$\sigma_{diff}$ (ADU)")
    ax.legend(fontsize=7)
    ax.set_title("noise model")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_qc_plot(qc_rows: list, path: str) -> None:
    sh = np.array([[r["shift_dy"], r["shift_dx"]] for r in qc_rows])
    al = np.array([r["alpha"] for r in qc_rows])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].hist(np.hypot(sh[:, 0], sh[:, 1]), bins=40)
    axes[0].set_xlabel("|residual shift| (px)")
    axes[0].set_title("def/ref residual misalignment")
    axes[1].hist(al, bins=40)
    axes[1].set_xlabel("alpha (gain match)")
    axes[1].set_title("photometric gain")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_gallery(events: list, h: np.ndarray, cfg, path: str,
                 n_show: int = 12) -> None:
    """Real | model | residual triplets for the strongest kept events."""
    kept = sorted([ev for ev in events if ev.kept], key=lambda e: -abs(e.a))
    kept = kept[:n_show]
    if not kept:
        return
    model = EPSFModel(h, cfg.oversample, cfg.window_radius())
    fig, axes = plt.subplots(len(kept), 3, figsize=(6.5, 2.1 * len(kept)))
    axes = np.atleast_2d(axes)
    for i, ev in enumerate(kept):
        m = ev.a * model.render(ev.cy, ev.cx) + ev.b
        v = max(abs(ev.data.min()), abs(ev.data.max()))
        for j, (img, name) in enumerate([(ev.data, "diff"), (m, "model"),
                                         (ev.data - m, "residual")]):
            ax = axes[i, j]
            ax.imshow(img, cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=9)
        axes[i, 0].set_ylabel(f"{ev.pair_id}\na={ev.a:.0f} chi2={ev.chi2:.1f}",
                              fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_reject_gallery(events_px: list, path: str, n_show: int = 12) -> None:
    """Crops of rejected PSF candidates, for threshold tuning by eye."""
    if not events_px:
        return
    n = min(n_show, len(events_px))
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (crop, label) in zip(axes, events_px[:n]):
        v = max(abs(crop.min()), abs(crop.max()), 1e-9)
        ax.imshow(crop, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(label, fontsize=6)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_psf_figs(h: np.ndarray, cfg, out_dir: str, prefix: str = "psf") -> None:
    s = cfg.oversample
    opt = cfg.optics
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    v = np.abs(h).max()
    im = axes[0].imshow(h, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[0].set_title("h (oversampled grid)")
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    im = axes[1].imshow(np.log10(np.abs(h) + 1e-5), cmap="magma")
    axes[1].set_title("log10 |h|")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_image.png"), dpi=130)
    plt.close(fig)

    r, p = radial_profile(h, s)
    f, m = mtf_curve(h, s)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(r * opt.pixel_nm, p / p.max())
    axes[0].axhline(0.5, color="gray", ls=":")
    axes[0].set_xlim(0, 6 * opt.fwhm_px * opt.pixel_nm)
    axes[0].set_xlabel("radius (nm)")
    axes[0].set_title("radial profile")
    axes[1].plot(f, m)
    axes[1].axvline(opt.cutoff_cyc_per_px, color="r", ls="--",
                    label=r"$2NA/\lambda_{min}$")
    axes[1].set_xlabel("frequency (cycles/px)")
    axes[1].set_xlim(0, min(1.0, 1.6 * opt.cutoff_cyc_per_px))
    axes[1].set_title("MTF (azimuthal avg)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_profile_mtf.png"), dpi=130)
    plt.close(fig)


def save_amplitude_plot(events: list, path: str) -> None:
    kept = [ev for ev in events if ev.kept]
    if not kept:
        return
    a = np.array([ev.a for ev in kept])
    il = np.array([ev.i_local for ev in kept])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(il, a, s=10, alpha=0.6)
    ax.set_xlabel("local background intensity (ADU)")
    ax.set_ylabel("fitted amplitude a (ADU)")
    ax.set_title("amplitude vs background (cross-term check)")
    ax.axhline(0, color="gray", lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_synth_gallery(out_dir: str, rows: list, R: int, path: str,
                       n_show: int = 8) -> None:
    """def/ref/diff/mask crops around inserted synthetic defects."""
    import os as _os
    with_def = [r for r in rows if r["has_defect"]]
    if not with_def:
        return
    sel = sorted(with_def, key=lambda r: -abs(r["a"]))[::max(
        1, len(with_def) // n_show)][:n_show]
    fig, axes = plt.subplots(len(sel), 4, figsize=(8.6, 2.1 * len(sel)))
    axes = np.atleast_2d(axes)
    w = R + 4
    for i, row in enumerate(sel):
        base = _os.path.join(out_dir, f"SYN_{row['index']:05d}")
        y, x = int(round(row["y"])), int(round(row["x"]))
        imgs = [np.load(base + "_def.npy").astype(float),
                np.load(base + "_ref1.npy").astype(float),
                np.load(base + "_diff.npy").astype(float),
                np.load(base + "_mask.npy").astype(float)]
        names = ["def", "ref", "diff", "mask"]
        for j, (img, name) in enumerate(zip(imgs, names)):
            crop = img[y - w: y + w + 1, x - w: x + w + 1]
            ax = axes[i, j]
            if name == "diff":
                v = max(abs(crop.min()), abs(crop.max()), 1e-9)
                ax.imshow(crop, cmap="RdBu_r", vmin=-v, vmax=v)
            elif name == "mask":
                ax.imshow(crop, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(crop, cmap="gray")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=9)
        axes[i, 0].set_ylabel(f"a={row['a']:.0f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_real_vs_synth(real_crops: list, synth_crops: list, path: str) -> None:
    """Side-by-side diff crops: do synthetic defects look real?"""
    n = min(len(real_crops), len(synth_crops), 8)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.2))
    for j in range(n):
        for i, crop in enumerate([real_crops[j], synth_crops[j]]):
            v = max(abs(crop.min()), abs(crop.max()), 1e-9)
            axes[i, j].imshow(crop, cmap="RdBu_r", vmin=-v, vmax=v)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    axes[0, 0].set_ylabel("real", fontsize=10)
    axes[1, 0].set_ylabel("synthetic", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_chi2_rho_plot(events: list, path: str) -> None:
    kept = [ev for ev in events if ev.kept]
    if not kept:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].hist([ev.chi2 for ev in kept], bins=40)
    axes[0].set_xlabel("reduced chi2 (kept events)")
    axes[1].hist([ev.rho for ev in kept], bins=40)
    axes[1].set_xlabel("correlation with h (shape-family check)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
