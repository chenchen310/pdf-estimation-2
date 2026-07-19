"""Frame/layer discovery for FID_{index}_bbp.npy + FID_{index}_{layer}.npy.

Layer rasters are accepted as bool, integer (binary or 0..255 coverage) or
float (0..1 coverage) and normalized to float32 area coverage in [0, 1]. The
normalization is recorded so an unexpected input convention is visible in the
run log instead of silently rescaled.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

BBP_SUFFIX = "_bbp.npy"


@dataclass
class FramePaths:
    frame_id: str
    bbp_path: str
    layer_paths: dict  # layer name -> path


def discover_layer_names(data_dir: str) -> list:
    """Layer names present next to the first frame (suffix set minus bbp)."""
    bbps = sorted(glob.glob(os.path.join(data_dir, f"*{BBP_SUFFIX}")))
    if not bbps:
        return []
    base = os.path.basename(bbps[0])[: -len(BBP_SUFFIX)]
    names = []
    for p in sorted(glob.glob(os.path.join(data_dir, base + "_*.npy"))):
        suffix = os.path.basename(p)[len(base) + 1: -len(".npy")]
        if suffix != "bbp":
            names.append(suffix)
    return names


def find_frames(data_dir: str, layer_names) -> list:
    frames = []
    for bbp_path in sorted(glob.glob(os.path.join(data_dir, f"*{BBP_SUFFIX}"))):
        base = os.path.basename(bbp_path)[: -len(BBP_SUFFIX)]
        lp = {n: os.path.join(data_dir, f"{base}_{n}.npy") for n in layer_names}
        if all(os.path.exists(p) for p in lp.values()):
            frames.append(FramePaths(base, bbp_path, lp))
    return frames


def load_bbp(fp: FramePaths, frame_px: int) -> np.ndarray:
    img = np.load(fp.bbp_path).astype(np.float64)
    if img.shape != (frame_px, frame_px):
        raise ValueError(f"{fp.frame_id}: bbp shape {img.shape}, "
                         f"expected {(frame_px, frame_px)}")
    return img


def load_layers(fp: FramePaths, layer_names, hr_n: int) -> tuple:
    """Return (list of float32 coverage rasters in [0,1], convention string)."""
    layers, conv = [], None
    for name in layer_names:
        a = np.load(fp.layer_paths[name])
        if a.shape != (hr_n, hr_n):
            raise ValueError(f"{fp.frame_id}/{name}: raster shape {a.shape}, "
                             f"expected {(hr_n, hr_n)}")
        if a.dtype == bool:
            c, this = a.astype(np.float32), "bool"
        elif np.issubdtype(a.dtype, np.integer):
            mx = int(a.max()) if a.size else 0
            if mx <= 1:
                c, this = a.astype(np.float32), "int-binary"
            else:
                c, this = a.astype(np.float32) / 255.0, "uint8-coverage"
        else:
            c, this = np.clip(a.astype(np.float32), 0.0, 1.0), "float-coverage"
        if conv is None:
            conv = this
        elif conv != this:
            conv = "mixed"
        layers.append(c)
    return layers, conv
