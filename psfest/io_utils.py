"""Pair discovery and loading for DID_{index}_def.npy / DID_{index}_ref1.npy."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

DEF_SUFFIX = "_def.npy"
REF_SUFFIX = "_ref1.npy"


@dataclass
class PairPaths:
    pair_id: str
    def_path: str
    ref_path: str


def find_pairs(data_dir: str) -> list[PairPaths]:
    pairs = []
    for def_path in sorted(glob.glob(os.path.join(data_dir, f"*{DEF_SUFFIX}"))):
        base = os.path.basename(def_path)[: -len(DEF_SUFFIX)]
        ref_path = os.path.join(data_dir, base + REF_SUFFIX)
        if os.path.exists(ref_path):
            pairs.append(PairPaths(base, def_path, ref_path))
    return pairs


def load_pair(p: PairPaths) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(p.def_path).astype(np.float64)
    r = np.load(p.ref_path).astype(np.float64)
    if d.shape != r.shape or d.ndim != 2:
        raise ValueError(f"{p.pair_id}: shape mismatch {d.shape} vs {r.shape}")
    return d, r
