"""Real-vs-synthetic acceptance test.

Trains a ridge-regularized logistic classifier (on PCA features of
noise-normalized diff crops) to distinguish real detected defects from
SNR-matched synthetic ones. Cross-validated AUC near 0.5 means the synthetic
population is statistically indistinguishable from the real one -- the
acceptance criterion for U-Net training data.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psfest.config import PipelineConfig  # noqa: E402
from psfest import io_utils  # noqa: E402
from psfest.diffimg import compute_diff  # noqa: E402
from psfest.noise import NoiseModel  # noqa: E402


def collect_crops(args, cfg, nm):
    R = cfg.window_radius()
    with open(os.path.join(args.run_dir, "run_summary.json")) as f:
        data_dir = json.load(f)["data_dir"]
    pairs = {p.pair_id: p for p in io_utils.find_pairs(data_dir)}

    real, real_snr = [], []
    with open(os.path.join(args.run_dir, "events.csv")) as f:
        evs = [r for r in csv.DictReader(f) if r.get("fit_kept") == "1"]
    for r in evs:
        p = pairs[r["pair_id"]]
        d, rf = io_utils.load_pair(p)
        dres = compute_diff(d, rf, cfg)
        y, x = int(r["y"]), int(r["x"])
        sig = float(nm.sigma_diff_at(float(r["i_local"])))
        real.append(dres.diff[y - R: y + R + 1, x - R: x + R + 1] / sig)
        real_snr.append(abs(float(r["fit_a"])) / sig)

    syn, syn_snr = [], []
    with open(os.path.join(args.synth_dir, "catalog.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["has_defect"] == "1"]
    for r in rows:
        dif = np.load(os.path.join(args.synth_dir,
                                   f"SYN_{int(r['index']):05d}_diff.npy"))
        y, x = int(round(float(r["y"]))), int(round(float(r["x"])))
        sig = float(nm.sigma_diff_at(float(r["i_local"])))
        syn.append(dif[y - R: y + R + 1, x - R: x + R + 1].astype(float) / sig)
        syn_snr.append(abs(float(r["a"])) / sig)

    # SNR-match: for each real event pick the unused synthetic closest in SNR
    order = np.argsort(real_snr)
    used = set()
    Xr, Xs = [], []
    for i in order:
        js = min((j for j in range(len(syn)) if j not in used),
                 key=lambda j: abs(syn_snr[j] - real_snr[i]), default=None)
        if js is None:
            break
        if abs(syn_snr[js] - real_snr[i]) > 0.3 * real_snr[i] + 2.0:
            continue
        used.add(js)
        Xr.append(real[i].ravel())
        Xs.append(syn[js].ravel())
    return np.array(Xr), np.array(Xs)


def logistic_cv_auc(X, ylab, n_pc=30, lam=1.0, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, k)
    aucs = []
    for fi in range(k):
        te = folds[fi]
        tr = np.concatenate([folds[j] for j in range(k) if j != fi])
        mu = X[tr].mean(axis=0)
        Xc = X - mu
        # PCA on train
        _, _, Vt = np.linalg.svd(Xc[tr], full_matrices=False)
        P = Vt[:n_pc].T
        Ztr, Zte = Xc[tr] @ P, Xc[te] @ P
        sd = Ztr.std(axis=0) + 1e-9
        Ztr, Zte = Ztr / sd, Zte / sd
        w = np.zeros(n_pc + 1)
        A_tr = np.hstack([Ztr, np.ones((len(Ztr), 1))])
        yt = ylab[tr]
        for _ in range(200):  # Newton-IRLS
            p = 1.0 / (1.0 + np.exp(-A_tr @ w))
            g = A_tr.T @ (p - yt) + lam * np.r_[w[:-1], 0]
            Wd = p * (1 - p)
            H = (A_tr * Wd[:, None]).T @ A_tr + lam * np.eye(n_pc + 1)
            step = np.linalg.solve(H, g)
            w -= step
            if np.linalg.norm(step) < 1e-8:
                break
        s = np.hstack([Zte, np.ones((len(Zte), 1))]) @ w
        yv = ylab[te]
        pos, neg = s[yv == 1], s[yv == 0]
        if len(pos) and len(neg):
            auc = np.mean(pos[:, None] > neg[None, :]) \
                + 0.5 * np.mean(pos[:, None] == neg[None, :])
            aucs.append(auc)
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--synth-dir", required=True)
    args = ap.parse_args()
    cfg = PipelineConfig.from_json(os.path.join(args.run_dir, "config.json"))
    nm = NoiseModel.load(os.path.join(args.run_dir, "noise_model.npz"))
    Xr, Xs = collect_crops(args, cfg, nm)
    print(f"matched {len(Xr)} real vs {len(Xs)} synthetic crops")
    X = np.vstack([Xr, Xs])
    y = np.r_[np.ones(len(Xr)), np.zeros(len(Xs))]
    auc, sd = logistic_cv_auc(X, y)
    print(json.dumps({"n_real": len(Xr), "n_synth": len(Xs),
                      "cv_auc": round(auc, 3), "cv_auc_std": round(sd, 3),
                      "target": "~0.5 (indistinguishable)"}, indent=2))


if __name__ == "__main__":
    main()
