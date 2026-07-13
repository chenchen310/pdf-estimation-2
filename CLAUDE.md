# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment quirks (read first)

System `python3` **and** system `git` are broken on this machine (`xcrun: invalid active developer path` — missing Command Line Tools). Always use:

- Python: `.venv/bin/python` (never bare `python3`)
- Git: `/opt/homebrew/bin/git` (never bare `git`)

Rebuild the venv if missing:
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python numpy scipy matplotlib
```
Dependencies are numpy/scipy/matplotlib only — no sklearn/torch/skimage; keep it that way unless asked.

The user communicates in Traditional Chinese and has asked to be *asked* about uncertain data conventions rather than have them guessed.

## Commands

```bash
# Generate simulated test data (DID_{i:05d}_def.npy / _ref1.npy + truth/)
.venv/bin/python scripts/make_test_data.py --out data/sim --n 650 --seed 0

# Run the estimation pipeline (works identically on real fab data dirs)
.venv/bin/python scripts/run_pipeline.py --data-dir data/sim --out-dir runs/sim650 --kfold

# Score a run against simulator ground truth (sim data only)
.venv/bin/python scripts/evaluate_sim.py --run-dir runs/sim650 --data-dir data/sim

# Generate synthetic U-Net training data (SYN_*_def/_ref1/_diff/_mask.npy)
.venv/bin/python scripts/run_synth.py --run-dir runs/sim650 --out-dir data/synth --n 600

# Real-vs-synthetic acceptance test (target: CV AUC ~ 0.5)
.venv/bin/python scripts/turing_test.py --run-dir runs/sim650 --synth-dir data/synth
```

There is no unit-test suite. Verification is end-to-end: generate sim data → run pipeline → `evaluate_sim.py` (the pipeline never reads `truth/`; the simulator in `sim/` deliberately uses its own 6x-oversampled optical chain so the estimator can't "cheat"). For a quick smoke loop use a small set:
```bash
.venv/bin/python scripts/make_test_data.py --out /tmp/smoke --n 40 --seed 7
.venv/bin/python scripts/run_pipeline.py --data-dir /tmp/smoke --out-dir /tmp/run_smoke --min-events 6
```
Reference scores on 650 sim pairs (regressions should stay near these): PSF NRMSE ≈ 2.3%, position RMSE ≈ 0.12 px, event purity 100%, held-out chi2 median ≈ 1.0, Turing-test AUC ≈ 0.57.

## Architecture

Purpose: estimate the effective impulse response `h` of point defects from KLA BBP inspection image pairs, then paste synthetic defects into clean images to train a detection U-Net. Per-event model: `D_i(x) = a_i·h(x−c_i) + b_i + noise`, with `h` unit-peak on an oversample-by-s grid (M = 2·R·s+1, default R=12, s=3) and the sign in `a_i` (defects are dark ⇒ a<0; `psf_polarity=-1` default).

Stage flow in `run_pipeline.py`, one module per stage in `psfest/`:

1. `io_utils` — pair discovery by naming convention `DID_{index}_def.npy` / `_ref1.npy` (uint16, 12-bit, 256×256).
2. `diffimg` — robust gain/offset match, then **signed** diff (diff is computed here, never loaded); residual-shift QC re-registers when >0.1 px.
3. `noise` — `sigma_diff(I, |grad I|)` from MAD-binned samples; the low-gradient marginal is the sensor curve used by synthesis.
4. `detect` — z-map peaks + point-source gates (concentration, RMS radius, single blob, saturation, crowding). No trusted labels exist, so gates are recorded per event and rejects go to a gallery for human review.
5. `epsf` — alternating estimation: per-event (a,b,c) fits (Nelder-Mead over c, closed-form a,b, Huber IRLS) ↔ Anderson–King accumulation of h, followed by apodize → **band-limit projection** → recenter → renormalize. chi2 pruning doubles as data cleaning.
6. `validate` + `report` — k-fold held-out fits, FWHM vs theory, band-limit violation of the *raw* accumulation (the projected h is band-limited by construction — never use its out-of-band energy as a metric), figures.
7. `synth` — samples *strength* = −a/(I_local − pedestal) from detected events (amplitude then scales with local background, matching cross-term physics), renders h at sub-pixel positions, adds a shot-noise increment, 12-bit clips, recomputes diff through the same pipeline so def/ref/diff stay consistent, and emits exact masks.

The run directory is the contract between scripts: `run_pipeline.py` writes `config.json`, `psf.npz`, `noise_model.npz`, `events.csv`, `run_summary.json` (clean-pair ids + strength distribution), `hot_map.npz`; `run_synth.py` and `turing_test.py` consume only these.

All optics-derived quantities (cutoff = 2NA/λ_min = 0.3 cyc/px here, FWHM, window radius, min separation) come from `PipelineConfig.optics` — changing optical mode means a new config JSON, never edited constants.

## Conventions that are easy to break

- **Fourier-shift sign in `EPSFModel.render`**: it must produce `h(x−c)`. The opposite sign renders `h(x+c)`, mirrors every fitted center, and smears h (this bug happened; the comment in `epsf.py` records the symptom).
- **`measure_shift` is gradient-based (Lucas–Kanade), not phase correlation** — phase correlation aliases on periodic wafer patterns and is unconstrained along line/space bars. The structure-tensor eigenvalue floor intentionally reports 0 for pattern-invariant directions; that is correct behavior, not a bug. Valid only for |t| ≲ 0.5 px.
- Fourier shifting is an *exact* interpolator here (images are band-limited below Nyquist); do not replace it with spline/linear interpolation anywhere in the align/render paths — that blurs h.
- Coordinates: event centers `c` are relative to the crop center; absolute position = integer peak + c. In the simulator, pixel i's center maps to high-res cell `i·s_hr + (s_hr−1)/2`.
- `data/` and `runs/` are gitignored and regenerable; `data/` is also where real fab images would land and must stay out of git.

## Known approximations (documented, do not "fix" silently)

- Pedestal is estimated from a low percentile of reference images → absolute strength scale is biased (~1.2x on sim) but self-consistent between fitting and synthesis.
- Dark synthetic defects carry slight local over-noise (shot noise cannot be removed from a real image); ≤15% at the deepest pixel, second-order.
- Single-h assumption: check `figs/chi2_rho.png` for bimodality (focus families / mixed modes) before trusting a run on new data.
