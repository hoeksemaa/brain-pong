#!/usr/bin/env python3
"""Bake figure 8.5's data: one recording, the game's chain, amplitude AND velocity.

Writes two files into web/essay-figures/data/:

  eog-ampvel.bin   int16, three arrays of n samples each, little-endian, in
                   `layout` order:
                     diff   R − L differential, deviation from its global mean, µV
                     amp    the game's filtered amplitude, µV
                     vel    the game's velocity, µV/s ÷ `vel_scale` (int16 range)
  eog-ampvel.json  metadata — spans, σ per row, cues, the chain's constants.

UNLIKE FIGURES 5-10 this does NOT use the committed john recording, and UNLIKE
the main bake it uses the GAME's corners, not the essay's. Both divergences are
the figure's argument, measured before they were chosen:

- Corners: rows 2 and 3 must be the same pipeline diverging only at the last
  step, and at the essay's LP 100 the derivative amplifies 40-100 Hz broadband
  so much that velocity LOSES on the corpus (531/960 cued glances vs amplitude's
  708/960 at 6σ) — showing that would argue the opposite of what happened. At
  the game's corners the corpus reads amplitude 526/960, velocity 926/960.
  The chain here is eog_core._eog_filter called verbatim, so it cannot drift.

- Subject: on john both detectors catch 79/80 — the one subject in the corpus
  where nothing separates them. David (run #3 of the mass collection,
  2026-07-02) is the argument: BOTH rows draw σ from the same noisy opening,
  each landing at the 100th percentile of its own recording's 5 s blocks
  (σ_amp 51.3 µV vs blockwise median 14.9; σ_vel 586.6 µV/s vs median 324) —
  the worst calibration either detector could have drawn. Amplitude's 6σ
  (±308 µV) sits ABOVE the recording's peak (265 µV): it cannot fire at all,
  0/80 cued glances. Velocity's slimmer margin (peaks 2.2× its 6σ) still
  clears: 77/80. The one-shot calibration fragility is a known wound: #36
  moved σ to MAD for exactly this, and a baseline-σ quality gate is planned.

The poll discipline mirrors bake_essay_figures.live_chain: 125-sample buffer
(0.4 s settle + 0.1 s new) per 0.1 s poll, chain applied from zero IIR state,
velocity (Engbert & Kliegl 5-point, eog_core._eog_velocity) computed over the
whole filtered buffer BEFORE slicing so no stencil straddles a poll boundary,
newest 25 samples kept. σ per row replays _run_eog_sm's CALIBRATING branch on
that row's own signal: 1.4826 × MAD over the first 5 s of polls from the
BASELINE cue.

Ground truth is the raw recording. Nothing here modifies data/.

Usage:  python scripts/bake_ampvel_figure.py
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "src")
from brainpong.eog_core import (NOTCH_BANDS, EOG_BASELINE_S, EOG_SIGMA_THR,
                                EOG_LPF_HZ, EOG_HPF_HZ,
                                _eog_filter, _eog_velocity)

REC = pathlib.Path("data/eog/20260702-182329-david.npz")
OUT_DIR = pathlib.Path("web/essay-figures/data")

VIEW_S = 10.0
HEADROOM = 1.25
EOG_SETTLE_S, EOG_POLL_S = 0.4, 0.1
VALID_FROM_S = 2.0
# 1 unit = 2 µV/s. Peak |v| here is ~16 300 µV/s and poll edges can spike
# higher; 2 µV/s quantisation is invisible against a σ of ~700.
VEL_SCALE = 2.0
CUE_DIR = {"LEFT": "L", "FAST_LEFT": "L", "RIGHT": "R", "FAST_RIGHT": "R",
           "REST": "C", "FAST_REST": "C"}


def live_chain(diff, fs, velocity=False):
    """Poll-by-poll, calling the game's own `_eog_filter` verbatim."""
    n_new = max(1, int(EOG_POLL_S * fs))
    win = int(EOG_SETTLE_S * fs) + n_new
    x = np.ascontiguousarray(diff.astype(np.float64))
    out = np.zeros_like(x)
    ends = list(range(win, x.size + 1, n_new))
    if ends[-1] < x.size:
        ends.append(x.size)
    prev = 0
    for end in ends:
        buf = _eog_filter(x[end - win:end].copy(), fs)
        if velocity:
            buf = _eog_velocity(buf, fs)
        lo = max(end - n_new, prev)
        out[lo:end] = buf[lo - (end - win):end - (end - win)]
        prev = end
    return out


def calibrate(sig, fs, base_t):
    """`_run_eog_sm`'s CALIBRATING branch, replayed poll by poll on `sig`."""
    n_new = max(1, int(EOG_POLL_S * fs))
    need = int(EOG_BASELINE_S * fs)
    i = int(np.ceil(base_t * fs / n_new)) * n_new
    acc = []
    while True:
        acc.append(sig[i:i + n_new])
        total = np.concatenate(acc)
        i += n_new
        if total.size >= need:
            med = float(np.median(total))
            return float(1.4826 * np.median(np.abs(total - med)))


def y_span(x, fs, skip_s):
    """Worst 10 s slice × HEADROOM, ceilinged to 50 — the main bake's rule."""
    x = x[int(skip_s * fs):]
    w = int(VIEW_S * fs)
    worst = max(np.ptp(x[i:i + w]) for i in range(0, len(x) - w, fs))
    return float(np.ceil(worst * HEADROOM / 50.0) * 50.0)


def to_int16(x, scale=1.0):
    q = np.rint(x / scale)
    assert np.abs(q).max() < 32767, f"overflow at scale {scale}"
    return q.astype("<i2")


def main():
    d = np.load(REC, allow_pickle=True)
    fs = int(np.atleast_1d(d["sample_rate"])[0])
    eeg = d["eeg"]
    ch_R = eeg[int(np.atleast_1d(d["eog_ch_R"])[0])] * 1e6
    ch_L = eeg[int(np.atleast_1d(d["eog_ch_L"])[0])] * 1e6
    diff = ch_R - ch_L
    n = diff.size

    amp = live_chain(diff, fs)
    vel = live_chain(diff, fs, velocity=True)

    labels = [str(v) for v in d["event_labels"]]
    ev = np.asarray(d["event_samples"])
    base_t = float(ev[labels.index("BASELINE")]) / fs
    sig_amp = calibrate(amp, fs, base_t)
    sig_vel = calibrate(vel, fs, base_t)
    mean_diff = float(diff.mean())

    layout = ["diff", "amp", "vel"]
    blob = (to_int16(diff - mean_diff).tobytes()
            + to_int16(amp).tobytes()
            + to_int16(vel, VEL_SCALE).tobytes())

    payload = {
        "recording": REC.name,
        "subject": "david",
        "fs": fs, "n": int(n),
        "duration_s": round(n / fs, 3),
        "view_s": VIEW_S,
        "valid_from_s": VALID_FROM_S,
        "layout": layout,
        "vel_scale": VEL_SCALE,
        "corners": {"lp_hz": EOG_LPF_HZ, "hp_hz": EOG_HPF_HZ,
                    "notch_bands": [[a, b] for a, b in NOTCH_BANDS],
                    "note": "the game's own _eog_filter, called verbatim"},
        "poll": {"settle_s": EOG_SETTLE_S, "poll_s": EOG_POLL_S},
        "sigma_thr": EOG_SIGMA_THR,
        "sigma": {"amp_uv": round(sig_amp, 2), "vel_uvs": round(sig_vel, 2),
                  "baseline_t": round(base_t, 3)},
        "mean": {"diff": round(mean_diff, 1)},
        # Threshold lines must sit on-panel: span covers both the worst slice
        # and ±6σ with air, else an inflated σ pushes the lines out of frame.
        "span": {"diff": y_span(diff, fs, 0.0),
                 "amp": max(y_span(amp, fs, VALID_FROM_S),
                            float(np.ceil(2.3 * EOG_SIGMA_THR * sig_amp / 50) * 50)),
                 "vel": max(y_span(vel, fs, VALID_FROM_S),
                            float(np.ceil(2.3 * EOG_SIGMA_THR * sig_vel / 50) * 50))},
        "cues": [{"t": round(float(s) / fs, 3), "dir": CUE_DIR[l]}
                 for s, l in zip(ev, labels) if l in CUE_DIR],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eog-ampvel.bin").write_bytes(blob)
    (OUT_DIR / "eog-ampvel.json").write_text(json.dumps(payload, separators=(",", ":")))
    binkb = (OUT_DIR / "eog-ampvel.bin").stat().st_size / 1024
    print(f"wrote {OUT_DIR}/eog-ampvel.bin ({binkb:.0f} KB) + eog-ampvel.json")
    print(f"  {REC.name}: {n} samples @ {fs} Hz = {n / fs:.2f} s, "
          f"{len(payload['cues'])} cues")
    print(f"  sigma  amp {sig_amp:.2f} uV -> 6sig {6 * sig_amp:.1f}   "
          f"vel {sig_vel:.2f} uV/s -> 6sig {6 * sig_vel:.0f}")
    print(f"  spans  diff {payload['span']['diff']:.0f}  amp {payload['span']['amp']:.0f} uV  "
          f"vel {payload['span']['vel']:.0f} uV/s")


if __name__ == "__main__":
    main()
