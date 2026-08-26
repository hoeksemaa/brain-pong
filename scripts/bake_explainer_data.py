#!/usr/bin/env python3
"""Bake real EOG clips out of data/eog/ into JSON for the scroll explainer.

Writes web/explainer/data/eog-clips.json. Read-only over data/eog/ — this never
writes back into the corpus (see CLAUDE.md, "Recording & data integrity").

Two clips of the same cued LEFT/RIGHT task, recorded on the SAME DAY on the same
rig about three hours apart, chosen because their mains behaviour is opposite:

  main — the montage working as intended. ~100 uV of 60 Hz on each electrode,
         correlated +0.99 between them, so the differential cancels it down to
         ~12 uV. Eye dipole intact (channels anti-correlated, -0.80).
  poor — the montage failing. The 60 Hz on the two electrodes is ANTI-correlated
         (-0.98), so subtracting them ADDS the hum instead of cancelling it:
         ~120 uV per electrode becomes ~187 uV in the difference.

That contrast is the point of the closing figure. Across the corpus the median
hum correlation is +0.72 (subtraction typically helps, ~30% reduction), but 24%
of sessions land on the wrong side of zero and the differential makes mains
worse. See docs/ for the electrode-impedance-mismatch work this relates to.

Both are shipped raw (per-electrode, pre-differential) so the page can animate
the differential and the filter chain itself rather than showing a canned result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "explainer" / "data" / "eog-clips.json"

# (key, recording stem, public label, t_start_s, duration_s)
CLIPS = [
    ("main", "20260706-193819-playerJ", "Session A", 41.0, 18.0),
    ("poor", "20260706-161306-playerI", "Session B", 23.0, 18.0),
]

CUE_LABELS = ("LEFT", "RIGHT", "REST")


def bake_clip(stem: str, label: str, t0: float, dur: float) -> dict:
    d = np.load(ROOT / "data" / "eog" / f"{stem}.npz", allow_pickle=True)
    fs = int(d["sample_rate"][0])
    ch_r, ch_l = int(d["eog_ch_R"][0]), int(d["eog_ch_L"][0])

    a, b = int(round(t0 * fs)), int(round((t0 + dur) * fs))
    # volts -> microvolts. Copy out of the npz; never mutate the source arrays.
    r = np.array(d["eeg"][ch_r][a:b], dtype=np.float64) * 1e6
    l = np.array(d["eeg"][ch_l][a:b], dtype=np.float64) * 1e6

    # Split off each electrode's DC half-cell offset (tens of mV) so the JSON
    # carries small numbers. The page adds it back when it wants to show why
    # the differential is needed in the first place.
    dc_r, dc_l = float(np.median(r)), float(np.median(l))

    cues = [
        {"t": round((int(s) - a) / fs, 3), "dir": str(lab)}
        for s, lab in zip(d["event_samples"], d["event_labels"])
        if a <= int(s) < b and str(lab) in CUE_LABELS
    ]

    return {
        "label": label,
        "fs": fs,
        "n": int(b - a),
        "dc_r_uv": round(dc_r, 1),
        "dc_l_uv": round(dc_l, 1),
        "r": [round(v, 2) for v in (r - dc_r)],
        "l": [round(v, 2) for v in (l - dc_l)],
        "cues": cues,
        "montage": str(d["montage"][0]),
        "gain": int(d["gain"][0]),
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    clips = {key: bake_clip(stem, label, t0, dur) for key, stem, label, t0, dur in CLIPS}
    payload = {
        "note": "Real recordings from the BrainPong corpus. Raw, per-electrode, in uV.",
        "clips": clips,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    kb = OUT.stat().st_size / 1024
    for key, c in clips.items():
        print(f"  {key:6s} {c['label']}  {c['n']} samples @ {c['fs']} Hz  {len(c['cues'])} cues")
    print(f"wrote {OUT.relative_to(ROOT)}  ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
