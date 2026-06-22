"""
Static-site data builder — npz → decimated JSON for the public raw-waveform viewer.

Reads every eog-v1-labeled recording, exports the two EOG channels (L/R) as
RAW signal (volts → µV, no filtering), min/max-decimated so envelope/spikes
survive, plus event markers. Output is committed and served statically by
Vercel — no Python runs in production.

Usage:
    source .venv/bin/activate
    python web/build.py
"""

import json
import numpy as np
from pathlib import Path

REPO      = Path(__file__).resolve().parent.parent
SRC_DIRS  = [REPO / "recordings" / "eog"]          # canonical labeled sessions
OUT_DIR   = Path(__file__).resolve().parent / "data"
TARGET_BUCKETS = 3000                               # ≈6000 pts/channel after min/max


def minmax_decimate(x, sr, target_buckets):
    """Bucket x into target_buckets; keep each bucket's min & max sample in
    time order. Preserves the raw envelope (rails, spikes, saccades) that
    naive stride decimation would hide. Returns (t_seconds, values)."""
    n = x.size
    if n <= target_buckets * 2:
        return np.arange(n) / sr, x.astype(float)
    bucket = int(np.ceil(n / target_buckets))
    t_out, v_out = [], []
    for start in range(0, n, bucket):
        seg = x[start:start + bucket]
        if seg.size == 0:
            continue
        i_min = int(np.argmin(seg)); i_max = int(np.argmax(seg))
        # emit the two extremes in their true temporal order
        pair = sorted(((start + i_min, seg[i_min]), (start + i_max, seg[i_max])))
        for idx, val in pair:
            t_out.append(idx / sr); v_out.append(float(val))
    return np.array(t_out), np.array(v_out)


def build_one(path):
    d = np.load(path, allow_pickle=True)
    if str(d.get("protocol_version", ["?"])[0]) not in ("eog-v1-labeled", "eog-v2-labeled"):
        return None
    sr      = int(d["sample_rate"][0])
    ch_L    = int(d["eog_ch_L"][0])
    ch_R    = int(d["eog_ch_R"][0])
    eeg     = d["eeg"]
    subject = str(d["subject_id"][0])
    n       = eeg.shape[1]

    chans = []
    for role, row in (("L", ch_L), ("R", ch_R)):
        uv = eeg[row].astype(float) * 1e6          # volts → µV, RAW
        t, v = minmax_decimate(uv, sr, TARGET_BUCKETS)
        chans.append({
            "name": f"{role}  (row {row})",
            "role": role,
            "row":  row,
            "t":    [round(x, 4) for x in t.tolist()],
            "uv":   [round(x, 2) for x in v.tolist()],
        })

    events = [
        {"t": round(int(s) / sr, 3), "label": str(l)}
        for s, l in zip(d["event_samples"].astype(int), d["event_labels"].astype(str))
    ]

    def _scalar(key, default=None):
        return d[key].ravel()[0].item() if key in d.files else default

    tags = [str(t) for t in d["tags"]] if "tags" in d.files else []

    return {
        "id":          path.stem,
        "subject":     subject,
        "sample_rate": sr,
        "n_samples":   n,
        "duration_s":  round(n / sr, 1),
        "unit":        "uV",
        "raw":         True,
        # v2 metadata (None / [] for legacy eog-v1 files)
        "gain":        _scalar("gain"),
        "montage":     _scalar("montage"),
        "notes":       _scalar("notes"),
        "unix_start":  _scalar("unix_start"),
        "tags":        tags,
        "channels":    chans,
        "events":      events,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for src in SRC_DIRS:
        for path in sorted(src.glob("*.npz")):
            rec = build_one(path)
            if rec is None:
                print(f"  skip {path.name} (not eog-v1-labeled)")
                continue
            (OUT_DIR / f"{rec['id']}.json").write_text(json.dumps(rec))
            manifest.append({
                "id": rec["id"], "subject": rec["subject"],
                "duration_s": rec["duration_s"], "sample_rate": rec["sample_rate"],
            })
            print(f"  built {rec['id']}  ({rec['duration_s']}s, {len(rec['events'])} events)")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest))
    print(f"\nWrote {len(manifest)} recordings → {OUT_DIR}")


if __name__ == "__main__":
    main()
