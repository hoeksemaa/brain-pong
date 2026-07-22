#!/usr/bin/env python3
"""Bake the committed EOG corpus into a static, PUBLIC-SAFE data portal.

The live diagnostic viewer (``serve_viewer.py`` + Flask ``/api``) reads the frozen
npz on demand and filters server-side. That needs a Python host and exposes the raw
recordings (which carry real subject names). This script instead pre-computes
everything the *public* single-page portal needs into flat JSON under
``web/portal-data/`` so the page can be served as pure static files (GitHub Pages),
with **no server, no write endpoints, and no real names anywhere in the output**.

Privacy by construction
-----------------------
Named subjects are replaced with stable pseudonyms (``<first-name>`` -> ``Player A``)
*before* anything is written; recording ids become ``<date>-<time>-<slug>`` so the
name never appears in a filename or URL either; free-form ``notes`` (which can name
people) are parsed for the pipeline tag and then dropped. The owner keeps his own
name by choice (``OWNER``); the already-anonymous ``P1``/``P2`` slot files pass
through unchanged. Result: the baked tree is name-free and safe to commit + deploy.

Outputs (all under ``web/portal-data/``)
  manifest.json  — one lean row per recording (metadata + 5 tags + pipeline + spark)
  meta.json      — corpus aggregates, tag taxonomy, subject list, lens catalogue
  rec/<id>.json  — per-recording detail: shared time axis + every paradigm-lens
                   ribbon (min/max decimated) + L/R raw + event markers

The signal itself is never modified; this only reads the npz. Regenerable.

Usage
    python scripts/bake_portal.py                 # -> web/portal-data/
    python scripts/bake_portal.py --width 1400    # decimation resolution
"""
import argparse
import functools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, savgol_filter

# Rail/ceiling math (mirrors brainpong.store._rail_status / _ceil_uv — kept inline
# so the bake stays pure numpy+scipy with no brainflow import, i.e. runnable in a
# minimal/CI env; store.py is the source of truth if this formula ever changes).
V_REF = 4.5            # ADS1299 reference; full-scale ≈ ±V_REF/gain referred to input
DEFAULT_GAIN = 24      # Cerelog PGA default; eog-v1 npz don't store gain


def _ceil_uv(gain):
    return V_REF / (gain or DEFAULT_GAIN) * 1e6


def _rail_status(wired, gain):
    """(status, rail_fraction) for a 2×N wired-channel slice (raw volts)."""
    if wired.shape[1] == 0:
        return "ok", 0.0
    ceil_v = V_REF / (gain or DEFAULT_GAIN)
    rail_frac = float(np.mean(np.abs(wired) > 0.9 * ceil_v))
    stds = wired.std(axis=1)
    if stds.min() < 1e-5:
        status = "flat"
    elif rail_frac > 0.01:
        status = "railing"
    else:
        status = "ok"
    return status, rail_frac


REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "eog"
OUT_DIR = REPO / "web" / "portal-data"

OWNER = "john"           # the project owner keeps his own name (his data, his call)
WIDTH = 1400             # min/max decimation buckets per ribbon
SPARK_BUCKETS = 64

# ── DSP paradigm lenses (zero-phase; offline display is non-causal by design) ────
# Each lens is a historical filtering paradigm the pipeline passed through, so the
# same raw recording can be viewed through every era. Order = chronological.
# unit: display unit; ceil: whether the ADC rail band is meaningful on this axis.


@functools.lru_cache(maxsize=64)
def _sos_bp(lo, hi, sr):
    return butter(4, [lo, hi], btype="band", fs=sr, output="sos")


@functools.lru_cache(maxsize=64)
def _sos_bs(lo, hi, sr):
    return butter(4, [lo, hi], btype="bandstop", fs=sr, output="sos")


def _bp(x, sr, lo, hi):
    return sosfiltfilt(_sos_bp(lo, hi, sr), x) if x.size >= 30 else x


def _notch(x, sr):
    for lo, hi in ((48.0, 52.0), (58.0, 62.0)):
        if x.size >= 30:
            x = sosfiltfilt(_sos_bs(lo, hi, sr), x)
    return x


def _velocity(x, sr):
    y = _bp(x, sr, 0.1, 30.0)
    if y.size < 21:
        return y
    return savgol_filter(y, window_length=21, polyorder=2, deriv=1, delta=1.0 / sr)


def _matched(x, sr):
    """Velocity cross-correlated with a ~120 ms unit-norm Hann saccade template."""
    v = _velocity(x, sr)
    n = max(3, int(round(0.120 * sr)))
    tmpl = np.hanning(n)
    nrm = np.linalg.norm(tmpl)
    if nrm > 0:
        tmpl = tmpl / nrm
    return np.correlate(v, tmpl, mode="same") if v.size >= n else v


def _robust_sigma(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    s = 1.4826 * mad
    return float(s) if s > 1e-9 else float(np.std(x)) or 1e-6


# (id, label, date, unit, has_ceil, blurb) — the chronological progression.
LENSES = [
    ("raw", "Raw", None, "µV", True,
     "Unfiltered L−R differential — the signal exactly as the board recorded it."),
    ("preflight", "EOG preflight", "2026-04-29", "µV", True,
     "0.5–30 Hz band + 50/60 Hz notch. The first EOG-tuned display band."),
    ("offline", "Offline baseline", "2026-05-21", "µV", True,
     "0.5–100 Hz. The wide low-pass deliberately passed EMG — the false-positive "
     "problem that drove every later narrowing."),
    ("clinical", "Clinical HEOG", "2026-05-21", "µV", True,
     "0.1–30 Hz literature band. Lower corners keep the saccade step, cut EMG."),
    ("velocity", "Velocity", "2026-07-09", "µV/s", False,
     "Engbert–Kliegl velocity (Savitzky–Golay derivative). The shift from amplitude "
     "to velocity — differentiation self-rejects slow drift."),
    ("matched", "Matched filter", "2026-07-09", "µV/s", False,
     "Velocity cross-correlated with a 120 ms Hann saccade template — the statistic "
     "the current detector fires on. Integrates over the glance shape for SNR."),
    ("znorm", "z-normalized", None, "σ", False,
     "0.1–30 Hz divided by the recording's own noise floor, so amplitudes are in σ "
     "units comparable across people regardless of signal strength. The dashed line "
     "marks 6σ above noise — a significant-excursion reference. (The live detector's "
     "actual fire threshold sits on the velocity/matched statistic, not this "
     "amplitude band, so this line is a scale guide, not a fire predictor.)"),
]


def _apply_lens(lid, diff_uv, sr, sigma):
    if lid == "raw":
        return np.asarray(diff_uv, float)
    if lid == "preflight":
        return _notch(_bp(diff_uv, sr, 0.5, 30.0), sr)
    if lid == "offline":
        return _bp(diff_uv, sr, 0.5, 100.0)
    if lid == "clinical":
        return _bp(diff_uv, sr, 0.1, 30.0)
    if lid == "velocity":
        return _velocity(diff_uv, sr)
    if lid == "matched":
        return _matched(diff_uv, sr)
    if lid == "znorm":
        return _bp(diff_uv, sr, 0.1, 30.0) / sigma
    raise ValueError(lid)


# ── decimation (min/max per pixel bucket; honest — rails/spikes survive) ─────────

def _round(a, nd):
    if nd <= 0:
        return [int(v) for v in np.round(a).astype(np.int64)]
    return [round(float(v), nd) for v in a]


def _decimate(y, width, sr, nd):
    seg = np.asarray(y, float)
    n = seg.size
    if n == 0:
        return [], [], []
    if n <= width:
        t = [round(float(v), 3) for v in np.arange(n) / sr]
        r = _round(seg, nd)
        return t, r, list(r)
    edges = np.linspace(0, n, width + 1).astype(int)
    starts = edges[:-1]
    mn = _round(np.minimum.reduceat(seg, starts), nd)
    mx = _round(np.maximum.reduceat(seg, starts), nd)
    centers = [round(float(v), 3) for v in (edges[:-1] + edges[1:]) / 2.0 / sr]
    return centers, mn, mx


# ── npz loading (full metadata, version tolerant) ───────────────────────────────

def _f(d, key, default=None):
    return d[key] if key in d.files else default


def _s(d, key, default=None):
    v = _f(d, key)
    return str(v[0]) if v is not None and len(v) else default


def load(path):
    d = np.load(path, allow_pickle=True)
    eeg = d["eeg"]
    gain_raw = _f(d, "gain")
    ev_s = d["event_samples"].astype(int) if "event_samples" in d.files else np.array([], int)
    ev_l = d["event_labels"].astype(str) if "event_labels" in d.files else np.array([], str)
    return {
        "eeg": eeg,
        "sr": int(_f(d, "sample_rate", [250])[0]),
        "ch_l": int(d["eog_ch_L"][0]),
        "ch_r": int(d["eog_ch_R"][0]),
        "n": int(eeg.shape[1]),
        "gain": int(gain_raw[0]) if gain_raw is not None else None,
        "subject": _s(d, "subject_id", "?"),
        "board": _s(d, "board"),
        "notes": _s(d, "notes", "") or "",
        "protocol": _s(d, "protocol_version", "?"),
        "n_players": int(_f(d, "n_players", [0])[0]) if _f(d, "n_players") is not None else None,
        "board_version": _s(d, "board_version"),
        "detector": _s(d, "detector"),
        "tags": [str(t) for t in _f(d, "tags", [])],
        "ev_s": ev_s,
        "ev_l": ev_l,
    }


# ── tag derivation (deterministic ruleset — see docs cross-reference) ────────────

def derive_tags(m, date):
    proto = m["protocol"]
    if proto == "eog-v2-labeled":
        session_type = "cued"
    elif "training" in m["tags"]:
        session_type = "training"
    else:
        session_type = "game"

    bv = m["board_version"]
    if bv in ("1.2", "1.3"):
        rig = "v" + bv
    elif (m["board"] or "").endswith("original"):
        rig = "v1.2"                      # the original board is hardware v1.2 (docs)
    else:
        rig = "unknown"

    return {
        "tournament": (date == "2026-07-13"),
        "session_type": session_type,
        "rig_board": rig,
        "electrode_type": "gold",         # corpus is documented all-gold; no clip files
        "cleaning_regimen": "prepped" if date == "2026-07-13" else "unknown",
    }


_PIPE_RE = re.compile(r"\[pipeline-(v\d)\]")
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{2,}$")   # drop char-splat artifacts


def pipeline_of(m):
    hit = _PIPE_RE.search(m["notes"])
    return hit.group(1) if hit else None


# ── anonymization ────────────────────────────────────────────────────────────────

def build_anon_map(subjects):
    """real subject -> (display pseudonym, url slug). Stable, name-free output."""
    special = {OWNER: (OWNER, OWNER), "P1": ("P1", "P1"), "P2": ("P2", "P2")}
    named = sorted(s for s in subjects if s not in special)
    out = {}
    for s in subjects:
        if s in special:
            out[s] = special[s]
    for i, s in enumerate(named):
        letter = chr(ord("A") + i) if i < 26 else f"A{i}"
        out[s] = (f"Player {letter}", f"player{letter}")
    return out


def parse_stem(stem):
    m = re.match(r"^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})-(.+)$", stem)
    if not m:
        return stem, stem[:8], "", stem
    y, mo, d, hh, mm, ss, subj = m.groups()
    date = f"{y}-{mo}-{d}"
    tm = f"{hh}:{mm}:{ss}"
    session = f"{y}{mo}{d}-{hh}{mm}{ss}"
    return date, tm, session, subj


# ── main bake ────────────────────────────────────────────────────────────────────

def bake(width):
    paths = sorted(DATA_DIR.glob("*.npz"))
    if not paths:
        sys.exit(f"no npz under {DATA_DIR}")

    # Pass 1: collect real subjects (from the stem token) for a stable anon map.
    subjects = set()
    for p in paths:
        _, _, _, subj = parse_stem(p.stem)
        subjects.add(subj)
    anon = build_anon_map(subjects)

    (OUT_DIR / "rec").mkdir(parents=True, exist_ok=True)

    manifest = []
    used_ids = {}
    sess_members = {}       # session key -> list of (anon_id, display subject)
    total_samples = 0
    lens_meta = [{"id": i, "label": l, "date": dt, "unit": u, "blurb": b}
                 for (i, l, dt, u, _c, b) in LENSES]

    for p in paths:
        stem = p.stem
        date, tm, session, subj = parse_stem(stem)
        disp, slug = anon.get(subj, (subj, re.sub(r"[^A-Za-z0-9]", "", subj) or "x"))

        rid = f"{session}-{slug}"
        if rid in used_ids:                       # same person, same second (rare)
            used_ids[rid] += 1
            rid = f"{rid}-{used_ids[rid]}"
        else:
            used_ids[rid] = 1

        m = load(str(p))
        sr = m["sr"]
        total_samples += m["n"]
        diff = (m["eeg"][m["ch_r"]] - m["eeg"][m["ch_l"]]) * 1e6
        wired = np.vstack([m["eeg"][m["ch_l"]], m["eeg"][m["ch_r"]]])
        status, rail_frac = _rail_status(wired, m["gain"])
        ceil = round(_ceil_uv(m["gain"]), 1)
        sigma = _robust_sigma(_bp(diff, sr, 0.1, 30.0))
        tags = derive_tags(m, date)
        pipe = pipeline_of(m)

        # spark: coarse midpoint of the clinical band, normalized
        clin = _bp(diff, sr, 0.5, 30.0)
        if clin.size >= SPARK_BUCKETS:
            e = np.linspace(0, clin.size, SPARK_BUCKETS + 1).astype(int)
            mid = (np.minimum.reduceat(clin, e[:-1]) + np.maximum.reduceat(clin, e[:-1])) / 2
        else:
            mid = clin
        amp = float(np.max(np.abs(mid))) or 1.0
        spark = [round(float(v / amp), 3) for v in mid]

        row = {
            "id": rid,
            "date": date,
            "time": tm,
            "session": session,
            "subject": disp,
            "duration": round(m["n"] / sr, 1),
            "fs": sr,
            "n_players": m["n_players"],
            "status": status,
            "rail_pct": round(rail_frac * 100, 2),
            "ribbon_peak": round(float(np.max(np.abs(diff))), 1),
            "detector": m["detector"],
            "pipeline": pipe,
            "tags": tags,
            "spark": spark,
        }
        manifest.append(row)
        sess_members.setdefault(session, []).append((rid, disp))

        # ── per-recording detail: shared time axis + every lens ribbon ──
        t_axis, _, _ = _decimate(diff, width, sr, 1)
        lenses = {}
        for (lid, _l, _dt, _u, has_ceil, _b) in LENSES:
            nd = 2 if lid == "znorm" else 0    # 1 µV resolution is plenty on ±k signals
            y = _apply_lens(lid, diff, sr, sigma)
            _, mn, mx = _decimate(y, width, sr, nd)
            lenses[lid] = {"mn": mn, "mx": mx,
                           "ceil": ceil if has_ceil else None,
                           "thr": 6 if lid == "znorm" else None}
        # raw L/R electrodes
        _, l_mn, l_mx = _decimate(m["eeg"][m["ch_l"]] * 1e6, width, sr, 0)
        _, r_mn, r_mx = _decimate(m["eeg"][m["ch_r"]] * 1e6, width, sr, 0)
        events = [{"t": round(int(s) / sr, 2), "label": str(lab)}
                  for s, lab in zip(m["ev_s"], m["ev_l"]) if _LABEL_RE.match(str(lab))]

        detail = {
            "id": rid, "date": date, "time": tm, "session": session, "subject": disp,
            "duration": round(m["n"] / sr, 1), "fs": sr, "n_players": m["n_players"],
            "status": status, "rail_pct": round(rail_frac * 100, 2),
            "ribbon_peak": round(float(np.max(np.abs(diff))), 1),
            "ceil_uv": ceil, "sigma": round(sigma, 2),
            "detector": m["detector"], "pipeline": pipe, "tags": tags,
            "t": t_axis, "lenses": lenses,
            "channels": {"l": {"mn": l_mn, "mx": l_mx}, "r": {"mn": r_mn, "mx": r_mx}},
            "events": events,
        }
        (OUT_DIR / "rec" / f"{rid}.json").write_text(json.dumps(detail, separators=(",", ":")))

    # opponents (2-player sessions share a timestamp)
    by_id = {r["id"]: r for r in manifest}
    for members in sess_members.values():
        if len(members) == 2:
            (a, an), (b, bn) = members
            by_id[a]["opponent"] = bn
            by_id[b]["opponent"] = an
        else:
            for rid, _ in members:
                by_id[rid]["opponent"] = None

    # ── corpus aggregates + tag taxonomy ──
    def dist(key_fn):
        c = {}
        for r in manifest:
            k = key_fn(r)
            c[k] = c.get(k, 0) + 1
        return c

    dates = sorted(r["date"] for r in manifest)
    subjects_present = sorted({r["subject"] for r in manifest})
    quality = dist(lambda r: r["status"])

    def tag_group(label, key, order=None):
        d = dist(lambda r: r["tags"][key])
        keys = order or sorted(d, key=lambda k: (k is None, str(k)))
        return {"label": label,
                "values": [{"v": k, "count": d.get(k, 0)} for k in keys if k in d]}

    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus": {
            "n_recordings": len(manifest),
            "n_sessions": len(sess_members),
            "n_subjects": len(subjects_present),
            "total_hours": round(total_samples / 250 / 3600, 2),
            "date_start": dates[0] if dates else None,
            "date_end": dates[-1] if dates else None,
            "quality": quality,
        },
        "tags": {
            "tournament": {"label": "Tournament",
                           "values": [{"v": True, "count": dist(lambda r: r["tags"]["tournament"]).get(True, 0)},
                                      {"v": False, "count": dist(lambda r: r["tags"]["tournament"]).get(False, 0)}]},
            "session_type": tag_group("Session type", "session_type", ["game", "training", "cued"]),
            "rig_board": tag_group("Rig / board", "rig_board", ["v1.2", "v1.3", "unknown"]),
            "electrode_type": tag_group("Electrode", "electrode_type", ["gold", "clips", "unknown"]),
            "cleaning_regimen": tag_group("Skin prep", "cleaning_regimen", ["prepped", "none", "unknown"]),
        },
        "pipeline": {
            "detector": dist(lambda r: r["detector"]),
            "version": dist(lambda r: r["pipeline"]),
        },
        "subjects": subjects_present,
        "lenses": lens_meta,
        "owner": OWNER,
    }

    (OUT_DIR / "manifest.json").write_text(
        json.dumps({"generated_at": meta["generated_at"], "recordings": manifest},
                   separators=(",", ":")))
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))

    # console summary
    print(f"baked {len(manifest)} recordings -> {OUT_DIR}")
    print(f"  sessions={len(sess_members)} subjects={len(subjects_present)} "
          f"hours={meta['corpus']['total_hours']} quality={quality}")
    print(f"  tournament={dist(lambda r: r['tags']['tournament'])}")
    print(f"  session_type={dist(lambda r: r['tags']['session_type'])}")
    print(f"  rig_board={dist(lambda r: r['tags']['rig_board'])}")
    print(f"  detector={meta['pipeline']['detector']} pipeline={meta['pipeline']['version']}")
    sz = sum(f.stat().st_size for f in OUT_DIR.rglob("*.json"))
    print(f"  output size: {sz/1e6:.1f} MB across {len(list(OUT_DIR.rglob('*.json')))} files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=WIDTH)
    bake(ap.parse_args().width)
