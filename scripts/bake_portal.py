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
EVERY subject is replaced with a stable pseudonym (``<first-name>`` -> ``Player A``)
*before* anything is written — the project owner included, so the published tree has
no real name in it at all. Recording ids become ``<date>-<time>-<slug>`` so a name
never reaches a filename or URL either, and free-form ``notes`` (which can name
people) are dropped entirely. The ``P1``/``P2`` slot recordings are pooled into a
single ``Unattributed`` bucket: they are station slots, not individuals. Pseudonym
letters are assigned in order of each subject's first recording, so adding newer
recordings never renames an already-published subject. Result: the baked tree is
name-free and safe to commit + deploy.

The corpus under ``data/eog/`` is NOT touched — it keeps the real names it was
recorded under, and this script only ever reads it.

Outputs (all under ``web/portal-data/``)
  manifest.json  — one lean row per recording (metadata + tags + spark)
  meta.json      — corpus aggregates, tag taxonomy, subject list
  rec/<id>.json  — per-recording detail: shared time axis + three raw min/max
                   ribbons (R−L difference, L electrode, R electrode) + events

The public portal deliberately shows only the raw signal — no derived/filtered
views. The signal itself is never modified; this only reads the npz. Regenerable.

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
from scipy.signal import butter, sosfiltfilt

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

WIDTH = 1400             # min/max decimation buckets per ribbon
SPARK_BUCKETS = 64

# ── spark-thumbnail bandpass (the only filtering left in the bake; the plots
#    themselves are raw). 0.5–30 Hz keeps the tiny list thumbnails readable —
#    raw traces are dominated by DC offset/drift at thumbnail size. ─────────────


@functools.lru_cache(maxsize=64)
def _sos_bp(lo, hi, sr):
    return butter(4, [lo, hi], btype="band", fs=sr, output="sos")


def _bp(x, sr, lo, hi):
    return sosfiltfilt(_sos_bp(lo, hi, sr), x) if x.size >= 30 else x


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
        "protocol": _s(d, "protocol_version", "?"),
        "n_players": int(_f(d, "n_players", [0])[0]) if _f(d, "n_players") is not None else None,
        "tags": [str(t) for t in _f(d, "tags", [])],
        "ev_s": ev_s,
        "ev_l": ev_l,
    }


# ── tag derivation (deterministic ruleset) ───────────────────────────────────────

# Dates on which a public BrainPong tournament was held. Being on one of these
# dates is necessary but NOT sufficient — see derive_tags.
TOURNAMENT_DATES = ("2026-07-13", "2026-08-17")


def derive_tags(m, date, session_size):
    """Per-recording tags. `session_size` is how many recordings share this session
    key, i.e. how many people were at the rig for that match.

    A tournament recording is one made on a tournament DATE in a TWO-PLAYER session.
    The head-to-head requirement is what separates the event from the owner's own
    solo work on the same evening: 2026-08-17 ran as paired matches from 18:26 to
    20:14 and then, after everyone had gone, seven solo recordings at 22:24-22:47.
    Tagging by date alone would file those seven as tournament matches.

    Note the tag marks "recorded during a tournament", not "was a competitive rally"
    -- the interleaved training runs inside a match are tagged too, which is how
    2026-07-13 has always been tagged (33 sessions, all two-player, all tournament).
    That date is unaffected by the two-player requirement."""
    proto = m["protocol"]
    if proto == "eog-v2-labeled":
        session_type = "cued"
    elif "training" in m["tags"]:
        session_type = "training"
    else:
        session_type = "game"

    return {
        "tournament": date in TOURNAMENT_DATES and session_size == 2,
        "session_type": session_type,
    }


_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{2,}$")   # drop char-splat artifacts


# ── anonymization ────────────────────────────────────────────────────────────────

# The two anonymous station slots the recorder falls back to when no name is typed.
# They are NOT two people: 72 recordings across both tournament nights share these
# two labels, and the identity behind them was never captured, so they can only be
# pooled — never split into individuals, and never resolved to anyone. Publishing
# them as "P1" and "P2" made meta.json report two extra "subjects", so they collapse
# into ONE honest bucket that is visibly not a person.
SLOT_SUBJECTS = ("p1", "p2")
SLOT_LABEL = ("Unattributed", "unattributed")


def canon_subject(subj):
    """Canonical identity key for the subject token parsed out of a filename stem.

    The recorder takes the player name as free text, so ONE person can arrive under
    several spellings: the 2026-08-17 tournament recorded 'bradley' and 'BRADLEY'
    sixteen seconds apart, plus 'brandon'/'BRANDON' and 'laika'/'LAIKA'. Case-fold so
    one person draws one pseudonym. Without this the map keys on the exact string,
    each casing draws its own letter, and the portal publishes phantom participants —
    28 tokens for 25 real identities, overstating the headline count by three."""
    return subj.strip().lower()


def build_anon_map(first_seen):
    """canonical subject -> (display pseudonym, url slug), from {canon: earliest stem}.

    EVERY person is pseudonymised, the project owner included. The portal is the
    public face of a corpus recorded from volunteers; the owner exempting himself
    while publishing everyone else under a letter is the wrong asymmetry, and it also
    leaves exactly one real name in an otherwise name-free tree for a reader to
    notice. Only the station slots (see SLOT_SUBJECTS) stay unlettered, because they
    are not people.

    Letters follow each subject's FIRST recording (not the alphabet), so baking in
    newer recordings — even from new people — never renames an already-published
    subject and never breaks a shared portal URL. That guarantee covers new DATA, not
    policy changes: dropping the owner exemption inserts him at his true first
    appearance and shifts every letter after it, once."""
    slots = {s: SLOT_LABEL for s in SLOT_SUBJECTS} if SLOT_LABEL else {}
    named = sorted((s for s in first_seen if s not in slots),
                   key=lambda s: (first_seen[s], s))
    out = {s: slots[s] for s in first_seen if s in slots}
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

    # Pass 1: earliest stem per real subject (paths are stem-sorted = time-sorted),
    # so pseudonym letters follow first appearance and stay stable across re-bakes.
    first_seen = {}
    session_size = {}
    for p in paths:
        _, _, session, subj = parse_stem(p.stem)
        first_seen.setdefault(canon_subject(subj), p.stem)
        session_size[session] = session_size.get(session, 0) + 1
    anon = build_anon_map(first_seen)

    # Clean rec/ so retired ids (renamed/removed recordings) don't linger in git.
    rec_dir = OUT_DIR / "rec"
    rec_dir.mkdir(parents=True, exist_ok=True)
    for f in rec_dir.glob("*.json"):
        f.unlink()

    manifest = []
    used_ids = {}
    sess_members = {}       # session key -> list of (anon_id, display subject)
    total_seconds = 0.0

    for p in paths:
        stem = p.stem
        date, tm, session, subj = parse_stem(stem)
        disp, slug = anon.get(canon_subject(subj),
                              (subj, re.sub(r"[^A-Za-z0-9]", "", subj) or "x"))

        rid = f"{session}-{slug}"
        if rid in used_ids:                       # same person, same second (rare)
            used_ids[rid] += 1
            rid = f"{rid}-{used_ids[rid]}"
        else:
            used_ids[rid] = 1

        m = load(str(p))
        sr = m["sr"]
        total_seconds += m["n"] / sr
        diff = (m["eeg"][m["ch_r"]] - m["eeg"][m["ch_l"]]) * 1e6
        wired = np.vstack([m["eeg"][m["ch_l"]], m["eeg"][m["ch_r"]]])
        status, rail_frac = _rail_status(wired, m["gain"])
        ceil = round(_ceil_uv(m["gain"]), 1)
        tags = derive_tags(m, date, session_size.get(session, 1))

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
            "tags": tags,
            "spark": spark,
        }
        manifest.append(row)
        sess_members.setdefault(session, []).append((rid, disp))

        # ── per-recording detail: shared time axis + three raw ribbons ──
        # diff = R − L (rightward gaze positive — same convention as eog_core).
        t_axis, _, _ = _decimate(diff, width, sr, 1)
        _, d_mn, d_mx = _decimate(diff, width, sr, 0)   # 1 µV resolution on ±k signals
        _, l_mn, l_mx = _decimate(m["eeg"][m["ch_l"]] * 1e6, width, sr, 0)
        _, r_mn, r_mx = _decimate(m["eeg"][m["ch_r"]] * 1e6, width, sr, 0)
        events = [{"t": round(int(s) / sr, 2), "label": str(lab)}
                  for s, lab in zip(m["ev_s"], m["ev_l"]) if _LABEL_RE.match(str(lab))]

        detail = {
            "id": rid, "date": date, "time": tm, "session": session, "subject": disp,
            "duration": round(m["n"] / sr, 1), "fs": sr, "n_players": m["n_players"],
            "status": status, "rail_pct": round(rail_frac * 100, 2),
            "ceil_uv": ceil, "tags": tags,
            "t": t_axis,
            "diff": {"mn": d_mn, "mx": d_mx},
            "channels": {"l": {"mn": l_mn, "mx": l_mx}, "r": {"mn": r_mn, "mx": r_mx}},
            "events": events,
        }
        (rec_dir / f"{rid}.json").write_text(json.dumps(detail, separators=(",", ":")))

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
            # Quote THIS wherever a number of PEOPLE is meant. n_subjects counts
            # display labels, and one label ("Unattributed") pools 72 recordings
            # from an unknown number of additional people, so quoting it as a
            # participant count overstates it.
            "n_named_subjects": len([s for s in subjects_present
                                     if not SLOT_LABEL or s != SLOT_LABEL[0]]),
            "n_slot_recordings": sum(1 for r in manifest
                                     if SLOT_LABEL and r["subject"] == SLOT_LABEL[0]),
            "total_hours": round(total_seconds / 3600, 2),
            "date_start": dates[0] if dates else None,
            "date_end": dates[-1] if dates else None,
            "quality": quality,
        },
        "tags": {
            "tournament": {"label": "Tournament",
                           "values": [{"v": True, "count": dist(lambda r: r["tags"]["tournament"]).get(True, 0)},
                                      {"v": False, "count": dist(lambda r: r["tags"]["tournament"]).get(False, 0)}]},
            "session_type": tag_group("Session type", "session_type", ["game", "training", "cued"]),
        },
        "subjects": subjects_present,
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
    sz = sum(f.stat().st_size for f in OUT_DIR.rglob("*.json"))
    print(f"  output size: {sz/1e6:.1f} MB across {len(list(OUT_DIR.rglob('*.json')))} files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=WIDTH)
    bake(ap.parse_args().width)
