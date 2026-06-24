"""
SQLite-backed store for the EOG diagnostic web viewer.

Design: the committed ``data/eog/*.npz`` recordings are FROZEN ground truth and
are never modified or duplicated. This DB (a regenerable derivative) holds only
*metadata*, *events*, and *trim annotations* — the signal itself is read on
demand from the npz (version-tolerant: eog-v1 carries fewer fields than v2).
Trims are mutable DB rows that gate downstream analysis without ever touching
the raw arrays.

  recording : one row per npz (metadata + computed health + sidebar spark)
  event     : sample-pinned cue markers (ingested from the npz)
  trim      : keep-window annotation, one (optional) row per recording

Signal access (:func:`ribbon`, :func:`channels`) loads the npz, derives/filters
in µV, and min/max-decimates to the requested pixel width — honest decimation so
rails and spikes survive. Filters come from
``brainpong.preprocess.VIEWER_FILTERS``.
"""
import json
import time
import logging
import sqlite3
from pathlib import Path
from functools import lru_cache

import numpy as np

from brainpong.preprocess import VIEWER_FILTERS

log = logging.getLogger("brainpong.store")

DEFAULT_GAIN = 24      # Cerelog PGA default; eog-v1 npz don't store gain
V_REF = 4.5            # ADS1299 reference; full-scale ≈ ±V_REF/gain referred to input

SCHEMA = """
CREATE TABLE IF NOT EXISTS recording (
    id           TEXT PRIMARY KEY,
    source_npz   TEXT NOT NULL,
    subject      TEXT,
    fs           INTEGER,
    n_samples    INTEGER,
    duration     REAL,
    gain         INTEGER,
    gain_assumed INTEGER,
    board        TEXT,
    montage      TEXT,
    notes        TEXT,
    protocol     TEXT,
    ch_l         INTEGER,
    ch_r         INTEGER,
    signal_unit  TEXT,
    ceil_uv      REAL,
    status       TEXT,
    rail_pct     REAL,
    ribbon_peak  REAL,
    tags         TEXT,
    spark        TEXT,
    ingested_at  REAL
);
CREATE TABLE IF NOT EXISTS event (
    recording_id TEXT NOT NULL,
    sample       INTEGER NOT NULL,
    t            REAL NOT NULL,
    label        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_event_rec ON event(recording_id);
CREATE TABLE IF NOT EXISTS trim (
    recording_id TEXT PRIMARY KEY,
    t0           REAL NOT NULL,
    t1           REAL NOT NULL,
    reason       TEXT,
    updated_at   REAL NOT NULL
);
"""


def connect(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=10.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=5000")
    return con


def init_db(db_path):
    con = connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    con.commit()
    log.info("schema ready at %s", db_path)
    return con


# ── npz loading (version-tolerant, cached) ───────────────────────────────────

def _field(d, key, default=None):
    return d[key] if key in d.files else default


@lru_cache(maxsize=16)
def _load_npz(npz_path):
    """Load a frozen recording's raw arrays + metadata. Cached by path."""
    log.info("loading npz %s", npz_path)
    d = np.load(npz_path, allow_pickle=True)
    eeg = d["eeg"]
    gain_raw = _field(d, "gain")
    return {
        "eeg": eeg,
        "sr": int(_field(d, "sample_rate", [250])[0]),
        "ch_l": int(d["eog_ch_L"][0]),
        "ch_r": int(d["eog_ch_R"][0]),
        "n": int(eeg.shape[1]),
        "gain": int(gain_raw[0]) if gain_raw is not None else None,
        "subject": str(_field(d, "subject_id", ["?"])[0]),
        "board": (str(_field(d, "board")[0]) if _field(d, "board") is not None else None),
        "montage": (str(_field(d, "montage")[0]) if _field(d, "montage") is not None else None),
        "notes": (str(_field(d, "notes")[0]) if _field(d, "notes") is not None else None),
        "protocol": str(_field(d, "protocol_version", ["?"])[0]),
        "signal_unit": (str(_field(d, "signal_unit")[0]) if _field(d, "signal_unit") is not None else "volts"),
        "tags": [str(t) for t in _field(d, "tags", [])],
        "ev_samples": d["event_samples"].astype(int) if "event_samples" in d.files else np.array([], int),
        "ev_labels": d["event_labels"].astype(str) if "event_labels" in d.files else np.array([], str),
    }


def _ceil_uv(gain):
    return V_REF / (gain or DEFAULT_GAIN) * 1e6


def _diff_uv(sig):
    return (sig["eeg"][sig["ch_r"]] - sig["eeg"][sig["ch_l"]]) * 1e6


def _rail_status(wired, gain):
    """(status, rail_fraction) for a 2×N wired-channel slice (raw volts).

    Shared by ingest (_health, whole recording) and the live windowed-health
    endpoint (window_health) so the rail metric is computed identically whether
    over the full recording or the trimmed analysis window.
    """
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


def _health(sig):
    """status / rail_pct / ribbon_peak from the two wired channels (whole recording)."""
    wired = np.vstack([sig["eeg"][sig["ch_l"]], sig["eeg"][sig["ch_r"]]])
    status, rail_frac = _rail_status(wired, sig["gain"])
    return status, round(rail_frac * 100, 3), round(float(np.max(np.abs(_diff_uv(sig)))), 1)


def _spark(sig, buckets=64):
    """Coarse normalized midpoint sparkline of the 0.5–30 Hz ribbon, for the list."""
    flt = VIEWER_FILTERS["bp_0530"](_diff_uv(sig), sig["sr"])
    n = len(flt)
    if n < buckets:
        mid = flt
    else:
        edges = np.linspace(0, n, buckets + 1).astype(int)
        mid = (np.minimum.reduceat(flt, edges[:-1]) + np.maximum.reduceat(flt, edges[:-1])) / 2.0
    amp = float(np.max(np.abs(mid))) or 1.0
    return [round(float(v / amp), 3) for v in mid]


# ── ingest ───────────────────────────────────────────────────────────────────

def ingest_dir(db_path, npz_dir):
    con = init_db(db_path)
    paths = sorted(Path(npz_dir).glob("*.npz"))
    n = 0
    for p in paths:
        rid = p.stem
        sig = _load_npz(str(p))
        status, rail_pct, ribbon_peak = _health(sig)
        con.execute("DELETE FROM recording WHERE id=?", (rid,))
        con.execute("DELETE FROM event WHERE recording_id=?", (rid,))
        con.execute(
            """INSERT INTO recording
               (id, source_npz, subject, fs, n_samples, duration, gain, gain_assumed,
                board, montage, notes, protocol, ch_l, ch_r, signal_unit, ceil_uv,
                status, rail_pct, ribbon_peak, tags, spark, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rid, str(p), sig["subject"], sig["sr"], sig["n"], round(sig["n"] / sig["sr"], 2),
             (sig["gain"] or DEFAULT_GAIN), 1 if sig["gain"] is None else 0, sig["board"], sig["montage"],
             sig["notes"], sig["protocol"], sig["ch_l"], sig["ch_r"], sig["signal_unit"],
             round(_ceil_uv(sig["gain"]), 1), status, rail_pct, ribbon_peak,
             json.dumps(sig["tags"]), json.dumps(_spark(sig)), time.time()),
        )
        con.executemany(
            "INSERT INTO event (recording_id, sample, t, label) VALUES (?,?,?,?)",
            [(rid, int(s), round(int(s) / sig["sr"], 3), str(l))
             for s, l in zip(sig["ev_samples"], sig["ev_labels"])],
        )
        n += 1
        log.info("ingested %s [%s] rail=%.1f%% peak=%.0fuV", rid, status, rail_pct, ribbon_peak)
    con.commit()
    con.close()
    return n


# ── decimation ────────────────────────────────────────────────────────────────

def _r1(a): return [round(float(v), 1) for v in a]
def _r3(a): return [round(float(v), 3) for v in a]


def _window_idx(sr, n, t0, t1):
    i0 = 0 if t0 is None else max(0, int(t0 * sr))
    i1 = n if t1 is None else min(n, int(t1 * sr))
    return i0, max(i0 + 1, i1)


def _decimate(y, i0, i1, width, sr):
    """min/max per pixel bucket over samples [i0,i1). Returns (t_sec, mn, mx)."""
    seg = np.asarray(y[i0:i1], dtype=np.float64)
    n = seg.size
    if n == 0:
        return [], [], []
    if n <= width:
        return _r3((i0 + np.arange(n)) / sr), _r1(seg), _r1(seg)
    edges = np.linspace(0, n, width + 1).astype(int)
    starts = edges[:-1]
    mn = np.minimum.reduceat(seg, starts)
    mx = np.maximum.reduceat(seg, starts)
    centers = (i0 + (edges[:-1] + edges[1:]) / 2.0) / sr
    return _r3(centers), _r1(mn), _r1(mx)


# ── queries ───────────────────────────────────────────────────────────────────

def _rec_dict(row):
    d = dict(row)
    d["tags"] = json.loads(d.get("tags") or "[]")
    d["spark"] = json.loads(d.get("spark") or "[]")
    return d


def list_recordings(db_path):
    con = connect(db_path)
    rows = con.execute("""
        SELECT r.*, t.t0 AS trim_t0, t.t1 AS trim_t1
        FROM recording r LEFT JOIN trim t ON t.recording_id = r.id
        ORDER BY (r.status='ok'), r.id
    """).fetchall()
    con.close()
    out = []
    for r in rows:
        d = _rec_dict(r)
        d["trim"] = {"t0": r["trim_t0"], "t1": r["trim_t1"]} if r["trim_t0"] is not None else None
        d.pop("trim_t0", None); d.pop("trim_t1", None)
        out.append(d)
    return out


def get_recording(db_path, rid):
    con = connect(db_path)
    r = con.execute("SELECT * FROM recording WHERE id=?", (rid,)).fetchone()
    if r is None:
        con.close()
        return None
    events = [dict(e) for e in con.execute(
        "SELECT sample, t, label FROM event WHERE recording_id=? ORDER BY sample", (rid,))]
    tr = con.execute("SELECT t0, t1, reason FROM trim WHERE recording_id=?", (rid,)).fetchone()
    con.close()
    d = _rec_dict(r)
    d["events"] = events
    d["trim"] = dict(tr) if tr else None
    return d


def _source(db_path, rid):
    con = connect(db_path)
    r = con.execute("SELECT source_npz FROM recording WHERE id=?", (rid,)).fetchone()
    con.close()
    return r["source_npz"] if r else None


def ribbon(db_path, rid, filt="raw", t0=None, t1=None, width=1500):
    src = _source(db_path, rid)
    if src is None:
        return None
    sig = _load_npz(src)
    if filt not in VIEWER_FILTERS:
        filt = "raw"
    y = VIEWER_FILTERS[filt](_diff_uv(sig), sig["sr"])
    i0, i1 = _window_idx(sig["sr"], sig["n"], t0, t1)
    t, mn, mx = _decimate(y, i0, i1, width, sig["sr"])
    # Rail ceiling is a µV (displacement) concept; it's meaningless on the
    # velocity (µV/s) axis, so suppress it there.
    ceil = None if filt == "velocity" else round(_ceil_uv(sig["gain"]), 1)
    return {"filter": filt, "unit": "µV/s" if filt == "velocity" else "µV",
            "ceil_uv": ceil, "t": t, "mn": mn, "mx": mx}


def channels(db_path, rid, rows=None, t0=None, t1=None, width=1500):
    src = _source(db_path, rid)
    if src is None:
        return None
    sig = _load_npz(src)
    if rows is None:
        rows = [sig["ch_l"], sig["ch_r"]]
    i0, i1 = _window_idx(sig["sr"], sig["n"], t0, t1)
    t = None
    chans = []
    for row in rows:
        tt, mn, mx = _decimate(sig["eeg"][row] * 1e6, i0, i1, width, sig["sr"])
        t = tt
        chans.append({
            "row": row,
            "label": "L" if row == sig["ch_l"] else "R" if row == sig["ch_r"] else f"CH{row}",
            "is_l": row == sig["ch_l"], "is_r": row == sig["ch_r"],
            "mn": mn, "mx": mx,
        })
    return {"t": t or [], "ceil_uv": round(_ceil_uv(sig["gain"]), 1), "channels": chans}


def eeg_rows(db_path, rid):
    """The board rows that carry the 8 ADS1299 channels (rows 1..8)."""
    src = _source(db_path, rid)
    if src is None:
        return []
    sig = _load_npz(src)
    return list(range(1, min(9, sig["eeg"].shape[0])))


def window_health(db_path, rid, t0=None, t1=None):
    """Rail %/status/peak over a sample window [t0,t1] (defaults to whole recording).

    Backs the live, trim-reactive rail readout: the viewer fetches this for the
    current keep-window so the rail metric tracks the *kept* signal (the analysis
    object), not the discarded tail. Same formula as ingest (_rail_status), so the
    untrimmed value matches the stored recording.rail_pct.
    """
    src = _source(db_path, rid)
    if src is None:
        return None
    sig = _load_npz(src)
    i0, i1 = _window_idx(sig["sr"], sig["n"], t0, t1)
    wired = np.vstack([sig["eeg"][sig["ch_l"]][i0:i1], sig["eeg"][sig["ch_r"]][i0:i1]])
    status, rail_frac = _rail_status(wired, sig["gain"])
    ribbon = (sig["eeg"][sig["ch_r"]][i0:i1] - sig["eeg"][sig["ch_l"]][i0:i1]) * 1e6
    peak = round(float(np.max(np.abs(ribbon))), 1) if ribbon.size else 0.0
    return {"rail_pct": round(rail_frac * 100, 3), "status": status,
            "ribbon_peak": peak, "n": int(wired.shape[1]),
            "t0": i0 / sig["sr"], "t1": i1 / sig["sr"]}


# ── trim annotations (mutable; never touch the npz) ──────────────────────────

def get_trim(db_path, rid):
    con = connect(db_path)
    r = con.execute("SELECT t0, t1, reason, updated_at FROM trim WHERE recording_id=?", (rid,)).fetchone()
    con.close()
    return dict(r) if r else None


def set_trim(db_path, rid, t0, t1, reason=None):
    con = connect(db_path)
    con.execute(
        """INSERT INTO trim (recording_id, t0, t1, reason, updated_at) VALUES (?,?,?,?,?)
           ON CONFLICT(recording_id) DO UPDATE SET
             t0=excluded.t0, t1=excluded.t1, reason=excluded.reason, updated_at=excluded.updated_at""",
        (rid, float(t0), float(t1), reason, time.time()),
    )
    con.commit()
    con.close()
    log.info("trim set %s = [%.1f, %.1f]", rid, t0, t1)
    return {"t0": float(t0), "t1": float(t1), "reason": reason}


def clear_trim(db_path, rid):
    con = connect(db_path)
    con.execute("DELETE FROM trim WHERE recording_id=?", (rid,))
    con.commit()
    con.close()
    log.info("trim cleared %s", rid)
