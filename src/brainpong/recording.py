"""In-game EOG recording writer.

Writes one npz per player in the same schema as ``scripts/record_eog.py``
(so the existing corpus + analysis tooling load it unchanged), extended with
the game-context fields the pong recorder needs: player count, board version,
serial port, player slot, and the live detector config (sigma / HPF / LPF /
glance window / deferred-confirm). Protocol tag ``eog-v3``.

Only the two EOG channels are stored (``eeg`` is ``(2, N)`` volts: row 0 = ch_L,
row 1 = ch_R), because CH3-8 are powered down in firmware and carry nothing.
``unix_start`` is passed in explicitly — the HOST-clock time of the span's first
sample (see :func:`map_events_to_samples`), not the board's own timestamp channel,
so it is immune to board-clock skew and the two players' files share one timebase.

Kept in the library (not the Dash script) so it is unit-testable without a board.
"""

import math
import numpy as np
from datetime import datetime
from pathlib import Path

SIGNAL_UNIT = "volts"
PROTOCOL_VERSION = "eog-v3"


def map_events_to_samples(events, start_t, stop_t, t_pull, n_pulled, sr):
    """Place a recording's span + event markers on the HOST clock — robust to board-clock skew.

    The X8 exposes a per-sample timestamp channel, but that channel can sit at a large
    *constant* offset from the host clock (one board was observed running ~801 s fast).
    Anchoring markers to the board clock then maps them to negative sample indices and
    silently drops them — exactly how every 2-player P1 file lost its calib/play markers.
    This function never reads the board timestamp. It treats the pulled buffer of
    ``n_pulled`` samples (rate ``sr``) as ending at host time ``t_pull``, so sample ``i``
    was acquired at ``t_pull - (n_pulled-1-i)/sr``.

    ``start_t``/``stop_t`` are the host times bracketing the recording (same clock as the
    event times). The buffer is trimmed to that window and each ``(label, host_time)``
    event is mapped onto the trimmed span.

    Returns ``(i0, i1, unix_start, ev_s, ev_l)``: inclusive slice bounds into the pulled
    buffer; the host time of the first stored sample (store it as ``unix_start`` so
    ``unix_start + i/fs`` is a host-referenced sample clock, which also gives the two
    players a shared timebase); and the in-span sample indices + labels of the events that
    fall inside the span. Only the host clock and nominal rate are used, so any board-clock
    offset is irrelevant by construction.
    """
    host0 = t_pull - (n_pulled - 1) / sr                    # host time of pulled sample 0
    i0 = max(0, int(math.ceil((start_t - host0) * sr)))     # first sample at/after start_t
    i1 = min(n_pulled - 1, int(math.floor((stop_t - host0) * sr)))  # last sample at/before stop_t
    if i1 - i0 < 1:                                          # window misses the buffer → keep all
        i0, i1 = 0, n_pulled - 1
    unix_start = host0 + i0 / sr                             # host time of the first stored sample
    n_span = i1 - i0 + 1
    ev_s, ev_l = [], []
    for label, t in events:
        s = int(round((t - unix_start) * sr))
        if 0 <= s < n_span:
            ev_s.append(s)
            ev_l.append(label)
    return i0, i1, float(unix_start), ev_s, ev_l


def save_eog_recording(out_dir, subject_id, eeg, unix_start, sr, *,
                       gain, board, montage, notes, tags, ch_L, ch_R,
                       n_players, board_version, serial_port, player_slot,
                       sigma_thr, hpf_hz, lpf_hz, glance_window_s,
                       detector='velocity', defer_ms=0.0, cand_sigma_thr=0.0,
                       event_samples=(), event_labels=(),
                       protocol_version=PROTOCOL_VERSION, stamp=None):
    """Write one player's EOG recording to ``<out_dir>/<ts>-<subject>.npz``.

    ``eeg`` is ``(2, N)`` volts, verbatim board output (unfiltered): row 0 = ch_L,
    row 1 = ch_R, so ``ch_L``/``ch_R`` should be 0/1. ``stamp`` (``YYYYMMDD-HHMMSS``)
    lets two simultaneous per-player saves share one timestamp so the files are
    linkable; if omitted it is generated now. Collision-safe: if the target path
    already exists (two players with the same name), a ``-<player_slot>`` suffix is
    added. Returns the written :class:`~pathlib.Path`.
    """
    eeg = np.ascontiguousarray(eeg)
    if eeg.ndim != 2 or eeg.shape[0] != 2:
        raise ValueError(f"eeg must be (2, N) [ch_L, ch_R]; got {eeg.shape}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = stamp or datetime.now().strftime('%Y%m%d-%H%M%S')
    path = out_dir / f"{ts}-{subject_id}.npz"
    if path.exists():
        path = out_dir / f"{ts}-{subject_id}-{player_slot}.npz"

    np.savez(
        path,
        # ── raw signal (verbatim board output, volts): row0=ch_L, row1=ch_R ──
        eeg              = eeg,
        # ── event markers (sample-pinned) ──
        event_samples    = np.array(list(event_samples), dtype=np.int64),
        event_labels     = np.array(list(event_labels), dtype='U16'),
        # ── per-recording metadata (record_eog schema) ──
        subject_id       = np.array([subject_id]),
        unix_start       = np.array([float(unix_start)]),
        sample_rate      = np.array([sr]),
        gain             = np.array([gain]),
        signal_unit      = np.array([SIGNAL_UNIT]),
        board            = np.array([board]),
        montage          = np.array([montage]),
        notes            = np.array([notes]),
        tags             = np.array(list(tags), dtype='U64'),
        eog_ch_L         = np.array([ch_L]),
        eog_ch_R         = np.array([ch_R]),
        protocol_version = np.array([protocol_version]),
        # ── game-context fields (new in eog-v3) ──
        n_players        = np.array([n_players]),
        board_version    = np.array([board_version]),
        serial_port      = np.array([serial_port]),
        player_slot      = np.array([player_slot]),
        sigma_thr        = np.array([sigma_thr]),
        hpf_hz           = np.array([hpf_hz]),
        lpf_hz           = np.array([lpf_hz]),
        glance_window_s  = np.array([glance_window_s]),
        detector         = np.array([detector]),
        # ── pipeline-v3 deferred-confirm detector ──
        # defer_ms = 0 means the pre-v3 immediate detector, which is also what every
        # archived file without these fields replays as. See eog_core._deferred_crossing.
        defer_ms         = np.array([defer_ms]),
        cand_sigma_thr   = np.array([cand_sigma_thr]),
    )
    return path
