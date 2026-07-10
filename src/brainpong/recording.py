"""In-game EOG recording writer.

Writes one npz per player in the same schema as ``scripts/record_eog.py``
(so the existing corpus + analysis tooling load it unchanged), extended with
the game-context fields the pong recorder needs: player count, board version,
serial port, player slot, and the live detector config (sigma / HPF / LPF /
glance window). Protocol tag ``eog-v3``.

Only the two EOG channels are stored (``eeg`` is ``(2, N)`` volts: row 0 = ch_L,
row 1 = ch_R), because CH3-8 are powered down in firmware and carry nothing.
``unix_start`` is passed in explicitly (the honest first board timestamp of the
captured span) rather than derived from a stored timestamp row.

Kept in the library (not the Dash script) so it is unit-testable without a board.
"""

import numpy as np
from datetime import datetime
from pathlib import Path

SIGNAL_UNIT = "volts"
PROTOCOL_VERSION = "eog-v3"


def save_eog_recording(out_dir, subject_id, eeg, unix_start, sr, *,
                       gain, board, montage, notes, tags, ch_L, ch_R,
                       n_players, board_version, serial_port, player_slot,
                       sigma_thr, hpf_hz, lpf_hz, glance_window_s,
                       detector='velocity', event_samples=(), event_labels=(),
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
    )
    return path
