"""
EOG labeled-data recorder — cued gaze paradigm for classifier training.

Displays LEFT/RIGHT/REST cues, records 2-channel horizontal EOG with event
labels injected at each cue onset. Subject ID is required and embedded in
both the filename and the npz metadata for cross-subject analysis.

Raw board data (ALL 11 board rows, in volts) is saved verbatim — no filtering
ever touches the stored signal. The display trace is filtered for visual
monitoring only. The stored array therefore also carries the board's own
package counter (row 0) and per-sample unix timestamps (row 10), so dropped
samples and the true wall-clock start are recoverable post-hoc.

Saves to:  recordings/eog/<timestamp>-<subject_id>.npz
Protocol:  eog-v2-labeled   (eog-v1 = same raw data, fewer metadata fields)

Schema (canonical in CLAUDE.md):
  Per-recording : unix_start, sample_rate, gain, signal_unit, board
                  (model + which physical unit, one str), subject_id,
                  montage, notes, tags (list[str], e.g. data-problem
                  labels), eog_ch_L/eog_ch_R (L/R board rows)
  Per-sample    : raw signal, one value per board row (row 0 = running
                  package counter → honest timeline + drop detection;
                  row 10 = per-sample unix timestamp)
  Events        : (sample_count, label) pairs, pinned to the signal

Usage:
    source .venv/bin/activate
    python record_eog.py --subject john
    python record_eog.py --subject alice --trials 15
    python record_eog.py --subject john --gain 24 --notes "battery, PD_BIAS off"
    python record_eog.py --subject john --board second --notes "new unit, untested"
"""

import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from pathlib import Path

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# ── Board / channel config ──────────────────────────────────────────────────
SERIAL_PORT = "/dev/cu.usbserial-1120"
BOARD_ID    = BoardIds.CERELOG_X8_BOARD
BOARD_NAME  = "CERELOG_X8"
EOG_SLOT_L  = 0    # left  canthus → index into get_eeg_channels() (CH1)
EOG_SLOT_R  = 1    # right canthus → index into get_eeg_channels() (CH2)

# Stored signal is volts: the Cerelog brainflow fork applies the count→volt
# scaling before handing data to Python, so the npz already holds real units.
SIGNAL_UNIT  = "volts"
# Cerelog default PGA gain; not reported by the board, so it must be asserted
# at record time. Sets the clip ceiling: ≈ ±4.5 V / gain referred to input.
DEFAULT_GAIN = 24
# WHICH physical board. board_name/board_id only name the *model* (CERELOG_X8),
# and BrainFlow can't distinguish two same-model units over serial — so the unit
# label must be asserted at record time. Matters now that a 2nd board exists and
# may be faulty: this is the only field that pins a recording to one of them.
# "original" = the known-good first unit (backs all prior recordings); pass the
# 2nd unit explicitly, e.g. --board second.
DEFAULT_BOARD_UNIT = "original"
# Default montage for the active EOG rig (override with --montage).
DEFAULT_MONTAGE = ("2 electrodes at outer canthi (differential) + "
                   "active bias on ear clip, no ground")

# ── Trial timing ────────────────────────────────────────────────────────────
N_TRIALS_EACH = 25    # trials per direction; total = 2 × this
BASELINE_SECS = 5.0   # initial eyes-forward settle (not a trial)
REST_SECS     = 1.5   # inter-trial rest
LOOK_SECS     = 2.0   # hold-gaze duration per trial

# ── Display ─────────────────────────────────────────────────────────────────
UPDATE_MS    = 50
DISPLAY_SECS = 8

# ── Filter chain (matches filtered_plot.py exactly — display only) ──────────
LPF_HZ      = 100.0
HPF_HZ      = 0.5
NOTCH_BANDS = ((48.0, 52.0), (58.0, 62.0))

OUT_DIR = Path(__file__).parent / "recordings" / "eog"

CUE_TEXT = {
    'BASELINE': 'BASELINE\nsit still — eyes forward',
    'REST':     'REST\neyes forward',
    'LEFT':     '◄   LOOK LEFT',
    'RIGHT':    'LOOK RIGHT   ►',
    'DONE':     'Session complete!\nSaving…',
}
CUE_COLOR = {
    'BASELINE': '#aaaaaa',
    'REST':     '#aaaaaa',
    'LEFT':     '#4499ff',
    'RIGHT':    '#ff8844',
    'DONE':     '#88cc44',
}


def build_sequence(n_each):
    trials = ['LEFT'] * n_each + ['RIGHT'] * n_each
    random.shuffle(trials)
    seq = [('BASELINE', BASELINE_SECS)]
    for label in trials:
        seq.append(('REST', REST_SECS))
        seq.append((label, LOOK_SECS))
    seq.append(('DONE', 0.0))
    return seq


def filter_for_display(raw_volts_window, sr):
    """Filtered µV copy for display only — never touches saved data."""
    x = np.ascontiguousarray((raw_volts_window * 1e6).astype(np.float64))
    if x.size < 20:
        return x
    DataFilter.detrend(x, DetrendOperations.CONSTANT.value)
    DataFilter.perform_lowpass(x,  sr, LPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    for lo, hi in NOTCH_BANDS:
        DataFilter.perform_bandstop(x, sr, lo, hi, 3, FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_highpass(x, sr, HPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    return x


def save_recording(buf, ev_samples, ev_labels, meta):
    """Write the raw board buffer + full per-recording metadata to npz.

    `buf` is the verbatim board array (n_rows × n_samples, volts). Nothing
    here filters or re-scales it. `meta` carries the per-recording schema.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d-%H%M%S')
    path = OUT_DIR / f"{ts}-{meta['subject_id']}.npz"

    # Honest wall-clock start: first per-sample timestamp the board emitted
    # (row 10). Falls back to the time.time() captured at start_stream if the
    # timestamp row is somehow empty/zero.
    ts_row = meta['timestamp_row']
    unix_start = float(buf[ts_row][0]) if buf.shape[1] and buf[ts_row][0] > 0 \
        else float(meta['unix_start_fallback'])

    np.savez(
        path,
        # ── raw signal (verbatim board output, volts) ──
        eeg              = buf,
        # ── event markers (sample-pinned) ──
        event_samples    = np.array(ev_samples, dtype=np.int64),
        event_labels     = np.array(ev_labels,  dtype='U10'),
        # ── per-recording metadata ──
        subject_id       = np.array([meta['subject_id']]),
        unix_start       = np.array([unix_start]),
        sample_rate      = np.array([meta['sr']]),
        gain             = np.array([meta['gain']]),
        signal_unit      = np.array([SIGNAL_UNIT]),
        board            = np.array([meta['board']]),
        montage          = np.array([meta['montage']]),
        notes            = np.array([meta['notes']]),
        tags             = np.array(meta['tags'], dtype='U64'),
        # ── channel / row map (L/R semantic, not derivable — see electrode swap) ──
        eog_ch_L         = np.array([meta['ch_L']]),
        eog_ch_R         = np.array([meta['ch_R']]),
        protocol_version = np.array(['eog-v2-labeled']),
    )
    counts = {l: ev_labels.count(l) for l in set(ev_labels)}
    sr = meta['sr']
    print(f"\nSaved → {path}")
    print(f"  {buf.shape[1]} samples  ({buf.shape[1] / sr:.1f} s)")
    print(f"  unix_start={unix_start:.3f}  gain=x{meta['gain']}  unit={SIGNAL_UNIT}")
    print(f"  board={meta['board']}")
    print(f"  events: {counts}")
    if meta['notes']:
        print(f"  notes: {meta['notes']}")
    if meta['tags']:
        print(f"  tags: {meta['tags']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', required=True,
                        help='Subject ID embedded in filename and metadata (e.g. "john")')
    parser.add_argument('--trials', type=int, default=N_TRIALS_EACH,
                        help=f'Trials per direction (default {N_TRIALS_EACH})')
    parser.add_argument('--gain', type=int, default=DEFAULT_GAIN,
                        help=f'PGA gain the board is set to (default x{DEFAULT_GAIN}). '
                             'Sets clip ceiling; the board does not report it.')
    parser.add_argument('--board', default=DEFAULT_BOARD_UNIT,
                        help=f'WHICH physical board (default "{DEFAULT_BOARD_UNIT}"). '
                             'Model is always CERELOG_X8; this disambiguates the '
                             'two units. Pass the 2nd unit explicitly, e.g. "second".')
    parser.add_argument('--montage', default=DEFAULT_MONTAGE,
                        help='Electrode montage description (free text)')
    parser.add_argument('--notes', default='',
                        help='Free-text session notes (rig state, battery, PD_BIAS, etc.)')
    parser.add_argument('--tags', nargs='*', default=[], metavar='TAG',
                        help='Zero or more structured labels for filtering, esp. '
                             'data problems (e.g. --tags flatline railing). '
                             'notes is prose; tags are the machine-filterable axis.')
    args = parser.parse_args()

    subject_id = args.subject.strip().lower()
    sequence   = build_sequence(args.trials)
    total_look = args.trials * 2

    print(f"Subject   : {subject_id}")
    print(f"Trials    : {total_look} ({args.trials} LEFT + {args.trials} RIGHT)")
    print(f"Duration  : ~{BASELINE_SECS + total_look * (REST_SECS + LOOK_SECS):.0f} s")

    params             = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT
    params.timeout     = 15
    board     = BoardShim(BOARD_ID, params)
    sr        = BoardShim.get_sampling_rate(BOARD_ID)
    n_rows    = BoardShim.get_num_rows(BOARD_ID)
    ch_L      = BoardShim.get_eeg_channels(BOARD_ID)[EOG_SLOT_L]
    ch_R      = BoardShim.get_eeg_channels(BOARD_ID)[EOG_SLOT_R]
    ts_row    = BoardShim.get_timestamp_channel(BOARD_ID)
    win       = int(DISPLAY_SECS * sr)

    meta = {
        'subject_id':          subject_id,
        'sr':                  sr,
        'gain':                args.gain,
        'board':               f"{BOARD_NAME} unit:{args.board.strip()}",
        'montage':             args.montage,
        'notes':               args.notes,
        'tags':                [t.strip() for t in args.tags if t.strip()],
        'ch_L':                ch_L,
        'ch_R':                ch_R,
        'timestamp_row':       ts_row,   # write-time only: derives unix_start, not stored
        'unix_start_fallback': time.time(),   # overwritten by row-10 ts at save
    }

    print(f"L channel : board row {ch_L}  |  R channel: board row {ch_R}")
    print(f"Gain      : x{args.gain}  |  unit: {SIGNAL_UNIT}")
    print(f"Board     : {meta['board']}")
    print("Connecting…")
    board.prepare_session()
    board.start_stream(int(5 * 60 * sr))
    meta['unix_start_fallback'] = time.time()
    time.sleep(2)
    print("Streaming. Complete all cues to save. Close window early to abort-save.\n")

    st = {
        'buf':      np.empty((n_rows, 0)),
        'n_samps':  0,
        'seq_idx':  0,
        'state_t0': None,
        'saved':    False,
    }
    ev_samples: list[int] = []
    ev_labels:  list[str] = []

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, (ax_cue, ax_sig) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={'height_ratios': [1, 2]},
    )
    fig.patch.set_facecolor('#111111')
    for ax in (ax_cue, ax_sig):
        ax.set_facecolor('#111111')

    ax_cue.set_xticks([]); ax_cue.set_yticks([])
    for sp in ax_cue.spines.values():
        sp.set_visible(False)

    cue_obj  = ax_cue.text(0.5, 0.55, '', transform=ax_cue.transAxes,
                            ha='center', va='center', fontsize=40,
                            fontweight='bold', color='white')
    prog_obj = ax_cue.text(0.5, 0.08, '', transform=ax_cue.transAxes,
                            ha='center', va='bottom', fontsize=11, color='#777777')

    ax_sig.set_title('HEOG  (R − L, filtered — display only)', color='white', fontsize=12)
    ax_sig.set_xlabel('Time (s from now)', color='#888888')
    ax_sig.set_ylabel('µV', color='#888888')
    ax_sig.tick_params(colors='#666666')
    for sp in ax_sig.spines.values():
        sp.set_edgecolor('#333333')
    ax_sig.set_xlim(-DISPLAY_SECS, 0)
    ax_sig.set_ylim(-200, 200)
    ax_sig.grid(True, color='#222222')

    t_axis   = np.linspace(-DISPLAY_SECS, 0, win)
    sig_line, = ax_sig.plot(t_axis, np.full(win, np.nan), lw=1.2, color='#aaffaa')

    plt.tight_layout()

    def _do_save():
        if not st['saved'] and st['buf'].shape[1] > 0:
            save_recording(st['buf'], ev_samples, ev_labels, meta)
            st['saved'] = True

    def on_close(event):
        try:
            board.stop_stream()
            board.release_session()
        except Exception:
            pass
        _do_save()

    fig.canvas.mpl_connect('close_event', on_close)

    def update(_frame):
        new = board.get_board_data()
        if new.shape[1] > 0:
            st['buf']     = np.hstack((st['buf'], new))
            st['n_samps'] += new.shape[1]

        now = time.monotonic()
        idx = st['seq_idx']

        if idx < len(sequence):
            label, duration = sequence[idx]

            if st['state_t0'] is None:
                st['state_t0'] = now
                ev_samples.append(st['n_samps'])
                ev_labels.append(label)

            elapsed   = now - st['state_t0']
            remaining = max(0.0, duration - elapsed)

            if duration > 0 and elapsed >= duration:
                st['seq_idx']  += 1
                st['state_t0']  = None
            else:
                cue_obj.set_text(CUE_TEXT.get(label, label))
                cue_obj.set_color(CUE_COLOR.get(label, 'white'))
                done = sum(1 for l in ev_labels if l in ('LEFT', 'RIGHT'))
                prog_obj.set_text(f'Trial {done} / {total_look}   |   {remaining:.1f} s')

        if st['seq_idx'] >= len(sequence):
            cue_obj.set_text(CUE_TEXT['DONE'])
            cue_obj.set_color(CUE_COLOR['DONE'])
            prog_obj.set_text('')
            _do_save()

        # Display trace — NaN-padded so the time axis never changes shape
        chunk = st['buf'][:, -win:]
        n_pts = chunk.shape[1]
        if n_pts >= 20:
            xl   = filter_for_display(chunk[ch_L], sr)
            xr   = filter_for_display(chunk[ch_R], sr)
            diff = np.full(win, np.nan)
            diff[-n_pts:] = xr - xl
            sig_line.set_ydata(diff)
            valid = diff[~np.isnan(diff)]
            peak  = max(float(np.abs(valid).max()) * 1.3, 50.0) if valid.size else 50.0
            ax_sig.set_ylim(-peak, peak)

    ani = FuncAnimation(fig, update, interval=UPDATE_MS, blit=False)  # noqa: F841
    plt.show()


if __name__ == '__main__':
    main()
