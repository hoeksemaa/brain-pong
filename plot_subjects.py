"""
Multi-subject continuous EOG overlay — compare timing and amplitude across subjects.

Three panels sharing a time axis, each clipped at the subject's DONE event:
  1. ch_L  (left canthus)  — one trace per subject
  2. ch_R  (right canthus) — one trace per subject
  3. HEOG diff  (R − L)   — one trace per subject

Each panel has a single y-scale spanning all subjects (amplitudes are
comparable within the panel). y-scales can differ across panels.

Event markers (LEFT cue = dashed, RIGHT cue = dotted) drawn in each
subject's color on all three panels.

Standard filter chain applied per channel before computing diff:
  detrend → LPF 100 Hz → notch 50/60 → HPF 0.5 Hz
Use --raw to skip filtering (shows electrode drift — saccades will be tiny).

Usage:
    python plot_subjects.py                          # all recordings/eog/*.npz
    python plot_subjects.py recordings/eog/foo.npz   # specific file(s)
    python plot_subjects.py --raw                    # no filtering
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from pathlib import Path

from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

RECORDINGS_DIR = Path(__file__).parent / "recordings" / "eog"

LPF_HZ      = 100.0
HPF_HZ      = 0.5
NOTCH_BANDS = ((48.0, 52.0), (58.0, 62.0))

# Per-subject colors — enough for 5 subjects
SUBJECT_COLORS = ['#00ccff', '#ffcc00', '#ff44cc', '#44ff99', '#ff7744']


def filt(x_uv, sr):
    y = np.ascontiguousarray(x_uv.astype(np.float64))
    if y.size < 20:
        return y
    DataFilter.detrend(y, DetrendOperations.CONSTANT.value)
    DataFilter.perform_lowpass(y, sr, LPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    for lo, hi in NOTCH_BANDS:
        DataFilter.perform_bandstop(y, sr, lo, hi, 3, FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_highpass(y, sr, HPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    return y


def load(path):
    d = np.load(path, allow_pickle=True)
    return {
        'eeg':     d['eeg'],
        'sr':      int(d['sample_rate'][0]),
        'ch_L':    int(d['eog_ch_L'][0]),
        'ch_R':    int(d['eog_ch_R'][0]),
        'samples': d['event_samples'].astype(int),
        'labels':  d['event_labels'].astype(str),
        'subject': str(d['subject_id'][0]),
        'name':    path.name,
    }


def prepare(rec, apply_filter=True, skip_s=2.0):
    """Return time axis + filtered traces clipped at DONE, trimmed at start.

    Filtering runs on the full session before slicing so the IIR startup
    transient stays at t=0 and gets cut off by the skip, not displayed.
    """
    eeg, sr, ch_L, ch_R = rec['eeg'], rec['sr'], rec['ch_L'], rec['ch_R']
    samples, labels = rec['samples'], rec['labels']
    n_samps = eeg.shape[1]

    done_idx = next((s for s, l in zip(samples, labels) if l == 'DONE'), n_samps)
    clip  = min(int(done_idx), n_samps)
    start = min(int(skip_s * sr), clip)

    # Filter the full [0:clip] window first so startup transient is before 'start'
    l_raw = (eeg[ch_L, :clip] * 1e6).copy()
    r_raw = (eeg[ch_R, :clip] * 1e6).copy()

    if apply_filter:
        l_full = filt(l_raw, sr)
        r_full = filt(r_raw, sr)
    else:
        l_full = l_raw
        r_full = r_raw

    # Slice off the leading skip — transient is gone
    l_sig = l_full[start:]
    r_sig = r_full[start:]
    diff  = r_sig - l_sig

    n = len(l_sig)
    t = np.arange(n) / sr + skip_s

    ev_mask = (samples > start) & (samples <= clip)
    ev_t = samples[ev_mask] / sr
    ev_l = labels[ev_mask]

    return t, l_sig, r_sig, diff, ev_t, ev_l


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*',
                        help='.npz recordings — default: all in recordings/eog/')
    parser.add_argument('--raw', action='store_true',
                        help='Skip filtering (shows electrode drift; saccades will be tiny)')
    parser.add_argument('--skip', type=float, default=2.0,
                        help='Seconds to trim from the start of each recording (default 2.0)')
    parser.add_argument('--pct', type=float, default=99.5,
                        help='Percentile of |signal| used for y-axis limits (default 99.5)')
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = sorted(RECORDINGS_DIR.glob("*.npz"))
        if not paths:
            print("No recordings found in recordings/eog/")
            return

    recs = [load(p) for p in paths]
    print(f"Loaded {len(recs)} recording(s): {', '.join(r['subject'] for r in recs)}")

    prepared = []
    all_l, all_r, all_d = [], [], []
    for rec in recs:
        t, l, r, d, ev_t, ev_l = prepare(rec, apply_filter=not args.raw,
                                          skip_s=args.skip)
        prepared.append((rec, t, l, r, d, ev_t, ev_l))
        all_l.append(np.abs(l[~np.isnan(l)]))
        all_r.append(np.abs(r[~np.isnan(r)]))
        all_d.append(np.abs(d[~np.isnan(d)]))

    def robust_lim(arrays, pct):
        combined = np.concatenate(arrays)
        return float(np.percentile(combined, pct)) if combined.size else 1.0

    lim_l = robust_lim(all_l, args.pct)
    lim_r = robust_lim(all_r, args.pct)
    lim_d = robust_lim(all_d, args.pct)

    fig, axes = plt.subplots(3, 1, figsize=(20, 11), sharex=True)
    fig.patch.set_facecolor('#111111')

    panel_info = [
        (axes[0], 'ch_L  (left canthus)',  lim_l),
        (axes[1], 'ch_R  (right canthus)', lim_r),
        (axes[2], 'HEOG diff  (R − L)',    lim_d),
    ]
    for ax, title, lim in panel_info:
        ax.set_facecolor('#0c0c0c')
        ax.set_title(title, color='white', fontsize=11, pad=4)
        ax.set_ylabel('µV', color='#888888', fontsize=9)
        ax.set_ylim(-lim * 1.1, lim * 1.1)
        ax.axhline(0, color='#2a2a2a', lw=0.8)
        ax.grid(True, color='#1a1a1a', lw=0.5)
        ax.tick_params(colors='#666666', labelsize=8, labelbottom=True)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.set_xlabel('Time (s)', color='#888888', fontsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a2a')

    legend_handles = []

    for i, (rec, t, l, r, d, ev_t, ev_l) in enumerate(prepared):
        color = SUBJECT_COLORS[i % len(SUBJECT_COLORS)]
        subj  = rec['subject']

        axes[0].plot(t, l, lw=0.7, color=color, alpha=0.85)
        axes[1].plot(t, r, lw=0.7, color=color, alpha=0.85)
        axes[2].plot(t, d, lw=0.9, color=color, alpha=0.85)

        # Event markers: LEFT = dashed, RIGHT = dotted, in subject's color
        for ts, label in zip(ev_t, ev_l):
            if label not in ('LEFT', 'RIGHT'):
                continue
            ls = '--' if label == 'LEFT' else ':'
            for ax in axes:
                ax.axvline(ts, color=color, lw=0.9, alpha=0.40, linestyle=ls)

        legend_handles.append(
            mlines.Line2D([], [], color=color, lw=2.5, label=subj)
        )

    # Append marker-style entries to legend
    legend_handles += [
        mlines.Line2D([], [], color='#aaaaaa', lw=1.2, linestyle='--',
                      alpha=0.8, label='LEFT cue'),
        mlines.Line2D([], [], color='#aaaaaa', lw=1.2, linestyle=':',
                      alpha=0.8, label='RIGHT cue'),
    ]
    axes[0].legend(
        handles=legend_handles, loc='upper right', fontsize=8,
        facecolor='#1c1c1c', labelcolor='white', framealpha=0.85,
    )

    filter_note = 'raw (unfiltered)' if args.raw else 'filtered: detrend → LPF 100 → notch 50/60 → HPF 0.5'
    fig.suptitle(
        f"EOG comparison — {', '.join(r['subject'] for r in recs)}   [{filter_note}]",
        color='white', fontsize=12, y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    plt.show()


if __name__ == '__main__':
    main()
