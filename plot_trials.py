"""
Per-trial EOG grid — all trials, both channels, and diff.

For each recording, plots a grid where each cell is one trial:
  ch_L filtered (blue), ch_R filtered (orange/red), HEOG diff R−L (green).

Border color = cue direction (blue=LEFT, orange=RIGHT).
Title color  = correct polarity (green ✓) vs wrong (red ✗).

Per-segment filtering is applied independently to each channel window before
computing the diff — avoids the full-session drift artifact that can flip sign.

Usage:
    python plot_trials.py                           # all recordings
    python plot_trials.py recordings/eog/foo.npz    # specific file(s)
    python plot_trials.py --cols 5                  # narrower grid (default 10)
    python plot_trials.py --pre 0.5 --post 2.5      # time window around cue
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

RECORDINGS_DIR = Path(__file__).parent / "recordings" / "eog"

LPF_HZ      = 100.0
HPF_HZ      = 0.5
NOTCH_BANDS = ((48.0, 52.0), (58.0, 62.0))

DEFAULT_PRE_S  = 0.3
DEFAULT_POST_S = 2.0


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
    return (
        d['eeg'],
        int(d['sample_rate'][0]),
        int(d['eog_ch_L'][0]),
        int(d['eog_ch_R'][0]),
        d['event_samples'].astype(int),
        d['event_labels'].astype(str),
        str(d['subject_id'][0]),
    )


def extract_trial(eeg, ch_L, ch_R, sr, onset, n_samps, pre_samp, post_samp):
    """
    Returns (fl, fr, diff) each of length pre_samp+post_samp.
    NaN-padded at edges when onset is near recording boundaries.
    """
    win = pre_samp + post_samp
    s0, s1 = onset - pre_samp, onset + post_samp

    pad_l = max(0, -s0);       s0 = max(0, s0)
    pad_r = max(0, s1 - n_samps); s1 = min(n_samps, s1)

    fl = filt((eeg[ch_L][s0:s1] * 1e6).copy(), sr)
    fr = filt((eeg[ch_R][s0:s1] * 1e6).copy(), sr)

    def pad(arr):
        return np.concatenate([
            np.full(pad_l, np.nan), arr, np.full(pad_r, np.nan),
        ])

    fl_p = pad(fl)
    fr_p = pad(fr)
    return fl_p, fr_p, fr_p - fl_p


def plot_subject(path, n_cols=10, pre_s=DEFAULT_PRE_S, post_s=DEFAULT_POST_S,
                 diff_only=False):
    eeg, sr, ch_L, ch_R, ev_samples, ev_labels, subject = load(path)
    n_samps   = eeg.shape[1]
    pre_samp  = int(pre_s * sr)
    post_samp = int(post_s * sr)
    win       = pre_samp + post_samp
    t         = np.linspace(-pre_s, post_s, win)

    trials = []
    for i, (s, label) in enumerate(zip(ev_samples, ev_labels)):
        if label not in ('LEFT', 'RIGHT'):
            continue
        fl, fr, diff = extract_trial(eeg, ch_L, ch_R, sr, s, n_samps, pre_samp, post_samp)

        post_diff = diff[pre_samp:]
        valid = post_diff[~np.isnan(post_diff)]
        peak = float(valid[np.argmax(np.abs(valid))]) if valid.size > 0 else 0.0
        expected = -1 if label == 'LEFT' else +1
        ok = np.sign(peak) == expected

        trials.append({
            'label': label, 'fl': fl, 'fr': fr, 'diff': diff,
            'peak': peak, 'ok': ok,
        })

    n       = len(trials)
    n_rows  = max(1, (n + n_cols - 1) // n_cols)
    correct = sum(tr['ok'] for tr in trials)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.2, n_rows * 1.9),
        sharex=True, sharey=True,
        squeeze=False,
    )
    fig.patch.set_facecolor('#111111')
    fig.suptitle(
        f"{path.name}  |  subject: {subject}  |  {correct}/{n} correct  "
        f"({100*correct/n:.0f}%)" if n else f"{path.name}  |  {subject}  |  no trials",
        fontsize=11, color='white', y=0.995,
    )

    axes_flat = axes.flatten()

    for idx, tr in enumerate(trials):
        ax = axes_flat[idx]
        ax.set_facecolor('#0c0c0c')

        if diff_only:
            diff_color = '#4499ff' if tr['label'] == 'LEFT' else '#ff7744'
            ax.plot(t, tr['diff'], lw=1.4, color=diff_color)
        else:
            ax.plot(t, tr['fl'],   lw=0.8, color='#4499ff', label='ch_L')
            ax.plot(t, tr['fr'],   lw=0.8, color='#ff7744', label='ch_R')
            ax.plot(t, tr['diff'], lw=1.4, color='#88ff88', label='R−L')

        ax.axvline(0, color='#ffffff', lw=0.5, alpha=0.35)
        ax.axhline(0, color='#444444', lw=0.8)
        ax.tick_params(colors='#555555', labelsize=4)
        ax.grid(True, color='#1a1a1a', lw=0.4)

        cue_color   = '#4499ff' if tr['label'] == 'LEFT' else '#ff7744'
        title_color = '#66ee33' if tr['ok'] else '#ff3333'
        label_char  = 'L' if tr['label'] == 'LEFT' else 'R'
        ok_char     = '✓' if tr['ok'] else '✗'

        for sp in ax.spines.values():
            sp.set_edgecolor(cue_color)
            sp.set_linewidth(1.5)

        ax.set_title(
            f"#{idx+1} {label_char} {ok_char}  {tr['peak']:+.0f}µV",
            fontsize=6.5, color=title_color, pad=2,
        )

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    if n > 0 and not diff_only:
        axes_flat[0].legend(
            fontsize=5, loc='upper right',
            facecolor='#1c1c1c', labelcolor='white', framealpha=0.8,
        )

    fig.text(0.5, 0.002, f'Time re: cue onset (s),  window [{-pre_s:.1f}, +{post_s:.1f}]',
             ha='center', color='#666666', fontsize=7)
    ylabel = 'HEOG diff  R−L  (µV)' if diff_only else 'µV'
    fig.text(0.003, 0.5, ylabel, va='center', rotation='vertical',
             color='#666666', fontsize=7)

    plt.tight_layout(rect=[0.015, 0.015, 1.0, 0.99])
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*',
                        help='.npz recording(s) — default: all in recordings/eog/')
    parser.add_argument('--cols', type=int, default=10,
                        help='Grid columns (default 10)')
    parser.add_argument('--pre',  type=float, default=DEFAULT_PRE_S,
                        help=f'Seconds before cue onset (default {DEFAULT_PRE_S})')
    parser.add_argument('--post', type=float, default=DEFAULT_POST_S,
                        help=f'Seconds after  cue onset (default {DEFAULT_POST_S})')
    parser.add_argument('--diff-only', action='store_true',
                        help='Plot only the R−L diff (no individual channels)')
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = sorted(RECORDINGS_DIR.glob("*.npz"))
        if not paths:
            print("No recordings found in recordings/eog/")
            return

    for path in paths:
        print(f"Plotting {path.name}…")
        plot_subject(path, n_cols=args.cols, pre_s=args.pre, post_s=args.post,
                     diff_only=args.diff_only)

    plt.show()


if __name__ == '__main__':
    main()
