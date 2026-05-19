"""
Offline EOG recording viewer.

Loads a recordings/eog/*.npz file and plots the full session with event
markers so you can visually verify signal quality and label alignment.

Usage:
    source .venv/bin/activate
    python plot_recording.py                         # latest recording
    python plot_recording.py recordings/eog/foo.npz  # specific file
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

RECORDINGS_DIR = Path(__file__).parent / "recordings" / "eog"

LPF_HZ      = 100.0
HPF_HZ      = 0.5
NOTCH_BANDS = ((48.0, 52.0), (58.0, 62.0))

LABEL_COLOR = {
    'BASELINE': '#888888',
    'REST':     '#888888',
    'LEFT':     '#4499ff',
    'RIGHT':    '#ff8844',
    'DONE':     '#888888',
}


def filt(x_uv, sr):
    y = np.ascontiguousarray(x_uv.astype(np.float64))
    if y.size < 20:
        return y
    DataFilter.detrend(y, DetrendOperations.CONSTANT.value)
    DataFilter.perform_lowpass(y,  sr, LPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    for lo, hi in NOTCH_BANDS:
        DataFilter.perform_bandstop(y, sr, lo, hi, 3, FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_highpass(y, sr, HPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    return y


def load(path):
    d = np.load(path, allow_pickle=True)
    sr      = int(d['sample_rate'][0])
    ch_L    = int(d['eog_ch_L'][0])
    ch_R    = int(d['eog_ch_R'][0])
    eeg     = d['eeg']
    samples = d['event_samples'].astype(int)
    labels  = d['event_labels'].astype(str)
    subject = str(d['subject_id'][0])
    return eeg, sr, ch_L, ch_R, samples, labels, subject


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?', help='Path to .npz recording (default: latest)')
    parser.add_argument('--clip', action='store_true',
                        help='Clip display to the DONE event — hides post-session artifacts')
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
    else:
        files = sorted(RECORDINGS_DIR.glob("*.npz"))
        if not files:
            print("No recordings found in recordings/eog/")
            return
        path = files[-1]

    print(f"Loading {path.name}…")
    eeg, sr, ch_L, ch_R, ev_samples, ev_labels, subject = load(path)

    n_samps = eeg.shape[1]

    if args.clip:
        done_idx = next((s for s, l in zip(ev_samples, ev_labels) if l == 'DONE'), None)
        if done_idx is not None:
            clip_samp = min(int(done_idx) + int(sr * 1.0), n_samps)  # +1s buffer
            eeg       = eeg[:, :clip_samp]
            ev_mask   = ev_samples < clip_samp
            ev_samples = ev_samples[ev_mask]
            ev_labels  = ev_labels[ev_mask]
            n_samps   = eeg.shape[1]
            print(f"Clipped to {n_samps/sr:.1f}s (DONE event + 1s)")
    t       = np.arange(n_samps) / sr

    l_uv   = eeg[ch_L] * 1e6
    r_uv   = eeg[ch_R] * 1e6
    diff   = r_uv - l_uv
    diff_f = filt(diff.copy(), sr)
    vel    = np.gradient(diff_f) * sr   # µV/s

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(
        f"{path.name}  |  subject: {subject}  |  {n_samps/sr:.0f}s @ {sr} Hz",
        fontsize=12,
    )

    panel_data = [
        (axes[0], diff,   'HEOG raw  (R − L)',              '#88aaff'),
        (axes[1], diff_f, 'HEOG filtered',                  '#aaffaa'),
        (axes[2], vel,    'HEOG velocity  (d/dt filtered)',  '#ffcc66'),
    ]

    for ax, data, title, color in panel_data:
        ax.plot(t, data, lw=0.6, color=color)
        ax.set_ylabel('µV')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='#444444', lw=0.8)

        # Event markers
        for i, (s, label) in enumerate(zip(ev_samples, ev_labels)):
            if label in ('BASELINE', 'REST', 'DONE'):
                continue
            ts = s / sr
            ax.axvline(ts, color=LABEL_COLOR.get(label, 'white'),
                       lw=1.2, alpha=0.8, linestyle='--')
            # Label at top of first panel only
            if ax is axes[0]:
                ax.text(ts + 0.05, ax.get_ylim()[1] * 0.85,
                        label[0],   # 'L' or 'R'
                        color=LABEL_COLOR.get(label, 'white'),
                        fontsize=7, va='top')

    axes[-1].set_xlabel('Time (s)')

    # Per-trial peak table — filter per segment to avoid full-session drift artifacts
    print(f"\n{'Trial':>5}  {'Label':>6}  {'Peak (µV)':>10}  {'Correct?':>8}")
    print("-" * 38)
    correct_count = 0; trial = 0
    for i, (s, label) in enumerate(zip(ev_samples, ev_labels)):
        if label not in ('LEFT', 'RIGHT'):
            continue
        end = int(ev_samples[i + 1]) if i + 1 < len(ev_samples) else n_samps
        seg_l = filt((eeg[ch_L][s:end] * 1e6).copy(), sr)
        seg_r = filt((eeg[ch_R][s:end] * 1e6).copy(), sr)
        seg   = seg_r - seg_l
        if seg.size < 2:
            continue
        peak          = seg[np.argmax(np.abs(seg))]
        expected_sign = -1 if label == 'LEFT' else +1
        ok            = '✓' if np.sign(peak) == expected_sign else '✗'
        if np.sign(peak) == expected_sign:
            correct_count += 1
        trial += 1
        print(f"{trial:>5}  {label:>6}  {peak:>+10.1f}  {ok:>8}")
    print(f"\nAccuracy: {correct_count}/{trial} = {100*correct_count/trial:.1f}%")

    # Legend
    patches = [
        mpatches.Patch(color=LABEL_COLOR['LEFT'],  label='LEFT cue onset'),
        mpatches.Patch(color=LABEL_COLOR['RIGHT'], label='RIGHT cue onset'),
    ]
    axes[0].legend(handles=patches, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
