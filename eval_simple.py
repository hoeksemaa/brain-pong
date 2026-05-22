"""
Simplest-possible EOG classifier eval.

Per trial:
  1. Extract window  [cue_onset - 0.3s, cue_onset + 0.5s]
  2. Compute diff = (ch_R - ch_L) * 1e6  (µV) for that window
  3. Filter the window (detrend → LPF 100 → notch 50/60 → HPF 0.5)
  4. Find the sample with the largest |amplitude| in [50ms, 500ms] post-cue
  5. Sign of that peak → LEFT (negative) or RIGHT (positive)
  6. Latency = time of that peak from cue onset

FPR:
  Skip the first 500ms of each REST window (return saccade).
  In the remaining portion, check whether |peak| > FPR_THRESHOLD_UV.
  FPR = fraction of REST windows that fire.

No normalization. No cross-subject LOO. No threshold sweep.
The only free parameter is FPR_THRESHOLD_UV (fixed, in µV).

Usage:
    source .venv/bin/activate
    python eval_simple.py
    python eval_simple.py --fpr-threshold 15
    python eval_simple.py recordings/eog/foo.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

RECORDINGS_DIR = Path(__file__).parent / "recordings" / "eog"

LPF_HZ      = 100.0
HPF_HZ      = 0.5
NOTCH_BANDS = ((48.0, 52.0), (58.0, 62.0))

PRE_S           = 0.3    # context before cue for IIR settling
DETECT_PRE_S    = 0.05   # detection window start (post-cue)
DETECT_POST_S   = 0.50   # detection window end   (post-cue)
FPR_SKIP_S      = 0.50   # skip this much at start of each REST (return saccade)
DEFAULT_FPR_THR = 10.0   # µV threshold for REST false-positive check


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
    return {
        'path':    path,
        'subject': str(d['subject_id'][0]),
        'sr':      int(d['sample_rate'][0]),
        'ch_L':    int(d['eog_ch_L'][0]),
        'ch_R':    int(d['eog_ch_R'][0]),
        'eeg':     d['eeg'],
        'ev_s':    d['event_samples'].astype(int),
        'ev_l':    d['event_labels'].astype(str),
    }


def eval_subject(rec, fpr_threshold_uv):
    sr, ch_L, ch_R = rec['sr'], rec['ch_L'], rec['ch_R']
    eeg, ev_s, ev_l = rec['eeg'], rec['ev_s'], rec['ev_l']
    n_samps = eeg.shape[1]

    pre_n  = int(PRE_S * sr)
    det_pre_n  = int(DETECT_PRE_S * sr)
    det_post_n = int(DETECT_POST_S * sr)

    # ── Trials ──────────────────────────────────────────────────────────────
    trial_results = []
    for s, label in zip(ev_s, ev_l):
        if label not in ('LEFT', 'RIGHT'):
            continue

        i0 = max(0, s - pre_n)
        i1 = min(n_samps, s + det_post_n)

        # Diff-first, then filter the window
        raw_diff = (eeg[ch_R, i0:i1] - eeg[ch_L, i0:i1]) * 1e6
        filtered = filt(raw_diff.copy(), sr)

        onset_in_seg = s - i0          # index of cue onset within the segment
        d0 = onset_in_seg + det_pre_n
        d1 = onset_in_seg + det_post_n
        d0 = max(0, min(d0, len(filtered)))
        d1 = max(0, min(d1, len(filtered)))

        detect = filtered[d0:d1]
        if detect.size == 0:
            continue

        peak_idx = int(np.argmax(np.abs(detect)))
        peak_val = float(detect[peak_idx])
        latency  = (det_pre_n + peak_idx) / sr

        predicted = 'RIGHT' if peak_val > 0 else 'LEFT'
        trial_results.append({
            'label':     label,
            'predicted': predicted,
            'correct':   (predicted == label),
            'latency':   latency,
            'peak_uv':   abs(peak_val),
        })

    # ── FPR on REST windows ──────────────────────────────────────────────────
    skip_n = int(FPR_SKIP_S * sr)
    rest_fires = []
    for i, (s, label) in enumerate(zip(ev_s, ev_l)):
        if label != 'REST':
            continue
        end = int(ev_s[i + 1]) if i + 1 < len(ev_s) else n_samps
        s_settled = s + skip_n
        if s_settled >= end:
            continue

        raw_diff = (eeg[ch_R, s_settled:end] - eeg[ch_L, s_settled:end]) * 1e6
        filtered = filt(raw_diff.copy(), sr)
        if filtered.size == 0:
            continue

        peak_val = float(np.max(np.abs(filtered)))
        rest_fires.append(peak_val > fpr_threshold_uv)

    n          = len(trial_results)
    n_correct  = sum(r['correct'] for r in trial_results)
    acc        = n_correct / n if n else 0.0
    lats       = [r['latency'] for r in trial_results]
    lat_med    = float(np.median(lats)) if lats else None
    peaks      = [r['peak_uv'] for r in trial_results]
    peak_med   = float(np.median(peaks)) if peaks else None
    fpr        = float(np.mean(rest_fires)) if rest_fires else 0.0

    return {
        'subject': rec['subject'],
        'acc': acc, 'n': n, 'n_correct': n_correct,
        'lat_med': lat_med, 'peak_med': peak_med, 'fpr': fpr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*',
                        help='.npz recordings (default: all in recordings/eog/)')
    parser.add_argument('--fpr-threshold', type=float, default=DEFAULT_FPR_THR,
                        help=f'µV threshold for REST false-positive check (default {DEFAULT_FPR_THR})')
    args = parser.parse_args()

    paths = [Path(f) for f in args.files] if args.files \
            else sorted(RECORDINGS_DIR.glob("*.npz"))
    if not paths:
        print("No recordings found.")
        sys.exit(1)

    recs    = [load(p) for p in paths]
    results = [eval_subject(r, args.fpr_threshold) for r in recs]

    print()
    print(f"Method : peak-sign  (per-segment diff-first filter, peak in "
          f"[{DETECT_PRE_S*1000:.0f}ms, {DETECT_POST_S*1000:.0f}ms] post-cue)")
    print(f"FPR    : |peak| > {args.fpr_threshold:.0f} µV in REST window "
          f"[{FPR_SKIP_S*1000:.0f}ms, end]")
    print()

    hdr = f"{'Subject':<12} {'Acc':>7} {'N':>4}  {'Lat p50':>8}  {'PeakAmp':>8}  {'FPR':>7}"
    print(hdr)
    print("─" * len(hdr))

    for r in results:
        lat_s  = f"{r['lat_med']*1000:.0f} ms" if r['lat_med'] else "   n/a"
        peak_s = f"{r['peak_med']:.1f} µV"    if r['peak_med'] else "   n/a"
        print(f"{r['subject']:<12} {r['acc']*100:>6.1f}% {r['n']:>4}  "
              f"{lat_s:>8}  {peak_s:>8}  {r['fpr']*100:>6.1f}%")

    print("─" * len(hdr))
    accs  = [r['acc']     for r in results]
    lats  = [r['lat_med'] for r in results if r['lat_med']]
    fprs  = [r['fpr']     for r in results]
    peaks = [r['peak_med'] for r in results if r['peak_med']]
    lat_s  = f"{np.mean(lats)*1000:.0f} ms" if lats else "   n/a"
    peak_s = f"{np.mean(peaks):.1f} µV"     if peaks else "   n/a"
    print(f"{'Average':<12} {np.mean(accs)*100:>6.1f}%  "
          f"{'':>3}  {lat_s:>8}  {peak_s:>8}  {np.mean(fprs)*100:>6.1f}%")
    print()


if __name__ == '__main__':
    main()
