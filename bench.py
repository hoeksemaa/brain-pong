"""
BrainPong latency + accuracy bench.

Replays a recorded session through the same algorithm the live game uses,
sliding a 1.5 s window every 300 ms (matching the live BCI cadence), and
collects per-trial accuracy + latency stats.

Usage:
    python bench.py --recording recordings/<id>.npz
    python bench.py --recording recordings/<id>.npz --hpf 8.0  # sweep HPF
"""

import argparse
import collections
import os

import numpy as np

import algorithm

# Marker codes mirror MARKER_* in pong_game_brainflow.py.
MARKER_CUE_LEFT     = 2
MARKER_CUE_RIGHT    = 3
MARKER_CUE_OFFSET   = 4
MARKER_PRESS        = 5
MARKER_RELEASE      = 6


# =============================================================================
# Mock board — replays a recording through a BoardShim-ish interface.
# =============================================================================
class MockBoard:
    """Loads a recording and exposes window slices the same way BoardShim does.

    Per the data integrity rules: every returned window is a fresh copy, never
    a view into the underlying npz array.
    """

    def __init__(self, recording_path):
        z = np.load(recording_path, allow_pickle=True)
        self._eeg      = z['eeg']        # (n_channels, n_samples)
        self._eeg_t    = z['eeg_t']      # (n_samples,)
        self._markers  = z['markers']    # (n_samples,)
        self._metadata = z['metadata'].item()

    # --- read-only metadata -------------------------------------------------
    @property
    def sampling_rate(self): return int(self._metadata['sampling_rate'])
    @property
    def n_channels(self):    return int(self._eeg.shape[0])
    @property
    def n_samples(self):     return int(self._eeg.shape[1])
    @property
    def session_id(self):    return self._metadata['session_id']
    @property
    def metadata(self):      return self._metadata

    # --- windowing ----------------------------------------------------------
    def t_session(self, sample_idx):
        """Session-relative time at a sample index, in seconds."""
        if sample_idx <= 0:
            return 0.0
        return float(self._eeg_t[sample_idx - 1] - self._eeg_t[0])

    def get_window(self, end_idx, n_samples):
        """Return a copy of (n_samples, n_channels) ending at sample end_idx.

        Returns None if there isn't enough data behind end_idx.
        """
        if end_idx < n_samples or end_idx > self.n_samples:
            return None
        start = end_idx - n_samples
        return self._eeg[:, start:end_idx].T.copy()

    # --- trial extraction ---------------------------------------------------
    def find_trials(self):
        """Walk the marker channel and return a list of trial dicts.

        Each trial: {'side', 't_cue', 't_press', 't_release'} in session-seconds.
        Only trials with all three timestamps are returned (the bench needs a
        labeled steady-state window to score against).
        """
        trials = []
        cur = None
        for i in np.nonzero(self._markers)[0]:
            code = int(round(self._markers[i]))
            t = self.t_session(i + 1)  # marker lands at sample i; treat its time as that sample's t
            if code in (MARKER_CUE_LEFT, MARKER_CUE_RIGHT):
                # close any prior open trial that still doesn't have a release
                if cur is not None and cur.get('t_press') and not cur.get('t_release'):
                    cur['t_release'] = cur.get('t_cue_offset')  # fall back to cue end
                if cur is not None and cur.get('t_press') and cur.get('t_release'):
                    trials.append(cur)
                cur = {'side': 'L' if code == MARKER_CUE_LEFT else 'R',
                       't_cue': t, 't_press': None, 't_release': None}
            elif code == MARKER_PRESS and cur is not None and cur.get('t_press') is None:
                cur['t_press'] = t
            elif code == MARKER_RELEASE and cur is not None and cur.get('t_press') is not None:
                cur['t_release'] = t
            elif code == MARKER_CUE_OFFSET and cur is not None:
                cur['t_cue_offset'] = t
                if cur.get('t_press') and cur.get('t_release'):
                    trials.append(cur)
                    cur = None
                # if press but no release, keep the trial open for now; the
                # next CUE_LEFT/RIGHT will close it w/ t_cue_offset as release
        # flush any trailing trial
        if cur is not None and cur.get('t_press') and cur.get('t_release'):
            trials.append(cur)
        return trials


# =============================================================================
# Run the algorithm on the mock board, collect per-window decisions.
# =============================================================================
def run_bench(board, args):
    fs = board.sampling_rate
    window_n = int(round(args.window_s * fs))
    step_n   = max(1, int(round((1.0 - args.overlap) * window_n)))

    state = {'smoothed': 0.0}
    decisions = []
    cursor = window_n  # first window ends at cursor

    while cursor <= board.n_samples:
        window = board.get_window(cursor, window_n)
        if window is None:
            break
        label, state, dbg = algorithm.cca_decision(
            window, state, fs,
            freq_l=args.freq_l, freq_r=args.freq_r,
            harmonics=args.harmonics,
            hpf_hz=args.hpf, lpf_hz=args.lpf,
            ema_alpha=args.ema_alpha,
            thresholds=None,  # raw-sign mode for v1
        )
        decisions.append({
            't_session_end': board.t_session(cursor),
            't_session_start': board.t_session(cursor - window_n),
            'label': label,
            'raw_score': dbg['raw_score'],
            'smoothed_score': dbg['smoothed_score'],
            'corr_L': dbg['corr_L'],
            'corr_R': dbg['corr_R'],
        })
        cursor += step_n
    return decisions


# =============================================================================
# Per-trial metrics.
# =============================================================================
def metrics_for_trial(trial, decisions):
    """Compute accuracy + latency metrics for a single trial.

    A decision "belongs" to a trial if its window-end time falls in
    [t_press, t_release]. We use window-end (not start) because that's the
    moment the algorithm actually emitted the decision.
    """
    in_trial = [d for d in decisions
                if trial['t_press'] <= d['t_session_end'] <= trial['t_release']]
    n = len(in_trial)
    if n == 0:
        return {'n_decisions': 0, 'n_correct': 0, 'accuracy': None,
                'latency_first': None, 'latency_sustained': None,
                'majority_label': None}

    correct = [d['label'] == trial['side'] for d in in_trial]
    n_correct = sum(correct)
    accuracy = n_correct / n

    # latency_first: first correct emission, measured from t_press
    latency_first = None
    for d in in_trial:
        if d['label'] == trial['side']:
            latency_first = d['t_session_end'] - trial['t_press']
            break

    # latency_sustained: 3 consecutive correct emissions; latency = end-time of
    # the FIRST of those 3, measured from t_press.
    latency_sustained = None
    for k in range(2, n):
        if correct[k] and correct[k-1] and correct[k-2]:
            latency_sustained = in_trial[k-2]['t_session_end'] - trial['t_press']
            break

    # majority over all in-trial decisions
    majority_label = collections.Counter(d['label'] for d in in_trial).most_common(1)[0][0]

    return {
        'n_decisions': n,
        'n_correct':   n_correct,
        'accuracy':    accuracy,
        'latency_first':     latency_first,
        'latency_sustained': latency_sustained,
        'majority_label':    majority_label,
    }


def compute_all(trials, decisions):
    return [{**t, **metrics_for_trial(t, decisions)} for t in trials]


# =============================================================================
# Reporting.
# =============================================================================
def _stats_ms(values):
    arr = np.array([v for v in values if v is not None])
    if arr.size == 0:
        return "n/a"
    return (f"p50={np.percentile(arr, 50)*1000:.0f}ms  "
            f"p95={np.percentile(arr, 95)*1000:.0f}ms  "
            f"max={arr.max()*1000:.0f}ms  "
            f"(n={arr.size})")


def render_report(args, board, rows):
    md = []
    md.append(f"# Bench results — `{board.session_id}`")
    md.append("")
    md.append(f"- recording: `{args.recording}`")
    md.append(f"- subject: `{board.metadata.get('subject_id', '?')}`  notes: `{board.metadata.get('headset_notes', '?')}`")
    md.append(f"- algorithm params: hpf={args.hpf}  lpf={args.lpf}  window={args.window_s}s  overlap={args.overlap}  harmonics={args.harmonics}  ema_alpha={args.ema_alpha}  freq_l={args.freq_l}  freq_r={args.freq_r}")
    md.append(f"- decision rule: raw_sign on smoothed score (no thresholds)")
    md.append("")

    # Summary
    if not rows:
        md.append("**No usable trials in this recording.**")
        return "\n".join(md)

    correct = sum(1 for r in rows if r['majority_label'] == r['side'])
    n = len(rows)
    md.append("## summary")
    md.append("")
    md.append(f"- trials: **{n}**")
    md.append(f"- accuracy (majority vote per trial): **{correct}/{n} = {100*correct/n:.1f}%**")
    md.append(f"- latency_first (first correct emission post-press): {_stats_ms([r['latency_first'] for r in rows])}")
    md.append(f"- latency_sustained (3 consecutive correct post-press): {_stats_ms([r['latency_sustained'] for r in rows])}")
    md.append("")

    # Confusion matrix
    md.append("## confusion matrix (true → majority predicted)")
    md.append("")
    md.append("| true \\ pred | L | R | NEUTRAL |")
    md.append("|---|---|---|---|")
    cm = collections.Counter()
    for r in rows:
        cm[(r['side'], r['majority_label'])] += 1
    for s in 'LR':
        md.append(f"| {s} | {cm[(s, 'L')]} | {cm[(s, 'R')]} | {cm[(s, 'NEUTRAL')]} |")
    md.append("")

    # Per-trial table
    md.append("## per-trial table")
    md.append("")
    md.append("| # | side | t_press(s) | hold(s) | n_dec | n_correct | acc | majority | lat_first | lat_sustained |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows):
        hold = r['t_release'] - r['t_press']
        lat_f = f"{r['latency_first']*1000:.0f}ms" if r['latency_first']     is not None else 'never'
        lat_s = f"{r['latency_sustained']*1000:.0f}ms" if r['latency_sustained'] is not None else 'never'
        acc   = f"{r['accuracy']*100:.0f}%" if r['accuracy'] is not None else 'n/a'
        md.append(f"| {i} | {r['side']} | {r['t_press']:.2f} | {hold:.2f} | {r['n_decisions']} | {r['n_correct']} | {acc} | {r['majority_label']} | {lat_f} | {lat_s} |")
    md.append("")

    return "\n".join(md)


# =============================================================================
# CLI.
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="BrainPong latency + accuracy bench")
    p.add_argument('--recording', required=True, help='Path to a recordings/<id>.npz file')
    p.add_argument('--hpf',         type=float, default=algorithm.DEFAULT_HPF_HZ)
    p.add_argument('--lpf',         type=float, default=algorithm.DEFAULT_LPF_HZ)
    p.add_argument('--window-s',    type=float, default=1.5,  help='FFT window length, seconds')
    p.add_argument('--overlap',     type=float, default=0.8,  help='window overlap fraction (0..1)')
    p.add_argument('--harmonics',   type=int,   default=algorithm.DEFAULT_HARMONICS)
    p.add_argument('--ema-alpha',   type=float, default=algorithm.DEFAULT_EMA_ALPHA)
    p.add_argument('--freq-l',      type=float, default=algorithm.DEFAULT_FREQ_L)
    p.add_argument('--freq-r',      type=float, default=algorithm.DEFAULT_FREQ_R)
    p.add_argument('--out',         default='plans/baseline-results.md')
    args = p.parse_args()

    board = MockBoard(args.recording)
    print(f"[bench] {board.session_id}: {board.n_samples} samples, {board.n_channels} ch @ {board.sampling_rate} Hz")

    trials = board.find_trials()
    print(f"[bench] found {len(trials)} usable trials in marker channel")

    decisions = run_bench(board, args)
    print(f"[bench] emitted {len(decisions)} window decisions")

    rows = compute_all(trials, decisions)

    report = render_report(args, board, rows)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(report)
    print(f"[bench] wrote {args.out}")

    # Console summary
    if rows:
        correct = sum(1 for r in rows if r['majority_label'] == r['side'])
        print()
        print(f"=== SUMMARY ===")
        print(f"trials: {len(rows)}")
        print(f"accuracy (majority): {correct}/{len(rows)} = {100*correct/len(rows):.1f}%")
        lat_s = [r['latency_sustained'] for r in rows if r['latency_sustained'] is not None]
        if lat_s:
            arr = np.array(lat_s)
            print(f"latency_sustained: p50={np.percentile(arr, 50)*1000:.0f}ms  "
                  f"p95={np.percentile(arr, 95)*1000:.0f}ms  n={arr.size}/{len(rows)}")
        lat_f = [r['latency_first'] for r in rows if r['latency_first'] is not None]
        if lat_f:
            arr = np.array(lat_f)
            print(f"latency_first:     p50={np.percentile(arr, 50)*1000:.0f}ms  "
                  f"p95={np.percentile(arr, 95)*1000:.0f}ms  n={arr.size}/{len(rows)}")


if __name__ == '__main__':
    main()
