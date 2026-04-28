"""
sweep.py — run a curated set of bench variants on a single recording, render
a single comparison table so you can scan many algorithms at once.

This is NOT a grid search. Each variant should represent a specific HYPOTHESIS
about why the algorithm fails (typically: why R-trials underperform L-trials,
which is the asymmetric failure mode in the canonical recording). Edit the
VARIANTS list below to add/remove tests. Each entry is a (name, overrides)
tuple where overrides override the bench's default params.

Usage:
    python sweep.py --recording recordings/<id>.npz
    python sweep.py --recording recordings/<id>.npz --out plans/sweep-2.md
"""

import argparse
import collections
import os

import numpy as np

import algorithm
import bench

# -----------------------------------------------------------------------------
# Variants — each one is a hypothesis. Add new entries as new questions arise.
# Defaults (no overrides) reproduce the live game's algorithm.
# -----------------------------------------------------------------------------
VARIANTS = [
    # baseline (= live game)
    ('baseline',     {}),
    # alpha-band hypothesis (informational only — see note below):
    # changing freq_l / freq_r tells the algorithm to look for SSVEP at those
    # freqs, but the recording's stimulus was at the freqs encoded in the
    # metadata. so these variants will tank unless a new recording is made
    # with matching stim. kept to make freq-mismatch failure mode visible.
    ('freq-12-18',   {'freq_l': 12.0, 'freq_r': 18.0}),
    ('freq-8-14',    {'freq_l': 8.0,  'freq_r': 14.0}),
    # higher HPF hypothesis: 5–8 Hz EOG drowns R-trials
    ('hpf-8',        {'hpf': 8.0}),
    ('hpf-12',       {'hpf': 12.0}),
    # harmonic content hypothesis: more harmonics gives CCA more dims to fit
    ('harmonics-5',  {'harmonics': 5}),
    ('harmonics-2',  {'harmonics': 2}),
    # window length hypothesis: more samples per CCA = more SNR
    ('window-2.0',   {'window_s': 2.0}),
    ('window-1.0',   {'window_s': 1.0}),
    # smoothing hypothesis: heavier EMA stabilizes noisy R-trials
    ('ema-0.7',      {'ema_alpha': 0.7}),
    ('ema-0.0',      {'ema_alpha': 0.0}),
    # 15 Hz harmonics (30, 45) hit the LPF cliff at 45 — try opening it up
    ('lpf-60',       {'lpf': 60.0}),
    # combo bets (informed guesses, not exhaustive)
    ('combo-harm5-w2',     {'harmonics': 5, 'window_s': 2.0}),
    ('combo-freq12-18-h5', {'freq_l': 12.0, 'freq_r': 18.0, 'harmonics': 5}),
]


# -----------------------------------------------------------------------------
def make_args(recording_path, **overrides):
    """Build an argparse.Namespace mimicking what bench.main parses."""
    defaults = dict(
        recording=recording_path,
        hpf=algorithm.DEFAULT_HPF_HZ,
        lpf=algorithm.DEFAULT_LPF_HZ,
        window_s=1.5,
        overlap=0.8,
        harmonics=algorithm.DEFAULT_HARMONICS,
        ema_alpha=algorithm.DEFAULT_EMA_ALPHA,
        freq_l=algorithm.DEFAULT_FREQ_L,
        freq_r=algorithm.DEFAULT_FREQ_R,
        out=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def run_variant(board, name, overrides):
    args = make_args(board.metadata.get('session_id', '?'), **overrides)
    args.recording = board._eeg  # not used downstream, just satisfies make_args
    trials = board.find_trials()
    decisions = bench.run_bench(board, args)
    rows = bench.compute_all(trials, decisions)

    n = len(rows)
    if n == 0:
        return {'name': name, 'overrides': overrides, 'n_trials': 0,
                'accuracy': 0.0, 'L_accuracy': 0.0, 'R_accuracy': 0.0,
                'lat_first_p50': None, 'lat_first_p95': None,
                'lat_sus_p50': None, 'lat_sus_p95': None, 'lat_sus_n': 0,
                'cm': collections.Counter()}

    correct = sum(1 for r in rows if r['majority_label'] == r['side'])
    cm = collections.Counter()
    for r in rows:
        cm[(r['side'], r['majority_label'])] += 1
    l_total = cm[('L','L')] + cm[('L','R')] + cm[('L','NEUTRAL')]
    r_total = cm[('R','L')] + cm[('R','R')] + cm[('R','NEUTRAL')]
    l_acc = cm[('L','L')] / l_total if l_total else 0.0
    r_acc = cm[('R','R')] / r_total if r_total else 0.0

    lat_first = [r['latency_first'] for r in rows if r['latency_first'] is not None]
    lat_sus   = [r['latency_sustained'] for r in rows if r['latency_sustained'] is not None]

    def pct(arr, p):
        if not arr: return None
        return float(np.percentile(np.array(arr), p))

    return {
        'name': name,
        'overrides': overrides,
        'n_trials': n,
        'accuracy': correct / n,
        'L_accuracy': l_acc,
        'R_accuracy': r_acc,
        'lat_first_p50': pct(lat_first, 50),
        'lat_first_p95': pct(lat_first, 95),
        'lat_sus_p50':   pct(lat_sus,   50),
        'lat_sus_p95':   pct(lat_sus,   95),
        'lat_sus_n': len(lat_sus),
        'cm': cm,
    }


# -----------------------------------------------------------------------------
def fmt_ms(v):
    return f"{int(round(v*1000))}ms" if v is not None else "n/a"


def render_report(recording_path, board, results):
    md = []
    md.append(f"# Sweep results — `{board.session_id}`")
    md.append("")
    md.append(f"- recording: `{recording_path}`")
    md.append(f"- subject: `{board.metadata.get('subject_id', '?')}`  notes: `{board.metadata.get('headset_notes', '?')}`")
    md.append(f"- variants: {len(results)}")
    md.append("")

    # Sort by accuracy desc; mark best
    sorted_results = sorted(results, key=lambda r: -r['accuracy'])
    best_acc = sorted_results[0]['accuracy'] if sorted_results else 0
    baseline = next((r for r in results if r['name'] == 'baseline'), None)
    baseline_acc = baseline['accuracy'] if baseline else None

    # Comparison table
    md.append("## comparison (sorted by accuracy)")
    md.append("")
    md.append("| variant | overrides | acc | Δ vs base | L→L | R→R | lat_p50 | lat_p95 | sus_p95 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted_results:
        ov = ' '.join(f"{k}={v}" for k, v in r['overrides'].items()) or '(defaults)'
        marker = '🏆 ' if r['accuracy'] == best_acc and best_acc > 0 else ''
        delta = ''
        if baseline_acc is not None and r['name'] != 'baseline':
            d = (r['accuracy'] - baseline_acc) * 100
            delta = f"{d:+.1f}pp"
        elif r['name'] == 'baseline':
            delta = '—'
        md.append(
            f"| {marker}**{r['name']}** | {ov} | "
            f"{r['accuracy']*100:.1f}% | {delta} | "
            f"{r['L_accuracy']*100:.0f}% | {r['R_accuracy']*100:.0f}% | "
            f"{fmt_ms(r['lat_first_p50'])} | {fmt_ms(r['lat_first_p95'])} | "
            f"{fmt_ms(r['lat_sus_p95'])} |"
        )
    md.append("")

    # Per-variant confusion matrices
    md.append("## confusion matrices")
    md.append("")
    md.append("(true row → majority predicted column)")
    md.append("")
    for r in sorted_results:
        cm = r['cm']
        ov = ' '.join(f"{k}={v}" for k, v in r['overrides'].items()) or '(defaults)'
        md.append(f"### {r['name']} — {ov}  (acc {r['accuracy']*100:.1f}%)")
        md.append("")
        md.append("| true \\ pred | L | R | NEUTRAL |")
        md.append("|---|---|---|---|")
        for s in 'LR':
            md.append(f"| **{s}** | {cm[(s,'L')]} | {cm[(s,'R')]} | {cm[(s,'NEUTRAL')]} |")
        md.append("")

    # Reading guide
    md.append("## reading the table")
    md.append("")
    md.append("- **acc** — % trials where majority vote matched the cue.")
    md.append("- **Δ vs base** — accuracy delta vs `baseline` in percentage points (positive = improvement).")
    md.append("- **L→L / R→R** — per-side accuracy. tells u where errors concentrate. all errors are R→L in the canonical recording, so R→R is the metric to watch.")
    md.append("- **lat_p50 / lat_p95** — first-correct-emission latency post-press, median + 95th percentile.")
    md.append("- **sus_p95** — sustained-correct (3 in a row) p95 latency.")
    md.append("- 🏆 marks the best accuracy row.")
    md.append("")
    md.append("## caveat on freq_l / freq_r variants")
    md.append("")
    md.append("The recording's actual stimulus frequencies are baked in by the recording session (see metadata). Changing `freq_l` / `freq_r` in the algorithm tells the bench to look for SSVEP at those frequencies — but the brain wasn't stimulated at those frequencies. So `freq-12-18` and `freq-8-14` variants tank, not because the hypothesis is wrong, but because there's no signal to find. Those variants are informational: they show what freq-mismatch failure looks like. To actually test a different freq pair, record a NEW session at that pair and run the sweep on that recording.")
    md.append("")

    return '\n'.join(md)


# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Run a sweep of bench variants and render a comparison table.")
    p.add_argument('--recording', required=True, help='Path to recordings/<id>.npz')
    p.add_argument('--out', default='plans/sweep-results.md', help='Where to write the markdown report')
    args = p.parse_args()

    board = bench.MockBoard(args.recording)
    print(f"[sweep] {board.session_id}: {board.n_samples} samples, {board.n_channels} ch @ {board.sampling_rate} Hz")
    print(f"[sweep] running {len(VARIANTS)} variants...")
    print()

    results = []
    for name, overrides in VARIANTS:
        r = run_variant(board, name, overrides)
        ov_str = ' '.join(f"{k}={v}" for k, v in overrides.items()) or '(defaults)'
        print(f"  [{name:22s}] {ov_str:38s} "
              f"acc={r['accuracy']*100:5.1f}%  L={r['L_accuracy']*100:3.0f}%  R={r['R_accuracy']*100:3.0f}%  "
              f"sus_p95={fmt_ms(r['lat_sus_p95']):>7s}")
        results.append(r)

    report = render_report(args.recording, board, results)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(report)
    print()
    print(f"[sweep] wrote {args.out}")

    # Surface the headline result
    sorted_results = sorted(results, key=lambda r: -r['accuracy'])
    baseline = next((r for r in results if r['name'] == 'baseline'), None)
    if sorted_results and baseline:
        winner = sorted_results[0]
        if winner['name'] != 'baseline':
            delta = (winner['accuracy'] - baseline['accuracy']) * 100
            print(f"[sweep] best variant: {winner['name']!r} at {winner['accuracy']*100:.1f}% "
                  f"({delta:+.1f}pp vs baseline)")


if __name__ == '__main__':
    main()
