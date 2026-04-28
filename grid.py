"""
grid.py — full grid search across the 5 sweepable algorithm knobs.

10 values × 5 knobs = 100,000 combinations (filtered down by hpf < lpf − 5).

Streaming CSV output makes this resumable: kill it any time, re-run picks up
where you left off (already-done combos are detected by exact param match).

Usage:
    # run as much as you can in this session (no limit)
    python grid.py --recording recordings/<id>.npz

    # run at most 500 more combos this session (good for chunking)
    python grid.py --recording recordings/<id>.npz --max 500

    # don't run anything, just regenerate the summary markdown from current CSV
    python grid.py --recording recordings/<id>.npz --summary-only

    # show progress check without running
    python grid.py --recording recordings/<id>.npz --status
"""

import argparse
import csv
import itertools
import os
import random
import time

import numpy as np

import algorithm
import bench


# =============================================================================
# THE GRID
# =============================================================================
# 5 values per knob, picked to span the interesting range w/o redundancy.
# freq_l / freq_r excluded — they're determined by the recording's actual
# stimulus and varying them just produces freq-mismatch.
#
# Reasoning for each value list (informed by the curated sweep):
#   hpf       — 3 (below all bands), 8 (alpha-band edge, didn't help in sweep),
#               12 (above-alpha, +12.5pp winner), 16 (well above), 25 (nuclear)
#   lpf       — 30 (narrow), 45 (default), 60 (preserves 30Hz harmonic),
#               80 (wide), 90 (very wide)
#   harmonics — 1 (minimal), 3 (default), 5 (modest), 7 (rich), 10 (max)
#   window_s  — 0.5 (latency-priority), 1.0, 1.5 (default), 2.5, 4.0 (SNR-priority)
#   ema_alpha — 0.0 (off), 0.3, 0.5, 0.7, 0.9 (near-frozen)
GRID_VALUES = {
    'hpf':       [3.0, 8.0, 12.0, 16.0, 25.0],
    'lpf':       [30.0, 45.0, 60.0, 80.0, 90.0],
    'harmonics': [1, 3, 5, 7, 10],
    'window_s':  [0.5, 1.0, 1.5, 2.5, 4.0],
    'ema_alpha': [0.0, 0.3, 0.5, 0.7, 0.9],
}
KNOB_ORDER = ['hpf', 'lpf', 'harmonics', 'window_s', 'ema_alpha']
HPF_LPF_MARGIN = 5.0  # require lpf > hpf + this; below ~5 Hz separation the bandpass dies

CSV_COLUMNS = KNOB_ORDER + [
    'accuracy', 'L_accuracy', 'R_accuracy', 'n_trials',
    'lat_first_p50_ms', 'lat_first_p95_ms',
    'lat_sus_p50_ms', 'lat_sus_p95_ms', 'lat_sus_n',
]


def all_combos():
    """Generate all valid (hpf < lpf - margin) combos in dict form.

    Shuffled with a fixed seed so partial chunks sample uniformly across
    the full grid instead of all-hpf=3-first. The seed is constant so
    re-running picks up exactly the same iteration order (resume works).
    """
    values = [GRID_VALUES[k] for k in KNOB_ORDER]
    out = []
    for combo in itertools.product(*values):
        params = dict(zip(KNOB_ORDER, combo))
        if params['hpf'] >= params['lpf'] - HPF_LPF_MARGIN:
            continue
        out.append(params)
    rng = random.Random(42)  # fixed seed — order is stable across invocations
    rng.shuffle(out)
    yield from out


def combo_key(params):
    """Tuple identity for dedup against an existing CSV."""
    return tuple(params[k] for k in KNOB_ORDER)


def _parse(s):
    """Best-effort parse of a CSV cell back to its native type."""
    try:
        return int(s)
    except ValueError:
        return float(s)


def load_existing_keys(csv_path):
    if not os.path.exists(csv_path):
        return set()
    keys = set()
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                keys.add(tuple(_parse(row[k]) for k in KNOB_ORDER))
            except (KeyError, ValueError):
                continue
    return keys


# =============================================================================
# RUN ONE COMBO
# =============================================================================
def run_combo(board, params):
    """Run one bench config; return a flat dict ready for CSV."""
    args = argparse.Namespace(
        recording=None,
        hpf=params['hpf'], lpf=params['lpf'],
        window_s=params['window_s'], overlap=0.8,
        harmonics=params['harmonics'], ema_alpha=params['ema_alpha'],
        freq_l=algorithm.DEFAULT_FREQ_L,
        freq_r=algorithm.DEFAULT_FREQ_R,
        out=None,
    )
    trials = board.find_trials()
    decisions = bench.run_bench(board, args)
    rows = bench.compute_all(trials, decisions)

    n = len(rows)
    correct   = sum(1 for r in rows if r['majority_label'] == r['side'])
    l_total   = sum(1 for r in rows if r['side'] == 'L')
    r_total   = sum(1 for r in rows if r['side'] == 'R')
    l_correct = sum(1 for r in rows if r['side'] == 'L' and r['majority_label'] == 'L')
    r_correct = sum(1 for r in rows if r['side'] == 'R' and r['majority_label'] == 'R')

    lat_first = [r['latency_first']     for r in rows if r['latency_first']     is not None]
    lat_sus   = [r['latency_sustained'] for r in rows if r['latency_sustained'] is not None]

    def pct_ms(arr, p):
        return float(np.percentile(np.array(arr), p)) * 1000 if arr else ''

    return {
        **params,
        'accuracy':   correct / n if n else 0,
        'L_accuracy': l_correct / l_total if l_total else 0,
        'R_accuracy': r_correct / r_total if r_total else 0,
        'n_trials':   n,
        'lat_first_p50_ms': pct_ms(lat_first, 50),
        'lat_first_p95_ms': pct_ms(lat_first, 95),
        'lat_sus_p50_ms':   pct_ms(lat_sus,   50),
        'lat_sus_p95_ms':   pct_ms(lat_sus,   95),
        'lat_sus_n':        len(lat_sus),
    }


def write_csv_row(csv_path, row):
    """Append one row + flush. Writes header if file is new."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


# =============================================================================
# SUMMARY MARKDOWN
# =============================================================================
def render_summary(csv_path, out_path):
    if not os.path.exists(csv_path):
        return
    rows = []
    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            try:
                row = {k: _parse(r[k]) for k in KNOB_ORDER}
                row['accuracy']   = float(r['accuracy'])
                row['L_accuracy'] = float(r['L_accuracy'])
                row['R_accuracy'] = float(r['R_accuracy'])
                row['lat_sus_p95_ms']   = float(r['lat_sus_p95_ms'])   if r.get('lat_sus_p95_ms')   else None
                row['lat_first_p50_ms'] = float(r['lat_first_p50_ms']) if r.get('lat_first_p50_ms') else None
                rows.append(row)
            except (ValueError, KeyError):
                continue
    if not rows:
        return

    md = []
    md.append(f"# Grid search summary — {len(rows)} combos completed")
    md.append("")
    md.append(f"Total grid space: {sum(1 for _ in all_combos())} valid combos.")
    md.append(f"Coverage: {100*len(rows)/sum(1 for _ in all_combos()):.1f}%")
    md.append("")

    # Top-20 by accuracy
    top = sorted(rows, key=lambda r: -r['accuracy'])[:20]
    md.append("## top 20 by accuracy")
    md.append("")
    md.append("| rank | hpf | lpf | harm | win | ema | accuracy | L→L | R→R | lat_p50 | sus_p95 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(top, 1):
        p50 = f"{r['lat_first_p50_ms']:.0f}ms" if r['lat_first_p50_ms'] is not None else 'n/a'
        p95 = f"{r['lat_sus_p95_ms']:.0f}ms"   if r['lat_sus_p95_ms']   is not None else 'n/a'
        md.append(
            f"| {i} | {r['hpf']:.0f} | {r['lpf']:.0f} | {r['harmonics']} | "
            f"{r['window_s']:.2f} | {r['ema_alpha']:.1f} | "
            f"**{r['accuracy']*100:.1f}%** | "
            f"{r['L_accuracy']*100:.0f}% | {r['R_accuracy']*100:.0f}% | "
            f"{p50} | {p95} |"
        )
    md.append("")

    # Per-knob marginal accuracy (mean over all combos with that knob fixed)
    md.append("## per-knob marginal accuracy")
    md.append("")
    md.append("(Mean accuracy across all combos that use that knob value. Tells u which knob values pull the average up vs down.)")
    md.append("")
    for knob in KNOB_ORDER:
        md.append(f"### {knob}")
        md.append("")
        md.append(f"| {knob} | mean accuracy | n combos | best in this group |")
        md.append(f"|---|---|---|---|")
        groups = {}
        for r in rows:
            groups.setdefault(r[knob], []).append(r)
        for v in sorted(groups.keys()):
            grp = groups[v]
            mean_acc = np.mean([r['accuracy'] for r in grp])
            best = max(grp, key=lambda r: r['accuracy'])
            md.append(f"| {v} | {mean_acc*100:.1f}% | {len(grp)} | {best['accuracy']*100:.1f}% |")
        md.append("")

    with open(out_path, 'w') as f:
        f.write('\n'.join(md))


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Resumable full grid search across algorithm knobs.")
    p.add_argument('--recording', required=True, help='Path to recordings/<id>.npz')
    p.add_argument('--csv',       default='plans/grid-results.csv', help='Streaming CSV (resume key)')
    p.add_argument('--summary',   default='plans/grid-results.md',  help='Markdown summary report')
    p.add_argument('--max', type=int, default=None, help='Run at most N more combos this invocation')
    p.add_argument('--summary-only', action='store_true', help='Skip running; emit summary from existing CSV')
    p.add_argument('--status',       action='store_true', help='Show progress and exit')
    args = p.parse_args()

    total_combos = sum(1 for _ in all_combos())
    done_keys = load_existing_keys(args.csv)
    todo = [c for c in all_combos() if combo_key(c) not in done_keys]

    if args.status:
        print(f"[grid] csv: {args.csv}")
        print(f"[grid] total valid combos: {total_combos}")
        print(f"[grid] completed:          {len(done_keys)}  ({100*len(done_keys)/total_combos:.1f}%)")
        print(f"[grid] remaining:          {len(todo)}")
        return

    if args.summary_only:
        render_summary(args.csv, args.summary)
        print(f"[grid] wrote summary {args.summary}")
        return

    os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
    board = bench.MockBoard(args.recording)
    print(f"[grid] {board.session_id}: {board.n_samples} samples @ {board.sampling_rate} Hz")
    print(f"[grid] grid: {total_combos} valid combos | done: {len(done_keys)} | remaining: {len(todo)}")

    if args.max is not None:
        todo = todo[:args.max]
        print(f"[grid] this session: running up to {len(todo)} combos")

    if not todo:
        print("[grid] nothing to do — grid is complete or already covered")
        render_summary(args.csv, args.summary)
        return

    t0 = time.time()
    for i, combo in enumerate(todo, 1):
        result = run_combo(board, combo)
        write_csv_row(args.csv, result)
        if i == 1 or i % 50 == 0 or i == len(todo):
            elapsed = time.time() - t0
            rate    = i / elapsed
            remain  = (len(todo) - i) / rate if rate > 0 else 0
            print(f"[grid] {i:5d}/{len(todo)}  "
                  f"acc={result['accuracy']*100:5.1f}%  L={result['L_accuracy']*100:3.0f}%  R={result['R_accuracy']*100:3.0f}%  "
                  f"({rate:.1f}/s, ~{remain/60:.0f} min remaining)")

    print()
    print(f"[grid] this session: {len(todo)} combos done in {(time.time()-t0)/60:.1f} min")
    render_summary(args.csv, args.summary)
    print(f"[grid] wrote summary {args.summary}")


if __name__ == '__main__':
    main()
