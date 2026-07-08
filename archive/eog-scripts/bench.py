"""
bench.py — run (preprocessor × detector) combinations against all recordings.

Usage:
    source .venv/bin/activate
    python scripts/bench.py                          # all recordings, full grid
    python scripts/bench.py data/eog/foo.npz  # specific file(s)

Prints a compact average-only summary table (one row per combination).
Saves a timestamped JSON to derivatives/results/ with full per-subject, per-trial detail.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from brainpong import preprocess
from brainpong import detect

RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "eog"
RESULTS_DIR    = Path(__file__).resolve().parent.parent / "derivatives" / "results"

# ── Preprocessor × Detector grid ──────────────────────────────────────────────
#
# abs_scale: baseline for absolute FPR thresholds, in units of the preprocessor's
#   output signal (µV for amplitude pipelines, µV/s for velocity, dimensionless
#   σ for z_normalized).  σ-relative thresholds (first_crossing, sustained_crossing)
#   adapt automatically via baseline_sigma and need no per-preprocessor tuning.

_PREPS = [
    # (prep_fn,                              label,                 abs_scale)
    (preprocess.segment_diff_filter,    'seg-diff-filter     ', 10.0),
    (preprocess.narrow_band,            'narrow-band         ', 10.0),
    (preprocess.narrow_band_detrend,    'narrow-band-detrend ', 10.0),
    (preprocess.narrow_lpf,             'narrow-lpf          ', 10.0),
    (preprocess.linear_detrend,         'linear-detrend      ', 10.0),
    (preprocess.velocity,               'velocity            ', 100.0),  # µV/s
    (preprocess.z_normalized,           'z-normalized        ', 3.0),    # σ units
]

_DETS = [
    # (det_fn,                       label,               kwargs_fn(abs_scale))
    (detect.peak_sign,          'peak-sign         ',
     lambda s: {'fpr_threshold_uv': s}),
    (detect.first_crossing,     'first-crossing    ',
     lambda s: {'sigma_threshold': 6.0, 'fpr_threshold_sigma': 6.0}),
    (detect.sustained_crossing, 'sustained-crossing',
     lambda s: {'sigma_threshold': 6.0, 'min_dur_ms': 12.0, 'fpr_threshold_sigma': 6.0}),
    (detect.mean_sign,          'mean-sign         ',
     lambda s: {'fpr_threshold_uv': s / 2}),
    (detect.template_corr,      'template-corr     ',
     lambda s: {'template_width_ms': 150.0, 'fpr_threshold_uv': s / 2}),
]

COMBINATIONS = [
    (prep_fn, det_fn, det_kwargs(abs_scale),
     f'{prep_label} / {det_label}')
    for prep_fn, prep_label, abs_scale in _PREPS
    for det_fn, det_label, det_kwargs   in _DETS
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_combination(paths, prep_fn, det_fn, det_kwargs):
    results = []
    for p in paths:
        try:
            prep   = prep_fn(p)
            result = det_fn(prep, **det_kwargs)
            results.append(result)
        except Exception as e:
            print(f"  ERROR {prep_fn.__name__}/{det_fn.__name__} {p.name}: {e}",
                  file=sys.stderr)
    return results


def _aggregate(results):
    accs = [r['acc']        for r in results]
    lats = [r['lat_med_ms'] for r in results if r['lat_med_ms'] is not None]
    fprs = [r['fpr']        for r in results]
    return {
        'acc_mean':     float(np.mean(accs))  if accs else None,
        'lat_med_mean': float(np.mean(lats))  if lats else None,
        'fpr_mean':     float(np.mean(fprs))  if fprs else None,
        'n_subjects':   len(results),
    }


def _fmt_lat(ms):
    return f"{ms:.0f} ms" if ms is not None else "  n/a "


def _fmt_pct(f):
    return f"{f * 100:.1f}%"


def _print_summary(combo_results):
    """One avg row per combination — compact overview of the full grid."""
    col_w = {'combo': 38, 'acc': 7, 'lat': 8, 'fpr': 7}
    hdr = (
        f"{'Combination':<{col_w['combo']}} "
        f"{'Acc':>{col_w['acc']}} "
        f"{'Lat p50':>{col_w['lat']}} "
        f"{'FPR':>{col_w['fpr']}}"
    )
    sep = "─" * len(hdr)

    current_prep = None
    print()
    print(hdr)
    print(sep)
    for label, results in combo_results:
        prep_label = label.split(' / ')[0].strip()
        if prep_label != current_prep:
            if current_prep is not None:
                print(sep)
            current_prep = prep_label
        agg = _aggregate(results)
        print(
            f"{label:<{col_w['combo']}} "
            f"{_fmt_pct(agg['acc_mean']):>{col_w['acc']}} "
            f"{_fmt_lat(agg['lat_med_mean']):>{col_w['lat']}} "
            f"{_fmt_pct(agg['fpr_mean']):>{col_w['fpr']}}"
        )
    print(sep)
    print()


def _print_full(combo_results):
    """Per-subject rows + avg row per combination."""
    col_w = {'combo': 38, 'subj': 10, 'acc': 7, 'n': 4, 'lat': 8, 'fpr': 7}
    hdr = (
        f"{'Combination':<{col_w['combo']}} "
        f"{'Subject':<{col_w['subj']}} "
        f"{'Acc':>{col_w['acc']}} "
        f"{'N':>{col_w['n']}} "
        f"{'Lat p50':>{col_w['lat']}} "
        f"{'FPR':>{col_w['fpr']}}"
    )
    sep = "─" * len(hdr)

    current_prep = None
    print()
    print(hdr)
    print(sep)
    for label, results in combo_results:
        prep_label = label.split(' / ')[0].strip()
        if prep_label != current_prep:
            if current_prep is not None:
                print(sep)
            current_prep = prep_label
        for i, r in enumerate(results):
            print(
                f"{(label if i == 0 else ''):<{col_w['combo']}} "
                f"{r['subject']:<{col_w['subj']}} "
                f"{_fmt_pct(r['acc']):>{col_w['acc']}} "
                f"{r['n']:>{col_w['n']}} "
                f"{_fmt_lat(r['lat_med_ms']):>{col_w['lat']}} "
                f"{_fmt_pct(r['fpr']):>{col_w['fpr']}}"
            )
        agg = _aggregate(results)
        print(
            f"{'':.<{col_w['combo']}} "
            f"{'avg':<{col_w['subj']}} "
            f"{_fmt_pct(agg['acc_mean']):>{col_w['acc']}} "
            f"{'':>{col_w['n']}} "
            f"{_fmt_lat(agg['lat_med_mean']):>{col_w['lat']}} "
            f"{_fmt_pct(agg['fpr_mean']):>{col_w['fpr']}}"
        )
    print(sep)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*',
                        help='.npz recordings (default: all in data/eog/)')
    parser.add_argument('--full', action='store_true',
                        help='show per-subject rows (default: avg-only summary)')
    args = parser.parse_args()

    paths = [Path(f) for f in args.files] if args.files \
            else sorted(p for p in RECORDINGS_DIR.glob("*.npz")
                        if not p.name.endswith('.tmp.npz'))
    if not paths:
        print("No recordings found.", file=sys.stderr)
        sys.exit(1)

    print(f"Recordings : {[p.name for p in paths]}")
    print(f"Combinations: {len(COMBINATIONS)}  ({len(_PREPS)} preprocessors × {len(_DETS)} detectors)")

    combo_results = []
    run_log = []

    for prep_fn, det_fn, det_kwargs, label in COMBINATIONS:
        results = _run_combination(paths, prep_fn, det_fn, det_kwargs)
        combo_results.append((label, results))

        agg = _aggregate(results)
        run_log.append({
            'label':        label,
            'preprocessor': prep_fn.__name__,
            'detector':     det_fn.__name__,
            'det_kwargs':   det_kwargs,
            'aggregate':    agg,
            'subjects':     results,
        })

    if args.full:
        _print_full(combo_results)
    else:
        _print_summary(combo_results)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_DIR / f"{ts}-bench.json"
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(out, 'w') as f:
        json.dump({
            'timestamp':    ts,
            'recordings':   [str(p) for p in paths],
            'combinations': run_log,
        }, f, indent=2, default=str)

    print(f"Results saved → {out}")


if __name__ == '__main__':
    main()
