"""
Phase 1 — offline EOG classifier eval.

Loads all data/eog/*.npz files, runs leave-one-subject-out (LOO)
cross-validation, and reports accuracy / detection latency / false-positive
rate across a sweep of z-score thresholds.

Classification rule:
  1. Per-subject baseline: fit µ/σ on the opening BASELINE segment only
     (matches the 15 s match-start normalization planned for Phase 2).
  2. Compute z-scored HEOG diff = (ch_R - ch_L - µ) / σ
  3. First-crossing detector: scan [50ms, 500ms] post-cue for the first
     sample where |z_diff| > τ; sign gives direction.
  4. FPR: fraction of REST-window samples where |z_diff| > τ.

LOO protocol:
  - τ is selected on the two *training* subjects (maximise F1 at ≤300ms).
  - Test subject is evaluated with that τ — no look-ahead.

Usage:
    source .venv/bin/activate
    python scripts/eval_classifier.py
    python scripts/eval_classifier.py --tau 2.0          # fixed τ, skip sweep
    python scripts/eval_classifier.py --no-plot          # suppress matplotlib
    python scripts/eval_classifier.py --post 0.4         # shorter detection window
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "eog"

# Filter chain shared by all callers (matches record_eog.py exactly)
LPF_HZ      = 100.0
HPF_HZ      = 0.5
NOTCH_BANDS = ((48.0, 52.0), (58.0, 62.0))

# Detection window (seconds, relative to cue onset)
DEFAULT_DETECT_PRE  = 0.05   # ignore first 50 ms (motor/blink artifact)
DEFAULT_DETECT_POST = 0.50   # 500 ms max detection window

# Baseline fitting: skip leading seconds to let IIR HPF settle
BASELINE_SETTLE_S = 2.0

# FPR: skip leading seconds of each REST window (return saccade period),
# then look for a threshold crossing in the remaining window.
FPR_SKIP_S = 0.5

# Threshold sweep range (σ units)
TAU_MIN, TAU_MAX, TAU_N = 0.3, 5.0, 60


# ── DSP ───────────────────────────────────────────────────────────────────────

def filt(x_uv, sr):
    """Filter a 1-D µV array in-place on a fresh copy."""
    y = np.ascontiguousarray(x_uv.astype(np.float64))
    if y.size < 20:
        return y
    DataFilter.detrend(y, DetrendOperations.CONSTANT.value)
    DataFilter.perform_lowpass(y,  sr, LPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    for lo, hi in NOTCH_BANDS:
        DataFilter.perform_bandstop(y, sr, lo, hi, 3, FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_highpass(y, sr, HPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    return y


# ── Loading ───────────────────────────────────────────────────────────────────

def load(path):
    d = np.load(path, allow_pickle=True)
    return {
        'path':    path,
        'subject': str(d['subject_id'][0]),
        'sr':      int(d['sample_rate'][0]),
        'ch_L':    int(d['eog_ch_L'][0]),
        'ch_R':    int(d['eog_ch_R'][0]),
        'eeg':     d['eeg'],                          # (n_rows, n_samples), raw volts
        'ev_s':    d['event_samples'].astype(int),
        'ev_l':    d['event_labels'].astype(str),
    }


# ── Per-subject feature extraction ───────────────────────────────────────────

def prepare(rec):
    """
    Returns a dict with:
      diff_z   — full-session z-scored HEOG diff (filtered)
      baseline_mu, baseline_sigma — normalisation constants
      ev_s, ev_l — event sample indices and labels
      sr

    Baseline µ/σ is fitted on the BASELINE segment, skipping the first
    BASELINE_SETTLE_S seconds to let the IIR HPF startup transient decay.

    Why not REST segments: the 1.5 s inter-trial REST windows begin with a
    return saccade (subject re-centring their gaze), so REST σ is inflated
    to roughly the same amplitude as the intentional cued saccades — making
    the z-score useless for discrimination.  The BASELINE (eyes-forward,
    undisturbed) gives a true noise-floor σ (~1-3 µV), giving 8-15 σ SNR.
    """
    sr, ch_L, ch_R = rec['sr'], rec['ch_L'], rec['ch_R']
    eeg, ev_s, ev_l = rec['eeg'], rec['ev_s'], rec['ev_l']

    # Diff-first, then filter WITHOUT detrend.
    #
    # Two issues resolved here:
    #
    # 1. Diff-first: individual channels can carry ±100 mV of common-mode
    #    noise (body motion, electrode artifact) that nearly cancels in the
    #    differential.  Filtering each channel separately then subtracting
    #    causes the HPF IIR transients to scale with the per-channel amplitude
    #    (>>  EOG signal).  Diff-first ensures the filter only sees the small
    #    differential signal (typically ±1 mV).
    #
    # 2. No detrend: DataFilter.detrend subtracts the FULL-SESSION mean.
    #    For subjects with large slow electrode drift (e.g., Player F's ch_R
    #    drifts ~66 mV over the session), the full-session diff mean is
    #    ~-66,000 µV even though the baseline diff is only ~-825 µV.
    #    Detrend then makes the initial condition +65,000 µV → massive HPF
    #    transient that dominates the BASELINE window.  Without detrend the
    #    IIR HPF sees the actual initial diff value (-825 µV), which decays
    #    to <2 µV by the 2-second settle skip.
    diff_raw = (eeg[ch_R] - eeg[ch_L]) * 1e6
    y = np.ascontiguousarray(diff_raw.astype(np.float64))
    DataFilter.perform_lowpass(y,  sr, LPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    for lo, hi in NOTCH_BANDS:
        DataFilter.perform_bandstop(y, sr, lo, hi, 3, FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_highpass(y, sr, HPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    diff = y

    # BASELINE segment (after IIR settling)
    baseline_onset = next((s for s, l in zip(ev_s, ev_l) if l == 'BASELINE'), None)
    if baseline_onset is None:
        raise ValueError(f"{rec['subject']}: no BASELINE event found")
    first_non_baseline = next(
        (s for s, l in zip(ev_s, ev_l) if l not in ('BASELINE',)),
        len(diff),
    )
    skip_n       = int(BASELINE_SETTLE_S * sr)
    baseline_seg = diff[baseline_onset + skip_n : first_non_baseline]
    if baseline_seg.size < 10:
        raise ValueError(f"{rec['subject']}: BASELINE too short after {BASELINE_SETTLE_S}s settle skip")

    mu    = float(np.mean(baseline_seg))
    sigma = float(np.std(baseline_seg))
    if sigma < 1e-6:
        sigma = 1e-6

    diff_z = (diff - mu) / sigma

    return {
        'subject':         rec['subject'],
        'sr':              sr,
        'diff_z':          diff_z,
        'baseline_mu':     mu,
        'baseline_sigma':  sigma,
        'ev_s':            ev_s,
        'ev_l':            ev_l,
        'n_samps':         diff.size,
    }


# ── Trial-level detector ──────────────────────────────────────────────────────

def classify_trial(diff_z, onset_s, n_samps, sr, tau, pre_s, post_s):
    """
    First-crossing detector on a z-scored diff trace.

    Returns:
      (decision, latency_s)
      decision  — 'LEFT' | 'RIGHT' | 'NONE'
      latency_s — seconds from onset to crossing, or None
    """
    i0 = onset_s + int(pre_s  * sr)
    i1 = onset_s + int(post_s * sr)
    i0 = max(0, min(i0, n_samps))
    i1 = max(0, min(i1, n_samps))

    window = diff_z[i0:i1]
    if window.size == 0:
        return 'NONE', None

    crossings = np.where(np.abs(window) > tau)[0]
    if crossings.size == 0:
        return 'NONE', None

    first = crossings[0]
    latency_s = (first + int(pre_s * sr)) / sr
    decision  = 'RIGHT' if window[first] > 0 else 'LEFT'
    return decision, latency_s


def fpr_for_subject(prep, tau, pre_s, post_s):
    """
    FPR = fraction of settled-fixation windows (late REST) where the detector fires.

    The first FPR_SKIP_S of each REST window is skipped — it contains the
    return saccade as the subject re-centres their gaze, which is a real eye
    movement, not a false detection.  Only the remaining portion (eyes
    truly settled) is used to estimate false-alarm rate.
    """
    diff_z  = prep['diff_z']
    ev_s    = prep['ev_s']
    ev_l    = prep['ev_l']
    sr      = prep['sr']
    n_samps = prep['n_samps']
    skip_n  = int(FPR_SKIP_S * sr)

    fired = []
    for i, (s, l) in enumerate(zip(ev_s, ev_l)):
        if l != 'REST':
            continue
        end  = int(ev_s[i + 1]) if i + 1 < len(ev_s) else n_samps
        seg  = diff_z[s + skip_n : end]
        if seg.size < 2:
            continue
        fired.append(bool(np.any(np.abs(seg) > tau)))

    return float(np.mean(fired)) if fired else 0.0


# ── Per-subject eval at a fixed τ ─────────────────────────────────────────────

def eval_subject(prep, tau, pre_s, post_s):
    """Evaluate all LEFT/RIGHT trials for one subject at threshold τ."""
    diff_z  = prep['diff_z']
    ev_s    = prep['ev_s']
    ev_l    = prep['ev_l']
    sr      = prep['sr']
    n_samps = prep['n_samps']

    trials = []
    for s, l in zip(ev_s, ev_l):
        if l not in ('LEFT', 'RIGHT'):
            continue
        decision, lat = classify_trial(diff_z, s, n_samps, sr, tau, pre_s, post_s)
        correct = (decision == l)
        trials.append({'label': l, 'decision': decision, 'correct': correct, 'latency': lat})

    n        = len(trials)
    n_corr   = sum(t['correct'] for t in trials)
    acc      = n_corr / n if n else 0.0
    lats     = [t['latency'] for t in trials if t['latency'] is not None]
    lat_med  = float(np.median(lats)) if lats else None
    fpr      = fpr_for_subject(prep, tau, pre_s, post_s)

    return {'acc': acc, 'n': n, 'n_correct': n_corr,
            'lat_med': lat_med, 'lats': lats, 'fpr': fpr, 'trials': trials}


# ── τ selection on training set ───────────────────────────────────────────────

def select_tau(train_preps, tau_grid, pre_s, post_s,
               max_latency_s=0.3, max_fpr=0.05):
    """
    Pick τ from tau_grid that maximises accuracy on training subjects
    subject to:  median latency ≤ max_latency_s  AND  mean FPR ≤ max_fpr.
    Tie-break: higher τ (lower FPR).  Falls back to best-accuracy τ if no
    τ satisfies both constraints simultaneously.
    """
    best_tau, best_acc = tau_grid[-1], -1.0
    fallback_tau, fallback_acc = tau_grid[-1], -1.0

    for tau in tau_grid:
        accs, lats, fprs = [], [], []
        for p in train_preps:
            res = eval_subject(p, tau, pre_s, post_s)
            accs.append(res['acc'])
            if res['lat_med'] is not None:
                lats.append(res['lat_med'])
            fprs.append(res['fpr'])

        mean_acc = float(np.mean(accs)) if accs else 0.0
        med_lat  = float(np.median(lats)) if lats else 999.0
        mean_fpr = float(np.mean(fprs)) if fprs else 1.0

        if mean_acc > fallback_acc:
            fallback_acc = mean_acc
            fallback_tau = tau

        if med_lat <= max_latency_s and mean_fpr <= max_fpr and mean_acc > best_acc:
            best_acc = mean_acc
            best_tau = tau

    return best_tau if best_acc > -1.0 else fallback_tau


# ── Tradeoff curve (single subject or aggregate) ──────────────────────────────

def sweep_tau(preps, tau_grid, pre_s, post_s):
    """Return arrays (tau_grid, mean_acc, mean_lat_med, mean_fpr) across subjects."""
    accs, lats, fprs = [], [], []
    for tau in tau_grid:
        a_list, l_list, f_list = [], [], []
        for p in preps:
            r = eval_subject(p, tau, pre_s, post_s)
            a_list.append(r['acc'])
            if r['lat_med'] is not None:
                l_list.append(r['lat_med'])
            f_list.append(r['fpr'])
        accs.append(np.mean(a_list))
        lats.append(np.mean(l_list) if l_list else np.nan)
        fprs.append(np.mean(f_list))
    return np.array(accs), np.array(lats), np.array(fprs)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_table(loo_results, selected_taus):
    header = f"{'Subject':<12} {'τ(σ)':>6} {'Acc':>7} {'N':>4} {'Lat p50(ms)':>12} {'FPR':>7}"
    print()
    print(header)
    print("─" * len(header))
    for res in loo_results:
        lat_str = f"{res['lat_med']*1000:.0f}" if res['lat_med'] is not None else " n/a"
        print(f"{res['subject']:<12} {res['tau']:>6.2f} {res['acc']*100:>6.1f}% "
              f"{res['n']:>4}  {lat_str:>10} ms  {res['fpr']*100:>5.1f}%")

    print("─" * len(header))
    accs = [r['acc'] for r in loo_results]
    lats = [r['lat_med'] for r in loo_results if r['lat_med'] is not None]
    fprs = [r['fpr'] for r in loo_results]
    lat_str = f"{np.mean(lats)*1000:.0f}" if lats else " n/a"
    print(f"{'LOO avg':<12} {'':>6} {np.mean(accs)*100:>6.1f}% "
          f"{'':>4}  {lat_str:>10} ms  {np.mean(fprs)*100:>5.1f}%")
    print()

    targets_met = (
        np.mean(accs) >= 0.85
        and (np.mean(lats) * 1000 <= 300 if lats else False)
        and np.mean(fprs) <= 0.05
    )
    if targets_met:
        print("Phase 1 targets MET  (≥85% acc, ≤300ms latency, <5% FPR)")
    else:
        reasons = []
        if np.mean(accs) < 0.85:
            reasons.append(f"acc {np.mean(accs)*100:.1f}% < 85%")
        if lats and np.mean(lats) * 1000 > 300:
            reasons.append(f"latency {np.mean(lats)*1000:.0f}ms > 300ms")
        if np.mean(fprs) > 0.05:
            reasons.append(f"FPR {np.mean(fprs)*100:.1f}% > 5%")
        print(f"Phase 1 targets NOT met: {', '.join(reasons)}")
    print()


def plot_tradeoff(preps, tau_grid, pre_s, post_s, selected_taus, loo_results):
    accs, lats, fprs = sweep_tau(preps, tau_grid, pre_s, post_s)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("EOG classifier — threshold sweep (all subjects pooled)", fontsize=12)

    axes[0].plot(tau_grid, accs * 100, color='#4499ff', lw=2)
    axes[0].axhline(85, color='#ffffff', lw=0.8, linestyle='--', alpha=0.4, label='85% target')
    axes[0].set_xlabel('Threshold τ (σ)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy vs τ')
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 105)

    axes[1].plot(tau_grid, lats * 1000, color='#ffcc44', lw=2)
    axes[1].axhline(300, color='#ffffff', lw=0.8, linestyle='--', alpha=0.4, label='300ms target')
    axes[1].set_xlabel('Threshold τ (σ)')
    axes[1].set_ylabel('Median latency (ms)')
    axes[1].set_title('Latency vs τ')
    axes[1].legend(fontsize=8)

    axes[2].plot(tau_grid, fprs * 100, color='#ff5555', lw=2)
    axes[2].axhline(5, color='#ffffff', lw=0.8, linestyle='--', alpha=0.4, label='5% FPR target')
    axes[2].set_xlabel('Threshold τ (σ)')
    axes[2].set_ylabel('False positive rate (%)')
    axes[2].set_title('FPR vs τ')
    axes[2].legend(fontsize=8)

    # Mark LOO-selected τ for each subject
    colors = ['#00ccff', '#ffcc00', '#ff44cc', '#44ff99', '#ff7744']
    for i, (res, c) in enumerate(zip(loo_results, colors)):
        for ax in axes:
            ax.axvline(res['tau'], color=c, lw=1.2, linestyle=':', alpha=0.7,
                       label=res['subject'] if ax is axes[0] else None)
    axes[0].legend(fontsize=7)

    for ax in axes:
        ax.set_facecolor('#111111')
        ax.tick_params(colors='#888888')
        ax.grid(True, color='#222222')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333333')
    fig.patch.set_facecolor('#0c0c0c')

    plt.tight_layout()
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*',
                        help='.npz recordings (default: all in data/eog/)')
    parser.add_argument('--tau',  type=float, default=None,
                        help='Fixed threshold in σ units (skip LOO τ-selection)')
    parser.add_argument('--pre',  type=float, default=DEFAULT_DETECT_PRE,
                        help=f'Detection window start post-cue (s), default {DEFAULT_DETECT_PRE}')
    parser.add_argument('--post', type=float, default=DEFAULT_DETECT_POST,
                        help=f'Detection window end post-cue (s), default {DEFAULT_DETECT_POST}')
    parser.add_argument('--no-plot', action='store_true', help='Suppress matplotlib output')
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = sorted(RECORDINGS_DIR.glob("*.npz"))

    if not paths:
        print("No recordings found.")
        sys.exit(1)

    print(f"Loading {len(paths)} recording(s)…")
    recs   = [load(p) for p in paths]
    preps  = [prepare(r) for r in recs]
    subjects = [p['subject'] for p in preps]
    print(f"Subjects: {', '.join(subjects)}")
    for p in preps:
        n_trials = sum(1 for l in p['ev_l'] if l in ('LEFT', 'RIGHT'))
        print(f"  {p['subject']}: {n_trials} trials, "
              f"baseline σ={p['baseline_sigma']:.2f} µV  "
              f"(τ=2σ → {2*p['baseline_sigma']:.1f} µV threshold)")

    tau_grid = np.linspace(TAU_MIN, TAU_MAX, TAU_N)

    loo_results   = []
    selected_taus = []

    if len(preps) < 2:
        print("\nOnly one subject — running single-subject eval (no LOO).")
        tau = args.tau if args.tau is not None else 2.0
        res = eval_subject(preps[0], tau, args.pre, args.post)
        res['subject'] = preps[0]['subject']
        res['tau']     = tau
        loo_results.append(res)
        selected_taus.append(tau)
    else:
        for i, test_prep in enumerate(preps):
            train_preps = [p for j, p in enumerate(preps) if j != i]

            if args.tau is not None:
                tau = args.tau
            else:
                tau = select_tau(train_preps, tau_grid, args.pre, args.post)

            res = eval_subject(test_prep, tau, args.pre, args.post)
            res['subject'] = test_prep['subject']
            res['tau']     = tau
            loo_results.append(res)
            selected_taus.append(tau)

    print_table(loo_results, selected_taus)

    if not args.no_plot:
        plot_tradeoff(preps, tau_grid, args.pre, args.post, selected_taus, loo_results)


if __name__ == '__main__':
    main()
