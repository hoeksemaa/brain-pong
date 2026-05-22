"""
Preprocessing functions: NPZ path → PrepResult dict.

Each function takes a path to a recordings/eog/*.npz file and returns a
PrepResult with these guaranteed keys:

    subject        str
    sr             int          samples per second
    diff_uv        ndarray      raw HEOG differential (or σ-scaled), shape (n_samples,)
    baseline_mu    float        noise-floor mean — diagnostic
    baseline_sigma float        noise-floor std  — diagnostic, not an opt target
    ev_s           ndarray int  event onset sample indices
    ev_l           ndarray str  event labels
    snr            float        median trial peak / baseline_sigma — diagnostic only

Optional key (set by all preprocessors except segment_diff_filter):

    filter_fn      callable     filter_fn(x, sr) → filtered array; used by detect.py
                                per-segment. If absent, detect.py falls back to its
                                own default IIR chain (0.5–100 Hz).

All fields are derived purely from the NPZ; no external inputs required.
"""

# ── Pipeline catalogue ─────────────────────────────────────────────────────────
#
# segment_diff_filter  [baseline]
#   Filter : 0.5–100 Hz causal IIR (brainflow Butterworth, constant detrend)
#   Notes  : Per-segment, diff-first. Highest accuracy (94% avg). Brainflow's
#            constant detrend handles the per-window edge conditions well.
#            Latency ~429 ms p50. FPR high at 10 µV threshold (55% avg) bc
#            the wide LPF passes EMG that trips the threshold in REST windows.
#
# narrow_band
#   Filter : 0.1–30 Hz causal IIR (scipy sosfilt, zi-settled)
#   Notes  : Literature-consensus HEOG band. Lower HPF (0.1 Hz) preserves
#            more of the saccade step; lower LPF (30 Hz) cuts EMG noise.
#            Accuracy 87% avg — slightly hurt by the scipy zi interaction with
#            the per-window edge vs brainflow's constant-detrend approach.
#            FPR modestly better (49% avg).
#
# narrow_band_detrend
#   Filter : mean-subtract per segment → 0.1–30 Hz causal IIR (scipy sosfilt, zi-settled)
#   Notes  : Hypothesis closer for the 87% vs 94% gap between narrow_band and
#            seg-diff-filter. Brainflow's CONSTANT detrend (DataFilter.detrend)
#            subtracts the segment mean before the IIR chain; narrow_band skips
#            this step. This variant adds it back explicitly. If accuracy lifts
#            to ~94%, mean-removal is the key variable, not the filter library.
#
# narrow_lpf
#   Filter : 0.5–30 Hz causal IIR (scipy sosfilt, zi-settled)
#   Notes  : Same HPF as baseline, LPF moved 100→30 Hz. Isolates the effect
#            of cutting EMG noise. Accuracy 89% avg; FPR best of the amplitude
#            pipelines (41% avg). German FPR drops to 8% — the EMG cut is
#            meaningful for at least one subject.
#
# linear_detrend
#   Filter : 0.5–100 Hz causal IIR (brainflow) with linear detrend instead
#            of constant (mean-subtraction) detrend per segment.
#   Notes  : Handles residual linear electrode drift within the pre-cue window.
#            Results nearly identical to baseline (93% acc, 433 ms, 51% FPR)
#            — subjects don't appear to have significant intra-window drift.
#            Safe to include; negligible cost.
#
# velocity
#   Filter : 0.1–30 Hz narrow_band → Savitzky-Golay first derivative
#            (order 2, window 21 samples = 84 ms at 250 Hz)
#   Notes  : Stores eye velocity (µV/s) rather than displacement. Velocity
#            peak precedes displacement peak by 50–150 ms → lower latency.
#            Latency 402 ms avg (27 ms faster than baseline). Accuracy 93%.
#            FPR 80% at 100 µV/s threshold — that threshold is a rough first
#            guess and needs sweeping. Best pipeline for latency reduction.
#
# z_normalized
#   Filter : Same as narrow_band; diff_uv stored as raw_diff / baseline_sigma
#   Notes  : Amplitudes in units of subject's own noise floor (dimensionless).
#            baseline_sigma = 1.0 by definition. Makes fpr_threshold_uv
#            cross-subject consistent (3σ means the same thing for everyone
#            regardless of electrode impedance). Accuracy same as narrow_band
#            (87%); FPR at 3σ: 38% avg — John drops from 100% to 56%.
#
# ── NOTE on zero-phase filtering ──────────────────────────────────────────────
# sosfiltfilt (forward+backward) was tried but creates anticausal pre-ringing
# with opposite sign to the saccade. The peak_sign detector's argmax(abs) picks
# up this pre-ringing and classifies the direction backwards. Zero-phase is
# valid offline but requires a detector that accounts for pre-ringing (e.g.,
# onset detection rather than peak-amplitude detection).
# ──────────────────────────────────────────────────────────────────────────────

import functools
import numpy as np
from pathlib import Path
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt, savgol_filter, detrend as sp_detrend

# Standard filter chain parameters
LPF_HZ      = 100.0
HPF_HZ      = 0.5
NOTCH_BANDS = ((48.0, 52.0), (58.0, 62.0))

# How far into the BASELINE segment to skip before fitting µ/σ (IIR settling)
BASELINE_SETTLE_S = 2.0

# Detection window used only for SNR diagnostic
SNR_PRE_S  = 0.05
SNR_POST_S = 0.50


def _load_npz(path):
    d = np.load(path, allow_pickle=True)
    return {
        'eeg':     d['eeg'],
        'sr':      int(d['sample_rate'][0]),
        'ch_L':    int(d['eog_ch_L'][0]),
        'ch_R':    int(d['eog_ch_R'][0]),
        'ev_s':    d['event_samples'].astype(int),
        'ev_l':    d['event_labels'].astype(str),
        'subject': str(d['subject_id'][0]),
    }


def _apply_filter(x_uv, sr):
    """Standard EOG filter chain on a 1-D µV array. Returns a fresh copy."""
    y = np.ascontiguousarray(x_uv.astype(np.float64))
    if y.size < 20:
        return y
    DataFilter.detrend(y, DetrendOperations.CONSTANT.value)
    DataFilter.perform_lowpass(y,  sr, LPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    for lo, hi in NOTCH_BANDS:
        DataFilter.perform_bandstop(y, sr, lo, hi, 3, FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_highpass(y, sr, HPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    return y


def _sosfilt_settled(sos, x):
    """Causal sosfilt with zi initialised to steady-state (avoids startup spike)."""
    zi = sosfilt_zi(sos) * x[0]
    y, _ = sosfilt(sos, x, zi=zi)
    return y


@functools.lru_cache(maxsize=8)
def _sos_narrow(sr):
    return butter(4, [0.1, 30.0], btype='band', fs=sr, output='sos')


@functools.lru_cache(maxsize=8)
def _sos_lpf30(sr):
    return butter(4, [0.5, 30.0], btype='band', fs=sr, output='sos')


def _apply_filter_narrow(x_uv, sr):
    """0.1–30 Hz bandpass, causal IIR (zi-settled sosfilt). Clinical HEOG band."""
    y = x_uv.astype(np.float64).copy()
    if y.size < 20:
        return y
    return _sosfilt_settled(_sos_narrow(sr), y)


def _apply_filter_zerophase(x_uv, sr):
    """0.5–30 Hz bandpass, causal IIR (zi-settled). Same HPF as baseline, LPF at 30 Hz."""
    y = x_uv.astype(np.float64).copy()
    if y.size < 20:
        return y
    return _sosfilt_settled(_sos_lpf30(sr), y)


def _apply_filter_linear_detrend(x_uv, sr):
    """Linear (slope+offset) detrend per segment, then 0.5–100 Hz causal IIR."""
    y = x_uv.astype(np.float64).copy()
    if y.size < 20:
        return y
    y = sp_detrend(y, type='linear')
    y = np.ascontiguousarray(y)
    DataFilter.perform_lowpass(y, sr, LPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    for lo, hi in NOTCH_BANDS:
        DataFilter.perform_bandstop(y, sr, lo, hi, 3, FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_highpass(y, sr, HPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    return y


def _apply_filter_velocity(x_uv, sr):
    """narrow_band → Savitzky-Golay first derivative (µV/s, window=21, order=2)."""
    y = _apply_filter_narrow(x_uv, sr)
    if y.size < 21:
        return y
    return savgol_filter(y, window_length=21, polyorder=2, deriv=1, delta=1.0 / sr)


def _snr(diff_uv, ev_s, ev_l, sr, baseline_sigma,
         pre_s=SNR_PRE_S, post_s=SNR_POST_S):
    """Median |peak| across all trials divided by baseline_sigma."""
    n = diff_uv.size
    peaks = []
    for s, l in zip(ev_s, ev_l):
        if l not in ('LEFT', 'RIGHT'):
            continue
        i0 = max(0, s + int(pre_s * sr))
        i1 = min(n, s + int(post_s * sr))
        seg = diff_uv[i0:i1]
        if seg.size > 0:
            peaks.append(float(np.max(np.abs(seg))))
    if not peaks or baseline_sigma < 1e-9:
        return float('nan')
    return float(np.median(peaks) / baseline_sigma)


def _fit_baseline(diff_raw, ev_s, ev_l, sr, filter_fn):
    """Compute baseline µ/σ from the BASELINE segment using the given filter_fn."""
    base_s   = next((s for s, l in zip(ev_s, ev_l) if l == 'BASELINE'), 0)
    first_nb = next((s for s, l in zip(ev_s, ev_l) if l not in ('BASELINE',)),
                    diff_raw.size)
    skip_n   = int(BASELINE_SETTLE_S * sr)
    bl_raw   = diff_raw[base_s + skip_n : first_nb]
    bl_filt  = filter_fn(bl_raw.copy(), sr)
    return float(np.mean(bl_filt)), float(np.std(bl_filt)) or 1e-6


def _make_result(raw, diff_uv, baseline_mu, baseline_sigma, filter_fn=None):
    snr = _snr(diff_uv, raw['ev_s'], raw['ev_l'], raw['sr'], baseline_sigma)
    result = {
        'subject':         raw['subject'],
        'sr':              raw['sr'],
        'diff_uv':         diff_uv,
        'baseline_mu':     baseline_mu,
        'baseline_sigma':  baseline_sigma,
        'ev_s':            raw['ev_s'],
        'ev_l':            raw['ev_l'],
        'snr':             snr,
    }
    if filter_fn is not None:
        result['filter_fn'] = filter_fn
    return result


# ── Preprocessors ─────────────────────────────────────────────────────────────

def segment_diff_filter(path):
    """
    Per-segment, diff-first filter. Baseline.

    Stores raw full-session differential; detect.py re-filters each window with
    its own 0.5–100 Hz causal IIR chain (constant detrend → LPF → notch → HPF).
    """
    raw      = _load_npz(path)
    diff_raw = (raw['eeg'][raw['ch_R']] - raw['eeg'][raw['ch_L']]) * 1e6
    mu, sigma = _fit_baseline(diff_raw, raw['ev_s'], raw['ev_l'], raw['sr'], _apply_filter)
    # No filter_fn stored — detect.py uses its own default IIR chain.
    return _make_result(raw, diff_raw.copy(), mu, sigma)


def narrow_band(path):
    """
    Per-segment diff-first filter, 0.1–30 Hz causal IIR.

    Literature-consensus HEOG band. Lower LPF (30 Hz) removes EMG noise;
    lower HPF (0.1 Hz) preserves more of the saccade step response.
    Causal sosfilt with zi-initialization avoids the anticausal pre-ringing
    that zero-phase (sosfiltfilt) introduces, which inverts peak_sign polarity.
    """
    raw      = _load_npz(path)
    diff_raw = (raw['eeg'][raw['ch_R']] - raw['eeg'][raw['ch_L']]) * 1e6
    mu, sigma = _fit_baseline(diff_raw, raw['ev_s'], raw['ev_l'], raw['sr'],
                               _apply_filter_narrow)
    return _make_result(raw, diff_raw.copy(), mu, sigma, _apply_filter_narrow)


def narrow_band_detrend(path):
    """
    narrow_band + explicit constant detrend (mean subtraction) per segment.

    Hypothesis: brainflow's CONSTANT detrend (mean-subtract before IIR) is
    responsible for the 94% vs 87% accuracy gap between seg-diff-filter and
    narrow_band. This variant replicates that step with an explicit mean
    subtraction before the zi-settled scipy IIR, keeping the 0.1–30 Hz band.
    """
    def _filter_fn(x_uv, sr):
        y = x_uv.astype(np.float64).copy()
        if y.size < 20:
            return y
        y -= y.mean()
        return _sosfilt_settled(_sos_narrow(sr), y)

    raw      = _load_npz(path)
    diff_raw = (raw['eeg'][raw['ch_R']] - raw['eeg'][raw['ch_L']]) * 1e6
    mu, sigma = _fit_baseline(diff_raw, raw['ev_s'], raw['ev_l'], raw['sr'],
                               _filter_fn)
    return _make_result(raw, diff_raw.copy(), mu, sigma, _filter_fn)


def narrow_lpf(path):
    """
    Per-segment diff-first filter, 0.5–30 Hz causal IIR.

    Same HPF as segment_diff_filter (0.5 Hz); LPF moved from 100→30 Hz.
    Isolates the effect of the LPF cutoff: removes EMG noise (>30 Hz) more
    aggressively while preserving the same low-frequency behaviour as baseline.
    """
    raw      = _load_npz(path)
    diff_raw = (raw['eeg'][raw['ch_R']] - raw['eeg'][raw['ch_L']]) * 1e6
    mu, sigma = _fit_baseline(diff_raw, raw['ev_s'], raw['ev_l'], raw['sr'],
                               _apply_filter_zerophase)
    return _make_result(raw, diff_raw.copy(), mu, sigma, _apply_filter_zerophase)


def linear_detrend(path):
    """
    Per-segment diff-first filter, linear detrend before 0.5–100 Hz causal IIR.

    Baseline segment_diff_filter removes only the mean (constant detrend).
    This variant fits and subtracts a linear trend from each window first,
    better handling residual linear electrode drift within the pre-cue window.
    """
    raw      = _load_npz(path)
    diff_raw = (raw['eeg'][raw['ch_R']] - raw['eeg'][raw['ch_L']]) * 1e6
    mu, sigma = _fit_baseline(diff_raw, raw['ev_s'], raw['ev_l'], raw['sr'],
                               _apply_filter_linear_detrend)
    return _make_result(raw, diff_raw.copy(), mu, sigma, _apply_filter_linear_detrend)


def velocity(path):
    """
    Per-segment narrow_band filter then Savitzky-Golay first derivative.

    Stores eye velocity (µV/s) rather than displacement. Velocity peak occurs
    50–150 ms before the displacement peak, so peak_sign fires earlier →
    lower latency. baseline_sigma is in µV/s; fpr_threshold_uv in bench.py
    must be scaled accordingly (~100 µV/s is a reasonable starting point).
    """
    raw      = _load_npz(path)
    diff_raw = (raw['eeg'][raw['ch_R']] - raw['eeg'][raw['ch_L']]) * 1e6
    mu, sigma = _fit_baseline(diff_raw, raw['ev_s'], raw['ev_l'], raw['sr'],
                               _apply_filter_velocity)
    return _make_result(raw, diff_raw.copy(), mu, sigma, _apply_filter_velocity)


def z_normalized(path):
    """
    narrow_band filter; diff_uv stored as raw_diff / baseline_sigma (dimensionless).

    Amplitudes are in units of baseline noise floors. baseline_sigma = 1.0 by
    definition. Makes fpr_threshold_uv cross-subject consistent: threshold=3
    means "fire if peak > 3× the subject's own noise floor" regardless of
    electrode impedance or recording quality.
    """
    raw      = _load_npz(path)
    diff_raw = (raw['eeg'][raw['ch_R']] - raw['eeg'][raw['ch_L']]) * 1e6
    _, sigma  = _fit_baseline(diff_raw, raw['ev_s'], raw['ev_l'], raw['sr'],
                               _apply_filter_narrow)
    return _make_result(raw, diff_raw / sigma, 0.0, 1.0, _apply_filter_narrow)
