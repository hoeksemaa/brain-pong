"""
SSVEP detection algorithms for BrainPong.

Single source of truth for the signal-processing pipeline. Both the live game
(pong_game_brainflow.py) and the bench (bench.py) import from here. Don't
re-inline algorithm logic in either of those — edit it here.

DATA INTEGRITY: every function returns a fresh array. BrainFlow's `DataFilter.*`
functions mutate their input in place; we always operate on copies of the
caller-supplied arrays.
"""

import numpy as np
from brainflow.data_filter import (
    DataFilter, FilterTypes, DetrendOperations, AggOperations,
)
from sklearn.cross_decomposition import CCA

# --- Defaults (mirror the live game's tuned values) ---------------------------
DEFAULT_FREQ_L = 10.0
DEFAULT_FREQ_R = 15.0
DEFAULT_HARMONICS = 3
DEFAULT_HPF_HZ = 5.0
DEFAULT_LPF_HZ = 45.0
DEFAULT_NOTCH_BANDS = ((48.0, 52.0), (58.0, 62.0))
DEFAULT_FILTER_ORDER = 4
DEFAULT_EMA_ALPHA = 0.4
DEFAULT_SCORE_AMPLIFIER = 2.5

# Single sklearn CCA instance reused across calls for speed; the .fit() call
# rebuilds internal state per window.
_cca_model = CCA(n_components=1)


def filter_window(eeg_window, sampling_rate,
                  hpf_hz=DEFAULT_HPF_HZ, lpf_hz=DEFAULT_LPF_HZ,
                  notch_bands=DEFAULT_NOTCH_BANDS,
                  filter_order=DEFAULT_FILTER_ORDER):
    """Apply BrainPong's standard filter chain to a multi-channel EEG window.

    detrend → LP → HP → bandstop notches → 3-tap rolling median, per channel.

    Args:
        eeg_window:    np.ndarray (n_samples, n_channels). NOT mutated.
        sampling_rate: int.
        hpf_hz:        highpass cutoff (default 5.0).
        lpf_hz:        lowpass cutoff (default 45.0).
        notch_bands:   iterable of (lo, hi) bandstop pairs (default 50/60 Hz line).
        filter_order:  Butterworth order for HP/LP (default 4).

    Returns:
        np.ndarray, same shape, fresh array.
    """
    if eeg_window.ndim == 1:
        eeg_window = eeg_window.reshape(-1, 1)
    if eeg_window.ndim != 2:
        raise ValueError(f"expected (n_samples, n_channels), got {eeg_window.shape}")

    # Channel-first copy (BrainFlow expects 1D contiguous arrays per call)
    out = np.ascontiguousarray(eeg_window.T.astype(np.float64))
    n_channels = out.shape[0]
    for i in range(n_channels):
        if out[i].size <= 20:
            continue
        DataFilter.detrend(out[i], DetrendOperations.CONSTANT.value)
        DataFilter.perform_lowpass(out[i], sampling_rate, float(lpf_hz),
                                   filter_order, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_highpass(out[i], sampling_rate, float(hpf_hz),
                                    filter_order, FilterTypes.BUTTERWORTH.value, 0)
        for lo, hi in notch_bands:
            DataFilter.perform_bandstop(out[i], sampling_rate,
                                        float(lo), float(hi), 3,
                                        FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_rolling_filter(out[i], 3, AggOperations.MEDIAN.value)
    return out.T


def make_ssvep_references(n_samples, sampling_rate, freq, harmonics=DEFAULT_HARMONICS):
    """Build the (n_samples, 2*harmonics) sin/cos reference matrix for CCA."""
    t = np.arange(n_samples) / float(sampling_rate)
    refs = []
    for h in range(1, harmonics + 1):
        refs.append(np.sin(2 * np.pi * h * freq * t))
        refs.append(np.cos(2 * np.pi * h * freq * t))
    return np.array(refs).T


def cca_correlation(eeg_window, references):
    """CCA correlation between an EEG window and a reference matrix.

    Args:
        eeg_window: (n_samples, n_channels). Pre-filtered ideally.
        references: (n_samples, n_refs).

    Returns:
        float in [-1, 1], or 0.0 on failure.
    """
    if eeg_window.shape[0] < eeg_window.shape[1] or eeg_window.shape[0] != references.shape[0]:
        return 0.0
    try:
        _cca_model.fit(eeg_window, references)
        U, V = _cca_model.transform(eeg_window, references)
        return float(np.corrcoef(U.T, V.T)[0, 1])
    except Exception:
        return 0.0


def cca_decision(eeg_window, state, sampling_rate,
                 freq_l=DEFAULT_FREQ_L, freq_r=DEFAULT_FREQ_R,
                 harmonics=DEFAULT_HARMONICS,
                 hpf_hz=DEFAULT_HPF_HZ, lpf_hz=DEFAULT_LPF_HZ,
                 ema_alpha=DEFAULT_EMA_ALPHA,
                 score_amplifier=DEFAULT_SCORE_AMPLIFIER,
                 thresholds=None):
    """One step of the CCA-based SSVEP decision pipeline.

    Pipeline: filter → CCA vs L freq → CCA vs R freq → diff → amplify → EMA → decide.

    Args:
        eeg_window: (n_samples, n_channels). NOT mutated.
        state:      dict with key 'smoothed' (float). Pass {} on first call.
        sampling_rate: int.
        freq_l, freq_r: target stimulus frequencies (Hz).
        harmonics: how many harmonics in the CCA reference set.
        hpf_hz, lpf_hz: bandpass cutoffs.
        ema_alpha: weight on the previous smoothed score.
        score_amplifier: scale on the raw (corr_R − corr_L) diff.
        thresholds: None for raw_sign mode (sign of smoothed score),
                    or {'left': float, 'right': float} for thresholded mode.

    Returns:
        (label, new_state, debug)
        label:     'L' | 'R' | 'NEUTRAL'
        new_state: {'smoothed': float}
        debug:     {'raw_score', 'smoothed_score', 'corr_L', 'corr_R'}
    """
    filtered = filter_window(eeg_window, sampling_rate, hpf_hz=hpf_hz, lpf_hz=lpf_hz)
    n_samples = filtered.shape[0]

    refs_l = make_ssvep_references(n_samples, sampling_rate, freq_l, harmonics)
    refs_r = make_ssvep_references(n_samples, sampling_rate, freq_r, harmonics)
    cl = cca_correlation(filtered, refs_l)
    cr = cca_correlation(filtered, refs_r)

    raw = (cr - cl) * score_amplifier
    prev = state.get('smoothed', 0.0)
    smoothed = ema_alpha * prev + (1.0 - ema_alpha) * raw

    if thresholds is None:
        # raw_sign mode — equivalent to thresholds at 0 in both directions
        if smoothed > 0:
            label = 'R'
        elif smoothed < 0:
            label = 'L'
        else:
            label = 'NEUTRAL'
    else:
        if smoothed > thresholds['right']:
            label = 'R'
        elif smoothed < thresholds['left']:
            label = 'L'
        else:
            label = 'NEUTRAL'

    return label, {'smoothed': smoothed}, {
        'raw_score': raw,
        'smoothed_score': smoothed,
        'corr_L': cl,
        'corr_R': cr,
    }
