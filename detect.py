"""
Detection functions: PrepResult → DetectResult dict.

Each function takes a PrepResult (from preprocess.py) and returns a
DetectResult with these guaranteed keys:

    subject      str
    acc          float   fraction of trials correctly classified
    n            int     total trials evaluated
    lat_med_ms   float   median detection latency in milliseconds
    fpr          float   false-positive rate on settled REST windows
    trials       list    per-trial dicts: label, predicted, correct,
                         latency_ms, amplitude_uv

Latency is measured from cue onset to the moment the classifier fires.
FPR is measured on the portion of REST windows after the return saccade
has settled (skip first FPR_SKIP_S).
"""

# ── Detector catalogue ────────────────────────────────────────────────────────
#
# All benchmarked below with segment_diff_filter preprocessing.
# FPR thresholds are first-guess starting points; sweep before drawing
# conclusions.
#
# peak_sign  [baseline]
#   Decision : argmax(|signal|) in window; direction = sign of peak sample
#   Latency  : time of peak (waits for full saccade to develop)
#   Always fires; no per-trial threshold
#   Result   : 94% acc  429 ms  FPR 55% @ 10µV
#   Notes    : Simple and surprisingly accurate. High FPR bc wide LPF (100 Hz)
#              passes EMG bursts that trip the 10µV threshold in REST windows.
#
# first_crossing
#   Decision : first sample where |signal| > σ_threshold × baseline_σ
#   Latency  : time of threshold crossing (earlier than peak)
#   Falls back to peak_sign if no crossing found
#   Result   : 94% acc  395 ms  FPR 52% @ 6σ  (34 ms faster than baseline)
#   Notes    : Engbert & Kliegl (2003) canonical approach (λ=6). Identical
#              accuracy to peak_sign; fires ~34 ms earlier because it catches
#              the rising edge of the saccade rather than the peak. FPR barely
#              better — single EMG spikes still fire the threshold.
#
# sustained_crossing  ← best overall
#   Decision : first crossing of σ_threshold × σ sustained for ≥ min_dur_ms
#   Latency  : onset of first sustained run (same as first_crossing when clean)
#   Falls back to peak_sign if no sustained crossing found
#   Result   : 94% acc  395 ms  FPR 19% @ σ=6 dur=12ms  ← FPR 3× better
#   Notes    : The 12 ms persistence gate (3 samples) kills single-spike EMG
#              artifacts without hurting latency on clean saccades. John's FPR
#              drops from 100% → 0%. Strongly recommended over first_crossing.
#
# mean_sign
#   Decision : sign of mean signal over full [DET_PRE_S, DET_POST_S] window
#   Latency  : fixed at DET_POST_S = 500 ms (needs full window to decide)
#   Always fires; no per-trial threshold
#   Result   : 94% acc  500 ms  FPR  9% @ 5µV  ← best FPR
#   Notes    : Averaging over 450 ms crushes brief noise; sustained saccade
#              dominates. Best FPR of all detectors (9%). The cost is that
#              latency is always 500 ms — no early firing possible. Use when
#              FPR is the primary concern and latency budget allows 500 ms.
#
# template_corr
#   Decision : peak of cross-correlation with half-Hann bump template
#   Latency  : time of peak correlation offset by half template width
#   Falls back to peak_sign on short windows
#   Result   : 93% acc  413 ms  FPR 71% @ 5µV
#   Notes    : Template (150 ms half-Hann) matches the filtered saccade shape
#              reasonably, giving accuracy on par with baseline. High FPR
#              suggests the template correlates well with EMG bursts in REST
#              windows too — the template may be too wide, making it sensitive
#              to non-saccade events. Template tuning (width, shape) is the
#              main lever; not recommended without further calibration.
#
# ── Summary ───────────────────────────────────────────────────────────────────
#   Latency-critical      : sustained_crossing (395 ms, −34 ms vs baseline)
#   FPR-critical          : mean_sign (9% FPR, accuracy unchanged)
#   Best all-around       : sustained_crossing (equal acc, lower lat, 3× FPR)
#   Needs calibration     : template_corr
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from scipy.signal import correlate
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

LPF_HZ      = 100.0
HPF_HZ      = 0.5
NOTCH_BANDS = ((48.0, 52.0), (58.0, 62.0))

PRE_S       = 0.30   # pre-cue context fed to filter for IIR settling
DET_PRE_S   = 0.05   # detection window start after cue onset
DET_POST_S  = 0.50   # detection window end   after cue onset
FPR_SKIP_S  = 0.50   # skip this much at the start of each REST window


def _apply_filter(x_uv, sr):
    y = np.ascontiguousarray(x_uv.astype(np.float64))
    if y.size < 20:
        return y
    DataFilter.detrend(y, DetrendOperations.CONSTANT.value)
    DataFilter.perform_lowpass(y,  sr, LPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    for lo, hi in NOTCH_BANDS:
        DataFilter.perform_bandstop(y, sr, lo, hi, 3, FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_highpass(y, sr, HPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    return y


def _filter_window(diff_raw, onset_s, n_samps, sr, filter_fn=None):
    """
    Slice [onset - PRE_S, onset + DET_POST_S] from diff_raw, filter it,
    and return (filtered_segment, onset_offset_in_segment).
    """
    if filter_fn is None:
        filter_fn = _apply_filter
    pre_n  = int(PRE_S * sr)
    post_n = int(DET_POST_S * sr)
    i0 = max(0, onset_s - pre_n)
    i1 = min(n_samps, onset_s + post_n)
    seg    = filter_fn(diff_raw[i0:i1].copy(), sr)
    offset = onset_s - i0
    return seg, offset


def _first_sustained(above, min_dur):
    """
    Index of the start of the first run of consecutive True values of
    length >= min_dur in boolean array `above`. Returns None if not found.
    """
    if min_dur <= 0 or len(above) < min_dur:
        return None
    kernel = np.ones(min_dur, dtype=np.int32)
    conv   = np.convolve(above.astype(np.int32), kernel, mode='valid')
    hits   = np.where(conv == min_dur)[0]
    return int(hits[0]) if len(hits) > 0 else None


def _make_saccade_template(sr, width_ms=150.0):
    """
    Normalised RIGHT-saccade template: half-Hann bump (positive).
    Represents the typical shape of a filtered saccade displacement
    peak at the given timescale.
    """
    n   = max(3, int(width_ms / 1000 * sr))
    t   = np.linspace(0, np.pi, n)
    tpl = np.sin(t)
    return tpl / np.sqrt(np.sum(tpl ** 2))


# ── Shared runner ──────────────────────────────────────────────────────────────

def _run_detection(prep, trial_fn, fpr_fn):
    """
    Generic trial + FPR runner.

    trial_fn(window, sr, baseline_sigma)
        → (predicted: str, latency_ms: float, amplitude_uv: float)

    fpr_fn(seg, sr, baseline_sigma) → bool
    """
    diff_raw       = prep['diff_uv']
    filter_fn      = prep.get('filter_fn', None)
    ev_s, ev_l, sr = prep['ev_s'], prep['ev_l'], prep['sr']
    baseline_sigma = prep.get('baseline_sigma', 1.0)
    n_samps        = diff_raw.size

    det_pre_n  = int(DET_PRE_S * sr)
    det_post_n = int(DET_POST_S * sr)

    trials = []
    for s, label in zip(ev_s, ev_l):
        if label not in ('LEFT', 'RIGHT'):
            continue
        seg, offset = _filter_window(diff_raw, s, n_samps, sr, filter_fn)
        d0 = min(offset + det_pre_n,  len(seg))
        d1 = min(offset + det_post_n, len(seg))
        window = seg[d0:d1]
        if window.size == 0:
            continue
        predicted, latency_ms, amplitude_uv = trial_fn(window, sr, baseline_sigma)
        trials.append({
            'label':        label,
            'predicted':    predicted,
            'correct':      predicted == label,
            'latency_ms':   latency_ms,
            'amplitude_uv': amplitude_uv,
        })

    skip_n = int(FPR_SKIP_S * sr)
    rest_fires = []
    for i, (s, label) in enumerate(zip(ev_s, ev_l)):
        if label != 'REST':
            continue
        end       = int(ev_s[i + 1]) if i + 1 < len(ev_s) else n_samps
        s_settled = s + skip_n
        if s_settled >= end:
            continue
        fn  = filter_fn if filter_fn is not None else _apply_filter
        seg = fn(diff_raw[s_settled:end].copy(), sr)
        if seg.size == 0:
            continue
        rest_fires.append(fpr_fn(seg, sr, baseline_sigma))

    n         = len(trials)
    n_correct = sum(t['correct'] for t in trials)
    lats      = [t['latency_ms'] for t in trials]

    return {
        'subject':    prep['subject'],
        'acc':        n_correct / n if n else 0.0,
        'n':          n,
        'lat_med_ms': float(np.median(lats)) if lats else None,
        'fpr':        float(np.mean(rest_fires)) if rest_fires else 0.0,
        'trials':     trials,
    }


# ── Detectors ─────────────────────────────────────────────────────────────────

def peak_sign(prep, fpr_threshold_uv=10.0):
    """
    Sign of the largest-amplitude sample in [DET_PRE_S, DET_POST_S] post-cue.

    Always fires; no per-trial threshold. Latency = time of peak sample.
    FPR: fraction of settled REST windows where |peak| > fpr_threshold_uv.
    """
    det_pre_n = int(DET_PRE_S * prep['sr'])

    def _trial(window, sr, sigma):
        pk  = int(np.argmax(np.abs(window)))
        val = float(window[pk])
        return ('RIGHT' if val > 0 else 'LEFT'), (det_pre_n + pk) / sr * 1000, abs(val)

    def _fpr(seg, sr, sigma):
        return float(np.max(np.abs(seg))) > fpr_threshold_uv

    return _run_detection(prep, _trial, _fpr)


def first_crossing(prep, sigma_threshold=6.0, fpr_threshold_sigma=6.0):
    """
    First sample in [DET_PRE_S, DET_POST_S] where |signal| > sigma_threshold × σ.

    Direction = sign at crossing point. Falls back to peak_sign if no crossing.
    Threshold adapts to each subject's baseline noise (σ from PrepResult).
    Engbert & Kliegl (2003) use λ=6 on velocity; applied here to displacement.

    FPR: any sample in settled REST window exceeds fpr_threshold_sigma × σ.
    """
    det_pre_n = int(DET_PRE_S * prep['sr'])

    def _trial(window, sr, sigma):
        thr  = sigma_threshold * sigma
        idxs = np.where(np.abs(window) > thr)[0]
        if len(idxs) == 0:
            pk  = int(np.argmax(np.abs(window)))
            val = float(window[pk])
            return ('RIGHT' if val > 0 else 'LEFT'), (det_pre_n + pk) / sr * 1000, abs(val)
        idx = int(idxs[0])
        val = float(window[idx])
        return ('RIGHT' if val > 0 else 'LEFT'), (det_pre_n + idx) / sr * 1000, abs(val)

    def _fpr(seg, sr, sigma):
        return bool(np.any(np.abs(seg) > fpr_threshold_sigma * sigma))

    return _run_detection(prep, _trial, _fpr)


def sustained_crossing(prep, sigma_threshold=6.0, min_dur_ms=12.0,
                        fpr_threshold_sigma=6.0):
    """
    First crossing of sigma_threshold × σ sustained for at least min_dur_ms.

    Adds a persistence gate on top of first_crossing. Single-sample spikes
    (EMG, movement artifact) are ignored; only deflections lasting ≥ min_dur_ms
    trigger detection. Engbert & Kliegl use 12 ms (3 samples at 250 Hz) as the
    minimum microsaccade duration; same value works well for voluntary saccades.

    Falls back to peak_sign if no sustained crossing found.
    """
    det_pre_n = int(DET_PRE_S * prep['sr'])

    def _trial(window, sr, sigma):
        thr     = sigma_threshold * sigma
        min_dur = max(1, int(min_dur_ms / 1000 * sr))
        above   = np.abs(window) > thr
        onset   = _first_sustained(above, min_dur)
        if onset is None:
            pk  = int(np.argmax(np.abs(window)))
            val = float(window[pk])
            return ('RIGHT' if val > 0 else 'LEFT'), (det_pre_n + pk) / sr * 1000, abs(val)
        val = float(window[onset])
        return ('RIGHT' if val > 0 else 'LEFT'), (det_pre_n + onset) / sr * 1000, abs(val)

    def _fpr(seg, sr, sigma):
        thr     = fpr_threshold_sigma * sigma
        min_dur = max(1, int(min_dur_ms / 1000 * sr))
        return _first_sustained(np.abs(seg) > thr, min_dur) is not None

    return _run_detection(prep, _trial, _fpr)


def mean_sign(prep, fpr_threshold_uv=5.0):
    """
    Sign of the mean signal over the full [DET_PRE_S, DET_POST_S] window.

    The most noise-robust direction classifier: random noise averages toward
    zero, while a real saccade maintains a consistent sign for most of the
    window. No per-trial threshold; always fires. Latency is fixed at
    DET_POST_S (500 ms) — the classifier needs the full window to decide.

    FPR: max of 500 ms sliding-window means over settled REST segment exceeds
    fpr_threshold_uv.
    """
    window_n = int(DET_POST_S * prep['sr'])

    def _trial(window, sr, sigma):
        val = float(np.mean(window))
        return ('RIGHT' if val > 0 else 'LEFT'), DET_POST_S * 1000, abs(val)

    def _fpr(seg, sr, sigma):
        n = max(1, window_n)
        if seg.size < n:
            return abs(float(np.mean(seg))) > fpr_threshold_uv
        kernel = np.ones(n) / n
        means  = np.convolve(seg, kernel, mode='valid')
        return float(np.max(np.abs(means))) > fpr_threshold_uv

    return _run_detection(prep, _trial, _fpr)


def template_corr(prep, template_width_ms=150.0, fpr_threshold_uv=5.0):
    """
    Matched-filter detection via cross-correlation with a canonical saccade shape.

    Template: normalised half-Hann bump (width = template_width_ms) representing
    a typical filtered saccade displacement peak. Cross-correlation of the
    detection window with the template peaks at the most saccade-like moment.
    Direction = sign of peak correlation. Latency = peak correlation time
    (offset by half-template to align with saccade onset rather than centre).

    FPR: max |correlation| over settled REST window exceeds fpr_threshold_uv.
    The threshold is in µV since the unit-energy template preserves signal scale.
    """
    sr  = prep['sr']
    tpl = _make_saccade_template(sr, template_width_ms)
    det_pre_n = int(DET_PRE_S * sr)
    half_tpl  = len(tpl) // 2

    def _trial(window, sr, sigma):
        if window.size < len(tpl):
            pk  = int(np.argmax(np.abs(window)))
            val = float(window[pk])
            return ('RIGHT' if val > 0 else 'LEFT'), (det_pre_n + pk) / sr * 1000, abs(val)
        corr = correlate(window, tpl, mode='valid')
        pk   = int(np.argmax(np.abs(corr)))
        val  = float(corr[pk])
        lat  = (det_pre_n + pk + half_tpl) / sr * 1000
        return ('RIGHT' if val > 0 else 'LEFT'), lat, abs(val)

    def _fpr(seg, sr, sigma):
        if seg.size < len(tpl):
            return False
        corr = correlate(seg, tpl, mode='valid')
        return float(np.max(np.abs(corr))) > fpr_threshold_uv

    return _run_detection(prep, _trial, _fpr)
