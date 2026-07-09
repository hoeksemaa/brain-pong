"""
Shared EOG detection core for BrainPong.

Single source of truth for the realtime EOG glance-pair detector. Both the
single-player path (`eog_state`) and the two-player path (`eog_state_p2`) in
pong_game_brainflow.py drive the *identical* functions here — they differ only
in which electrode slots feed them and which state dict they carry. Any future
BrainPong variant (N-player, calibration tools, replay) should import from here
rather than re-inline detection logic.

Everything in this module is pure / deterministic and hardware-free:
  - no `board`, no Dash, no CLI, no global mutable singletons beyond the
    caller-supplied state dict.
  - it imports only numpy + brainflow.data_filter (the DSP half of BrainFlow),
    NOT brainflow.board_shim, so it loads without the Cerelog board fork and is
    cheap to import inside a test process.

DATA INTEGRITY: filters operate on copies. BrainFlow's `DataFilter.*` functions
mutate their input in place, so `_eog_filter` first makes a contiguous float64
copy and filters that. Callers that pass a view into a recording array are safe.

Detection model (glance-pair):
  CALIBRATING → (collect EOG_BASELINE_S of signal, set σ) → IDLE
  IDLE        → sustained crossing in dir A           → ARMED(first_dir=A)
  ARMED       → opposite crossing B within GLANCE_WINDOW_S (and after
                ARMED_MIN_WAIT_S)                      → FIRE cmd=A → REFRACTORY
  ARMED       → no opposite within GLANCE_WINDOW_S     → IDLE (timeout)
  REFRACTORY  → after REFRACTORY_S                     → IDLE

The pair requirement (look one way, then the other) is a deliberate debounce:
a single involuntary saccade should not move the paddle. See the module-level
NOTE on oscillating noise — the pair logic has a known blind spot there.
"""

import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# ── Detection constants (the algorithm's tunable surface) ───────────────────────
# These are the *defaults*. Where a function exposes a matching keyword arg, the
# default is sourced from here so production behaviour is defined in one place
# while tests can pin behaviour at explicit values independent of these defaults.
EOG_LPF_HZ       = 50.0    # low-pass corner. At 50 Hz the 58–62 Hz notch sits above the cutoff (60 Hz
                           # mains already attenuated by the LPF); the 48–52 Hz notch straddles the corner.
EOG_HPF_HZ       = 0.1     # high-pass corner. 0.1 Hz (τ≈1.6 s) keeps more low-frequency signal than
                           # 0.5 Hz but its recovery tail does NOT fully settle inside the ~0.4 s per-poll
                           # window; acceptable because the VELOCITY detector rejects the slow tail (a
                           # slow tail is low-velocity). Offline whole-window filtering favoured 0.1 Hz on hit-rate.
NOTCH_BANDS      = ((48.0, 52.0), (58.0, 62.0))
EOG_SIGMA_THR    = 4.0     # crossing threshold in units of baseline σ. Lower than the
                           # original 5σ: the glance-PAIR debounce rejects stray singles,
                           # so a more sensitive primitive misses fewer real glances.
EOG_MIN_DUR_MS   = 12.0    # a crossing must persist this long (kills single spikes)
GLANCE_WINDOW_S  = 0.5     # max time between the two glances of a pair
ARMED_MIN_WAIT_S = 0.05    # min time before the opposite glance counts
REFRACTORY_S     = 0.8     # dead time after a fired command
EOG_BASELINE_S   = 5.0     # baseline collected before σ is fixed


# ── State factory ───────────────────────────────────────────────────────────────

def _make_eog_state():
    """Fresh per-player EOG state dict. ch_L/ch_R/sr are filled in at board setup;
    the runtime knobs (sigma_thr, glance_window_s, lpf_hz, hpf_hz) default here and
    are updated live from the in-game browser sliders."""
    return {
        'ch_L': None, 'ch_R': None, 'sr': None,
        'sm': 'CALIBRATING',
        'baseline_acc': [],
        'baseline_sigma': None,
        'first_dir': None,
        'arm_time': None,
        'last_cmd_time': 0.0,
        'cmd_seq': 0,
        # ── tunable config (set at board setup; survive recalibration) ──────────
        'sigma_thr': EOG_SIGMA_THR,          # ×; a glance must exceed this MULTIPLE of baseline σ
        'glance_window_s': GLANCE_WINDOW_S,  # s; max gap between the two glances of a pair
        'lpf_hz': EOG_LPF_HZ,                # Hz; low-pass corner of the filter chain
        'hpf_hz': EOG_HPF_HZ,                # Hz; high-pass corner of the filter chain
    }


def _reset_eog_st(eog_st):
    """Return a player's state to fresh CALIBRATING (keeps ch_L/ch_R/sr/cmd_seq
    and the config knobs sigma_thr/glance_window_s)."""
    eog_st['sm']             = 'CALIBRATING'
    eog_st['baseline_acc']   = []
    eog_st['baseline_sigma'] = None
    eog_st['first_dir']      = None
    eog_st['arm_time']       = None
    eog_st['last_cmd_time']  = 0.0


# ── Differential + filter (pure DSP) ────────────────────────────────────────────

def eog_diff(data, ch_R, ch_L):
    """Horizontal EOG differential (R − L) in µV from a channel-major window.

    `data` is (n_channels, n_samples) as BrainFlow returns it. Sign convention:
    rightward gaze is positive. Swapping ch_R/ch_L flips every downstream
    decision (see the John electrode-swap note), so this is the one place the
    polarity contract is defined. Returns a fresh 1-D array.
    """
    return (data[ch_R] - data[ch_L]).astype(np.float64) * 1e6


def _eog_filter(x_uv, sr, lpf_hz=EOG_LPF_HZ, hpf_hz=EOG_HPF_HZ,
                notch_bands=NOTCH_BANDS):
    """0.5–100 Hz causal IIR chain — mirrors segment_diff_filter preprocessing.

    detrend(constant) → lowpass → bandstop notches → highpass, on a private
    copy. Arrays shorter than 20 samples are returned (as float64) unfiltered —
    the IIR has no room to settle. Cutoffs are keyword args so tests can probe
    the passband/stopband without depending on the production defaults.
    """
    y = np.ascontiguousarray(x_uv.astype(np.float64))
    if y.size < 20:
        return y
    DataFilter.detrend(y, DetrendOperations.CONSTANT.value)
    DataFilter.perform_lowpass(y, sr, lpf_hz, 4, FilterTypes.BUTTERWORTH.value, 0)
    for lo, hi in notch_bands:
        DataFilter.perform_bandstop(y, sr, lo, hi, 3, FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_highpass(y, sr, hpf_hz, 4, FilterTypes.BUTTERWORTH.value, 0)
    return y


def _eog_velocity(x_uv, sr):
    """Engbert & Kliegl (2003) 5-point smoothed velocity of a filtered signal.

    v[n] = (x[n+2] + x[n+1] − x[n−1] − x[n−2]) / (6·dt),  dt = 1/sr   (µV/s).
    Differentiates *and* lightly low-pass smooths (the 5-tap kernel). Velocity is
    the field-standard saccade statistic: differentiation is a high-pass operator,
    so slow drift and the high-pass filter's slow recovery tail (low slope) are
    attenuated while a saccade's steep edge is amplified — a tail can match a
    saccade in amplitude but never in velocity. Its sign is the direction of gaze
    change (rightward = +), so it slots straight into _sustained_crossing in place
    of amplitude with the direction contract unchanged. The 2-sample stencil is an
    8 ms group delay (negligible vs the ≤500 ms budget); edges are replicated so
    the 2 endpoint samples get a conservative one-sided estimate. Returns float64,
    same length as the input.
    """
    y = np.ascontiguousarray(x_uv.astype(np.float64))
    n = y.size
    if n < 5:
        return np.zeros(n, dtype=np.float64)
    yp = np.pad(y, 2, mode='edge')
    dt = 1.0 / sr
    return (yp[4:] + yp[3:-1] - yp[1:-3] - yp[:-4]) / (6.0 * dt)


# ── Crossing detector (pure) ─────────────────────────────────────────────────────

def _sustained_crossing(signal, sigma, sr, sigma_thr=EOG_SIGMA_THR,
                        min_dur_ms=EOG_MIN_DUR_MS):
    """Direction of the first sustained threshold crossing, else None.

    Returns 'RIGHT'/'LEFT' (sign at onset) if |signal| exceeds sigma_thr×σ for a
    run of at least min_dur_ms; None otherwise. The persistence gate is what
    rejects single-sample EMG spikes. NOTE: it does *not* reject oscillation —
    a sustained run of either sign satisfies it, so an oscillating artifact reads
    as a stream of alternating crossings.
    """
    if sigma < 1e-9 or signal.size == 0:
        return None
    thr     = sigma_thr * sigma
    min_dur = max(1, int(min_dur_ms / 1000 * sr))
    above   = np.abs(signal) > thr
    conv    = np.convolve(above.astype(np.int32), np.ones(min_dur, dtype=np.int32), mode='valid')
    hits    = np.where(conv == min_dur)[0]
    if len(hits) == 0:
        return None
    onset = int(hits[0])
    return 'RIGHT' if signal[onset] > 0 else 'LEFT'


# ── Glance-pair state machine (deterministic given `now`) ────────────────────────

def _run_eog_sm(eog_st, new_sig, now, label='EOG'):
    """Advance one EOG state machine tick. Returns a command dict or None.

    `now` is injected (wallclock seconds) rather than read from time.time(), so
    the machine is fully deterministic and testable: drive it with crafted
    `new_sig` windows and explicit timestamps.
    """
    if eog_st['sm'] == 'CALIBRATING':
        eog_st['baseline_acc'].append(new_sig.copy())
        total = np.concatenate(eog_st['baseline_acc'])
        if total.size >= int(EOG_BASELINE_S * eog_st['sr']):
            # Robust (MAD-based) noise scale: median-based, so an involuntary
            # saccade or blink during the eyes-forward calibration can't inflate σ
            # and desensitise the detector (Engbert & Kliegl use a median velocity
            # estimator for exactly this). 1.4826·MAD is a consistent estimator of
            # the Gaussian σ; fall back to std if the baseline is degenerate (flat).
            med   = float(np.median(total))
            sigma = 1.4826 * float(np.median(np.abs(total - med)))
            if sigma < 1e-9:
                sigma = float(np.std(total))
            eog_st['baseline_sigma'] = sigma or 1e-6
            eog_st['sm'] = 'IDLE'
            print(f"[{label}] baseline σ = {eog_st['baseline_sigma']:.2f} — ready")
        return None

    if eog_st['sm'] == 'REFRACTORY':
        if now - eog_st['last_cmd_time'] > REFRACTORY_S:
            eog_st['sm'] = 'IDLE'
        return None

    sigma    = eog_st['baseline_sigma']
    crossing = _sustained_crossing(new_sig, sigma, eog_st['sr'],
                                   sigma_thr=eog_st.get('sigma_thr', EOG_SIGMA_THR))

    if eog_st['sm'] == 'IDLE':
        if crossing is not None:
            eog_st['sm']        = 'ARMED'
            eog_st['first_dir'] = crossing
            eog_st['arm_time']  = now

    elif eog_st['sm'] == 'ARMED':
        if now - eog_st['arm_time'] > eog_st.get('glance_window_s', GLANCE_WINDOW_S):
            eog_st['sm']        = 'IDLE'
            eog_st['first_dir'] = None
        elif now - eog_st['arm_time'] > ARMED_MIN_WAIT_S and crossing is not None:
            opposite = {'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
            if crossing == opposite.get(eog_st['first_dir']):
                cmd = eog_st['first_dir']
                eog_st['cmd_seq']      += 1
                eog_st['last_cmd_time'] = now
                eog_st['sm']            = 'REFRACTORY'
                eog_st['first_dir']     = None
                print(f"[{label}] command={cmd}  seq={eog_st['cmd_seq']}")
                return {'command': cmd, 'seq': eog_st['cmd_seq']}
    return None
