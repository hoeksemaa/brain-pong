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
EOG_LPF_HZ       = 100.0
EOG_HPF_HZ       = 0.5
NOTCH_BANDS      = ((48.0, 52.0), (58.0, 62.0))
EOG_SIGMA_THR    = 5.0     # crossing threshold in units of baseline σ
EOG_MIN_DUR_MS   = 12.0    # a crossing must persist this long (kills single spikes)
GLANCE_WINDOW_S  = 0.7     # max time between the two glances of a pair
ARMED_MIN_WAIT_S = 0.05    # min time before the opposite glance counts
REFRACTORY_S     = 0.8     # dead time after a fired command
EOG_BASELINE_S   = 5.0     # baseline collected before σ is fixed


# ── State factory ───────────────────────────────────────────────────────────────

def _make_eog_state():
    """Fresh per-player EOG state dict. ch_L/ch_R/sr are filled in at board setup."""
    return {
        'ch_L': None, 'ch_R': None, 'sr': None,
        'sm': 'CALIBRATING',
        'baseline_acc': [],
        'baseline_sigma': None,
        'first_dir': None,
        'arm_time': None,
        'last_cmd_time': 0.0,
        'cmd_seq': 0,
    }


def _reset_eog_st(eog_st):
    """Return a player's state to fresh CALIBRATING (keeps ch_L/ch_R/sr/cmd_seq)."""
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
            eog_st['baseline_sigma'] = float(np.std(total))
            eog_st['sm'] = 'IDLE'
            print(f"[{label}] baseline σ = {eog_st['baseline_sigma']:.2f} µV — ready")
        return None

    if eog_st['sm'] == 'REFRACTORY':
        if now - eog_st['last_cmd_time'] > REFRACTORY_S:
            eog_st['sm'] = 'IDLE'
        return None

    sigma    = eog_st['baseline_sigma']
    crossing = _sustained_crossing(new_sig, sigma, eog_st['sr'])

    if eog_st['sm'] == 'IDLE':
        if crossing is not None:
            eog_st['sm']        = 'ARMED'
            eog_st['first_dir'] = crossing
            eog_st['arm_time']  = now

    elif eog_st['sm'] == 'ARMED':
        if now - eog_st['arm_time'] > GLANCE_WINDOW_S:
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
