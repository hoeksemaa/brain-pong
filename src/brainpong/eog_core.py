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
EOG_LPF_HZ       = 30.0    # low-pass corner. Detection runs on VELOCITY (a derivative), which amplifies
                           # high frequencies, so 30–50 Hz broadband/EMG noise dominates the velocity noise
                           # floor. Dropping the corner 50→30 Hz cuts ~1/3 off that floor with no hit-rate
                           # cost on labeled saccades (their velocity energy sits <30 Hz). Both notch bands
                           # (48–52, 58–62) now fall entirely above the corner — redundant but harmless.
                           # MUST move together with the raised σ below: a lower floor lowers the ABSOLUTE
                           # threshold (σ·floor), so LPF-down alone fires MORE, not less.
EOG_HPF_HZ       = 0.1     # high-pass corner. 0.1 Hz (τ≈1.6 s) keeps more low-frequency signal than
                           # 0.5 Hz but its recovery tail does NOT fully settle inside the ~0.4 s per-poll
                           # window; acceptable because the VELOCITY detector rejects the slow tail (a
                           # slow tail is low-velocity). Offline whole-window filtering favoured 0.1 Hz on hit-rate.
NOTCH_BANDS      = ((48.0, 52.0), (58.0, 62.0))
EOG_SIGMA_THR    = 6.0     # crossing threshold in units of baseline σ. Raised 4→6 to spend the lower
                           # LP30 noise floor (above) on FEWER phantom fires rather than more sensitivity.
                           # At LP30 the floor is ~27% lower, so σ≈5.5 reproduces the old LP50/σ4 absolute
                           # bar and σ=6 tightens it ~9%; labeled-cue hit-rate stays ~0.89 while resting
                           # phantom pair-commands drop. Drop toward 5.5 (live slider) if a noisy rig starts
                           # missing real glances — an under-firing rig wants a lower multiplier, not this.
EOG_MIN_DUR_MS   = 12.0    # a crossing must persist this long (kills single spikes)
GLANCE_WINDOW_S  = 0.5     # max time between the two glances of a pair
ARMED_MIN_WAIT_S = 0.05    # min time before the opposite glance counts
REFRACTORY_S     = 0.8     # dead time after a fired command
PLAY_SETTLE_S    = 0.7     # detector muted for this long after PLAY begins, so the
                           # filter/velocity startup transient (a large edge artifact
                           # on the first freshly-filtered window) can't phantom-fire
                           # and jitter the paddle in the opening second. Symptom guard;
                           # the root-cause edge-artifact fix lives in the offline eval.
EOG_BASELINE_S   = 5.0     # baseline collected before σ is fixed
MATCHED_TEMPLATE_MS = 120.0  # saccade-velocity template width for the matched-filter detector


# ── State factory ───────────────────────────────────────────────────────────────

def _make_eog_state():
    """Fresh per-player EOG state dict. ch_L/ch_R/sr/mf_template are filled in at
    board setup; the runtime knobs (sigma_thr, glance_window_s, lpf_hz, hpf_hz,
    detector) default here and are updated live from the in-game browser controls."""
    return {
        'ch_L': None, 'ch_R': None, 'sr': None,
        'sm': 'CALIBRATING',
        'baseline_acc': [],
        'baseline_sigma': None,
        'first_dir': None,
        'arm_time': None,
        'last_cmd_time': 0.0,
        'cmd_seq': 0,
        'settle_until': 0.0,                 # wallclock until which fires are muted (startup guard)
        # ── tunable config (set at board setup; survive recalibration) ──────────
        'sigma_thr': EOG_SIGMA_THR,          # ×; a glance must exceed this MULTIPLE of baseline σ
        'glance_window_s': GLANCE_WINDOW_S,  # s; max gap between the two glances of a pair
        'lpf_hz': EOG_LPF_HZ,                # Hz; low-pass corner of the filter chain
        'hpf_hz': EOG_HPF_HZ,                # Hz; high-pass corner of the filter chain
        'detector': 'velocity',             # 'velocity' | 'matched' — detection method (UI toggle)
        'mf_template': None,                 # saccade template for the matched filter (set at board setup)
    }


def _reset_eog_st(eog_st):
    """Return a player's state to fresh CALIBRATING for a new game (keeps ch_L/ch_R/sr
    and the config knobs sigma_thr/glance_window_s/lpf_hz/hpf_hz/detector/mf_template).

    cmd_seq resets to 0 so each game's command sequence restarts clean; the browser
    command stores are cleared to a matching seq 0 on New Game
    (clear_bci_stores_on_new_game in the game script), so no command from a previous
    game can survive into the new game's first PLAYING tick."""
    eog_st['sm']             = 'CALIBRATING'
    eog_st['baseline_acc']   = []
    eog_st['baseline_sigma'] = None
    eog_st['first_dir']      = None
    eog_st['arm_time']       = None
    eog_st['last_cmd_time']  = 0.0
    eog_st['settle_until']   = 0.0
    eog_st['cmd_seq']        = 0


def begin_play_settle(eog_st, now, settle_s=PLAY_SETTLE_S):
    """Call the instant the game enters PLAY: drop the detector to IDLE and mute
    it for ``settle_s`` so the filter startup transient can't phantom-fire and
    jitter the paddle. Idempotent — safe to call every tick while PLAYING; it only
    re-arms the window on the transition (guarded by the caller)."""
    eog_st['sm']           = 'IDLE'
    eog_st['first_dir']    = None
    eog_st['settle_until'] = now + settle_s


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


# ── Matched filter (pure) ────────────────────────────────────────────────────────

def _make_velocity_template(sr, width_ms=MATCHED_TEMPLATE_MS):
    """Unit-norm saccade-velocity template for the matched filter.

    A saccade is a UNIPOLAR velocity pulse, so the template is a single positive
    Hann bump ~width_ms wide. The sign (LEFT vs RIGHT) is carried by the signal,
    not the template, so one template matches both directions. Unit-norm keeps the
    response scaled sanely; the σ-relative threshold absorbs the absolute scale.
    """
    n = max(3, int(round(width_ms / 1000.0 * sr)))
    t = np.hanning(n).astype(np.float64)
    return t / (np.linalg.norm(t) + 1e-12)


def _matched_filter(vel, template):
    """Matched-filter response: velocity cross-correlated with the saccade template
    (zero-lag-centred, same length as `vel`). Peaks where the velocity matches the
    saccade shape; the peak's SIGN is the gaze-change direction, so it drops
    straight into _sustained_crossing in place of raw velocity. Integrating over the
    template width lifts a coherent saccade above incoherent noise (~√len gain);
    the shape-match also rejects transients that don't look like a saccade.
    """
    v = np.ascontiguousarray(vel, dtype=np.float64)
    if v.size < template.size:
        return np.zeros_like(v)
    return np.correlate(v, template, mode='same')


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

    # Startup-transient guard: for the first PLAY_SETTLE_S of play, stay muted and
    # IDLE so the filter/velocity edge artifact on the opening windows can't fire.
    if now < eog_st.get('settle_until', 0.0):
        if eog_st['sm'] == 'ARMED':
            eog_st['sm']        = 'IDLE'
            eog_st['first_dir'] = None
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


# ── Pipeline self-description (for recording notes) ───────────────────────────
# UPDATE pipeline_description() AND bump PIPELINE_VERSION whenever the
# preprocessing or detection METHOD changes — i.e. the *shape* of the pipeline,
# not the tunable cutoff/threshold VALUES (those are stored per-recording in
# their own fields). The parameterised bits (notch bands) are derived from the
# constants above so they can't drift; the method prose is hand-maintained and
# names the exact functions, so the CODE stays the source of truth for a reader.
PIPELINE_VERSION = "pipeline-v2"


def pipeline_description(detector='velocity'):
    """One-line English description of the current live preprocessing + detection
    METHOD (for the requested `detector`), embedded in each recording's notes so the
    pipeline is reconstructable from the file alone. Omits the numeric cutoffs/
    thresholds (stored as the lpf_hz/hpf_hz/sigma_thr/glance_window_s fields) — it
    describes HOW, not WITH-WHAT."""
    notches = ', '.join(f"{lo:g}-{hi:g} Hz" for lo, hi in NOTCH_BANDS)
    pre = (
        "PREPROCESS: HEOG differential (R-L, µV) -> causal Butterworth chain "
        f"(constant detrend -> 4th-order low-pass -> bandstop notches [{notches}] -> "
        "4th-order high-pass) -> Engbert & Kliegl (2003) 5-point smoothed velocity. "
    )
    if detector == 'matched':
        det = (
            "DETECT (matched-filter): the velocity is cross-correlated with a "
            f"~{MATCHED_TEMPLATE_MS:g} ms unit-norm Hann saccade template (matched "
            "filter); the per-player glance-PAIR state machine then runs on that "
            "response -- a crossing of (sigma_thr x robust MAD baseline sigma of the "
            "response) sustained >= min-duration arms a direction, the OPPOSITE "
            "crossing within glance_window fires, then a refractory dead-time. "
        )
        code = "_eog_filter / _eog_velocity / _matched_filter / _sustained_crossing / _run_eog_sm"
    else:
        det = (
            "DETECT (velocity): per-player glance-PAIR state machine on the velocity "
            "signal -- a crossing of (sigma_thr x robust MAD baseline sigma) sustained "
            ">= min-duration arms a direction, the OPPOSITE crossing within "
            "glance_window fires, then a refractory dead-time. "
        )
        code = "_eog_filter / _eog_velocity / _sustained_crossing / _run_eog_sm"
    tail = ("Cutoff/threshold values are in the lpf_hz/hpf_hz/sigma_thr/"
            f"glance_window_s fields. Code: eog_core.{code}. [{PIPELINE_VERSION}]")
    return pre + det + tail
