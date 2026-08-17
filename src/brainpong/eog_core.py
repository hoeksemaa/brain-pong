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
  IDLE        → CONFIRMED crossing in dir A           → ARMED(first_dir=A)
  ARMED       → opposite crossing B within GLANCE_WINDOW_S (and after
                ARMED_MIN_WAIT_S)                      → FIRE cmd=A → REFRACTORY
  ARMED       → no opposite within GLANCE_WINDOW_S     → IDLE (timeout)
  REFRACTORY  → after REFRACTORY_S                     → IDLE

A crossing is CONFIRMED, not immediate: a permissive gate nominates a candidate,
then its same-sign peak is accumulated for defer_ms before being tested against
the real bar — see _deferred_crossing for why the poll boundary makes this
necessary, and for the opposite-sign trap it has to dodge.

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
EOG_DEFER_MS     = 0.0     # OFF BY DEFAULT — the deferral does not survive the state machine.
                           # The idea: hold a candidate crossing this long, then decide on the
                           # PEAK it reached rather than on the first sample over the bar,
                           # because the 100 ms poll truncates a 100-150 ms saccade pulse. At
                           # CANDIDATE level that is worth +6 to +8 points of glance hit-rate
                           # at matched false-alarm budgets. At COMMAND level, through the real
                           # _run_eog_sm on 19 cued recordings, it LOSES: hit-rate 0.433 -> 0.378
                           # and phantom commands 7.79 -> 8.79/min at a fixed 6σ, and still
                           # 0.433 -> 0.361 after raising σ_thr to re-match the phantom rate
                           # (1 win / 14 losses / 4 ties, Wilcoxon p=0.0053).
                           # Why it fails: accumulating a peak over 200 ms (50 samples) rather
                           # than one 25-sample poll raises the NOISE exceedance probability too,
                           # and holding a 2σ candidate blocks the SM from nominating the real
                           # crossing behind it. The candidate-level gain never reaches a command.
                           # Kept, wired and slider-exposed so the idea can be re-tested on a
                           # live rig (in-game signal differs from the cued corpus) — but any
                           # value > 0 is an EXPERIMENT, not the shipping detector.
EOG_CAND_SIGMA_THR = 2.0   # permissive gate that OPENS the deferral window. Deliberately far
                           # below sigma_thr: it only nominates a candidate: the real bar
                           # (sigma_thr x sigma) is applied to the peak when the window closes.
                           # Clamped to sigma_thr, so it can never be the stricter of the two.
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
        # ── deferred-confirm candidate (see _deferred_crossing); None = nothing pending ──
        'pend_dir': None, 'pend_sign': 0.0, 'pend_peak': 0.0, 'pend_until': None,
        'last_cmd_time': 0.0,
        'cmd_seq': 0,
        'settle_until': 0.0,                 # wallclock until which fires are muted (startup guard)
        'fire_log': [],                      # recent detected motions [{'t': wallclock_s, 'cmd': 'LEFT'|'RIGHT'}]
                                             # every sustained crossing the SM counts as a directional
                                             # motion (arming saccade + return saccade), for the live-feed
                                             # detection overlay (display only)
        # ── tunable config (set at board setup; survive recalibration) ──────────
        'sigma_thr': EOG_SIGMA_THR,          # ×; a glance must exceed this MULTIPLE of baseline σ
        'defer_ms': EOG_DEFER_MS,            # ms; hold a candidate this long, decide on its peak (0 = off)
        'cand_sigma_thr': EOG_CAND_SIGMA_THR,  # ×; permissive gate that opens the deferral window
        'glance_window_s': GLANCE_WINDOW_S,  # s; max gap between the two glances of a pair
        'lpf_hz': EOG_LPF_HZ,                # Hz; low-pass corner of the filter chain
        'hpf_hz': EOG_HPF_HZ,                # Hz; high-pass corner of the filter chain
        'detector': 'velocity',              # always 'velocity' live — the matched-filter
                                             # toggle is retired. Kept because every recording
                                             # writes this to its npz and pipeline.replay reads
                                             # it back (archived files may say 'matched').
        'mf_template': None,                 # saccade template, set by pipeline.replay only
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
    eog_st['fire_log']       = []
    _clear_pending(eog_st)


def begin_play_settle(eog_st, now, settle_s=PLAY_SETTLE_S):
    """Call the instant the game enters PLAY: drop the detector to IDLE and mute
    it for ``settle_s`` so the filter startup transient can't phantom-fire and
    jitter the paddle. Idempotent — safe to call every tick while PLAYING; it only
    re-arms the window on the transition (guarded by the caller).

    REFUSES TO LEAVE CALIBRATION WITHOUT A THRESHOLD. The calibration countdown is
    WALLCLOCK but σ locks on SAMPLES accumulated, so the two can disagree: a throttled
    BCI tick (backgrounded browser tab, slow callback) or an interrupted calibration
    can start play with ``baseline_sigma`` still None. Forcing 'IDLE' there sends the
    next poll into _sustained_crossing with sigma=None, which raises TypeError on
    ``sigma < 1e-9`` — and in the live Dash app that kills the detector callback for
    the WHOLE game, silently. So when σ is missing, stay CALIBRATING: the baseline
    keeps accumulating and _run_eog_sm promotes itself to IDLE the moment it holds
    EOG_BASELINE_S of signal. The mute window and the overlay reset happen either way.
    """
    if eog_st.get('baseline_sigma') is None:
        eog_st['sm'] = 'CALIBRATING'       # σ not locked yet — finish calibrating first
        print("[EOG] play began before σ locked — staying in calibration")
    else:
        eog_st['sm'] = 'IDLE'
    eog_st['first_dir']    = None
    eog_st['settle_until'] = now + settle_s
    eog_st['fire_log']     = []            # drop calibration-tail fires from the overlay
    _clear_pending(eog_st)                 # no calibration-era candidate may confirm into play


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


# ── Matched filter (pure) — REPLAY ONLY, retired from the live game ──────────────
#
# The live game always detects on velocity; the in-game Velocity/Matched toggle is
# gone (see scripts/pong_game_brainflow.py). These two functions stay because 63 of
# the 144 recordings in data/eog — the whole 2026-07-13 tournament among them — were
# PLAYED in matched mode and store detector='matched' in their npz. pipeline.replay
# reads that field, so deleting these would silently change how 44% of the corpus
# replays. Do not wire them back into the live path: _matched_filter zero-pads the
# future (np.correlate mode='same'), which costs up to 33% of the response in the
# newest ~40 ms — exactly the samples the live loop reads.
#
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

def _first_sustained_run(mask, min_dur):
    """Start index of the first run of >= min_dur consecutive True in `mask`, else None."""
    if min_dur <= 0 or mask.size < min_dur:
        return None
    conv = np.convolve(mask.astype(np.int32), np.ones(min_dur, dtype=np.int32), mode='valid')
    hits = np.where(conv == min_dur)[0]
    return int(hits[0]) if len(hits) else None


def _sustained_onset(signal, sigma, sr, sigma_thr=EOG_SIGMA_THR,
                     min_dur_ms=EOG_MIN_DUR_MS):
    """(onset_index, direction) of the first sustained threshold crossing, else (None, None).

    Same gate as _sustained_crossing; this variant also hands back WHERE the crossing
    started, which the deferred-confirm path needs to know where to begin measuring
    the saccade's peak.
    """
    if sigma is None or sigma < 1e-9 or signal.size == 0:
        return None, None
    min_dur = max(1, int(min_dur_ms / 1000 * sr))
    onset = _first_sustained_run(np.abs(signal) > sigma_thr * sigma, min_dur)
    if onset is None:
        return None, None
    return onset, ('RIGHT' if signal[onset] > 0 else 'LEFT')


def _sustained_crossing(signal, sigma, sr, sigma_thr=EOG_SIGMA_THR,
                        min_dur_ms=EOG_MIN_DUR_MS):
    """Direction of the first sustained threshold crossing, else None.

    Returns 'RIGHT'/'LEFT' (sign at onset) if |signal| exceeds sigma_thr×σ for a
    run of at least min_dur_ms; None otherwise. The persistence gate is what
    rejects single-sample EMG spikes. NOTE: it does *not* reject oscillation —
    a sustained run of either sign satisfies it, so an oscillating artifact reads
    as a stream of alternating crossings.
    """
    return _sustained_onset(signal, sigma, sr, sigma_thr, min_dur_ms)[1]


def _deferred_crossing(eog_st, signal, sigma, now):
    """Crossing detection that decides on the saccade's PEAK, not on its first sample.

    STATUS: OFF by default (EOG_DEFER_MS = 0). Measured worse at command level — see the
    numbers on that constant. This docstring describes what the code does, not a
    recommendation to enable it.

    WHY IT WAS TRIED. The loop hands the SM one 100 ms poll at a time, but a saccade's
    velocity pulse is 100-150 ms wide, so the poll boundary chops it: whichever fraction
    landed in this poll is the entire evidence _sustained_crossing gets. A real glance
    whose pulse straddles the boundary can fail to reach sigma_thr inside its slice and
    be thrown away. At candidate level, deciding on the peak over the whole event instead
    is worth +6 to +8 points of hit-rate. That gain does not survive the glance-pair SM.

    HOW. Two stages. A permissive gate (cand_sigma_thr, default 2σ) opens a PENDING
    window; the peak is then accumulated across polls and tested against the REAL bar
    (sigma_thr·σ) when the window closes. Noise does not gain from this the way a
    saccade does, because noise has no coherent 100-150 ms pulse to accumulate.

    THE SIGN TRAP. The window must never reach the RETURN saccade, which is the
    opposite sign — measured on 372 cued glances, an opposite-sign crossing lands
    within 200 ms of onset on 20.4% of them (a fast return, or post-saccadic
    overshoot). Taking |peak| blindly over a fixed 200 ms would therefore arm the
    WRONG DIRECTION on one glance in five. So only same-sign samples accumulate, and
    a sustained opposite-sign run CLOSES the window early and forces the decision on
    what the outward pulse alone reached. Early closure is a bonus, not a cost: those
    glances resolve in less than defer_ms.

    Returns 'RIGHT'/'LEFT' at the moment a candidate is CONFIRMED (defer_ms after its
    onset, or sooner if the sign flipped), else None. With defer_ms <= 0 it degrades to
    exactly _sustained_crossing, so legacy replay is bit-identical.
    """
    sr        = eog_st['sr']
    sigma_thr = eog_st.get('sigma_thr', EOG_SIGMA_THR)
    defer_ms  = eog_st.get('defer_ms', EOG_DEFER_MS)
    min_dur   = max(1, int(EOG_MIN_DUR_MS / 1000 * sr))

    if defer_ms <= 0:                       # deferral disabled → today's behaviour
        return _sustained_crossing(signal, sigma, sr, sigma_thr=sigma_thr)

    cand_thr = min(eog_st.get('cand_sigma_thr', EOG_CAND_SIGMA_THR), sigma_thr)

    # ── 1. extend an open candidate with this poll ──────────────────────────────
    if eog_st.get('pend_until') is not None:
        sign  = eog_st['pend_sign']
        same  = signal * sign                       # > 0 while the pulse holds its sign
        close = _first_sustained_run(same < 0, min_dur)   # sustained flip → return saccade
        upto  = close if close is not None else signal.size
        if upto > 0:
            eog_st['pend_peak'] = max(eog_st['pend_peak'], float(np.max(same[:upto])))
        if close is not None or now >= eog_st['pend_until']:
            peak, direction = eog_st['pend_peak'], eog_st['pend_dir']
            _clear_pending(eog_st)
            return direction if peak > sigma_thr * sigma else None
        return None

    # ── 2. no candidate open — look for one at the permissive gate ──────────────
    onset, direction = _sustained_onset(signal, sigma, sr, sigma_thr=cand_thr)
    if onset is None:
        return None
    sign = 1.0 if direction == 'RIGHT' else -1.0
    same = signal[onset:] * sign
    close = _first_sustained_run(same < 0, min_dur)
    upto  = close if close is not None else same.size
    peak  = float(np.max(same[:upto])) if upto > 0 else 0.0
    if close is not None:                   # whole pulse already inside this poll
        return direction if peak > sigma_thr * sigma else None
    eog_st.update({'pend_dir': direction, 'pend_sign': sign, 'pend_peak': peak,
                   'pend_until': now + defer_ms / 1000.0})
    return None


def _clear_pending(eog_st):
    """Drop any half-measured candidate. Called on refractory, settle and reset, so a
    candidate can never survive across a state change and confirm into the wrong era."""
    eog_st['pend_dir']   = None
    eog_st['pend_sign']  = 0.0
    eog_st['pend_peak']  = 0.0
    eog_st['pend_until'] = None


# ── Glance-pair state machine (deterministic given `now`) ────────────────────────

def _log_motion(eog_st, now, direction):
    """Record a single directional motion the detector counted, for the live-feed
    overlay (display only; never read back by the detector). One entry per sustained
    crossing the SM acts on — the arming saccade and the return saccade of a glance
    pair each get one. Capped so it can't grow unbounded if the wave payload isn't
    being pulled."""
    log = eog_st.setdefault('fire_log', [])
    log.append({'t': now, 'cmd': direction})
    if len(log) > 64:
        del log[:-64]


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
        _clear_pending(eog_st)          # nothing measured during dead time may confirm after it
        if now - eog_st['last_cmd_time'] > REFRACTORY_S:
            eog_st['sm'] = 'IDLE'
        return None

    # Startup-transient guard: for the first PLAY_SETTLE_S of play, stay muted and
    # IDLE so the filter/velocity edge artifact on the opening windows can't fire.
    if now < eog_st.get('settle_until', 0.0):
        _clear_pending(eog_st)          # the edge artifact must not survive the mute window
        if eog_st['sm'] == 'ARMED':
            eog_st['sm']        = 'IDLE'
            eog_st['first_dir'] = None
        return None

    sigma = eog_st['baseline_sigma']
    if sigma is None:
        # Invariant backstop: only CALIBRATING may hold a None σ, and that branch
        # returns above. Reaching here means something moved the SM out of calibration
        # without a threshold — recover by calibrating again rather than raising
        # TypeError inside _sustained_crossing. See begin_play_settle for the path
        # that used to do exactly this and take the detector down for a whole game.
        eog_st['sm'] = 'CALIBRATING'
        return None

    crossing = _deferred_crossing(eog_st, new_sig, sigma, now)

    if eog_st['sm'] == 'IDLE':
        if crossing is not None:
            eog_st['sm']        = 'ARMED'
            eog_st['first_dir'] = crossing
            eog_st['arm_time']  = now
            # the arming saccade is a directional motion the detector counted — band it
            _log_motion(eog_st, now, crossing)

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
                # the return saccade that completes the pair is itself a directional
                # motion the detector counted — band it (display only; never read back).
                _log_motion(eog_st, now, crossing)
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
PIPELINE_VERSION = "pipeline-v3"


def pipeline_description(detector='velocity'):
    """One-line English description of the current live preprocessing + detection
    METHOD (for the requested `detector`), embedded in each recording's notes so the
    pipeline is reconstructable from the file alone. Omits the numeric cutoffs/
    thresholds (stored as the lpf_hz/hpf_hz/sigma_thr/glance_window_s/defer_ms fields)
    — it describes HOW, not WITH-WHAT."""
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
            "signal. A crossing of (cand_sigma_thr x robust MAD baseline sigma) sustained "
            ">= min-duration NOMINATES a candidate; its SAME-SIGN peak is then accumulated "
            "across polls for defer_ms (or until a sustained opposite-sign run closes the "
            "window early, so the return saccade can never flip the direction) and compared "
            "to the real bar (sigma_thr x sigma). A confirmed crossing arms a direction, the "
            "OPPOSITE confirmed crossing within glance_window fires, then a refractory "
            "dead-time. With defer_ms = 0 the candidate stage is skipped and the first "
            "sustained crossing decides immediately (the pre-v3 detector). "
        )
        code = "_eog_filter / _eog_velocity / _deferred_crossing / _run_eog_sm"
    tail = ("Cutoff/threshold values are in the lpf_hz/hpf_hz/sigma_thr/"
            f"glance_window_s/defer_ms fields. Code: eog_core.{code}. [{PIPELINE_VERSION}]")
    return pre + det + tail
