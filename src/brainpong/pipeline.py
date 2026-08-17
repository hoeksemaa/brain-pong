"""Offline replay of the LIVE BrainPong detection pipeline over a frozen recording.

Feeds a recording through the exact production code path — ``eog_core.eog_diff →
_eog_filter → _eog_velocity → (_matched_filter) → _run_eog_sm`` —
the same functions, in the same order, with the
same chunking the game uses live: one poll every ``POLL_S`` (100 ms), each poll
filtering a fresh ``SETTLE_S + POLL_S`` (0.5 s) window and handing only the newest
``POLL_S`` of detector signal to the state machine (mirrors ``_poll_eog`` in
scripts/pong_game_brainflow.py). Per-recording tuning (sigma_thr / lpf_hz / hpf_hz /
glance_window_s / detector) is read from the npz so each game replays under the
settings it was actually played with; the ``calib_start`` / ``play_start`` /
``train_start`` event markers place calibration and the play-settle mute exactly
where they happened in the session.

Every intermediate stage is captured and stitched from the per-poll windows (the
newest-``POLL_S`` slice of each freshly filtered window), so the displayed traces
are literally the samples the live detector saw — including per-window filter
transients — not an idealized whole-recording filter pass. The filter sub-steps
re-run the same ``DataFilter`` calls as ``_eog_filter`` in the same order, copying
after each op; ``_eog_filter`` itself remains the source of truth for the chain.

Known replay limit: live wallclock jitter of the 100 ms Dash poll is replaced by
exact sample-aligned steps.

Pure numpy + brainflow.data_filter (via eog_core); no board, no Dash.
"""
import functools
import math

import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

from brainpong import eog_core
from brainpong.eog_core import (
    EOG_SIGMA_THR, EOG_LPF_HZ, EOG_HPF_HZ, GLANCE_WINDOW_S, EOG_BASELINE_S,
    EOG_MIN_DUR_MS, REFRACTORY_S, PLAY_SETTLE_S, NOTCH_BANDS, MATCHED_TEMPLATE_MS,
    _make_eog_state, _eog_velocity, _matched_filter, _make_velocity_template,
    _run_eog_sm, begin_play_settle, eog_diff,
)

# Live game-loop poll timing — mirrors EOG_POLL_S / EOG_SETTLE_S in
# scripts/pong_game_brainflow.py (they live in the game script, not eog_core;
# keep in sync if that script's values ever change).
POLL_S   = 0.1
SETTLE_S = 0.4


def _field(d, key, default=None):
    if key not in d.files:
        return default
    v = d[key]
    try:
        return v.item() if v.shape == () else v[0]
    except Exception:
        return default


def _filter_steps(x_uv, sr, lpf_hz, hpf_hz):
    """The exact op sequence of eog_core._eog_filter, capturing a copy after each
    stage. Same DataFilter calls, same order, same orders/params — so the final
    'highpass' array is bit-identical to _eog_filter's return value."""
    y = np.ascontiguousarray(x_uv.astype(np.float64))
    if y.size < 20:                       # _eog_filter passthrough for tiny windows
        return {'detrend': y, 'lowpass': y, 'notch': y, 'highpass': y}
    DataFilter.detrend(y, DetrendOperations.CONSTANT.value)
    det = y.copy()
    DataFilter.perform_lowpass(y, sr, lpf_hz, 4, FilterTypes.BUTTERWORTH.value, 0)
    lp = y.copy()
    for lo, hi in NOTCH_BANDS:
        DataFilter.perform_bandstop(y, sr, lo, hi, 3, FilterTypes.BUTTERWORTH.value, 0)
    nc = y.copy()
    DataFilter.perform_highpass(y, sr, hpf_hz, 4, FilterTypes.BUTTERWORTH.value, 0)
    return {'detrend': det, 'lowpass': lp, 'notch': nc, 'highpass': y}


def _decimate(arr, i0, i1, width, sr):
    """Min/max envelope decimation of arr[i0:i1] to <=width buckets.
    Returns (t, mn, mx) as rounded python lists; t in seconds of the recording."""
    seg = arr[i0:i1]
    n = seg.size
    if n == 0:
        return [], [], []
    width = max(10, min(int(width), n))
    edges = np.linspace(0, n, width + 1).astype(int)
    edges = np.unique(edges)
    mn = np.minimum.reduceat(seg, edges[:-1])
    mx = np.maximum.reduceat(seg, edges[:-1])
    tc = (edges[:-1] + np.diff(edges) / 2.0 + i0) / sr
    r = 1 if np.nanmax(np.abs(seg)) >= 100 else 3
    return ([round(float(v), 3) for v in tc],
            [round(float(v), r) for v in mn],
            [round(float(v), r) for v in mx])


def _valid_span(arr):
    """(i0, i1) of the finite region of a NaN-padded stitched array (i1 exclusive)."""
    ok = np.flatnonzero(np.isfinite(arr))
    if ok.size == 0:
        return 0, 0
    return int(ok[0]), int(ok[-1]) + 1


@functools.lru_cache(maxsize=4)
def replay(npz_path, width=1400):
    """Replay one recording through the live pipeline. Returns a JSON-safe dict:
    step traces (decimated), calibration/threshold info, state-machine timeline,
    commands, and the resulting paddle-zone trace."""
    d = np.load(npz_path, allow_pickle=True)
    eeg  = d['eeg']
    sr   = int(_field(d, 'sample_rate', 250))
    ch_l = int(d['eog_ch_L'][0])
    ch_r = int(d['eog_ch_R'][0])
    n    = eeg.shape[1]
    dur  = n / sr

    # Per-recording tuning — the settings this game was actually played with
    # (eog-v3 in-game recordings); older files fall back to the eog_core defaults.
    sigma_thr = float(_field(d, 'sigma_thr',       EOG_SIGMA_THR))
    lpf_hz    = float(_field(d, 'lpf_hz',          EOG_LPF_HZ))
    hpf_hz    = float(_field(d, 'hpf_hz',          EOG_HPF_HZ))
    glance_s  = float(_field(d, 'glance_window_s', GLANCE_WINDOW_S))
    detector  = str(_field(d, 'detector', 'velocity'))
    # Absent on every pre-v3 file → 0.0 → _deferred_crossing degrades to the immediate
    # detector those games actually ran, so legacy replay stays bit-identical.
    defer_ms  = float(_field(d, 'defer_ms',       0.0))
    cand_thr  = float(_field(d, 'cand_sigma_thr', 0.0))
    matched   = detector == 'matched'

    ev_s = d['event_samples'].astype(int) if 'event_samples' in d.files else np.array([], int)
    ev_l = d['event_labels'].astype(str)  if 'event_labels'  in d.files else np.array([], str)
    events = {}                                   # first occurrence of each label wins
    for s, l in zip(ev_s, ev_l):
        events.setdefault(str(l), int(s))
    t_calib = events.get('calib_start', 0) / sr
    i_play  = events.get('play_start', events.get('train_start'))
    t_play  = (i_play / sr) if i_play is not None else None

    # ── the exact live state, configured as at board setup + slider settings ──
    st = _make_eog_state()
    st.update({'ch_L': ch_l, 'ch_R': ch_r, 'sr': sr,
               'sigma_thr': sigma_thr, 'glance_window_s': glance_s,
               'lpf_hz': lpf_hz, 'hpf_hz': hpf_hz, 'detector': detector,
               'defer_ms': defer_ms, 'cand_sigma_thr': cand_thr,
               'mf_template': _make_velocity_template(sr)})

    stage_keys = ['detrend', 'highpass', 'velocity'] + (['matched'] if matched else [])
    stitched = {k: np.full(n, np.nan) for k in stage_keys}

    n_settle = int(SETTLE_S * sr)
    n_new    = max(1, int(POLL_S * sr))
    n_win    = n_settle + n_new

    poll_states = []            # (t, sm-state) after every poll
    motions     = []            # every directional motion the SM counted (fire_log)
    commands    = []            # every fired command (+ live/discarded flag)
    sigma_lock_t = None
    settle_until = None
    played = False
    last_motion_t = -1.0    # fire_log is capped at 64 entries, so track new motions
                            # by their strictly-increasing timestamps, not list length

    i = max(n_win, int(math.ceil(t_calib * sr)))   # first poll with a full window
    while i <= n:
        now = i / sr
        # PLAYING transition: live, begin_play_settle fires once when status flips.
        if t_play is not None and not played and now >= t_play:
            begin_play_settle(st, now)             # SM → IDLE, mute PLAY_SETTLE_S
            settle_until = st['settle_until']
            played = True

        window = eeg[:, i - n_win:i]
        diff   = eog_diff(window, ch_r, ch_l)      # same op as _poll_eog
        steps  = _filter_steps(diff, sr, lpf_hz, hpf_hz)
        vel    = _eog_velocity(steps['highpass'], sr)
        det    = _matched_filter(vel, st['mf_template']) if matched else vel
        sl = slice(i - n_new, i)                   # stitch the newest POLL_S only —
        for k in ('detrend', 'highpass'):          # what the live loop computed fresh
            stitched[k][sl] = steps[k][-n_new:]
        stitched['velocity'][sl] = vel[-n_new:]
        if matched:
            stitched['matched'][sl] = det[-n_new:]

        result = _run_eog_sm(st, det[-n_new:], now, label='replay')
        if st['baseline_sigma'] is not None and sigma_lock_t is None:
            sigma_lock_t = now
        for f in st.get('fire_log', []):
            if f['t'] > last_motion_t:
                motions.append({'t': round(f['t'], 2), 'dir': f['cmd']})
                last_motion_t = f['t']
        if result is not None:
            live = (t_play is None) or (now >= t_play)   # calibration-tail fires are
            commands.append({'t': round(now, 2), 'cmd': result['command'],   # discarded
                             'seq': result['seq'], 'live': bool(live)})      # live
        poll_states.append((round(now, 2), st['sm']))
        i += n_new

    sigma = st['baseline_sigma']
    thr   = (sigma_thr * sigma) if sigma else None

    # ── SM state timeline → compressed segments ──
    segments = []
    for t, s in poll_states:
        if segments and segments[-1]['state'] == s:
            segments[-1]['t1'] = t
        else:
            segments.append({'t0': t - POLL_S, 't1': t, 'state': s})

    # ── assemble step panels (name + one-sentence description live HERE, next to
    #    the code they describe, so the UI text can't drift from the pipeline) ──
    notches = ' + '.join(f'{lo:g}–{hi:g}' for lo, hi in NOTCH_BANDS)

    def sig_step(key, name, desc, arr, unit='µV', **extra):
        i0, i1 = _valid_span(arr)
        t, mn, mx = _decimate(arr, i0, i1, width, sr)
        return {'key': key, 'kind': 'signal', 'name': name, 'desc': desc,
                'unit': unit, 't': t, 'mn': mn, 'mx': mx, **extra}

    # Raw L/R and the R−L differential are deliberately NOT steps here — the
    # DERIVED and RAW zones above the pipeline already show them.
    steps_out = [
        sig_step('detrend', 'Detrend (constant)',
                 'Subtracts each 0.5 s poll window’s mean (DataFilter.detrend CONSTANT) so the '
                 'electrode’s DC offset doesn’t bias the filters that follow.',
                 stitched['detrend']),
        sig_step('filtered', 'Band-limit filters',
                 f'Three causal Butterworth passes in sequence — 4th-order low-pass {lpf_hz:g} Hz '
                 '(strips EMG/broadband noise the velocity derivative would amplify), band-stops at '
                 f'{notches} Hz (mains hum, redundant above the corner but kept as the live chain '
                 f'runs it), and 4th-order high-pass {hpf_hz:g} Hz (slow electrode drift) — shown '
                 'after all three; this is the “filtered” trace the game’s live feed displays.',
                 stitched['highpass']),
        sig_step('velocity', 'Velocity (Engbert–Kliegl)',
                 '5-point smoothed derivative (µV/s): drift and filter-recovery tails are slow and '
                 'vanish, while a saccade’s steep edge becomes a sharp unipolar pulse.',
                 stitched['velocity'], unit='µV/s'),
    ]
    if matched:
        steps_out.append(sig_step(
            'matched', 'Matched filter',
            f'Velocity cross-correlated with a {MATCHED_TEMPLATE_MS:g} ms unit-norm Hann saccade '
            'template: a coherent saccade-shaped pulse gains over incoherent noise, and this '
            'response (not raw velocity) is what this game thresholded.',
            stitched['matched'], unit='µV/s'))
    det_key  = 'matched' if matched else 'velocity'
    thr_step = sig_step(
        'threshold', 'Baseline σ → threshold',
        f'The first {EOG_BASELINE_S:g} s of detector signal (eyes still) set a robust MAD σ; a '
        f'crossing must exceed {sigma_thr:g}×σ for ≥{EOG_MIN_DUR_MS:g} ms to count as a directional '
        'eye motion.',
        stitched[det_key], unit='µV/s')
    thr_step.update({'thr': round(float(thr), 1) if thr else None,
                     'sigma': round(float(sigma), 2) if sigma else None})
    steps_out.append(thr_step)
    steps_out.append({
        'key': 'sm', 'kind': 'sm', 'name': 'Glance-pair state machine',
        'desc': 'A sustained crossing ARMS its direction; the OPPOSITE crossing within the '
                f'{glance_s:g} s glance window FIRES the first direction as a command, then a '
                f'{REFRACTORY_S:g} s refractory blocks repeats — one involuntary saccade alone '
                'never moves the paddle.',
        'segments': segments, 'motions': motions, 'commands': commands,
        'glance_window_s': glance_s})
    for i, s in enumerate(steps_out):
        s['name'] = f'{i + 1} · {s["name"]}'

    return {
        'sr': sr, 'duration': round(dur, 2), 'detector': detector,
        'params': {'sigma_thr': sigma_thr, 'lpf_hz': lpf_hz, 'hpf_hz': hpf_hz,
                   'glance_window_s': glance_s, 'defer_ms': defer_ms},
        'poll_s': POLL_S, 'settle_s': SETTLE_S,
        'calib': {'t_start': round(t_calib, 2),
                  't_locked': round(sigma_lock_t, 2) if sigma_lock_t else None,
                  'sigma': round(float(sigma), 2) if sigma else None,
                  'thr': round(float(thr), 1) if thr else None},
        'play_start': round(t_play, 2) if t_play is not None else None,
        'settle_until': round(settle_until, 2) if settle_until else None,
        'n_commands': len([c for c in commands if c['live']]),
        'n_motions': len(motions),
        'steps': steps_out,
    }
