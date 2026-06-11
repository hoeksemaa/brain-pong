"""
_run_eog_sm — the glance-pair state machine shared by P1 and P2.

`now` is injected, so the machine is fully deterministic: we drive it with flat
±σ windows (which _sustained_crossing reads as RIGHT/LEFT) and explicit
timestamps. These tests pin the *protocol* — arm on one direction, fire on the
opposite within the glance window, then go refractory — and crucially that two
players' states are independent.
"""
import numpy as np
import pytest

from eog_core import (
    _run_eog_sm, _make_eog_state, _reset_eog_st,
    GLANCE_WINDOW_S, ARMED_MIN_WAIT_S, REFRACTORY_S, EOG_BASELINE_S,
)
from synth import SR, calibrated_state, const_window

RIGHT_SIG = const_window(6.0)   # σ defaults to 1.0 in calibrated_state → 6σ
LEFT_SIG = const_window(-6.0)


# ── calibration ──────────────────────────────────────────────────────────────────

def test_calibration_sets_sigma_and_goes_idle():
    st = _make_eog_state()
    st['sr'] = SR
    # One window holding the full baseline duration of noise.
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(int(EOG_BASELINE_S * SR) + 10) * 3.0
    assert _run_eog_sm(st, noise, now=0.0) is None
    assert st['sm'] == 'IDLE'
    assert st['baseline_sigma'] == pytest.approx(np.std(noise), rel=1e-6)


def test_calibration_waits_for_enough_samples():
    st = _make_eog_state()
    st['sr'] = SR
    _run_eog_sm(st, np.ones(int(EOG_BASELINE_S * SR) // 2), now=0.0)
    assert st['sm'] == 'CALIBRATING'      # not enough yet
    assert st['baseline_sigma'] is None


# ── the happy path ────────────────────────────────────────────────────────────────

def test_glance_pair_fires_first_direction():
    st = calibrated_state()
    assert _run_eog_sm(st, RIGHT_SIG, now=0.0) is None   # arm RIGHT
    assert st['sm'] == 'ARMED' and st['first_dir'] == 'RIGHT'
    cmd = _run_eog_sm(st, LEFT_SIG, now=0.2)             # opposite within window
    assert cmd == {'command': 'RIGHT', 'seq': 1}
    assert st['sm'] == 'REFRACTORY'


def test_left_then_right_fires_left():
    st = calibrated_state()
    _run_eog_sm(st, LEFT_SIG, now=0.0)
    cmd = _run_eog_sm(st, RIGHT_SIG, now=0.2)
    assert cmd['command'] == 'LEFT'


# ── debounce / rejection paths ────────────────────────────────────────────────────

def test_single_glance_does_not_fire():
    st = calibrated_state()
    _run_eog_sm(st, RIGHT_SIG, now=0.0)                  # arm
    assert _run_eog_sm(st, RIGHT_SIG, now=0.2) is None   # same dir, no fire
    assert st['sm'] == 'ARMED'


def test_armed_times_out_then_returns_to_idle():
    st = calibrated_state()
    _run_eog_sm(st, RIGHT_SIG, now=0.0)
    # Opposite arrives *after* the glance window → counts as timeout, no fire.
    assert _run_eog_sm(st, LEFT_SIG, now=GLANCE_WINDOW_S + 0.1) is None
    assert st['sm'] == 'IDLE' and st['first_dir'] is None


def test_opposite_too_soon_does_not_fire():
    st = calibrated_state()
    _run_eog_sm(st, RIGHT_SIG, now=0.0)
    # Before ARMED_MIN_WAIT_S the opposite glance is ignored.
    assert _run_eog_sm(st, LEFT_SIG, now=ARMED_MIN_WAIT_S / 2) is None
    assert st['sm'] == 'ARMED'


def test_refractory_blocks_then_clears():
    st = calibrated_state()
    _run_eog_sm(st, RIGHT_SIG, now=0.0)
    fire_t = 0.2
    assert _run_eog_sm(st, LEFT_SIG, now=fire_t)['seq'] == 1
    # Within refractory: input ignored.
    assert _run_eog_sm(st, RIGHT_SIG, now=fire_t + REFRACTORY_S / 2) is None
    assert st['sm'] == 'REFRACTORY'
    # After refractory: a tick clears back to IDLE...
    assert _run_eog_sm(st, np.zeros(25), now=fire_t + REFRACTORY_S + 0.1) is None
    assert st['sm'] == 'IDLE'
    # ...and a fresh pair fires again with an incremented seq.
    _run_eog_sm(st, RIGHT_SIG, now=fire_t + REFRACTORY_S + 0.2)
    cmd = _run_eog_sm(st, LEFT_SIG, now=fire_t + REFRACTORY_S + 0.4)
    assert cmd == {'command': 'RIGHT', 'seq': 2}


# ── the headline: P1 and P2 are independent ───────────────────────────────────────

def test_two_players_do_not_interfere():
    p1 = calibrated_state(ch_L=0, ch_R=1)
    p2 = calibrated_state(ch_L=2, ch_R=3)

    # Arm P1 RIGHT but leave it hanging.
    _run_eog_sm(p1, RIGHT_SIG, now=0.0, label='P1')
    assert p1['sm'] == 'ARMED'

    # Drive a full LEFT-pair on P2.
    _run_eog_sm(p2, LEFT_SIG, now=0.0, label='P2')
    cmd2 = _run_eog_sm(p2, RIGHT_SIG, now=0.2, label='P2')

    assert cmd2 == {'command': 'LEFT', 'seq': 1}
    # P1 is untouched by anything P2 did.
    assert p1['sm'] == 'ARMED' and p1['first_dir'] == 'RIGHT'
    assert p1['cmd_seq'] == 0


def test_reset_returns_to_calibrating_but_keeps_channels():
    st = calibrated_state(ch_L=2, ch_R=3)
    st['cmd_seq'] = 7
    _reset_eog_st(st)
    assert st['sm'] == 'CALIBRATING'
    assert st['baseline_sigma'] is None
    assert st['ch_L'] == 2 and st['ch_R'] == 3   # channel mapping survives reset
