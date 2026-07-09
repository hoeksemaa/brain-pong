"""
_eog_velocity — the Engbert-Kliegl velocity statistic the detector now uses.

The detector thresholds VELOCITY, not amplitude. These pin (1) the velocity
computation and (2) the property that motivated the switch: a fast saccade and a
slow high-pass recovery tail can reach the SAME amplitude, yet the velocity gate
fires on the saccade and ignores the tail.
"""
import numpy as np
import pytest

from brainpong.eog_core import _eog_velocity, _sustained_crossing
from synth import SR


def _mad_sigma(v):
    """Production noise-scale estimator: 1.4826 · MAD."""
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


# ── the velocity computation ──────────────────────────────────────────────────────

def test_velocity_of_ramp_is_slope():
    t = np.arange(SR) / SR
    v = _eog_velocity(120.0 * t, SR)          # 120 µV/s ramp
    assert np.median(v) == pytest.approx(120.0, rel=0.05)


def test_velocity_of_flat_is_zero():
    assert np.allclose(_eog_velocity(np.full(SR, 37.0), SR), 0.0, atol=1e-6)


def test_sign_is_direction_of_gaze_change():
    t = np.arange(SR) / SR
    assert np.median(_eog_velocity(100.0 * t, SR)) > 0    # rising (rightward) → +
    assert np.median(_eog_velocity(-100.0 * t, SR)) < 0   # falling (leftward) → −


def test_short_input_safe():
    assert _eog_velocity(np.array([1.0, 2.0]), SR).shape == (2,)


# ── the headline: velocity separates a saccade from a same-amplitude slow tail ────

def _saccade_and_tail():
    """Two deflections of the SAME 200 µV amplitude: a ballistic saccade (200 µV
    over 40 ms) and a slow recovery tail (200 µV over 1.0 s)."""
    x = np.arange(2 * SR) / SR
    saccade = np.clip((x - 1.0) / 0.04, 0.0, 1.0) * 200.0
    tail    = np.clip((x - 1.0) / 1.00, 0.0, 1.0) * 200.0
    return saccade, tail


def test_fast_saccade_fires_slow_tail_does_not_at_equal_amplitude():
    saccade, tail = _saccade_and_tail()
    rng = np.random.default_rng(0)
    sigma_v = _mad_sigma(_eog_velocity(rng.standard_normal(3 * SR) * 1.0, SR))

    v_sacc = _eog_velocity(saccade, SR)
    v_tail = _eog_velocity(tail, SR)
    assert saccade.max() == pytest.approx(tail.max(), rel=0.01)          # same amplitude
    assert np.max(np.abs(v_sacc)) > 10 * np.max(np.abs(v_tail))          # ≫ velocity
    # the detector fires the saccade, ignores the tail (Engbert λ = 6)
    assert _sustained_crossing(v_sacc, sigma_v, SR, sigma_thr=6.0) == 'RIGHT'
    assert _sustained_crossing(v_tail, sigma_v, SR, sigma_thr=6.0) is None


def test_amplitude_gate_would_be_fooled_by_the_tail():
    # Contrast: on AMPLITUDE (the old detector) the slow tail crosses just like a
    # saccade — this is exactly the false-trigger the velocity switch removes.
    _, tail = _saccade_and_tail()
    assert _sustained_crossing(tail, 5.0, SR, sigma_thr=6.0) == 'RIGHT'   # amplitude fooled
