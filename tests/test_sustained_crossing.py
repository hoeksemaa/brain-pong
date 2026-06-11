"""
_sustained_crossing — direction of the first sustained threshold crossing.

The persistence gate here is BrainPong's only line of defence against spiky
artifact, so its boundaries matter. Thresholds/durations are passed explicitly
so each test pins one behaviour independent of the production defaults.
"""
import numpy as np
import pytest

from eog_core import _sustained_crossing
from synth import SR, const_window

SIGMA = 2.0
THR = 5.0  # sigma_thr used throughout (so the crossing threshold is 5·σ = 10 µV)


def test_below_threshold_returns_none():
    sig = const_window(4.0 * SIGMA, dur_s=0.2)  # 8 µV < 10 µV threshold
    assert _sustained_crossing(sig, SIGMA, SR, sigma_thr=THR) is None


def test_sustained_positive_is_right():
    sig = const_window(6.0 * SIGMA, dur_s=0.2)
    assert _sustained_crossing(sig, SIGMA, SR, sigma_thr=THR) == 'RIGHT'


def test_sustained_negative_is_left():
    sig = const_window(-6.0 * SIGMA, dur_s=0.2)
    assert _sustained_crossing(sig, SIGMA, SR, sigma_thr=THR) == 'LEFT'


def test_single_sample_spike_is_rejected():
    # One lone supra-threshold sample (classic EMG spike) must NOT cross.
    sig = np.zeros(60)
    sig[30] = 100.0 * SIGMA
    assert _sustained_crossing(sig, SIGMA, SR, sigma_thr=THR) is None


def test_run_shorter_than_min_dur_rejected():
    # min_dur_ms=12 ms at 250 Hz = 3 samples; a 2-sample run must fail.
    sig = np.zeros(60)
    sig[30:32] = 100.0 * SIGMA
    assert _sustained_crossing(sig, SIGMA, SR, sigma_thr=THR, min_dur_ms=12.0) is None


def test_run_exactly_min_dur_fires():
    sig = np.zeros(60)
    sig[30:33] = 100.0 * SIGMA  # exactly 3 samples
    assert _sustained_crossing(sig, SIGMA, SR, sigma_thr=THR, min_dur_ms=12.0) == 'RIGHT'


def test_direction_taken_at_onset_not_peak():
    # If the first sustained run is negative, a later bigger positive run is
    # irrelevant: the detector commits to the onset sign.
    sig = np.zeros(120)
    sig[20:40] = -6.0 * SIGMA
    sig[60:80] = 50.0 * SIGMA
    assert _sustained_crossing(sig, SIGMA, SR, sigma_thr=THR) == 'LEFT'


def test_zero_sigma_guarded():
    sig = const_window(100.0, dur_s=0.2)
    assert _sustained_crossing(sig, 0.0, SR, sigma_thr=THR) is None


def test_empty_signal_returns_none():
    assert _sustained_crossing(np.array([]), SIGMA, SR, sigma_thr=THR) is None


def test_threshold_is_a_real_lever():
    sig = const_window(7.0 * SIGMA, dur_s=0.2)  # 14 µV
    assert _sustained_crossing(sig, SIGMA, SR, sigma_thr=6.0) == 'RIGHT'   # 12 µV thr
    assert _sustained_crossing(sig, SIGMA, SR, sigma_thr=8.0) is None      # 16 µV thr
