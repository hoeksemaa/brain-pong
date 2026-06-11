"""
_eog_filter — the shared 0.5–100 Hz causal IIR chain.

These assert *contract and frequency response*, not exact sample values, so they
survive a re-tuning of the chain as long as its job (pass HEOG band, kill DC,
notch mains, don't mutate input) is intact. Where a behaviour depends on a
cutoff, the cutoff is passed explicitly so the test means the same thing even if
the production default moves.
"""
import numpy as np
import pytest

from eog_core import _eog_filter
from synth import SR, sine, interior_rms


def _rms_ratio(freq, sr=SR, dur_s=8.0, **kw):
    x = sine(freq, amp=10.0, dur_s=dur_s, sr=sr)
    y = _eog_filter(x.copy(), sr, **kw)
    return interior_rms(y, sr) / interior_rms(x, sr)


# ── contract ────────────────────────────────────────────────────────────────────

def test_does_not_mutate_input():
    x = sine(10, 10.0, 2.0)
    x0 = x.copy()
    _eog_filter(x, SR)
    np.testing.assert_array_equal(x, x0)


def test_preserves_shape_and_dtype():
    x = sine(10, 10.0, 2.0)
    y = _eog_filter(x.copy(), SR)
    assert y.shape == x.shape
    assert y.dtype == np.float64


def test_short_input_returned_unfiltered_as_float64():
    x = np.arange(10, dtype=np.float64)  # < 20 samples: no room for IIR to settle
    y = _eog_filter(x.copy(), SR)
    assert y.dtype == np.float64
    np.testing.assert_array_equal(y, x)


def test_deterministic():
    x = sine(12, 7.0, 3.0)
    np.testing.assert_array_equal(_eog_filter(x.copy(), SR), _eog_filter(x.copy(), SR))


# ── frequency response ───────────────────────────────────────────────────────────

def test_removes_dc_offset():
    x = 500.0 + sine(8, 10.0, 4.0)   # big DC pedestal + small saccade-band wiggle
    y = _eog_filter(x.copy(), SR)
    assert abs(np.mean(y)) < 1.0      # detrend + HPF crush the pedestal


def test_passes_heog_band():
    # 8 Hz sits squarely in the EOG band; most energy should survive.
    assert _rms_ratio(8.0) > 0.5


def test_attenuates_below_highpass():
    # 0.2 Hz is below the 0.5 Hz HPF → strongly attenuated relative to passband.
    assert _rms_ratio(0.2) < 0.3


def test_attenuates_above_lowpass():
    # 110 Hz is above the 100 Hz LPF (and below Nyquist=125) → attenuated.
    assert _rms_ratio(110.0) < 0.3


def test_notches_mains_60hz():
    # 60 Hz mains should be cut hard vs a neighbouring in-band tone.
    r60 = _rms_ratio(60.0)
    r20 = _rms_ratio(20.0)
    assert r60 < 0.5
    assert r60 < r20


def test_cutoffs_are_honoured_as_parameters():
    # Tightening the LPF to 12 Hz must attenuate 20 Hz that the default passes.
    # This is the anti-calcification lever: behaviour is a function of the arg,
    # so the test stays meaningful when the production default changes.
    wide = _rms_ratio(20.0)
    narrow = _rms_ratio(20.0, lpf_hz=12.0)
    assert narrow < wide
