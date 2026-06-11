"""
eog_diff — the horizontal differential + polarity contract.

This is the single most widely-shared function: every player, every mode, every
poll computes (ch_R − ch_L) through it. Its sign convention is load-bearing —
get ch_R/ch_L backwards and every paddle moves the wrong way (cf. the recorded
electrode-swap on John's sessions). These tests pin the contract, not the
arithmetic.
"""
import numpy as np
import pytest

from eog_core import eog_diff


def _data(n=64, n_ch=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_ch, n)).astype(np.float64)


def test_diff_is_R_minus_L_in_microvolts():
    data = _data()
    out = eog_diff(data, ch_R=1, ch_L=0)
    np.testing.assert_allclose(out, (data[1] - data[0]) * 1e6)


def test_rightward_is_positive():
    # ch_R larger than ch_L → positive deflection → 'RIGHT' downstream.
    data = np.zeros((4, 32))
    data[1] = 5e-4   # 500 µV on the right electrode
    out = eog_diff(data, ch_R=1, ch_L=0)
    assert np.all(out > 0)


def test_swapping_channels_inverts_sign():
    # The electrode-swap failure mode: wrong mapping flips the whole signal.
    data = _data()
    normal = eog_diff(data, ch_R=1, ch_L=0)
    swapped = eog_diff(data, ch_R=0, ch_L=1)
    np.testing.assert_allclose(swapped, -normal)


def test_player_channel_independence():
    # P1 uses slots 0/1, P2 uses 2/3. Each diff must read only its own rows,
    # with zero crosstalk — the core invariant that makes 2-player work.
    data = _data()
    p1 = eog_diff(data, ch_R=1, ch_L=0)
    # Stomp P2's rows; P1's result must be unchanged.
    data2 = data.copy()
    data2[2] = 999.0
    data2[3] = -999.0
    p1_after = eog_diff(data2, ch_R=1, ch_L=0)
    np.testing.assert_allclose(p1, p1_after)
    # And P2 actually reflects its own rows.
    p2 = eog_diff(data2, ch_R=3, ch_L=2)
    np.testing.assert_allclose(p2, (data2[3] - data2[2]) * 1e6)


def test_returns_fresh_float64_not_a_view():
    data = _data().astype(np.float32)  # force a dtype that must be promoted
    out = eog_diff(data, ch_R=1, ch_L=0)
    assert out.dtype == np.float64
    out[:] = 0.0
    # Mutating the result must not reach back into the recording array.
    assert not np.all(data[1] == 0.0)
