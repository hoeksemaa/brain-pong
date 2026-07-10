"""Matched-filter detector DSP: template shape, direction contract, SNR gain."""

import numpy as np

from brainpong.eog_core import (
    _make_velocity_template, _matched_filter, _sustained_crossing,
    MATCHED_TEMPLATE_MS,
)


def test_template_is_unit_norm_positive_bump():
    sr = 250
    t = _make_velocity_template(sr)
    assert t.ndim == 1 and len(t) == round(MATCHED_TEMPLATE_MS / 1000 * sr)   # 30
    assert np.all(t >= 0)                    # a saccade is a unipolar velocity pulse
    assert abs(np.linalg.norm(t) - 1.0) < 1e-9


def _synthetic(sign, seed=0):
    """500-sample velocity: white noise + a saccade bump at 200–230, given sign."""
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, 1.0, 500)
    v[200:230] += np.hanning(30) * 6.0 * sign
    return v


def test_matched_filter_direction_positive():
    tmpl = _make_velocity_template(250)
    r = _matched_filter(_synthetic(+1), tmpl)
    pk = int(np.argmax(np.abs(r)))
    assert r[pk] > 0            # rightward (positive) bump → positive peak
    assert 180 <= pk <= 250     # peak lands near the injected saccade


def test_matched_filter_direction_negative():
    tmpl = _make_velocity_template(250)
    r = _matched_filter(_synthetic(-1), tmpl)
    assert r[int(np.argmax(np.abs(r)))] < 0     # leftward → negative peak


def test_matched_filter_widens_separation():
    tmpl = _make_velocity_template(250)
    v = _synthetic(+1, seed=3)
    r = _matched_filter(v, tmpl)
    snr = lambda x: np.max(np.abs(x)) / (1.4826 * np.median(np.abs(x - np.median(x))) + 1e-9)
    assert snr(r) > snr(v)      # integrating over the template lifts signal above noise


def test_direction_contract_into_sustained_crossing():
    # the matched-response peak sign drives _sustained_crossing's RIGHT/LEFT contract
    tmpl = _make_velocity_template(250)
    r = _matched_filter(_synthetic(+1), tmpl)
    sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
    assert _sustained_crossing(r, sigma, 250, sigma_thr=4.0) == 'RIGHT'


def test_matched_filter_short_signal_is_safe():
    tmpl = _make_velocity_template(250)
    out = _matched_filter(np.zeros(5), tmpl)   # shorter than the template
    assert out.shape == (5,) and not np.any(out)
