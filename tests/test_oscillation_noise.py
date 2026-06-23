"""
Oscillating-noise behaviour of the EOG detector.

Motivation: the reported failure mode is *oscillating* noise, not single spikes.
The sustained-crossing gate (test_sustained_crossing) already kills single
spikes — but it does nothing about oscillation, and the glance-pair state
machine is actively *vulnerable* to it: an oscillation that swings above ±5σ
presents as an endless stream of alternating LEFT/RIGHT crossings, which is
exactly the "look one way then the other" pattern the machine fires on.

This file:
  1. confirms the filter kills 60 Hz mains (the notch) but PASSES low-frequency
     oscillation (electrode sway / motion / sub-notch mains harmonics), so the
     detector actually sees it;
  2. reproduces the spurious command end-to-end  (test_*_reproduces_*);
  3. shows a cheap run-count discriminator cleanly separates a real glance-pair
     from oscillation — evidence a mitigation is feasible (NOT wired into prod).

Tests in (2) document a *known defect*. If/when a mitigation lands, they should
flip to asserting zero spurious fires.
"""
import numpy as np
import pytest

from brainpong.eog_core import _eog_filter, _run_eog_sm, EOG_SIGMA_THR
from synth import SR, sine, calibrated_state, interior_rms, threshold_runs


# ── 1. the filter lets low-freq oscillation through ───────────────────────────────

def test_filter_kills_60hz_but_passes_low_freq_oscillation():
    sixty = sine(60.0, 10.0, 6.0)
    low = sine(5.0, 10.0, 6.0)        # cable sway / motion band
    r60 = interior_rms(_eog_filter(sixty.copy(), SR), SR) / interior_rms(sixty, SR)
    r5 = interior_rms(_eog_filter(low.copy(), SR), SR) / interior_rms(low, SR)
    assert r60 < 0.5          # mains notched
    assert r5 > 0.7           # 5 Hz oscillation survives → detector will see it


# ── 2. reproduce the spurious fire end-to-end ─────────────────────────────────────

def _feed_oscillation(freq, amp_sigma, sigma=1.0, dur_s=3.0, poll_s=0.1):
    """Filter an oscillation, chop into poll windows, run the state machine.

    Mirrors the live loop: _poll_eog hands _run_eog_sm the last poll_s of
    filtered signal each tick. Returns the list of fired commands.
    """
    osc = sine(freq, amp_sigma * sigma, dur_s, sr=SR)
    filt = _eog_filter(osc.copy(), SR)
    st = calibrated_state(sigma=sigma)
    n_new = max(1, int(poll_s * SR))
    cmds = []
    for k in range(filt.size // n_new):
        w = filt[k * n_new:(k + 1) * n_new]
        out = _run_eog_sm(st, w, now=k * poll_s, label='OSC')
        if out is not None:
            cmds.append(out)
    return cmds


@pytest.mark.parametrize("freq", [3.0, 5.0, 8.0])
def test_suprathreshold_oscillation_reproduces_spurious_fire(freq):
    # KNOWN DEFECT: a clean oscillation above threshold drives phantom commands.
    cmds = _feed_oscillation(freq, amp_sigma=10.0)
    assert len(cmds) >= 1, (
        f"{freq} Hz oscillation produced no command — if a mitigation landed, "
        "flip this test to assert zero spurious fires."
    )


def test_subthreshold_oscillation_is_safe():
    # Sanity floor: oscillation that never reaches 5σ must never fire, so the
    # defect above is genuinely about amplitude crossing threshold, not a bug
    # in the harness.
    cmds = _feed_oscillation(5.0, amp_sigma=2.0)
    assert cmds == []


# ── 3. a candidate discriminator that separates pair vs oscillation ───────────────

def test_run_count_discriminates_glance_pair_from_oscillation():
    """Over one glance window: a real pair is 2 supra-threshold runs; an
    oscillation is many. A cheap run-count (or its cousin, a 'return to
    baseline between glances' gate) would let the detector reject oscillation
    without touching its latency on real saccades. Proven here, not shipped.
    """
    sigma, thr = 1.0, EOG_SIGMA_THR * 1.0
    win_s = 0.7

    # Real glance-pair: RIGHT pulse, gap at baseline, LEFT pulse.
    pair = np.zeros(int(win_s * SR))
    pair[10:40] = 10.0 * sigma
    pair[120:150] = -10.0 * sigma

    # Oscillation over the same window.
    osc = sine(6.0, 10.0 * sigma, win_s, sr=SR)

    pair_runs = threshold_runs(pair, thr)
    osc_runs = threshold_runs(osc, thr)

    assert pair_runs == 2
    assert osc_runs >= 4
    assert osc_runs > pair_runs
