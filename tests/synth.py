"""
Synthetic EOG signal + state helpers shared across the test suite.

Everything here is deterministic. No recordings are read (those are read-only
ground truth and not needed to exercise the detection *logic*); we synthesize
the minimal signal shapes that pin each behaviour. Sample rate defaults to
250 Hz to match the Cerelog X8.
"""
import numpy as np

from brainpong.eog_core import _make_eog_state

SR = 250  # Hz, matches the X8


def calibrated_state(sigma=1.0, sr=SR, ch_L=0, ch_R=1):
    """A state dict already past calibration: sm='IDLE', baseline_sigma set."""
    st = _make_eog_state()
    st['sr'] = sr
    st['ch_L'] = ch_L
    st['ch_R'] = ch_R
    st['baseline_sigma'] = float(sigma)
    st['sm'] = 'IDLE'
    return st


def const_window(value, dur_s=0.1, sr=SR):
    """Flat window of `value`, long enough to clear the sustained-crossing gate."""
    n = max(1, int(round(dur_s * sr)))
    return np.full(n, float(value), dtype=np.float64)


def sine(freq, amp, dur_s, sr=SR, phase=0.0):
    """amp * sin(2π·freq·t + phase) over dur_s seconds."""
    t = np.arange(int(round(dur_s * sr))) / float(sr)
    return amp * np.sin(2 * np.pi * freq * t + phase)


def interior_rms(x, sr=SR, skip_s=1.0):
    """RMS of the interior of x, dropping skip_s at each end (IIR edge/settle)."""
    k = int(round(skip_s * sr))
    core = x[k:-k] if x.size > 2 * k else x
    return float(np.sqrt(np.mean(core ** 2)))


def threshold_runs(sig, thr):
    """Number of distinct runs where |sig| > thr — i.e. how many separate events.

    A real glance-pair is 2 runs; an oscillation is many. Used by the
    oscillation tests as a *candidate* discriminator (not wired into production).
    """
    above = np.abs(sig) > thr
    if above.size == 0:
        return 0
    rises = int(np.sum((~above[:-1]) & above[1:]))
    return rises + (1 if above[0] else 0)


def confirm_crossing(st, sig, t0, poll_s=0.1, max_polls=8):
    """Drive the SM with `sig` from t0 until its deferred detector RESOLVES the
    candidate, returning (result, t_resolved).

    The live loop hands the SM one poll_s slice per tick, and since pipeline-v3 a
    crossing is confirmed defer_ms after it is nominated — not on the poll that
    found it. Tests that care about the glance-PAIR protocol (arm / fire / timeout /
    refractory) should use this so they exercise the real detector instead of
    assuming a crossing lands on the first window. With defer_ms=0 it resolves on
    the first poll, so it is correct for the legacy path too.
    """
    from brainpong.eog_core import _run_eog_sm
    t = t0
    for _ in range(max_polls):
        r = _run_eog_sm(st, sig, now=t)
        if st.get('pend_until') is None:      # nominated candidate now confirmed/discarded
            return r, t
        t += poll_s
    raise AssertionError(f"crossing never resolved within {max_polls} polls")
