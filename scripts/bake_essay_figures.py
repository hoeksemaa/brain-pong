#!/usr/bin/env python3
"""Bake the blog-essay figure data from one committed recording.

Writes two files into web/essay-figures/data/:

  eog-full.bin    int16, µV. Ten arrays of n samples, little-endian, in the
                  order named by `layout` in the JSON:

                    ch_R, ch_L     deviation from each channel's global mean
                    detrend_live   the differential, detrended the way the game
                                   detrends it — see live_chain()
                    lp / notch / hp / lp_notch / lp_hp / notch_hp /
                    lp_notch_hp    the same, plus each SUBSET of the game's
                                   three filters (see FILTERS)

                  Every stage is computed here, through BrainFlow, once. The
                  browser only interpolates and draws.

  eog-figures.json  metadata — sample rate, per-signal means and y-spans, the
                  calibration σ, and the gaze cues.

The unfiltered differential is not stored; it is ch_R − ch_L and the browser
computes it.

WHY EIGHT SUBSETS. Figure 8 gives each of the three filters its own slider. The
filters are linear and commute, so blending them independently is exactly a
multilinear interpolation of the eight subset outputs:

    ∏ᵢ[(1−sᵢ)·I + sᵢ·Hᵢ]  =  Σ_S (∏_{i∈S} sᵢ)(∏_{i∉S}(1−sᵢ)) · ∏_{i∈S} Hᵢ

so every slider position is a filter the pipeline could actually run, not a
crossfade between two pictures. Storing the eight corners is what lets the
browser reach any of them with three multiplications per sample.

FIDELITY. The filter parameters are imported from `brainpong.eog_core`, not
restated, and the chain is applied the way `_poll_eog` applies it: to the
125-sample buffer the board hands over, every 100 ms, keeping the newest 25.
`_chain(..., True, True, True)` is asserted bit-identical to `_eog_filter`.

Ground truth is the raw recording. Nothing here modifies data/.

Usage:  python scripts/bake_essay_figures.py
"""

import json
import pathlib

import numpy as np
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes

# The live detector's own constants. Imported rather than restated so the
# figures cannot drift from the software they depict.
from brainpong.eog_core import (
    EOG_LPF_HZ, EOG_HPF_HZ, NOTCH_BANDS, EOG_BASELINE_S,
    EOG_MIN_DUR_MS,
    _eog_filter, _eog_velocity,
)

REC = pathlib.Path("data/eog/20260703-163542-john.npz")
OUT_DIR = pathlib.Path("web/essay-figures/data")

VIEW_S = 10.0      # the scrolling window the figures show
HEADROOM = 1.25    # y-span = worst slice in the recording × this

# The live acquisition window, mirrored from
# pong_game_brainflow.EOG_SETTLE_S / EOG_POLL_S.
EOG_SETTLE_S, EOG_POLL_S = 0.4, 0.1

# The three switchable stages of `_eog_filter`, in the order it applies them.
# Order is provably free here — running the game's order against any other with
# the same corners differs by 2.4e-8 µV rms, i.e. float round-off — but the
# game's order is what is reproduced, because there is no reason not to.
FILTERS = ("lp", "notch", "hp")

# THE ESSAY'S CORNERS, deliberately not the game's.
#
# eog_core runs LP 30 / HP 0.1, with the notch at 3rd order. The essay teaches
# the primitive version at 4th order throughout, and a 100 Hz low-pass corner,
# because that is what leaves mains INSIDE the passband and so gives the notch
# something visible to remove. At the game's real 30 Hz corner both notch bands
# sit above the low-pass and the notch moves the trace 0.26 px — true, and
# eog_core says as much ("redundant but harmless"), but it makes for a figure
# where a third of the controls does nothing.
#
# The cost of this choice is that figure 8 is NOT the game's filter chain. The
# machinery is still verified against the real one — see
# assert_mirrors_eog_filter(), which drives the same code with eog_core's own
# constants and demands bit-equality with `_eog_filter`. What diverges is the
# numbers, on purpose, and both sets are written into the JSON so the prose can
# name the difference instead of hiding it.
ESSAY_LPF_HZ = 100.0
ESSAY_HPF_HZ = 0.5
ESSAY_ORDER = 4          # every stage, including the notch (the game uses 3)

# Figure 9 holds still over the calibration instead of scrolling, with this
# much recording either side of it for context.
CALIB_PAD_S = 3.0

# Where a figure showing a filtered stage may start scrolling. Each poll filters
# a fresh 125-sample buffer from zero state, so there is no whole-record
# ring-down to skip any more — but the first polls of the record land on the
# electrode still settling, so figures 7-10 still start here.
VALID_FROM_S = 2.0

# Cue label → gaze direction. BASELINE / FAST_INTRO / DONE are protocol
# bookkeeping, not gaze states, so they are left out.
CUE_DIR = {
    "LEFT": "L", "FAST_LEFT": "L",
    "RIGHT": "R", "FAST_RIGHT": "R",
    "REST": "C", "FAST_REST": "C",
}


def bandpass(x, fs, lo, hi):
    y = np.ascontiguousarray(x.astype(np.float64))
    DataFilter.perform_bandpass(y, fs, lo, hi, 4, FilterTypes.BUTTERWORTH.value, 0)
    return y


def _chain(y, sr, lp, notch, hp,
           lpf_hz=ESSAY_LPF_HZ, hpf_hz=ESSAY_HPF_HZ,
           order=ESSAY_ORDER, notch_order=ESSAY_ORDER):
    """`eog_core._eog_filter` with each of its three filters switchable.

    Mutates and returns `y`. Same structure and same order of operations as
    `_eog_filter`; the corners default to the essay's rather than the game's
    (see ESSAY_LPF_HZ). The detrend is not switchable because the game never
    switches it: `_eog_filter` opens with it unconditionally, so it is part of
    every subset.

    Driven with eog_core's own constants this is bit-identical to
    `_eog_filter` — that is what assert_mirrors_eog_filter() checks.
    """
    DataFilter.detrend(y, DetrendOperations.CONSTANT.value)
    if lp:
        DataFilter.perform_lowpass(y, sr, lpf_hz, order, FilterTypes.BUTTERWORTH.value, 0)
    if notch:
        for lo, hi in NOTCH_BANDS:
            DataFilter.perform_bandstop(y, sr, lo, hi, notch_order,
                                        FilterTypes.BUTTERWORTH.value, 0)
    if hp:
        DataFilter.perform_highpass(y, sr, hpf_hz, order, FilterTypes.BUTTERWORTH.value, 0)
    return y


def assert_mirrors_eog_filter(diff, fs):
    """`_chain` must reproduce `_eog_filter` exactly when given its constants.

    The essay bakes different corners on purpose, so the baked arrays cannot be
    compared against the game directly. This checks the thing that must not
    drift regardless: that the CODE here is the same chain, in the same order,
    on the same buffer — only the numbers differ. If `_eog_filter` ever gains a
    stage or reorders one, this fails.
    """
    n_new = max(1, int(EOG_POLL_S * fs))
    win = int(EOG_SETTLE_S * fs) + n_new
    for end in (win, win + 37 * n_new, win + 501 * n_new):
        buf = diff[end - win:end].copy()
        _chain(buf, fs, True, True, True,
               lpf_hz=EOG_LPF_HZ, hpf_hz=EOG_HPF_HZ, order=4, notch_order=3)
        ref = _eog_filter(diff[end - win:end].copy(), fs)
        assert np.array_equal(buf, ref), f"_chain diverges from _eog_filter at {end}"


def live_chain(diff, fs, lp=False, notch=False, hp=False, velocity=False,
               settle_s=EOG_SETTLE_S, poll_s=EOG_POLL_S, **chain_kw):
    """Run `_chain` the way the game runs `_eog_filter` — poll by poll.

    `pong_game_brainflow._poll_eog` calls `get_current_board_data(n_settle +
    n_new)` — 0.4 s of context plus 0.1 s of new samples, 125 at 250 Hz — hands
    that whole buffer to `_eog_filter`, and then keeps only the newest `n_new`.
    So the entire chain, detrend included, is recomputed on a fresh overlapping
    buffer ten times a second, and each poll's IIR filters start from zero
    state. That startup transient is real and the game lives with it: the 0.1 Hz
    high-pass has τ ≈ 1.6 s and cannot settle inside a 0.5 s buffer, which
    `eog_core` documents and accepts because the velocity detector rejects a
    slow tail.

    With no filters enabled this is the detrend alone — one constant, the mean
    of the 125 samples ending at that poll, subtracted from all 25 samples the
    poll delivers. That is a BLOCK operation, not a sliding filter: for 24 of
    every 25 samples the constant is the mean of a window reaching up to 96 ms
    past the sample. The per-sample sliding equivalent is a different operator
    and differs by 12.2 µV rms on this recording.

    POLL ALIGNMENT. Live, the 100 ms interval is not sample-aligned and its
    phase drifts; the recording carries no record of where the boundaries fell.
    This uses a fixed `n_new` stride with the first poll ending at sample `win`,
    plus one short final poll ending at the last recorded sample so the tail is
    covered. Any other phase is equally faithful and moves the result by up to
    10.8 µV rms.

    Each sample takes the value from the poll that first delivered it, which is
    the value the detector was handed. Samples before the first full buffer are
    left at zero: the game returns None until the board has one.
    """
    n_new = max(1, int(poll_s * fs))
    win = int(settle_s * fs) + n_new
    x = np.ascontiguousarray(diff.astype(np.float64))
    out = np.zeros_like(x)

    ends = list(range(win, x.size + 1, n_new))
    # A recording rarely ends on a poll boundary. One more poll, ending at the
    # last sample, delivers the remainder exactly as a live poll would have.
    if ends[-1] < x.size:
        ends.append(x.size)

    prev = 0
    for end in ends:
        # .copy() is load-bearing: BrainFlow's DataFilter.* mutate in place, and
        # ascontiguousarray on an already-contiguous slice returns a VIEW, so
        # filtering would write back into `x` and corrupt every later poll.
        buf = x[end - win:end].copy()                  # get_current_board_data
        _chain(buf, fs, lp, notch, hp, **chain_kw)     # _eog_filter
        # _poll_eog takes the derivative over the WHOLE settled buffer and only
        # then slices, so no velocity estimate ever straddles a poll boundary.
        if velocity:
            buf = _eog_velocity(buf, fs)
        lo = max(end - n_new, prev)                    # only what this poll delivers
        out[lo:end] = buf[lo - (end - win):end - (end - win)]
        prev = end
    return out


def subset_key(subset):
    """Layout key for a subset of FILTERS. The empty subset is the detrend."""
    return "_".join(subset) if subset else "detrend_live"


def all_subsets():
    """The eight subsets of FILTERS, each as a tuple in FILTERS order."""
    out = []
    for bits in range(8):
        out.append(tuple(f for j, f in enumerate(FILTERS) if bits >> j & 1))
    return out


def calibrate(vel, fs, base_t, baseline_s=EOG_BASELINE_S, poll_s=EOG_POLL_S):
    """Reproduce `_run_eog_sm`'s CALIBRATING branch, poll by poll.

    The state machine appends each poll's `new_sig` to `baseline_acc`, and the
    FIRST time the accumulated length reaches `baseline_s * sr` it computes

        sigma = 1.4826 * median(|total - median(total)|)

    once, latches it, and leaves CALIBRATING.

    `x` is whatever the poll delivers. The GAME delivers velocity, so its sigma
    is in µV/s; the essay hands this the filtered amplitude instead and gets µV.
    See the figure 9 block in main() for why.

    Returns (i0, i1, sigma, sigma_by_poll) where [i0, i1) is the span of samples
    that actually entered the estimate. `sigma_by_poll[k]` is what sigma WOULD
    be had calibration stopped after k+1 polls — the code never computes it, so
    it is offered for the figure and must not be described as the game's.
    """
    n_new = max(1, int(poll_s * fs))
    need = int(baseline_s * fs)
    i0 = int(np.ceil(base_t * fs / n_new)) * n_new     # first poll at/after the cue
    by_poll, acc = [], []
    i = i0
    while True:
        acc.append(vel[i:i + n_new])
        total = np.concatenate(acc)
        med = float(np.median(total))
        by_poll.append(float(1.4826 * np.median(np.abs(total - med))))
        i += n_new
        if total.size >= need:
            return i0, i, by_poll[-1], by_poll


# Figure 10's slider. Range and default are historical: sigma_thr has held five
# values in this repo — 6.0, 5.0, 2.5, 4.0, 6.0 — and 2.5 is the one that was
# reached by dropping it "esp. on noisy-baseline players where 5σ was
# unreachable" (f90fecf, 2026-07-09).
DETECT_K = (1.0, 5.0, 0.05)          # min, max, step

# A detection counts as real if it lands in this window after a cue. The median
# lag from cue to peak |dx/dt| on this recording is 276 ms (IQR 259-312), so the
# window closes well after the response.
#
# It opens at zero, not at 0.10 s. A floor was meant to stop a detection being
# credited to the wrong cue, but in the FAST block the cues alternate every
# ~0.55 s and are entirely predictable, so the subject anticipates: two of the
# 8σ "misses" were 16σ and 24σ saccades detected at +0.036 s and +0.080 s and
# excluded for arriving too promptly. Nothing is lost by opening at zero — the
# preceding detection sits 0.32-0.35 s BEFORE the cue, outside any window.
RESP_WIN = (0.0, 1.00)


def detect_sweep(x, fs, sigma, cue_t, ks,
                 min_dur_ms=EOG_MIN_DUR_MS, poll_s=EOG_POLL_S, win=RESP_WIN):
    """Run `_sustained_crossing`'s rule across the whole record, for each k.

    The rule is the detector's own: |x| over k*sigma, held for at least
    min_dur_ms. Crossings are then thinned to one per POLL, because that is the
    granularity at which the detector can report: `_poll_eog` hands
    `_sustained_crossing` 25 samples every 0.1 s and it returns at most one
    crossing from them.

    NOT the refractory period. REFRACTORY_S is dead time after `_run_eog_sm`
    FIRES A COMMAND — it belongs to the state machine, and this figure stops
    short of the state machine. Using it here was a real error and it cost 18
    detections: fast-block cues are 0.548-0.604 s apart, so every
    return-to-centre saccade fell inside the dead time opened by the outbound
    one and was discarded, while the figure went on painting it red. The counts
    and the colour must obey the same rule.

    What this still does NOT include is the glance PAIR: a sustained crossing is
    a spike, not a command. The game needs an opposite crossing within
    GLANCE_WINDOW_S before it moves a paddle.

    Scoring is against EVERY cue, not just LEFT/RIGHT. A REST cue is "look back
    to centre", which is as real a saccade as the outbound one; counting only
    the directional cues makes 40 genuine eye movements look like false alarms
    and turns the false-alarm rate into nonsense.
    """
    min_dur = max(1, int(min_dur_ms / 1000 * fs))
    out = []
    for k in ks:
        above = np.abs(x) > k * sigma
        conv = np.convolve(above.astype(np.int32), np.ones(min_dur, np.int32), "valid")
        held = conv == min_dur
        onset = np.where(held & ~np.r_[False, held[:-1]])[0] / fs
        t = []
        for v in onset:
            if not t or v - t[-1] > poll_s:
                t.append(v)
        t = np.asarray(t)
        hit, matched = set(), np.zeros(t.size, bool)
        for ci, ct in enumerate(cue_t):
            m = (t >= ct + win[0]) & (t <= ct + win[1])
            if m.any():
                hit.add(ci)
                matched |= m
        out.append([int(t.size), len(hit), int((~matched).sum())])
    return out


def robust_sigma(x):
    """Noise floor, MAD estimator — the same one `_run_eog_sm` calibrates with.

    Median-based so a blink inside the baseline window widens σ far less than it
    would widen a standard deviation.
    """
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def y_span(x, fs, skip_s=0.0):
    """Panel height that comfortably fits ANY VIEW_S slice of this signal.

    Sized to the worst slice in the whole recording, not the average, so the
    trace can never overflow its panel as the window scrolls. The centre tracks
    the window mean, so the span is all the browser needs.

    A fixed ABSOLUTE axis was tried and rejected: electrode drift across 2m13s
    is 4-5x larger than anything that happens inside one slice, so a typical
    slice occupied ~4% of the panel and the glances were unreadable.
    """
    x = x[int(skip_s * fs):]
    w = int(VIEW_S * fs)
    worst = max(np.ptp(x[i:i + w]) for i in range(0, len(x) - w, fs))
    return float(np.ceil(worst * HEADROOM / 50.0) * 50.0)


def main():
    d = np.load(REC, allow_pickle=True)
    fs = int(np.atleast_1d(d["sample_rate"])[0])
    eeg = d["eeg"]

    # Channel roles come from the recording, not from a constant here.
    ch_r_i = int(np.atleast_1d(d["eog_ch_R"])[0])
    ch_l_i = int(np.atleast_1d(d["eog_ch_L"])[0])
    ch_R = eeg[ch_r_i] * 1e6          # µV, referred to input
    ch_L = eeg[ch_l_i] * 1e6
    diff = ch_R - ch_L
    n = ch_R.size

    mean_R, mean_L = float(ch_R.mean()), float(ch_L.mean())

    # Every subset of the three filters, each applied poll by poll.
    stages = {}
    for subset in all_subsets():
        flags = {f: (f in subset) for f in FILTERS}
        stages[subset_key(subset)] = live_chain(diff, fs, **flags)

    # The chain here must be structurally the game's, even though it is baked
    # with the essay's corners.
    n_new = max(1, int(EOG_POLL_S * fs))
    win = int(EOG_SETTLE_S * fs) + n_new
    assert_mirrors_eog_filter(diff, fs)

    labels = [str(v) for v in d["event_labels"]]
    ev = np.asarray(d["event_samples"])

    # ── Figure 9 ──────────────────────────────────────────────────────────────
    # σ is measured on the filtered AMPLITUDE — figure 8's output, the same
    # array it ends on — and not on the velocity.
    #
    # That diverges from the game, deliberately, and it is worth being blunt
    # about rather than quiet. `_poll_eog` returns
    # `_eog_velocity(filtered)[-n_new:]`, so the real detector calibrates on a
    # derivative, in µV/s. The essay keeps the whole chain in one quantity and
    # one unit: a figure that changes what it is measuring halfway down the
    # pipeline costs a reader more than the extra step buys. Both numbers are
    # recorded below; only the amplitude one is drawn.
    amp = stages["lp_notch_hp"]
    base_t = float(ev[labels.index("BASELINE")]) / fs
    c_i0, c_i1, sigma, sigma_by_poll = calibrate(amp, fs, base_t)

    # What the detector really calibrates on, for prose that wants to name the
    # gap: the same calibration on the velocity, at the essay's corners and at
    # the game's. Computed, not baked — no figure draws either.
    _, _, sigma_vel, _ = calibrate(
        live_chain(diff, fs, lp=True, notch=True, hp=True, velocity=True),
        fs, base_t)
    _, _, sigma_vel_game, _ = calibrate(
        live_chain(diff, fs, lp=True, notch=True, hp=True, velocity=True,
                   lpf_hz=EOG_LPF_HZ, hpf_hz=EOG_HPF_HZ, order=4, notch_order=3),
        fs, base_t)

    # The window figure 9 plays through: the calibration plus CALIB_PAD_S either
    # side. Centred on zero (the signal is zero-mean by construction), so the
    # span is sized to the worst excursion inside that window rather than to the
    # worst 10 s slice of the whole record.
    view_lo = max(0.0, base_t - CALIB_PAD_S)
    view_hi = c_i1 / fs + CALIB_PAD_S
    w0, w1 = int(view_lo * fs), int(np.ceil(view_hi * fs))
    span_amp = float(np.ceil(2 * np.abs(amp[w0:w1]).max() * HEADROOM / 50.0) * 50.0)

    # ── Figure 10 ─────────────────────────────────────────────────────────────
    # The same σ figure 9 just measured, swept across the slider's range.
    cue_t = [float(s) / fs for s, l in zip(ev, labels) if l in CUE_DIR]
    k_lo, k_hi, k_step = DETECT_K
    ks = [round(k_lo + i * k_step, 3)
          for i in range(int(round((k_hi - k_lo) / k_step)) + 1)]
    sweep = detect_sweep(amp, fs, sigma, cue_t, ks)

    layout = ["ch_R", "ch_L"] + [subset_key(s) for s in all_subsets()]
    arrays = {"ch_R": ch_R - mean_R, "ch_L": ch_L - mean_L, **stages}
    blob = b""
    for k in layout:
        assert np.abs(arrays[k]).max() < 32767, f"{k} overflows int16"
        blob += np.rint(arrays[k]).astype("<i2").tobytes()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eog-full.bin").write_bytes(blob)

    hum_R = bandpass(ch_R, fs, 58, 62)
    hum_L = bandpass(ch_L, fs, 58, 62)

    sigs = {"ch_R": ch_R, "ch_L": ch_L, "diff": diff, **stages}

    payload = {
        "recording": REC.name,
        "subject": "john",
        "fs": fs,
        "n": int(n),
        "duration_s": round(n / fs, 3),
        "view_s": VIEW_S,
        "units": "uV, referred to input",
        # Order of the int16 arrays in eog-full.bin, n samples each.
        "layout": layout,
        # What to add back to each stored deviation array to recover the signal.
        # The filter stages are already centred, so only the raw channels carry one.
        "mean": {"ch_R": round(mean_R, 1), "ch_L": round(mean_L, 1),
                 "diff": round(mean_R - mean_L, 1),
                 **{subset_key(s): 0.0 for s in all_subsets()}},
        # Panel heights, µV. Each sized so no slice of that signal can overflow.
        "span": {k: y_span(x, fs, VALID_FROM_S if k in stages else 0.0)
                 for k, x in sigs.items()},
        "valid_from_s": VALID_FROM_S,
        # Figure 8. The three switchable filters, in the order _eog_filter
        # applies them, each with the label its slider carries.
        "filters": {
            "order": list(FILTERS),
            "lp": {"hz": ESSAY_LPF_HZ, "order": ESSAY_ORDER,
                   "label": f"low-pass {ESSAY_LPF_HZ:g} Hz"},
            "notch": {"bands": [[lo, hi] for lo, hi in NOTCH_BANDS],
                      "order": ESSAY_ORDER,
                      "label": "notch "
                               + " / ".join(f"{(lo + hi) / 2:g}" for lo, hi in NOTCH_BANDS)
                               + " Hz"},
            "hp": {"hz": ESSAY_HPF_HZ, "order": ESSAY_ORDER,
                   "label": f"high-pass {ESSAY_HPF_HZ:g} Hz"},
        },
        # What the GAME runs, for prose that wants to name the divergence. The
        # figure is deliberately not this; see ESSAY_LPF_HZ in the bake.
        "game_filters": {"lp_hz": EOG_LPF_HZ, "hp_hz": EOG_HPF_HZ,
                         "notch_bands": [[lo, hi] for lo, hi in NOTCH_BANDS],
                         "order": 4, "notch_order": 3},
        # Figure 7. The game's own detrend, so the figure can name it.
        "detrend": {"settle_s": EOG_SETTLE_S, "poll_s": EOG_POLL_S,
                    "window_s": round(EOG_SETTLE_S + EOG_POLL_S, 3),
                    "window_n": win,
                    "poll_n": n_new,
                    "kind": "constant (mean subtraction), one per poll block"},
        # Figure 9. Everything here is in µV/s, on the velocity, because that is
        # what `_run_eog_sm` accumulates. `sigma_by_poll` is what sigma would be
        # had calibration stopped early — the game never computes it.
        "calib": {
            "baseline_t": round(base_t, 3),
            "baseline_s": EOG_BASELINE_S,
            "poll_s": EOG_POLL_S,
            "i0": int(c_i0), "i1": int(c_i1),
            "t0": round(c_i0 / fs, 4), "t1": round(c_i1 / fs, 4),
            "sigma": round(sigma, 2),
            "sigma_by_poll": [round(v, 2) for v in sigma_by_poll],
            "units": "uV",
            # Not drawn. What the live detector actually calibrates on — the
            # velocity — at the essay's corners and at the game's, in µV/s.
            "sigma_velocity": round(sigma_vel, 2),
            "sigma_velocity_game_corners": round(sigma_vel_game, 2),
            # The figure holds still over the calibration plus this much either
            # side, instead of scrolling the whole record.
            "pad_s": CALIB_PAD_S,
            "view": [round(view_lo, 4), round(view_hi, 4)],
            "span": span_amp,
        },
        # Figure 10. `sweep[i]` is [detections, cues hit, false alarms] at
        # k = k_min + i*k_step, on the same signal and the same σ as figure 9.
        "detect": {
            "k_min": k_lo, "k_max": k_hi, "k_step": k_step, "k_start": 2.5,
            "sigma": round(sigma, 2),
            "n_cues": len(cue_t),
            "resp_win": list(RESP_WIN),
            "min_dur_ms": EOG_MIN_DUR_MS,
            "poll_s": EOG_POLL_S,
            "sweep": sweep,
        },
        "gain": int(np.atleast_1d(d["gain"])[0]),
        "fullscale_uv": 187500.0,          # VREF/gain = 4.5 V / 24
        "lsb_uv": 187500.0 / (2 ** 23),    # ~0.022 µV
        "hum_R_rms_uv": round(float(hum_R.std()), 2),
        "hum_L_rms_uv": round(float(hum_L.std()), 2),
        # Gaze state at each cue.
        "cues": [
            {"t": round(float(s) / fs, 3), "dir": CUE_DIR[l]}
            for s, l in zip(ev, labels) if l in CUE_DIR
        ],
    }
    (OUT_DIR / "eog-figures.json").write_text(json.dumps(payload, separators=(",", ":")))

    binkb = (OUT_DIR / "eog-full.bin").stat().st_size / 1024
    jskb = (OUT_DIR / "eog-figures.json").stat().st_size / 1024
    print(f"wrote {OUT_DIR}/eog-full.bin ({binkb:.0f} KB) + eog-figures.json ({jskb:.0f} KB)")
    print(f"  {REC.name}: {n} samples @ {fs} Hz = {n / fs:.2f} s")
    print(f"  chain  detrend -> {' -> '.join(FILTERS)}  "
          f"(lp {ESSAY_LPF_HZ:g} Hz, notch {NOTCH_BANDS}, hp {ESSAY_HPF_HZ:g} Hz, "
          f"order {ESSAY_ORDER}), per {EOG_POLL_S:g} s poll over "
          f"{EOG_SETTLE_S + EOG_POLL_S:g} s")
    print(f"  ESSAY corners, not the game's (game: lp {EOG_LPF_HZ:g}, hp {EOG_HPF_HZ:g}, "
          f"notch order 3) — deliberate, see ESSAY_LPF_HZ")
    print(f"  _chain verified bit-identical to eog_core._eog_filter "
          f"when driven with the game's constants")
    print(f"  means  R {mean_R:+.0f}  L {mean_L:+.0f}  diff {mean_R - mean_L:+.0f} µV")
    print("  spans  " + "  ".join(f"{k} {v:.0f}" for k, v in payload["span"].items()) + " µV")
    print(f"  fig 9  calibration t={c_i0/fs:.3f}-{c_i1/fs:.3f}s "
          f"({(c_i1-c_i0)/fs:.1f}s, {(c_i1-c_i0)//n_new} polls), view "
          f"{view_lo:.2f}-{view_hi:.2f}s")
    print(f"         sigma = {sigma:.2f} µV on the filtered AMPLITUDE, span {span_amp:.0f} µV")
    print(f"         (not drawn: the detector's own sigma is {sigma_vel:.1f} µV/s on the "
          f"velocity, {sigma_vel_game:.1f} µV/s at the game's corners)")
    print(f"  cues   {len(payload['cues'])} gaze markers")
    sw = payload["detect"]["sweep"]
    print(f"  fig 10 sigma {sigma:.2f} uV, k {k_lo}-{k_hi} step {k_step} "
          f"({len(sw)} points), scored against {len(cue_t)} cues")
    for kk in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        r = sw[int(round((kk - k_lo) / k_step))]
        print(f"         k={kk:.1f}  det {r[0]:3d}  hit {r[1]:2d}/{len(cue_t)}  false {r[2]:3d}  "
              f"FPR {100*r[2]/max(1,r[0]):5.1f}%")


if __name__ == "__main__":
    main()
