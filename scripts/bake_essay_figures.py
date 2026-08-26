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

REC = pathlib.Path("data/eog/20260703-163542-playerG.npz")
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

# ── Figure 5. The ADC ────────────────────────────────────────────────────────
#
# THE RATE THE FIGURE DRAWS IS NOT THE RATE THE BOARD RUNS, and that is the
# whole design. At 250 Hz a 10 s window holds 2500 samples: the dots merge into
# the line they came from, and a reader learns nothing except that the trace is
# made of ink. So the figure holds a DIGI_VIEW_S window and draws DIGI_N_DOTS
# samples across it — an honest decimation of the real recording, at a rate
# chosen to be legible rather than accurate. `digitize.fs_real` and
# `digitize.fs_shown` are both in the JSON so the prose can name the gap.
#
# The window is the SAME at both ends of the toggle. An earlier cut zoomed the
# axis as the samples appeared, which moved the camera and the operator at once
# and left the reader unable to tell which of the two had changed the picture.
#
# THE RATE IS THE CONSTANT, not the dot count. The window has been retuned twice
# with the rate held at 10 Hz, so the rate is stated and the number of dots falls
# out of it — widening the axis adds samples rather than spreading the same ones
# further apart, which is what a real rate does.
DIGI_VIEW_S = 3.0        # seconds of recording on the axis, throughout
DIGI_FS_HZ = 10.0        # samples drawn per second -> one per 100 ms, not 250

# ── Figure 3. Bias drive ─────────────────────────────────────────────────────
#
# THE ONLY INVENTED NUMBERS IN THE SERIES. Everything from figure 5 down is the
# committed recording; this is not, and cannot be. The array only ever contains
# SRB1-referenced, bias-ON channels — the corpus has no PD_BIAS-off session — so
# the common-mode the bias loop suppresses was never measured. What follows is a
# model with stated parameters, and the figure is built so that turning it fully
# on returns the real recording exactly.
#
# THE MECHANISM. Electrode impedances and the amplifier's input impedance form a
# bridge; common-mode leaks differentially in proportion to how UNBALANCED that
# bridge is, which dominates over finite CMRR (Metting van Rijn et al., and the
# ADS1299's own CMRR of -110 dB is far too good to explain any visible hum). For
# a canthus electrode i measured against the reference earlobe:
#
#     ch_i_measured = ch_i_true + V_cm * (Z_srb1 - Z_i) / Z_in
#
# Three consequences, one per figure: bias shrinks V_cm (figure 3); SRB1
# subtraction leaves the per-channel (Z_srb1 - Z_i) term (figure 2); and R - L
# cancels Z_srb1 entirely, leaving only (Z_L - Z_R) (figure 6). Three different
# residuals, which is what stops the three figures being one animation.
#
# WHAT SETS Z_in. Not the chip — its input impedance is gigaohms and irrelevant
# here. The bottom of the bridge is the capacitance from each input node to
# ground, dominated by the unshielded electrode leads. 30-100 pF is normal for
# plain wire; 100 pF at 60 Hz is 26.5 MOhm.
#
# WHAT THIS FIGURE MUST NOT CLAIM. That the ~9 uV of 60 Hz actually present in
# the recording is what bias left behind. With bias on, this model leaks 1.7-4 uV
# — and the measured hum is several times that, is 4x larger on R than on L, and
# correlates between the channels at only +0.24, none of which looks like a
# shared common-mode. Lead-loop magnetic pickup (differential by cable geometry,
# and immune to both bias and CMRR), electrode noise and broadband noise in the
# band are all candidates. The figure shows what bias REMOVES. It says nothing
# about the composition of what remains.
# WHAT THE INTERFERENCE LOOKS LIKE. Not a pure sine. Mains carries harmonic
# distortion, and capacitive coupling makes it worse: the displacement current
# into the body is C·dV/dt, so it scales with frequency and a 3rd harmonic that is
# 2 % of the mains VOLTAGE couples in at roughly 6 % of the fundamental's current.
# The amplitudes below are that, rounded.
#
# ONLY ODD HARMONICS, because that is what a symmetric distorted waveform has,
# and they flatten the peaks rather than skewing them — hence the pi phase on the
# 3rd. Stopping at the 5th: the 7th would be ~2 % and needs finer rendering than
# it earns.
#
# NO ENVELOPE, though the phenomenon is real. Mains amplitude on a body genuinely
# wanders — coupling capacitance changes when the subject shifts — and it is
# measurable in this very recording: the 58-62 Hz Hilbert envelope has sd/mean of
# 0.22 on R and 0.42 on L, and its own power peaks at 0.061 Hz. A 30 % sinusoidal
# envelope at 0.13 Hz was tried and dropped: it got the depth and the timescale
# about right but the CHARACTER wrong. Only 19-25 % of the real envelope's variance
# sits below 0.3 Hz, so the real thing is erratic — it jumps when someone moves and
# then holds — where one slow sinusoid breathes too regularly to be believed.
#
# WHAT IS NOT MODELLED, and why. Electrode-skin noise is 1-20 uV rms with a 1/f^a
# spectrum (a between 1.5 and 2), and it dominates the thermal noise of the
# electrode impedance — but it is ALREADY IN THE RECORDING, along with the EMG,
# the drift and the lead-loop pickup. Adding it here would count it twice. These
# two figures model only the interference that gets REMOVED; everything that
# survives to figure 5 is measured.
#
# Grid frequency also wanders by tens of mHz, which is the reason figure 8's notch
# uses 58-62 Hz bands instead of a single frequency. Too slow to see in 3 s, so it
# is named in the JSON and not drawn.
INTERFERENCE_HARMONICS = (   # (order, amplitude relative to fundamental, phase)
    (1, 1.00, 0.0),
    (3, 0.07, np.pi),        # pi flattens the peaks, which is what distortion does
    (5, 0.04, 0.0),
)
# Sub-sample rendering steps the browser uses for the analytic part. The 5th
# harmonic is 300 Hz and the recording is sampled at 250, so drawing this on
# sample positions would alias it to 50 Hz. 8 steps per sample is 2 kHz, which
# gives the 300 Hz component ~6.7 points per cycle.
INTERFERENCE_SUBSTEPS = 8

BIAS_F_HZ            = 60.0        # mains, North America
BIAS_VCM_OFF_V       = 1.5         # body common-mode with the loop open, amplitude.
                                   # Kept well inside the ADS1299's input
                                   # common-mode window (AVSS+0.3 to AVDD-0.3,
                                   # i.e. ~+/-2.2 V about mid-supply) so the front
                                   # end is operable in every frame the figure draws.
BIAS_SUPPRESSION_DB  = 40.0        # what the loop buys. A DRL study measuring this
                                   # directly reports -92 dB with the loop against
                                   # -61 dB for electrodes tied to system ground,
                                   # so ~30 dB over a passive wire; 40 dB total is
                                   # comfortably in range.
BIAS_LEAD_C_PF       = 100.0       # lead capacitance to ground -> Z_in
# Prepped gel gold cups, 5-20 kOhm at 60 Hz. All four electrodes are gel on this
# rig (both canthi, both earlobes), so the mismatch is a few kOhm — not the tens
# of kOhm that dry or drying contact would give.
Z_CANTHUS_R_OHM      = 8_000.0
Z_CANTHUS_L_OHM      = 12_000.0
Z_SRB1_OHM           = 15_000.0    # the reference earlobe. Its impedance sets the
                                   # per-channel residual and cancels in R - L.
                                   # The BIAS earlobe is a different electrode and
                                   # a different failure mode: it sits inside the
                                   # loop and sets the suppression above.

# The converter, as specified. ADS1299-class AFE (SBAS499C): section 9.3.1.3.3
# for the modulator, 9.3.2.1.1 for the sinc filter, Table 4 for the bandwidth.
# Only these three numbers are stated; everything else in `adc_chain()` is
# derived from them, so the block cannot drift from the part.
ADC_F_CLK_HZ   = 2_048_000.0   # internal oscillator, nominal (SBAS499C 7.5)
ADC_MOD_DIV    = 2             # f_MOD = f_CLK / 2            (9.3.1.3.3)
ADC_MOD_ORDER  = 2             # second-order delta-sigma     (9.3.1.3.3)
ADC_SINC_ORDER = 3             # third-order sinc decimator   (9.3.2.1.1)

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


def interference_unit(t, harmonics=None):
    """The common-mode waveform, normalised to unit peak.

    One function, shared by figures 2 and 3, so the chain between them cannot
    drift: figure 2 subtracts the reference electrode's share of this and figure 3
    suppresses the whole thing, and both scale the SAME shape.

    Amplitudes are normalised so max|unit| = 1, which is what lets V_cm be quoted
    as a peak and stay inside the amplifier's input window.
    """
    h = harmonics if harmonics is not None else INTERFERENCE_HARMONICS
    return sum(a * np.sin(2 * np.pi * k * BIAS_F_HZ * t + p) for k, a, p in h)


def interference_model():
    """Normalise the harmonic amplitudes so the composite peaks at exactly 1."""
    # Several fundamental periods at fine resolution, so the peak found is true.
    t = np.arange(0, 5 / BIAS_F_HZ, 1 / (BIAS_F_HZ * 4000))
    peak = float(np.abs(interference_unit(t)).max())
    harmonics = [[int(k), round(a / peak, 6), round(p, 6)]
                 for k, a, p in INTERFERENCE_HARMONICS]
    return {
        "f_hz": BIAS_F_HZ,
        # [order, amplitude, phase], already normalised: the browser evaluates
        # sum(a·sin(2π·k·f·t + p)) and gets a unit-peak waveform.
        "harmonics": harmonics,
        "substeps": INTERFERENCE_SUBSTEPS,
        "thd_note": "Odd harmonics only, amplitudes scaled for capacitive coupling "
                    "(displacement current is C·dV/dt, so it rises with frequency). "
                    "300 Hz is above the recording's 125 Hz Nyquist, which is why "
                    "the analytic part is drawn at substeps per sample.",
        "not_modelled": "Electrode-skin noise (1-20 µV rms, 1/f^1.5-2), EMG, drift "
                        "and lead-loop pickup are already in the recording and are "
                        "not added here. Grid-frequency wander of tens of mHz — the "
                        "reason figure 8 notches 58-62 Hz rather than 60 — is too "
                        "slow to see in a 3 s window. No amplitude envelope: the "
                        "real one (sd/mean 0.22 on R, 0.42 on L, power peaking at "
                        "0.061 Hz) is erratic rather than periodic, and a single "
                        "slow sinusoid read as too tidy.",
    }


def bias_model(sigs, fs, interf):
    """Figure 3's common-mode leak, per channel, and the panel it needs.

    Returns the amplitude in µV of the 60 Hz term the bias loop REMOVES from each
    channel — leak(loop open) minus leak(loop closed) — plus the parameters it was
    derived from, so the prose can name every one of them. See the BIAS_* block
    for the model and for what this figure is not allowed to claim.

    The browser reconstructs the waveform as amplitude * sin(2*pi*f*t) off the
    absolute sample index, so nothing needs baking as an array.
    """
    z_in = 1.0 / (2 * np.pi * BIAS_F_HZ * BIAS_LEAD_C_PF * 1e-12)
    closed = 10 ** (-BIAS_SUPPRESSION_DB / 20)      # V_cm with the loop closed
    z_ch = {"ch_R": Z_CANTHUS_R_OHM, "ch_L": Z_CANTHUS_L_OHM}

    # The same waveform the browser reconstructs, so the span is measured on the
    # array the figure actually draws. Unit peak; see interference_unit().
    t = np.arange(next(iter(sigs.values())).size) / fs
    wave = interference_unit(t, harmonics=interf["harmonics"])

    ch, span = {}, {}
    for key, z in z_ch.items():
        # Signed: the leak's polarity follows which electrode is the higher
        # impedance, so a mismatch that flips sign flips the phase on that channel.
        alpha = (Z_SRB1_OHM - z) / z_in
        off = BIAS_VCM_OFF_V * alpha * 1e6          # µV, loop open
        on = off * closed                           # µV, loop closed
        ch[key] = {
            "z_ohm": z,
            "mismatch_ohm": Z_SRB1_OHM - z,
            "leak_open_uv": round(off, 2),
            "leak_closed_uv": round(on, 3),
            # What the toggle actually removes. At s=1 the injected term is zero
            # and the trace is the recording, bit for bit.
            "removed_uv": round(off - on, 2),
        }
        # Panel for the loop-OPEN state, by the same worst-slice rule as
        # everything else but applied to the constructed array.
        span[key] = y_span(sigs[key] + (off - on) * wave, fs, 0.0, DIGI_VIEW_S)

    # The anti-signal the loop drives into the body, as figure 3's middle panel
    # draws it. ONE loop produces ONE signal, but the two channels sit behind
    # different mismatches, so no single anti-signal cancels both exactly — the
    # panel is drawn at the mean of what it removes from each, and the per-channel
    # remainder is the (Z_L − Z_R) term figure 6 spends. The arithmetic applied to
    # R and L stays per-channel exact; only this display trace is averaged.
    mid = float(np.mean([ch[k]["removed_uv"] for k in ("ch_R", "ch_L")]))
    span["mid"] = y_span(-mid * wave, fs, 0.0, DIGI_VIEW_S)

    # The differential leak cancels Z_srb1 outright, so it depends only on how the
    # two canthi differ from each other. Not drawn here; figure 6 spends it.
    d_alpha = (Z_CANTHUS_L_OHM - Z_CANTHUS_R_OHM) / z_in
    return {
        "mid_uv": round(mid, 2),
        "f_hz": BIAS_F_HZ,
        "vcm_open_v": BIAS_VCM_OFF_V,
        "vcm_closed_v": round(BIAS_VCM_OFF_V * closed, 6),
        "suppression_db": BIAS_SUPPRESSION_DB,
        "lead_c_pf": BIAS_LEAD_C_PF,
        "z_in_ohm": round(z_in, 1),
        "z_srb1_ohm": Z_SRB1_OHM,
        "ch": ch,
        "span": span,
        "diff_leak_open_uv": round(BIAS_VCM_OFF_V * d_alpha * 1e6, 2),
        "diff_leak_closed_uv": round(BIAS_VCM_OFF_V * closed * d_alpha * 1e6, 3),
        "invented": True,
        "note": "Model, not measurement: the corpus has no PD_BIAS-off session, so "
                "V_cm was never recorded. Applying the loop fully returns the real "
                "recording exactly. Says nothing about what the residual is made of.",
    }


def srb1_model(sigs, fs, bias, interf):
    """Figure 2: what the amplifier pins sit at BEFORE the reference is subtracted.

    Same scenario as figure 3 with the loop still open, so the two figures share
    every parameter. For a canthus electrode i and the reference earlobe:

        V_pin_i  = V_cm·(1 − Z_i/Z_in)     + true_i
        V_srb1   = V_cm·(1 − Z_srb1/Z_in)
        ------------------------------------------------------------------
        V_pin_i − V_srb1 = V_cm·(Z_srb1 − Z_i)/Z_in + true_i

    — which is exactly figure 3's loop-open state. So `V_pin_i = (figure 3's
    opening trace) + V_srb1`, and subtracting V_srb1 lands on figure 3 exactly.
    One term, one seam, no slack.

    THE HALF-CELLS NEED NO SPOOFING. V_srb1 also carries the earlobe's own
    half-cell potential, but the recording's DC offsets — −11.1 mV on R, −38.9 mV
    on L — ARE already (electrode half-cell − earlobe half-cell). Adding the
    earlobe's offset to both channels and then subtracting it again is a no-op, so
    the term subtracted here is the common-mode alone. That the offsets in the
    recording are differences of half-cells, not absolute electrode potentials, is
    worth a sentence of prose.

    At the volts scale the two channels are indistinguishable: they differ by
    ~400 µV out of 3 V, i.e. 0.013 %. That is the figure — two electrodes that
    look like the same signal, and that signal is mains.
    """
    z_in = bias["z_in_ohm"]
    # (1 − Z_srb1/Z_in) is 0.99943 here; kept explicit because it is the same
    # divider the channels see, and dropping it would quietly break the identity
    # above.
    a_cm = BIAS_VCM_OFF_V * (1 - Z_SRB1_OHM / z_in) * 1e6      # µV

    t = np.arange(next(iter(sigs.values())).size) / fs
    wave = interference_unit(t, harmonics=interf["harmonics"])

    span = {}
    for key in ("ch_R", "ch_L"):
        pin = sigs[key] + (bias["ch"][key]["removed_uv"] + a_cm) * wave
        span[key] = y_span(pin, fs, 0.0, DIGI_VIEW_S)
    # The reference electrode gets its own panel, so it needs its own height. It
    # carries no biopotential and no drift — only the common-mode — which is
    # exactly why subtracting it removes the one and leaves the other.
    span["srb1"] = y_span(a_cm * wave, fs, 0.0, DIGI_VIEW_S)

    return {
        "subtracted_uv": round(a_cm, 1),
        "span": span,
        "invented": True,
        "note": "Same open-loop scenario as `bias`, so the two figures share every "
                "parameter and figure 2 lands on figure 3's opening frame exactly. "
                "The reference electrode's half-cell is already inside the "
                "recording's DC offsets and so is not spoofed.",
    }


def adc_chain(fs):
    """What the converter does to one sample, derived from the datasheet numbers.

    The answer to "is a sample the first of its interval, the last, or the
    average" is none of those. Each channel's second-order delta-sigma modulator
    samples at f_MOD = f_CLK / 2, and a third-order sinc filter decimates that by
    N = f_MOD / f_DR (SBAS499C 9.3.2.1.1):

        |H(z)| = |(1 - z^-N) / (1 - z^-1)|^3

    which is a boxcar of length N convolved with itself three times. So the
    impulse response is 3N-2 modulator periods wide — exactly 3 output periods,
    which is what the datasheet means by "with a step change at input, the filter
    takes 3 x tDR to settle" — and it is a BELL, peaking at 2.25x its own mean
    and tapering to zero at both ends. Consecutive samples therefore overlap
    three deep; they are not independent snapshots of adjacent slots.

    Being symmetric, its centre of mass sits half its width back, so the number
    stamped at t is an average centred 1.5 output periods EARLIER.

    FIGURE 5 DOES NOT DO THIS. It takes the plain mean of the real samples in
    each slot — a boxcar, i.e. a first-order sinc — because then the highlighted
    band tells the whole truth about where its dot came from. The real weighting
    reaches into the two neighbouring slots, which a one-slot band would lie
    about. Everything here is baked so the prose can name that gap; nothing here
    is drawn.
    """
    f_mod = ADC_F_CLK_HZ / ADC_MOD_DIV
    n = int(round(f_mod / fs))

    # The kernel itself, so support and group delay are counted rather than
    # asserted from a formula that could be mistranscribed.
    k = np.ones(n) / n
    for _ in range(ADC_SINC_ORDER - 1):
        k = np.convolve(k, np.ones(n) / n)
    support = k.size                      # 3N - 2
    group_delay = (support - 1) / 2.0     # symmetric FIR

    def mag(f):
        """|H(f)| of the sinc^ADC_SINC_ORDER decimator. (SBAS499C eq. 7.)"""
        return abs((np.sin(np.pi * f * n / f_mod)
                    / (n * np.sin(np.pi * f / f_mod))) ** ADC_SINC_ORDER)

    # -3 dB corner by bisection. Monotone from DC out to the first null at f_DR,
    # so a bisection on [tiny, f_DR) is exact to the tolerance.
    target = 10 ** (-3 / 20)
    lo, hi = 1e-6, fs * 0.999
    for _ in range(200):
        mid = (lo + hi) / 2
        if mag(mid) > target:
            lo = mid
        else:
            hi = mid
    f_3db = (lo + hi) / 2

    return {
        "part": "ADS1299-class",
        "datasheet": "SBAS499C 9.3.1.3.3, 9.3.2.1.1, Table 4",
        "f_clk_hz": ADC_F_CLK_HZ,
        "f_mod_hz": f_mod,
        "modulator_order": ADC_MOD_ORDER,
        "sinc_order": ADC_SINC_ORDER,
        "decimation_n": n,
        "filter": "third-order sinc, |H(z)| = |(1-z^-N)/(1-z^-1)|^3",
        # In output periods, which is the unit that survives a rate change.
        "support_t_dr": round(support / n, 4),
        "support_ms": round(support / f_mod * 1e3, 3),
        "group_delay_t_dr": round(group_delay / n, 4),
        "group_delay_ms": round(group_delay / f_mod * 1e3, 3),
        "peak_weight_ratio": round(float(k.max() / k.mean()), 3),
        "bw_3db_hz": round(f_3db, 2),
        "bw_3db_fraction_fdr": round(f_3db / fs, 4),
        # The first null is at the DATA RATE, not at Nyquist — so there is no
        # brick wall at fs/2 and mains passes nearly intact into the arithmetic.
        "first_null_hz": float(fs),
        "atten_nyquist_db": round(float(20 * np.log10(mag(fs / 2))), 2),
        "atten_60hz_db": round(float(20 * np.log10(mag(60.0))), 2),
    }


def y_span(x, fs, skip_s=0.0, view_s=VIEW_S):
    """Panel height that comfortably fits ANY `view_s` slice of this signal.

    Sized to the worst slice in the whole recording, not the average, so the
    trace can never overflow its panel as the window scrolls. The centre tracks
    the window mean, so the span is all the browser needs.

    A fixed ABSOLUTE axis was tried and rejected: electrode drift across 2m13s
    is 4-5x larger than anything that happens inside one slice, so a typical
    slice occupied ~4% of the panel and the glances were unreadable.

    `view_s` is a parameter because figure 5 zooms the time axis by ten and needs
    the span that goes with the narrow end. The same argument that rejected an
    absolute axis rejects keeping the 10 s span through that zoom: a 1 s slice of
    a raw channel moves a tenth of what a 10 s slice does, so the samples flatten
    into a straight row of dots and the step has nothing left to show.
    """
    x = x[int(skip_s * fs):]
    w = int(view_s * fs)
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

    # ── Figure 5 ──────────────────────────────────────────────────────────────
    # Nothing to bake but numbers: the figure decimates ch_R / ch_L in the
    # browser, and those are already in the blob.
    adc = adc_chain(fs)

    # ── Figure 3 ──────────────────────────────────────────────────────────────
    sigs_pre = {"ch_R": ch_R - mean_R, "ch_L": ch_L - mean_L}
    interf = interference_model()
    bias = bias_model(sigs_pre, fs, interf)

    # ── Figure 2 ──────────────────────────────────────────────────────────────
    srb1 = srb1_model(sigs_pre, fs, bias, interf)

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
        "subject": "playerG",
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
        # Figures 2 and 3 share this one waveform, so the chain between them
        # cannot drift. See interference_model().
        "interference": interf,
        # Figure 2. Same invented open-loop scenario as `bias`; see srb1_model().
        "srb1": srb1,
        # Figure 3. A MODEL, not a measurement — the only invented numbers in the
        # series. See the BIAS_* block and bias_model().
        "bias": bias,
        # Figure 5. Both rates, because the figure draws the wrong one on
        # purpose: `fs_real` is the board, `fs_shown` is what the figure samples
        # at. `adc` is what the converter really does to a sample and is not
        # drawn — see adc_chain().
        "digitize": {
            "fs_real": fs,
            "fs_shown": DIGI_FS_HZ,
            "view_s": DIGI_VIEW_S,
            "n_dots": int(round(DIGI_VIEW_S * DIGI_FS_HZ)),
            "slot_s": round(1.0 / DIGI_FS_HZ, 6),
            "samples_per_dot": round(fs / DIGI_FS_HZ, 3),
            "kind": "mean of the real samples in each slot (boxcar, i.e. sinc^1)",
            # Panel heights for a view_s window, same worst-slice rule as `span`
            # but over 3 s rather than 10. The top-level `span` is sized for a
            # 10 s slice, and a short slice of a raw channel moves a fraction as
            # far, so reusing it flattens the samples into a straight row of dots.
            "span": {k: y_span(sigs[k], fs, 0.0, DIGI_VIEW_S) for k in ("ch_R", "ch_L")},
            "adc": adc,
        },
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
    sr = payload["srb1"]
    print(f"  fig 2  INVENTED: subtracts {sr['subtracted_uv']/1e6:.4f} V of common-mode "
          f"(the reference earlobe), panels "
          + "  ".join(f"{k} {v/1e6:.2f} V" for k, v in sr["span"].items())
          + " -> figure 3's open state")
    b = payload["bias"]
    print(f"  fig 3  INVENTED (no PD_BIAS-off session exists): V_cm {b['vcm_open_v']:g} V "
          f"-> {b['vcm_closed_v']*1e3:g} mV at {b['suppression_db']:g} dB, "
          f"Z_in {b['z_in_ohm']/1e6:.1f} MΩ ({b['lead_c_pf']:g} pF leads)")
    for k, c in b["ch"].items():
        print(f"         {k}  Z {c['z_ohm']/1e3:g}k, mismatch {c['mismatch_ohm']/1e3:+g}k "
              f"-> leak {c['leak_open_uv']:.1f} µV open, {c['leak_closed_uv']:.2f} µV closed; "
              f"removes {c['removed_uv']:.1f} µV, panel {b['span'][k]:.0f} µV")
    print(f"         R-L leak {b['diff_leak_open_uv']:.1f} µV open -> "
          f"{b['diff_leak_closed_uv']:.2f} µV closed (Z_srb1 cancels; figure 6 spends this)")
    dg = payload["digitize"]
    print(f"  fig 5  {dg['fs_shown']:g} Hz shown, NOT the board's {dg['fs_real']} Hz — "
          f"{dg['n_dots']} dots over {dg['view_s']:g} s, one per {1e3*dg['slot_s']:g} ms "
          f"({dg['samples_per_dot']:g} real samples per dot, boxcar mean)")
    print(f"         spans at {dg['view_s']:g} s  "
          + "  ".join(f"{k} {v:.0f} (vs {payload['span'][k]:.0f} at {VIEW_S:g} s)"
                      for k, v in dg["span"].items()) + " µV")
    print(f"         ADC: sinc^{adc['sinc_order']} decimator, N={adc['decimation_n']}, "
          f"f_MOD {adc['f_mod_hz']/1e6:g} MHz — one sample is a bell-weighted mean "
          f"{adc['support_ms']:.2f} ms wide ({adc['support_t_dr']:g} t_DR),")
    print(f"         centred {adc['group_delay_ms']:.2f} ms back "
          f"({adc['group_delay_t_dr']:g} t_DR), peak weight {adc['peak_weight_ratio']:g}x mean; "
          f"-3 dB {adc['bw_3db_hz']:g} Hz ({adc['bw_3db_fraction_fdr']:g} f_DR)")
    print(f"         first null {adc['first_null_hz']:g} Hz, not Nyquist: "
          f"{adc['atten_nyquist_db']:g} dB at {fs/2:g} Hz, "
          f"{adc['atten_60hz_db']:g} dB at 60 Hz")
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
