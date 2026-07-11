"""Host-clock event mapping — the fix for the board-clock skew that dropped 2-player
P1 markers (see brainpong.recording.map_events_to_samples).

One X8 board's timestamp channel ran a fixed ~801 s fast; anchoring event markers to
that board clock mapped them to negative sample indices and silently dropped them, so
every 2-player P1 (.npz) came out with an empty event_samples array. map_events_to_samples
places the span and markers on the HOST clock instead, so it is immune to any board-clock
offset by construction.
"""
import pytest

from brainpong.recording import map_events_to_samples

SR = 250
START, STOP = 1000.0, 1100.0          # host times bracketing a 100 s recording
T_PULL = STOP                          # buffer pulled right at stop
N_PULLED = int((STOP - START + 2.0) * SR) + 1   # duration + 2 s pre-roll, as the recorder pulls
EVENTS = [('calib_start', 1005.0), ('play_start', 1010.5), ('p1_left', 1042.0)]


def _old_board_clock_mapping(events, board_unix_start, sr, n_span):
    """The pre-fix mapping: anchor markers to the BOARD clock (span_ts[0])."""
    out = []
    for label, t in events:
        s = int(round((t - board_unix_start) * sr))
        if 0 <= s < n_span:
            out.append((s, label))
    return out


def test_markers_land_at_correct_host_offsets():
    i0, i1, unix_start, ev_s, ev_l = map_events_to_samples(EVENTS, START, STOP, T_PULL, N_PULLED, SR)
    assert unix_start == pytest.approx(START, abs=1e-3)          # span starts at the recording start
    assert ev_l == ['calib_start', 'play_start', 'p1_left']
    assert ev_s == [1250, 2625, 10500]                          # 5.0 s, 10.5 s, 42.0 s into the span


def test_span_covers_exactly_the_recording_window():
    i0, i1, unix_start, _, _ = map_events_to_samples(EVENTS, START, STOP, T_PULL, N_PULLED, SR)
    assert (i1 - i0 + 1) == int((STOP - START) * SR) + 1        # 100 s * 250 + 1 samples


def test_skewed_board_clock_survives_here_but_broke_the_old_way():
    # NEW: host-clock mapping is unaffected by the board clock — markers survive.
    i0, i1, unix_start, ev_s, ev_l = map_events_to_samples(EVENTS, START, STOP, T_PULL, N_PULLED, SR)
    assert ev_s == [1250, 2625, 10500]
    # OLD: if the board reported timestamps +801 s fast, span_ts[0] ≈ unix_start + 801,
    # and every marker maps to a large negative index → all dropped (the observed bug).
    n_span = i1 - i0 + 1
    dropped = _old_board_clock_mapping(EVENTS, unix_start + 801.05, SR, n_span)
    assert dropped == []                                        # reproduces the silent-drop defect


def test_events_outside_the_window_are_dropped():
    events = [('before', 999.0), ('calib_start', 1005.0), ('after', 1100.5)]
    _, _, _, ev_s, ev_l = map_events_to_samples(events, START, STOP, T_PULL, N_PULLED, SR)
    assert ev_l == ['calib_start']                              # only the in-window event kept
    assert ev_s == [1250]


def test_pull_lag_shifts_all_markers_by_a_constant_only():
    # A realistic small lag between the newest sample and the time.time() pull shifts every
    # marker by the same tiny amount (here 40 ms = 10 samples), never drops or reorders them.
    _, _, _, ev_s, _ = map_events_to_samples(EVENTS, START, STOP, STOP + 0.040, N_PULLED, SR)
    base = [1250, 2625, 10500]
    assert all(abs(a - b) <= 10 for a, b in zip(ev_s, base))
    assert ev_s == sorted(ev_s)


def test_no_events_is_empty():
    _, _, _, ev_s, ev_l = map_events_to_samples([], START, STOP, T_PULL, N_PULLED, SR)
    assert ev_s == [] and ev_l == []
