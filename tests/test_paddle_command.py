"""Paddle-command application — the game-start / after-every-point twitch guard.

Regression tests for the bug where a command sitting in the command store *before*
a rally began (a leftover from the previous game, or a fire during the ~0.5 s
calibration tail) was applied to the paddle on the first PLAYING physics tick,
snapping it one slot sideways. The physics loop resets its per-rally seq tracker
(`last_bci_seq`) to None on every New Game and after every point, while the store is
not reset — so a stale command read as "new" and moved the paddle.

`advance_paddle_zone` fixes this by ADOPTING the store's current seq on the first
tick of a rally (last_seq is None) without moving, mirroring the keyboard-label
"adopt without acting" pattern. These tests pin that contract, plus the ordinary
move/clamp behaviour so the fix can't silently break real glances.
"""
from brainpong.game_logic import advance_paddle_zone

# The game tiles N_PANELS=5 slots; centre is 2, clamps are [0, 4].
MIN_Z, MAX_Z, CENTER = 0, 4, 2


def _adopt(zone, cmd, last):
    return advance_paddle_zone(zone, cmd, last, min_zone=MIN_Z, max_zone=MAX_Z)


# ── the headline: a pre-rally command must NOT twitch the paddle ──────────────────

def test_leftover_command_does_not_move_on_first_tick():
    """First tick of a rally with a stale {LEFT, seq:7} in the store: adopt, don't move."""
    stale = {'command': 'LEFT', 'seq': 7}
    zone, last = _adopt(CENTER, stale, None)     # last_bci_seq is None at rally start
    assert zone == CENTER                         # paddle stayed put — no twitch
    assert last == 7                              # ...but the seq is now adopted


def test_after_point_reset_does_not_replay_last_command():
    """After a point, last_bci_seq is reset to None while the store still holds the
    last in-rally command. The next tick must adopt it, not re-apply it."""
    store = {'command': 'RIGHT', 'seq': 8}
    zone, last = _adopt(CENTER, store, None)      # None == just after get_initial_game_state
    assert zone == CENTER                          # no phantom move at the new serve
    assert last == 8


def test_genuine_new_command_still_moves_after_adopt():
    """The fix must not swallow real glances: once the first tick has adopted, a
    strictly-new seq moves the paddle."""
    zone, last = _adopt(CENTER, {'command': 'LEFT', 'seq': 7}, None)   # adopt
    assert (zone, last) == (CENTER, 7)
    zone, last = _adopt(zone, {'command': 'LEFT', 'seq': 8}, last)     # real new fire
    assert zone == CENTER - 1
    assert last == 8
    zone, last = _adopt(zone, {'command': 'RIGHT', 'seq': 9}, last)
    assert zone == CENTER                                              # back to centre
    assert last == 9


def test_repeated_seq_is_a_noop():
    zone, last = _adopt(CENTER, {'command': 'RIGHT', 'seq': 3}, 3)     # same seq as last
    assert zone == CENTER and last == 3


def test_neutral_command_is_a_noop():
    zone, last = _adopt(CENTER, {'command': 'NEUTRAL', 'seq': 4}, 3)
    assert zone == CENTER and last == 4


def test_clamps_at_edges():
    # Rightward past the right edge stays clamped.
    zone, last = _adopt(MAX_Z, {'command': 'RIGHT', 'seq': 2}, 1)
    assert zone == MAX_Z
    # Leftward past the left edge stays clamped.
    zone, last = _adopt(MIN_Z, {'command': 'LEFT', 'seq': 2}, 1)
    assert zone == MIN_Z


def test_store_reset_to_seq0_is_harmless():
    """New Game clears the store to {NEUTRAL, seq:0}. Even if last_seq were somehow
    non-None (it isn't — it's reset to None too), a NEUTRAL is a no-op."""
    zone, last = _adopt(CENTER, {'command': 'NEUTRAL', 'seq': 0}, 5)
    assert zone == CENTER and last == 0


def test_missing_or_none_command_tolerated():
    zone, last = _adopt(CENTER, None, None)
    assert zone == CENTER and last == 0
    # empty dict → seq defaults to 0 (differs from last=2) but command defaults to
    # NEUTRAL, so the paddle does not move and the seq is adopted as 0.
    zone, last = _adopt(CENTER, {}, 2)
    assert zone == CENTER and last == 0
