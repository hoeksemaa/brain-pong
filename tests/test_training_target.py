"""TRAINING-mode prompt targets — the left ↔ right forever-loop contract.

In TRAINING the field shows each player a prompt ("move all the way LEFT/RIGHT");
`next_training_target` decides when that prompt flips. The contract: the prompt
flips exactly when the paddle reaches the COMMANDED edge, holds everywhere else
(including on the un-commanded edge), and therefore alternates left → right →
left … forever as the player sweeps. Each player has their own target, so these
semantics per-call are the whole 2-player story.
"""
from brainpong.game_logic import next_training_target

# The game tiles N_PANELS=5 slots; centre is 2, clamps are [0, 4].
MIN_Z, MAX_Z, CENTER = 0, 4, 2


def _next(zone, target):
    return next_training_target(zone, target, min_zone=MIN_Z, max_zone=MAX_Z)


# ── flips exactly at the commanded edge ────────────────────────────────────────────

def test_reaching_left_edge_flips_to_right():
    assert _next(MIN_Z, 'left') == 'right'


def test_reaching_right_edge_flips_to_left():
    assert _next(MAX_Z, 'right') == 'left'


# ── holds everywhere else ──────────────────────────────────────────────────────────

def test_holds_mid_field():
    assert _next(CENTER, 'left') == 'left'
    assert _next(CENTER, 'right') == 'right'


def test_one_slot_short_of_the_edge_does_not_flip():
    assert _next(MIN_Z + 1, 'left') == 'left'
    assert _next(MAX_Z - 1, 'right') == 'right'


def test_uncommanded_edge_does_not_flip():
    """Sitting on the RIGHT edge while asked to go LEFT (the state right after a
    flip, before the player starts back) must keep asking LEFT — otherwise the
    prompt would oscillate every tick at either edge."""
    assert _next(MAX_Z, 'left') == 'left'
    assert _next(MIN_Z, 'right') == 'right'


def test_stable_at_edge_across_ticks():
    """The physics loop calls this every 16 ms; a paddle parked on the commanded
    edge must flip ONCE, then the new target holds on subsequent ticks."""
    target = next_training_target(MIN_Z, 'left', min_zone=MIN_Z, max_zone=MAX_Z)
    assert target == 'right'
    assert _next(MIN_Z, target) == 'right'        # still parked left: no re-flip


# ── the forever loop ───────────────────────────────────────────────────────────────

def test_full_sweep_cycle():
    """Walk a paddle centre → left edge → right edge → left edge and check the
    prompt sequence is left, right, left — the endless training loop."""
    target = 'left'
    seen = []
    for zone in (CENTER, 1, MIN_Z,               # sweep to the left edge
                 1, CENTER, 3, MAX_Z,            # then all the way right
                 3, CENTER, 1, MIN_Z):           # and back left again
        new = _next(zone, target)
        if new != target:
            seen.append(new)
            target = new
    assert seen == ['right', 'left', 'right']
