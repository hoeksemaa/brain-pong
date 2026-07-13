"""Pure, hardware-free game-logic helpers for BrainPong.

Extracted from the Dash physics callback so the paddle-command application — the
piece that carried the game-start / after-every-point "phantom twitch" bug — is
unit-testable without standing up Dash or a board. No game constants are imported
here: the caller passes the zone bounds, so this stays a pure function.
"""


def advance_paddle_zone(zone_idx, command, last_seq, *, min_zone, max_zone):
    """Apply one BCI command to a paddle zone using a seq-tracked, adopt-on-first-tick rule.

    ``command`` is the command-store value ``{'command': 'LEFT'|'RIGHT'|'NEUTRAL',
    'seq': int}`` (``None``/missing tolerated). ``last_seq`` is the caller's
    last-applied seq, or ``None`` on the first tick of a rally.

    Returns ``(new_zone_idx, new_last_seq)``.

    On the FIRST tick of a rally (``last_seq is None``) the store's current seq is
    ADOPTED without moving the paddle: a command that predates the rally — a leftover
    from the previous game, or a fire during the calibration tail — must not twitch the
    paddle. This mirrors the keyboard-label "adopt the live counter without acting on
    the first tick" pattern in update_game_physics. Because the physics loop resets
    ``last_bci_seq`` to ``None`` on every New Game AND on every point (ball-clear
    rebuilds the game state), this also stops the paddle jumping at the start of each
    rally, not just at game start.

    Thereafter a strictly-new seq applies the command: LEFT/RIGHT move one slot,
    clamped to ``[min_zone, max_zone]``; NEUTRAL and a repeated seq are no-ops. A store
    reset (seq back to 0, command NEUTRAL) is therefore a harmless no-op move.
    """
    seq = (command or {}).get('seq', 0)
    if last_seq is None:                      # first tick of a rally: adopt, don't move
        return zone_idx, seq
    if seq == last_seq:                       # nothing new since last tick
        return zone_idx, last_seq
    move = (command or {}).get('command', 'NEUTRAL')
    if move == 'LEFT':
        zone_idx = max(min_zone, zone_idx - 1)
    elif move == 'RIGHT':
        zone_idx = min(max_zone, zone_idx + 1)
    return zone_idx, seq


def next_training_target(zone_idx, target, *, min_zone, max_zone):
    """Advance one player's TRAINING-mode prompt target given their paddle zone.

    ``target`` is the direction the on-field prompt is currently asking for
    (``'left'`` or ``'right'``). Once the paddle reaches the commanded edge the
    prompt flips to the opposite direction; anywhere else — including sitting on
    the *un*-commanded edge — it holds. Each player's prompt therefore loops
    left ↔ right forever, independently of the other player's.
    """
    if target == 'left' and zone_idx <= min_zone:
        return 'right'
    if target == 'right' and zone_idx >= max_zone:
        return 'left'
    return target
