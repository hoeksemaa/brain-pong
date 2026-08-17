import argparse
import math
import os
import random
import sys
import time
import numpy as np
from dash import Dash, dcc, html, Output, Input, State, no_update, clientside_callback, ctx
from dash.exceptions import PreventUpdate
import logging
from datetime import datetime
from pathlib import Path

# --- CLI ---
_cli_parser = argparse.ArgumentParser(
    prog='pong_game_brainflow.py',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=(
        'BrainPong — EOG eye-tracking pong on the Cerelog X8.\n'
        'Glance left/right to move your paddle (or use the keyboard), then open\n'
        'the local web UI at http://127.0.0.1:8050/.\n'
        '\n'
        'Modes:\n'
        '  (default)    Single-player on a connected board; keyboard control (P1: ←/→).\n'
        '  --eog        Single-player EOG glance-pair detector (needs board).\n'
        '  --2player    Two-player EOG.  Two boards, one per player.\n'
        '  --no-board   Skip hardware; keyboard only. Combine with --2player for\n'
        '               keyboard two-player.\n'
        '\n'
        'Controls / intention labels  (P1 = arrows: Left/Right/Up,  P2 = WASD: A/D/W):\n'
        '  Left/Right or A/D = look left / right,  Up or W = blink.\n'
        '  In keyboard modes these move the paddle. Under EOG the eyes drive the\n'
        '  paddle and the keys instead log a sample-pinned intention label into the\n'
        '  recording (p1_left/p1_right/p1_blink, p2_*) to build a labeled training set.\n'
        '\n'
        'In any mode, the dumbbell button (next to pause / new-game) enters TRAINING:\n'
        'no ball, no score — on-field prompts cue each player to sweep their paddle\n'
        'all the way left, then right, in a loop. New Game exits back to a real game.\n'
        '\n'
        'Note: --eog cannot be combined with --no-board or --2player.'
    ),
    epilog=(
        'examples:\n'
        '  python scripts/pong_game_brainflow.py --no-board             # keyboard only, no hardware\n'
        '  python scripts/pong_game_brainflow.py --no-board --2player   # keyboard two-player\n'
        '  python scripts/pong_game_brainflow.py --eog                  # single-player EOG (board)\n'
        '  python scripts/pong_game_brainflow.py --2player              # two-player EOG (board)'
    ),
)
_cli_parser.add_argument('--no-board', action='store_true',
                         help='Run without BrainFlow hardware; keyboard only (P1: arrows, P2: WASD).')
_cli_parser.add_argument('--eog', action='store_true',
                         help='Single-player EOG glance-pair detector (requires hardware).')
_cli_parser.add_argument('--2player', dest='two_player', action='store_true',
                         help='Two-player EOG: two boards, one per player (P1 arrows, P2 WASD). Exclusive with --eog.')
# NOTE: detector tuning (σ-threshold, glance-window) is done LIVE via in-game
# sliders in the browser, not CLI flags — see the 'EOG detector tuning' layout block.
# parse_known_args (not parse_args) keeps the historically lenient behavior of
# ignoring unrecognized args; add_help defaults to True so -h/--help now works.
_cli_args, _ = _cli_parser.parse_known_args()
NO_BOARD_MODE   = _cli_args.no_board
EOG_MODE        = _cli_args.eog
TWO_PLAYER_MODE = _cli_args.two_player

if EOG_MODE and NO_BOARD_MODE:
    print("ERROR: --eog requires hardware; cannot combine with --no-board.", file=sys.stderr)
    sys.exit(2)
if TWO_PLAYER_MODE and EOG_MODE:
    print("ERROR: --2player and --eog are mutually exclusive.", file=sys.stderr)
    sys.exit(2)

# --- BrainFlow ---
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Shared EOG detection core — identical for the 1-player and 2-player paths.
# Detection logic + its constants live in eog_core.py so they're testable
# without standing up Dash/the board; don't re-inline them here.
from brainpong.eog_core import (
    EOG_SIGMA_THR, GLANCE_WINDOW_S, EOG_BASELINE_S, EOG_LPF_HZ, EOG_HPF_HZ,
    eog_diff, _make_eog_state, _eog_filter, _eog_velocity, _sustained_crossing,
    _run_eog_sm, _reset_eog_st, pipeline_description, begin_play_settle,
)
from brainpong.game_logic import advance_paddle_zone, next_training_target
from brainpong.recording import save_eog_recording, map_events_to_samples

# ==============================================================================
# === 1. CONFIGURATION =========================================================
# ==============================================================================
BOARD_ID             = BoardIds.CERELOG_X8_BOARD

# Two boards — one per player — on separate serial ports. Hardcoded for now: the
# two X8s are the same model, so BrainFlow can't tell them apart over serial and
# board identity == which port. Find the live paths with:  ls /dev/cu.usbserial-*
# (they only appear while a board is plugged in). Single-player modes use P1 only.
P1_SERIAL_PORT       = "/dev/cu.usbserial-1120"   # Player 1 board (v1.2 / original)
P2_SERIAL_PORT       = "/dev/cu.usbserial-1110"   # Player 2 board — 2nd port on the hub (confirm which physical board at the paddle)

# In-game recording (one npz per player per game, data/eog/). Board VERSION is
# looked up from the port: v1.2 is always on 1120, v1.3 always on 1110 (asserted,
# since BrainFlow can't tell same-model units apart over serial).
BOARD_STREAM_BUFFER  = 450000     # ring size for start_stream; also caps the recording pull
PORT_TO_VERSION      = {P1_SERIAL_PORT: "1.2", P2_SERIAL_PORT: "1.3"}
REC_OUT_DIR          = Path(__file__).resolve().parent.parent / "data" / "eog"
REC_MONTAGE          = "2 electrodes at outer canthi (differential) + active bias on ear clip, no ground"
REC_GAIN             = 24         # Cerelog X8 default (assumed, not read from board); matches the corpus
INITIAL_BALL_SPEED_Y = -4
WAVE_YMAX_DEFAULT    = 10000  # µV; fixed waveform half-range (live slider: 5k–50k)
GAME_INTERVAL_MS     = 16
GAME_WIDTH           = 800
GAME_HEIGHT          = 800   # square play field (UI revamp). Was 600; N_PANELS still
                             # tiles WIDTH so paddle slots/width are unchanged — only the
                             # vertical extent changes (all physics scale off GAME_HEIGHT).
# Each side has N_PANELS discrete paddle slots that tile the width evenly:
# the paddle is one panel wide and snaps to panel centers. Bump N_PANELS to
# re-tile everything (width, slot centers, clamps) — keep assets/render.js's
# ghost-slot loop in sync (it mirrors this formula).
N_PANELS             = 5
PADDLE_WIDTH         = GAME_WIDTH // N_PANELS - 2
PADDLE_POSITIONS     = [GAME_WIDTH * (2 * i + 1) // (2 * N_PANELS) for i in range(N_PANELS)]
CENTER_ZONE          = N_PANELS // 2   # starting / reset slot index
MAX_ZONE             = N_PANELS - 1    # clamp ceiling for rightward moves
PADDLE_HEIGHT        = 20
BALL_RADIUS          = 10
POWERUP_RADIUS       = 14
POWERUP_FALL_SPEED   = 3
AI_REACTION_FRAMES   = 31

BCI_UPDATE_INTERVAL_MS = 100

# Serve-hold countdowns — state-level, NOT app statuses: a frame counter in the game
# state that the 16 ms physics tick decrements while the ball stays parked; paddles
# remain fully live throughout. Game/training start gets READY→SET→GO! words (ball
# launches / training prompts appear ON the GO); every post-point serve is a plain
# stationary-ball hold. render.js draws the words and plays the audio ticks from
# serve_hold / serve_hold_total / hold_kind in the game state.
START_COUNTDOWN_S = 1.5   # New Game / Training entry: READY → SET → GO!
SERVE_HOLD_S      = 1.0   # between points: stationary ball, ticks only
START_HOLD_FRAMES = round(START_COUNTDOWN_S * 1000 / GAME_INTERVAL_MS)
SERVE_HOLD_FRAMES = round(SERVE_HOLD_S      * 1000 / GAME_INTERVAL_MS)

# EOG electrode slot indices into get_eeg_channels().
# L/R inputs physically swapped PERMANENTLY 2026-07-02 (left-eye → pin2/CH2/row2, right-eye →
# pin1/CH1/row1). Slots reflect that so diff = data[ch_R]-data[ch_L] is canonical (rightward → +).
# Do NOT revert. Each player now has their OWN board with the SAME canthi montage, so P2 reads the
# SAME slots as P1 — just from board_p2. Both boards must be wired identically or P2's L/R inverts.
EOG_SLOT_L  = 1   # left  → CH2/row2  (both players, each on their own board)
EOG_SLOT_R  = 0   # right → CH1/row1
EOG_SLOT_L2 = EOG_SLOT_L   # P2: identical montage on its own board
EOG_SLOT_R2 = EOG_SLOT_R

# Detection constants (EOG_LPF/HPF_HZ, EOG_SIGMA_THR, EOG_MIN_DUR_MS,
# GLANCE_WINDOW_S, ARMED_MIN_WAIT_S, REFRACTORY_S, EOG_BASELINE_S) now live in
# eog_core. The knobs below are game-loop poll/UI timing, not the detector.
INSTRUCTIONS_S   = 5.0
EOG_POLL_S       = 0.1
EOG_SETTLE_S     = 0.4

# ==============================================================================
# === 2. RUNTIME STATE =========================================================
# ==============================================================================
board_p1      = None   # Player 1's board (also the sole board in --eog / default single-player)
board_p2      = None   # Player 2's board — only opened in --2player
sampling_rate = 0      # model-level; both X8s report the same rate


eog_state    = _make_eog_state()
eog_state_p2 = _make_eog_state()

# In-game recording state: snapshotted on New Game / Training, flushed to npz on
# Game Over (or save-and-restart on the next New Game / Training click).
# mode: 'game' | 'training' — training sessions get a 'training' tag in the npz.
_rec = {'active': False, 'start_time': None, 'players': [], 'events': [],
        'last_status': None, 'mode': 'game'}


def _poll_eog(eog_st, brd):
    """Pull latest window from board `brd`, filter → velocity, return new_sig slice or None.

    Detection runs on VELOCITY (the derivative), not amplitude: velocity is a
    high-pass, so slow drift and the high-pass filter's recovery tail self-reject
    while saccades' steep edges fire (Engbert & Kliegl). The velocity is computed
    on the full settled window (so the returned new samples have context) and only
    the last n_new are handed to the state machine. Hardware-bound (reads the
    passed-in board `brd` — one per player); the pure parts (eog_diff, _eog_filter,
    _eog_velocity) live in eog_core and are tested there.
    """
    if brd is None or eog_st['ch_L'] is None or eog_st['sr'] is None:
        return None
    sr       = eog_st['sr']
    n_settle = int(EOG_SETTLE_S * sr)
    n_new    = max(1, int(EOG_POLL_S * sr))
    data = brd.get_current_board_data(n_settle + n_new)
    if data.shape[1] < n_settle + n_new:
        return None
    diff_raw = eog_diff(data, eog_st['ch_R'], eog_st['ch_L'])
    filtered = _eog_filter(diff_raw.copy(), sr,
                           lpf_hz=eog_st.get('lpf_hz', EOG_LPF_HZ),
                           hpf_hz=eog_st.get('hpf_hz', EOG_HPF_HZ))
    vel      = _eog_velocity(filtered, sr)
    return vel[-n_new:]


# ==============================================================================
# === 3. APP LAYOUT ============================================================
# ==============================================================================
# assets/ lives at the repo root (next to the README demo media), not beside
# this script under scripts/. Dash resolves a relative assets_folder against
# the script's own dir, so 'assets' would point at the nonexistent
# scripts/assets and the clientside renderer (assets/render.js) would 404 —
# leaving a blank canvas. Pin the real absolute path instead. (Regression from
# the src/scripts restructure in #25, which moved the script but not assets/.)
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
app = Dash(__name__, assets_folder=_ASSETS_DIR)
app.title = "BrainPong — 2-Player EOG" if TWO_PLAYER_MODE else "BrainPong — EOG"

POINTS_TO_WIN = 10

_P1_LABEL = 'P1' if TWO_PLAYER_MODE else 'Player'
_P2_LABEL = 'P2' if TWO_PLAYER_MODE else 'AI'


def get_initial_game_state():
    return {
        # Bumped in app-status by New Game / Training clicks; update_game_physics —
        # the SOLE game-state writer — sees the mismatch and performs the reset
        # itself, so an in-flight stale physics write can never clobber a fresh game.
        'game_id': 0,
        'player_x': GAME_WIDTH / 2, 'ai_x': GAME_WIDTH / 2,
        'balls': [{'x': GAME_WIDTH / 2, 'y': GAME_HEIGHT / 2,
                   'vx': 0, 'vy': random.choice((-1, 1)) * abs(INITIAL_BALL_SPEED_Y)}],
        'powerups': [],
        'speed_mult': 1.0,
        'ai_zone_idx': CENTER_ZONE, 'ai_move_timer': 0,
        'zone_idx': CENTER_ZONE,    'last_label_seq': None,    'last_bci_seq': None,
        'zone_idx_p2': CENTER_ZONE, 'last_label_seq_p2': None, 'last_bci_seq_p2': None,
        # TRAINING-mode prompt targets (render.js draws them; harmless during play).
        # Both prompts always start by asking for LEFT.
        'train_target': 'left', 'train_target_p2': 'left',
        # Fresh state = game/training start → the READY/SET/GO hold. The between-point
        # reset in update_game_physics overrides these with the shorter 'serve' hold.
        'serve_hold': START_HOLD_FRAMES, 'serve_hold_total': START_HOLD_FRAMES,
        'hold_kind': 'start',
        'player_score': 0, 'ai_score': 0, 'winner': None,
    }


# ── UI assets: button icons (inline SVG data URIs; html.Img renders them crisply) ──
import urllib.parse
_ICON = lambda svg: "data:image/svg+xml," + urllib.parse.quote(svg)
_PAUSE_ICON = _ICON(
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
    "<rect x='5' y='4' width='5' height='16' rx='1.5' fill='#f4ead2'/>"
    "<rect x='14' y='4' width='5' height='16' rx='1.5' fill='#f4ead2'/></svg>")
_NEWGAME_ICON = _ICON(
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='#f4ead2' "
    "stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'>"
    "<path d='M20 12a8 8 0 1 1-2.34-5.66'/><path d='M20 3.5V8.2H15.3'/></svg>")
_TRAINING_ICON = _ICON(   # dumbbell: outer + inner plates, bar through the middle
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='#f4ead2'>"
    "<rect x='1.5' y='9' width='2.6' height='6' rx='1.1'/>"
    "<rect x='4.9' y='6.4' width='3' height='11.2' rx='1.2'/>"
    "<rect x='7' y='10.6' width='10' height='2.8' rx='1.2'/>"
    "<rect x='16.1' y='6.4' width='3' height='11.2' rx='1.2'/>"
    "<rect x='19.9' y='9' width='2.6' height='6' rx='1.1'/></svg>")

_MK = lambda v: {'label': str(v), 'style': {'color': '#a9bcd0'}}   # slider marks (dark bg)


def _tuning_row(label, control):
    return html.Div(
        [html.Label(label, style={'display': 'inline-block', 'width': '160px', 'fontSize': '12px'}),
         html.Div(control, style={'display': 'inline-block', 'width': 'min(560px, 68%)',
                                  'verticalAlign': 'middle'})],
        style={'margin': '10px 0'})


def _player_card(slot):
    """A scoreboard card. P1 = John = purple = TOP; P2 = Esther/AI = yellow = BOTTOM.
    The name field doubles as the recording subject; both p1-name/p2-name always exist
    so the recording callback's State always resolves (p2 is the AI in single-player)."""
    pid = 'p1' if slot == 1 else 'p2'
    return html.Div(
        className=f'card player {pid} dimmable',
        children=[
            dcc.Input(id=f'{pid}-name', className='nameField pixel', type='text',
                      value='', placeholder=f'Player {slot}', debounce=True, maxLength=14,
                      autoComplete='off'),
            html.Div('00', id=f'{pid}-score', className='score pixel'),
        ],
    )


app.layout = html.Div(
    className='bp bp-wrap',
    children=[
        html.Div(
            className='stage', id='game-stage',
            children=[
                html.Div(
                    className='grid',
                    children=[
                        # LEFT — P1 card / controls / P2 card
                        html.Div(className='col left', children=[
                            _player_card(1),
                            html.Div(className='controls', children=[
                                html.Button(html.Img(src=_PAUSE_ICON, alt='Pause / Resume'),
                                            id='pause-button', className='btn pause', n_clicks=0,
                                            title='Pause / Resume'),
                                html.Button(html.Img(src=_NEWGAME_ICON, alt='New Game'),
                                            id='start-button', className='btn newgame', n_clicks=0,
                                            title='New Game'),
                                html.Button(html.Img(src=_TRAINING_ICON, alt='Training Mode'),
                                            id='training-button', className='btn training', n_clicks=0,
                                            title='Training Mode'),
                            ]),
                            _player_card(2),
                        ]),
                        # CENTER — square play field
                        html.Div(className='col center', children=[
                            html.Div(className='cabinet dimmable', id='cabinet', children=[
                                html.Canvas(id='pong-game-canvas', width=GAME_WIDTH, height=GAME_HEIGHT),
                            ]),
                        ]),
                        # RIGHT — live EOG feeds (P1 purple top, P2 yellow bottom)
                        html.Div(className='col right', children=[
                            html.Div(className='card wave p1 dimmable', children=[
                                html.Div(className='screen', children=html.Canvas(id='wave-canvas-p1'))]),
                            html.Div(className='card wave p2 dimmable', children=[
                                html.Div(className='screen', children=html.Canvas(id='wave-canvas-p2'))]),
                        ]),
                    ],
                ),
                # Centered overlay text (calibration / game over / paused / start). Driven
                # clientside by renderPong from app-status-store; the controls stay bright.
                html.Div(className='game-overlay', id='game-overlay', children=[
                    html.Div([
                        html.Div(id='overlay-title', className='otitle pixel'),
                        html.Div(id='overlay-sub', className='osub'),
                        html.Div(id='overlay-hint', className='ohint'),
                    ]),
                    # CALIBRATING shows this dot only — no text; see updateOverlay.
                    html.Div(className='calib-dot'),
                ]),
                # Kept only so manage_app_flow's status-display Output resolves; not shown.
                html.H3(id='status-display', className='bp-hidden'),
            ],
        ),
        # ── Operator settings drawer: HIDDEN from players by default; toggled with the
        #    G key (handled in render.js). Keeps every existing slider id so
        #    update_settings / update_eog_tuning are byte-for-byte unchanged. ──
        html.Div(id='tuning-drawer', className='tuning', style={'display': 'none'}, children=[
            html.Div('⚙  Game & detector settings — press G to hide', className='tuning-head'),
            _tuning_row('Ball speed', dcc.Slider(
                id='ball-speed-slider', min=1, max=12, step=0.5, value=abs(INITIAL_BALL_SPEED_Y),
                marks={i: _MK(i) for i in range(1, 13, 2)})),
            # (The Velocity / Matched-filter toggle lived here. Matched filtering is
            # retired from the live game — see _poll_eog. eog_core._matched_filter is
            # kept for pipeline.replay of the 63 archived matched-mode recordings.)
            _tuning_row('Sigma threshold (×σ)', dcc.Slider(
                id='sigma-thr-slider', min=1, max=10, step=0.5, value=EOG_SIGMA_THR,
                marks={i: _MK(i) for i in range(1, 11)})),
            _tuning_row('Glance window (s)', dcc.Slider(
                id='glance-window-slider', min=0.1, max=2.0, step=0.1, value=GLANCE_WINDOW_S,
                marks={v: _MK(v) for v in (0.1, 0.5, 1.0, 1.5, 2.0)})),
            _tuning_row('High-pass (Hz)', dcc.Slider(
                id='hpf-slider', min=0.1, max=2.0, step=0.1, value=EOG_HPF_HZ,
                marks={v: _MK(v) for v in (0.1, 0.5, 1.0, 1.5, 2.0)})),
            _tuning_row('Low-pass (Hz)', dcc.Slider(
                id='lpf-slider', min=10, max=100, step=5, value=EOG_LPF_HZ,
                marks={v: _MK(v) for v in (10, 30, 50, 70, 100)})),
            _tuning_row('Waveform ±µV', dcc.Slider(
                id='wave-ymax-slider', min=5000, max=50000, step=5000, value=WAVE_YMAX_DEFAULT,
                marks={v: _MK(f'{v // 1000}k') for v in (5000, 10000, 30000, 50000)})),
        ]),
        # ── stores + intervals (unchanged) + two new waveform-display stores ──
        dcc.Store(id='settings-store',       data={'ball_speed': abs(INITIAL_BALL_SPEED_Y), 'paddle_width': PADDLE_WIDTH,
                                                    'two_player': TWO_PLAYER_MODE}),
        dcc.Store(id='eog-tuning-store',      data={'sigma_thr': EOG_SIGMA_THR, 'glance_window_s': GLANCE_WINDOW_S,
                                                    'lpf_hz': EOG_LPF_HZ, 'hpf_hz': EOG_HPF_HZ, 'detector': 'velocity',
                                                    'wave_ymax': WAVE_YMAX_DEFAULT}),
        dcc.Store(id='game-state-store',     data=get_initial_game_state()),
        dcc.Store(id='app-status-store',     data={'status': 'STARTING', 'countdown': 0, 'mode': 'game',
                                                    'game_id': 0}),
        dcc.Store(id='bci-command-store',    data={'command': 'NEUTRAL', 'seq': 0}),
        dcc.Store(id='bci-command-store-p2', data={'command': 'NEUTRAL', 'seq': 0}),
        dcc.Store(id='key-press-store',      data={'p1_last': 'None', 'p1_seq': 0, 'p2_last': 'None', 'p2_seq': 0}),
        dcc.Store(id='rec-dummy-store'),   # sink for the recording callback (side-effect only)
        dcc.Store(id='wave-store-p1'),     # display-only: filtered detector signal for the P1 feed
        dcc.Store(id='wave-store-p2'),     # display-only: filtered detector signal for the P2 feed
        dcc.Interval(id='game-interval',   interval=GAME_INTERVAL_MS,       n_intervals=0, disabled=False),
        dcc.Interval(id='bci-interval',    interval=BCI_UPDATE_INTERVAL_MS,  n_intervals=0, disabled=True),
        dcc.Interval(id='status-interval', interval=500,                     n_intervals=0),
    ],
)

# ==============================================================================
# === 4. CLIENTSIDE CALLBACKS ==================================================
# ==============================================================================
clientside_callback(
    """
    function(n_intervals) {
        var ds = window.dash_clientside = window.dash_clientside || {};
        if (!ds.label_listener_added) {
            ds.label_listener_added = true;
            ds.p1_last = 'None'; ds.p1_seq = 0;   // P1 = arrows (Left/Right/Up) — bottom paddle
            ds.p2_last = 'None'; ds.p2_seq = 0;   // P2 = WASD  (A=left, D=right, W=blink) — top paddle
            var P1 = {ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'blink'};
            var P2 = {a: 'left', d: 'right', w: 'blink'};
            document.addEventListener('keydown', function(e) {
                if (e.repeat) return;             // one label per physical press (ignore auto-repeat)
                if (e.key.indexOf('Arrow') === 0) e.preventDefault();   // stop arrow-key page scroll
                if (P1.hasOwnProperty(e.key))      { ds.p1_last = P1[e.key]; ds.p1_seq++; }
                else if (P2.hasOwnProperty(e.key)) { ds.p2_last = P2[e.key]; ds.p2_seq++; }
            });
        }
        return {p1_last: ds.p1_last, p1_seq: ds.p1_seq,
                p2_last: ds.p2_last, p2_seq: ds.p2_seq};
    }
    """,
    Output('key-press-store', 'data'),
    Input('game-interval', 'n_intervals'),
)

clientside_callback(
    """
    function(gameState, appStatus, settings) {
        if (window.dash_clientside && window.dash_clientside.renderPong) {
            window.dash_clientside.renderPong('pong-game-canvas', gameState, appStatus, settings);
        }
        return null;
    }
    """,
    Output('pong-game-canvas', 'className'),
    Input('game-state-store', 'data'),
    Input('app-status-store', 'data'),
    Input('settings-store', 'data'),
)

# Live EOG feed rendering (presentation only): draw each player's waveform canvas
# whenever its wave store updates. Glance feedback now lives in the waveform's
# green (LEFT) / red (RIGHT) bands, which replaces the old corner flash-dots.
clientside_callback(
    """
    function(data) {
        if (window.dash_clientside && window.dash_clientside.renderWave) {
            window.dash_clientside.renderWave('wave-canvas-p1', data);
        }
        return null;
    }
    """,
    Output('wave-canvas-p1', 'className'),
    Input('wave-store-p1', 'data'),
)

clientside_callback(
    """
    function(data) {
        if (window.dash_clientside && window.dash_clientside.renderWave) {
            window.dash_clientside.renderWave('wave-canvas-p2', data);
        }
        return null;
    }
    """,
    Output('wave-canvas-p2', 'className'),
    Input('wave-store-p2', 'data'),
)

# ==============================================================================
# === 5. SERVER CALLBACKS ======================================================
# ==============================================================================
@app.callback(
    Output('bci-command-store', 'data'),
    Input('bci-interval', 'n_intervals'),
    State('app-status-store', 'data'),
    prevent_initial_call=True,
)
def update_bci_command(_, app_status):
    if board_p1 is None or eog_state['ch_L'] is None:
        return no_update
    status = app_status.get('status')
    # TRAINING is a full detector consumer, exactly like PLAYING: settle on entry,
    # poll every tick, and let fires move the paddle (that's the point of training).
    if status in ('PLAYING', 'TRAINING') and \
            eog_state.get('_prev_status') not in ('PLAYING', 'TRAINING'):
        begin_play_settle(eog_state, time.time())   # mute the opening-second transient
    eog_state['_prev_status'] = status
    if status not in ('PLAYING', 'CALIBRATING', 'TRAINING'):
        return no_update
    new_sig = _poll_eog(eog_state, board_p1)
    if new_sig is None:
        return no_update
    result = _run_eog_sm(eog_state, new_sig, time.time(), label='P1')
    # Only PLAYING/TRAINING may move a paddle. During CALIBRATING the SM still runs
    # (it must accumulate the baseline), but once σ locks it can fire in the ~0.5 s
    # tail before play begins — discard that so no command reaches the store early.
    if status not in ('PLAYING', 'TRAINING'):
        return no_update
    return result if result is not None else no_update


@app.callback(
    Output('bci-command-store-p2', 'data'),
    Input('bci-interval', 'n_intervals'),
    State('app-status-store', 'data'),
    prevent_initial_call=True,
)
def update_bci_command_p2(_, app_status):
    if not TWO_PLAYER_MODE:
        return no_update
    if board_p2 is None or eog_state_p2['ch_L'] is None:
        return no_update
    status = app_status.get('status')
    if status in ('PLAYING', 'TRAINING') and \
            eog_state_p2.get('_prev_status') not in ('PLAYING', 'TRAINING'):
        begin_play_settle(eog_state_p2, time.time())   # mute the opening-second transient
    eog_state_p2['_prev_status'] = status
    if status not in ('PLAYING', 'CALIBRATING', 'TRAINING'):
        return no_update
    new_sig = _poll_eog(eog_state_p2, board_p2)
    if new_sig is None:
        return no_update
    result = _run_eog_sm(eog_state_p2, new_sig, time.time(), label='P2')
    # Only PLAYING/TRAINING may move a paddle (closes the calibration-tail fire; see P1).
    if status not in ('PLAYING', 'TRAINING'):
        return no_update
    return result if result is not None else no_update


@app.callback(
    Output('bci-command-store',    'data', allow_duplicate=True),
    Output('bci-command-store-p2', 'data', allow_duplicate=True),
    Input('start-button', 'n_clicks'),
    Input('training-button', 'n_clicks'),
    State('app-status-store', 'data'),
    prevent_initial_call=True,
)
def clear_bci_stores_on_restart(start_clicks, training_clicks, app_status):
    """Wipe both command stores to NEUTRAL/seq0 on New Game / Training (belt-and-suspenders).

    The primary guard against a start-of-rally twitch is advance_paddle_zone's
    adopt-on-first-tick rule; this second layer keeps the store itself clean so no
    command from the previous game (or a fire during the calibration tail) can linger
    in it. Pairs with _reset_eog_st resetting cmd_seq, so a new game's seq numbering
    restarts from 0. allow_duplicate: update_bci_command / _p2 also write these stores.
    """
    if not (start_clicks or training_clicks):
        return no_update, no_update
    # A redundant Training click mid-training is a no-op in manage_app_flow; be a
    # strict no-op here too — the wipe could otherwise swallow a just-fired EOG
    # command the 16 ms physics tick hasn't consumed yet (one lost glance per click).
    st = app_status or {}
    in_training_flow = (st.get('status') == 'TRAINING' or
                        (st.get('mode') == 'training' and
                         st.get('status') in ('INSTRUCTIONS', 'CALIBRATING')))
    triggered = {t['prop_id'].split('.')[0] for t in (ctx.triggered or [])}
    if 'start-button' not in triggered and in_training_flow:
        return no_update, no_update
    fresh = {'command': 'NEUTRAL', 'seq': 0}
    return dict(fresh), dict(fresh)


WAVE_WINDOW_S = 5.0     # seconds of live EOG feed shown per panel
WAVE_POINTS   = 480     # min/max-envelope downsample target sent to the canvas
WAVE_RANGE_FLOOR_UV = 400.0    # min half-range (µV) so a quiet trace doesn't zoom into noise
WAVE_RANGE_HEADROOM = 1.25     # show the tallest gaze pulse with this much headroom
WAVE_RANGE_EMA      = 0.25     # per-tick smoothing of the auto-range (0=frozen, 1=instant)
WAVE_RAIL_UV        = 187500.0 # ADS1299 ±full-scale referred to input at gain ×24
WAVE_RAIL_FRAC      = 0.02     # fraction of raw samples at ≥95% rail that flags "no contact"


def _decimate_minmax(sig, n_out):
    """Downsample to n_out (lo, hi) envelope pairs. Unlike a peak (max-|·|) decimator,
    this keeps BOTH extremes per bucket, so a tall spike is drawn at its true height in
    a thin band rather than smeared into a full-height block, and nothing is hidden.
    Presentation-only (feed display), never the detector."""
    n = len(sig)
    if n <= n_out:
        v = [round(float(x), 1) for x in sig]
        return v, v
    idx = np.linspace(0, n, n_out + 1).astype(int)
    lo, hi = [], []
    for a, b in zip(idx[:-1], idx[1:]):
        seg = sig[a:b] if b > a else sig[a:a + 1]
        lo.append(round(float(seg.min()), 1))
        hi.append(round(float(seg.max()), 1))
    return lo, hi


def _wave_payload(eog_st, brd):
    """Compact display payload for a player's live EOG feed.

    Plots the FILTERED GAZE AMPLITUDE (eog_diff → _eog_filter, in µV) — the legible
    "where the eyes moved" trace — NOT the wide-dynamic-range velocity/matched-filter
    response the detector thresholds. (The old panel plotted that response, hard-clamped
    to a fixed ±range and peak-decimated; on a strong signal the response runs ~10× the
    range, so the clamp+peak turned clean glances into a full-height "stair-step".)
    Three fixes: plot amplitude, AUTO-SCALE the range to the signal (smoothed), and use a
    MIN/MAX envelope instead of peak decimation. When the raw electrodes are railing the
    payload reports quality='railed' so the panel can show an explicit NO-CONTACT state
    instead of drawing garbage. DISPLAY ONLY — does not feed the detector, mutate detector
    decisions, or change σ (the state machine still polls independently)."""
    sr = eog_st['sr']
    if sr is None or brd is None or eog_st['ch_L'] is None:
        return None
    data = brd.get_current_board_data(int(WAVE_WINDOW_S * sr))
    if data.shape[1] < 20:
        return None
    diff = eog_diff(data, eog_st['ch_R'], eog_st['ch_L'])
    filt = _eog_filter(diff.copy(), sr,
                       lpf_hz=eog_st.get('lpf_hz', EOG_LPF_HZ),
                       hpf_hz=eog_st.get('hpf_hz', EOG_HPF_HZ))

    # ── contact quality: are the raw electrode channels sitting at the rail? ──
    raw_uv    = np.abs(data[[eog_st['ch_L'], eog_st['ch_R']]]) * 1e6
    rail_frac = float(np.mean(raw_uv > 0.95 * WAVE_RAIL_UV)) if raw_uv.size else 0.0
    sigma     = eog_st['baseline_sigma']
    if rail_frac > WAVE_RAIL_FRAC:
        quality = 'railed'
    elif sigma is None:
        quality = 'calibrating'
    else:
        quality = 'ok'

    # ── auto-scale: half-range = tallest recent gaze pulse + headroom, floored and
    #    EMA-smoothed so the trace doesn't "breathe" every tick. Railed/pre-calib
    #    frames don't poison the smoothed range. ──
    # Scale to the tallest gaze pulse so glances are never clipped. Use the 99.9th
    # percentile rather than the raw max so a single artifact sample (~1 in 1250)
    # can't blow up the range; the EMA below damps whatever slips through.
    peak   = float(np.percentile(np.abs(filt), 99.9)) if filt.size else 0.0
    target = max(WAVE_RANGE_FLOOR_UV, peak * WAVE_RANGE_HEADROOM)
    prev   = eog_st.get('wave_autorange') or WAVE_RANGE_FLOOR_UV
    if quality == 'ok':
        autorange = (1.0 - WAVE_RANGE_EMA) * prev + WAVE_RANGE_EMA * target
        eog_st['wave_autorange'] = autorange
    else:
        autorange = prev

    lo, hi = _decimate_minmax(filt, WAVE_POINTS)

    # Real detected motions, mapped onto this window's time axis. The window's right
    # edge is ~now (newest sample); a motion at wallclock t sits at
    # x = 1 - (now - t)/WAVE_WINDOW_S, so bands scroll left with the trace and drop off
    # once older than the window. Overlays every directional motion the glance-pair
    # state machine COUNTED (each arming + each return saccade) — see render.js. Prune
    # in place so fire_log stays bounded to the visible window (+ small margin).
    now = time.time()
    log = eog_st.get('fire_log')
    fires = []
    if log:
        log[:] = [f for f in log if now - f['t'] <= WAVE_WINDOW_S + 1.0]
        for f in log:
            x = 1.0 - (now - f['t']) / WAVE_WINDOW_S
            if 0.0 <= x <= 1.0:
                fires.append({'x': round(x, 4), 'dir': f['cmd']})
    return {'lo': lo, 'hi': hi, 'ymax': round(float(autorange), 1),
            'win_s': WAVE_WINDOW_S, 'quality': quality, 'fires': fires}


# Live-feed store fills, on the BCI tick (100 ms, enabled only in CALIBRATING/PLAYING),
# so each feed is live during play/calibration and holds its last frame (dimmed) while
# paused or on the game-over screen. A clientside callback draws the canvas from the store.
@app.callback(
    Output('wave-store-p1', 'data'),
    Input('bci-interval', 'n_intervals'),
    State('app-status-store', 'data'),
    prevent_initial_call=True,
)
def update_wave_store_p1(_, app_status):
    if not (EOG_MODE or TWO_PLAYER_MODE) or board_p1 is None or eog_state['ch_L'] is None:
        raise PreventUpdate
    if (app_status or {}).get('status', '') not in ('CALIBRATING', 'PLAYING', 'TRAINING'):
        raise PreventUpdate
    payload = _wave_payload(eog_state, board_p1)
    if payload is None:
        raise PreventUpdate
    return payload


@app.callback(
    Output('wave-store-p2', 'data'),
    Input('bci-interval', 'n_intervals'),
    State('app-status-store', 'data'),
    prevent_initial_call=True,
)
def update_wave_store_p2(_, app_status):
    if not TWO_PLAYER_MODE or board_p2 is None or eog_state_p2['ch_L'] is None:
        raise PreventUpdate
    if (app_status or {}).get('status', '') not in ('CALIBRATING', 'PLAYING', 'TRAINING'):
        raise PreventUpdate
    payload = _wave_payload(eog_state_p2, board_p2)
    if payload is None:
        raise PreventUpdate
    return payload


@app.callback(
    Output('settings-store', 'data'),
    Input('ball-speed-slider', 'value'),
)
def update_settings(ball_speed):
    # two_player rides along so render.js knows whether the bottom paddle is a human
    # (draw a P2 training prompt) or the AI (no prompt).
    return {'ball_speed': ball_speed, 'paddle_width': PADDLE_WIDTH,
            'two_player': TWO_PLAYER_MODE}


@app.callback(
    Output('eog-tuning-store', 'data'),
    Input('sigma-thr-slider', 'value'),
    Input('glance-window-slider', 'value'),
    Input('hpf-slider', 'value'),
    Input('lpf-slider', 'value'),
    Input('wave-ymax-slider', 'value'),
)
def update_eog_tuning(sigma_thr, glance_window, hpf_hz, lpf_hz, wave_ymax):
    """Live-apply the detector knobs from the browser controls.

    Writes straight into both players' state dicts (mutated in place). _run_eog_sm
    reads sigma_thr / glance_window_s each tick, and _poll_eog reads hpf_hz / lpf_hz
    each poll — so every knob takes effect on the next detector poll without a
    restart, and all survive recalibration (_reset_eog_st keeps them), so a New Game
    locks in whatever the controls read. Harmless in keyboard-only mode.

    `detector` is no longer a knob: the live game always runs on velocity. It stays
    in the store and in the state dict at the constant 'velocity' because every
    recording writes it to its npz and pipeline.replay reads it back.
    """
    for st in (eog_state, eog_state_p2):
        st['sigma_thr']       = sigma_thr
        st['glance_window_s'] = glance_window
        st['hpf_hz']          = hpf_hz
        st['lpf_hz']          = lpf_hz
        st['wave_ymax']       = wave_ymax
    return {'sigma_thr': sigma_thr, 'glance_window_s': glance_window,
            'hpf_hz': hpf_hz, 'lpf_hz': lpf_hz, 'detector': 'velocity',
            'wave_ymax': wave_ymax}


def _advance_training(state, bci_command, bci_command_p2, key_data):
    """One 16 ms tick of TRAINING mode.

    Paddles move EXACTLY as in play — same key seq-adopt pattern, same
    advance_paddle_zone for EOG commands (the input blocks mirror
    update_game_physics' PLAYING path byte-for-byte) — but there is no ball, no
    powerups, no AI motion, and the score never changes. Each human player has a
    prompt target ('left'/'right'); reaching the commanded edge flips it (see
    game_logic.next_training_target) and, when recording, logs a sample-pinned
    pN_target_<new-direction> event — i.e. the marker names the NEW instruction,
    so the completed sweep is the marker's opposite.
    """
    kbd_moves = NO_BOARD_MODE or not (EOG_MODE or TWO_PLAYER_MODE)
    # Belt-and-suspenders invariants, forced EVERY tick (the game_id reset in
    # update_game_physics already guarantees a fresh state on entry, but these must
    # hold unconditionally in training no matter what state arrives): never a ball,
    # never a score, and NO countdown — prompts are live from the first tick.
    state['balls'], state['powerups'] = [], []
    state['player_score'], state['ai_score'], state['winner'] = 0, 0, None
    state['serve_hold'] = 0

    # --- P1 keys + EOG (mirrors the PLAYING path) ---
    zone_idx = state.get('zone_idx', CENTER_ZONE)
    p1_seq = key_data.get('p1_seq', 0)
    prev_p1_seq = state.get('last_label_seq')
    if prev_p1_seq is None:                       # first tick: adopt without logging
        state['last_label_seq'] = p1_seq
    elif p1_seq != prev_p1_seq:
        state['last_label_seq'] = p1_seq
        intent = key_data.get('p1_last', 'None')
        if intent in ('left', 'right', 'blink'):
            _log_event(f'p1_{intent}')
            if kbd_moves and intent == 'left':    zone_idx = max(0, zone_idx - 1)
            elif kbd_moves and intent == 'right': zone_idx = min(MAX_ZONE, zone_idx + 1)
    zone_idx, state['last_bci_seq'] = advance_paddle_zone(
        zone_idx, bci_command, state.get('last_bci_seq'), min_zone=0, max_zone=MAX_ZONE)
    state['zone_idx'] = zone_idx
    state['player_x'] = PADDLE_POSITIONS[zone_idx]
    new_target = next_training_target(zone_idx, state.get('train_target', 'left'),
                                      min_zone=0, max_zone=MAX_ZONE)
    if new_target != state.get('train_target'):
        _log_event(f'p1_target_{new_target}')
        state['train_target'] = new_target

    # --- P2 (human in 2-player only; in single-player the AI paddle idles at HOME —
    #     forced every tick like the invariants above — and it gets no prompt) ---
    if not TWO_PLAYER_MODE:
        state['ai_zone_idx'] = CENTER_ZONE
        state['ai_x']        = PADDLE_POSITIONS[CENTER_ZONE]
    if TWO_PLAYER_MODE:
        zone_idx_p2 = state.get('zone_idx_p2', CENTER_ZONE)
        p2_seq = key_data.get('p2_seq', 0)
        prev_p2_seq = state.get('last_label_seq_p2')
        if prev_p2_seq is None:                   # first tick: adopt without logging
            state['last_label_seq_p2'] = p2_seq
        elif p2_seq != prev_p2_seq:
            state['last_label_seq_p2'] = p2_seq
            intent = key_data.get('p2_last', 'None')
            if intent in ('left', 'right', 'blink'):
                _log_event(f'p2_{intent}')
                if kbd_moves and intent == 'left':    zone_idx_p2 = max(0, zone_idx_p2 - 1)
                elif kbd_moves and intent == 'right': zone_idx_p2 = min(MAX_ZONE, zone_idx_p2 + 1)
        zone_idx_p2, state['last_bci_seq_p2'] = advance_paddle_zone(
            zone_idx_p2, bci_command_p2, state.get('last_bci_seq_p2'),
            min_zone=0, max_zone=MAX_ZONE)
        state['zone_idx_p2'] = zone_idx_p2
        state['ai_x']        = PADDLE_POSITIONS[zone_idx_p2]
        new_target_p2 = next_training_target(zone_idx_p2, state.get('train_target_p2', 'left'),
                                             min_zone=0, max_zone=MAX_ZONE)
        if new_target_p2 != state.get('train_target_p2'):
            _log_event(f'p2_target_{new_target_p2}')
            state['train_target_p2'] = new_target_p2

    return state


@app.callback(
    Output('game-state-store', 'data', allow_duplicate=True),
    Input('game-interval', 'n_intervals'),
    State('game-state-store', 'data'),
    State('bci-command-store', 'data'),
    State('bci-command-store-p2', 'data'),
    State('app-status-store', 'data'),
    State('key-press-store', 'data'),
    State('settings-store', 'data'),
    prevent_initial_call=True,
)
def update_game_physics(_, state, bci_command, bci_command_p2, app_status, key_data, settings):
    # A New Game / Training click bumps app-status game_id; the reset happens HERE,
    # in the sole game-state writer, so it applies race-free within one 16 ms tick
    # (an in-flight stale tick can no longer clobber a fresh game — the next tick
    # sees the id mismatch and rebuilds). Runs in ANY status so the scoreboard is
    # already fresh behind the INSTRUCTIONS/CALIBRATING overlay.
    gid = app_status.get('game_id', 0)
    if state.get('game_id', 0) != gid:
        state = get_initial_game_state()
        state['game_id'] = gid
        if app_status.get('mode') == 'training':   # training: no READY/SET/GO hold
            state.update({'serve_hold': 0, 'serve_hold_total': 0})
        return state
    if app_status.get('status') == 'TRAINING':
        return _advance_training(state, bci_command, bci_command_p2, key_data)
    if app_status.get('status') != 'PLAYING':
        return no_update

    settings    = settings or {}
    ball_speed  = settings.get('ball_speed', abs(INITIAL_BALL_SPEED_Y))
    zone_idx    = state.get('zone_idx', CENTER_ZONE)
    speed_mult  = state.get('speed_mult', 1.0)
    balls       = state.get('balls', [])
    powerups    = state.get('powerups', [])

    # --- P1 intention keys (WASD: A=left, D=right, W=blink) ---
    # Each physical press logs a sample-pinned intention label into the recording
    # (see _log_event) to build the labeled training set. In keyboard-only modes the
    # left/right keys also move the paddle; in EOG modes the EYES drive the paddle and
    # these keys are label-ONLY, so a logged label always coincides with a real eye
    # movement the player made (the whole point of collecting it).
    # Keys move the paddle only when no live EOG is driving it: keyboard-only modes
    # (default, --no-board, --no-board --2player). Under real EOG the eyes drive the
    # paddle and the keys are label-only.
    kbd_moves = NO_BOARD_MODE or not (EOG_MODE or TWO_PLAYER_MODE)
    p1_seq = key_data.get('p1_seq', 0)
    prev_p1_seq = state.get('last_label_seq')
    if prev_p1_seq is None:                       # first tick of a game: adopt the live
        state['last_label_seq'] = p1_seq          # counter without logging a pre-game press
    elif p1_seq != prev_p1_seq:
        state['last_label_seq'] = p1_seq
        intent = key_data.get('p1_last', 'None')
        if intent in ('left', 'right', 'blink'):
            _log_event(f'p1_{intent}')
            if kbd_moves and intent == 'left':    zone_idx = max(0, zone_idx - 1)
            elif kbd_moves and intent == 'right': zone_idx = min(MAX_ZONE, zone_idx + 1)

    # --- P1 EOG ---
    # adopt-on-first-tick: a command sitting in the store before this rally began (a
    # leftover from the previous game, or a fire during the calibration tail) must NOT
    # move the paddle — it is adopted without acting, mirroring the keyboard-label seq
    # handling above. Because last_bci_seq is reset to None on New Game AND after every
    # point, this also kills the start-of-rally twitch. See brainpong.game_logic.
    zone_idx, state['last_bci_seq'] = advance_paddle_zone(
        zone_idx, bci_command, state.get('last_bci_seq'), min_zone=0, max_zone=MAX_ZONE)

    state['zone_idx'] = zone_idx
    state['player_x'] = PADDLE_POSITIONS[zone_idx]

    # --- Top paddle: P2 (2-player) or AI (single-player) ---
    if TWO_PLAYER_MODE:
        zone_idx_p2 = state.get('zone_idx_p2', CENTER_ZONE)

        # --- P2 intention keys (arrows: ←=left, →=right, ↑=blink) ---
        # Under real 2-player EOG the paddle is eye-driven and these are label-only;
        # in keyboard two-player (--no-board --2player) they move P2's paddle.
        p2_seq = key_data.get('p2_seq', 0)
        prev_p2_seq = state.get('last_label_seq_p2')
        if prev_p2_seq is None:                   # first tick: adopt without logging (see P1)
            state['last_label_seq_p2'] = p2_seq
        elif p2_seq != prev_p2_seq:
            state['last_label_seq_p2'] = p2_seq
            intent = key_data.get('p2_last', 'None')
            if intent in ('left', 'right', 'blink'):
                _log_event(f'p2_{intent}')
                if kbd_moves and intent == 'left':    zone_idx_p2 = max(0, zone_idx_p2 - 1)
                elif kbd_moves and intent == 'right': zone_idx_p2 = min(MAX_ZONE, zone_idx_p2 + 1)

        # Same adopt-on-first-tick rule as P1 (see brainpong.game_logic).
        zone_idx_p2, state['last_bci_seq_p2'] = advance_paddle_zone(
            zone_idx_p2, bci_command_p2, state.get('last_bci_seq_p2'),
            min_zone=0, max_zone=MAX_ZONE)

        state['zone_idx_p2'] = zone_idx_p2
        state['ai_x']        = PADDLE_POSITIONS[zone_idx_p2]

    else:
        ai_zone_idx   = state.get('ai_zone_idx', CENTER_ZONE)
        ai_move_timer = state.get('ai_move_timer', 0)
        if ai_move_timer <= 0:
            target_ball = next(
                (b for b in sorted(balls, key=lambda b: b['y']) if b['vy'] < 0),
                balls[0] if balls else None,
            )
            if target_ball:
                bx = target_ball['x']
                ai_zone_idx = max(0, min(MAX_ZONE, int(bx / GAME_WIDTH * N_PANELS)))
            ai_move_timer = AI_REACTION_FRAMES
        else:
            ai_move_timer -= 1
        state['ai_zone_idx']   = ai_zone_idx
        state['ai_move_timer'] = ai_move_timer
        state['ai_x']          = PADDLE_POSITIONS[ai_zone_idx]

    # --- Serve hold: READY/SET/GO at game start, plain 1 s hold after every point ---
    # The ball stays parked at center (its serve velocity is already set) until the
    # counter runs out; everything above still ran, so paddles stay live and players
    # can re-position during the hold. Nothing can score and no powerup moves.
    hold = state.get('serve_hold', 0)
    if hold > 0:
        state['serve_hold'] = hold - 1
        return state

    target_speed       = ball_speed * speed_mult
    balls_remaining    = []
    new_powerup_spawns = []

    for ball in balls:
        spd = math.hypot(ball['vx'], ball['vy'])
        if spd > 0.01:
            s = target_speed / spd
            ball['vx'] *= s
            ball['vy'] *= s
        else:
            ball['vy'] = -target_speed

        ball['x'] += ball['vx']
        ball['y'] += ball['vy']

        if ball['x'] <= BALL_RADIUS or ball['x'] >= GAME_WIDTH - BALL_RADIUS:
            ball['vx'] *= -1

        # P1 paddle (bottom)
        if ball['vy'] > 0 and ball['y'] + BALL_RADIUS >= GAME_HEIGHT - PADDLE_HEIGHT:
            if abs(state['player_x'] - ball['x']) < PADDLE_WIDTH / 2 + BALL_RADIUS:
                spd = math.hypot(ball['vx'], ball['vy'])
                a = random.uniform(-math.pi / 3, math.pi / 3)
                ball['vx'] = spd * math.sin(a)
                ball['vy'] = -spd * math.cos(a)
                ball['y']  = GAME_HEIGHT - PADDLE_HEIGHT - BALL_RADIUS
                # In 2-player, spawn powerup rising toward P2
                if TWO_PLAYER_MODE and random.random() < 0.7:
                    ptype = random.choice(['fire', 'multi'])
                    new_powerup_spawns.append({
                        'x':    float(ball['x']),
                        'y':    float(GAME_HEIGHT - PADDLE_HEIGHT - BALL_RADIUS - POWERUP_RADIUS - 2),
                        'vy':   float(-POWERUP_FALL_SPEED),
                        'type': ptype,
                    })

        # P2/AI paddle (top)
        if ball['vy'] < 0 and ball['y'] - BALL_RADIUS <= PADDLE_HEIGHT:
            if abs(state['ai_x'] - ball['x']) < PADDLE_WIDTH / 2 + BALL_RADIUS:
                spd = math.hypot(ball['vx'], ball['vy'])
                a = random.uniform(-math.pi / 3, math.pi / 3)
                ball['vx'] = spd * math.sin(a)
                ball['vy'] = spd * math.cos(a)
                ball['y']  = PADDLE_HEIGHT + BALL_RADIUS
                if random.random() < 0.7:
                    ptype = random.choice(['fire', 'multi'])
                    new_powerup_spawns.append({
                        'x':    float(ball['x']),
                        'y':    float(PADDLE_HEIGHT + BALL_RADIUS + POWERUP_RADIUS + 2),
                        'vy':   float(POWERUP_FALL_SPEED),
                        'type': ptype,
                    })

        if ball['y'] - BALL_RADIUS > GAME_HEIGHT:
            state['ai_score'] += 1
            if state['ai_score'] >= POINTS_TO_WIN:
                state.update({'winner': 'P2' if TWO_PLAYER_MODE else 'AI',
                              'balls': [], 'powerups': []})
                return state
        elif ball['y'] + BALL_RADIUS < 0:
            state['player_score'] += 1
            if state['player_score'] >= POINTS_TO_WIN:
                state.update({'winner': 'P1' if TWO_PLAYER_MODE else 'Player',
                              'balls': [], 'powerups': []})
                return state
        else:
            balls_remaining.append(ball)

    if not balls_remaining:
        p, a = state['player_score'], state['ai_score']
        state = get_initial_game_state()
        state.update({
            'player_score': p, 'ai_score': a,
            'game_id': gid,   # same game — MUST carry, or the next tick re-resets it
            'balls': [{'x': GAME_WIDTH / 2, 'y': GAME_HEIGHT / 2, 'vx': 0,
                       'vy': random.choice((-1, 1)) * ball_speed}],
            # point over → the short between-point hold, not the full READY/SET/GO
            'serve_hold': SERVE_HOLD_FRAMES, 'serve_hold_total': SERVE_HOLD_FRAMES,
            'hold_kind': 'serve',
        })
        return state

    # --- Powerup physics and catch detection ---
    active_powerups = []
    multi_triggered = False

    for pu in powerups:
        pu['y'] += pu['vy']

        # Off screen (either end)
        if pu['y'] - POWERUP_RADIUS > GAME_HEIGHT:
            continue
        if pu['y'] + POWERUP_RADIUS < 0:
            continue

        caught = False

        # P1 catch (bottom)
        if (pu['y'] + POWERUP_RADIUS >= GAME_HEIGHT - PADDLE_HEIGHT and
                abs(state['player_x'] - pu['x']) < PADDLE_WIDTH / 2 + POWERUP_RADIUS):
            caught = True

        # P2 catch (top, 2-player only)
        if (not caught and TWO_PLAYER_MODE and
                pu['y'] - POWERUP_RADIUS <= PADDLE_HEIGHT and
                abs(state['ai_x'] - pu['x']) < PADDLE_WIDTH / 2 + POWERUP_RADIUS):
            caught = True

        if caught:
            if pu['type'] == 'fire':
                speed_mult = min(4.0, speed_mult * 2.0)
            elif pu['type'] == 'multi' and not multi_triggered:
                multi_triggered = True
                split = []
                for b in balls_remaining:
                    spd = math.hypot(b['vx'], b['vy'])
                    if spd < 0.01:
                        spd = target_speed
                    # Preserve the source ball's vertical direction so the split
                    # balls continue the way it was already travelling (a ball
                    # heading down splits into two heading down), instead of
                    # always launching up toward the top paddle.
                    vdir = 1.0 if b['vy'] >= 0 else -1.0
                    for _ in range(2):
                        a = random.uniform(-math.pi / 3, math.pi / 3)
                        split.append({
                            'x':  float(b['x']),
                            'y':  float(b['y']),
                            'vx': float(spd * math.sin(a)),
                            'vy': float(vdir * abs(spd * math.cos(a))),
                        })
                balls_remaining = split
            continue

        active_powerups.append(pu)

    active_powerups.extend(new_powerup_spawns)
    state['balls']      = balls_remaining
    state['powerups']   = active_powerups
    state['speed_mult'] = speed_mult

    return state


@app.callback(
    Output('app-status-store', 'data', allow_duplicate=True),
    Input('game-state-store', 'data'),
    State('app-status-store', 'data'),
    prevent_initial_call=True,
)
def check_winner(game_state, app_status):
    if not game_state or not game_state.get('winner'):
        return no_update
    if (app_status or {}).get('status') == 'PLAYING':
        return {**app_status, 'status': 'GAME_OVER', 'winner': game_state['winner']}
    return no_update


# The GAME_OVER (and calibration / paused / start) overlay is rendered clientside by
# render.js::renderPong straight from app-status-store + game-state-store, so there is
# no server-side winner-overlay callback here anymore. check_winner still drives the
# GAME_OVER transition (game logic); the overlay is pure presentation on top of it.


@app.callback(
    Output('status-display', 'children'),
    Output('app-status-store', 'data'),
    Output('bci-interval', 'disabled'),
    Output('game-interval', 'disabled'),
    Input('status-interval', 'n_intervals'),
    Input('pause-button', 'n_clicks'),
    Input('start-button', 'n_clicks'),
    Input('training-button', 'n_clicks'),
    State('app-status-store', 'data'),
    prevent_initial_call=True,
)
def manage_app_flow(status_n, pause_clicks, start_clicks, training_clicks, app_status):
    # ctx.triggered can hold MORE than one prop: under this app's constant 16 ms
    # callback churn the renderer can coalesce a button click with a 500 ms interval
    # tick into ONE dispatch, and branching on the single ctx.triggered_id then lets
    # the interval mask the click (observed live: a swallowed Training click).
    # Collect every trigger and give buttons priority — the interval's countdown work
    # just waits for its next tick.
    triggered      = {t['prop_id'].split('.')[0] for t in (ctx.triggered or [])}
    status         = app_status.get('status', 'STARTING')
    countdown      = app_status.get('countdown', 0)
    mode           = app_status.get('mode', 'game')   # 'game' | 'training' — where the
    new_status     = status                           # calibration countdown lands
    # This callback does NOT write game-state (a click's reset used to race the
    # in-flight 16 ms physics write and could be clobbered — observed live as a
    # skipped READY/SET/GO). Instead accepted clicks bump game_id and the physics
    # tick, the sole game-state writer, performs the reset itself.
    game_id        = app_status.get('game_id', 0)

    needs_eog_flow = EOG_MODE or (TWO_PLAYER_MODE and not NO_BOARD_MODE)
    # Already in training, or counting down toward it? (used by the button guards)
    in_training_flow = (status == 'TRAINING' or
                        (mode == 'training' and status in ('INSTRUCTIONS', 'CALIBRATING')))

    if 'pause-button' in triggered and pause_clicks > 0:
        # Pause is inert in training (and on the way into it): there is nothing to
        # pause — no ball, no score — and PAUSED's toggle always resumes to PLAYING,
        # which would silently eject the players from training.
        if not in_training_flow:
            new_status = 'PAUSED' if status != 'PAUSED' else 'PLAYING'
    elif 'start-button' in triggered and start_clicks > 0:
        mode     = 'game'
        game_id += 1                       # physics performs the reset on this tick
        if needs_eog_flow:
            new_status = 'INSTRUCTIONS'
            countdown  = INSTRUCTIONS_S
            states_to_reset = [eog_state, eog_state_p2] if TWO_PLAYER_MODE else [eog_state]
            for st in states_to_reset:
                _reset_eog_st(st)
        else:
            new_status = 'PLAYING'
    elif 'training-button' in triggered and training_clicks > 0:
        # Re-clicking while already in (or counting down into) training does nothing;
        # from anywhere else it behaves like New Game but lands in TRAINING (the
        # physics reset sees mode=='training' and zeroes the READY/SET/GO hold).
        if not in_training_flow:
            mode     = 'training'
            game_id += 1
            if needs_eog_flow:
                new_status = 'INSTRUCTIONS'
                countdown  = INSTRUCTIONS_S
                states_to_reset = [eog_state, eog_state_p2] if TWO_PLAYER_MODE else [eog_state]
                for st in states_to_reset:
                    _reset_eog_st(st)
            else:
                new_status = 'TRAINING'
    else:   # status-interval tick (or the pre-ctx fallback)
        if status == 'STARTING':
            new_status = 'PLAYING' if not needs_eog_flow else status
        elif status == 'INSTRUCTIONS':
            countdown -= 0.5
            if countdown <= 0:
                new_status = 'CALIBRATING'
                countdown  = EOG_BASELINE_S + 0.5
        elif status == 'CALIBRATING':
            countdown -= 0.5
            if countdown <= 0:
                new_status = 'TRAINING' if mode == 'training' else 'PLAYING'

    if new_status == 'INSTRUCTIONS':
        msg = html.Div([
            html.Div(f"Remove hands from keyboard — calibrating in {max(0, int(countdown))}s",
                     style={'fontSize': '20px', 'color': 'yellow', 'marginBottom': '6px'}),
            html.Div("1. Stare at the CENTER of the screen",   style={'fontSize': '15px', 'color': '#ccc'}),
            html.Div("2. Do NOT blink",                        style={'fontSize': '15px', 'color': '#ccc'}),
            html.Div("3. Keep your eyes completely still",     style={'fontSize': '15px', 'color': '#ccc'}),
            html.Div("4. 🔇 Do NOT talk or clench your jaw — it corrupts the signal",
                     style={'fontSize': '15px', 'color': '#ffcc33'}),
        ])
    elif new_status == 'CALIBRATING':
        who = "Both players — " if TWO_PLAYER_MODE else ""
        msg = f"Calibrating — {who}hold still, eyes forward, 🔇 no talking...  {max(0, int(countdown))}s"
    elif new_status == 'PLAYING':
        if TWO_PLAYER_MODE and NO_BOARD_MODE:
            msg = "2-PLAYER — P1: ←/→  |  P2: A/D"
        elif TWO_PLAYER_MODE:
            msg = "2-PLAYER EOG — glance to move  |  🔇 no talking  |  P1: ←/→ override  |  P2: A/D override"
        elif NO_BOARD_MODE:
            msg = "NO BOARD — keyboard only (←/→)"
        elif EOG_MODE:
            msg = "PLAYING — glance left/right to move  |  🔇 no talking  |  ←/→ to override"
        else:
            msg = "PLAYING — ←/→ keys"
    elif new_status == 'TRAINING':
        msg = "TRAINING — follow the on-field prompts (no ball, no score)"
    elif new_status == 'PAUSED':
        msg = "PAUSED"
    else:
        msg = ""

    bci_disabled  = NO_BOARD_MODE or (new_status not in ('PLAYING', 'CALIBRATING', 'TRAINING'))
    game_disabled = new_status in ('PAUSED', 'GAME_OVER')

    new_app_status = {'status': new_status, 'countdown': countdown, 'mode': mode,
                      'game_id': game_id}
    if app_status.get('winner') and new_status == 'GAME_OVER':
        new_app_status['winner'] = app_status['winner']

    return msg, new_app_status, bci_disabled, game_disabled


# ==============================================================================
# === 5b. IN-GAME RECORDING ====================================================
# ==============================================================================
# One npz per player per game (data/eog/, eog-v3 schema, extends record_eog).
# Driven by status transitions: New Game → INSTRUCTIONS starts it; CALIBRATING /
# PLAYING drop event markers; GAME_OVER pulls each board's span and saves. Only
# the 2 EOG channels are stored (CH3-8 firmware-off). Board version from the port.

def _rec_boards():
    """(slot, board, state, port) for each board to record this game — [] if none."""
    out = []
    if board_p1 is not None:
        out.append(('P1', board_p1, eog_state, P1_SERIAL_PORT))
    if TWO_PLAYER_MODE and board_p2 is not None:
        out.append(('P2', board_p2, eog_state_p2, P2_SERIAL_PORT))
    return out


def _start_recording(p1_name, p2_name, mode='game'):
    """Snapshot per-player metadata + live config at New Game / Training.

    Save-and-restart: if a recording is still in progress (a game abandoned via New
    Game without reaching GAME_OVER — or any training session, which never reaches
    GAME_OVER), flush it to disk FIRST so no electrode data is lost, then start the
    fresh block. _stop_and_save_recording is a no-op when nothing is active (first
    game of the session, or already saved at GAME_OVER), so this never double-saves.

    The tunables captured here are only a FALLBACK: each player entry keeps a live
    reference to its state dict under 'state', and _lock_recording_tuning re-reads
    them at CALIBRATING, which is the value that reaches the npz."""
    if _rec['active']:
        _stop_and_save_recording()
    boards = _rec_boards()
    if not boards:
        return
    _rec['mode'] = mode
    names = {'P1': (p1_name or 'P1').strip() or 'P1',
             'P2': (p2_name or 'P2').strip() or 'P2'}
    players = [{
        'slot': slot, 'board': brd, 'port': port, 'state': st,
        'version': PORT_TO_VERSION.get(port, '?'), 'subject': names[slot],
        'ch_L': st['ch_L'], 'ch_R': st['ch_R'], 'sr': st['sr'],
        'sigma_thr': st.get('sigma_thr'), 'hpf_hz': st.get('hpf_hz'),
        'lpf_hz': st.get('lpf_hz'), 'glance_window_s': st.get('glance_window_s'),
        'detector': st.get('detector', 'velocity'),
    } for slot, brd, st, port in boards]
    _rec['active']     = True
    _rec['start_time'] = time.time()
    _rec['players']    = players
    _rec['events']     = []
    summary = ', '.join(f"{p['slot']}={p['subject']}(v{p['version']})" for p in players)
    print(f"[rec] started ({_rec['mode']}) — {len(players)} player(s): {summary}")


def _lock_recording_tuning():
    """Freeze each player's detector config into the recording at CALIBRATING.

    sigma_thr / hpf_hz / lpf_hz / glance_window_s / detector are LIVE slider values
    (update_eog_tuning writes them straight into the state dict). _start_recording runs
    at INSTRUCTIONS, so a slider moved between New Game and the calibration countdown
    was stored with the value it had BEFORE the move. pipeline.replay trusts these
    fields to reconstruct a session, so a stale one silently replays the game under
    settings it was never played with.

    Calibration is the instant that makes them true: σ is measured under exactly this
    filter chain and detector, and the whole game then runs against that σ. (A slider
    moved mid-game still changes live behaviour without changing the stored value —
    but lpf/hpf/detector changes already invalidate the locked σ, so the calibration-
    time config is the one the recording should carry.)"""
    if not _rec['active']:
        return
    changed = []
    for p in _rec['players']:
        st = p.get('state')
        if st is None:                      # no live state (shouldn't happen) → keep the
            continue                        # INSTRUCTIONS fallback rather than blank it
        locked = {'sigma_thr': st.get('sigma_thr'), 'hpf_hz': st.get('hpf_hz'),
                  'lpf_hz': st.get('lpf_hz'),
                  'glance_window_s': st.get('glance_window_s'),
                  'detector': st.get('detector', 'velocity')}
        if any(p.get(k) != v for k, v in locked.items()):
            changed.append(p['slot'])
        p.update(locked)
    if changed:
        print(f"[rec] tuning re-locked at calibration for {', '.join(changed)} "
              f"(sliders moved during the countdown)")


def _log_event(label):
    if _rec['active']:
        _rec['events'].append((label, time.time()))


def _stop_and_save_recording():
    """Flush each recording board's span to npz. Robust to per-board errors."""
    if not _rec['active']:
        return
    _rec['active'] = False
    start_t, stop_t = _rec['start_time'], time.time()
    n_players = len(_rec['players'])
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')   # shared → the two files are linkable
    for p in _rec['players']:
        try:
            _save_one_recording(p, start_t, stop_t, n_players, stamp)
        except Exception as e:
            print(f"[rec] save FAILED for {p['slot']} ({p['subject']}): {e}")


def _save_one_recording(p, start_t, stop_t, n_players, stamp):
    brd, sr = p['board'], p['sr']
    n_pull  = min(BOARD_STREAM_BUFFER, int((max(0.0, stop_t - start_t) + 2.0) * sr) + 1)
    full    = brd.get_current_board_data(n_pull)          # all rows, a fresh copy
    t_pull  = time.time()                                 # host time of the newest pulled sample
    if full.shape[1] == 0:
        print(f"[rec] no samples for {p['slot']} ({p['subject']}) — skipped")
        return
    # Place the span + markers on the HOST clock, never the board's timestamp channel:
    # one X8 ran a fixed ~801 s fast, which (via the board clock) mapped every P1 marker
    # to a negative sample index and silently dropped it in 2-player mode. This is immune
    # to any board-clock skew and gives the two players' files a shared timebase.
    i0, i1, unix_start, ev_s, ev_l = map_events_to_samples(
        _rec['events'], start_t, stop_t, t_pull, full.shape[1], sr)
    span = full[:, i0:i1 + 1]
    eeg  = np.vstack([np.ascontiguousarray(span[p['ch_L']].astype(np.float64)),
                      np.ascontiguousarray(span[p['ch_R']].astype(np.float64))])   # (2,N)
    n_samp = eeg.shape[1]
    detector = p.get('detector', 'velocity')
    training = _rec.get('mode') == 'training'
    session  = 'training-mode' if training else 'pong'
    notes = (f"in-game {session} recording ({n_players}-player); board v{p['version']} on "
             f"{p['port']}; CH3-8 firmware-off so only the EOG pair stored "
             f"(row0=ch_L, row1=ch_R). {pipeline_description(detector)} "
             f"Serve-hold: the first {START_COUNTDOWN_S:.1f}s after play_start is the "
             f"READY/SET/GO countdown (ball parked); each post-point serve holds "
             f"{SERVE_HOLD_S:.1f}s with the ball parked at center. Training has no "
             f"countdown — prompts are live from train_start.")
    if training:
        notes += (" Training mode: no ball; on-field prompts cue full-width paddle sweeps."
                  " pN_target_<dir> events mark each prompt flip (the NEW instruction;"
                  " the just-completed sweep is its opposite); both prompts start LEFT"
                  " at train_start.")
    path = save_eog_recording(
        REC_OUT_DIR, p['subject'], eeg, unix_start, sr,
        gain=REC_GAIN, board=f"CERELOG_X8 unit:v{p['version']}", montage=REC_MONTAGE,
        notes=notes, tags=['in-game', f'{n_players}-player', f"board-v{p['version']}",
                           f"detector-{detector}"] + (['training'] if training else []),
        ch_L=0, ch_R=1, n_players=n_players, board_version=p['version'],
        serial_port=p['port'], player_slot=p['slot'], sigma_thr=p['sigma_thr'],
        hpf_hz=p['hpf_hz'], lpf_hz=p['lpf_hz'], glance_window_s=p['glance_window_s'],
        detector=detector, event_samples=ev_s, event_labels=ev_l, stamp=stamp)
    print(f"[rec] saved {p['slot']} ({p['subject']}) → {path}  [{n_samp} samp, {n_samp / sr:.1f}s]")


@app.callback(
    Output('rec-dummy-store', 'data'),
    Input('app-status-store', 'data'),
    State('p1-name', 'value'),
    State('p2-name', 'value'),
    prevent_initial_call=True,
)
def manage_recording(app_status, p1_name, p2_name):
    """Drive recording off status transitions. No-op without a board to record."""
    if NO_BOARD_MODE or not (EOG_MODE or TWO_PLAYER_MODE):
        return no_update
    status = (app_status or {}).get('status')
    mode   = (app_status or {}).get('mode', 'game')
    prev   = _rec['last_status']
    if status == prev:
        # A mid-INSTRUCTIONS mode switch (Training clicked during a game's countdown,
        # or New Game during training's) flips app-status mode WITHOUT a status
        # transition, so the dedup above would keep _rec['mode'] stale and the npz
        # would be saved with the wrong training/game identity. Retag the live block —
        # the countdown restarted with the click, so its content matches the new mode.
        if status == 'INSTRUCTIONS' and _rec['active'] and mode != _rec['mode']:
            _rec['mode'] = mode
            print(f"[rec] mode switched mid-countdown — block retagged '{mode}'")
        return no_update
    _rec['last_status'] = status
    if status == 'INSTRUCTIONS':
        _start_recording(p1_name, p2_name, mode=mode)
    elif status == 'CALIBRATING' and prev == 'INSTRUCTIONS':
        _lock_recording_tuning()           # freeze the live config σ is measured under
        _log_event('calib_start')
    elif status == 'PLAYING' and prev == 'CALIBRATING':
        _log_event('play_start')
    elif status == 'TRAINING' and prev == 'CALIBRATING':
        _log_event('train_start')                  # both prompts start by asking LEFT
        _log_event('p1_target_left')
        if TWO_PLAYER_MODE:
            _log_event('p2_target_left')
    elif status == 'GAME_OVER':
        _stop_and_save_recording()
    return no_update


# ==============================================================================
# === 6. MAIN ==================================================================
# ==============================================================================
def main():
    global board_p1, board_p2, sampling_rate

    if NO_BOARD_MODE:
        mode_msg = ("2-PLAYER NO-BOARD: P1=←/→, P2=A/D." if TWO_PLAYER_MODE
                    else "NO-BOARD MODE: keyboard only (←/→).")
        print(mode_msg)
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        print("Open http://127.0.0.1:8050/ in your browser.")
        app.run(debug=False, use_reloader=False)
        return

    if TWO_PLAYER_MODE:
        print("=" * 60)
        print("2-PLAYER EOG MODE — two independent boards, one per player")
        print(f"P1 (bottom, green): board {P1_SERIAL_PORT}, canthi → ch1–2.")
        print(f"P2 (top,    red  ): board {P2_SERIAL_PORT}, canthi → ch1–2 (own board).")
        print("Each board is fully independent: its own 2 canthi electrodes, its own")
        print("SRB1 reference, its own active bias electrode. Do NOT share electrodes.")
        print("Tip: run the laptop on BATTERY so the two boards don't couple via mains ground.")
        print("Both players sit still together for the calibration countdown — each")
        print("board measures its OWN baseline σ separately.")
        print("=" * 60)
    elif EOG_MODE:
        print("=" * 60)
        print("EOG MODE: glance-pair detector active.")
        print("Glance left→center to move left; right→center to move right.")
        print("Sit still for the first 3 s (baseline calibration).")
        print("=" * 60)

    def _open_board(port, label):
        p             = BrainFlowInputParams()
        p.timeout     = 15
        p.serial_port = port
        b = BoardShim(BOARD_ID, p)
        print(f"Connecting to {label} board on {port} ...")
        b.prepare_session()
        return b

    try:
        # Player 1 board — also the only board in --eog / default single-player.
        board_p1 = _open_board(P1_SERIAL_PORT, "P1")
        sampling_rate    = BoardShim.get_sampling_rate(BOARD_ID)
        all_eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
        print(f"P1 board connected. Sampling rate: {sampling_rate} Hz")

        if EOG_MODE or TWO_PLAYER_MODE:
            eog_state['sr']          = sampling_rate
            eog_state['ch_L']        = all_eeg_channels[EOG_SLOT_L]
            eog_state['ch_R']        = all_eeg_channels[EOG_SLOT_R]
            print(f"P1 EOG channels: ch_L={eog_state['ch_L']}  ch_R={eog_state['ch_R']}  "
                  f"σ_thr={eog_state['sigma_thr']:.2f}×  glance_window={eog_state['glance_window_s']:.2f}s  "
                  f"(adjust live via the browser sliders)")

        if TWO_PLAYER_MODE:
            # Player 2's board: a SECOND, independent X8 on its own serial port. Two
            # same-model BoardShim coexist in ONE process because BrainFlow keys its
            # session registry on (board_id, serial_port), not board_id alone.
            board_p2 = _open_board(P2_SERIAL_PORT, "P2")
            eog_state_p2['sr']          = sampling_rate     # same model → same rate
            eog_state_p2['ch_L']        = all_eeg_channels[EOG_SLOT_L2]
            eog_state_p2['ch_R']        = all_eeg_channels[EOG_SLOT_R2]
            print(f"P2 board connected. EOG channels: ch_L={eog_state_p2['ch_L']}  "
                  f"ch_R={eog_state_p2['ch_R']}  σ_thr={eog_state_p2['sigma_thr']:.2f}×  "
                  f"glance_window={eog_state_p2['glance_window_s']:.2f}s")

        print("Starting data stream(s)...")
        board_p1.start_stream(BOARD_STREAM_BUFFER)
        if board_p2 is not None:
            board_p2.start_stream(BOARD_STREAM_BUFFER)
        time.sleep(1.0)

        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        print("Open http://127.0.0.1:8050/ in your browser.")
        if TWO_PLAYER_MODE:
            print("Both players sit still for baseline calibration, then glance to play.")
        elif EOG_MODE:
            print("Sit still for 3 s — baseline calibrating, then glance to play.")
        app.run(debug=False, use_reloader=False)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        for _lbl, _b in (("P1", board_p1), ("P2", board_p2)):
            if _b is not None and _b.is_prepared():
                print(f"Stopping {_lbl} stream and releasing session.")
                try:
                    _b.stop_stream()
                except Exception:
                    pass
                _b.release_session()


if __name__ == "__main__":
    main()
