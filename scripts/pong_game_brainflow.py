import argparse
import math
import random
import sys
import time
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Output, Input, State, no_update, clientside_callback, ctx
from dash.exceptions import PreventUpdate
import logging

# --- CLI ---
_cli_parser = argparse.ArgumentParser(add_help=False)
_cli_parser.add_argument('--no-board', action='store_true',
                         help='Run without BrainFlow hardware (keyboard-only).')
_cli_parser.add_argument('--eog', action='store_true',
                         help='Single-player EOG glance-pair detector (requires hardware).')
_cli_parser.add_argument('--2player', dest='two_player', action='store_true',
                         help='Two-player EOG: P1=ch1-2, P2=ch3-4. Exclusive with --eog.')
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
    EOG_SIGMA_THR, EOG_BASELINE_S,
    eog_diff, _make_eog_state, _eog_filter, _sustained_crossing,
    _run_eog_sm, _reset_eog_st,
)

# ==============================================================================
# === 1. CONFIGURATION =========================================================
# ==============================================================================
BOARD_ID             = BoardIds.CERELOG_X8_BOARD
INITIAL_BALL_SPEED_Y = -4
GAME_INTERVAL_MS     = 16
GAME_WIDTH           = 800
GAME_HEIGHT          = 600
PADDLE_WIDTH         = GAME_WIDTH // 3 - 2
PADDLE_POSITIONS     = [GAME_WIDTH // 6, GAME_WIDTH // 2, 5 * GAME_WIDTH // 6]
PADDLE_HEIGHT        = 20
BALL_RADIUS          = 10
POWERUP_RADIUS       = 14
POWERUP_FALL_SPEED   = 3
AI_REACTION_FRAMES   = 31

BCI_UPDATE_INTERVAL_MS = 100

# EOG electrode slot indices into get_eeg_channels()
EOG_SLOT_L  = 0   # P1 left
EOG_SLOT_R  = 1   # P1 right
EOG_SLOT_L2 = 2   # P2 left
EOG_SLOT_R2 = 3   # P2 right

# Detection constants (EOG_LPF/HPF_HZ, EOG_SIGMA_THR, EOG_MIN_DUR_MS,
# GLANCE_WINDOW_S, ARMED_MIN_WAIT_S, REFRACTORY_S, EOG_BASELINE_S) now live in
# eog_core. The knobs below are game-loop poll/UI timing, not the detector.
INSTRUCTIONS_S   = 5.0
EOG_POLL_S       = 0.1
EOG_SETTLE_S     = 0.4

# ==============================================================================
# === 2. RUNTIME STATE =========================================================
# ==============================================================================
board         = None
sampling_rate = 0


eog_state    = _make_eog_state()
eog_state_p2 = _make_eog_state()


def _poll_eog(eog_st):
    """Pull latest window from board, filter, return new_sig slice or None.

    Hardware-bound (reads the global `board`), so it stays here; the pure parts
    (eog_diff, _eog_filter) live in eog_core and are tested there.
    """
    if board is None or eog_st['ch_L'] is None or eog_st['sr'] is None:
        return None
    sr       = eog_st['sr']
    n_settle = int(EOG_SETTLE_S * sr)
    n_new    = max(1, int(EOG_POLL_S * sr))
    data = board.get_current_board_data(n_settle + n_new)
    if data.shape[1] < n_settle + n_new:
        return None
    diff_raw = eog_diff(data, eog_st['ch_R'], eog_st['ch_L'])
    filtered = _eog_filter(diff_raw.copy(), sr)
    return filtered[-n_new:]


# ==============================================================================
# === 3. APP LAYOUT ============================================================
# ==============================================================================
app = Dash(__name__, assets_folder='assets')
app.title = "BrainPong — 2-Player EOG" if TWO_PLAYER_MODE else "BrainPong — EOG"

POINTS_TO_WIN = 10

_P1_LABEL = 'P1' if TWO_PLAYER_MODE else 'Player'
_P2_LABEL = 'P2' if TWO_PLAYER_MODE else 'AI'


def get_initial_game_state():
    return {
        'player_x': GAME_WIDTH / 2, 'ai_x': GAME_WIDTH / 2,
        'balls': [{'x': GAME_WIDTH / 2, 'y': GAME_HEIGHT / 2,
                   'vx': 0, 'vy': INITIAL_BALL_SPEED_Y}],
        'powerups': [],
        'speed_mult': 1.0,
        'ai_zone_idx': 1, 'ai_move_timer': 0,
        'zone_idx': 1,    'prev_key': 'None',    'last_bci_seq': 0,
        'zone_idx_p2': 1, 'prev_key_p2': 'None', 'last_bci_seq_p2': 0,
        'player_score': 0, 'ai_score': 0, 'winner': None,
    }


app.layout = html.Div(
    id='main-container',
    style={'backgroundColor': '#111', 'color': '#DDD', 'fontFamily': 'monospace', 'textAlign': 'center'},
    children=[
        html.H1("BrainPong — 2-Player EOG" if TWO_PLAYER_MODE else "BrainPong — EOG"),
        html.Div([
            html.Button('Pause / Resume', id='pause-button', n_clicks=0, style={'marginRight': '20px'}),
            html.Button('Start New Game', id='start-button', n_clicks=0),
        ], style={'marginBottom': '10px'}),
        html.H3(id='status-display', style={'fontSize': '24px', 'color': 'yellow', 'minHeight': '80px'}),
        html.Div(
            style={'display': 'inline-flex', 'alignItems': 'stretch'},
            children=[
                html.Div(
                    style={'position': 'relative'},
                    children=[
                        html.Div(
                            html.Canvas(id='pong-game-canvas', width=GAME_WIDTH, height=GAME_HEIGHT),
                            style={'border': '2px solid #555'},
                        ),
                        html.Div(id='winner-overlay', style={'display': 'none'}),
                    ],
                ),
                html.Div(
                    style={
                        'display': 'flex', 'flexDirection': 'column',
                        'justifyContent': 'space-between',
                        'paddingLeft': '18px', 'paddingTop': '8px', 'paddingBottom': '8px',
                    },
                    children=[
                        html.Div([
                            html.Div(_P2_LABEL, style={'fontSize': '13px', 'color': '#888', 'marginBottom': '2px'}),
                            html.Div(id='ai-score-display', children='0',
                                     style={'fontSize': '32px', 'color': '#ff5252', 'fontWeight': 'bold'}),
                        ]),
                        html.Div([
                            html.Div(_P1_LABEL, style={'fontSize': '13px', 'color': '#888', 'marginBottom': '2px'}),
                            html.Div(id='player-score-display', children='0',
                                     style={'fontSize': '32px', 'color': '#33ff66', 'fontWeight': 'bold'}),
                        ]),
                    ],
                ),
            ],
        ),
        html.Div(
            style={'width': '800px', 'margin': '15px auto', 'textAlign': 'left',
                   'padding': '10px', 'border': '1px solid #333', 'borderRadius': '6px'},
            children=[html.Div([
                html.Label('Ball Speed', style={'display': 'inline-block', 'width': '120px'}),
                html.Div(
                    dcc.Slider(
                        id='ball-speed-slider', min=1, max=12, step=0.5,
                        value=abs(INITIAL_BALL_SPEED_Y),
                        marks={i: {'label': str(i), 'style': {'color': 'black', 'fontWeight': 'bold'}}
                               for i in range(1, 13, 2)},
                    ),
                    style={'display': 'inline-block', 'width': '640px', 'verticalAlign': 'middle'},
                ),
            ])],
        ),
        # P1 EOG live plot
        html.Div(
            [
                html.Div('Player 1 — HEOG' if TWO_PLAYER_MODE else '',
                         style={'color': '#33ff66', 'fontSize': '13px', 'marginBottom': '2px'}),
                dcc.Graph(id='eog-live-plot', config={'displayModeBar': False}),
            ],
            style={'width': '800px', 'margin': '0 auto',
                   'display': 'block' if (EOG_MODE or TWO_PLAYER_MODE) else 'none'},
        ),
        # P2 EOG live plot (2-player only)
        html.Div(
            [
                html.Div('Player 2 — HEOG',
                         style={'color': '#ff5252', 'fontSize': '13px', 'marginBottom': '2px', 'marginTop': '10px'}),
                dcc.Graph(id='eog-live-plot-p2', config={'displayModeBar': False}),
            ],
            style={'width': '800px', 'margin': '0 auto',
                   'display': 'block' if TWO_PLAYER_MODE else 'none'},
        ),
        dcc.Store(id='settings-store',       data={'ball_speed': abs(INITIAL_BALL_SPEED_Y), 'paddle_width': PADDLE_WIDTH}),
        dcc.Store(id='game-state-store',     data=get_initial_game_state()),
        dcc.Store(id='app-status-store',     data={'status': 'STARTING', 'countdown': 0}),
        dcc.Store(id='bci-command-store',    data={'command': 'NEUTRAL', 'seq': 0}),
        dcc.Store(id='bci-command-store-p2', data={'command': 'NEUTRAL', 'seq': 0}),
        dcc.Store(id='key-press-store',      data={'key': 'None', 'key_p2': 'None'}),
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
        if (!window.dash_clientside) { window.dash_clientside = {}; }
        if (!window.dash_clientside.key_listener_added) {
            window.dash_clientside.key_listener_added = true;
            window.dash_clientside.current_key    = 'None';
            window.dash_clientside.current_key_p2 = 'None';
            document.addEventListener('keydown', function(e) {
                if (e.key === 'a' || e.key === 'd') {
                    window.dash_clientside.current_key = e.key;
                }
                if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                    e.preventDefault();
                    window.dash_clientside.current_key_p2 = e.key;
                }
            });
            document.addEventListener('keyup', function(e) {
                if (e.key === 'a' || e.key === 'd') {
                    window.dash_clientside.current_key = 'None';
                }
                if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                    window.dash_clientside.current_key_p2 = 'None';
                }
            });
        }
        return {key: window.dash_clientside.current_key,
                key_p2: window.dash_clientside.current_key_p2};
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
    if board is None or eog_state['ch_L'] is None:
        return no_update
    if app_status.get('status') not in ('PLAYING', 'CALIBRATING'):
        return no_update
    new_sig = _poll_eog(eog_state)
    if new_sig is None:
        return no_update
    result = _run_eog_sm(eog_state, new_sig, time.time(), label='P1')
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
    if board is None or eog_state_p2['ch_L'] is None:
        return no_update
    if app_status.get('status') not in ('PLAYING', 'CALIBRATING'):
        return no_update
    new_sig = _poll_eog(eog_state_p2)
    if new_sig is None:
        return no_update
    result = _run_eog_sm(eog_state_p2, new_sig, time.time(), label='P2')
    return result if result is not None else no_update


EOG_DISPLAY_SECS = 8.0


def _make_eog_figure(eog_st, title, line_color, thr_color):
    sr = eog_st['sr']
    if sr is None:
        return None
    n_req = int(EOG_DISPLAY_SECS * sr)
    data  = board.get_current_board_data(n_req)
    if data.shape[1] < 20:
        return None
    diff_raw = eog_diff(data, eog_st['ch_R'], eog_st['ch_L'])
    filtered = _eog_filter(diff_raw.copy(), sr)
    n_pts    = len(filtered)
    t        = np.linspace(-n_pts / sr, 0, n_pts)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=filtered, mode='lines', name='R−L',
                             line=dict(color=line_color, width=1)))
    sigma = eog_st['baseline_sigma']
    if sigma is not None:
        thr = EOG_SIGMA_THR * sigma
        fig.add_hline(y= thr, line_dash='dash', line_color=thr_color,
                      annotation_text=f'+{EOG_SIGMA_THR:.0f}σ', annotation_font_color=thr_color)
        fig.add_hline(y=-thr, line_dash='dash', line_color=thr_color,
                      annotation_text=f'-{EOG_SIGMA_THR:.0f}σ', annotation_font_color=thr_color)
    peak = max(float(np.abs(filtered).max()) * 1.3, 50.0) if filtered.size else 50.0
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#111111', plot_bgcolor='#111111',
        margin=dict(l=50, r=30, t=30, b=40), height=200,
        title=dict(text=title, font=dict(size=12, color='#aaa'), x=0.5),
        xaxis=dict(title='time (s)', range=[-EOG_DISPLAY_SECS, 0], color='#666'),
        yaxis=dict(title='µV', range=[-peak, peak], color='#666'),
        showlegend=False,
    )
    return fig


@app.callback(
    Output('eog-live-plot', 'figure'),
    Input('status-interval', 'n_intervals'),
    State('app-status-store', 'data'),
    prevent_initial_call=True,
)
def update_eog_plot(_, app_status):
    if not (EOG_MODE or TWO_PLAYER_MODE) or board is None or eog_state['ch_L'] is None:
        raise PreventUpdate
    if (app_status or {}).get('status', '') not in ('CALIBRATING', 'PLAYING', 'PAUSED'):
        raise PreventUpdate
    fig = _make_eog_figure(eog_state, 'P1 HEOG  (R − L, filtered)', '#aaffaa', 'orange')
    if fig is None:
        raise PreventUpdate
    return fig


@app.callback(
    Output('eog-live-plot-p2', 'figure'),
    Input('status-interval', 'n_intervals'),
    State('app-status-store', 'data'),
    prevent_initial_call=True,
)
def update_eog_plot_p2(_, app_status):
    if not TWO_PLAYER_MODE or board is None or eog_state_p2['ch_L'] is None:
        raise PreventUpdate
    if (app_status or {}).get('status', '') not in ('CALIBRATING', 'PLAYING', 'PAUSED'):
        raise PreventUpdate
    fig = _make_eog_figure(eog_state_p2, 'P2 HEOG  (R − L, filtered)', '#ffaaaa', 'cyan')
    if fig is None:
        raise PreventUpdate
    return fig


@app.callback(
    Output('settings-store', 'data'),
    Input('ball-speed-slider', 'value'),
)
def update_settings(ball_speed):
    return {'ball_speed': ball_speed, 'paddle_width': PADDLE_WIDTH}


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
    if app_status.get('status') != 'PLAYING':
        return no_update

    settings    = settings or {}
    ball_speed  = settings.get('ball_speed', abs(INITIAL_BALL_SPEED_Y))
    key_command = key_data.get('key', 'None')
    prev_key    = state.get('prev_key', 'None')
    zone_idx    = state.get('zone_idx', 1)
    speed_mult  = state.get('speed_mult', 1.0)
    balls       = state.get('balls', [])
    powerups    = state.get('powerups', [])

    # --- P1 keyboard ---
    if key_command != prev_key:
        if key_command == 'a':   zone_idx = max(0, zone_idx - 1)
        elif key_command == 'd': zone_idx = min(2, zone_idx + 1)
    state['prev_key'] = key_command

    # --- P1 EOG ---
    bci_seq = (bci_command or {}).get('seq', 0)
    if bci_seq != state.get('last_bci_seq', 0):
        state['last_bci_seq'] = bci_seq
        bci_move = (bci_command or {}).get('command', 'NEUTRAL')
        if bci_move == 'LEFT':    zone_idx = max(0, zone_idx - 1)
        elif bci_move == 'RIGHT': zone_idx = min(2, zone_idx + 1)

    state['zone_idx'] = zone_idx
    state['player_x'] = PADDLE_POSITIONS[zone_idx]

    # --- Top paddle: P2 (2-player) or AI (single-player) ---
    if TWO_PLAYER_MODE:
        zone_idx_p2 = state.get('zone_idx_p2', 1)
        prev_key_p2 = state.get('prev_key_p2', 'None')
        key_p2      = key_data.get('key_p2', 'None')

        if key_p2 != prev_key_p2:
            if key_p2 == 'ArrowLeft':  zone_idx_p2 = max(0, zone_idx_p2 - 1)
            elif key_p2 == 'ArrowRight': zone_idx_p2 = min(2, zone_idx_p2 + 1)
        state['prev_key_p2'] = key_p2

        bci_seq_p2 = (bci_command_p2 or {}).get('seq', 0)
        if bci_seq_p2 != state.get('last_bci_seq_p2', 0):
            state['last_bci_seq_p2'] = bci_seq_p2
            bci_move_p2 = (bci_command_p2 or {}).get('command', 'NEUTRAL')
            if bci_move_p2 == 'LEFT':  zone_idx_p2 = max(0, zone_idx_p2 - 1)
            elif bci_move_p2 == 'RIGHT': zone_idx_p2 = min(2, zone_idx_p2 + 1)

        state['zone_idx_p2'] = zone_idx_p2
        state['ai_x']        = PADDLE_POSITIONS[zone_idx_p2]

    else:
        ai_zone_idx   = state.get('ai_zone_idx', 1)
        ai_move_timer = state.get('ai_move_timer', 0)
        if ai_move_timer <= 0:
            target_ball = next(
                (b for b in sorted(balls, key=lambda b: b['y']) if b['vy'] < 0),
                balls[0] if balls else None,
            )
            if target_ball:
                bx = target_ball['x']
                if bx < GAME_WIDTH / 3:       ai_zone_idx = 0
                elif bx < 2 * GAME_WIDTH / 3: ai_zone_idx = 1
                else:                          ai_zone_idx = 2
            ai_move_timer = AI_REACTION_FRAMES
        else:
            ai_move_timer -= 1
        state['ai_zone_idx']   = ai_zone_idx
        state['ai_move_timer'] = ai_move_timer
        state['ai_x']          = PADDLE_POSITIONS[ai_zone_idx]

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
                    ptype = random.choice(['fire', 'ice', 'multi'])
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
                    ptype = random.choice(['fire', 'ice', 'multi'])
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
            'balls': [{'x': GAME_WIDTH / 2, 'y': GAME_HEIGHT / 2, 'vx': 0, 'vy': -ball_speed}],
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
            elif pu['type'] == 'ice':
                speed_mult = max(0.25, speed_mult * 0.5)
            elif pu['type'] == 'multi' and not multi_triggered:
                multi_triggered = True
                tripled = []
                for b in balls_remaining:
                    spd = math.hypot(b['vx'], b['vy'])
                    if spd < 0.01:
                        spd = target_speed
                    for _ in range(3):
                        a = random.uniform(-math.pi / 3, math.pi / 3)
                        tripled.append({
                            'x':  float(b['x']),
                            'y':  float(b['y']),
                            'vx': float(spd * math.sin(a)),
                            'vy': float(-abs(spd * math.cos(a))),
                        })
                balls_remaining = tripled
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


@app.callback(
    Output('winner-overlay', 'style'),
    Output('winner-overlay', 'children'),
    Input('app-status-store', 'data'),
)
def update_winner_overlay(app_status):
    if app_status and app_status.get('status') == 'GAME_OVER':
        winner = app_status.get('winner', '???')
        top_wins = winner in ('AI', 'P2')
        text_color = '#ff5252' if top_wins else '#33ff66'
        display_map = {'AI': 'AI', 'Player': 'Player', 'P1': 'Player 1', 'P2': 'Player 2'}
        display_name = display_map.get(winner, winner)
        return (
            {
                'position': 'absolute', 'top': '0', 'left': '0',
                'width': '100%', 'height': '100%',
                'backgroundColor': 'rgba(0,0,0,0.82)',
                'display': 'flex', 'flexDirection': 'column',
                'alignItems': 'center', 'justifyContent': 'center',
                'zIndex': '10',
            },
            html.Div([
                html.Div(f"{display_name} Wins!!!",
                         style={'fontSize': '64px', 'color': text_color, 'fontWeight': 'bold'}),
                html.Div("Click  Start New Game  to play again",
                         style={'fontSize': '18px', 'color': '#ccc', 'marginTop': '16px'}),
            ]),
        )
    return {'display': 'none'}, ''


@app.callback(
    Output('status-display', 'children'),
    Output('app-status-store', 'data'),
    Output('game-state-store', 'data', allow_duplicate=True),
    Output('bci-interval', 'disabled'),
    Output('game-interval', 'disabled'),
    Input('status-interval', 'n_intervals'),
    Input('pause-button', 'n_clicks'),
    Input('start-button', 'n_clicks'),
    State('app-status-store', 'data'),
    prevent_initial_call=True,
)
def manage_app_flow(status_n, pause_clicks, start_clicks, app_status):
    triggered_id   = ctx.triggered_id or 'status-interval'
    status         = app_status.get('status', 'STARTING')
    countdown      = app_status.get('countdown', 0)
    new_status     = status
    new_game_state = no_update

    needs_eog_flow = EOG_MODE or (TWO_PLAYER_MODE and not NO_BOARD_MODE)

    if triggered_id == 'pause-button' and pause_clicks > 0:
        new_status = 'PAUSED' if status != 'PAUSED' else 'PLAYING'
    elif triggered_id == 'start-button' and start_clicks > 0:
        new_game_state = get_initial_game_state()
        if needs_eog_flow:
            new_status = 'INSTRUCTIONS'
            countdown  = INSTRUCTIONS_S
            states_to_reset = [eog_state, eog_state_p2] if TWO_PLAYER_MODE else [eog_state]
            for st in states_to_reset:
                _reset_eog_st(st)
        else:
            new_status = 'PLAYING'
    elif triggered_id == 'status-interval':
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
                new_status = 'PLAYING'

    if new_status == 'INSTRUCTIONS':
        msg = html.Div([
            html.Div(f"Remove hands from keyboard — calibrating in {max(0, int(countdown))}s",
                     style={'fontSize': '20px', 'color': 'yellow', 'marginBottom': '6px'}),
            html.Div("1. Stare at the CENTER of the screen",   style={'fontSize': '15px', 'color': '#ccc'}),
            html.Div("2. Do NOT blink",                        style={'fontSize': '15px', 'color': '#ccc'}),
            html.Div("3. Keep your eyes completely still",     style={'fontSize': '15px', 'color': '#ccc'}),
        ])
    elif new_status == 'CALIBRATING':
        who = "Both players — " if TWO_PLAYER_MODE else ""
        msg = f"Calibrating — {who}hold still, eyes forward...  {max(0, int(countdown))}s"
    elif new_status == 'PLAYING':
        if TWO_PLAYER_MODE and NO_BOARD_MODE:
            msg = "2-PLAYER — P1: A/D  |  P2: ←/→"
        elif TWO_PLAYER_MODE:
            msg = "2-PLAYER EOG — glance to move  |  P1: A/D override  |  P2: ←/→ override"
        elif NO_BOARD_MODE:
            msg = "NO BOARD — keyboard only (A/D)"
        elif EOG_MODE:
            msg = "PLAYING — glance left/right to move  |  A/D to override"
        else:
            msg = "PLAYING — A/D keys"
    elif new_status == 'PAUSED':
        msg = "PAUSED"
    else:
        msg = ""

    bci_disabled  = NO_BOARD_MODE or (new_status not in ('PLAYING', 'CALIBRATING'))
    game_disabled = new_status in ('PAUSED', 'GAME_OVER')

    new_app_status = {'status': new_status, 'countdown': countdown}
    if app_status.get('winner') and new_status == 'GAME_OVER':
        new_app_status['winner'] = app_status['winner']

    return msg, new_app_status, new_game_state, bci_disabled, game_disabled


# ==============================================================================
# === 6. MAIN ==================================================================
# ==============================================================================
def main():
    global board, sampling_rate

    if NO_BOARD_MODE:
        mode_msg = ("2-PLAYER NO-BOARD: P1=A/D, P2=←/→." if TWO_PLAYER_MODE
                    else "NO-BOARD MODE: keyboard only (A/D).")
        print(mode_msg)
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        print("Open http://127.0.0.1:8050/ in your browser.")
        app.run(debug=False, use_reloader=False)
        return

    if TWO_PLAYER_MODE:
        print("=" * 60)
        print("2-PLAYER EOG MODE")
        print("P1 (bottom, green): ch1–2. Glance left→center / right→center.")
        print("P2 (top,    red  ): ch3–4. Same gestures.")
        print("Players hold hands — shared bias electrode is fine.")
        print("Both sit still during baseline calibration.")
        print("=" * 60)
    elif EOG_MODE:
        print("=" * 60)
        print("EOG MODE: glance-pair detector active.")
        print("Glance left→center to move left; right→center to move right.")
        print("Sit still for the first 3 s (baseline calibration).")
        print("=" * 60)

    params             = BrainFlowInputParams()
    params.timeout     = 15
    params.serial_port = "/dev/cu.usbserial-1120"
    board = BoardShim(BOARD_ID, params)
    try:
        print("Connecting to board...")
        board.prepare_session()
        sampling_rate    = BoardShim.get_sampling_rate(BOARD_ID)
        all_eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
        print(f"Board connected. Sampling rate: {sampling_rate} Hz")

        if EOG_MODE or TWO_PLAYER_MODE:
            eog_state['sr']   = sampling_rate
            eog_state['ch_L'] = all_eeg_channels[EOG_SLOT_L]
            eog_state['ch_R'] = all_eeg_channels[EOG_SLOT_R]
            print(f"P1 EOG channels: ch_L={eog_state['ch_L']}  ch_R={eog_state['ch_R']}")

        if TWO_PLAYER_MODE:
            eog_state_p2['sr']   = sampling_rate
            eog_state_p2['ch_L'] = all_eeg_channels[EOG_SLOT_L2]
            eog_state_p2['ch_R'] = all_eeg_channels[EOG_SLOT_R2]
            print(f"P2 EOG channels: ch_L={eog_state_p2['ch_L']}  ch_R={eog_state_p2['ch_R']}")

        print("Starting data stream...")
        board.start_stream(450000)
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
        if board and board.is_prepared():
            print("Stopping stream and releasing session.")
            board.stop_stream()
            board.release_session()


if __name__ == "__main__":
    main()
