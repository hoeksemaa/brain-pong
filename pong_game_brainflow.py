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
                         help='Use EOG glance-pair detector (requires hardware).')
_cli_args, _ = _cli_parser.parse_known_args()
NO_BOARD_MODE = _cli_args.no_board
EOG_MODE      = _cli_args.eog

if EOG_MODE and NO_BOARD_MODE:
    print("ERROR: --eog requires hardware; cannot combine with --no-board.", file=sys.stderr)
    sys.exit(2)

# --- BrainFlow ---
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# ==============================================================================
# === 1. CONFIGURATION =========================================================
# ==============================================================================
BOARD_ID             = BoardIds.CERELOG_X8_BOARD
INITIAL_BALL_SPEED_Y = -4
GAME_INTERVAL_MS     = 16
GAME_WIDTH           = 800
GAME_HEIGHT          = 600
PADDLE_WIDTH         = GAME_WIDTH // 3 - 2   # fills one of three equal zones; 2 px gap
PADDLE_POSITIONS     = [GAME_WIDTH // 6, GAME_WIDTH // 2, 5 * GAME_WIDTH // 6]
PADDLE_HEIGHT        = 20
BALL_RADIUS          = 10
POWERUP_RADIUS        = 14
POWERUP_FALL_SPEED    = 3
AI_REACTION_FRAMES    = 31  # frames between AI zone decisions (0.5 s at 16 ms/frame)

BCI_UPDATE_INTERVAL_MS = 100  # fast enough to catch return saccades within glance window

# EOG glance-pair detector
EOG_SLOT_L       = 0      # index into get_eeg_channels() — left electrode
EOG_SLOT_R       = 1      # index into get_eeg_channels() — right electrode
EOG_LPF_HZ      = 100.0
EOG_HPF_HZ      = 0.5
EOG_SIGMA_THR   = 5.0    # sustained_crossing threshold multiplier
EOG_MIN_DUR_MS  = 12.0   # min crossing duration — rejects EMG spikes
GLANCE_WINDOW_S  = 0.7   # max seconds between outward and return saccade
ARMED_MIN_WAIT_S = 0.05  # ignore crossings this soon after arming
REFRACTORY_S     = 0.8   # cooldown after firing a command
EOG_BASELINE_S   = 5.0   # seconds of quiet signal to estimate baseline noise
INSTRUCTIONS_S   = 5.0   # hands-off buffer after button click; auto-advances to CALIBRATING
EOG_POLL_S       = 0.1
EOG_SETTLE_S     = 0.4   # extra signal pulled for IIR filter settling

# ==============================================================================
# === 2. RUNTIME STATE =========================================================
# ==============================================================================
board         = None
sampling_rate = 0

eog_state = {
    'ch_L': None, 'ch_R': None, 'sr': None,
    'sm': 'CALIBRATING',   # CALIBRATING | IDLE | ARMED | REFRACTORY
    'baseline_acc': [],
    'baseline_sigma': None,
    'first_dir': None,
    'arm_time': None,
    'last_cmd_time': 0.0,
    'cmd_seq': 0,
}


def _eog_filter(x_uv, sr):
    """0.5–100 Hz causal IIR chain — mirrors segment_diff_filter preprocessing."""
    y = np.ascontiguousarray(x_uv.astype(np.float64))
    if y.size < 20:
        return y
    DataFilter.detrend(y, DetrendOperations.CONSTANT.value)
    DataFilter.perform_lowpass(y, sr, EOG_LPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    for lo, hi in ((48.0, 52.0), (58.0, 62.0)):
        DataFilter.perform_bandstop(y, sr, lo, hi, 3, FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_highpass(y, sr, EOG_HPF_HZ, 4, FilterTypes.BUTTERWORTH.value, 0)
    return y


def _sustained_crossing(signal, sigma, sr):
    """Returns 'LEFT'/'RIGHT' if |signal| > EOG_SIGMA_THR×σ for ≥ EOG_MIN_DUR_MS, else None."""
    if sigma < 1e-9 or signal.size == 0:
        return None
    thr     = EOG_SIGMA_THR * sigma
    min_dur = max(1, int(EOG_MIN_DUR_MS / 1000 * sr))
    above   = np.abs(signal) > thr
    conv    = np.convolve(above.astype(np.int32), np.ones(min_dur, dtype=np.int32), mode='valid')
    hits    = np.where(conv == min_dur)[0]
    if len(hits) == 0:
        return None
    onset = int(hits[0])
    return 'RIGHT' if signal[onset] > 0 else 'LEFT'


# ==============================================================================
# === 3. APP LAYOUT ============================================================
# ==============================================================================
app = Dash(__name__, assets_folder='assets')
app.title = "BrainPong — EOG"


POINTS_TO_WIN = 10

def get_initial_game_state():
    return {
        'player_x': GAME_WIDTH / 2, 'ai_x': GAME_WIDTH / 2,
        'balls': [{'x': GAME_WIDTH / 2, 'y': GAME_HEIGHT / 2,
                   'vx': 0, 'vy': INITIAL_BALL_SPEED_Y}],
        'powerups': [],
        'speed_mult': 1.0,
        'ai_zone_idx': 1, 'ai_move_timer': 0,
        'zone_idx': 1, 'prev_key': 'None', 'last_bci_seq': 0,
        'player_score': 0, 'ai_score': 0, 'winner': None,
    }


app.layout = html.Div(
    id='main-container',
    style={'backgroundColor': '#111', 'color': '#DDD', 'fontFamily': 'monospace', 'textAlign': 'center'},
    children=[
        html.H1("BrainPong — EOG"),
        html.Div([
            html.Button('Pause / Resume', id='pause-button',  n_clicks=0, style={'marginRight': '20px'}),
            html.Button('Start New Game', id='start-button',  n_clicks=0),
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
                        html.Div(
                            id='winner-overlay',
                            style={'display': 'none'},
                        ),
                    ],
                ),
                html.Div(
                    style={
                        'display': 'flex', 'flexDirection': 'column',
                        'justifyContent': 'space-between',
                        'paddingLeft': '18px', 'paddingTop': '8px', 'paddingBottom': '8px',
                    },
                    children=[
                        html.Div(id='ai-score-display',     children='0',
                                 style={'fontSize': '32px', 'color': '#ff5252', 'fontWeight': 'bold'}),
                        html.Div(id='player-score-display', children='0',
                                 style={'fontSize': '32px', 'color': '#33ff66', 'fontWeight': 'bold'}),
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
        html.Div(
            dcc.Graph(id='eog-live-plot', config={'displayModeBar': False}),
            style={'width': '800px', 'margin': '0 auto', 'display': 'block' if EOG_MODE else 'none'},
        ),
        dcc.Store(id='settings-store',    data={'ball_speed': abs(INITIAL_BALL_SPEED_Y), 'paddle_width': PADDLE_WIDTH}),
        dcc.Store(id='game-state-store',  data=get_initial_game_state()),
        dcc.Store(id='app-status-store',  data={'status': 'STARTING', 'countdown': 0}),
        dcc.Store(id='bci-command-store', data={'command': 'NEUTRAL', 'seq': 0}),
        dcc.Store(id='key-press-store',   data={'key': 'None'}),
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
            window.dash_clientside.current_key = 'None';
            document.addEventListener('keydown', function(e) {
                if (e.key === 'a' || e.key === 'd') { window.dash_clientside.current_key = e.key; }
            });
            document.addEventListener('keyup', function(e) {
                if (e.key === 'a' || e.key === 'd') { window.dash_clientside.current_key = 'None'; }
            });
        }
        return {key: window.dash_clientside.current_key};
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

    sr   = eog_state['sr']
    ch_L = eog_state['ch_L']
    ch_R = eog_state['ch_R']

    n_settle = int(EOG_SETTLE_S * sr)
    n_new    = max(1, int(EOG_POLL_S * sr))
    data = board.get_current_board_data(n_settle + n_new)
    if data.shape[1] < n_settle + n_new:
        return no_update

    diff_raw = (data[ch_R] - data[ch_L]) * 1e6
    filtered = _eog_filter(diff_raw.copy(), sr)
    new_sig  = filtered[-n_new:]
    now      = time.time()

    if eog_state['sm'] == 'CALIBRATING':
        eog_state['baseline_acc'].append(new_sig.copy())
        total = np.concatenate(eog_state['baseline_acc'])
        if total.size >= int(EOG_BASELINE_S * sr):
            eog_state['baseline_sigma'] = float(np.std(total))
            eog_state['sm'] = 'IDLE'
            print(f"[EOG] baseline σ = {eog_state['baseline_sigma']:.2f} µV — ready")
        return no_update

    if eog_state['sm'] == 'REFRACTORY':
        if now - eog_state['last_cmd_time'] > REFRACTORY_S:
            eog_state['sm'] = 'IDLE'
        return no_update

    sigma    = eog_state['baseline_sigma']
    crossing = _sustained_crossing(new_sig, sigma, sr)

    if eog_state['sm'] == 'IDLE':
        if crossing is not None:
            eog_state['sm']        = 'ARMED'
            eog_state['first_dir'] = crossing
            eog_state['arm_time']  = now

    elif eog_state['sm'] == 'ARMED':
        if now - eog_state['arm_time'] > GLANCE_WINDOW_S:
            eog_state['sm']        = 'IDLE'
            eog_state['first_dir'] = None
        elif now - eog_state['arm_time'] > ARMED_MIN_WAIT_S and crossing is not None:
            opposite = {'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
            if crossing == opposite.get(eog_state['first_dir']):
                cmd = eog_state['first_dir']
                eog_state['cmd_seq']      += 1
                eog_state['last_cmd_time'] = now
                eog_state['sm']            = 'REFRACTORY'
                eog_state['first_dir']     = None
                print(f"[EOG] command={cmd}  seq={eog_state['cmd_seq']}")
                return {'command': cmd, 'seq': eog_state['cmd_seq']}

    return no_update


EOG_DISPLAY_SECS = 8.0   # rolling window shown in the live plot

@app.callback(
    Output('eog-live-plot', 'figure'),
    Input('status-interval', 'n_intervals'),
    State('app-status-store', 'data'),
    prevent_initial_call=True,
)
def update_eog_plot(_, app_status):
    if not EOG_MODE or board is None or eog_state['ch_L'] is None:
        raise PreventUpdate
    status = (app_status or {}).get('status', '')
    if status not in ('CALIBRATING', 'PLAYING', 'PAUSED'):
        raise PreventUpdate

    sr   = eog_state['sr']
    ch_L = eog_state['ch_L']
    ch_R = eog_state['ch_R']

    n_req = int(EOG_DISPLAY_SECS * sr)
    data  = board.get_current_board_data(n_req)
    if data.shape[1] < 20:
        raise PreventUpdate

    diff_raw = (data[ch_R] - data[ch_L]) * 1e6
    filtered = _eog_filter(diff_raw.copy(), sr)
    n_pts    = len(filtered)
    t        = np.linspace(-n_pts / sr, 0, n_pts)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=filtered, mode='lines', name='R−L',
                             line=dict(color='#aaffaa', width=1)))

    sigma = eog_state['baseline_sigma']
    if sigma is not None:
        thr = EOG_SIGMA_THR * sigma
        fig.add_hline(y= thr, line_dash='dash', line_color='orange',
                      annotation_text=f'+{EOG_SIGMA_THR:.0f}σ', annotation_font_color='orange')
        fig.add_hline(y=-thr, line_dash='dash', line_color='orange',
                      annotation_text=f'-{EOG_SIGMA_THR:.0f}σ', annotation_font_color='orange')

    peak = max(float(np.abs(filtered).max()) * 1.3, 50.0) if filtered.size else 50.0

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',
        margin=dict(l=50, r=30, t=30, b=40),
        height=200,
        title=dict(text='HEOG  (R − L, filtered)', font=dict(size=12, color='#aaa'), x=0.5),
        xaxis=dict(title='time (s)', range=[-EOG_DISPLAY_SECS, 0], color='#666'),
        yaxis=dict(title='µV', range=[-peak, peak], color='#666'),
        showlegend=False,
    )
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
    State('app-status-store', 'data'),
    State('key-press-store', 'data'),
    State('settings-store', 'data'),
    prevent_initial_call=True,
)
def update_game_physics(_, state, bci_command, app_status, key_data, settings):
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

    # Keyboard: step one zone per new key press (hold doesn't repeat)
    if key_command != prev_key:
        if key_command == 'a':   zone_idx = max(0, zone_idx - 1)
        elif key_command == 'd': zone_idx = min(2, zone_idx + 1)
    state['prev_key'] = key_command

    # EOG: one-shot — only act when seq increments
    bci_seq = (bci_command or {}).get('seq', 0)
    if bci_seq != state.get('last_bci_seq', 0):
        state['last_bci_seq'] = bci_seq
        bci_move = (bci_command or {}).get('command', 'NEUTRAL')
        if bci_move == 'LEFT':    zone_idx = max(0, zone_idx - 1)
        elif bci_move == 'RIGHT': zone_idx = min(2, zone_idx + 1)

    state['zone_idx'] = zone_idx
    state['player_x'] = PADDLE_POSITIONS[zone_idx]

    # AI re-evaluates its zone every AI_REACTION_FRAMES frames then snaps instantly.
    ai_zone_idx   = state.get('ai_zone_idx', 1)
    ai_move_timer = state.get('ai_move_timer', 0)
    if ai_move_timer <= 0:
        target_ball = next(
            (b for b in sorted(balls, key=lambda b: b['y']) if b['vy'] < 0),
            balls[0] if balls else None,
        )
        if target_ball:
            bx = target_ball['x']
            if bx < GAME_WIDTH / 3:
                ai_zone_idx = 0
            elif bx < 2 * GAME_WIDTH / 3:
                ai_zone_idx = 1
            else:
                ai_zone_idx = 2
        ai_move_timer = AI_REACTION_FRAMES
    else:
        ai_move_timer -= 1
    state['ai_zone_idx']   = ai_zone_idx
    state['ai_move_timer'] = ai_move_timer
    state['ai_x']          = PADDLE_POSITIONS[ai_zone_idx]

    target_speed       = ball_speed * speed_mult
    balls_remaining    = []
    last_exit_scorer   = None   # 'player' or 'ai' — updated for every ball that exits
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

        if ball['vy'] > 0 and ball['y'] + BALL_RADIUS >= GAME_HEIGHT - PADDLE_HEIGHT:
            if abs(state['player_x'] - ball['x']) < PADDLE_WIDTH / 2 + BALL_RADIUS:
                spd = math.hypot(ball['vx'], ball['vy'])
                a = random.uniform(-math.pi / 3, math.pi / 3)
                ball['vx'] = spd * math.sin(a)
                ball['vy'] = -spd * math.cos(a)
                ball['y']  = GAME_HEIGHT - PADDLE_HEIGHT - BALL_RADIUS

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
                        'x':   float(ball['x']),
                        'y':   float(PADDLE_HEIGHT + BALL_RADIUS + POWERUP_RADIUS + 2),
                        'vy':  float(POWERUP_FALL_SPEED),
                        'type': ptype,
                    })

        if ball['y'] - BALL_RADIUS > GAME_HEIGHT:
            state['ai_score'] += 1
            if state['ai_score'] >= POINTS_TO_WIN:
                state.update({'winner': 'AI', 'balls': [], 'powerups': []})
                return state
        elif ball['y'] + BALL_RADIUS < 0:
            state['player_score'] += 1
            if state['player_score'] >= POINTS_TO_WIN:
                state.update({'winner': 'Player', 'balls': [], 'powerups': []})
                return state
        else:
            balls_remaining.append(ball)

    # New round only once every ball has been scored
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

        if pu['y'] - POWERUP_RADIUS > GAME_HEIGHT:
            continue  # fell off screen

        if (pu['y'] + POWERUP_RADIUS >= GAME_HEIGHT - PADDLE_HEIGHT and
                abs(state['player_x'] - pu['x']) < PADDLE_WIDTH / 2 + POWERUP_RADIUS):
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
            continue  # caught, remove from active

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
        color_name  = 'Red'   if winner == 'AI'     else 'Green'
        text_color  = '#ff5252' if winner == 'AI'   else '#33ff66'
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
                html.Div(f"{color_name} Wins!!!",
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

    if triggered_id == 'pause-button' and pause_clicks > 0:
        new_status = 'PAUSED' if status != 'PAUSED' else 'PLAYING'
    elif triggered_id == 'start-button' and start_clicks > 0:
        new_status     = 'INSTRUCTIONS'
        countdown      = INSTRUCTIONS_S
        new_game_state = get_initial_game_state()
    elif triggered_id == 'status-interval':
        if status == 'STARTING':
            new_status = 'PLAYING' if not EOG_MODE else status
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
        msg = f"Calibrating — hold still, eyes forward...  {max(0, int(countdown))}s"
    elif new_status == 'PLAYING':
        if NO_BOARD_MODE: msg = "NO BOARD — keyboard only (A/D)"
        elif EOG_MODE:    msg = "PLAYING — glance left/right to move  |  A/D to override"
        else:             msg = "PLAYING — A/D keys"
    elif new_status == 'PAUSED':
        msg = "PAUSED"
    else:
        msg = ""

    bci_disabled  = NO_BOARD_MODE or (new_status not in ('PLAYING', 'CALIBRATING'))
    game_disabled = new_status in ('PAUSED', 'GAME_OVER')

    return msg, {'status': new_status, 'countdown': countdown}, new_game_state, bci_disabled, game_disabled


# ==============================================================================
# === 6. MAIN ==================================================================
# ==============================================================================
def main():
    global board, sampling_rate

    if NO_BOARD_MODE:
        print("NO-BOARD MODE: keyboard only (A/D).")
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        print("Open http://127.0.0.1:8050/ in your browser.")
        app.run(debug=False, use_reloader=False)
        return

    if EOG_MODE:
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

        if EOG_MODE:
            eog_state['sr']   = sampling_rate
            eog_state['ch_L'] = all_eeg_channels[EOG_SLOT_L]
            eog_state['ch_R'] = all_eeg_channels[EOG_SLOT_R]
            print(f"EOG channels: ch_L={eog_state['ch_L']}  ch_R={eog_state['ch_R']}")

        print("Starting data stream...")
        board.start_stream(450000)
        time.sleep(1.0)

        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        print("Open http://127.0.0.1:8050/ in your browser.")
        if EOG_MODE:
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
