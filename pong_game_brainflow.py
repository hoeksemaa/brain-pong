import argparse
import os
import sys
import time
from datetime import datetime, timezone
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Output, Input, State, no_update, clientside_callback, ctx
from dash.exceptions import PreventUpdate
import logging

# --- CLI ---
_cli_parser = argparse.ArgumentParser(add_help=False)
_cli_parser.add_argument('--no-board', action='store_true',
                         help='Run the game without BrainFlow hardware (keyboard-only, skips calibration).')
_cli_parser.add_argument('--record', action='store_true',
                         help='Run in recording mode: capture labeled SSVEP-EEG trials to recordings/<id>.npz. Requires hardware (incompatible with --no-board).')
_cli_parser.add_argument('--trials', type=int, default=40,
                         help='Total number of trials per recording session (default 40 = 20 LEFT + 20 RIGHT alternating). Must be even.')
_cli_args, _ = _cli_parser.parse_known_args()
NO_BOARD_MODE = _cli_args.no_board
RECORDING_MODE = _cli_args.record
TOTAL_TRIALS = _cli_args.trials

if RECORDING_MODE and NO_BOARD_MODE:
    print("ERROR: --record requires hardware; cannot combine with --no-board.", file=sys.stderr)
    sys.exit(2)
if RECORDING_MODE and TOTAL_TRIALS % 2 != 0:
    print(f"ERROR: --trials must be even (got {TOTAL_TRIALS}); strict L/R alternation requires equal counts.", file=sys.stderr)
    sys.exit(2)

# --- BrainFlow Integration ---
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes, DetrendOperations, AggOperations

# --- Signal Processing & Machine Learning Libraries ---
# (Removed unused Scipy imports like butter, lfilter)
from sklearn.cross_decomposition import CCA as SklearnCCA

# ==============================================================================
# === 1. TUNABLE CONFIGURATION ===============================================
# ==============================================================================
# --- BrainFlow Board ---
BOARD_ID = BoardIds.CERELOG_X8_BOARD

# --- Game Feel & Speed Adjustments ---
PADDLE_SPEED = 30
INITIAL_BALL_SPEED_Y = -4

# --- BCI & Signal Processing ---
CHANNELS_TO_USE = [1, 2, 3, 4] 
SSVEP_FREQ_LEFT = 10
SSVEP_FREQ_RIGHT = 15.0
FFT_WINDOW_SECONDS = 1.5
FFT_OVERLAP_PERCENT = 0.8
USE_EMA_SMOOTHING = True
EMA_SMOOTHING_FACTOR = 0.4
BCI_SCORE_AMPLIFIER = 2.5

# NOTE: These now control the BrainFlow filters directly
FILTER_LOW_CUT_HZ = 5.0   # Highpass cutoff (removes frequencies BELOW this)
FILTER_HIGH_CUT_HZ = 45.0 # Lowpass cutoff (removes frequencies ABOVE this)
FILTER_ORDER = 4          # BrainFlow usually works best with order 4 for Butterworth

CCA_NUM_HARMONICS = 3
CALIBRATION_DURATION_S = 7
CALIBRATION_THRESHOLD_STD_FACTOR = 0.8
MIN_THRESHOLD_GAP = 0.05
GAME_INTERVAL_MS = 16
AI_PADDLE_SPEED = 6
BALL_SPIN_FACTOR = 0.06
GAME_WIDTH = 800
GAME_HEIGHT = 600
PADDLE_WIDTH = 150
PADDLE_HEIGHT = 20
BALL_RADIUS = 10

# --- Recording mode (per plans/recording-protocol.md v1) ---
RECORDINGS_DIR = 'recordings'
RECORD_TRIAL_DURATION_S = 15.0
RECORD_REST_DURATION_S = 3.0
PROTOCOL_VERSION = 'v1'

# Marker codes injected into the BrainFlow marker channel via insert_marker.
# These show up as float values in the marker channel of the saved EEG, giving
# us cue/press/release events with native board-clock timestamps and zero
# cross-clock sync error.
MARKER_SESSION_START = 1.0
MARKER_CUE_LEFT      = 2.0
MARKER_CUE_RIGHT     = 3.0
MARKER_CUE_OFFSET    = 4.0
MARKER_PRESS         = 5.0
MARKER_RELEASE       = 6.0
MARKER_SESSION_END   = 7.0

# ==============================================================================
# === 2. CORE SETUP ============================================================
# ==============================================================================
board = None
sampling_rate = 0
bci_eeg_channels = []
fft_samples = 0
cca_ref_signals = {}
cca_model = SklearnCCA(n_components=1)

BCI_UPDATE_INTERVAL_MS = int((FFT_WINDOW_SECONDS * (1 - FFT_OVERLAP_PERCENT)) * 1000)

# --- Recording mode: server-side mirrors so finally-block can save partial sessions ---
recording_session = {
    'session_id':     None,
    'subject_id':     None,
    'headset_notes':  None,
    'started_at_iso': None,
    'started':        False,   # True after user pressed space in RECORD_READY
    'finalized':      False,   # True after save_session_npz has been called
    'events':         [],      # list of {kind, side, t_browser_ms, trial_idx}
    'last_event_seq': -1,      # de-dup guard for the events store
}

def preprocess_eeg_window(eeg_data):
    """
    Applies the BrainFlow specific filtering pipeline using global config settings.
    """
    if eeg_data.ndim == 1: 
        eeg_data = eeg_data.reshape(-1, 1)
    
    # Work on a contiguous copy to ensure C-compatibility
    data_to_process = np.ascontiguousarray(eeg_data.T) 
    n_channels = data_to_process.shape[0]

    for i in range(n_channels):
        if data_to_process[i].size > 20:
            # 1. Detrend (Center at 0)
            DataFilter.detrend(data_to_process[i], DetrendOperations.CONSTANT.value)
            
            # 2. Low-Pass (Remove High Freq Noise) using FILTER_HIGH_CUT_HZ (e.g., 45Hz)
            DataFilter.perform_lowpass(data_to_process[i], sampling_rate, float(FILTER_HIGH_CUT_HZ), FILTER_ORDER, FilterTypes.BUTTERWORTH.value, 0)
            
            # 3. High-Pass (Remove Low Freq Drift) using FILTER_LOW_CUT_HZ (e.g., 5Hz)
            DataFilter.perform_highpass(data_to_process[i], sampling_rate, float(FILTER_LOW_CUT_HZ), FILTER_ORDER, FilterTypes.BUTTERWORTH.value, 0)

            # 4. Notch Filters (Grid Noise) - Kept hardcoded as grid noise is always 50/60Hz
            DataFilter.perform_bandstop(data_to_process[i], sampling_rate, 48.0, 52.0, 3, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data_to_process[i], sampling_rate, 58.0, 62.0, 3, FilterTypes.BUTTERWORTH.value, 0)
            
            # 5. Rolling Median (Smoothing)
            DataFilter.perform_rolling_filter(data_to_process[i], 3, AggOperations.MEDIAN.value)

    return data_to_process.T

def get_cca_correlation(eeg_data_multi_channel, ref_signals):
    if eeg_data_multi_channel.shape[0] < eeg_data_multi_channel.shape[1] or eeg_data_multi_channel.shape[0] != ref_signals.shape[0]: return 0.0
    try:
        cca_model.fit(eeg_data_multi_channel, ref_signals)
        U, V = cca_model.transform(eeg_data_multi_channel, ref_signals)
        return np.corrcoef(U.T, V.T)[0, 1]
    except Exception: return 0.0

# ==============================================================================
# === Recording mode helpers ===================================================
# ==============================================================================
def safe_insert_marker(value):
    """Inject a marker into the BrainFlow stream; never raise (best-effort)."""
    try:
        if board is not None and board.is_prepared():
            board.insert_marker(float(value))
    except Exception as e:
        print(f"[record] insert_marker({value}) failed: {e}")

def collect_session_metadata_from_cli():
    print("=" * 60)
    print(f"RECORDING MODE — protocol {PROTOCOL_VERSION}, {TOTAL_TRIALS} trials")
    print("=" * 60)
    subject_id = input("Subject ID (e.g. 'jh'): ").strip() or 'unknown'
    headset_notes = input("Headset notes (impedance, fit, anything weird): ").strip()
    session_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    print(f"Session ID: {session_id}")
    print(f"Output dir: {os.path.abspath(RECORDINGS_DIR)}")
    print("=" * 60)
    return session_id, subject_id, headset_notes

def save_session_npz(final_payload, incomplete=False):
    """Persist the current session to recordings/<session_id>.npz.
    final_payload comes from the JS final-state flush and contains:
      {'edge_log': [...], 'measurement': {...}, 'browser_ua': '...'}
    """
    if recording_session['finalized']:
        return None
    session_id = recording_session['session_id']
    if not session_id:
        return None

    out_path = os.path.join(RECORDINGS_DIR, f'{session_id}.npz')

    eeg_full = None
    eeg = None
    eeg_t = None
    markers = None
    try:
        if board is not None and board.is_prepared():
            eeg_full = board.get_board_data()
    except Exception as e:
        print(f"[record] get_board_data failed: {e}")

    if eeg_full is not None and eeg_full.size > 0:
        try:
            ts_channel = BoardShim.get_timestamp_channel(BOARD_ID)
            marker_channel = BoardShim.get_marker_channel(BOARD_ID)
            eeg_channels = bci_eeg_channels if bci_eeg_channels else BoardShim.get_eeg_channels(BOARD_ID)
            eeg = eeg_full[eeg_channels].astype(np.float32)
            eeg_t = eeg_full[ts_channel].astype(np.float64)
            markers = eeg_full[marker_channel].astype(np.float32)
        except Exception as e:
            print(f"[record] failed to slice BrainFlow channels: {e}")
            eeg = eeg_full.astype(np.float32)  # fallback: raw matrix

    events_arr = np.array(
        [(e.get('t_browser_ms', 0.0), e.get('kind', ''), e.get('side', ''), e.get('trial_idx', -1))
         for e in recording_session['events']],
        dtype=np.dtype([('t_browser_ms', 'f8'), ('kind', 'U16'), ('side', 'U2'), ('trial_idx', 'i4')]),
    )

    edge_log = (final_payload or {}).get('edge_log') or []
    edge_log_arr = np.array(
        [(e.get('ms', 0.0), int(e.get('frame', 0)), e.get('side', ''), bool(e.get('isOn', False)))
         for e in edge_log],
        dtype=np.dtype([('ms', 'f8'), ('frame', 'i8'), ('side', 'U1'), ('is_on', '?')]),
    )

    measurement = (final_payload or {}).get('measurement') or {}
    metadata = {
        'protocol_version': PROTOCOL_VERSION,
        'session_id':       session_id,
        'subject_id':       recording_session['subject_id'],
        'headset_notes':    recording_session['headset_notes'],
        'started_at_iso':   recording_session['started_at_iso'],
        'ended_at_iso':     datetime.now(timezone.utc).isoformat(),
        'incomplete':       bool(incomplete),
        'total_trials':     TOTAL_TRIALS,
        'board_id':         int(BOARD_ID),
        'sampling_rate':    int(sampling_rate) if sampling_rate else None,
        'eeg_channel_indices': list(bci_eeg_channels),
        'serial_port':      "/dev/cu.usbserial-1120",
        'stimulus_left_hz':  measurement.get('actualLeftHz'),
        'stimulus_right_hz': measurement.get('actualRightHz'),
        'display_refresh_hz': measurement.get('measuredHz'),
        'chosen_refresh_hz':  measurement.get('chosenRefreshHz'),
        'left_period_frames':  measurement.get('leftPeriodFrames'),
        'right_period_frames': measurement.get('rightPeriodFrames'),
        'browser_user_agent': (final_payload or {}).get('browser_ua'),
        'filter_low_cut_hz':  FILTER_LOW_CUT_HZ,
        'filter_high_cut_hz': FILTER_HIGH_CUT_HZ,
        'filter_order':       FILTER_ORDER,
        'marker_codes': {
            'session_start':  MARKER_SESSION_START,
            'cue_left':       MARKER_CUE_LEFT,
            'cue_right':      MARKER_CUE_RIGHT,
            'cue_offset':     MARKER_CUE_OFFSET,
            'press':          MARKER_PRESS,
            'release':        MARKER_RELEASE,
            'session_end':    MARKER_SESSION_END,
        },
    }

    save_kwargs = {
        'events':   events_arr,
        'edge_log': edge_log_arr,
        'metadata': np.array(metadata, dtype=object),
    }
    if eeg is not None:       save_kwargs['eeg'] = eeg
    if eeg_t is not None:     save_kwargs['eeg_t'] = eeg_t
    if markers is not None:   save_kwargs['markers'] = markers

    np.savez(out_path, **save_kwargs)
    recording_session['finalized'] = True
    print(f"[record] saved {out_path}  (incomplete={incomplete}, events={len(recording_session['events'])}, edges={len(edge_log)})")
    return out_path

# ==============================================================================
# === 3. DASH APP LAYOUT =======================================================
# ==============================================================================
app = Dash(__name__, assets_folder='assets')
app.title = "BrainFlow BCI Pong"

def get_initial_game_state():
    return { 'player_x': GAME_WIDTH / 2, 'ai_x': GAME_WIDTH / 2, 'ball_x': GAME_WIDTH / 2,
             'ball_y': GAME_HEIGHT / 2, 'ball_vx': 0, 'ball_vy': INITIAL_BALL_SPEED_Y,
             'player_score': 0, 'ai_score': 0 }

app.layout = html.Div(id='main-container', style={'backgroundColor': '#111', 'color': '#DDD', 'fontFamily': 'monospace', 'textAlign': 'center'}, children=[
    html.H1("BrainFlow BCI Pong"),
    html.Div([
        html.Button('Pause / Resume', id='pause-button', n_clicks=0, style={'marginRight': '20px'}),
        html.Button('Restart Game', id='restart-button', n_clicks=0),
    ], style={'marginBottom': '10px'}),
    html.H3(id='status-display', style={'fontSize': '24px', 'color': 'yellow', 'height': '30px'}),
    html.Div(html.Canvas(id='pong-game-canvas', width=GAME_WIDTH, height=GAME_HEIGHT), style={'width': f'{GAME_WIDTH}px', 'margin': 'auto', 'border': '2px solid #555'}),
    html.Div(style={'width': '800px', 'margin': '15px auto', 'textAlign': 'left', 'padding': '10px', 'border': '1px solid #333', 'borderRadius': '6px'}, children=[
        html.Div([
            html.Label('Ball Speed', style={'display': 'inline-block', 'width': '120px'}),
            html.Div(dcc.Slider(id='ball-speed-slider', min=1, max=12, step=0.5, value=abs(INITIAL_BALL_SPEED_Y),
                                marks={i: {'label': str(i), 'style': {'color': 'black', 'fontWeight': 'bold'}} for i in range(1, 13, 2)}),
                     style={'display': 'inline-block', 'width': '640px', 'verticalAlign': 'middle'}),
        ]),
        html.Div([
            html.Label('Paddle Size', style={'display': 'inline-block', 'width': '120px'}),
            html.Div(dcc.Slider(id='paddle-size-slider', min=50, max=300, step=10, value=PADDLE_WIDTH,
                                marks={i: {'label': str(i), 'style': {'color': 'black', 'fontWeight': 'bold'}} for i in range(50, 301, 50)}),
                     style={'display': 'inline-block', 'width': '640px', 'verticalAlign': 'middle'}),
        ]),
        html.Div([
            html.Label('Paddle Speed', style={'display': 'inline-block', 'width': '120px'}),
            html.Div(dcc.Slider(id='paddle-speed-slider', min=5, max=80, step=1, value=PADDLE_SPEED,
                                marks={i: {'label': str(i), 'style': {'color': 'black', 'fontWeight': 'bold'}} for i in range(10, 81, 10)}),
                     style={'display': 'inline-block', 'width': '640px', 'verticalAlign': 'middle'}),
        ]),
        html.Div([
            html.Label('AI Difficulty', style={'display': 'inline-block', 'width': '120px'}),
            html.Div(dcc.Slider(id='ai-difficulty-slider', min=1, max=3, step=1, value=2,
                                marks={1: {'label': '1 (easy)', 'style': {'color': 'black', 'fontWeight': 'bold'}},
                                       2: {'label': '2 (normal)', 'style': {'color': 'black', 'fontWeight': 'bold'}},
                                       3: {'label': '3 (hard)', 'style': {'color': 'black', 'fontWeight': 'bold'}}}),
                     style={'display': 'inline-block', 'width': '640px', 'verticalAlign': 'middle'}),
        ]),
    ]),
    html.Div(style={'width': '1000px', 'margin': '20px auto', 'display': 'none' if NO_BOARD_MODE else 'flex', 'justifyContent': 'space-around'}, children=[
        dcc.Graph(id='psd-plot', style={'width': '60%'}),
        dcc.Graph(id='control-plot', style={'width': '35%'})
    ]),
    dcc.Store(id='settings-store', data={
        'paddle_width': PADDLE_WIDTH,
        'paddle_speed': PADDLE_SPEED,
        'ball_speed': abs(INITIAL_BALL_SPEED_Y),
        'ai_difficulty': 2,
    }),
    dcc.Store(id='game-state-store', data=get_initial_game_state()),
    dcc.Store(id='app-status-store', data={'status': 'STARTING', 'countdown': 0}),
    dcc.Store(id='calibration-store', data={'scores_left': [], 'scores_right': [], 'scores_rest': [], 'thresholds': None}),
    dcc.Store(id='bci-command-store', data={'command': 'NEUTRAL', 'raw_score': 0.0, 'smoothed_score': 0.0}),
    dcc.Store(id='key-press-store', data={'key': 'None'}),
    dcc.Store(id='recording-events-store', data={'events': [], 'seq': 0}),
    dcc.Store(id='recording-state-store', data={'trial_idx': 0, 'side': None, 'phase_start_t': None}),
    dcc.Store(id='recording-final-store', data=None),
    dcc.Store(id='recording-save-status-store', data={'saved_path': None, 'error': None}),
    dcc.Store(id='eeg-live-store', data=None),
    dcc.Interval(id='game-interval', interval=GAME_INTERVAL_MS, n_intervals=0, disabled=False),
    dcc.Interval(id='bci-interval', interval=BCI_UPDATE_INTERVAL_MS, n_intervals=0, disabled=True),
    dcc.Interval(id='status-interval', interval=500, n_intervals=0)
])

# ==============================================================================
# === 4. CLIENTSIDE CALLBACKS ==================================================
# ==============================================================================
clientside_callback(
    """ function(n_intervals) {
        if (!window.dash_clientside) { window.dash_clientside = {}; }
        if (!window.dash_clientside.key_listener_added) {
            window.dash_clientside.key_listener_added = true;
            window.dash_clientside.current_key = 'None';
            document.addEventListener('keydown', function(event) {
                if (event.key === 'a' || event.key === 'd') { window.dash_clientside.current_key = event.key; }
            });
            document.addEventListener('keyup', function(event) {
                if (event.key === 'a' || event.key === 'd') { window.dash_clientside.current_key = 'None'; }
            });
        }
        return {key: window.dash_clientside.current_key};
    } """,
    Output('key-press-store', 'data'),
    Input('game-interval', 'n_intervals')
)

if RECORDING_MODE:
    # Capture sub-ms keypress timestamps via performance.now() and flush to a
    # Dash store on every game-interval tick. Sets up the keydown/keyup
    # listeners on first call (idempotent). The pending-events buffer in JS
    # is the authoritative source of timing precision; the Dash poll just
    # ferries them server-side at ~16 ms granularity.
    clientside_callback(
        """
        function(n_intervals, appStatus, currentStore) {
            if (!window.dash_clientside) { window.dash_clientside = {}; }
            if (!window.dash_clientside.brainpong_record_listeners_added) {
                window.dash_clientside.brainpong_record_listeners_added = true;
                window.dash_clientside.brainpong_pending_input = [];
                window.dash_clientside.brainpong_seq = 0;
                window.dash_clientside.brainpong_space_held = false;
                document.addEventListener('keydown', function(ev) {
                    if (ev.code !== 'Space' && ev.key !== ' ') return;
                    // Suppress browser's native page-scroll on EVERY space
                    // keydown (including auto-repeat) before any guards.
                    ev.preventDefault();
                    if (ev.repeat) return;
                    if (window.dash_clientside.brainpong_space_held) return;
                    window.dash_clientside.brainpong_space_held = true;
                    window.dash_clientside.brainpong_pending_input.push({
                        kind: 'press',
                        side: '',
                        t_browser_ms: performance.now(),
                    });
                }, {passive: false});
                document.addEventListener('keyup', function(ev) {
                    if (ev.code !== 'Space' && ev.key !== ' ') return;
                    ev.preventDefault();
                    if (!window.dash_clientside.brainpong_space_held) return;
                    window.dash_clientside.brainpong_space_held = false;
                    window.dash_clientside.brainpong_pending_input.push({
                        kind: 'release',
                        side: '',
                        t_browser_ms: performance.now(),
                    });
                }, {passive: false});
            }
            var pending = window.dash_clientside.brainpong_pending_input;
            if (!pending.length) {
                return window.dash_clientside.no_update;
            }
            var existing = (currentStore && currentStore.events) || [];
            var nextEvents = existing.concat(pending);
            window.dash_clientside.brainpong_pending_input = [];
            window.dash_clientside.brainpong_seq = (window.dash_clientside.brainpong_seq || 0) + 1;
            return { events: nextEvents, seq: window.dash_clientside.brainpong_seq };
        }
        """,
        Output('recording-events-store', 'data'),
        Input('game-interval', 'n_intervals'),
        State('app-status-store', 'data'),
        State('recording-events-store', 'data'),
    )

    # On RECORD_DONE entry, dump the JS edge log + measurement + UA into the
    # final-store so the server can persist them as part of the .npz. Guarded
    # so it only fires once per session.
    clientside_callback(
        """
        function(appStatus) {
            if (!appStatus || appStatus.status !== 'RECORD_DONE') {
                return window.dash_clientside.no_update;
            }
            if (window.dash_clientside.brainpong_final_flushed) {
                return window.dash_clientside.no_update;
            }
            window.dash_clientside.brainpong_final_flushed = true;
            return {
                edge_log: (window.dash_clientside.brainpong_edge_log || []).slice(),
                measurement: window.dash_clientside.brainpong_measurement || null,
                browser_ua: navigator.userAgent,
            };
        }
        """,
        Output('recording-final-store', 'data'),
        Input('app-status-store', 'data'),
    )

clientside_callback(
    f"""
    function(gameState, appStatus, settings, recState, saveStatus, eegLive) {{
        if (window.dash_clientside && window.dash_clientside.renderPong) {{
            window.dash_clientside.renderPong(
                'pong-game-canvas',
                gameState,
                appStatus,
                settings,
                {str(NO_BOARD_MODE).lower()},
                {SSVEP_FREQ_LEFT},
                {SSVEP_FREQ_RIGHT},
                {{
                    recordingMode: {str(RECORDING_MODE).lower()},
                    totalTrials: {TOTAL_TRIALS},
                    trialDurationS: {RECORD_TRIAL_DURATION_S},
                    restDurationS: {RECORD_REST_DURATION_S},
                    recState: recState,
                    saveStatus: saveStatus,
                    eegLive: eegLive
                }}
            );
        }}
        return null;
    }}
    """,
    Output('pong-game-canvas', 'className'),
    Input('game-state-store', 'data'),
    Input('app-status-store', 'data'),
    Input('settings-store', 'data'),
    Input('recording-state-store', 'data'),
    Input('recording-save-status-store', 'data'),
    Input('eeg-live-store', 'data')
)

# ==============================================================================
# === 5. CORE BCI & GAME LOGIC CALLBACKS =======================================
# ==============================================================================
@app.callback(
    Output('bci-command-store', 'data'),
    Output('calibration-store', 'data', allow_duplicate=True),
    Input('bci-interval', 'n_intervals'),
    State('app-status-store', 'data'),
    State('calibration-store', 'data'),
    State('bci-command-store', 'data'),
    prevent_initial_call=True
)
def update_bci_command(_, app_status, cal_data, last_bci_command):
    if board is None:
        return no_update, no_update
    status = app_status.get('status', 'STARTING')
    data = board.get_current_board_data(fft_samples)
    if data.shape[1] < fft_samples:
        return no_update, no_update
    eeg_window = data[bci_eeg_channels].T
    processed_eeg = preprocess_eeg_window(eeg_window)
    corr_left = get_cca_correlation(processed_eeg, cca_ref_signals[SSVEP_FREQ_LEFT])
    corr_right = get_cca_correlation(processed_eeg, cca_ref_signals[SSVEP_FREQ_RIGHT])
    raw_score = (corr_right - corr_left) * BCI_SCORE_AMPLIFIER
    if status.startswith('CALIBRATING'):
        if 'LEFT' in status: cal_data['scores_left'].append(raw_score)
        elif 'RIGHT' in status: cal_data['scores_right'].append(raw_score)
        elif 'REST' in status: cal_data['scores_rest'].append(raw_score)
        return {'command': 'NEUTRAL', 'raw_score': raw_score, 'smoothed_score': 0.0}, cal_data
    elif status == 'PLAYING':
        smoothed_score = raw_score
        if USE_EMA_SMOOTHING:
            last_smoothed = last_bci_command.get('smoothed_score', 0.0)
            smoothed_score = (EMA_SMOOTHING_FACTOR * last_smoothed) + ((1 - EMA_SMOOTHING_FACTOR) * raw_score)
        thresholds = cal_data.get('thresholds')
        if not thresholds: return no_update, no_update
        if smoothed_score > thresholds['right']: command = 'RIGHT'
        elif smoothed_score < thresholds['left']: command = 'LEFT'
        else: command = 'NEUTRAL'
        return {'command': command, 'raw_score': raw_score, 'smoothed_score': smoothed_score}, no_update
    return no_update, no_update

@app.callback(
    Output('settings-store', 'data'),
    Input('ball-speed-slider', 'value'),
    Input('paddle-size-slider', 'value'),
    Input('paddle-speed-slider', 'value'),
    Input('ai-difficulty-slider', 'value'),
)
def update_settings(ball_speed, paddle_size, paddle_speed, ai_difficulty):
    return {'ball_speed': ball_speed, 'paddle_width': paddle_size, 'paddle_speed': paddle_speed, 'ai_difficulty': ai_difficulty}

@app.callback(
    Output('game-state-store', 'data', allow_duplicate=True),
    Input('game-interval', 'n_intervals'),
    State('game-state-store', 'data'),
    State('bci-command-store', 'data'),
    State('app-status-store', 'data'),
    State('key-press-store', 'data'),
    State('settings-store', 'data'),
    prevent_initial_call=True
)
def update_game_physics(_, state, bci_command, app_status, key_data, settings):
    if app_status.get('status') != 'PLAYING':
        return no_update
    settings = settings or {}
    paddle_speed = settings.get('paddle_speed', PADDLE_SPEED)
    paddle_width = settings.get('paddle_width', PADDLE_WIDTH)
    ball_speed = settings.get('ball_speed', abs(INITIAL_BALL_SPEED_Y))
    ai_speed = {1: 3, 2: 6, 3: 10}.get(settings.get('ai_difficulty', 2), AI_PADDLE_SPEED)
    key_command = key_data.get('key', 'None')
    if key_command == 'a': state['player_x'] -= paddle_speed
    elif key_command == 'd': state['player_x'] += paddle_speed
    else:
        bci_move = bci_command.get('command', 'NEUTRAL')
        if bci_move == 'LEFT': state['player_x'] -= paddle_speed
        elif bci_move == 'RIGHT': state['player_x'] += paddle_speed
    state['player_x'] = max(paddle_width / 2, min(GAME_WIDTH - paddle_width / 2, state['player_x']))
    if state['ai_x'] < state['ball_x']: state['ai_x'] += ai_speed
    if state['ai_x'] > state['ball_x']: state['ai_x'] -= ai_speed
    state['ai_x'] = max(paddle_width / 2, min(GAME_WIDTH - paddle_width / 2, state['ai_x']))
    # Rescale ball velocity to match current ball_speed slider (live tuning)
    current_speed = (state['ball_vx']**2 + state['ball_vy']**2) ** 0.5
    if current_speed > 0.01:
        scale = ball_speed / current_speed
        state['ball_vx'] *= scale; state['ball_vy'] *= scale
    else:
        state['ball_vy'] = -ball_speed
    state['ball_x'] += state['ball_vx']; state['ball_y'] += state['ball_vy']
    if state['ball_x'] <= BALL_RADIUS or state['ball_x'] >= GAME_WIDTH - BALL_RADIUS: state['ball_vx'] *= -1
    if state['ball_vy'] > 0 and state['ball_y'] + BALL_RADIUS >= GAME_HEIGHT - PADDLE_HEIGHT:
        if abs(state['player_x'] - state['ball_x']) < paddle_width / 2 + BALL_RADIUS:
            state['ball_vy'] *= -1; state['ball_vx'] += (state['ball_x'] - state['player_x']) * BALL_SPIN_FACTOR
            state['ball_y'] = GAME_HEIGHT - PADDLE_HEIGHT - BALL_RADIUS
    if state['ball_vy'] < 0 and state['ball_y'] - BALL_RADIUS <= PADDLE_HEIGHT:
        if abs(state['ai_x'] - state['ball_x']) < paddle_width / 2 + BALL_RADIUS:
            state['ball_vy'] *= -1; state['ball_vx'] += (state['ball_x'] - state['ai_x']) * BALL_SPIN_FACTOR
            state['ball_y'] = PADDLE_HEIGHT + BALL_RADIUS
    if state['ball_y'] - BALL_RADIUS > GAME_HEIGHT:
        state['ai_score'] += 1; p_score, a_score = state['player_score'], state['ai_score']
        state = get_initial_game_state(); state.update({'player_score': p_score, 'ai_score': a_score})
        state['ball_vy'] = -ball_speed
    elif state['ball_y'] + BALL_RADIUS < 0:
        state['player_score'] += 1; p_score, a_score = state['player_score'], state['ai_score']
        state = get_initial_game_state(); state.update({'player_score': p_score, 'ai_score': a_score})
        state['ball_vy'] = -ball_speed
    return state

# ==============================================================================
# === 5b. RECORDING-MODE SERVER CALLBACKS ======================================
# ==============================================================================
if RECORDING_MODE:
    @app.callback(
        Output('recording-events-store', 'data', allow_duplicate=True),
        Input('recording-events-store', 'data'),
        State('recording-state-store', 'data'),
        State('app-status-store', 'data'),
        prevent_initial_call=True,
    )
    def consume_recording_events(events_payload, rec_state, app_status):
        """For each new press/release event in the store, insert a BrainFlow
        marker and append to the Python-side mirror used by the finally-block
        save. Idempotent via the seq counter so re-firing doesn't double-log.
        """
        events_payload = events_payload or {}
        events = events_payload.get('events') or []
        seq = events_payload.get('seq', 0)
        if seq <= recording_session['last_event_seq']:
            return no_update

        # Walk all events; only record those we haven't seen yet.
        # We use len(recording_session['events']) as the high-water mark since
        # the JS store accumulates monotonically.
        already = len(recording_session['events'])
        new = events[already:]
        if not new:
            recording_session['last_event_seq'] = seq
            return no_update

        status = (app_status or {}).get('status', '')
        trial_idx = (rec_state or {}).get('trial_idx', -1)
        side      = (rec_state or {}).get('side') or ''

        for ev in new:
            kind = ev.get('kind')
            if kind == 'press':
                safe_insert_marker(MARKER_PRESS)
            elif kind == 'release':
                safe_insert_marker(MARKER_RELEASE)
            ev_full = {
                'kind':         kind or '',
                'side':         side if status == 'RECORD_TRIAL' else '',
                't_browser_ms': float(ev.get('t_browser_ms', 0.0)),
                'trial_idx':    int(trial_idx if status == 'RECORD_TRIAL' else -1),
            }
            recording_session['events'].append(ev_full)

        recording_session['last_event_seq'] = seq
        return no_update

    @app.callback(
        Output('eeg-live-store', 'data'),
        Input('status-interval', 'n_intervals'),
        State('app-status-store', 'data'),
        prevent_initial_call=True,
    )
    def update_eeg_live_status(_, app_status):
        """Best-effort 'is the headset actually streaming?' readout. Fires
        every 500 ms; reads the last ~0.5 s of board data and computes per-
        channel mean/std for the live indicator.
        """
        status = (app_status or {}).get('status', '')
        if not status.startswith('RECORD_'):
            return no_update
        if board is None:
            return {'streaming': False, 'reason': 'no board object'}
        try:
            if not board.is_prepared():
                return {'streaming': False, 'reason': 'session not prepared'}
            n = int(sampling_rate * 0.5) if sampling_rate else 125
            data = board.get_current_board_data(n)
        except Exception as e:
            return {'streaming': False, 'reason': str(e)}
        if data is None or data.size == 0 or data.shape[1] == 0:
            return {'streaming': False, 'reason': 'no samples'}
        chans = []
        for ch_idx in (bci_eeg_channels or [])[:4]:
            ch = data[ch_idx]
            chans.append({
                'mean':   float(np.mean(ch)),
                'std':    float(np.std(ch)),
                'latest': float(ch[-1]),
            })
        return {
            'streaming':  True,
            'n_samples':  int(data.shape[1]),
            'channels':   chans,
            'srate':      int(sampling_rate) if sampling_rate else None,
        }

    @app.callback(
        Output('recording-save-status-store', 'data'),
        Output('app-status-store', 'data', allow_duplicate=True),
        Input('recording-final-store', 'data'),
        State('app-status-store', 'data'),
        prevent_initial_call=True,
    )
    def save_recording(final_payload, app_status):
        if not final_payload:
            return no_update, no_update
        if (app_status or {}).get('status') != 'RECORD_DONE':
            return no_update, no_update
        try:
            path = save_session_npz(final_payload, incomplete=False)
            new_status = {**(app_status or {}), 'status': 'RECORD_SAVED'}
            return {'saved_path': path, 'error': None}, new_status
        except Exception as e:
            print(f"[record] save_recording failed: {e}")
            new_status = {**(app_status or {}), 'status': 'RECORD_SAVED'}
            return {'saved_path': None, 'error': str(e)}, new_status

# ==============================================================================
# === 6. STATE MACHINE AND FEEDBACK PLOTS ======================================
# ==============================================================================
@app.callback(
    Output('status-display', 'children'),
    Output('app-status-store', 'data'),
    Output('calibration-store', 'data', allow_duplicate=True),
    Output('game-state-store', 'data', allow_duplicate=True),
    Output('bci-interval', 'disabled'),
    Output('game-interval', 'disabled'),
    Output('recording-state-store', 'data', allow_duplicate=True),
    Input('status-interval', 'n_intervals'),
    Input('pause-button', 'n_clicks'),
    Input('restart-button', 'n_clicks'),
    State('app-status-store', 'data'),
    State('calibration-store', 'data'),
    State('recording-state-store', 'data'),
    prevent_initial_call=True
)
def manage_app_flow(status_n, pause_clicks, restart_clicks, app_status, cal_data, rec_state):
    triggered_id = ctx.triggered_id if ctx.triggered_id else 'status-interval'
    status = app_status.get('status', 'STARTING'); countdown = app_status.get('countdown', 0)
    new_status, new_cal_data, new_game_state = status, no_update, no_update
    new_rec_state = no_update
    if triggered_id == 'restart-button' and restart_clicks > 0:
        new_status = 'STARTING'
        new_cal_data = {'scores_left': [], 'scores_right': [], 'scores_rest': [], 'thresholds': None}
        new_game_state = get_initial_game_state()
    elif triggered_id == 'pause-button' and pause_clicks > 0 and not RECORDING_MODE:
        # Pause/resume is meaningless in recording mode; ignore.
        new_status = 'PAUSED' if status != 'PAUSED' else 'PLAYING'
    elif triggered_id == 'status-interval':
        if status == 'STARTING':
            if RECORDING_MODE:
                new_status = 'RECORD_INIT'
                countdown = 0
            elif NO_BOARD_MODE:
                new_status = 'PLAYING'
            else:
                new_status, countdown = 'CALIBRATING_LEFT', CALIBRATION_DURATION_S
        elif status == 'RECORD_INIT':
            # Brief pause to let the JS refresh-rate probe (~1s) finish, then prompt.
            new_status = 'RECORD_READY'
            countdown = 0
            recording_session['ready_press_baseline'] = len(recording_session['events'])
        elif status == 'RECORD_READY':
            baseline = recording_session.get('ready_press_baseline', 0)
            new_events = recording_session['events'][baseline:]
            if any(e.get('kind') == 'press' for e in new_events):
                # First press in READY = session start. Flush BrainFlow's pre-
                # session ring buffer so the saved EEG starts at t=0 of the
                # actual session, not stream-start. Trial 0 is LEFT by convention.
                try:
                    if board is not None and board.is_prepared():
                        board.get_board_data()  # discards everything buffered
                except Exception as e:
                    print(f"[record] pre-session buffer flush failed: {e}")
                safe_insert_marker(MARKER_SESSION_START)
                safe_insert_marker(MARKER_CUE_LEFT)
                recording_session['started'] = True
                recording_session['started_at_iso'] = datetime.now(timezone.utc).isoformat()
                new_rec_state = {'trial_idx': 0, 'side': 'L', 'phase_start_t': time.time()}
                new_status = 'RECORD_TRIAL'
                countdown = RECORD_TRIAL_DURATION_S
        elif status == 'RECORD_TRIAL':
            countdown -= 0.5
            if countdown <= 0:
                safe_insert_marker(MARKER_CUE_OFFSET)
                new_status = 'RECORD_REST'
                countdown = RECORD_REST_DURATION_S
        elif status == 'RECORD_REST':
            countdown -= 0.5
            if countdown <= 0:
                next_idx = (rec_state or {}).get('trial_idx', 0) + 1
                if next_idx >= TOTAL_TRIALS:
                    safe_insert_marker(MARKER_SESSION_END)
                    new_status = 'RECORD_DONE'
                    countdown = 0
                else:
                    next_side = 'L' if next_idx % 2 == 0 else 'R'
                    new_rec_state = {'trial_idx': next_idx, 'side': next_side, 'phase_start_t': time.time()}
                    safe_insert_marker(MARKER_CUE_LEFT if next_side == 'L' else MARKER_CUE_RIGHT)
                    new_status = 'RECORD_TRIAL'
                    countdown = RECORD_TRIAL_DURATION_S
        elif status == 'RECORD_DONE':
            # Save is owned by save_recording (fires on recording-final-store push).
            pass
        elif status == 'RECORD_SAVED':
            pass
        elif status.startswith('CALIBRATING'):
            countdown -= 0.5
            if countdown <= 0:
                if status == 'CALIBRATING_LEFT': new_status, countdown = 'CALIBRATING_RIGHT', CALIBRATION_DURATION_S
                elif status == 'CALIBRATING_RIGHT': new_status, countdown = 'CALIBRATING_REST', CALIBRATION_DURATION_S
                elif status == 'CALIBRATING_REST': new_status = 'ANALYZING'
        elif status == 'ANALYZING':
            mean_left = np.mean(cal_data['scores_left']) if cal_data['scores_left'] else 0
            std_left = np.std(cal_data['scores_left']) if cal_data['scores_left'] else 0.1
            mean_right = np.mean(cal_data['scores_right']) if cal_data['scores_right'] else 0
            std_right = np.std(cal_data['scores_right']) if cal_data['scores_right'] else 0.1
            threshold_left = mean_left - CALIBRATION_THRESHOLD_STD_FACTOR * std_left
            threshold_right = mean_right + CALIBRATION_THRESHOLD_STD_FACTOR * std_right
            cal_data['thresholds'] = {'left': min(threshold_left, -MIN_THRESHOLD_GAP/2), 'right': max(threshold_right, MIN_THRESHOLD_GAP/2)}
            new_cal_data = cal_data; new_status, countdown = 'READY', 3
        elif status == 'READY':
            countdown -= 0.5
            if countdown <= 0: new_status = 'PLAYING'
    msg = ""
    if new_status == 'CALIBRATING_LEFT': msg = f"Focus on the LEFT flicker... {int(max(0, countdown))}"
    elif new_status == 'CALIBRATING_RIGHT': msg = f"Focus on the RIGHT flicker... {int(max(0, countdown))}"
    elif new_status == 'CALIBRATING_REST': msg = f"Look at the CENTER (rest)... {int(max(0, countdown))}"
    elif new_status == 'ANALYZING': msg = "Analyzing calibration data..."
    elif new_status == 'READY': msg = f"Get Ready! Starting in {int(max(0, countdown)) + 1}..."
    elif new_status == 'PLAYING':
        msg = "NO BOARD — keyboard only (A/D)" if NO_BOARD_MODE else "PLAYING! Use 'A' and 'D' to override."
    elif new_status == 'PAUSED': msg = "PAUSED"
    elif new_status == 'RECORD_INIT': msg = "Recording — initializing display probe..."
    elif new_status == 'RECORD_READY': msg = "Recording — press SPACE to begin"
    elif new_status == 'RECORD_TRIAL':
        side_label = 'LEFT' if (rec_state or {}).get('side') == 'L' else 'RIGHT'
        idx = (rec_state or {}).get('trial_idx', 0) + 1
        msg = f"Trial {idx}/{TOTAL_TRIALS} — look {side_label} ({int(max(0, countdown))}s)"
    elif new_status == 'RECORD_REST': msg = f"Rest ({int(max(0, countdown))}s)"
    elif new_status == 'RECORD_DONE': msg = "Saving session..."
    elif new_status == 'RECORD_SAVED': msg = "DONE — session saved"
    if NO_BOARD_MODE:
        bci_interval_disabled = True
    else:
        bci_interval_disabled = not (new_status.startswith('CALIBRATING') or new_status == 'PLAYING')
    game_interval_disabled = (new_status == 'PAUSED')
    app_status_out = {'status': new_status, 'countdown': countdown}
    return msg, app_status_out, new_cal_data, new_game_state, bci_interval_disabled, game_interval_disabled, new_rec_state

@app.callback(
    Output('psd-plot', 'figure'), Output('control-plot', 'figure'),
    Input('bci-interval', 'n_intervals'),
    State('bci-command-store', 'data'), State('calibration-store', 'data'),
    prevent_initial_call=True
)
def update_feedback_plots(_, bci_command, cal_data):
    if board is None:
        raise PreventUpdate
    data = board.get_current_board_data(fft_samples)
    if data.shape[1] < fft_samples: raise PreventUpdate
    y_data = data[bci_eeg_channels[0]]; y_processed = preprocess_eeg_window(y_data).flatten()
    y_win = y_processed * np.hanning(len(y_processed)); N = len(y_win)
    yf = np.fft.fft(y_win); xf = np.fft.fftfreq(N, 1.0/sampling_rate)[:N//2]
    psd = 10 * np.log10(np.abs(yf[0:N//2])**2 + 1e-12)
    psd_fig = go.Figure(layout=go.Layout(title=f'Live PSD (Channel {bci_eeg_channels[0]})', template='plotly_dark', xaxis_title='Frequency (Hz)', yaxis_title='Power (dB)'))
    psd_fig.add_trace(go.Scatter(x=xf, y=psd, mode='lines', name='PSD'))
    for h in range(1, CCA_NUM_HARMONICS + 1):
        opacity = 1.0 / h
        psd_fig.add_vline(x=SSVEP_FREQ_LEFT * h, line_dash="dash", line_color="cyan", opacity=opacity, annotation_text=f"{SSVEP_FREQ_LEFT*h:.1f}Hz" if h==1 else "")
        psd_fig.add_vline(x=SSVEP_FREQ_RIGHT * h, line_dash="dash", line_color="magenta", opacity=opacity, annotation_text=f"{SSVEP_FREQ_RIGHT*h:.1f}Hz" if h==1 else "")
    psd_fig.update_xaxes(range=[FILTER_LOW_CUT_HZ, FILTER_HIGH_CUT_HZ + 5])
    smoothed_score = (bci_command or {}).get('smoothed_score', 0.0)
    control_fig = go.Figure(layout=go.Layout(title='Live BCI Score (Smoothed)', template='plotly_dark', yaxis=dict(range=[-3.0, 3.0])))
    control_fig.add_trace(go.Bar(x=['Score'], y=[smoothed_score], marker_color=['cyan' if smoothed_score < 0 else 'magenta']))
    if cal_data and cal_data.get('thresholds'):
        thresholds = cal_data['thresholds']
        control_fig.add_hline(y=thresholds['left'], line_dash="dot", line_color="cyan", annotation_text="Left Threshold")
        control_fig.add_hline(y=thresholds['right'], line_dash="dot", line_color="magenta", annotation_text="Right Threshold")
    return psd_fig, control_fig

# ==============================================================================
# === 7. MAIN EXECUTION ========================================================
# ==============================================================================
def main():
    global board, sampling_rate, bci_eeg_channels, fft_samples, cca_ref_signals

    if NO_BOARD_MODE:
        print("=" * 60)
        print("NO-BOARD MODE: skipping BrainFlow hardware connection.")
        print("Calibration is skipped. Use keyboard 'A' / 'D' to play.")
        print("=" * 60)
        log = logging.getLogger('werkzeug'); log.setLevel(logging.ERROR)
        print("\nDash server is running. Open http://127.0.0.1:8050/ in your browser.")
        app.run(debug=False, use_reloader=False)
        return

    if RECORDING_MODE:
        sid, subj, notes = collect_session_metadata_from_cli()
        recording_session['session_id']    = sid
        recording_session['subject_id']    = subj
        recording_session['headset_notes'] = notes

    params = BrainFlowInputParams(); params.timeout = 15; params.serial_port = "/dev/cu.usbserial-1120"
    board = BoardShim(BOARD_ID, params)
    try:
        print("Connecting to board..."); board.prepare_session()
        sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)
        all_eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
        bci_eeg_channels = all_eeg_channels[:len(CHANNELS_TO_USE)]
        fft_samples = int(sampling_rate * FFT_WINDOW_SECONDS)
        print(f"Board connected. Sampling Rate: {sampling_rate} Hz"); print(f"Using EEG Channels: {bci_eeg_channels} for BCI")

        # --- PREPARE REFERENCE SIGNALS FOR CCA ---
        time_points = np.arange(0, FFT_WINDOW_SECONDS, 1.0 / sampling_rate)[:fft_samples]
        for freq in [SSVEP_FREQ_LEFT, SSVEP_FREQ_RIGHT]:
            refs = [];
            for h in range(1, CCA_NUM_HARMONICS + 1):
                refs.append(np.sin(2 * np.pi * h * freq * time_points)); refs.append(np.cos(2 * np.pi * h * freq * time_points))
            cca_ref_signals[freq] = np.array(refs).T

        print("Starting data stream..."); board.start_stream(450000); time.sleep(FFT_WINDOW_SECONDS)
        log = logging.getLogger('werkzeug'); log.setLevel(logging.ERROR)
        print("\nDash server is running. Open http://127.0.0.1:8050/ in your browser.")
        if RECORDING_MODE:
            print(f"Recording will write to {os.path.join(RECORDINGS_DIR, recording_session['session_id'])}.npz")
            print("Press SPACE in the browser to begin recording trials.")
        else:
            print("The calibration routine will begin automatically.")
        app.run(debug=False, use_reloader=False)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # If a recording session was in flight and the save callback never ran,
        # do a best-effort save here so the data isn't lost on Ctrl-C.
        if RECORDING_MODE and recording_session['started'] and not recording_session['finalized']:
            try:
                save_session_npz({}, incomplete=True)
            except Exception as e:
                print(f"[record] finally-block save failed: {e}")
        if board and board.is_prepared():
            print("Stopping stream and releasing session."); board.stop_stream(); board.release_session()

if __name__ == "__main__":
    main()