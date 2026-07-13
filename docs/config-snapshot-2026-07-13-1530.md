# BrainPong config snapshot — 2026-07-13, 3:30 PM

Baseline captured **before** a performance-tuning session. If tuning goes poorly,
revert to these values (they are the shipped defaults on branch
`feat/start-countdown`).

- **Git commit:** `0d6b2290e402178d489e9fc94a4e8f08680f701a`
  (`feat(game): READY/SET/GO start countdown + 1 s between-point serve hold`)
- **Branch:** `feat/start-countdown`

> Note: sigma_thr / glance_window_s / hpf_hz / lpf_hz / detector are also live
> in-browser sliders. Values below are the code defaults each slider initializes to.

## Detector / DSP knobs — `src/brainpong/eog_core.py`

| Constant | Value | Meaning |
|---|---|---|
| `EOG_SIGMA_THR` | **6.0** | crossing threshold in units of baseline σ (slider `sigma-thr`, range 1–10, step 0.5) |
| `GLANCE_WINDOW_S` | **0.5** s | max time between the two glances of a pair (slider `glance-window`, 0.1–2.0, step 0.1) |
| `EOG_HPF_HZ` | **0.1** Hz | high-pass corner (slider `hpf`, 0.1–2.0, step 0.1) |
| `EOG_LPF_HZ` | **30.0** Hz | low-pass corner (slider `lpf`, 10–100, step 5) |
| `NOTCH_BANDS` | **((48,52),(58,62)) Hz** | bandstop notches (50/60 Hz) |
| `detector` (default) | **'velocity'** | detection method — 'velocity' \| 'matched' (UI toggle) |
| `MATCHED_TEMPLATE_MS` | **120.0** ms | saccade-velocity template width (matched filter) |
| `EOG_MIN_DUR_MS` | **12.0** ms | min crossing persistence (kills single spikes) |
| `ARMED_MIN_WAIT_S` | **0.05** s | min time before the opposite glance counts |
| `REFRACTORY_S` | **0.8** s | dead time after a fired command |
| `PLAY_SETTLE_S` | **0.7** s | detector muted after PLAY begins |
| `EOG_BASELINE_S` | **5.0** s | baseline collected before σ is fixed (σ via robust MAD, 1.4826·MAD) |
| `PIPELINE_VERSION` | **"pipeline-v2"** | detector version tag |

Filter chain order: `detrend(constant) → Butterworth LP(4th) → bandstop notches → Butterworth HP(4th)`.

## Game / timing knobs — `scripts/pong_game_brainflow.py`

| Constant | Value | Meaning |
|---|---|---|
| `POINTS_TO_WIN` | **10** | first to this wins |
| `N_PANELS` | **5** | discrete paddle slots (CENTER_ZONE=2, MAX_ZONE=4) |
| `INITIAL_BALL_SPEED_Y` | **-4** | serve speed (slider `ball-speed`, 1–12, step 0.5, default abs=4) |
| `AI_REACTION_FRAMES` | **31** | AI paddle reaction delay (1-player mode) |
| `GAME_WIDTH` × `GAME_HEIGHT` | **800 × 800** | square play field |
| `PADDLE_WIDTH` | GAME_WIDTH//N_PANELS − 2 = **158** | |
| `PADDLE_HEIGHT` | **20** | |
| `BALL_RADIUS` | **10** | |
| `POWERUP_RADIUS` / `POWERUP_FALL_SPEED` | **14** / **3** | |
| `START_COUNTDOWN_S` | **1.5** s | New Game / Training entry READY→SET→GO! |
| `SERVE_HOLD_S` | **1.0** s | between-point stationary-ball hold |
| `INSTRUCTIONS_S` | **5.0** s | INSTRUCTIONS screen duration |

### Intervals / acquisition
| Constant | Value |
|---|---|
| `GAME_INTERVAL_MS` | **16** ms (physics + render) |
| `BCI_UPDATE_INTERVAL_MS` | **100** ms (EOG poll → detector) |
| `status-interval` | **500** ms (app-flow SM) |
| `EOG_POLL_S` / `EOG_SETTLE_S` | **0.1** s / **0.4** s |
| `BOARD_STREAM_BUFFER` | **450000** |

### Hardware / recording (not tuning, for completeness)
| Constant | Value |
|---|---|
| `BOARD_ID` | `BoardIds.CERELOG_X8_BOARD` |
| `P1_SERIAL_PORT` / `P2_SERIAL_PORT` | `/dev/cu.usbserial-1120` (v1.2) / `/dev/cu.usbserial-1110` (v1.3) |
| `REC_GAIN` | **24** |
| EOG slots | R → CH1/row0 (`EOG_SLOT_R=0`), L → CH2/row1 (`EOG_SLOT_L=1`) |
