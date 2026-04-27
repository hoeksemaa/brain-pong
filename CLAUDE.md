# BrainPong — Claude project notes

Single-binary Dash app: SSVEP-driven Pong using a Cerelog X8 EEG board over BrainFlow.

## File map

- `pong_game_brainflow.py` — everything server-side: BrainFlow I/O, DSP, sklearn CCA, Dash layout, all callbacks, state machine, feedback plots. ~475 LoC.
- `assets/render.js` — clientside canvas renderer + SSVEP flicker stimulus. Auto-loaded by Dash from the `assets/` folder.
- `requirements.txt` — pinned-by-name (not version) deps: `brainflow dash plotly numpy scikit-learn`.
- `ROADMAP.md` — future improvements organized as fun/UX, signal+ML, hardware, user testing.
- `plans/today.md` — the active day's plan.
- `plans/automated-benchmark-test-suite.md` — longer-arc plan: build a latency+accuracy benchmark.
- `README.md` — three lines, mostly aspirational.

## Running it locally

There is a project-local venv at `.venv/` (Python 3.13). **Always activate it before running any Python script in this repo** — otherwise BrainFlow / Dash / sklearn imports will fail or pull from the wrong interpreter.

```bash
source .venv/bin/activate
python pong_game_brainflow.py            # real hardware mode
python pong_game_brainflow.py --no-board # keyboard-only dev mode
```

If deps drift, `pip install -r requirements.txt` from inside the activated venv. The venv is gitignored.

## Pipeline at a glance

```
Cerelog X8 → serial(/dev/cu.usbserial-1120, hardcoded) → BrainFlow ringbuffer
  → every 300 ms: pull 1.5 s × 4 ch
  → detrend → LP(45) → HP(5) → notch(50,60) → rolling-median(3)
  → CCA against sin/cos refs @ 10 Hz (LEFT) and 15 Hz (RIGHT), 3 harmonics
  → raw_score = (corr_R - corr_L) * 2.5
  → EMA (α=0.4) → calibrated thresholds → {LEFT, RIGHT, NEUTRAL}
  → drives player_x in 16 ms game tick
```

## State machine

`STARTING → CALIBRATING_LEFT (7s) → CALIBRATING_RIGHT (7s) → CALIBRATING_REST (7s) → ANALYZING → READY (3s) → PLAYING ⇄ PAUSED`

`--no-board` flag skips hardware and calibration entirely; arrow goes `STARTING → PLAYING` and the user controls with A/D keys only.

## Three Dash intervals

- `game-interval` 16 ms (~60 Hz physics + render)
- `bci-interval` 300 ms (window step = `FFT_WINDOW_SECONDS × (1 − overlap)`)
- `status-interval` 500 ms (state machine tick; countdowns decrement by 0.5 per fire)

## Playability targets (v1)

- Latency ≤ 500 ms (intent → paddle move). Stretch: 200 ms.
- Accuracy ≥ 75% (look-direction matches paddle direction).

These trade off against each other; the benchmark plan exists to map the curve and pick a knee.

## Known smells / latent bugs

Don't treat these as features:

- **`CHANNELS_TO_USE = [1,2,3,4]` is used as a length only.** `main()` does `all_eeg_channels[:len(CHANNELS_TO_USE)]` — the actual indices are ignored. If we ever want non-contiguous channels (e.g. `[3,5,7,9]`), the indexing logic must change.
- **`scores_rest` is collected during calibration and never read.** `manage_app_flow` only uses left/right scores when computing thresholds. Free 3-class training data sitting on the floor — relevant when we add a learned classifier.
- **Pause-from-calibration is broken.** `manage_app_flow` toggles `'PAUSED' if status != 'PAUSED' else 'PLAYING'`. Unpausing during a calibration phase yeets the user into PLAYING with `cal_data['thresholds'] = None`, so `update_bci_command` early-returns forever.
- **`render.js` flicker time source uses `n_intervals * interval_ms`, not `performance.now()`.** Stimulus phase is tied to Dash callback fire count; if the browser throttles, the effective SSVEP frequency drifts off 10/15 Hz, while the Python-side CCA references are built against true wallclock time — they'll desync.
- **Monitor refresh assumption is implicit and undocumented.** 10/15 Hz are clean divisors of 60 Hz; on 75/120/144 Hz monitors the rendered flicker aliases and the SNR craters.
- **`threading` imported but unused** — vestigial.
- **PSD plot only shows ch0**, even though CCA uses all four channels.
- **Hardcoded serial port** `/dev/cu.usbserial-1120` — not CLI-configurable.
- **No raw EEG logging.** Means no post-hoc analysis, no replay testing, no labeled training data. Phase 1 of the benchmark plan fixes this with a `--record` flag.
- **No `requirements.txt` / `pyproject.toml`** and no tests of any kind.

## Optimization knobs we expect to tune

- `FFT_WINDOW_SECONDS` (currently 1.5) — hard floor on latency. Try 0.5 / 0.3 / 0.2 once benchmarking exists.
- SSVEP frequencies — 10 Hz has a 100 ms cycle; pushing to 30 Hz or 60 Hz on a 60 Hz monitor would shrink the integration floor. Stretch: detect monitor refresh rate at runtime and pick clean divisors.
- Spatial filtering (CAR, Laplacian) — currently zero, each channel filtered independently.
- FBCCA / TRCA over plain CCA — both well-known SSVEP wins.
- Replace FFT-based filter chain with Goertzel at the two target freqs for cheaper narrow-band detection.

## Working agreements

- Don't add features beyond what the active task asks for; this is a small repo and accidental scope creep shows.
- The two `.md`s in this repo (`ROADMAP.md`, `plans/automated-benchmark-test-suite.md`) are user-authored intent. Update them when scope changes; don't fork parallel docs.
- Project owner is interning at Cerelog (the board vendor), so domain feedback on signal processing should be treated as authoritative — verify code-level claims, but defer on hardware/signal intuition.
