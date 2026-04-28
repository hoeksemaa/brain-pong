# BrainPong — Claude project notes

Single-binary Dash app: SSVEP-driven Pong using a Cerelog X8 EEG board over BrainFlow.

## File map

- `pong_game_brainflow.py` — everything server-side: BrainFlow I/O, DSP, sklearn CCA, Dash layout, all callbacks, state machine, feedback plots. ~475 LoC.
- `assets/render.js` — clientside canvas renderer + SSVEP flicker stimulus. Auto-loaded by Dash from the `assets/` folder.
- `requirements.txt` — pinned-by-name (not version) deps: `brainflow dash plotly numpy scikit-learn`.
- `refresh-rate.html` — standalone browser probe for measuring display rAF rate + jitter. Run when in doubt about flicker precision.
- `filtered_plot.py` — owner's **gold-standard** real-time EEG plotter (8 channels, 5-stage filter chain). Use as the pre-flight signal-quality check before any recording session. Verbatim copy from `cerelog/Shared_brainflow-cerelog/python_package/cerelog_tests/filtered_plot.py`; do not edit unless syncing with upstream.
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

### Cerelog brainflow (real-hardware mode)

Hardware modes (default play, `--record`) reference `BoardIds.CERELOG_X8_BOARD`, which only exists in Cerelog's brainflow fork — **not** upstream PyPI brainflow. To run with the real board, install the fork over the public package:

```bash
source .venv/bin/activate
pip install -e /Users/john/Dev/cerelog/Shared_brainflow-cerelog/python_package
```

The fork imports `pkg_resources`, which is why `requirements.txt` pins `setuptools<81`. Without that pin, you'll hit `ModuleNotFoundError: No module named 'pkg_resources'` at import time. `--no-board` mode also reads the constant at module-load and so requires the fork (or stub) to be installed even though it doesn't use the board.

### Recordings policy

**`recordings/*.npz` is committed to git.** This is a personal project; the owner is fine with biosignal data being public. Don't add `recordings/` to `.gitignore`. Don't omit recordings from PRs.

### Recording mode (`--record`)

```bash
python pong_game_brainflow.py --record                 # 40 trials (20 L + 20 R)
python pong_game_brainflow.py --record --trials 4      # smoke-test session, 4 trials
```

Requires hardware. Errors loud if combined with `--no-board` or with an odd `--trials` count. Prompts for `subject_id` and `headset_notes` at startup, then writes `recordings/<YYYYMMDD-HHMMSS>.npz` on RECORD_DONE. Ctrl-C does a finally-block save with `incomplete=True`.

## Display / browser setup (matters a LOT for SSVEP precision)

The flicker stimulus must hit precise frequencies. The owner runs a 14"/16" 2021 MBP (M1 Pro, Liquid Retina XDR, ProMotion adaptive 24–120 Hz). Empirical findings from `refresh-rate.html`:

- **Use Chrome, not Safari.** Chrome on Apple Silicon delivers stable 120 Hz rAF (measured: 120.5 Hz median, 8.30 ms median Δ, p99 = 9.40 ms, 0 drops over 10 s / 1202 frames). Safari quantizes `performance.now()` to 1 ms (privacy hardening) AND tends to settle ProMotion at 60 Hz instead of 120 for canvas content.
- **Display setting must be "ProMotion"** (System Settings → Displays → Refresh Rate). The fixed-rate options (60 / 59.94 / 50 / 48 / 47.95) are below 120, and macOS does NOT expose a fixed "120 Hz" option for built-in ProMotion displays — that's an Apple API gap, not something we control.
- **Run fullscreen** during recording sessions. ProMotion is more likely to honor 120 Hz when the page is fullscreen and other apps aren't competing for the compositor.
- **No external monitor** assumed. If that changes, re-run the probe — most external displays are 60 Hz only.

### Target stimulus parameters (assumes 120 Hz refresh)

| direction | freq | period (frames) | duty | edge resolution |
|---|---|---|---|---|
| LEFT  | 10 Hz | 12 | 6 on / 6 off | 8.33 ms |
| RIGHT | 15 Hz | 8  | 4 on / 4 off | 8.33 ms |

Flicker is **black ↔ white** (max luminance contrast for strongest SSVEP evoked response), not the cyan/magenta of the original cosmetic palette.

If anything about the display, browser, or refresh rate changes, **re-run `refresh-rate.html` first** before debugging downstream signal issues.

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

## Data integrity — CRITICAL

**`recordings/*.npz` is read-only ground truth.** It backs the step-6 latency bench, future ML training, and every algorithm comparison. Never mutate the underlying data; never re-save over a session file.

Concrete rules when handling recorded data:

- **Filtering algorithms (DSP, CCA preprocessing, etc.) operate on copies, not the loaded arrays.** BrainFlow's `DataFilter.*` functions mutate their input *in place*. If a caller does `DataFilter.detrend(eeg[i], ...)` on a slice of the loaded npz array, it corrupts the source. Always copy first: `x = np.ascontiguousarray(eeg[i].astype(np.float64))`, then filter `x`.
- **The mock-board adapter (step 6) returns copies** of the requested window, never views into the underlying recording array. Matches BrainFlow's real behavior.
- **`np.savez` over an existing recording is forbidden** unless we're explicitly migrating a session to a new format (and even then, write to `<id>.npz.tmp` first, verify load round-trips, then atomic rename).
- **Outputs (analysis results, baseline numbers, plots) live elsewhere** — `plans/baseline-results.md`, notebooks, derived files. Never write back into `recordings/`.

The reason: as algorithms change (HPF cutoff, freq pair, harmonics, classifier), we want to compare apples-to-apples against the same reference recordings. If we ever silently mutate the source data, comparisons across PRs become meaningless.

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
- **`render.js` flicker time source uses `n_intervals * interval_ms`, not `performance.now()`.** Stimulus phase is tied to Dash callback fire count; if the browser throttles, the effective SSVEP frequency drifts off 10/15 Hz, while the Python-side CCA references are built against true wallclock time — they'll desync. **Slated for replacement** with a frame-counted, rAF-driven flicker loop (see "Display / browser setup" above).
- **Monitor refresh assumption was implicit.** Now explicit: target is 120 Hz on Chrome + ProMotion. 10/15 Hz are clean integer divisors at both 60 and 120; the rewrite assumes 120 and falls loud if rAF measurement says otherwise.
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

## Shipping policy

**NEVER open a PR or merge to main without an explicit user request.** Default flow per change:
1. Branch + commit + push to remote.
2. **STOP.** Wait for the user to say "PR + merge" (or equivalent) before running `gh pr create` / `gh pr merge`.

This applies even when the change feels obviously complete. The user is the only reviewer; PRs and merges are user-initiated actions.

## Working agreements

- Don't add features beyond what the active task asks for; this is a small repo and accidental scope creep shows.
- The two `.md`s in this repo (`ROADMAP.md`, `plans/automated-benchmark-test-suite.md`) are user-authored intent. Update them when scope changes; don't fork parallel docs.
- Project owner is interning at Cerelog (the board vendor), so domain feedback on signal processing should be treated as authoritative — verify code-level claims, but defer on hardware/signal intuition.
