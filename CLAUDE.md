# BrainPong — Claude project notes

Cerelog X8 EEG/EOG project. **Active work is EOG-based eye-tracking** (2-electrode horizontal setup). The original SSVEP pong game and all prior scripts/data are preserved in `archive/`.

## Repository layout

```
src/brainpong/    — importable library (eog_core, preprocess, detect); `pip install -e .`
scripts/          — runnable entrypoints (record_eog, eog_display, bench, eval_*, plot_*)
data/eog/         — EOG session recordings (.npz, protocol_version 'eog-v1'/'eog-v2')
derivatives/      — regenerable outputs (derivatives/results/ = bench JSON)
tests/            — pytest suite (imports brainpong from src/ via conftest)
archive/          — all prior SSVEP pong work (scripts, recordings, plans, assets)
docs/             — project documentation (roadmaps, protocols)
```

Library code goes in `src/brainpong/`; new runnable scripts go in `scripts/` and
import the library as `from brainpong.X import …` (run after `pip install -e .`).
Don't put EOG work inside `archive/`. Root stays minimal (README, CLAUDE.md,
pyproject.toml only).

## EOG diagnostic dashboard (active build)

Goal: a **live raw-data dashboard** (frontend display + backend store) to tease apart an oscillating-noise problem on the current rig — 2 electrodes at the outer canthi (differential pair into one channel) + a single **active bias-drive** ear clip, **no ground**. Scope is deliberately minimal: display + store raw signal only. **No filtering, no real-signal extraction, no formal `.npz` recording protocol yet.**

Why live (not record→offline): every diagnostic test is an *interactive manipulation* (re-prep electrode, switch to battery, toggle `PD_BIAS`/`MUX`), so the signal must respond under your hands in real time. "Raw" means genuinely raw: pre-filter, pre-HPF, in real units, with an explicit clip/rail indicator — quietly running the existing filter chain would hide the exact failures (#2 CMRR-collapse, #3 DC-rail) we're hunting. Adapt `archive/eog_filtered_plot.py` / `archive/filtered_plot.py`; don't greenfield a new Dash app.

### Datapoints each recording stores

**Per-recording (session metadata, constant within a recording):**

| field | notes |
|---|---|
| Unix start time | when the session began; doubles as a natural ID |
| Sample rate (fs) | ticks/sec (e.g. 250); denominator for the time axis |
| Gain | amp setting (e.g. ×24); converts counts→volts *and* sets the clip ceiling. **Changing gain = new recording** |
| Signal unit | what the stored numbers are: raw counts vs µV (BrainFlow may pre-convert — record which) |
| Board used | single `board` str = model + **which physical unit**, e.g. `"CERELOG_X8 unit:original"`. Model is constant; the unit label (user-asserted via `--board`, default "original") is the part that matters now that a 2nd, possibly-faulty board exists — every recording must pin to one unit. BrainFlow can't distinguish two same-model units over serial, hence asserted not detected. (Numeric board id 65 lives only in the hardcoded `BOARD_ID` constant — runtime needs it, but it's 1:1 with the name so not stored) |
| Person recorded | subject |
| Method / montage | e.g. "2 electrodes at outer canthi (differential) + active bias on ear clip, no ground" |
| Free-text notes | catch-all prose; defaults to blank |
| Tags | `list[str]`, defaults to empty. Structured, machine-filterable labels — esp. **data-problem markers** so a corpus of known-bad recordings stays queryable (e.g. `flatline`, `railing`, `loose-electrode`). `notes` is prose; `tags` is the filter axis. Stored as a 1-D string array (not a wrapped singleton) since it's multi-valued |

**Per-sample (every tick):**

| field | notes |
|---|---|
| Running sample count | 0,1,2,3… — gives the honest evenly-spaced timeline and catches dropped data |
| Raw signal | measured value(s), one per channel, in the unit named above |

**Event markers (sparse):** a label + the sample count (or time) it occurred at, so it pins to the signal.

Sample time is **derived, not stored**: `start_time + count / fs`. Use that uniform `n/fs` axis for any spectral/frequency analysis — *not* host arrival timestamps (USB delivers samples in bursts, so arrival times are lumpy and would smear the PSD). Clip/rail = signal approaching the ceiling implied by gain (≈ ±4.5 V / gain referred to input; ±187 mV at ×24).

**Deliberately deferred** (flagged, not adopted in v1 — don't treat as oversights): structured per-segment instrument condition (`PD_BIAS`/`MUX` state per chunk — for now lives in free-text notes), structured electrode→channel→site map (cf. [[project_electrode_swap_john]]), lead-off / status-word ΔZ readout, and versioning (dashboard SHA / brainflow-fork / schema tag).

### Raw-waveform web viewer — REMOVED

A static Plotly.js viewer (`web/`, npz→decimated-JSON, Vercel-deployed) was the
first attempt at a public raw-waveform website. **Removed 2026-06-22 — it never
worked.** The underlying *goal* (a public, globally-accessible site to view raw
waveforms; data is public by policy) may be revisited, but any future viewer
starts fresh, ideally reading the live SQLite store rather than pre-built JSON.

### Storage format + live architecture (decided, not yet built)

Live capture will use **SQLite (WAL)**, not npz — npz is write-once (RAM-buffered, no crash durability) with a poor metadata story. Schema: `recording` (per-session metadata), `chunk` (one row per `get_board_data()` chunk; signal as `(n_channels,n_samples)` C-order BLOB, dtype/n_channels in `recording`), `event` (sample-pinned markers). Commit per chunk; `PRAGMA busy_timeout`. Architecture: two decoupled processes through the DB file — `recorder.py` (owns board, only acquires+stores) + frontend (reads via WAL, renders); **SQLite is the bus**, no sockets. Sample time derived `unix_start + sample_index/fs`. Existing 9 npz stay frozen ground truth — conform via a load-time sidecar shim, never re-save. Deduced constants for the old corpus: unit=volts, board=CERELOG_X8 (id 65), start=`eeg[10][0]` (EOG)/`started_at_iso` (SSVEP); **gain not stored anywhere — deduce ×24 (Cerelog default), confirm against fork**.

### Oscillating-noise diagnostic protocol (what the dashboard serves)

Turn each hypothesis into a falsifiable prediction; toggle one variable at a time:

1. **Raw PSD** — frequency IDs the culprit: 60.0 Hz locked + harmonics → mains/CMRR (#2); odd non-mains peak → bias-loop self-oscillation (#1); sub-1 Hz / sawtooth → DC-rail (#3).
2. **Input-short (`CHnSET` MUX=001)** — partitions board-internal vs electrode-side: oscillation persists → internal; vanishes → input-side.
3. **`PD_BIAS` off** — splits #1 from #2: dies → #1 (bias loop generating its own oscillation via high-Z ear contact); worse/rails → #2 (mains leaking differential through canthus impedance mismatch ΔZ).

Root of #2 is **electrode-impedance asymmetry** between the two canthi (system CMRR ≈ chip CMRR + ΔZ/Zin_cm), so a matched-electrode bench rig hides it — verdicts must be reproduced on the real canthus montage. See [[project_noise_60hz_cmrr]].

## Archive contents (SSVEP pong — reference only)

- `archive/pong_game_brainflow.py` — SSVEP Dash app: BrainFlow I/O, DSP, CCA, game loop. ~475 LoC.
- `archive/assets/render.js` — canvas renderer + SSVEP flicker stimulus.
- `archive/requirements.txt` — deps for the pong game.
- `archive/refresh-rate.html` — browser rAF rate probe.
- `archive/filtered_plot.py` — gold-standard 8-channel EEG plotter. Verbatim copy from upstream cerelog repo; don't edit unless syncing.
- `archive/eog_filtered_plot.py` — EOG-tuned plotter (the starting point for new EOG work).
- `archive/recordings/` — SSVEP session `.npz` files (protocol_version: 'v1'). Read-only ground truth.
- `archive/plans/` — all prior day-plans and benchmark specs.
- `archive/ROADMAP.md` — future improvements for the pong game.

## Running it locally

There is a project-local venv at `.venv/` (Python 3.13). **Always activate it before running any Python script in this repo** — otherwise BrainFlow / Dash / sklearn imports will fail or pull from the wrong interpreter.

```bash
source .venv/bin/activate
pip install -e .                 # once: makes `brainpong` importable (src/ layout)
python scripts/<script>.py       # entrypoints live in scripts/
```

The library lives in `src/brainpong/` and is imported as `from brainpong.X import …`. `pip install -e .` is editable, so source edits take effect without reinstall; `tests/conftest.py` also adds `src/` to the path so pytest runs even without the install. If deps drift, `pip install -r archive/requirements.txt` from inside the activated venv. The venv is gitignored.

### Cerelog brainflow (real-hardware mode)

Hardware modes (default play, `--record`) reference `BoardIds.CERELOG_X8_BOARD`, which only exists in Cerelog's brainflow fork — **not** upstream PyPI brainflow. To run with the real board, install the fork over the public package:

```bash
source .venv/bin/activate
pip install -e /Users/john/Dev/cerelog/Shared_brainflow-cerelog/python_package
```

The fork imports `pkg_resources`, which is why `requirements.txt` pins `setuptools<81`. Without that pin, you'll hit `ModuleNotFoundError: No module named 'pkg_resources'` at import time. `--no-board` mode also reads the constant at module-load and so requires the fork (or stub) to be installed even though it doesn't use the board.

### Recordings policy

**All `.npz` recordings are committed to git.** Personal project; owner is fine with biosignal data being public. Don't gitignore any recordings dir.

- SSVEP sessions: `archive/recordings/` — protocol_version `v1`, schema defined in `archive/plans/recording-protocol.md`. Read-only ground truth; never mutate.
- EOG sessions: `data/eog/` — protocol_version `eog-v1`/`eog-v2`. Same immutability rules apply.

### SSVEP recording mode (archived — for reference)

```bash
source .venv/bin/activate
python archive/pong_game_brainflow.py --record                 # 40 trials
python archive/pong_game_brainflow.py --record --trials 4      # smoke-test
```

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

**All `.npz` recordings are read-only ground truth.** They back benchmarks, future ML training, and every algorithm comparison. Never mutate the underlying data; never re-save over a session file.

Concrete rules when handling recorded data:

- **Filtering algorithms (DSP, CCA preprocessing, etc.) operate on copies, not the loaded arrays.** BrainFlow's `DataFilter.*` functions mutate their input *in place*. If a caller does `DataFilter.detrend(eeg[i], ...)` on a slice of the loaded npz array, it corrupts the source. Always copy first: `x = np.ascontiguousarray(eeg[i].astype(np.float64))`, then filter `x`.
- **The mock-board adapter (step 6) returns copies** of the requested window, never views into the underlying recording array. Matches BrainFlow's real behavior.
- **`np.savez` over an existing recording is forbidden** unless we're explicitly migrating a session to a new format (and even then, write to `<id>.npz.tmp` first, verify load round-trips, then atomic rename).
- **Outputs (analysis results, baseline numbers, plots) live elsewhere** — `derivatives/` or `docs/`. Never write back into any `data/` recordings directory.

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
- User-authored intent docs live in `docs/` (active EOG) and `archive/ROADMAP.md` / `archive/plans/` (prior SSVEP). Update them when relevant; don't fork parallel docs. New EOG planning docs go in `docs/`.
- Project owner is interning at Cerelog (the board vendor), so domain feedback on signal processing should be treated as authoritative — verify code-level claims, but defer on hardware/signal intuition.
