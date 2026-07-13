# BrainPong — Claude project notes

**Play Pong with your eyes.** A 2-electrode horizontal **EOG** (electrooculography)
montage on a Cerelog X8 reads left/right eye glances and drives the paddle in real
time. Current focus: a **live, working 2-player EOG game** aimed at an in-person
tournament (see `docs/tournament-prep-plan.md`). Prior SSVEP-flicker work is
superseded and lives in `archive/` (reference only).

## Context — how this device works (read first)

Personal hobby project: play a game using eye movements. The sensor is
**receive-only** — it records the tiny naturally-occurring voltages that appear on
the skin near the eyes when you look left or right (standard EOG), and maps them to
paddle moves. There is **no output path to the person**: no actuator, nothing applied
back to the body. A few pieces of engineering shorthand below could be misread out of
context, so:

- **The Cerelog X8 is a biopotential *amplifier*** — an input-only measuring
  instrument, essentially a sensitive voltmeter for microvolt–millivolt EOG. The
  "±187 mV at ×24 gain" figure is its *input measuring range*, i.e. the largest signal
  it can read — not anything applied to a person.
- **The "bias" feedback on the reference electrode is a noise-canceller** — the
  standard right-leg-drive common-mode servo found in every ECG/EEG front-end, there
  to reject mains hum. "No ground" describes the referential montage wiring, not
  electrical safety.
- **The only person measured is the project owner** (plus consenting tournament
  players). No third-party data, no medical or clinical use.
- **The owner's own recordings are shared publicly by choice** — hence "committed to
  git" below.

## Hardware & acquisition → see the `cerelog-x8` skill

The reusable board layer (BrainFlow fork install, serial/multi-board identity,
gain/units/clip ceiling, channel slots, the live-acquisition loop, signal-quality
gotchas) is documented once in the global **`cerelog-x8`** skill so it's reusable
across projects. Project-specific facts:

- **Board id**: `BoardIds.CERELOG_X8_BOARD` (Cerelog fork only). Fork lives at
  `/Users/john/Dev/cerelog/Shared_brainflow-cerelog/python_package`.
- **Serial ports are hardcoded** (not CLI): P1 `/dev/cu.usbserial-1120` (v1.2),
  P2 `/dev/cu.usbserial-1110` (v1.3). Two same-model boards are keyed by port.
- **EOG slots**: right eye → CH1/row0, left eye → CH2/row1 (L/R physically swapped
  PERMANENTLY 2026-07-02 so `diff = ch_R − ch_L` is canonical, rightward → +).
  CH3–8 are firmware-off; only the EOG pair is stored.

## Repository layout

```
src/brainpong/   — importable library (`pip install -e .`)
  eog_core.py    — LIVE realtime glance-pair detector (single source of truth, 1- & 2-player)
  recording.py   — LIVE in-game recorder → eog-v3 .npz (one per player)
  preprocess.py  — OFFLINE eval preprocessors + VIEWER_FILTERS (display filters)
  detect.py      — OFFLINE eval detectors (trial-based; not used by the live game)
  store.py       — viewer SQLite store over frozen npz (metadata/events/trims only)
scripts/         — pong_game_brainflow.py (game), record_eog.py (cued recorder),
                   filtered_plot.py (live signal sanity plot),
                   serve_viewer.py + ingest_npz.py (diagnostic viewer)
web/             — "EOG Studio" diagnostic viewer frontend (canvas + Flask API)
data/eog/        — labeled EOG recordings (eog-v1/v2/v3). Read-only ground truth.
derivatives/     — regenerable outputs (results/ JSON; viewer.db, gitignored)
archive/         — SSVEP pong + early EOG probes + archived offline eval/plot tooling
                   (archive/eog-scripts/). Reference only; nothing live imports it.
docs/            — active: tournament-prep-plan.md (supersedes tournament-roadmap.md);
                   plus electrode/rig characterization docs.
tests/           — pytest suite over the live EOG core (conftest adds src/ to path)
```

Library code → `src/brainpong/`; runnable entrypoints → `scripts/`, importing
`from brainpong.X import …`. Don't put live EOG work in `archive/`. Root stays
minimal (README, CLAUDE.md, pyproject.toml).

## Running it

Project-local venv at `.venv/` (Python 3.13). **Always activate it first** or
BrainFlow/Dash/sklearn imports pull from the wrong interpreter.

```bash
source .venv/bin/activate
pip install -e .                                    # once: makes `brainpong` importable
python scripts/filtered_plot.py                     # is the signal alive? (live plot)
python scripts/record_eog.py --subject john         # record cued LEFT/RIGHT/REST data
python scripts/pong_game_brainflow.py --no-board    # keyboard only, no hardware
python scripts/pong_game_brainflow.py --eog         # 1-player EOG (needs board + fork)
python scripts/pong_game_brainflow.py --2player     # 2-player EOG (two boards)
```

Diagnostic viewer (no board needed — reads committed recordings):

```bash
python scripts/ingest_npz.py     # build derivatives/viewer.db from data/eog/
python scripts/serve_viewer.py   # http://localhost:8770  (needs only flask + numpy/scipy)
```

**Modes** (`pong_game_brainflow.py`, `parse_known_args`): default = keyboard only
(opens P1 board but leaves the detector idle — a mild foot-gun); `--no-board` =
keyboard, no hardware; `--eog` = 1-player EOG; `--2player` = two boards. Guarded:
`--eog + --no-board` and `--2player + --eog` exit with an error.

## The live EOG pipeline (board → paddle)

Per player, per BCI tick (all DSP lives in `eog_core.py`):

```
2 EEG rows (CH1=right, CH2=left) → get_current_board_data(0.4s settle + 0.1s new)
  → differential: (ch_R − ch_L) × 1e6  → HEOG in µV, rightward +
  → detrend → Butterworth LP(30 Hz) → notch(48–52, 58–62) → Butterworth HP(0.1 Hz)
  → Engbert–Kliegl 5-point velocity (µV/s)                  [the detector statistic]
  → detector signal: 'matched' (default; ~120 ms Hann velocity template,
    cross-correlated — integrates over the saccade shape, rescues weak-signal
    players velocity misses) OR 'velocity' (UI toggle)
  → sustained crossing: first run where |signal| > sigma_thr·σ persists ≥ 12 ms
  → glance-PAIR state machine (look-one-way-then-back) → {LEFT, RIGHT}
  → v3 commitment gates (see below) → paddle snaps one of N_PANELS=5 slots (clamped)
```

A **glance pair** = the outgoing saccade + the return saccade of a deliberate look;
keying on that signature (not raw amplitude) keeps it robust to drift. Velocity is
itself a high-pass, so slow drift and HPF-recovery tails self-reject.

**pipeline-v3 "committed, glance-shaped out-and-back" gates** (`eog_core`, 2026-07-13
tuning analysis): on top of the pair machine, a fired command must also clear
(1) **run-count ≤ `RUN_COUNT_MAX`=2** threshold lobes since arming (rejects oscillating
noise / continuous sweeps — the min-wait delays the fire past the extra lobes);
(2) a **per-player self-calibrating amplitude floor** (`AMP_CONFIRM_FRAC`=0.55 × running
median committed-glance peak; rejects casual look-around saccades ~14σ vs committed
glances ~37σ — cuts drift ~73%; never binds below the σ threshold, so weak rigs are
unaffected); and (3) a **baseline-σ quality gate** (`SIGMA_QUALITY_CEIL`=3000: a railing
electrode calibrates to σ 50k+ and is suppressed, not fired on). Direction stays
velocity crossing-order (filtered-position sign is corrupted by the short-window HPF
edge transient). Defaults: `matched` / `GLANCE_WINDOW_S`=0.7 / `ARMED_MIN_WAIT_S`=0.2 /
`REFRACTORY_S`=0.6.

**Dash intervals**: `game-interval` 16 ms (physics + render), `bci-interval` 100 ms
(EOG poll → detector, enabled only in PLAYING/CALIBRATING/TRAINING), `status-interval`
500 ms (app-flow SM, countdowns, live plots, recording driver).

## State machines

**App/UI flow** (500 ms tick): `STARTING → INSTRUCTIONS (5 s) → CALIBRATING (~5.5 s)
→ PLAYING ⇄ PAUSED → GAME_OVER` (first to `POINTS_TO_WIN = 10`). Keyboard-only modes
skip INSTRUCTIONS/CALIBRATING. **TRAINING** is a sibling of PLAYING: the dumbbell
button runs the same INSTRUCTIONS→CALIBRATING flow (app-status `mode: game|training`
picks the landing state) into a no-ball, no-score drill where on-field prompts cue
each player to sweep their paddle full-left then full-right in a loop
(`game_logic.next_training_target`; prompts drawn by render.js, never the dimming
overlay). Pause is inert in training; New Game exits it; re-clicking Training is a
no-op. Training sessions record normally, tagged `training`, with `train_start` +
`pN_target_<dir>` prompt-flip event markers. **Serve holds** (state-level, not app
statuses — a `serve_hold` frame counter the 16 ms physics tick decrements while the
ball stays parked and paddles remain live): PLAYING opens with a 1.5 s
READY→SET→GO! word countdown (ball launches ON the GO), and every post-point serve
holds a plain 1 s beat; render.js draws the words and plays the tick/launch tones
off `serve_hold`/`hold_kind`. TRAINING has **no** countdown — prompts are live the
moment it begins (`_advance_training` forces `serve_hold` to 0). Game resets are
**single-writer**: New Game/Training clicks only bump `game_id` in app-status;
`update_game_physics` (the sole game-state writer) sees the mismatch and rebuilds
the state itself, so an in-flight stale physics write can never clobber a fresh
game (this race previously ate scores/countdowns).

**Per-player detector SM** (`_run_eog_sm`): `CALIBRATING → IDLE → ARMED →
REFRACTORY → IDLE`.
- **CALIBRATING**: collect `EOG_BASELINE_S = 5 s` of the detector signal, set the
  noise floor **σ via robust MAD** (1.4826·MAD, std fallback). This is a single
  *omnidirectional* baseline — there is **no** per-direction (left/right/rest)
  calibration.
- **IDLE→ARMED**: a sustained crossing arms on `first_dir`.
- **ARMED→FIRE**: an opposite-direction crossing within `GLANCE_WINDOW_S = 0.7 s`
  (after `ARMED_MIN_WAIT_S = 0.2 s`) fires the command → REFRACTORY. Timeout → IDLE.
- **REFRACTORY**: `REFRACTORY_S = 0.6 s` dead time → IDLE.

## Detection tuning knobs (live, in-browser sliders — not CLI)

`sigma_thr` (default **6.0**, crossing threshold in σ), `glance_window_s` (**0.7**),
HPF/LPF corners (0.1 / 30 Hz), and the **detector toggle** (default **matched** |
velocity). All are captured into each recording's eog-v3 metadata so a session is
reconstructable. Tune these against recorded corpora, not by feel alone. The v3
commitment gates (run-count / amplitude / quality) are not sliders — they are
constants in `eog_core`.

## Recording & data integrity — CRITICAL

- **In-game recorder** writes one **eog-v3** `.npz` per player to `data/eog/`
  (`recording.save_eog_recording`): the 2 EOG channels `(2, N)` in volts, plus
  session metadata + the live detector config + sample-pinned event markers
  (calib_start / play_start). `record_eog.py` is the separate *cued* recorder
  (saves all 11 board rows). Detector version tag: `pipeline-v3`.
- **All `.npz` recordings are committed to git and are read-only ground truth.**
  They back benchmarks, ML training, and every algorithm comparison. Never mutate;
  never re-save over a session file.
- **DSP operates on copies.** BrainFlow's `DataFilter.*` mutate in place; always
  `x = np.ascontiguousarray(arr.astype(np.float64))` first. The viewer store reads
  signal on-demand from the frozen npz and only ever writes *annotations* (trims),
  never the npz.
- **Outputs live in `derivatives/` or `docs/`**, never written back into `data/`.
  `np.savez` over an existing recording is forbidden (migration only: `.tmp` +
  verify round-trip + atomic rename).

Sample time is **derived** (`unix_start + sample_index / fs`), never host arrival
timestamps — USB bursts make arrival times lumpy and would smear any PSD.

## Known current issues (EOG game — not features)

- **Oscillating-noise false fires — largely mitigated (pipeline-v3).** `_sustained_crossing`
  itself still doesn't reject oscillation, but the v3 gates do: the run-count gate
  catches the mid-band and, once a few real glances prime the per-player amplitude gate,
  supra-threshold oscillation weaker than a committed glance is fully rejected (0 fires
  in `tests/test_oscillation_noise.py`). Residual: **unprimed** slow/fast oscillation can
  still leak a few fires before the first committed glance is seen. Related: the 60 Hz /
  CMRR-asymmetry hunt in the electrode-comparison docs — system CMRR degrades with
  electrode-impedance mismatch (ΔZ) between the two canthi, so verdicts must be
  reproduced on the real montage, not a matched bench rig.
- **Default (no-flag) mode opens the P1 board but never uses the detector** — a
  board is required to launch yet contributes nothing (keyboard only).
- **`_sustained_crossing` is imported but unused** in `pong_game_brainflow.py`
  (it's called internally by `_run_eog_sm`); vestigial import.

## Shipping policy

**NEVER open a PR or merge to main without an explicit user request.** Per change:
branch + commit + push, then **STOP** and wait for "PR + merge" (or equivalent)
before `gh pr create` / `gh pr merge`. Applies even when the change feels complete.
The user is the only reviewer.

## Working agreements

- **Do NOT use the memory tool / auto-memory for this project.** Record any notes,
  findings, or context worth persisting as **local files in the repo** (`docs/` for
  durable notes — e.g. `docs/claude-memory-snapshot.md` holds the prior memory
  export). Local files only; nothing goes into Claude's memory store.
- Don't add features beyond the active task; this is a small repo and scope creep
  shows.
- User-authored intent docs live in `docs/`. Keep the active tournament plan
  current; don't fork parallel docs.
- The owner interns at Cerelog (the board vendor), so **defer to their hardware /
  signal intuition** (verify code-level claims, but treat domain feedback as
  authoritative).
