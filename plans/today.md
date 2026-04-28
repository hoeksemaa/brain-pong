# Today's plan — BrainPong v1 push

Goal: get BrainPong meaningfully closer to playable. By end of day: real labeled data, an automated latency benchmark, and the flicker bug fixed. UI overhaul deferred.

Playability targets (from prior conversation):
- Latency ≤ 500 ms (intent → paddle move). Stretch: 200 ms.
- Accuracy ≥ 75% per direction.

---

## Step 1 — Fix `render.js` flicker time source ✅ DONE (PR #2, 2026-04-27)

**Why:** `render.js` currently uses `n_intervals * interval_ms` as its clock for the SSVEP flicker. That ties stimulus phase to Dash callback fire count, not wallclock. If the browser throttles or a callback skips, the effective flicker frequency drifts off 10/15 Hz — and the Python-side CCA references are built against true wallclock, so they desync. This bug poisons everything downstream, including any data we record.

**What we did (scope grew during execution):**
- Built `refresh-rate.html`, a standalone browser probe for measuring rAF rate + jitter.
- Empirically determined: Chrome on macOS ProMotion delivers stable 120 Hz (median 8.30 ms, p99 9.40 ms, 0 drops/1202 frames). Safari quantizes `performance.now()` to 1 ms and lands at 60 Hz. **Project targets Chrome.**
- Rewrote `assets/render.js` as a self-bootstrapping rAF loop with **frame-counted** flicker — period in frames, not milliseconds, so phase is exact relative to display refresh. (Stronger than the originally-planned `performance.now()` swap.)
- Added runtime refresh-rate measurement at startup (picks nearest of {60, 90, 120}, fails loud if outside tolerance).
- Switched to **black ↔ white** flicker for max-luminance SSVEP response. Repainted paddles + ball + scores so they don't fight the new flicker.
- Added an edge-time ring buffer (`window.dash_clientside.brainpong_edge_log`) so step 3 can log exact stimulus transitions alongside EEG.
- Added `visibilitychange` listener to flag warnings when the tab is hidden.
- Dropped `game-interval` from the render callback's inputs; rAF owns its own clock.
- Documented the Chrome/ProMotion/120 Hz setup in `CLAUDE.md`.

**Verified visually:** `--no-board` mode on Chrome, both bands flicker at correct relative speeds, console reports detected refresh rate.

**Actual time:** ~2 hrs (vs. 30 min estimated). Scope creep was warranted — original plan would have only fixed one of five sources of jitter.

---

## Step 2 — Design the recording protocol (on paper) ✅ DONE (2026-04-27)

**Why:** rushed protocol = noisy labels = wasted data. 30 min of paper design saves a day of regret.

**Output:** [`plans/recording-protocol.md`](recording-protocol.md) — versioned (v1) protocol that step 3 implements verbatim.

**Decisions locked:**
- 10 s trials (revised down from 15 s after first hardware run; users hold cleanly within 10 s and fatigue is lower), 20 LEFT + 20 RIGHT, strict L/R alternation, 3 s rest w/ central fixation cross.
- Cue: centered arrow `←` / `→`. No text, no audio.
- Keypress: press at gaze onset, release at offset. Steady-state window = `[t_press + 1.5 s, t_release]`.
- Metadata captured liberally — protocol version, subject id, sampling rate, channels used, **actual measured stimulus freqs** (not nominal 10/15), display refresh, browser UA, filter chain, headset notes.
- File format: one `.npz` per session at `recordings/<YYYYMMDD-HHMMSS>.npz` with `eeg`, `eeg_t`, `events`, `edge_log`, `metadata` arrays.
- Loading snippet (5 lines, numpy + matplotlib) documented inside the protocol.

---

## Step 3 — Add `--record` flag and recording schema ✅ DONE (2026-04-27)

**Implemented:**
- `--record` CLI flag w/ `--trials N` override; fails loud on `--no-board` or odd N.
- CLI prompts for `subject_id` and `headset_notes`, auto-creates `recordings/<YYYYMMDD-HHMMSS>.npz`.
- New RECORD_INIT → RECORD_READY → RECORD_TRIAL ⇄ RECORD_REST → RECORD_DONE → RECORD_SAVED state machine inside `manage_app_flow`.
- Sub-ms keypress capture clientside via `performance.now()`, batched to `recording-events-store` on every game-interval tick (~16 ms granularity for delivery, but timestamps are sub-ms).
- BrainFlow `insert_marker` injection on every cue/press/release/session boundary — events land natively on the EEG marker channel, no cross-clock sync needed.
- `save_session_npz` writes EEG (per-channel + timestamp + marker channels), events array, edge log array, and a fat metadata dict (protocol version, subject id, sampling rate, actual stimulus freqs from the JS measurement, browser UA, filter chain, marker code map).
- `render.js` extended w/ a record-cue layer: arrow `←` / `→` (~160 px) + trial counter + 15 s countdown bar + `● HOLDING` indicator while space is down + `+` fixation during rest + final saved-path display.
- Edge-log ring buffer bumped 4096 → 65536 (covers ~22 min of two-band flicker).
- Ctrl-C / exception triggers a finally-block partial save w/ `incomplete=True`.
- Hardware path verified: CLI prompts work, recordings dir auto-creates, board-connect attempt reaches the right code path. **Real-hardware end-to-end run is the user's job; I have no headset.**

**Side effects (documented in CLAUDE.md):**
- Cerelog's brainflow fork must be installed (`pip install -e /Users/john/Dev/cerelog/Shared_brainflow-cerelog/python_package`) — upstream brainflow is missing `BoardIds.CERELOG_X8_BOARD`.
- `setuptools<81` pinned in `requirements.txt` because the fork imports `pkg_resources`, which setuptools 81 removed.

---

## Step 4 — Collect data with the headset on ✅ DONE (2026-04-27)

**Done:** 4 sessions on disk in `recordings/`. Canonical session is `20260427-191502.npz` — full 40-trial run (20 L + 20 R) with sponge electrodes, no incomplete trials.

**Per-trial behavior:** response time μ=1.56 s (cue → press), hold time μ=6.61 s (well above the 1.5 s discard + 1 s steady-state minimum). 40/40 trials had a press; 38/40 had a release within the cue window (2 trials held into rest, still valid).

---

## Step 5 — Sanity check the data ✅ DONE (2026-04-28)

**Done:** went well beyond the "plot one trial's PSD" plan. Full eval on the 40-trial canonical session produced:

- **Structural integrity:** 0 dropped samples, dt jitter ≤1 ms, no NaN/flat channels, all 40 trials cleanly extracted from the marker channel.
- **CCA discrimination accuracy: 34/40 = 85%** — well above the v1 75% target. t-statistic between L and R diff distributions = 5.87 (p << 0.001). signal is real.
- **Asymmetric performance:** L→L 100% (20/20), R→R 70% (14/20). all 6 misclassifications are R trials predicted as L. failures have ~3× smaller CCA margins than successes — 15 Hz response is weaker than 10 Hz alpha baseline.
- **No temporal drift:** 80/90/80/90% accuracy across the 4 trial-buckets — 9 min sessions don't fatigue the user.
- **Multi-channel CCA is essential:** per-channel accuracy is 50–55% (chance). Spatial combination across 4 channels is what gets us to 85%.
- **FFT analysis revealed unexpected EOG contamination:** dominant peaks are at 5–8 Hz, not at the stimulus frequencies. R-trials have 5–9× more low-frequency energy than L-trials, almost certainly eye-movement artifact. raw FFT band power "would predict L on every trial" — CCA's pattern matching is what saves us. This points to a high-ROI experiment: bump HPF from 5 → 8 Hz.

---

## Step 6 — Build the automated latency test bench

**Why:** this is the lever that turns optimization from vibes-based into measurement-based. Every algorithm change from here on gets benchmarked.

**Do:**
- **Mock board adapter.** A `MockBoard` class with the same interface as `BoardShim` (specifically `get_current_board_data(n)`), backed by a recording from Step 4. Advances a virtual clock so `bci-interval` callback semantics still work.
- **Test harness.** Pytest test (or plain script) that:
  - Loads a labeled recording.
  - Plays it through the existing preprocessing + CCA + thresholding pipeline.
  - Measures `latency = t_command - t_event` for each ground-truth event.
  - Asserts `latency < LATENCY_BUDGET_MS` (start: 500 ms).
  - Reports the latency distribution + per-direction accuracy.

**Definition of "event time":** the moment the labeled gaze-onset enters the rolling FFT window. Document this — there's another reasonable definition (the keypress itself) and they differ by up to one window length.

**Time:** 2–4 hrs.

---

## Step 7 — Run the bench, write down the numbers

**Why:** establishes the baseline. Every future optimization gets compared against this.

**Do:**
- Run the bench against the recordings.
- Record p50 / p95 latency, accuracy per direction, confusion matrix.
- Drop the numbers into `plans/baseline-results.md` (create it).
- This is the "before" snapshot that justifies any optimization work tomorrow.

---

## Discovered today (post-step-5) — high-ROI experiments to queue after step 6

These are findings from the 40-trial analysis that suggest concrete optimization targets. All gated on step 6 / step 7 (need the bench to validate).

- **Bump HPF cutoff 5 → 8 Hz.** FFT shows EOG artifacts dominate at 5–8 Hz, especially during R-trials. Cutoff at 8 Hz should kill most contamination while preserving stimulus fundamentals (10/15 Hz) and harmonics. Single-line change in `pong_game_brainflow.py`. Sweep on the bench, keep if accuracy stays ≥85% and the spectrum cleans up.
- **Try stimulus pair (8 Hz, 14 Hz) or (12 Hz, 18 Hz).** Move 10 Hz off the natural alpha-band peak to rebalance L/R signal strength. Current setup makes 15 Hz fight 10 Hz alpha at a baseline disadvantage.
- **5-harmonic CCA** (currently 3). Cheap upgrade.
- **Threshold-based classifier from calibration.** L→L is 100%, R→R is 70%. Adding a learned offset bias would close the gap.

## Explicitly deferred to another day

- **UI overhaul / design system.** Decoupled from BCI quality, doesn't fit in today, will be half-assed if jammed in. Own day.
- **SSVEP frequency sweep (10/15 → 30/40, etc.).** Don't preemptively switch — once the bench exists, sweep frequencies as a benchmark experiment and pick empirically. Right tool for the job is the bench we're building today.
- **ML classifier on calibration data.** Needs the data corpus from today; build tomorrow at earliest. (We have 40 labeled trials now — enough to start.)
- **Spatial filtering (CAR, Laplacian), FBCCA, TRCA.** All wait for the bench.
- **Pause-from-calibration bug fix.** Known but doesn't block today.

---

## Risks / things that could blow up the day

- Headset contact is bad (~50% reliability per prior notes) → Step 5 fails → debug eats the day. Mitigation: budget for it; if contact is the issue, pivot to algorithmic work using prior recordings (if any) or accept that today produces a record-only artifact.
- Protocol design takes longer than 30 min. Mitigation: timebox it; perfect is the enemy.
- `--record` flag scope creeps into a full benchmark mode. Mitigation: today's record flag does the minimum — raw EEG + event log to disk. That's it.
