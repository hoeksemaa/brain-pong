# Today's plan — BrainPong v1 push

Goal: get BrainPong meaningfully closer to playable. By end of day: real labeled data, an automated latency benchmark, and the flicker bug fixed. UI overhaul deferred.

Playability targets (from prior conversation):
- Latency ≤ 500 ms (intent → paddle move). Stretch: 200 ms.
- Accuracy ≥ 75% per direction.

---

## Step 1 — Fix `render.js` flicker time source ✅ DONE (PR #2, 2026-04-27)

**Why:** `render.js` currently uses `n_intervals * interval_ms` as its clock for the SSVEP flicker. That ties stimulus phase to Dash callback fire count, not wallclock. If the browser throttles or a callback skips, the effective flicker frequency drifts off 10/15 Hz — and the Python-side CCA references are built against true wallclock, so they desync. This bug poisons everything downstream, including any data we record.

**What we did (scope grew during execution):**
- Built `tools/refresh-rate.html`, a standalone browser probe for measuring rAF rate + jitter.
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
- 15 s trials, 20 LEFT + 20 RIGHT, strict L/R alternation, 3 s rest w/ central fixation cross.
- Cue: centered arrow `←` / `→`. No text, no audio.
- Keypress: press at gaze onset, release at offset. Steady-state window = `[t_press + 1.5 s, t_release]`.
- Metadata captured liberally — protocol version, subject id, sampling rate, channels used, **actual measured stimulus freqs** (not nominal 10/15), display refresh, browser UA, filter chain, headset notes.
- File format: one `.npz` per session at `recordings/<YYYYMMDD-HHMMSS>.npz` with `eeg`, `eeg_t`, `events`, `edge_log`, `metadata` arrays.
- Loading snippet (5 lines, numpy + matplotlib) documented inside the protocol.

---

## Step 3 — Add `--record` flag and recording schema

**Why:** we currently log nothing. Need raw EEG + ground-truth event timestamps on disk.

**Do:**
- Add `--record [session_id]` CLI flag to `pong_game_brainflow.py`.
- New "recording mode" in the Dash app that:
  - Hides game UI; shows protocol prompts.
  - Captures keydown/keyup with `t = performance.now()` (and translates to a shared clock with the EEG stream).
  - Dumps raw EEG buffer + event log to `recordings/<session_id>.npz` (or `.h5`) with metadata: sampling rate, channel indices, stimulus freqs, protocol version.
- Format choice: probably `.npz` for simplicity. Include event log as a structured array.

**Time:** 1–2 hrs.

---

## Step 4 — Collect data with the headset on

**Why:** the actual point. Without recordings, none of the rest works.

**Do:**
- Wear the cap. Run the recording protocol from Step 2.
- Aim for ~20 trials per direction. Probably ~15–20 min of session time including setup.

---

## Step 5 — Sanity check the data

**Why:** if the data is garbage, every downstream step is garbage. 5 min of looking at a plot now saves a day later.

**Do:**
- Quick notebook / script: load one LEFT trial, plot the PSD on the steady-state window.
- Expected: a clear peak at 10 Hz (and ideally harmonics at 20 / 30 Hz). Magnitude should be visibly above the surrounding noise floor.
- Same for one RIGHT trial → expect a peak at 15 Hz.
- If no peak: STOP. Debug headset contact / electrode placement / filter chain before continuing.

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

## Explicitly deferred to another day

- **UI overhaul / design system.** Decoupled from BCI quality, doesn't fit in today, will be half-assed if jammed in. Own day.
- **SSVEP frequency sweep (10/15 → 30/40, etc.).** Don't preemptively switch — once the bench exists, sweep frequencies as a benchmark experiment and pick empirically. Right tool for the job is the bench we're building today.
- **ML classifier on calibration data.** Needs the data corpus from today; build tomorrow at earliest.
- **Spatial filtering (CAR, Laplacian), FBCCA, TRCA.** All wait for the bench.
- **Pause-from-calibration bug fix.** Known but doesn't block today.

---

## Risks / things that could blow up the day

- Headset contact is bad (~50% reliability per prior notes) → Step 5 fails → debug eats the day. Mitigation: budget for it; if contact is the issue, pivot to algorithmic work using prior recordings (if any) or accept that today produces a record-only artifact.
- Protocol design takes longer than 30 min. Mitigation: timebox it; perfect is the enemy.
- `--record` flag scope creeps into a full benchmark mode. Mitigation: today's record flag does the minimum — raw EEG + event log to disk. That's it.
