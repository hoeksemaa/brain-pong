# Today's plan — BrainPong v1 push

Goal: get BrainPong meaningfully closer to playable. By end of day: real labeled data, an automated latency benchmark, and the flicker bug fixed. UI overhaul deferred.

Playability targets (from prior conversation):
- Latency ≤ 500 ms (intent → paddle move). Stretch: 200 ms.
- Accuracy ≥ 75% per direction.

---

## Step 1 — Fix `render.js` flicker time source

**Why:** `render.js` currently uses `n_intervals * interval_ms` as its clock for the SSVEP flicker. That ties stimulus phase to Dash callback fire count, not wallclock. If the browser throttles or a callback skips, the effective flicker frequency drifts off 10/15 Hz — and the Python-side CCA references are built against true wallclock, so they desync. This bug poisons everything downstream, including any data we record.

**Do:**
- Replace `n_intervals * interval_ms / 1000.0` with a `performance.now() / 1000.0` clock.
- Sanity check: open the page, eyeball the flicker, confirm it looks the same speed as before (it should — just less jittery).

**Time:** ~30 min. Blocks data collection.

---

## Step 2 — Design the recording protocol (on paper)

**Why:** rushed protocol = noisy labels = wasted data. 30 min of paper design saves a day of regret.

**Decide:**
- Trial structure. Proposed: **10–15 s gaze trials, 3 s rest between, 20 trials per direction (LEFT / RIGHT)**.
- Prompt sequence (random or alternating?).
- What the screen shows during rest (blank? fixation cross? center flicker?).
- Keypress semantics: "press space the instant you BEGIN looking; release when prompted to stop."
- Steady-state window for training: discard first 1.5 s after keypress; use seconds 1.5–4.5 (or wider if trials are longer) as the supervised signal.
- Filename / metadata schema for recordings.

**Output:** a short protocol description committed alongside the data.

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
