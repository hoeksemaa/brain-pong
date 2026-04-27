# Plan: Automated Benchmark Test Suite for BrainPong

## Goal

Get the game into a state where it *plays like a game*: low-enough latency to feel responsive and high-enough accuracy to feel controllable.

### Concrete success metrics (v1 targets)

- **Latency**: when the user looks left, the paddle moves left within **≤ 500 ms**.
  - Stretch: ≤ 200 ms.
- **Accuracy**: when the user looks left, the paddle moves left **≥ 75%** of the time. Same for right.

These are the two numbers we will benchmark. They trade off against each other — longer integration windows raise accuracy but add latency. The point of the test suite is to map that tradeoff curve so we can pick a sane operating point instead of guessing.

---

## Strategy: two complementary test tracks

### Track A — Automated pipeline benchmark (no human in the loop)

Measures: *latency from "signal present in input stream" → "command emitted by pipeline."*

This isolates the off-board portion of the pipeline (DSP + CCA + threshold + smoothing) — i.e. the part we can actually optimize. Board acquisition latency is fixed by hardware and not in scope here.

**How it works:**
1. Replace the live BrainFlow board with a mock data source that replays pre-recorded EEG.
2. The recording has *labeled ground-truth events*: e.g. "at t = 2.000 s, the subject began looking left."
3. Feed the stream through the existing preprocessing + CCA + thresholding pipeline as if it were live.
4. Measure the wall-clock delta between `event_t` (signal first present in queue) and `command_t` (pipeline emits `LEFT`/`RIGHT`).
5. Assert: `command_t - event_t < LATENCY_BUDGET_MS`. Fail the test otherwise.

**Why this is useful even though it doesn't measure accuracy on real users:**
- Catches regressions when we change the DSP / window size / classifier.
- Lets us optimize the algorithm in tight loops without strapping on the headset.
- Gives a clean number for "how much of the latency budget is software."

### Track B — Manual test bench (human in the loop)

Measures: *both* latency and accuracy, end-to-end, including the user's brain.

**How it works:**
1. User puts on headset, calibrates as normal.
2. Test harness runs a scripted prompt sequence: shows "LOOK LEFT" or "LOOK RIGHT" at random.
3. User presses a key (e.g. spacebar) the instant they begin looking.
4. Harness logs:
   - `t_intent` = key-press time
   - `t_command` = time the BCI pipeline emits a directional command
   - `direction_intended` vs. `direction_emitted`
5. Run **20 trials** per direction.
6. Compute:
   - Median + p95 latency = `t_command - t_intent`
   - Accuracy = `mean(direction_intended == direction_emitted)`

This gives the *real* number we care about, with the human + hardware in the loop.

---

## Phase 1 — Data collection

Prerequisite for Track A. We currently log nothing.

- Add a `--record` CLI flag that dumps the raw EEG buffer + a timestamped event log to disk.
- Run a controlled session:
  - Subject follows scripted prompts (LOOK LEFT for 5 s, REST for 3 s, LOOK RIGHT for 5 s, REST for 3 s, …).
  - Save the EEG stream + event timestamps to `recordings/<session_id>.npz` (or similar).
- Collect ~5–10 minutes of clean labeled data initially.
- One labeled five-second window with a clear left/right transition is the minimum viable input for the first automated test.

**Output:** a corpus of EEG recordings with ground-truth event markers, usable as mock input for Track A and as training/validation data for any future ML classifier.

---

## Phase 2 — Build Track A (automated) first

Cheaper to iterate on. Build before Track B.

1. **Mock board adapter.** Drop-in replacement for `BoardShim` that reads from a saved recording and exposes the same `get_current_board_data(n)` interface, but advances a virtual clock so `n_intervals * BCI_UPDATE_INTERVAL_MS` semantics still work.
2. **Test runner.** A pytest (or plain script) test that:
   - Loads a recording with a known ground-truth `LOOK LEFT @ t = 2.0 s` event.
   - Plays it through the pipeline.
   - Asserts the pipeline emits `LEFT` within `LATENCY_BUDGET_MS` of the event.
   - Reports the actual latency for the report log.
3. **One unit first, then scale.** Start with a single labeled instance. Once the harness works, parameterize over many recordings.
4. **Output:** a CI-runnable test that fails when latency regresses and prints a latency distribution.

---

## Phase 3 — Build Track B (manual bench)

Once Track A is green, layer on the human test.

1. Add a "benchmark mode" to the Dash app:
   - Hides game elements.
   - Shows scripted prompts.
   - Listens for spacebar press as `t_intent`.
   - Logs `(trial_id, direction_intended, t_intent, t_command, direction_emitted)` to CSV.
2. Run 20 trials per direction. Eat the data.
3. Build a small notebook / script that computes p50/p95 latency, accuracy, and confusion matrix from the CSV.

---

## Optimization levers (what we'll actually tune once we can measure)

The benchmark is the means; these are the ends.

### Window length
- Currently `FFT_WINDOW_SECONDS = 1.5`. That's a hard floor of 1.5 s on the latency before any other processing.
- Try shrinking to 0.5 s, 0.3 s, 0.2 s. Plot accuracy at each.
- The classical tradeoff: shorter windows → lower frequency resolution → noisier CCA scores → lower accuracy. Quantify it.

### SSVEP stimulus frequencies
- Current freqs: 10 Hz (left), 15 Hz (right). The 10 Hz signal has a 100 ms cycle, so any detector needs to integrate at least one cycle — that's a floor.
- **Bump to higher frequencies (e.g. 30 Hz / 60 Hz).** Shorter cycles → less integration time needed → less latency.
- Constraint: the stimulus frequency must be a clean divisor of the monitor refresh rate, otherwise the rendered flicker is aliased and the brain doesn't see a clean sinusoid.
  - 60 Hz monitor: 30 Hz (2 frames/cycle) and 60 Hz (1 frame/cycle, but degenerate) work cleanly. 15 Hz (4 frames/cycle) and 10 Hz (6 frames/cycle) also clean.
  - 75 / 120 / 144 Hz monitors: current 10/15 Hz are NOT clean divisors. Real problem.
- **v1 plan**: pick 30 Hz on the assumption of a 60 Hz monitor. Print a startup disclaimer warning users on non-60 Hz displays.
- **Stretch**: detect the monitor's actual refresh rate at runtime and pick stimulus frequencies dynamically to be clean divisors. This would be genuinely cool.

### Algorithm itself
- We may not even need a full FFT. Try:
  - Direct CCA on filtered time-domain signal (already what we do — but with a much shorter window).
  - Goertzel filter at the two target freqs (cheap, narrow-band).
  - Drop the 80% overlap and run shorter non-overlapping windows.
- Every choice gets benchmarked through Track A. No more vibes-based tuning.

---

## Tradeoff to keep in mind

Latency and accuracy trade off directly. Pushing window size from 1.5 s → 0.3 s will tank accuracy if nothing else changes. The point of the test bench is to find the *Pareto frontier* and pick a knee point — probably somewhere we can hit 500 ms with ~75% accuracy, then tighten over time as DSP and ML improve.

---

## Suggested execution order

1. **Phase 1** — add recording, collect ~5 min of labeled data. (Few hours.)
2. **Phase 2.1** — mock board adapter. (Few hours.)
3. **Phase 2.2** — first single-instance latency test. (Few hours.)
4. **Phase 2.3** — parameterize over the recorded corpus, generate latency histogram.
5. **Phase 3** — manual benchmark mode + 20-trial run.
6. **Optimization loop** — sweep window length × stimulus freq × algorithm variant against Track A; validate winners with Track B.

---

## Open questions

- What format for the recording? `.npz` is easy from numpy; could also use BrainFlow's native `write_file` for compatibility.
- Do we need per-channel latency, or just end-to-end? (End-to-end is enough for v1.)
- For Track A, how do we define "signal first present in queue"? Two reasonable definitions:
  - The instant the labeled event timestamp is reached in the playback clock.
  - The instant the new sample first enters the rolling FFT window.
  - These differ by up to one window length and we should be consistent.
- Should Track B's `t_intent` be the keypress, or should we use a visual go-cue with a fixed reaction-time offset subtracted? Keypress is simpler and probably good enough.
- Does the monitor-refresh-matching feature need OS-level APIs, or can we get it from the browser (`window.screen.refreshRate` exists on some browsers, unreliable on others)?
