# BrainPong — Roadmap / Future Improvements

A working list of things to build next, ordered roughly from "most fun, lowest cost" to "most foundational, highest cost."

---

## 1. Fun & Experience

The game today is functionally Pong in a Dash canvas. It works, but it doesn't *feel* like an app you'd want to keep open.

### 1a. Game-mode expansion (Breakout-style hybrid)
- Move beyond plain Pong. Take inspiration from Breakout / brick-breakers / rhythm games where hitting the ball produces *rewards that fall from the opponent's side* (tokens, power-ups, score multipliers, paddle buffs).
- Each successful return should feel earned and visually celebrated — the BCI is high-effort, so the per-hit reward should be high too.
- Possible mechanics: falling power-ups (wider paddle, slower ball, multi-ball, score-2x), combos for consecutive returns, escalating difficulty waves, boss-style enemy paddles.
- Goal: make the game compelling on its own merits, so the BCI is a *thrilling control scheme*, not a novelty wrapped around a 1972 game.

### 1b. Visual design system
- Adopt a real design system rather than ad-hoc inline styles in `pong_game_brainflow.py`. Candidates: a token-based system (colors, spacing, typography, motion) inspired by Linear/Arc/Vercel.
- Concrete steps:
  - Define a palette that doesn't fight with the SSVEP flicker (left = cyan @ 10 Hz, right = magenta @ 15 Hz must remain visually dominant for calibration).
  - Replace Dash's default sliders/buttons with styled components.
  - Pick a typography pair (display + mono) and use it consistently.
  - Add motion: easing on score changes, particle effects on paddle hits, screen shake on goals.
- Goal: a "truly beautiful experience" — not just functional UI.

### 1c. UI/UX polish
- Onboarding: first-run explainer overlay describing what SSVEP is and what the user should do.
- Calibration screen: replace the bare flickering rectangles with a focused visual target (crosshair, dot, animated focus ring) to actually anchor the user's gaze.
- Feedback: real-time confidence meter showing how strongly the BCI is reading left vs right *before* a command commits — currently the user only sees the paddle move.
- End-of-session summary: per-session stats (avg accuracy, peak streak, calibration drift over time).

---

## 2. Technical Accuracy — Signal Processing & ML

The current pipeline is intentionally simple: Butterworth filters + notch + rolling median + sklearn CCA. This works, but leaves a lot of accuracy on the table.

### 2a. State-of-the-art DSP on the 4-channel signal
- Replace or augment the current filter chain with techniques that are standard in modern SSVEP literature but missing here:
  - **Spatial filtering**: Common Average Reference (CAR), Laplacian filtering, or task-specific spatial filters across the 4 channels — currently each channel is filtered independently.
  - **Adaptive artifact rejection**: detect blinks / muscle spikes per window and either reject or interpolate, instead of letting them poison the CCA score.
  - **FBCCA (Filter Bank CCA)**: split the signal into multiple sub-bands and run CCA per band, then combine — meaningfully better than vanilla CCA at the same SNR.
  - **TRCA (Task-Related Component Analysis)**: outperforms CCA when you have calibration data, which we already collect.
  - **Better windowing**: explore shorter windows with overlap (current 1.5 s @ 80% overlap → 300 ms decision rate) vs. longer windows for accuracy. Trade-off: latency vs. confidence.

### 2b. Machine learning layer
- We currently do *no* ML — CCA is correlation, not learned. Add a learned classifier on top:
  - Use the calibration phase (left / right / rest) as labeled training data per session.
  - Try a small classifier (LDA, logistic regression, shallow MLP) on CCA features (corr_L, corr_R, harmonics, frequency power) → command.
  - For per-user models that improve over time: store labeled session data, fine-tune across sessions.
  - Aspirational: end-to-end deep learning (EEGNet, conformer-style architectures) once we have enough data. Almost certainly overkill for 4 channels, but worth a baseline.
- Note: `scores_rest` is currently collected during calibration but discarded — exactly the data we'd need for a 3-class classifier.

---

## 3. Hardware — Data Collection

Software ceiling is set by signal quality, and signal quality is currently set by a sponge-and-saltwater cap that "works maybe 50% of the time."

### 3a. Custom 3D-printed headset
- Design and print a headset sized for the new dry electrodes coming in soon.
- Goals:
  - **Equal or better scalp contact** vs. the current spongy electrodes.
  - **No saltwater dribble** down the neck — current cap is gross and disqualifying for any real user demo.
  - **Repeatable electrode placement** — fixed positions for occipital channels (Oz, O1, O2, POz) so calibration generalizes session-to-session.
  - **Comfort for 10+ minute sessions** — current setup is fatiguing.
- Iterate: print → fit-test → redesign. Track which electrode positions yield the best CCA scores empirically.

---

## 4. User Testing

Cheapest, most informative thing on the list. Once the game is fun-enough and the headset is comfortable-enough:

- Recruit **3 testers** (varied head sizes, hair types, prior gaming experience).
- Have each play a calibration + 5-minute play session.
- Capture:
  - Did calibration succeed? (objective: were thresholds set sensibly)
  - Did they feel in control of the paddle? (subjective)
  - What did they *enjoy*? What was frustrating?
  - Where did they want to look during play vs. where SSVEP forced them to look?
- Use feedback to prioritize the next round of work — don't guess at what matters.

---

## Suggested ordering

1. **User testing first** with the *current* build — cheapest, sets the bar for what "good enough" means before more engineering.
2. **Hardware (3D headset)** — unblocks everything else, since better signal makes both DSP and ML improvements actually visible.
3. **DSP upgrades (FBCCA, spatial filtering)** — fastest software wins, no new infra.
4. **ML classifier on calibration data** — moderate effort, leverages data we already collect.
5. **Game-mode expansion + design system** — highest creative effort, do once the BCI feels reliable so polish isn't wasted on a flaky core.

---

## Open questions

- What's the target device — desktop only, or eventually a packaged app?
- Do we want session data persisted (for cross-session ML) or kept ephemeral (privacy)?
- How many channels does the new hardware support? More electrodes → spatial filtering becomes much more powerful.
- Single-player vs. two-player BCI duel — is this a stretch goal worth designing toward now?
