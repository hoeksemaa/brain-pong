# BrainPong — improvements master plan

Captured 2026-07-10 from a planning interview over a ~20-item wishlist. Work is
**paused** after Wave 1. This doc is the resume point: it records the decoded
intent of each item, the decisions made, what shipped, and what's still open.

## Status legend
- ✅ done (see "Completed work" for files/lines)
- ⏸️ planned, not started
- 🙋 handed to the user (manual / their machine)
- ❓ needs a decision before building

## ⚠️ Resume state (read first)
All Wave 0 + Wave 1 code changes are **committed? NO — uncommitted on `main`** as of
the pause. Before resuming, decide whether to branch + commit the Wave 1 work
(`feat/wave1-quickwins`) or keep iterating. Files touched:
`scripts/pong_game_brainflow.py`, `src/brainpong/eog_core.py`. 62/62 tests pass;
app imports clean; live-on-board visual check still pending.

Sequencing was chosen as **quick-wins-first**. Tournament (`docs/tournament-prep-plan.md`)
is the deadline driver.

---

## Wave 0 — macOS config (no repo risk)
- 🙋 **`.md` default app** → change Launch Services so Markdown opens in **TextEdit**,
  not Emacs. User is doing this via Finder → Get Info → Open with → TextEdit →
  "Change All". (Alternative if they want me to own it: `brew install duti` then
  `duti -s com.apple.TextEdit .md all`.) — reported done by user.

## Wave 1 — quick, isolated, tournament-relevant fixes  ✅ COMPLETE
- ✅ **Purple power-up direction bug.** The ball-split power-up (1→3 balls) always
  launched the splits *up* toward the top paddle. Fixed to inherit the source ball's
  vertical direction (a down-moving ball splits into three down-moving balls).
- ✅ **Paddle jitter in first second.** Filter/velocity startup transient (a large
  edge artifact on the first freshly-filtered windows) phantom-fired at PLAY onset.
  Added a `PLAY_SETTLE_S = 0.7 s` detector mute, armed on the PLAYING transition
  inside the 100 ms BCI callbacks so it engages before the first possible fire.
  This is a **symptom guard**; the root-cause edge-artifact fix is Wave 3.
- ✅ **No-talking warning.** 🔇 reminder added to the instructions screen, the
  calibration banner, and both EOG play-status messages (talking = jaw/EMG
  contamination throughout, so it's persistent, not calibration-only).
- ✅ **Green flash dots** (the achievable form of "green dot to show where software
  thinks you're looking"). Four corner indicators — P2 top-left/right (red), P1
  bottom-left/right (green) — that blink+glow and fade (~450 ms) when that player's
  detector fires LEFT/RIGHT. Purely reactive to the command stores the state machine
  already emits: **no estimator, no new signal path**. Decision: the ambitious
  continuous "intent predictor" was explicitly **nixed** — we do not want to build an
  absolute-position estimator from the relative/velocity signal.

## Wave 2 — appearance revamp + glitch fixes  ⏸️
- ❓ **Blocked on a glitch inventory.** Either the user lists specific visual
  glitches/bugs, or I do a play-through pass and compile a list for approval.
- Includes: general "revamp appearance," fix visual glitches.

## Wave 3 — detection / signal science  ⏸️
- **Edge/startup artifact rejection.** Root cause of the opening-second jitter and the
  "out-of-scale" / "HPF-LPF endcaps" cluster: filtering a fixed-length window makes
  the band-edge frequencies ring hardest, so the signal shoots to huge values at the
  window edges. Standard fixes to evaluate: window edge-trimming (analyze a 10 s
  window, discard first/last second), reflection/mirror padding, detrend-before-filter.
  Research the cleanest fit for the Butterworth chain in `eog_core.py`/`preprocess.py`.
- **Other filtering/detection tests + re-eval.** Resume the 7prep×5det re-eval on the
  Anthony-and-later corpus (see memory `project_eog_detection_reeval`). This is the
  tournament's real blocker (hit-rate + phantom fires).
- **Training mode** (guided glance practice). A **separate mode/tab** — no enemy
  paddle, no ball. Show the 5 paddle slots + the player's live brainwave trace, and a
  gentle guided regimen ("move the paddle all the way left… now all the way right") so
  a new player gets a feel for the controls before real competition. The green flash
  dots belong here too (live "what am I registering" feedback).
- **Build labelled dataset.** Mostly emergent from the in-game eog-v3 recorder;
  formalize a manifest/export.

## Wave 4 — dev infrastructure  ⏸️
- **claude.md → AGENTS.md migration (global + this project).** Move real content into
  `AGENTS.md`; leave `CLAUDE.md` as a thin pointer so any model/agent works, not just
  Claude. Scope: both `~/.claude` global config **and** this repo.
- **Philosophy trim (do this during the migration).** Strip generic advice that's
  already baked into models at train time ("try harder," "think longer," "raise your
  IQ 2 std devs," etc.). Instruction files should contain **only** what cannot be
  trained in: facts specific to this project, to the user's life, or to their digital
  infrastructure.
- **C11 adoption.** An agentic, tmux-like monitoring tool by a developer named **Atin**
  (A-T-I-N; also appears in the recording corpus as a consenting player,
  `data/eog/*-atin.npz`). "Like tmux but better." Research it, then propose a workflow
  for tmux-like agent monitoring. ("Use C11 for tmux-like things.")
- **Fable.** Route some low-risk subtasks to the Fable model and evaluate. Candidates
  TBD (I pick and report back, unless the user names some).
- **Board skill enrichment.** Extend the global `cerelog-x8` skill with additional
  board info the user has in mind. ("Skills for information for the board.")
- **Meta principle guiding this wave:** "designing a good working environment is more
  important than doing the work directly." Favor the infra that makes later work
  faster/cheaper.

## Wave 5 — future / hardware  ⏸️
- **Isolate USBs.** Research USB isolators / ground-loop mitigation between the two
  boards (related: memory `project_cross_board_noise`, the v1.2-vs-v1.3 domain shift).
  Parked as a future problem.

---

## Already done before this plan (verified, no work needed)
- **Record all data from pong** — the in-game recorder already writes eog-v3 npz.
- **Add 1-player/2-player datapoint** — `n_players` field already stored.
- **Record sigma / HPF / LPF / glance duration** — `sigma_thr`, `hpf_hz`, `lpf_hz`,
  `glance_window_s`, `detector` all already stored in eog-v3 (`recording.py`).
- **Board version field** — `board_version` is stored, BUT: **enforcement** of the
  "v1.2 is always slot/player 1" rule is NOT done yet. That enforcement is the one
  remaining piece of the recording cluster → fold into Wave 3/4 when resumed.

## Decoded cryptic items (for future-me)
- "Purple power one-way" = the ball-split power-up bug (Wave 1, done).
- "c11 / tmux-like things" = C11, Atin's agentic tmux-like tool (Wave 4).
- "Modify .md reader" = macOS default-app change (Wave 0, handed to user).
- "Brainflow artifact at beginning of scale / out-of-scale distribution / HPF-LPF
  endcaps" = filter window edge artifacts (Wave 3).
- "Green dot" = the flash-dot feedback (Wave 1, done); estimator version nixed.

## Completed work — file references
- `scripts/pong_game_brainflow.py`
  - Purple split: preserves `vdir` from source ball's `vy` (was hard-coded `-abs(...)`).
  - `begin_play_settle` imported; armed on PLAYING transition in both BCI callbacks.
  - No-talking text in instructions/calibration/play messages.
  - `_FLASH_DOT_BASE` + four `flash-p{1,2}-{left,right}` divs + clientside fade callback.
- `src/brainpong/eog_core.py`
  - `PLAY_SETTLE_S` constant; `settle_until` in state; `begin_play_settle()`;
    settle-guard branch in `_run_eog_sm`.
