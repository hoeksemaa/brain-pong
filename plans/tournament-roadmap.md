# Brain Pong Tournament — Roadmap

**Goal:** 8-player single-elimination Brain Pong tournament at Fractal Tech, ~end of May 2026.  
**Hardware:** 2× Cerelog X8 boards (one per player per match), horizontal EOG (2 active electrodes each).  
**Game:** 2-player pong, each player controls their paddle by looking left/right.

---

## Architecture decisions

- **Universal model, no per-player calibration training.** The EOG dipole signal is consistent across subjects; inter-subject variance is mainly amplitude scale and baseline offset, not morphology.
- **15-second baseline normalization at match start.** Each player looks forward for 15s; µ/σ from that window normalizes their signal for the match. Not a calibration session — just a zero-point.
- **Subject ID in all training recordings** so cross-subject generalization can be measured and stratified.

---

## Phases

### Phase 0 — Labeled data capture
*Before leaving for Michigan (~May 14)*

Record cued LEFT/RIGHT/REST sessions using `record_eog.py`. Prioritize recording multiple subjects (not just the developer) since the universal model's validity is only demonstrable cross-subject. Target: ≥5 sessions, ≥3 subjects.

**Exit criterion:** recordings in `recordings/eog/`, visually confirmed event-to-waveform alignment.

---

### Phase 1 — Offline classifier
*Michigan, no hardware*

Build and validate a universal classifier on the labeled recordings. Expected approach: z-score normalization using REST periods → threshold on `(ch_R − ch_L)`. Validate with leave-one-subject-out cross-validation.

**Target metrics:** ≥85% accuracy, ≤300 ms detection latency, <5% false-positive rate at rest.  
**Exit criterion:** eval script reports metrics on held-out subjects.

---

### Phase 2 — Real-time pipeline
*Michigan, mock-board replay*

Wire the Phase 1 classifier into a streaming loop. The 15-second baseline normalization runs at stream start. Validate using recording replay (mock board) — real-time output should match offline metrics.

**Exit criterion:** mock-board replay hits Phase 1 targets.

---

### Phase 3 — Pong game
*Michigan*

Port `archive/pong_game_brainflow.py` from SSVEP/CCA to EOG. Replace the classification block; add 2-player mode (one board stream per player → one paddle each). Include an AI player slot for solo development and testing.

**Exit criterion:** game is playable in `--no-board` / mock-board mode end-to-end.

---

### Phase 4 — Hardware integration + human subjects
*Back in NYC, ~May 21–29*

Connect real boards, run real people through the system. Measure real latency and accuracy. Record additional subjects if LOO accuracy is marginal. Fix electrode placement protocol issues — impedance consistency between players is the expected primary variance source at this stage.

**Exit criterion:** 3 consecutive matches complete, ≥80% in-game accuracy across subjects.

---

### Phase 5 — Tournament rehearsal
*Late May*

Full dry run at Fractal Tech. Shake out bracket logistics, electrode prep time per player, match duration, and anything that breaks under event conditions.

---

### Phase 6 — Tournament
*~End of May 2026*

---

## Key risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Universal model fails on outlier subjects | Medium | LOO eval in Phase 1 surfaces this early; fallback is a quick per-player amplitude calibration (one left, one right) |
| Electrode placement variance ruins consistency | High | Standardize placement protocol in Phase 4; prep checklist per player |
| Game latency feels bad even if metrics are fine | Medium | Tune window size / step during Phase 4 with real subjects |
| 2-board sync issues | Low | Boards are independent streams; sync only matters for the display layer |
