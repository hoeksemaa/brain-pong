# Brain Pong Tournament — Roadmap

**Goal:** Host an EOG Brain Pong tournament. Players control a paddle by looking left/right.
**Deadline:** Thursday **2026-06-25**. Dry run target: Wed 6/24.
**Today:** 2026-06-19 (Fri) → **6 days out.**
**Hardware reality:** 1 confirmed working board + **a 2nd board from Simon, UNVERIFIED** (needs flashing; probably works out of the box, but neither of you is sure). If #2 checks out → **true simultaneous 2-player head-to-head is on the table.** Until verified, keep designing so a one-board tournament works regardless.

> Supersedes the May-12 plan (which assumed 2 boards, end-of-May, and no noise problem). Architecture decisions still in force are kept in "Carried-over decisions" below.

---

## The two things that can sink this (front-load them)

1. **Noise.** Oscillation has blown the signal out of the water before; spike detection just fails when it's present. Unknown cause. **No rig = no tournament.** Critical path.
2. **Board availability.** Simon gave a 2nd board, **unverified** — flash + test it ASAP. Works → simultaneous head-to-head unlocks; doesn't → fall back to one-board score-attack. Resolve early so the format isn't in limbo.

Everything else (game, cal UX, snacks, trophy) is downstream of these two. Order the week so #1 and #2 are answered first; keep a working fallback for both.

---

## Critical path (dependency order)

```
board #2 flash+verify ┐
                      ├─→ tournament FORMAT locked ─→ runtime wiring ─→ dry run ─→ TOURNAMENT
noise: record ─→ pin freq ─→ find cause ─→ prep protocol ─→ clean data ─→ robust detection ─┘
```

---

## Phase 0 — Hardware unblock (DO EARLY)

- [ ] **Flash + verify board #2.** Simon's spare — neither of you is sure it works; probable it's fine post-flash. Flash it, stream from it, sanity-check a clean channel, record a test npz. *Done when:* board #2 streams and you can record from it. **This gates the format decision (Phase 5).**
- [ ] **(Optional) Ping Simon on noise gotchas.** Board's handled, but his intuition on no-ground 2-electrode bias-drive config could save Phase 2 time (draft at bottom). *Done when:* asked, or consciously skipped.

---

## Phase 0b — Logistics (LOW PRIORITY — do Tue/Wed 6/23–6/24)

None of these need much prep, and there's a 3D printer on-site so the trophy has no shipping lead time. Knock them out around the dry run.

- [ ] **[LOW] Print the brain trophy.** 3D printer at the space → just need a model + print time (hours, runs unattended). *Done when:* model sourced and print queued/done.
- [ ] **[LOW] Get spherical snacks** (theme: the pong ball). *Done when:* bought.
- [ ] **[LOW] Recruit 2 dry-run subjects** for Wed 6/24. *Done when:* 2 people confirmed.
- [ ] **[LOW] Line up promo media capture.** Simon wants photos + video of the event as promotional material for Cerelog. Assign a dedicated person (you'll be running the rig, so don't self-assign) + rough shot list: rig/electrode close-ups, players mid-match, the bracket, trophy, candid crowd. *Done when:* a person is committed to shoot on 6/25.

---

## Phase 1 — Pin the noise (6/19–6/20)

You already have the tool: `record_eog.py` now stores genuinely raw signal (all board rows, volts) + `--notes`. Use it to capture the noise, then PSD it. **Per CLAUDE.md, compute PSD on the derived `n/fs` axis, NOT host arrival timestamps** (USB delivers in bursts → smeared PSD).

- [ ] **Reproduce + record the noise.** Recreate whatever conditions produced the blow-out before; record a session with `--notes "noise-repro: <rig state, power source, PD_BIAS, electrode prep>"`. *Done when:* an npz visibly containing the oscillation exists in `recordings/eog/`.
- [ ] **Build a minimal offline PSD/inspect script** (`psd_inspect.py`, repo root). Load an npz → plot raw per-channel trace + Welch PSD per channel on the `n/fs` axis + a clip/rail marker (±4.5 V / gain ≈ ±187 mV @ ×24). Don't gold-plate. *Done when:* it renders a PSD for the noise recording.
- [ ] **Classify the frequency** (CLAUDE.md decision tree): 60 Hz locked + harmonics → mains/CMRR collapse (#2, the leading hypothesis per `project_noise_60hz_cmrr`); odd non-mains peak → bias-loop self-oscillation (#1); sub-1 Hz / sawtooth → DC-rail (#3). *Done when:* you can name the culprit class with a labeled PSD plot.

---

## Phase 2 — Find the cause & lock a rock-solid prep protocol (6/20–6/21)

Toggle **one variable at a time**. Start with the no-firmware-needed manipulations (likely the actual fix per the CMRR hypothesis), then register toggles only if still ambiguous.

**Cheap external manipulations (no register access):**
- [ ] **Battery vs wall power.** Mains/CMRR noise should drop hard on battery. *Done when:* PSD compared, delta recorded.
- [ ] **Electrode contact symmetry.** Re-prep both canthi (clean skin, fresh gel/contact), aim for matched impedance — the #2 root cause is ΔZ asymmetry between the two canthi. *Done when:* a prep procedure that reliably suppresses the noise is written down.
- [ ] **Bias / ear-clip contact.** Verify the active bias ear clip is making good contact; reseat. *Done when:* effect on noise recorded.
- [ ] **Environment.** Distance from laptop charger / monitor / mains cables; dress the leads. *Done when:* effect recorded.

**Register toggles (only if needed — confirm the fork exposes these via `config_board`/firmware FIRST):**
- [ ] **Input-short (`CHnSET` MUX=001).** Partitions board-internal vs electrode-side: persists → internal; vanishes → input-side. *Done when:* verdict recorded.
- [ ] **`PD_BIAS` off.** Splits #1 from #2: dies → #1; worse/rails → #2. *Done when:* verdict recorded.

**Exit:**
- [ ] **Write the prep protocol** (`plans/electrode-prep-checklist.md`): exact steps that produce a clean signal every time, plus a go/no-go check. *Done when:* doc exists and you've reproduced clean signal twice from cold.

> ⚠️ Risk: cause may be hardware and unfixable in 6 days. Fallback baked into the protocol — battery power + matched prep + good bias + a detection-side noise gate (Phase 4). The bench-matched rig hides #2, so **verdicts must be reproduced on the real canthus montage.**

---

## Phase 3 — More clean labeled data (6/21)

Robust detection needs clean ground truth captured under the locked protocol.

- [ ] **Record ≥3 fresh clean sessions** under the new prep protocol (yourself + dry-run subjects if available). *Done when:* npz in `recordings/eog/`, event→waveform alignment eyeballed.
- [ ] **Rebuild the public viewer** (`python web/build.py`) and confirm new sessions render raw. *Done when:* manifest updated, page shows them.

---

## Phase 4 — Detection robustness (6/22)

The detector must fire on real glances and **not** false-fire on residual noise (`project_oscillation_false_fire`: glance-pair detector currently fires phantom commands on oscillation; reproduced in tests; `eog_core.py` is the shared 1p/2p core — modify, don't delete, per `feedback_no_delete_pipeline_fns`).

- [ ] **Re-tune spike/glance detection on clean data.** *Done when:* `bench.py`/eval reports accuracy on the new clean sessions.
- [ ] **Add a noise gate.** If band-power in the noise band exceeds a floor, or the signal is railing/clipping, suppress detection (don't emit a command). *Done when:* feeding a noise recording through the detector yields ~zero false fires.
- [ ] **Validate at-rest false-positive rate.** *Done when:* <5% FP during REST on held-out data.

---

## Phase 5 — Tournament runtime (6/22–6/23)

**Format decision.** Default to **single-board score-attack → bracket** (one-board-safe, zero stream-sync): each player does a timed solo pong run (rallies in 60s, or survival time); higher score advances a single-elim bracket. **If board #2 verifies (Phase 0), upgrade to true simultaneous 2-player head-to-head** — two independent board streams, one paddle each (what the original 2-board design assumed). Choose based on the Phase 0 result + remaining time; don't let head-to-head's extra integration risk the whole event.

- [ ] **Audit current runtime state.** Inspect `eog_core.py` + `pong_game_brainflow.py` — how much real-time EOG play already works end-to-end with the real board vs mock. *Done when:* you can list what's built vs missing.
- [ ] **Real-board end-to-end run.** One player plays a full pong round driven by live EOG glances. *Done when:* paddle tracks intent for a full round, latency feels ≤500 ms.
- [ ] **[HEAD-TO-HEAD ONLY] Dual-board input.** Drive two paddles from two independent boards in one app/loop. Concretely: two `BoardShim` sessions on two serial ports (note `record_eog.py`'s `SERIAL_PORT` is currently hardcoded → parameterize per board), two acquisition reads per tick, the detector run twice (the `eog_core.py` 1p/2p split is the head start — modify, don't delete per `feedback_no_delete_pipeline_fns`), each command → its own paddle. *Done when:* two boards stream concurrently and two paddles move independently from live glances. **Gated on board #2 verifying (Phase 0); skip if running score-attack.**
- [ ] **Fast per-player setup.** Baseline/zero-point (15 s eyes-forward µ/σ normalization, or a quick one-left-one-right amplitude cal) so a new player is playing in <60 s after electrodes are on. *Done when:* timed a cold start < ~2 min total (prep + cal).
- [ ] **Score-attack mode + bracket sheet.** Solo timed run with a visible score; a paper/printable bracket. *Done when:* you can run 2 back-to-back solo runs and compare scores.

---

## Phase 6 — Dry run (Wed 6/24)

- [ ] **Full dry run with 2 subjects** under event conditions. Shake out: prep time per player, electrode swap between players, match duration, noise under a non-developer's skin/prep, score-attack feel. *Done when:* 2 people complete matches start-to-finish; breakages logged.
- [ ] **Fix the top breakages.** *Done when:* the showstoppers from the dry run are resolved or have a documented workaround.

---

## Phase 7 — Tournament (Thu 6/25)

- [ ] Snacks present. [ ] Trophy present. [ ] Bracket sheet printed. [ ] Boards/electrodes/gel/battery charged & packed. [ ] Prep checklist on hand. [ ] Buffer the morning for fixes.
- [ ] **Promo media captured for Cerelog.** Photographer/videographer present and shooting per the shot list (rig close-ups, players mid-match, bracket, trophy, crowd). *Done when:* photos + video are in the can and shared with Simon.

---

## Suggested day-by-day

| Day | Focus |
|---|---|
| **Fri 6/19** | Phase 0 (flash + verify board #2) + Phase 1 (record noise, PSD, classify) |
| **Sat 6/20** | Phase 2 (cause hunt — power/electrodes/bias first) |
| **Sun 6/21** | Phase 2 exit (lock prep protocol) + Phase 3 (clean data) |
| **Mon 6/22** | Phase 4 (robust detection) + start Phase 5 (runtime audit) |
| **Tue 6/23** | Phase 5 (real-board e2e, fast cal, score-attack) + Phase 0b logistics (trophy print, snacks, recruit, media) |
| **Wed 6/24** | Phase 6 (dry run + fixes) + any Phase 0b spillover |
| **Thu 6/25** | Phase 7 (tournament; morning buffer) |

---

## Open decisions (yours to make)

- **Format:** score-attack→bracket (one-board-safe fallback) vs simultaneous head-to-head (now feasible if board #2 verifies in Phase 0).
- **Player count / bracket size** (drives total prep+play time; 8 single-elim = 14 solo runs).
- **Calibration:** universal model + 15 s baseline (carried-over assumption) vs per-player amplitude cal (more robust to inter-subject + prep variance; likely safer given the noise history).
- **Trophy:** local 3D print vs fallback.

---

## Risk / fallback

| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| Noise cause is hardware, unfixable by 6/25 | Medium | Battery + matched prep + bias + detection noise gate; require go/no-go signal check per player |
| Board #2 doesn't work after flashing | Medium | Falls back to one-board score-attack→bracket (needs only one board) |
| Register toggles (MUX/PD_BIAS) not exposed in fork | Medium | Diagnose via external manipulations (power/electrode/env) which need no firmware |
| Non-developer skin/prep reintroduces noise | High | Standardized prep checklist; per-player baseline; gate trials on signal quality |
| Game latency feels bad even if metrics fine | Medium | Tune `FFT_WINDOW_SECONDS`/step during dry run |
| Trophy doesn't arrive | Medium | Local print or fallback trophy chosen today |

---

## Carried-over decisions (from May-12 plan, still in force)

- **Subject ID in every recording** (now also unix_start/gain/montage/notes via `eog-v2-labeled`) for cross-subject stratification.
- **15 s baseline normalization** as the zero-point at match start — *candidate*, but the noise history may force a fuller per-player amplitude cal (see Open decisions).
- Universal-model hypothesis (signal morphology consistent across subjects; variance is amplitude + offset) — validate, don't assume, given noise.

---

## Draft message to Simon (optional — board's already handled)

> Hey Simon — thanks for the spare board, I'll flash + test it. One thing I'm still chasing: an oscillation that blows the EOG signal out of the water — leading hypothesis is 60 Hz mains via CMRR collapse from canthus electrode-impedance asymmetry (no ground, active bias on ear clip). Any quick intuition on bias-drive config or known gotchas on the X8 for a no-ground 2-electrode montage?
