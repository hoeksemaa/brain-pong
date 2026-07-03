# BrainPong — Pre-Tournament Goals & TODOs

_2026-07-01. Rewritten around 6 goals (supersedes the prior matrix plan).
Companion: `docs/eog-flat-signal-debug.md`._

**Target:** 2-player horizontal-EOG pong, one Cerelog X8 (×24, 250 SPS), **all-gold
montage**. **Hard date: Jul 13.**

Six goals below exceed the window → each item tagged **(J13)** must-land-for-tournament
or **(post)** continues-after. Owners `[john]` / `[claude]`.
Dates: **Jul 5** go/no-go · **Jul 6** invite out · **Jul 13** event.
Dependency spine: **G3** clean signal → **G1** dataset → **G4** detection; **G2**
health gates G1/G5; **G5** logistics; **G6** mostly post.

## G1 — Dataset: 10 people, gold/gold

Comparable multi-subject corpus on the all-gold montage (existing 30-rec corpus
predates the gold decision → not a clean basis). `record_eog` already works well.
- [ ] (J13, **tmrw**) Capture **≥10 distinct subjects** on the all-gold montage. `[john]`
- [ ] (J13) Keep it **comparable**: same `record_eog` flags / cue script / durations for every subject (pin the exact invocation, don't vary per person). `[john]`
- [ ] (J13) Commit to `data/eog/`; curate + label saccades. `[john]`

## G2 — Data-health pipeline (3 checks — try in software, document if not possible)

Flag bad per-component state before trusting a recording.
- [ ] (J13) **Firmware check**: print firmware version + register config on launch, verify vs expected. Kills wrong-flash / wrong-unit (2 boards). Feasible. `[claude/john]`
- [ ] (J13) **Electrode-connection check**: wrap `eog_monitor` verdicts (OK/float/rail/dead) + investigate ADS1299 lead-off regs (LOFF / LOFF_STATP/N) → per-channel go/no-go. Feasible. `[claude/john]`
- [ ] (post) **Cable-resistance check**: investigate if the ADS1299 lead-off current source yields a usable resistance proxy; if not, document as a manual multimeter step. Likely partial. `[john]`
- [ ] (J13) Surface all three as one **pre-session self-test**. `[claude]`

## G3 — Reduce noise + prove robustness

Clean saccade signal on an **arbitrary person**; cross-person noise/variance mustn't tank performance.
- [ ] (J13) *Investigate*: on-skin B/C/D on `eog_monitor` — blink (both jump) → L/R hold (anti-phase + diff steps) → rail; ID dominant noise (60 Hz/CMRR vs oscillation vs rail), one variable at a time. `[john]`
- [ ] (J13) *Mitigate*: A/B **behind-ear/mastoid** ref-bias vs on-ear; **bipolar rewire** if blinks show but no anti-phase; verify bias-recenter (CH1/2 off ±187.5 mV rail) under live contact. `[john]`
- [ ] (J13) *Prove*: acceptance = clean anti-phase + diff steps on **≥2 non-John people**, noise within bound. `[john]`
- [ ] (post) Quantify per-subject variance (SNR/offset) across the G1 corpus. `[claude]`

## G4 — Reevaluate detection w/ new data

Fires on real saccades, not noise, across people.
- [ ] (J13) Lock scheme = **single-saccade** (bench ~94% vs glance-pair 8%); wire `detect.sustained_crossing`/`peak_sign` into the game path; verify via `replay_eog`. `[claude]`
- [ ] (J13) **Robust auto-calibration**: `max(k·σ, abs_floor)` + reject dead-flat/railing baselines. `[claude]`
- [ ] (J13) **Oscillation/false-fire gate**: amplitude-consistency + quiet-interval; test on railing/flat recordings. `[claude]`
- [ ] (J13) Cross-subject sanity now: replay `aaron` + `german`. `[claude]`
- [ ] (J13) Re-run `bench` across the full G1 gold corpus (available once tmrw's capture lands); retune/compare algs. `[claude]`

## G5 — Tournament logistics

- [ ] (J13) **Format**: single-elim 8 (7 matches), timed 90 s / higher score; size to rehearsal per-player time (60/7 ≈ 8.5 min/match incl. swap+recal). `[john]`
- [x] (J13) **Throughput — supplies**: gold electrodes ordered; alcohol swabs obtained (enough for now). `[2026-07-02]`
- [ ] (J13) **Throughput — remaining**: paste/tape/gauze; prep-ahead helper; bench 2nd X8 (MAC `8c:4f:00:a9:7b:90`) → 1 vs 2 stations. `[john]`
- [ ] (J13) Hygiene (wipe/swap electrodes); roles (prep / runner / MC). `[john]`
- [ ] (J13) **Keyboard fallback** verified 2-player (`--no-board`; A/D + arrows) → event runs even if signal dies. `[claude/john]`
- [ ] (J13) **Attendance**: Luma event → QR poster `[claude]` → post outside Precision Neuro + Synchron; **invite out by Jul 6**. `[john]`
- [ ] (J13) **Go/no-go Jul 5** (slip before the 6th, don't cancel after); runbook below. `[john+claude]`

## G6 — Unified public site (view + capture + play)

One public site merging portal + game: view data · capture data · play the game.
- [ ] (J13) **Minimum**: game tournament-ready — 2-player calibration flow + port SSVEP smell fixes + basic feel/scoring. `[claude]`
- [ ] (post) **Unify + deploy**: single public site = view (viewer, reads SQLite) + capture (recorder over the SQLite-WAL bus) + play. Public by policy. `[claude]`
- Note: prior static viewer died of deploy friction, not perf → new site reads the live store; capture needs a host-side board owner (browser can't drive BrainFlow serial). See [[project_eog_viewer]].

---

# Reference

**DONE firmware/HW:** SRB1 on (MISC1 0x15 `0x00→0x20`); BIAS_SENSP (0x0D) `0xFF→0x03`,
CH3–8 (0x07–0x0C) `0x60→0x81` powerdown — flashed 2026-06-30 (CH3–8=0; bias-recenter
pending live contact). Flash `arduino-cli ... esp32:esp32:esp32:UploadSpeed=115200`
(921600 fails), port `/dev/cu.usbserial-1120`; backups `*.pre-srb1/biasfix.bak`.
Board healthy (250 Hz). Wiring: L/R canthi→CH1/CH2(+) (P2→CH3/CH4), ref→SRB1,
bias→Bias, (−) empty. Gain ×24, rail ±187.5 mV. [[project_firmware_bias_sense_bug]]

**DONE electrodes:** all gold cups (retire carbon; better CMRR/DC). Ag/AgCl ideal.
Technique: alcohol+abrade; paste=contact, tape=clamp; no bridges; settle 1–2 min.

**Key finding (30 recs):** CUED 21/264 (8%), median 1.30 s; FALSE 73/16.1 min
(4.5/min). 5σ fragile both ways: dead baseline→phantom storm; railing→dead.
Game wants glance-pairs, cues are holds. `bench` single-saccade ~94%. [[project_oscillation_false_fire]]

**Detector constants** (`eog_core.py`): `SIGMA_THR=5`, `MIN_DUR=12ms`,
`GLANCE_WINDOW=0.7s`, `ARMED_MIN_WAIT=0.05s`, `REFRACTORY=0.8s`, `BASELINE=5s`,
0.5–100 Hz + 50/60 notch. Cadence 0.1 s poll / 0.5 s window / 5 s calib.

**Tools:** `eog_monitor.py` (DC-coupled raw + contact verdicts); `replay_eog.py`
(recording → game's detector; cue-hit/latency/phantom); viewer `localhost:8770`
(`serve_viewer.py`, `ingest_npz.py`, `?filter=raw`).

**Signal status:** cleared firmware/wiring/leads/stream. Suspect = on-skin canthi
contact (SRB1 float → identical channels → flat diff). Live display HPs 0.5 Hz →
judge raw.

**Runbook:** launch `... pong_game_brainflow.py --2player`; electrodes per wiring,
confirm mid-range on `eog_monitor`; 5 s eyes-forward calib (redo if dead/railing);
**record every match** (G1 data); FALLBACK `--no-board` keyboard.
