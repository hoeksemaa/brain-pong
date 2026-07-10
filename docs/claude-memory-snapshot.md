# Claude memory snapshot — BrainPong

Exported 2026-07-10 from Claude's per-project auto-memory so the notes live **in
the repo** instead of in Claude's memory store. Going forward Claude records
project notes as local files (see the policy note in `CLAUDE.md`), not memory.

This is a verbatim consolidation of every memory file that existed for this
project at export time (29 files). It's a point-in-time dump for John to review —
some entries are superseded (noted inline where the original said so). Grouped by
type: **user**, **feedback**, **project**, **reference**.

---

## User

### John's BCI environment preference
*(was `user_bci_env_preference.md`)*

John does **NOT** prefer medically-credentialed, doctor-heavy, established
environments — that was a mis-inference (and the Honcho profile still states it
backwards as of 2026-06-02). When he's described companies like Precision
Neuroscience as "medically credentialed, less startup-like, many doctors/serious
people," he was **describing what he observes/expects of BCI startups**, not
stating a desire to work there.

His actual lean is **the opposite**: scrappier, earlier-stage, higher individual
autonomy and ownership. Consistent with his autonomy-focused / founder-control /
individual-agency traits. He still tracks the big credentialed players (Precision,
Synchron) as industry context, but for personal fit, weight scrappy/seed-stage
roles with broad signal-processing ownership higher.

He IS genuinely interested in minimally-invasive neural recording (e.g. Layer
7-style cortical interfaces) as a technical domain — that part stands.

---

## Feedback

### Never delete pipeline functions
*(was `feedback_no_delete_pipeline_fns.md`)*

Never delete any function in `preprocess.py` or `detect.py`. Only modify existing
functions or add new ones.

**Why:** All pipeline functions are part of the benchmark history. Removing one
would make past results (saved JSON in `results/`) unreproducible and break
comparisons across runs.

**How to apply:** If a function is superseded or found to be wrong, fix it in place
or create a new variant with a different name. Do not remove the old one.

### Prefer deployed over local
*(was `feedback_prefer_deployed_over_local.md`)*

When John says "frontend" / "dashboard" / "website" for this project, he means a
**public, globally-accessible deployed site that anyone can view from anywhere** —
not a localhost tool. Claude twice defaulted to "run it locally" and John had to
correct it both times ("I want to deploy it to a website, that's the whole
point... I don't want it local").

**Why:** the recordings are public by policy (committed to git, owner fine with
biosignal data being public), and the explicit goal is shareability. He also likes
simple managed hosting (Vercel) over self-managed.

**How to apply:** assume deployment is the target. For static/fixed data → static
site (pre-built JSON + JS plotting) on Vercel. Don't pitch local-only
matplotlib/Streamlit-local unless he asks.

### Response length
*(was `feedback_response_length.md`)*

For simple questions, John wants answers condensed to ~2 paragraphs — up to 3 when
the topic is genuinely meaty and he's finding it helpful. Just the important info,
not a full analysis. He explicitly pushed back on 10-paragraph replies to simple
questions, then later okayed bumping to 3 for a substantive thread.

**Why:** Verbose multi-section answers (headers, tables, every caveat) bury the
point and waste his time; he reads fast and wants signal.

**How to apply:** Lead with the direct answer. Cut the section headers, exhaustive
enumerations, and "here's everything related" tangents unless he asks to go deep.
Terse-but-dense still holds — just shorter. Offer to expand rather than
pre-expanding.

### Verify before interpreting
*(was `feedback_verify_before_interpreting.md`)*

When John reports something he saw on the live display, describes a rig
manipulation, or shares a viewer/plot screenshot, treat it as a **hypothesis to
test against the raw npz**, not an established fact — load the data and quantify
before drawing conclusions or launching heavy analysis.

**Why:** twice in one session (2026-07-01) Claude over-committed to an
interpretation and John had to correct it:
1. He described a wire twist→noise A-B-A; Claude spun up a whole Workflow
   explaining "why twisting *reproducibly* increases noise" — then he corrected
   that run 3 (re-twisted) was actually *clean*. Twist was exonerated; the real
   cause was electrode/contact settling (first-run noise). Built on an unconfirmed
   premise.
2. From a viewer screenshot Claude asserted the L-electrode spikes "grow over the
   hour." The data showed them present from the start and roughly stationary (count
   highest early).

**How to apply:** verify first, interpret second. Before spinning up a Workflow or
committing to a mechanism, run a quick numeric check on the actual samples. For
anything per-channel / electrode-contact, compute **per-channel min/max** (sparse
pops hide in RMS/s2s). John does careful A-B-A controls and honest partial reports
— match that with data-grounded verification. He values calibrated epistemics and
magnitudes over confident narratives, and he catches over-reach, so flag
uncertainty explicitly rather than smoothing it.

---

## Project

### Phase 0 status
*(was `project_phase0_status.md`)*

Phase 0 complete. Three subjects recorded (50 trials each): john, aaron, german.
The file `20260519-153629-german.npz` had subject_id "german2" corrected to
"german" (the original german.npz was junk data and moved to `archive/`).

Phase 1 (offline classifier) is underway. Baseline `eval_simple.py` (peak-sign
classifier, no normalization): john 100%, aaron 88%, german 94% accuracy; latency
p50 ~430ms; FPR varies by threshold.

**Why:** Need labeled cross-subject EOG data to build universal classifier
offline. Aaron has one channel with ~66mV electrode drift during session —
individual channel stats are wild but the differential is clean.

Key technical findings:
- Board slots 0/1 = live EOG channels (physical CH1/CH2, board rows 1/2)
- Purple wire = slot 1 = ch_R (right electrode, board row 2)
- Classifier should trigger on saccade velocity peak (~100-300ms transient), NOT
  sustained gaze
- Recording saves all 11 board rows raw; ch_L=row1, ch_R=row2 stored in metadata
- Baseline voltage drift (±1000µV) is normal — diff signal cancels it, leaves clean
  ±28µV saccade signal

### John electrode swap (May)
*(was `project_electrode_swap_john.md`)*

John's recording (`20260513-175241-john.npz`) had the left/right electrode wires
connected to opposite board slots compared to Aaron and German. The raw eeg data is
unchanged, but the metadata was corrected on 2026-05-19: `eog_ch_L`: 1 → **2**;
`eog_ch_R`: 2 → **1**.

**Why:** John had the electrodes on opposite ears from Aaron/German. This caused
LEFT saccades to appear as positive diff peaks, opposite to the other two subjects.

**How to apply:** superseded/generalized by the permanent polarity swap below.

### Benchmarking design
*(was `project_benchmarking_design.md`)*

**Goal:** Run multiple (preprocessor, detector) combinations against the same
labeled recordings, save all results, find the best accuracy/latency tradeoff.

Pipeline API: NPZ → PrepResult (in memory) → DetectResult → saved JSON in
`results/`. PrepResult fields: subject, sr, ev_s, ev_l, diff_uv ((ch_R−ch_L)×1e6
after filtering), baseline_mu, baseline_sigma. DetectResult: acc, lat_med_ms, fpr,
trials.

Optimize, in priority order: (1) Accuracy ≥85%, (2) Latency p50 ≤300ms, (3) FPR
<5%. Accuracy and latency trade off — lower threshold fires earlier but catches
more noise. **Do NOT optimize** SNR / noise floor (diagnostics only; Goodharting
risk — a preprocessor could blur the saccade to improve SNR while hurting latency).

Conventions: `preprocess_<slug>(npz_path) → PrepResult`; `detect_<slug>(prep,
**params) → DetectResult`; `bench.py` runs combos and saves `results/` JSON; all
scripts kept — nothing deleted.

Frontrunner pipeline (2026-05-21): **velocity + mean-sign** — 94% avg acc, 500ms
fixed latency, 9.3% avg FPR (aaron's FPR spike is an electrode artifact).

> Note: the offline benchmark harness was later deprioritized — `bench.py` and the
> `eval_*` scripts live in `archive/eog-scripts/`. `preprocess.py`/`detect.py`
> remain in `src/brainpong/` as the algorithm source of truth.

### 60Hz / CMRR noise
*(was `project_noise_60hz_cmrr.md`)*

The "unmitigatable oscillating noise ~50% of sessions" that breaks the σ threshold
is **60 Hz utility mains**, confirmed by PSD on all 3 EOG recordings: 75–96% of
band power at 60.00–60.03 Hz (locked integer → mains fingerprint).

The bandstop notch (58–62 Hz, order 3) is NOT the problem — it delivers 58–62 dB
and crushes 60 Hz to 0.01 µV **when input is sane**. Per session: john 13µV→0.01µV
clean; german 4µV→0.01µV clean; aaron **8542µV**→78µV, std 9802µV — analog
front-end railed, dead.

Mechanism: `R−L` montage rejects mains only via CMRR. When one electrode has
poor/asymmetric contact (high-Z mismatch, dried gel, flaky bias/DRL), 60Hz becomes
*differential*, amplitude explodes ~1000×, amp saturates → clipping spawns
broadband harmonics + IIR ringing no digital notch recovers. The "50%" is a binary
electrode-contact/CMRR coin flip.

Suspects ranked: (1) electrode impedance asymmetry, (2) bias/ground contact, (3)
ground loop via laptop charger (run on battery), (4) long unshielded leads.

**2026-06-02:** John killed the ground electrode — now **bias input only**. Bias
contact is the *sole* CMRR path, so suspect (2) elevated. **2026-06-24:** an
**external monitor** plugged into the laptop re-grounds it to mains earth — "run on
battery" is NOT sufficient if a monitor/charger/wired-ethernet is attached.
Generalizes suspect (3) to *any* mains-referenced peripheral. Lit deep-dive durable
numbers: CMRR ≈ Z_in/|Z₁−Z₂| (match the pair > absolute kΩ; skin-prep drops
imbalance ~8×; ISCEV target <5 kΩ). ±187.5 mV rail = ADS1299 full-scale at ×24.
ISCEV EOG band DC–30 Hz → 0.1 Hz HPF + 30 Hz LPF (current 0.5 Hz distorts the
saccade step; 100 Hz LPF lets EMG in). Refuted: a precise µV/° slope; 30° saccade
≈ 250–1000 µV is a sanity band only.

### Oscillation false-fire
*(was `project_oscillation_false_fire.md`)*

John reports persistent **oscillating** noise (not single spikes) in the live EOG
game as of 2026-06. Root cause is structural: the glance-pair detector
(`_run_eog_sm` in `eog_core`) arms on a sustained crossing in one direction and
fires on the opposite direction within GLANCE_WINDOW_S — so any oscillation
swinging above ±σ reads as an endless LEFT/RIGHT/LEFT… stream, exactly the
"look one way then the other" pattern it fires on. The persistence gate kills
single spikes but does nothing here.

Reproduced in `tests/test_oscillation_noise.py`: a 3–8 Hz, 10σ oscillation → ~3
phantom commands per 3 s (rate-limited by the 0.8 s refractory). Sub-threshold
oscillation is clean. The notch kills mains but LOW-freq oscillation (cable sway /
motion / sub-notch harmonics) passes through.

Candidate mitigations (NOT yet implemented — defer to John): return-to-baseline
gate between the two glances; run-count / sign-alternation veto; non-physiological
amplitude ceiling for railing. A run-count discriminator is demonstrated working in
the same test file.

### Tournament prep (Jun 25)
*(was `project_tournament_jun25.md`)*

EOG Brain Pong tournament — players drive a paddle by looking left/right. Game:
`scripts/pong_game_brainflow.py --2player`. Originally Thu 2026-06-25; slipped (John
in a multi-day sleep trial). Full living plan: `docs/tournament-prep-plan.md`.

Blocker understanding (2026-06-26) — TWO stacked problems:
1. **Signal/contact** (needs hardware): flat HEOG from on-skin contact. Led to
   flashing SRB1 on. Board/firmware/wiring/stream all confirmed healthy.
2. **Detector — the bigger, OFFLINE-fixable blocker.** As measured 2026-06-26 by an
   early offline replay harness (`archive/eog-scripts/replay_eog.py`; the current
   harness is `scripts/replay_detector.py`), the glance-pair detector hit only a
   small fraction of cued gazes and phantom-fired. (The Replay-harness entry has the
   current, higher numbers and the decision to keep the glance-pair debounce.)

### SRB1 ground fix
*(was `project_srb1_ground_fix.md`)*

The board's ADS1299 montage is configured ONLY in firmware:
`/Users/john/Dev/pong-firmware/esp32_firmware/esp32_firmware.ino` (separate repo,
NOT under git — back up before editing). The brainflow host CANNOT set registers;
it only sends a baud byte. Flash via `arduino-cli compile/upload --fqbn
esp32:esp32:esp32`. GOTCHA: default UploadSpeed 921600 fails on this adapter —
append `:UploadSpeed=115200` to the FQBN.

Montage is **2-channel referential** (EOG = software diff CH2−CH1), NOT 1-ch
bipolar. Each channel needs a shared reference on its −input → SRB1.

**Bug found 2026-06-25:** firmware had MISC1 (0x15) = 0x00 → SRB1 switches OPEN →
channel −inputs floating. Both EOG channels parked at +145–148 mV DC (~78% of the
±187.5 mV rail @ ×24). **Fix:** MISC1 0x00 → 0x20 (SRB1 closed). **Requires
physically clipping a reference electrode to the SRB1 pin** — flashing alone without
the electrode ties all −inputs to a floating SRB1 pin (worse).

### EOG flat-signal debug
*(was `project_eog_flat_signal_debug.md`)*

2026-06-25 debug of "HEOG differential reads flat / no saccade response." Checklist
in `docs/eog-flat-signal-debug.md`. Findings from the raw npz:
- **Board + stream are HEALTHY** — row-10 unix timestamps advance at clean 250 Hz.
  (Row-0 brainflow package counter is unused/always-0 on this board — ignore it.)
- **Flat runs = both canthi inputs FLOATING.** ch1≡ch2 (e.g. both −171 mV, corr≈1),
  parked near the −187.5 mV rail. The referential montage's insidious failure mode:
  a contact break looks like a quiet flat line, not an obvious rail.
- **The live display LIES**: `record_eog`'s display HP-filters at 0.5 Hz, removing
  the near-DC gaze plateau. Judge from RAW stored data.

**Verdict:** primary bottleneck = unreliable canthi electrode contact (gold is
polarizable, bad for near-DC EOG; Ag/AgCl preferred).

### Firmware bias-sense bug
*(was `project_firmware_bias_sense_bug.md`)*

Firmware: `~/Dev/pong-firmware/esp32_firmware/esp32_firmware.ino` (no git; register
table `ADS1299_REGISTER_LS` ~line 237; backups `.pre-srb1.bak`, `.pre-biasfix.bak`).

**Montage (confirmed by John 2026-06-29): REFERENTIAL.** Canthus L → IN1P (CH1+),
canthus R → IN2P (CH2+); both −inputs tied to SRB1 via MISC1=0x20; bias drive on
ear clip. HEOG = CH1−CH2 in software. John chose referential over bipolar (values
observing both canthi independently; referential is an information superset).

**The bug (fixed 2026-06-29):** `BIAS_SENSP` (0x0D) was `0xFF` — routed all 8
+inputs into the bias derivation, but only CH1/CH2 connected. The 6 floating inputs
(CH3-8) made the bias loop chase uncontrollable nodes → common-mode shoved toward
the −187.5 mV rail. Fix: `BIAS_SENSP 0xFF→0x03` and CH3-8 `0x60→0x81` (power-down +
MUX input-short). **Flashed & verified 2026-06-30** via arduino-cli.

**Flashing:** `arduino-cli compile --upload -p /dev/cu.usbserial-1120 --fqbn
esp32:esp32:esp32:UploadSpeed=115200 .` — must pin UploadSpeed=115200.

### Clips vs gold electrodes
*(was `project_clips_vs_gold_electrodes.md`)*

Compared `20260630-173432-john` (CLIPS = carbon/rubber ear pads + Signa gel) vs
`20260630-175142-john` (GOLD cups + Ten20). Only the two **ear electrodes** (bias +
SRB1 reference) changed.

**Verdict: go all-gold, ideally all-Ag/AgCl.** Clips DC offset −178 mV = **96% of
the ±187.5 mV rail**; 60 Hz mains **~350× worse** (1149 vs 3.3 µV) = CMRR collapse.
Differential HEOG: 60 Hz cancels, both classify L/R ~98–100% (easy-task ceiling).
**The trap:** clips' "quiet" differential is a **silent ~11× attenuation** (blinks
scale 11.4× matching cued deflections 11.0×), not low noise. Carbon has no Ag/AgCl
couple → stays polarizable; gel is a red herring. Track per-channel DC-offset-%-of-
rail + per-channel 60 Hz + effective gain, NOT differential noise.

### 67-min gold endurance test
*(was `project_67min_gold_torture_test.md`)*

`data/eog/20260701-133403-john.npz` — 67.6 min, all-gold, soap-washed skin,
free-run. Writeup: `docs/eog-67min-gold-freerun-characterization.md`.

**Rig validated for long sessions:** rail-safe the whole hour (worst 21% of rail),
drift tiny (~2–3 mV/channel, settles by ~10 min), gel didn't dry. Once filtered to
0.5–15 Hz, R−L noise (~73 µV) matches the short gold session — dominated by real
ocular activity.

**Key finding — the L canthus electrode was marginal ALL session** (John's
electrodes are physically reversed vs others). Two symptoms: (1) **L-only contact
"pops":** CH1 had 265 large transient spikes (>400 µV, up to 1200 µV); CH2 zero; 0%
coincidence → NOT ocular; ~700 µV artifacts in R−L → false events. (2)
**Differential 60 Hz:** per-channel 60 Hz anti-correlated (corr −0.93) → R−L 60 Hz
(~40 µV) did NOT cancel. Both point to a weak/high-Z L electrode → **re-seat the L
canthus electrode**. **Method lesson:** aggregate RMS/s2s MISS sparse per-channel
transients — use per-channel min/max decimation.

### EOG dashboard schema
*(was `project_eog_dashboard_schema.md`)*

Building a **live raw-data dashboard** to diagnose oscillating noise: 2 electrodes
at the outer canthi (differential pair) + a single active bias-drive ear clip, no
ground. Scope minimal — display + store raw signal only, no filtering.

**Canonical schema lives in `CLAUDE.md`** — don't duplicate. Summary: per-recording
metadata {unix start, fs, gain, unit, board, person, montage, notes}; per-sample
{running count, raw signal/channel}; sparse event markers. Sample time is derived
`start + count/fs`, NOT lumpy USB arrival timestamps.

**Decisions (2026-06-17):** storage = SQLite (WAL), chunk-as-BLOB. Live
architecture = two processes decoupled through the DB file. **Actually shipped
first: a static public web viewer (`web/`).** Diagnostic protocol: raw PSD IDs
culprit by frequency (60.0Hz locked → mains/CMRR; odd non-mains peak → bias-loop
self-oscillation; sub-1Hz → DC-rail); input-short MUX=001 partitions
board-internal vs electrode-side; PD_BIAS off splits self-oscillation from mains.

### EOG viewer website
*(was `project_eog_viewer.md`)*

Building a **data-viewing website** for the EOG corpus (offline; distinct from the
live dashboard). Owner struggles with aesthetics → design-first workflow.

**Locked (2026-06-24):** offline corpus viewer; **DB + server** (SQLite + small
API, HTTP from day one); npz stay frozen → ingest into regenerable SQLite, never
re-save npz. Default view: 2 wired channels stacked + derived L−R ribbon pinned on
top. Layout: two labeled zones (tinted DERIVED zone over tinted RAW zone),
dev-SaaS/"Studio" dark aesthetic. Trim feature: keep-window (drag two handles);
gates all derived analysis; stored as a DB annotation, npz never touched.

**Why the removed `web/` viewer "never worked": DEPLOY/BUILD FRICTION** (Vercel +
`build.py` npz→JSON export-and-commit), NOT Plotly perf. Lesson: a server reading
SQLite sidesteps it. Honesty rules: min/max-per-pixel decimation (never stride),
explicit clip/rail band, filters as non-destructive overlays, viridis-not-jet.

**BUILT (2026-06-24), 57 tests green:** `src/brainpong/store.py`,
`scripts/ingest_npz.py`, `scripts/serve_viewer.py`, `web/{index.html,style.css,
viewer.js}`. Filters = `preprocess.VIEWER_FILTERS` (zero-phase filtfilt).
**Merged as PR #26.** **Confirmed working 2026-07-01:** 40 recordings ingested;
per-channel min/max view exposed the L-canthus contact pops. Still localhost-only
(public deployment remained open). `store.py`, `ingest_npz.py`, and
`serve_viewer.py` are live (`serve_viewer.py`/`ingest_npz.py` in `scripts/`).

### Jul-2 4-subject polarity
*(was `project_4subject_jul2_polarity.md`)* — **SUBSUMED by the permanent polarity
swap entry below.**

Analyzed four non-John gold recordings from 2026-07-02 (anthony/german/david/hart).
**Signal is excellent in ALL four** — not the bottleneck. **Polarity split:**
anthony CANONICAL; german/david/hart all **INVERTED** (right gaze reads negative).
Their ~0–5% detector accuracy = the canonical-sign assumption reading a perfect
signal backwards; flipping the sign → 95–100%. With 3 of 4 inverted, "inverted" may
be the rig's default wiring → pin the sign per-recording. 60 Hz/CMRR blemish weak-
to-collapsed on 3 of 4, culprit consistently the R/right-canthus electrode.
Detector blockers = sign convention + FPR, not signal. The 5th file
`20260702-170507-john` is "fake data" — ignore.

### Permanent L/R polarity swap — AUTHORITATIVE
*(was `project_polarity_swap_permanent.md`)*

John physically swapped the Left/Right EOG electrode inputs right after recording
`anthony` and stated it is **permanent**. Confirmed empirically (2026-07-06): every
recording after anthony reads **polarity-inverted** relative to the code contract
`diff = eeg[ch_R] − eeg[ch_L]`. This is the single source of truth for EOG polarity
— supersedes the Jul-2 4-subject entry and generalizes the May John swap.

**Evidence (unanimous, two independent methods + visual):** filtered signed-diff
AUC — anthony 0.975 = CANONICAL; all 10 post-anthony recordings AUC < 0.06 =
INVERTED. Raw unfiltered per-trial median agrees (75–98% of trials wrong way).
Categorical step change, not a gradient. Signal EXCELLENT (peaks 160–580 µV) — pure
sign issue.

**Boundary:** last canonical = `20260702-172120-anthony`; first inverted =
`20260702-174754-german`.

**Mechanism (confirmed by John, connector/pin level):** left-eye electrode → pin
2(+) → CH2 → row 2; right-eye → pin 1(+) → CH1 → row 1. Canonical metadata for this
wiring is **ch_L=2, ch_R=1**.

**SCOPE COLLAPSED (John, 2026-07-06/07), later partly reversed:** offline analysis
was deprioritized and 10 offline/viewer scripts + `eog_display.py` moved to
`archive/eog-scripts/`. The viewer was then restored: `scripts/` currently holds
`record_eog.py`, `pong_game_brainflow.py`, `filtered_plot.py`, `serve_viewer.py`,
`ingest_npz.py`, and `replay_detector.py`. `store.py`/`preprocess.py`/`detect.py`
are live in `src/brainpong/`.

**✅ DONE 2026-07-08 — swap applied on BOTH software and data:** `record_eog.py`
(EOG_SLOT_L=1/R=0) and `pong_game_brainflow.py` edited. All 11 post-swap files
metadata-MIGRATED (`eog_ch_L`/`eog_ch_R` swapped 1/2→2/1 via tmp→verify→atomic-
rename; raw `eeg` byte-identical). **Corpus is now UNIFORMLY CANONICAL — 0 inverted
files.** **MERGED to main as PR #32.** `detect.py` is orphaned in src/ (offline
eval detectors, consumers archived); `store.py` + `preprocess.py` back the restored
viewer.

### EOG detection re-eval
*(was `project_eog_detection_reeval.md`)*

Working-notes log of the EOG filter/detector re-evaluation on the **Anthony-and-
later** clean corpus (2026-07-08 session). Kept so a future session doesn't
re-derive the approach or repeat dead ends.

Task: re-eval filtering (`preprocess.py`, 7 preprocessors) × detection (`detect.py`,
5 detectors) on the clean gold-montage corpus (12 recordings / 10 subjects), for
accuracy + FP + FN. Polarity already canonical (PR #32). Built fresh harness
(`scratchpad/eval_harness.py`) reusing the real preprocessors + re-implementing each
detector's decision rule, with two changes: peak_sign FALLBACK removed (to measure
FN honestly) + threshold as k·baseline_sigma for all detectors. Fidelity check: 0
mismatches vs shipped predictions.

**Load-bearing findings:**
1. Direction accuracy near-ceiling (~90-97%) and filter-insensitive — the DETECTOR
   decides it. Sign-based detectors win (peak_sign ~96.6% cued).
2. The shipped `sustained_crossing` "best overall, 94%" is **fallback-driven** — at
   6σ it MISSES ~76% of real saccades; the 94% was the peak_sign fallback.
3. The real wall is a **14× baseline-noise spread (σ 18-262 µV).** anthony
   peak/σ=0.89 → ANY amplitude threshold gives ~100% FN on him; simon σ=18µV →
   ~100% FP. Sign detectors sidestep this (read direction, not amplitude).
4. Free-running (deployable) has NO good operating point: best ~84% acc / ~14% FN /
   ~23% FP. Fix = robust per-subject baseline + calibration + sign decision, NOT a
   different filter/detector.
5. Caveat: `preprocess._snr` overstates SNR ~24× for drift-heavy subjects.

Round 2 (absolute-threshold hypothesis): `abs_mean` gave best-balanced (wide ×
abs_mean = 84.2% success / 14.8% FN / 12.9% FP). Rescue asymmetry: abs_mean rescued
anthony (noisy baseline) but did NOT fix simon (doesn't hold still in REST). TWO
free-running failure modes → per-subject calibration is mandatory.

**Live-detector arc — SHIPPED 2026-07-09:** root cause of false/reversed paddle
commands = the 0.5 Hz causal high-pass MANUFACTURES an opposite-sign recovery-tail
artifact the amplitude detector reads as a real crossing. Fixes: (1) **EOG_HPF_HZ
0.5→0.1** (PR #36) — necessary but not sufficient alone; (2) **velocity-based
detection** (PR #36) — `_eog_velocity` Engbert-Kliegl 5-point derivative feeds
VELOCITY to the unchanged glance-pair SM; velocity's sign = gaze-change direction;
slow tail self-rejects. Calibration σ now robust 1.4826·MAD. (3) **Untried:**
LOW-pass 100→~35 Hz (differentiating a 0.1-100 Hz signal amplifies HF noise; at LPF
35 hit 87% vs 57% at 100). 0.1-35 Hz is the textbook EOG band.

Dead ends: don't trust `detect.py` accuracy at face value (peak_sign fallback masks
misses); `detect.py` is orphaned but IS the offline-detector source of truth
(`preprocess.py` also backs the viewer's filters).

### Board MAC fingerprint
*(was `project_board_mac_fingerprint.md`)*

BrainFlow can't tell two same-model X8 units apart over serial, but the **ESP32 MAC
is hardware-unique and CAN** — printed by esptool in the arduino-cli flash output.

**Version labels (corrected 2026-07-09, John authoritative):** the **original
board is hardware v1.2** (the good one — every committed recording and all detector
tuning are on it); the **new/second board (from Simon) is hardware v1.3** (noisier).
- **new/second = v1.3** (`--board v1.3`): MAC `8c:4f:00:a9:7b:90`, chip
  ESP32-D0WDQ6 rev v1.1, 40 MHz crystal (same MCU as original). First bring-up
  2026-07-01: `20260701-142113-john` (57s) quantified HEALTHY (per-channel DC
  −29/−25 mV, real anti-phase dipole corr −0.62, clean mV-scale HEOG). Watch-item:
  differential 60 Hz ~6.9 µV vs ~0.7 µV on gold ref (~10× worse, likely electrode
  ΔZ). NOT a controlled A/B.
- **original = v1.2** (`--board original`, default): MAC not yet captured (TODO).

### Two-board build
*(was `project_two_board_build.md`)*

Real two-player EOG pong (two independent Cerelog X8 boards, one per player) in
`scripts/pong_game_brainflow.py`. **MERGED to main via PR #37.** Supersedes the old
fake "2-player" (two electrode-pairs CH1-2/CH3-4 on ONE board). Same PR added live
HPF/LPF sliders + default LPF 100→50 Hz. **VALIDATED on real two-board hardware
2026-07-09** — John + German played live, got good data. **Best config = current
shipped defaults: LPF 50 Hz, HPF 0.1 Hz, σ-threshold 4.0, glance-window 0.5 s.**

Architecture: ONE Dash process, TWO `BoardShim` handles. The Cerelog fork keys its
session registry on `(board_id, serial_port)` → two same-model X8s coexist in one
process. P2 slots = P1 slots (CH1/2) on its OWN board.

**Decisions (John):** hardcode ports; simultaneous shared-countdown calibration but
each board computes its OWN baseline σ; live-play only (no in-game recording in this
build). **Open hardware items:** set real `P2_SERIAL_PORT`; wire both boards to
IDENTICAL L/R canthi polarity (no sign-cal exists); run laptop on BATTERY (two
boards share USB ground → bias loops couple); both boards need the bias-fix flash.

### In-game recording
*(was `project_ingame_recording.md`)*

The pong game now auto-records to `data/eog/` on every New Game — one npz per player
(2 players → 2 files sharing a timestamp). Built for the 1-board-vs-2-board
electrical-interference study. **MERGED to main via PR #38.** Has run on real
two-board hardware — the 2026-07-09 session produced paired eog-v3 npz
(`20260709-*-{atin,john}.npz`, shape `(2,N)`, `n_players=2`, boards 1.2/1.3).

**New protocol `eog-v3`** (writer = `recording.py::save_eog_recording`): `eeg` is
**(2, N)** — ONLY the 2 EOG channels, raw volts (row0=ch_L, row1=ch_R; eog_ch_L=0,
eog_ch_R=1). **Older loaders that assume 8 rows will break — use eog_ch_L/R
indices.** New fields: n_players, board_version, serial_port, player_slot,
sigma_thr, hpf_hz, lpf_hz, glance_window_s. Markers: calib_start, play_start. Board
version derived from PORT (`PORT_TO_VERSION` = {usbserial-1120: "1.2", usbserial-
1110: "1.3"}).

Capture: span = New Game → Game Over (incl. calibration), pulled non-destructively.
Raw-only (no pre-processed data — regenerable), but `notes` embeds a plain-English
pipeline description via `eog_core.pipeline_description()` (version-tagged
`pipeline-v1`) — **update that string + bump the version whenever the method
changes.**

### Matched-filter detector
*(was `project_matched_filter_detector.md`)*

The pong game now has a **matched-filter detector** selectable against velocity via
a "Detector" radio toggle (Velocity | Matched filter). Built because analysis showed
matched filtering is the highest-value upgrade for tight-margin data (on German's
data it widened the glance/noise margin ~2.16× → ~4.17×). **MERGED to main via PR
#39.** NOT yet run on real hardware.

**How it fits (detection-stage swap, velocity stays preprocessing):** the matched
filter cross-correlates the velocity signal with a **~120 ms unit-norm Hann saccade
template**; the SAME glance-pair SM then thresholds that response. Only change vs
velocity mode is the signal handed to `_run_eog_sm`. `PIPELINE_VERSION` bumped to
`pipeline-v2`.

**GOTCHA:** the toggle changes which signal the baseline σ is measured on, so it
only detects correctly from the **next New Game** (recalibration). Swap per-run, not
mid-game.

### Cross-board noise mismatch
*(was `project_cross_board_noise.md`)*

The two-board plan hits a hardware-mismatch problem: **v1.2 (original) controls work
well; v1.3 (second/Simon board) is noisier.** Framing: a *device domain-shift*
problem. Two bars: (a) "each board works on its own" = high confidence; (b)
"fair/identical head-to-head" = achievable only at the operating-point level (equal
false-fire rate), NOT raw signal.

**Approach stack (decided 2026-07-09):** (1) self-normalization — threshold in each
board's own robust σ (already in `eog_core.py`); make baseline σ rolling/adaptive;
(2) morphology/template gate on armed candidates (highest-value vs in-band
false-fires); (3) per-board profile keyed off ESP32 MAC (one parameterized pipeline,
not forked code); (4) adaptive filtering (synthetic-ref LMS notch); (5) reference
electrode + ANC; (6) Kalman constant-velocity detector (horizon). **Honest floor:**
ΔZ contact asymmetry + in-band SNR overlap are fundamental walls. Fairness = equal
phantom-fire rate on each board's own baseline, not equal volts.

**UPDATE 2026-07-09 — v1.3 "spiky garbage" is very likely a MIS-FLASH, not a noisy
hardware revision.** v1.3 velocity was a dense forest of spikes to ~5M µV/s.
Diagnosis: v1.3's OWN 07-01 bring-up npz is CLEAN (velocity max 11k µV/s) →
hardware is fine; current garbage is a firmware/config STATE. The spiky signature
reproduces from a known-railing recording → fingerprint = front-end RAILING / bias
instability, not eye signal. Root cause (John's hypothesis, strongly supported):
v1.3 has the PRE-fix flash where floating CH3–8 feed the bias drive
(`BIAS_SENSP=0xFF`). **CONFIRMED 2026-07-09: reflashing v1.3 fixed the spiky
garbage.** Implication: v1.3 is NOT fundamentally noisy → de-risks the two-board
plan; the elaborate cross-device DSP stack may be unnecessary for v1.3 specifically.

### Replay harness + glance window
*(was `project_replay_harness_glance_window.md`)*

First end-to-end measurement of the LIVE glance-pair detector
(`eog_core._run_eog_sm`), built 2026-07-10 as `scripts/replay_detector.py`. It
replays recordings through the exact game tick (0.5s window, step 0.1s, last 0.1s →
SM; faithful to `pong_game_brainflow._poll_eog`) and scores three axes separately:
SENSITIVITY (fired-on-glance, polarity-agnostic), DIRECTION (correct L/R of fires),
FALSE FIRES (fires in REST/confound windows, per-min). Reads frozen npz only; writes
`derivatives/results/replay_*.json`.

**Key findings (12 tournament subjects, FAST/game-realistic glances):**
- The felt "German 80%" = **misses, not phantom fires.** Harness scores German's
  recording at 80% fired-on-glance, direction 92% correct, ~0 false fires/min.
  Validates the harness. The 20% failure is "glanced, paddle didn't move."
- **The dominant miss cause is the pair window, not the threshold.**
  `glance_window_s` was tightened 0.7→0.5s in commit 919fab5; that drops returns that
  land late. Widening it: 0.5s→67% hit, 0.7s→86%, 0.9s→92% — with NO direction
  penalty (~94%) and negligible false-fire rise (1.3→2.2/min on quiet REST). Lowering
  sigma_thr 4→3 instead buys ~nothing and DOUBLES wrong-direction (8→16%). Window ≫
  threshold.
- **67% average hides a bimodal split** fixed by the wider window: christy 35→95%,
  john 55→100%, atin 50→95%, hart 80→100% at 0.9s. Only simon stays low (15→50%) — a
  genuine low-SNR outlier needing per-person calibration.
- **Unmeasured risk:** cued REST has no casual glances, so it understates gameplay
  false fires; a wider window gives a casual look-and-back more time to phantom-fire.
  The `CASUAL_GLANCE` confound epoch (added to `record_eog.py` same day) measures
  this; the fix if needed is an absolute amplitude floor (deliberate > casual).

**Evidence-based path to 90%+:** (1) widen glance_window to ~0.8s [live slider,
~free]; (2) collect confounds, re-run harness at 0.8s for honest false-fire cost; (3)
if too high add abs-floor + oscillation/run-count gate + blink common-mode veto (uses
the 2nd channel, currently discarded); (4) per-person calibration for low-SNR
outliers. Do NOT swap to single-saccade (stale plan; pair debounce earns the 80%).
Matched filter raises hit 67→78% but false fires 1.3→10.2/min — not a free win.

---

## Reference

### ADS1299 datasheet notes
*(was `reference_ads1299_datasheet.md`)*

ADS1299-x datasheet (TI SBAS499C, Jan 2017). PDF at
`~/Desktop/cerelog/ads1299-datasheet.pdf`. The analog front-end inside the Cerelog
X8 — the chip that produces every `.npz` sample.

- Low-noise **24-bit** simultaneous-sampling ΔΣ ADC w/ per-channel PGA. 8ch. Input-
  referred noise 1 µVpp @ gain 24. CMRR −110 dB typ. DC input impedance 1 GΩ.
  Supplies AVDD 5 V, fCLK 2.048 MHz. PGA gain ∈ {1,2,4,6,8,12,24}.
- Data: 24-bit two's complement. 1 LSB = ±FS/2²³. SPI frame = 24-bit status word +
  N×24-bit channels, MSB first.
- Key registers: `01h CONFIG1` DR[2:0] — **110 = 250 SPS, the power-on default;
  fs=250 is fixed for this rig** (bottleneck is ESP32 SPI→USB transport, not the
  ADC). `03h CONFIG3` PD_REFBUF/BIASREF_INT/PD_BIAS. `05h–0Ch CHnSET` bit7
  power-down, bits6:4 gain, bits2:0 MUX (000=normal, 001=input-short, 101=test
  signal). `0Dh BIAS_SENSP`/`0Eh BIAS_SENSN` per-channel bias routing. `15h MISC1`
  bit5 SRB1.
- **CMRR is gated by external mismatch, not the chip.** Referential montage (Fig
  73) = the project's SRB1 wiring. **Input CM range narrows with gain** (Eq 4) →
  full-scale referred to input = ±187.5 mV @ gain 24. **Bias drive is the designed
  countermeasure but a stability hazard** — high-Z ear-clip contact adds loop poles
  → can self-oscillate. **Lead-off** (LOFF reg + status bits) can directly measure
  the ΔZ asymmetry and flag a popped electrode in real time.

### Pokémon grouping game
*(was `reference_pokemon_grouping_game.md`)*

John plays a Pokémon puzzle game: Connections-style 4×4 grid — 16 mons into 4 groups
of 4, sometimes with a hidden/unloaded sprite. He asks for charts (types, 4x
weak/resist, immunities, gen) then pattern-hunts. Guess feedback: "N of these were
in a group together."

Category pool (June 2026): type-based axes, generation/region, plus rarer "vibes" —
legendaries, sub-legendaries, mythicals, pikachu clones, regis, eeveelutions,
starters, baby pokemon, stone evolutions. Ultra Beasts, Mega, and Gmax are each
their OWN category (NOT lumped into legendaries). Evolution-based categories valid
(branch evolutions, not-fully-evolved, evolution types). Type-based categories only
count 4× weaknesses, ¼× resistances, and immunities — normal 2× weak / ½× resist do
NOT count. Game tends to include one trap mon straddling both groups.
