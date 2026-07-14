# EOG detector — numerical analysis & tuning recommendations (2026-07-13)

Follow-up to `eog-detector-tuning-analysis.md` (2026-07-11), driven by the goal of
pushing subjective "intended-move → paddle-move" quality above the current ~80%.
Covers responsiveness, robust L/R detection, blink/noise rejection, noisy-player
robustness, the direction-flip failure, velocity-vs-matched detector, and explicit
FPR/FNR. **No production code changed** — everything below was measured in a faithful
offline replay harness (scratchpad, not committed).

## Method & harness fidelity

A replay harness imports the real `eog_core` DSP + state machine and reproduces the
live loop tick-for-tick: 125-sample window (100 settle + 25 new) advanced 25 samples
(100 ms) per tick, velocity on the full window, last 25 samples → `_run_eog_sm`, MAD
calibration on the first 5 s. It reproduces the shipped detector to **1484 fires** at
default config across the 53-file corpus (prior harness: 1534 on 47 files; difference
is the 6 new 07-13 files + minor calibration convention). Per-subject fire streams are
clean and mechanistically sensible (e.g. John's cued FAST block: 12 fires, all correct
direction, landing ~0.8 s after each cue).

### Ground-truth corpus (what each slice is good for)
- **12 structured cued files** (`anthony`…`xiq`, 10 people, `record_eog` protocol):
  20 FAST glance-pairs each = **240 labelled pairs**. The reliable **accuracy /
  direction / FNR** backbone. Caveat: cued returns are cue-paced (~0.5–0.6 s), i.e.
  **slower** than practiced gameplay flicks, so absolute recall understates a fast
  player and overstates the benefit of a long glance window.
- **2 John hand-labelled gameplay files**: 29 glances + 36 blinks. Used for blink
  separation and **soft** recall. **Labels are incomplete** (many real glances
  unlabelled) → John blink-FP is an **upper bound**, gameplay recall is indicative
  only.
- **07-13 training/2-player files**: rapid full-left/full-right sweeps → the
  direction-flip / oscillation testbed; also a live **railing-electrode** example (P2).

### Metric definitions (per your request)
- **CC** = correct-command rate = P(fires within the cue window **and** correct
  direction). = TPR × direction-accuracy.
- **FNR** = false-negative rate = fraction of cued glances with **no** correct command
  = 1 − TPR (a miss).
- **FPR** = false-positive rate = spurious fires per **minute of rest** (eyes-forward,
  no cued glance).
- **blinkFP** = fires within ±0.35 s of a labelled blink (John, upper bound).

## Baseline (shipped default: velocity, σ6, gw0.5, min-wait0.05, refr0.8)

| metric | value |
|---|---|
| Correct-command (cued, 240 pairs) | **36.7%** |
| FNR (missed cued glances) | **57.5%** |
| Direction accuracy \| fired | 86.3% |
| FPR (fires/min, rest) | 3.50 |
| blink-FP (John, upper bound) | 5 / 36 |
| Fire latency after glance onset | ~0.8 s cued / ~0.4 s (368 ms median) gameplay |

The 36.7% cued CC is dragged down by weak/noisy subjects; strong sessions (david, hart,
john, atin) sit at 55–65% with ~90–100% direction — consistent with John's felt ~80%.

## Findings

### 1. Amplitude is NOT the bottleneck for committed glances — the return gap vs the window is
In the live regime, outgoing glance velocity is healthy for almost everyone (cued median
6–19σ; John gameplay **37σ**, min 30σ). What kills recall is the **0.5 s pairing window**:
median glance return-gap is **430–700 ms** (cued) — right at or past the cutoff — so the
return arrives after the window closes and the pair times out. Per-cue pairs recovered by
widening the window (cued): **john 6→20, hart 7→20, atin 10→20, christy 3→18, david 12→18**
going 0.5→0.7 s.

### 2. glance_window 0.5 → 0.7 s is the single biggest recall lever (cued)

| gw (s) | CC | FNR | dir | FPR |
|---|---|---|---|---|
| 0.4 | 6.2% | 87.5% | 50% | 3.3 |
| **0.5 (default)** | 36.7% | 57.5% | 86.3% | 3.50 |
| 0.6 | 46.2% | 47.9% | 88.8% | 3.6 |
| **0.7** | **53.8%** | **40.4%** | **90.2%** | 4.11 |
| 0.8 | 58.8% | 35.4% | 91.0% | 4.4 |

Direction accuracy *improves* with a longer window (waiting for the true return beats
pairing on a noise crossing). On John's practiced gameplay, recall is nearly flat with gw
(his flicks already return in ~370 ms) — but **new tournament players behave like the
deliberate cued subjects, not like practiced John**, so gw0.7 is expected to help the
population even though it does little for John. 0.7 s is the knee; ≥0.8 s adds FPR for
little more recall.

### 3. armed_min_wait 0.05 → 0.2 s is the "timing window" — the blink/oscillation gate
This is the finding the prior instance flagged. Blinks complete their biphasic twin-spike
in **68–84 ms** (median); real glance returns take **368 ms** (gameplay) / ~500 ms (cued).
Requiring ≥0.2 s between the two crossings of a pair rejects blinks while keeping glances:

| min-wait (s) | CC | FPR | blinkFP | fires |
|---|---|---|---|---|
| **0.05 (default)** | 36.7% | 3.50 | 5/36 | 85 |
| 0.15 | 35.8% | 3.1 | 4/36 | 77 |
| **0.20** | 35.4% | **2.6** | **3/36** | 63 |
| 0.25 | 33.3% | 2.2 | 3/36 | 51 |

0.20 s cuts FPR ~25% and blink-FP ~40% at ~1 pp CC cost. Beyond 0.25 s it starts eating
real glances (p25 of gameplay return-gap ≈ 220 ms).

### 4. The Pareto move: gw0.7 + min-wait0.2 (velocity)
**CC 36.7 → 53.3%, FNR 57.5 → 41.2%, direction 86.3 → 90.8%, FPR 3.50 → 3.43 (≈flat),
blink-FP 5 → 3.** Strictly better on every axis but a ~0.5 pp CC vs gw0.7-alone. This is
the safe, low-risk default change.

### 5. Matched filter is viable AFTER the min-wait gate (revised vs prior)
The prior rejected the matched filter because blink-fires jumped to ~65%. But that was
**without** the timing gate. With it:

| detector / config | CC | FNR | dir | FPR | blinkFP |
|---|---|---|---|---|---|
| velocity gw0.7 mw0.2 | 53.3% | 41.2% | 90.8% | 3.43 | 3/36 |
| matched gw0.5 mw0.05 | 57.5% | 32.1% | 84.7% | 7.13 | 16/36 |
| matched gw0.7 mw0.05 | 76.7% | 12.5% | 87.6% | 7.40 | 16/36 |
| **matched gw0.7 mw0.2** | **76.7%** | **14.2%** | **89.3%** | **5.48** | **5/36** |

The matched filter integrates over the saccade shape (~√len SNR gain), so it recovers the
weak/slow glances velocity misses — **+23 pp CC, FNR 41 → 14%** — and the min-wait gate
pulls its blink-FP back down to velocity-baseline levels (16 → 5). The remaining cost is
**FPR 5.48 vs 3.43/min** (~1 extra false move/min of rest). This is the highest-recall
option on the table and the biggest lever for "80% → higher," at a controllable FPR cost.

### 6. Direction-flip / oscillation ("look L-R-L-R read as R-L") — run-count discriminator
Reproduced in the 07-13 training sweeps: during a continuous full-left/full-right sweep
the detector fires `L,L,R,L` — phase-ambiguous, because a continuous scan is an
*oscillation* and the pair machine locks onto arbitrary phase (it arms on whichever
crossing it sees first after refractory). A **threshold-crossing run-count** in the pair
window separates the cases cleanly:

- real John glances: **median 2 runs** (a clean out+back)
- training sweeps / oscillation: **median 4 runs** (up to 24)

Rejecting a pair whose window contains **>2 runs** (equivalently: require the signal to
return toward baseline between the two lobes) drops the oscillation/sweep fires while
passing clean glances. This is the same run-count idea already prototyped in
`tests/test_oscillation_noise.py` — not yet wired into production. It is the direct fix
for the direction-flip **and** the known oscillating-noise false-fire.

### 7. Noisy / railing electrodes need an absolute sanity gate, not a σ-relative one
Today's P2 calibrated to **σ = 58,000–226,000 µV** (vs P1 ~200) — a saturated/railing
channel — yet still fired 26–28 times, because a σ-relative threshold scales *with* the
garbage. No threshold multiplier fixes this. Needs an **absolute signal-quality gate**:
flag when baseline σ is implausibly high (or raw |signal| rides the clip ceiling) and
tell the player to re-seat the electrode rather than pretending to detect. This is the
"expect noisier players" case, and it is a hardware-contact (ΔZ) problem — consistent
with the electrode-comparison docs.

### 8. Amplitude confirm-gate — complementary to timing, not a silver bullet
Within a player, glances are much larger than blinks (John: glance median 40σ, blink
median 6σ). But a per-player confirm-gate at 40% of median glance (16σ for John) rejects
9/9 blinks **while also dropping 7/29 real glances** — the weakest real glances (8σ)
overlap the strongest blinks (12σ). So amplitude alone over-rejects; **amplitude + the
0.2 s timing gate together** are strong where either alone is leaky. Training mode already
has the player sweep on cue — the natural place to capture per-player glance amplitude for
this gate.

### 9. Responsiveness has two independent components
- **Latency to a single move** ≈ human out-and-back gesture (368 ms gameplay median) +
  ~40 ms detection. The pair requirement means the paddle moves on the *return*, not the
  outgoing glance — this is inherent to the debounce and can't be cut without abandoning
  pair-based blink rejection.
- **Repeat rate** is capped by refractory: at 0.8 s the min inter-fire gap is **1000 ms
  with zero fast repeats**; dropping to **0.5–0.6 s** enables 4–5 rapid same-direction
  repeats with negligible FPR change. This is the direct knob for "move left several
  times quickly," and it is nearly free.

## Recommendations (priority order)

**Tier 1 — ship, low-risk, Pareto-positive (velocity):**
1. `GLANCE_WINDOW_S` 0.5 → **0.7 s**. Biggest recall lever for new/deliberate players;
   improves direction; small FPR cost.
2. `ARMED_MIN_WAIT_S` 0.05 → **0.2 s**. The timing window: cuts FPR ~25% and blink-FP
   ~40% at ~1 pp recall cost. Together with (1): CC 36.7→53.3%, FNR 57→41%, dir→90.8%,
   FPR flat, blinks down.
3. `REFRACTORY_S` 0.8 → **0.5–0.6 s**. Doubles rapid-repeat capability; ~free.

**Tier 2 — highest ceiling, needs a decision / small pipeline addition:**
4. **Switch the default detector to matched filter** *paired with min-wait 0.2* →
   CC ≈ 77%, FNR ≈ 14%. Accept FPR ≈ 5.5/min, or pair with (5) to tame it. This is the
   single biggest quality jump available. Suggest A/B it live before committing.
5. **Run-count (oscillation) discriminator** — reject pairs with >2 threshold runs in the
   window. Fixes the direction-flip and the oscillating-noise false-fire; also offsets the
   matched filter's higher FPR. Prototype exists in `test_oscillation_noise.py`.

**Tier 3 — robustness for noisy players:**
6. **Absolute signal-quality gate** on calibration σ / clip-ceiling → "re-seat electrode"
   instead of firing on a railing channel (today's P2).
7. **Per-player amplitude confirm-gate** captured during training-mode sweeps, set well
   below median glance (~30%, not 40%) so it complements — not replaces — the timing gate.

## Caveats
- Cued absolute recall understates fast players (cue-paced returns); **relative** setting
  comparisons transfer, absolute rates don't. gw0.7's population benefit rests on new
  players resembling the cued subjects.
- John gameplay recall/blink-FP are soft (incomplete labels); blink-FP is an upper bound.
- Matched-filter FPR (5.5/min) is real and worth confirming live before it ships as
  default; the run-count gate (rec 5) is the intended mitigation.
- σ sweeps: lowering σ raises recall but FPR scales badly (σ4 → FPR ~2×, blink-FP ~2×);
  prefer the window/detector levers over lowering σ. Keep σ per-rig via the live slider.

---

# Second pass (deeper) — 2026-07-13

Pushed past knob-tuning to interrogate the detector's design. Five findings materially
change the recommendation.

## A. gw0.7 + min-wait0.2 helps 12/12 subjects — proven transferable
Per-subject correct-command, `velocity` default vs `gw0.7+mw0.2`: improves or ties for
**every one of the 12 cued subjects** (never regresses). This is the transferability
proof the prior doc could only assert. Ship it as the conservative default.

## B. The matched filter RESCUES the players the current detector abandons
Per-subject correct-command, default → `matched gw0.7 mw0.2`:

| subject | default | matched | | subject | default | matched |
|---|---|---|---|---|---|---|
| german | 10% | **85%** | | hart | 65% | **100%** |
| aaron² | 10% | **90%** | | john | 60% | **100%** |
| xiq | 10% | **95%** | | atin | 60% | **95%** |
| david | 60% | 90% | | simon | 15% | 35% |
| anthony | 20% | 15% | | christy | 40% | 50% |

The velocity detector's "barely fires" group (german/xiq/aaron² at 10%) is exactly who
the matched filter fixes — it integrates over the saccade shape (~√len SNR gain), so it
recovers weak/slow glances velocity misses. Aggregate CC 36.7 → 76.7%, FNR 57 → 14%,
direction 89%. **For a tournament of diverse new players this is the difference between
"half can't move the paddle" and "almost everyone can."** Only anthony regresses — his
signal is present (~9σ) but his *return* lobe is weak (~2σ) and his direction is near
chance: a gesture/coaching problem, not an algorithm one.

## C. The dominant "false" fires are real unintended saccades, not noise
Over a **67-min free-run** (John, natural eye movement, no deliberate glances) the detector
fires ~**13/min at every config** — because natural looking-around *is* out-and-back eye
movement, which the detector is built to catch. The matched filter adds only ~20% over
velocity here (14.5 vs 12/min), and `matched+σ8` drops to 9/min (below velocity default) —
so the earlier "matched doubles FPR" (from contaminated cued-rest windows) **overstates the
real cost**. The true robustness ceiling is intent: a no-intent EOG scheme cannot separate
"I meant to move" from "I glanced at the ball." Which leads to:

## D. A per-player amplitude gate suppresses drift ~73% — the biggest gameplay lever
Casual free-run saccades are **much smaller** than committed control glances:

| | median | p75 | min (committed) |
|---|---|---|---|
| free-run saccades (σ) | 14 | 22 | — |
| John control glances (σ) | ~37 | — | ~30 |

A confirm-gate at **20–25σ** (below John's 30σ control-glance floor) cuts free-run nuisance
fires from 13→**3.6/min (−73%) while keeping every committed glance**. This is the direct
fix for "the paddle drifts when I glance around." **It must be per-player-relative** (~50–60%
of that player's calibrated control-glance median), because weak-signal players glance at
only 5–9σ and a fixed 20σ gate would erase them. Training-mode sweeps already have the
player produce committed glances on cue — capture the amplitude there and set the gate.
Amplitude and timing gates are complementary: amplitude carries clean/strong rigs, timing
carries weak rigs where glances overlap blinks.

## E. Direction should come from position sign, not crossing order
For a detected pair, direction from **peak filtered-position sign** is ≥ as accurate as the
current first-crossing-order logic and far more robust to window/timing (John: position 100%
across all windows; crossing-order degrades to 72% at a poorly-aligned window). Position is
unambiguous (it reflects where the eye *was*); crossing-order is what inverts during rapid
alternation. Cheap, low-risk robustness upgrade. (A biphasic direction-encoding matched
filter was prototyped and rejected — it underperforms once you realize the existing
pair+min-wait machine already *is* a biphasic "out-then-back-after-a-gap" detector.)

## Revised recommendation — a small redesign, not just knobs
Reconceive detection as **"a committed, glance-shaped out-and-back":**
1. **Detector: matched filter** (sensitivity — rescues weak players). Biggest single lever.
2. **Shape gate: min-wait 0.2 s + glance_window 0.7 s** (rejects blinks & catches slow
   returns). Slider-testable today.
3. **Shape gate 2: run-count ≤ 2** in the pair window (rejects oscillation → fixes the
   direction-flip and oscillating-noise false-fire). Prototype in `test_oscillation_noise.py`.
4. **Intent gate: per-player amplitude confirm** at ~55% of calibrated control-glance median
   (rejects casual-saccade drift). Calibrate in training mode. −73% nuisance fires.
5. **Direction from peak-position sign**, not crossing order.
6. **Refractory 0.5–0.6 s** (repeat-glance rate).
7. **Absolute signal-quality gate** on calibration σ / clip ceiling (railing electrodes,
   e.g. today's P2 at σ=58k–226k).

Suggested rollout: (2)+(6) are slider/constant changes — do now (Tier 1, proven 12/12).
Then A/B (1) matched live. (3),(4),(5),(7) are small, isolated code additions; (4) needs a
one-line amplitude capture in the training-mode calibration. Keep the shipped config
snapshot as the revert point.
