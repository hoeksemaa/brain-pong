# EOG flat-signal debug — paranoid checklist

Status: active debug (2026-06-25). HEOG differential reads ~flat despite a working
board. **This doc is grounded in John's actual recordings, not just theory** — see
"What the data already told us" first, then the ranked test list.

**Update 2026-06-30:** the firmware bias fix (row 7) was **flashed** — `BIAS_SENSP 0xFF→0x03` + CH3–8 power-down. A post-flash headless capture confirmed **CH3–8 read exactly 0** (the flash took); CH1/CH2 still rail at −187.5 mV with the inputs *open*, so the in-vivo bias-recenter check is pending real electrode contact. See [[project_firmware_bias_sense_bug]].

## What the data already told us (don't re-derive)

Analysis of the 7 runs from 2026-06-25 (`data/eog/20260625-*.npz`, all raw/unfiltered):

- **The board + stream are healthy.** Row-10 unix timestamps advance at a clean 250 Hz
  in every file; per-channel ADC noise is present (≠ frozen). The "dead" runs are NOT a
  USB/stream freeze. (Row-0 package counter is unused — always 0 on this board; ignore it.)
- **The flat runs = both canthi inputs FLOATING.** In the dead runs (`132918`, `133011`,
  `133203`, `133544`) ch1 and ch2 are *identical* (e.g. both −171.1 / −171.0 mV, 30 µV
  noise, correlated ≈ 1.0) → their difference is a constant 65 µV. Open inputs get pulled
  to the same SRB1/bias-driven level → zero differential, parked near the −187.5 mV rail.
- **When "live", contact is one-sided.** In `133701` ch1 is quiet (297 µV std) while ch2
  swings 2380 µV — one electrode intermittently making contact, the other not.
- **The eye dipole is barely there even in the best labeled run.** `131331`: RIGHT−LEFT
  separation only **+34 µV** (want hundreds), and *both* canthi move the **same** direction
  for R-vs-L gaze (L Δ=+44, R Δ=+78) → that's common-mode, not an anti-phase dipole.
  The left canthus is not contributing its opposite-polarity half.
- **The handling/tap run hit ±100 mV.** So the full electrode→board→capture path can
  produce huge signal — the hardware works; contact/montage is the bottleneck.

**Verdict:** primary cause is **unreliable canthi electrode contact**, whose failure mode
in this *referential* montage is insidious — floating inputs collapse to an identical
common level and look like a quiet flat line rather than an obvious break. Secondary: the
live display HP-filters (0.5 Hz) away the near-DC EOG plateau, so it lies even when raw is OK.

## Do these IN ORDER (decisive bisection)

1. **Multimeter every electrode** (30 s). Resistance through pasted skin: canthus↔canthus
   and each canthus↔reference should be ~5–100 kΩ. Multi-MΩ / open-circuit = bad contact
   or broken lead → fix before anything else.
2. **Tap test on the live raw trace** (30 s). Tap each cup; a contacting electrode throws
   an obvious artifact on *its* channel. No artifact = that electrode/lead/header pin is dead.
3. **Headless per-channel + blink + DC-step capture** (1 run; Claude runs it). Look hard
   L (hold 3 s) → R (hold 3 s) ×3, then blink. Confirms each canthus *individually*
   deflects and which side is weak. Judge from RAW, not the display.
4. **If contact is good but diff still weak → bipolar rewire test.** Both canthi onto ONE
   channel's +/− (e.g. L→CH3+, R→CH3−); watch CH3 raw. Bipolar captures the dipole
   directly with no SRB1. If bipolar works and referential doesn't, the referential/SRB1
   path (or one electrode's contact) is the culprit.
5. **If still bad → on-chip firmware diagnostics** (reflash): MUX=001 input-short (all
   channels should read the ~µV chip floor → proves board internal is clean) and MUX=101
   test-signal (square wave on all 8 → proves ADC+SPI+stream end-to-end).

## Ranked checklist (likelihood × test-cheapness)

| # | Layer | Hypothesis | Why plausible (this rig) | Tiny test | Pass / Fail |
|---|---|---|---|---|---|
| 1 | Electrode/contact | Canthi cups not contacting (dry/under-pasted, poor prep, peeling tape) | Data: channels flip open-identical ↔ one-flailing; dipole barely present | Multimeter pair Z; tap test; re-abrade + fill cup + secure | 5–100 kΩ & tap artifact = OK; MΩ/no-artifact = fail |
| 2 | Electrode/biology | Gold cups polarize on a near-DC signal (EOG is DC-ish); half-cell drift | Gold is polarizable; Ag/AgCl is the EOG standard | Swap to Ag/AgCl if available; watch for slow drift to rail | Stable mid-range DC = OK |
| 3 | Montage/HW | Referential (SRB1) collapses floating inputs to identical → flat diff | Data: dead runs = ch1≡ch2 via shared SRB1 | **Bipolar rewire** (both canthi one channel +/−) | Bipolar shows dipole = referential/contact issue |
| 4 | Software/display | Live trace HPF 0.5 Hz removes the near-DC gaze plateau | Confirmed: step→decays to baseline; display ≠ raw | Compare stored raw vs live; use DC-coupled monitor | Raw shows step the display hid = display artifact |
| 5 | Wiring/physical | Intermittent lead/connector/header-pin (DIY harness) | Data: state flips between runs | Wiggle leads during tap test; buzz each wire end-to-end | Stable = OK; flicker = reseat/resolder |
| 6 | Electrode | Paste bridge shorting a canthus to ref/neighbor | Smeared paste → shared node → flat diff | Visual inspect; clean + re-paste tightly | No bridge = OK |
| 7 | Firmware | BIAS_SENSP=0xFF feeds 6 floating inputs into bias → common-mode shoved to rail | Open canthi sit at −171 mV (near rail) | ✅ **DONE (flashed 2026-06-30)**: BIAS_SENSP 0xFF→0x03 (CH1+CH2 only) + CH3–8 power-down | CH3–8=0 confirmed (capture); CM-recenter pending live contact |
| 8 | Biology | EOG genuinely small: dim light, near fixation, fatigue, un-settled electrodes | Corneo-retinal potential is light-dependent | Bright room; big 30°+ saccades to targets; settle 2–5 min | Bigger deflection = was biology |
| 9 | Software/mapping | Python reads wrong rows as L/R | Mostly ruled out (rows 1,2 carry signal in live file) | Tap L electrode → must move row 1 only | Correct row moves = OK |
| 10 | Firmware/board | ADC not converting / SPI stale / gain wrong | Mostly ruled out (decoded gain×24, MUX=000, timestamps live) | MUX=101 test-signal reflash → square wave all ch | Square wave = ADC/stream OK |
| 11 | Stream | USB/buffer freeze | **Ruled out** — timestamps advance 250 Hz in all files | (n/a) | — |

## Notes / fixes worth building
- **DC-coupled raw monitor**: a headless CLI that prints per-channel mean/std/rail% + the
  L/R diff with NO HPF, updating live — so contact quality and gaze steps are visible
  without the lying display. (record_eog's display should get a "raw/DC" toggle too.)
- **Lead-off detection** (LOFF reg + status word) could flag an open canthus in real time;
  currently disabled in firmware. A future flash could surface it in the recorder.
- Montage call: **bipolar (canthus-to-canthus, one channel)** is the classic 2-electrode
  HEOG montage and is more robust + easier to debug than the current referential+software-diff
  setup — its failure modes are obvious (channel rails) rather than insidious (flat diff).
  Worth considering if contact fixes don't fully resolve it.
