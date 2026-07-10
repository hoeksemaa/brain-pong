# EOG ear electrodes: carbon-rubber CLIPS vs GOLD cups — a numerical comparison

**Sessions:** `data/eog/20260630-173432-john.npz` (CLIPS) vs `data/eog/20260630-175142-john.npz` (GOLD)
**Date analyzed:** 2026-06-30. Script + figures: `docs/assets/clips-vs-gold/`.

## Setup (what was and wasn't held constant)

Both recordings are the same subject (John), same board (`CERELOG_X8 unit:original`,
ADS1299 inside), gain ×24, fs 250 Hz, same `record_eog.py` paradigm (25 LEFT + 25 RIGHT
cued 2 s gaze holds), back-to-back (~18 min apart), **zero sample drops** → directly comparable.

The **two outer-canthi signal electrodes were identical gold cups in both.** The *only*
change was the **two earlobe electrodes** — the common-mode **bias-reference servo**
on one ear + the **SRB1 reference** on the other:

- **CLIPS:** carbon/rubber clip pads + **Signa gel** (a *low*-impedance wet gel).
- **GOLD:** gold-plated cups + **Ten20** paste, taped on.

Gel on both, so this is a material/geometry comparison, not wet-vs-dry. Channel model:
`CH1 = Lcanthus − SRB1`, `CH2 = Rcanthus − SRB1`, `HEOG = CH2 − CH1 = Rcanthus − Lcanthus`
(the shared ear reference cancels mathematically in the differential).

## The headline

**The difference is enormous where it concerns signal-chain *health*, and essentially
zero where it concerns *L/R task accuracy* in this dataset.** Both montages classified
left-vs-right at ~98–100%. But the gold montage operated in a clean, robust regime while
the clips operated on the edge of saturation with collapsed common-mode rejection and a
~11× silent attenuation of the real signal. The clips "worked" by luck (a huge eye dipole
and differential mains-cancellation), not because they are adequate.

## Magnitudes (this is what you asked for — how *large*)

| Property (what it measures) | CLIPS | GOLD | Difference |
|---|---|---|---|
| **Per-ch DC offset** (rail proximity) | −178 / −179 mV = **96% of rail** | −28 / −47 mV = 16–26% | clips ~**15–25× less headroom** (6–7 mV vs 138–158 mV) |
| **Per-ch 60 Hz** (CMRR / contact health) | **~1149 µV** RMS | ~3.3 µV | clips **~350× worse** |
| CH1↔CH2 correlation | 0.9988 | 0.969 | clips near-total common-mode |
| **Differential 60 Hz** (after R−L) | 0.59 µV | 0.65 µV | **equal — cancels** |
| **L/R deflection** (true signal size) | 41 µV sep | 452 µV sep | gold **~11× larger** |
| **Blink amplitude** (involuntary ref) | 9.4 µV | 107 µV | gold **~11.4×** larger |
| Per-ch hi-freq contact noise | 60 Hz-dominated | **5–7 µV** | gold floor genuinely low |
| Slow drift wander (<0.1 Hz, p2p) | ~4 mV | ~3 mV | **similar** (shared canthi) |
| **L/R classification accuracy** | **100%** | **98%** | **negligible** |

So: signal-chain health differs by **1–2.5 orders of magnitude**; task accuracy differs by
**~nothing**. Both extremes are true simultaneously.

## The trap (and the proof) — clips look "quieter" but aren't

Naively the clips' differential trace is *cleaner*: rest std 3.9 µV vs gold's 68.9 µV. **This
is an artifact, not a virtue.** The clips apply a flat ~11× attenuation to *everything*, so
signal and noise shrink together. Evidence it's a measurement-side attenuation, not John
moving his eyes less:

- **Blinks** (involuntary, biologically identical across two back-to-back sessions) are
  **11.4× smaller** with clips — matching the **11.0×** cued-deflection ratio almost exactly.
  You cannot consciously scale blinks 11×, so the chain is attenuating.
- The cue-averaged waveform **shape is identical** (clean sustained DC step, no droop over
  the 2 s hold) — rules out AC-coupling/high-pass; and there is **no flat-topping** — rules
  out hard clipping. A flat, broadband, instantaneous scaling.
- The **differential PSD ratio** (gold/clips) is ~11–12× across the mid/high bands; it rises
  to ~17–20× only at low frequency, where gold faithfully captures extra real ocular biology.
- Un-scale the clips noise by 11× → ~43 µV, same order as gold's 69 µV. The "quiet" clips
  trace is **compressed**, not low-noise.

## Mechanism — leading hypothesis (not proven)

3 of 4 expert reviewers converge on: **PGA small-signal gain-compression from a near-rail
DC operating point.** The carbon-rubber electrode is a *polarizable* junction with a large,
unstable half-cell EMF. Because the ear electrode is shared by both channels, that EMF shows
up as the ~−178 mV input-referred offset on CH1 *and* CH2 (note they rail nearly equally).
At ×24 that's ~4.27 V at the PGA output against a ±4.5 V swing — ~95% of range — where the
amplifier's *incremental* gain droops, compressing the small EOG riding on top by ~11×,
flatly and broadband. This elegantly **unifies the rail proximity and the attenuation: the
attenuation *is* the rail proximity.** The same weak/saturated bias servo + high-Z contact
collapses CMRR (the 350× per-channel 60 Hz).

**Honest caveat:** this is *inferred*, not measured. There's a real puzzle — `R−L` cancels
the ear node by construction, so an ear-only swap has *no first-order path* to attenuate the
differential; the gain-compression story works only via the shared front-end operating
point. A pure resistive input-divider is implausible (ADS1299 inputs ~GΩ). The behavioral
explanation is killed by the blink test. **Definitive test:** apply the ADS1299's built-in
calibration reference in each electrode state (`INT_TEST` square wave, or an external mV
bench source) and read the recovered gain — if it drops ~11× with clips, gain-compression is confirmed.

## "More conductive gel, yet worse" — resolved

Signa gel lowers *bulk* series resistance but cannot fix the *interfacial* half-cell offset
and drift, which are thermodynamic/kinetic electrode properties. Carbon has no Ag to form a
reversible Ag/AgCl/Cl⁻ couple, so even a chloride-rich gel leaves a large polarizable offset.
This confirms the **electrode material** is the bottleneck, not the gel.

## Answers to the three questions

**Q1 — clear difference, and how large?**
- **Task performance: no meaningful difference** (100% vs 98% — a ceiling artifact of an easy
  task; don't read clips' higher Cohen's d = 14.7 vs 5.87 as "better" — that's the
  compression flattering it).
- **Noise / chain health: massive, all favoring gold** — ~350× less per-channel mains,
  ~15–25× more rail headroom, a genuinely low ~5–7 µV contact-noise floor, and no silent
  ~11× attenuation. Gold is decisively better as a *measurement*; clips only survived because
  the eye dipole is large and mains cancels differentially.

**Q2 — all-gold vs gold-eyes + clip-ears?**
**Go all-gold; ideally all-Ag/AgCl.** The ear (ref + bias) contact is the entire bottleneck
here. The two ear electrodes have *different* requirements, both of which carbon clips fail:
- **SRB1 reference** sets the DC offset → wants a **low, stable, non-polarizable half-cell**
  → **Ag/AgCl ideal, gold good, carbon terrible.** (Material matters more than area here —
  a bigger clip wouldn't pull −178 mV off the rail.)
- **The common-mode servo electrode** carries the µA-scale return of the right-leg-drive
  feedback → wants **low impedance / large contact area** for CMRR and to keep the bias
  amp out of saturation.

Preference order: **all-Ag/AgCl > all-gold > anything with carbon-rubber.** Since EOG is
essentially a DC/sub-Hz signal, offset stability and low-frequency electrode noise dominate —
exactly where Ag/AgCl's non-polarizable advantage is largest.

**Q3 — which numerical properties to track?** (ranked by diagnostic value)
1. **Per-channel DC offset as % of rail (headroom).** Would have flagged the clips instantly.
   *Add a live rail-proximity guard to the dashboard.*
2. **Per-channel 60 Hz amplitude** — the contact-quality canary. **Look per-channel, not
   differential**, since R−L cancels it and hides the problem.
3. **Effective differential gain** (blink or calibration amplitude) — catches *silent
   attenuation* that SNR/accuracy won't.
4. **Drift** — both your candidates (noise, drift) matter, but drift matters mostly via its
   *interaction with headroom*: drift magnitude was similar (~3–4 mV) in both, yet only
   dangerous in clips because it eats a 6 mV margin. Track slope (mV/min) **relative to
   headroom**, not in isolation.
5. Differential task SNR / Cohen's d — useful but **misleadingly high under compression**;
   never use it alone.

## Caveats / confounds (epistemics)

- **Session order not counterbalanced:** clips preceded gold by ~18 min; gel drying / skin
  hydration / fatigue are uncontrolled. The blink 11.4× cross-check makes the ear-swap the
  dominant driver, but an **A-B-A interleave** would close this.
- **"Canthi identical" is asserted, not measured** — no lead-off/ΔZ readout exists, and
  re-taping the ears could perturb canthus contact. Capture per-channel impedance to confirm.
- **Easy task ceiling:** both classify ~perfectly, so accuracy *cannot* distinguish them here;
  a harder regime (small/fast saccades, continuous control, motion) would expose the clips.

## Recommended next steps

1. Replace carbon clips with gold (or Ag/AgCl) at the ears; re-record. **Highest-value change.**
2. Calibration-injection gain readback (`INT_TEST`) to settle the 11× mechanism definitively.
3. A-B-A interleaved A/B to kill the order confound; capture per-channel impedance.
4. Add **per-channel rail-% and 60 Hz** guards to the diagnostic dashboard.
