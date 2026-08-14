
## Structure John settled on

The section was reorganised twice. Final shape: **the numbered pipeline is a
list of operations** — things that happen to the signal, in order — and the
hardware-hygiene material (gel, skin prep, wire resistance, unplug your laptop,
the checklist) is **broken out into a separate debugging section** rather than
occupying pipeline steps.

There is a hard boundary partway down: everything before the ADC is analogue
physics, everything after is arithmetic on arrays. The ADC is the hinge, and it
is where **real recorded data first appears**. Before it, figures are
reconstructed *backwards* from the real recording.

| # | operation | data | status |
|---|---|---|---|
| 1 | eyes move — corneoretinal dipole rotates | diagram | **DONE** |
| 2 | measured against SRB1 | spoofed back | **DONE** |
| 3 | bias driven into the body | spoofed back | **DONE** |
| — | ~~amplified ×24~~ | — | **cut** |
| 5 | digitized — one sample per interval | → real from here | **DONE** |
| 6 | left subtracted from right | real | **DONE** |
| 7 | detrend — one constant per 0.1 s poll, over 0.5 s | real | **DONE** |
| 8 | filter — LP 100, notch 48–52/58–62, HP 0.5 | real | **DONE** |
| 8.5 | threshold amplitude, or threshold velocity | real (anthony) | built — NOT signed off |
| 9 | calibrate → σ | real | **DONE** |
| 10 | threshold → spike detected | real | **DONE** |

**DONE** means John has signed it off. Every step is now built, signed off and
live in the harness. What is left is the prose.

**Gain is cut, not deferred.** It muddied the water for what it bought. Three
reasons, in order of how badly they hurt:

- **There is no amplified array to draw.** The Cerelog fork returns volts
  *referred to input* — it has already divided the 24 back out. Verifiable rather
  than assumed: the LSB in the committed recording is 0.0223517 µV, which is
  `4.5/24/2²³`, and that only comes out right if gain 24 was divided out. A gain
  figure would be relabelling a y-axis, which is the one thing this series does
  not do.
- **On an autoscaled trace ×24 is the identity.** Every figure centres on the
  window mean and scales to a baked span, so a flat multiplier moves zero pixels.
- **The usual justification is false on this rig.** "Gain lifts µV above the LSB"
  needs quantisation to be the limiting error. At *unity* gain the LSB would be
  0.536 µV against ~11 µV of sample-to-sample movement — already 21× finer. At
  ×24 it is 506× finer. Quantisation was never the limit; the skin-electrode
  interface is.

What gain really is on this rig is a **headroom** decision, and that belongs in
the debugging section next to the electrode-offset material, not in a list of
operations performed on the signal: the rail referred to input is ±4.5 V/gain, so
±187.5 mV at ×24, and ch_L's −38.9 mV half-cell offset alone eats 24 % of it
before any signal arrives. ×96 would sit at 94 % and clip on a quiet channel.

The numbers keep their existing values — there is no step 4, rather than a
renumbered 4–9. Renumbering would rename every mount id and invalidate the
dozens of "figure 6"/"figure 7" cross-references in `figures.js`'s comments, for
no reader-visible gain until the prose is written. Decide it then.

```bash
python -m http.server --directory web/essay-figures 8901   # the dev harness
python scripts/bake_essay_figures.py                       # rebuild the data
```

## Figures are transformations, not animations

Each figure is a **composition of the pipeline's own functions** applied to the
committed recording; the control moves between two real states of that
composition. Nothing is drawn that the software would not compute.

So before building a figure, **read the function it depicts and match it** —
window length, order of operations, constant detrend versus linear. The first
draft of figure 7 subtracted the mean of the displayed 10 s window, which is not
what the game does, and got both the period and the character of the operation
wrong.

Two rules that fell out of building 7 through 10, both learned the hard way:

- **What is drawn and what is counted must obey the same rule.** Figure 10
  painted every sample over the threshold red while counting detections through
  a refractory period, so a spike could be bright red and tallied as missed at
  the same time. If the colour says "detected", the count must mean the same
  thing by the same test.
- **Quantise geometry at the sample, not at the poll.** Figure 9's play-head
  advanced in whole 0.1 s polls, which is faithful to how the detector receives
  data and reads on screen as chop. Sample resolution is 0.27 px per step and
  reads as motion; everything that moves still comes off one integer sample
  index so the edges cannot drift apart.

## House style

Established across 6–10, then extended by 1–5. Where 1–5 depart from it, the
departure is listed below and reasoned in the figure's own header comment.

- **Seams.** A figure opens on the state the previous one closed on — same
  array, same span, same centring rule. The reader should feel the steps
  accumulate rather than restart.
- **Colour.** Black is what goes in, red is what the step produces, grey is
  axes and gridlines. Green is a second input channel — figures 2, 3, 5 and 6,
  wherever ch_L is drawn beside ch_R. A figure's input and output are never the
  same colour even when they are the same array: figure 6's red differential is
  figure 7's black. **Red also carries what a step SUBTRACTS**, not just what it
  produces — SRB1 in figure 2, the anti-signal in figure 3, the dipole's poles in
  figure 1 — on the grounds that the subtrahend and the difference it makes are
  one operation.
- **One second** for every toggle transition (`TRANSITION_S`) — except where a
  figure has two stages to show, and then it is two seconds so each stage still
  gets its one. Figures 2, 3 and 5 are the two-stage ones.
- **No text on the figure** beyond axis labels, cue guides, and a readout that
  is genuinely the step's output — σ in 9, the counts in 10. The figures are
  read inline beside prose that explains them; anything the sentence next to it
  already says is ink competing with the trace.
- **Geometry.** One plot size everywhere: 430 px wide, 150 px tall, 60 px from
  the top of the figure. Controls reserve a fixed 150 px gutter on the right
  whether or not a figure has any, otherwise the canvas flexes and the same ten
  seconds gets drawn at different scales in different steps. The CANVAS may grow:
  figures 2 and 3 stack three panels and so run 614 px tall, and their top two
  panels sit at figure 5's exact y so the seam is still pixel-for-pixel. Figure 1
  is the one figure with no plot box at all — it is a diagram.
- **Names change with the arithmetic**, as figure 6's R and L become R − L:
  figure 2 goes R/L → R−SRB1/L−SRB1 and figure 3 goes R/L → R−Bias/L−Bias. They
  switch rather than cross-fade, because both variants start at x=6 and differ in
  length, so overlapping alphas draw both sets of glyphs at once. And they must
  stay clear of x=63: the tick gutter is 108 px with the tick labels right-aligned
  into it, which is what rules out longer forms at the standard 15 px.

## Where the figures diverge from the live game, on purpose

Both sets of numbers are in the baked JSON so prose can name the difference.

- **Filter corners.** The figures use LP 100 / HP 0.5 at 4th order; the game
  runs LP 30 / HP 0.1 with a 3rd-order notch. At the game's 30 Hz corner both
  notch bands sit above the low-pass and the notch toggle moves the trace 0.26
  px, which makes a third of figure 8's controls do nothing. **These are also
  the original values** — `72d7387`, 2026-05-22 — so the figures depict the
  detector as first built rather than a simplification invented for the essay.
- **Amplitude, not velocity.** The live detector thresholds the Engbert &
  Kliegl 5-point derivative, so its σ is 342 µV/s. The figures stay on
  amplitude end to end and σ is 8.02 µV. One quantity, one unit, the whole way
  down. `meta.calib.sigma_velocity` carries the detector's real number.
- **Figure 5's sample rate is wrong on purpose, by a factor of 25.** The board
  runs 250 Hz, so the 3 s window holds 750 samples per channel — more than one
  per pixel. Drawn as dots they merge back into the line they came from. So the
  figure samples at 10 Hz instead: 30 dots across the window, one per 100 ms,
  each the mean of 25 real samples. What survives the exaggeration is the only
  thing the step has to teach — a continuous quantity became one number per
  interval, and the intervals are all there is from here down.
  `digitize.fs_real` and `digitize.fs_shown` are both baked, and the RATE is the
  constant in the bake rather than the dot count, so retuning the window adds
  samples instead of spreading the same ones further apart.
- **Figure 5 holds its camera still, and pays for it at the seam.** Same 3 s
  window, same span, same 1 s-per-second scroll at both ends of the toggle, so
  the only thing the control changes is the sampling. An earlier cut opened on
  figure 6's 10 s frame and zoomed in as the dots appeared — the seam was then
  exact, but the camera and the operator moved together and a reader watching the
  picture change could not tell which of the two had done it. The cost is that
  figure 6 opens three times wider, so 5 → 6 is a zoom-out no figure performs.
  Worth a sentence of prose at the handoff.
- **The 3 s window resolves the mains**, which is an accident worth keeping. At
  this scale the 60 Hz ripple is visible on both raw traces with the toggle at 0,
  and it vanishes into the dots at 1 — averaging 25 samples spans 1.5 mains
  cycles. A free preview of what figure 8's notch is for, at no cost in ink.
- **Figure 5 averages each slot; the converter does not.** The dot is the plain
  mean of the real samples in its band, which is what makes the band the whole
  truth about where the dot came from. The ADS1299 decimates with a *third-order*
  sinc — see below — whose weighting reaches into both neighbouring slots, and a
  one-slot band would lie about that. `digitize.adc` carries the real filter.
- **σ converging in figure 9** is the essay's construction. `_run_eog_sm`
  evaluates 1.4826 × MAD exactly once, when the buffer first reaches 5 s; there
  is no running estimate in the game. The loop rests on the value it does
  commit to.

What is *not* divergent, and is asserted on every bake: the chain's structure.
`assert_mirrors_eog_filter()` drives the same code with `eog_core`'s own
constants and demands bit-equality with `_eog_filter`, so if the live chain ever
gains a stage or reorders one, the bake fails.

## Figure 8.5 — threshold amplitude, or threshold velocity

Built 2026-08-14, not yet signed off. Three rows off one poll-by-poll
computation — R − L, the game's filtered amplitude, the same chain's velocity —
each detector row carrying its own ±6σ from the recording's own 5 s
calibration. No control; the loop is the animation. Sidecar bake:
`scripts/bake_ampvel_figure.py` → `data/eog-ampvel.{bin,json}`, fetched only
when the `fig-085` mount is present.

Two deliberate departures from the series, both measured before chosen:

- **The game's corners, not the essay's** — baked through `_eog_filter`
  verbatim. At the essay's LP 100 the derivative amplifies 40–100 Hz broadband
  and velocity *loses* on the 12-recording corpus (531/960 cued glances vs
  amplitude's 708/960 at 6σ); at the game's corners it is amplitude 526/960,
  velocity 926/960.
- **David, not the committed john recording.** On john both rows catch 79/80
  — nothing separates them. David is the total shutout: BOTH rows draw σ from
  the same noisy opening, each at the 100th percentile of its own recording's
  5 s blocks (σ_amp 51.3 µV vs blockwise median 14.9; σ_vel 586.6 µV/s vs
  median 324) — the worst calibration either detector could have drawn,
  inherited equally. Amplitude's 6σ (±308 µV) sits above the recording's peak
  (265 µV), so it cannot fire at all: 0/80. Velocity's slimmer margin (peaks
  2.2× its 6σ) still clears: 77/80. (Alternatives, one-line switch in the
  bake: christy — same shape with a fatter velocity margin, 1/80 vs 79/80,
  peaks 7.5× its 6σ; anthony — milder, 20/80 vs 74/80, amplitude catches the
  big glances and drowns the moderate ones.)

House-style departure: row names sit in the gutter per figures 2/3 but stacked
at 10.5 px over their band (`Plot.label2`) — "amplitude" at the standard 15 px
would cross x=63 into the tick labels.

## Open

- ~~**The velocity step has no figure.**~~ Figure 8.5 (above) now carries it,
  pending sign-off. The essay prose should still say that the real detector
  differentiates *between* steps 8 and 9, and that figures 9–10 stay on
  amplitude while the game thresholds velocity.
- **Figures 2 and 3 are the only invented numbers in the series**, and they are
  flagged `invented: true` in the JSON. They have to be: the array only ever
  contains SRB1-referenced, bias-ON channels, and no PD_BIAS-off session exists in
  the corpus, so the common-mode was never measured. What keeps them honest is
  that the model is an invertible function of the real data — applying it fully
  returns the recording bit for bit — and that every parameter is baked where the
  prose can name it. Recording a bias-off session, or routing BIASOUT to a spare
  channel per SBAS499C §9.3.2.4.2, would turn both into measurements.
- **Figure 3's middle panel is drawn at the leak it removes, not at the bias
  signal's own amplitude.** 280 µV against a real anti-signal of ~1.5 V at the
  body — a factor of ~5300. The prose has to say "the interference the loop
  removes" rather than "the bias signal", or relabel the panel. It also has no
  amplitude wander, which is why it reads as a standing wave: the real envelope is
  measurable (sd/mean 0.22 on R, 0.42 on L, power peaking at 0.061 Hz) but
  erratic, and a single slow sinusoid looked too tidy.
- **Figure 1 asserts a sign, and the recording confirms it.** Averaged over the
  cued glances, a LEFT cue puts ch_L at +39.0 µV and ch_R at −25.4 µV; a RIGHT cue
  puts ch_R at +29.6 µV and ch_L at −34.2 µV — R−L of ∓64 µV, symmetric to within
  a microvolt. The canthus the cornea swings toward is the one that goes positive.
  Note the handedness trap: the figure draws eyes looking OUT at the reader, so
  screen-left is the subject's RIGHT, and as drawn (`GAZE_DIR = −1`) it is a
  RIGHTWARD glance. Flip that constant for a leftward one.
- **Figure 1 carries no gaze angle and no µV.** It is a diagram of where the
  signal comes from, and it stops there. If the prose wants degrees: the canthi
  anti-correlate at −0.92 bandpassed 0.5–20 Hz, and a left→right plateau
  separation of ≈288 µV over a plausible ~27° target separation puts this rig at
  ~11 µV/deg against a literature 15–20. That constant is an assumption, not a
  measurement, and the recording does not store viewing distance.
- **"24-bit" cannot be a figure, and step 5 no longer claims to be one.** The
  LSB is 0.0223517 µV referred to input — verified against the recording, whose
  smallest nonzero sample-to-sample step matches `4.5/24/2²³` to the last digit.
  But the median step between consecutive raw samples is 507 LSB (11.3 µV), so
  the converter is ~500× finer than the noise on the wire and no amount of zoom
  turns a real trace into a staircase. Amplitude quantisation is free here; the
  rate is what costs you. Related and undrawn: decimating this recording to 25 Hz
  drops the measured peak saccade velocity from 13 047 to 5 352 µV/s, which is
  the honest link from the sample rate to figure 10's threshold.
- **The anti-alias filter is the decimator**, so folding it into step 5 as a
  separate operation would double-count. Per SBAS499C §9.3.2.1.1 the ADS1299's
  sinc³ has its first null at f_DR = 250 Hz and *not* at Nyquist: 125 Hz is only
  −11.8 dB and 60 Hz passes at −2.5 dB, which is why the notch downstream has
  work left to do. The passband repeats at every f_MOD = 1.024 MHz, and that
  repetition is the only thing the external R-C is there for.
- **Chrome restores the toggles' checked state across a reload**, so a figure can
  come back with its control saying 1 while the canvas draws 0. `Figure` sets
  `input.checked = false` to prevent exactly this, but it does so *before*
  `mount.appendChild`, and restoration happens on insertion. Pre-existing, and it
  hits every figure — figure 5 only made it obvious because it is now the first
  checkbox in the document. Not fixed: it is the shared shell, not this figure.
- `_eog_filter`'s docstring in `eog_core.py` still says "0.5–100 Hz causal IIR
  chain" while the constants above it read 30 and 0.1. Written 2026-05-22, never
  updated. Not the essay's problem, but it will mislead anyone checking the
  figures against the code.
