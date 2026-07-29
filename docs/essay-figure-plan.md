
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
| 1 | eyes move — corneoretinal dipole rotates | simulated | **not started** |
| 2 | measured against SRB1 | spoofed back | **not started** |
| 3 | bias driven into the body *(simultaneous with 2)* | spoofed back | **not started** |
| 4 | amplified ×24 | spoofed back | **not started** |
| 5 | digitized — 250 Hz, 24-bit *(fold anti-alias in here)* | → real from here | **not started** |
| 6 | left subtracted from right | real | **DONE** |
| 7 | detrend — one constant per 0.1 s poll, over 0.5 s | real | **DONE** |
| 8 | filter — LP 100, notch 48–52/58–62, HP 0.5 | real | **DONE** |
| 9 | calibrate → σ | real | **DONE** |
| 10 | threshold → spike detected | real | **DONE** |

**DONE** means John has signed it off. Steps 1–5 have nothing built — no draft,
no data baked. Everything from 6 down is built, signed off, and live in the
harness.

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

Established across 6–10, and worth holding to for 1–5.

- **Seams.** A figure opens on the state the previous one closed on — same
  array, same span, same centring rule. The reader should feel the steps
  accumulate rather than restart.
- **Colour.** Black is what goes in, red is what the step produces, grey is
  axes and gridlines. Green is a second input channel and appears only in
  figure 6. A figure's input and output are never the same colour even when
  they are the same array: figure 6's red differential is figure 7's black.
- **One second** for every toggle transition, everywhere (`TRANSITION_S`).
- **No text on the figure** beyond axis labels, cue guides, and a readout that
  is genuinely the step's output — σ in 9, the counts in 10. The figures are
  read inline beside prose that explains them; anything the sentence next to it
  already says is ink competing with the trace.
- **Geometry.** One plot size everywhere: 430 px wide, 150 px tall, 60 px from
  the top of the figure. Controls reserve a fixed 150 px gutter on the right
  whether or not a figure has any, otherwise the canvas flexes and the same ten
  seconds gets drawn at different scales in different steps.

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
- **σ converging in figure 9** is the essay's construction. `_run_eog_sm`
  evaluates 1.4826 × MAD exactly once, when the buffer first reaches 5 s; there
  is no running estimate in the game. The loop rests on the value it does
  commit to.

What is *not* divergent, and is asserted on every bake: the chain's structure.
`assert_mirrors_eog_filter()` drives the same code with `eog_core`'s own
constants and demands bit-equality with `_eog_filter`, so if the live chain ever
gains a stage or reorders one, the bake fails.

## Open

- **The velocity step has no figure.** It is a real operation — it is what turns
  a filter chain into a saccade detector — and the step list skips it. Figures 8
  and 9 hand amplitude to amplitude so nothing is broken, but the essay should
  say in prose that the real detector differentiates here, or it needs a step
  8.5.
- **Steps 1–5.** Nothing built. The hard part is that each pre-ADC figure has to
  be an invertible function applied to the real recording rather than an invented
  waveform, and step 1 needs a gaze angle that the recording does not store.
- `_eog_filter`'s docstring in `eog_core.py` still says "0.5–100 Hz causal IIR
  chain" while the constants above it read 30 and 0.1. Written 2026-05-22, never
  updated. Not the essay's problem, but it will mislead anyone checking the
  figures against the code.
