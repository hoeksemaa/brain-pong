# 67-minute all-gold free-run — drift/headroom characterization

**Recording:** `data/eog/20260701-133403-john.npz` — all-gold (canthi + earlobes, Ten20), soap-washed skin,
free-run, board original, gain ×24, fs 250, **67.6 min / 1,014,436 samples**. Figures + metrics + script in
`docs/assets/long-freerun/`. Baseline compared against the short gold session `20260630-175142-john.npz` (3.25 min).

## Headline

**Over a full hour the gold rig was rock-solid on the bulk-integrity metrics** — DC offset barely moved
(~2–3 mV total), it never came within 148 mV of the rail, the RMS noise floor was excellent, and the gel did not
dry out. **But two per-channel issues, both traced to a marginal L-canthus electrode, were missed by the aggregate
stats and only surfaced in the viewer's per-channel min/max view (see "L-electrode pops" below):** (1) a
**residual 60 Hz in R−L (~40 µV) that did NOT cancel** — because it's *differential* (anti-correlated, corr −0.93),
a sign of L/R impedance asymmetry — and (2) **persistent L-only contact "pops."** The 60 Hz is entirely removed by
any EOG-band filter; the pops are not (they look like events). Both point to re-prepping the L electrode.

## Drift, railing, headroom (the core question)

| metric | value | verdict |
|---|---|---|
| Per-channel DC offset | CH1 −38 mV, CH2 −35 mV | healthy, mid-range |
| Total per-channel drift over 67 min | **+2.1 / +3.2 mV** | tiny |
| Drift rate | ~+0.03 mV/min (settles to ~0 by min ~10) | settled |
| Worst-case rail use (whole hour) | **20.9% of ±187.5 mV** | huge margin |
| Minimum headroom (whole hour) | **≥148 mV** | never remotely close to rail |
| Railing events | **zero** | — |

Per-channel drift **settles like a decelerating exponential** (rate +0.3–0.45 mV/min in the first ~3 min → ~0 by
min 10) and then holds flat for the rest of the hour. The two channels settle on **different schedules** (CH2
rises in the first ~15 min, CH1 rises in the second half), so their difference produces a **~2.2 mV slow hump in
R−L** (rises to a peak near min 25, declines after) — still trivially small vs 183 mV of R−L headroom, and
linearly detrendable.

### Prediction scorecard (I predicted the 30-min behavior earlier)
- ✅ Ramp bends over and plateaus (~10–15 min) — **correct**.
- ✅ Rail-safe with large headroom the whole time — **correct, decisively**.
- ❌ Magnitude: predicted ~10 mV per-channel rise; actual ~2–3 mV — **overestimated** (soap prep likely reduced it).
- ❌ "R−L stays flat within tens of µV": actual R−L had a ~2.2 mV differential hump from asynchronous channel
  settling — **missed** (still tiny, but a real mV-scale feature I didn't foresee).
- ✅ No sweat waves; ✅ gel didn't dry (60 Hz kept *falling*, didn't rise late).

## Comparison to the short (3.25 min) gold session

| | LONG 67 min | SHORT 3 min | note |
|---|---|---|---|
| DC offset CH1/CH2 | −38 / −35 mV | −28 / −47 mV | both healthy gold range |
| Per-ch drift slope | +0.03 mV/min | +0.9 mV/min | short caught only the early transient; long shows the full settle |
| Per-ch hi-freq noise (s2s) | 23–37 µV | 5–7 µV | long higher — but it's the differential-60Hz leaking per-channel |
| **Per-ch 60 Hz** | ~10–25 µV (falls over hour) | ~3 µV | more mains this session |
| **R−L 60 Hz** | **~40 µV (does NOT cancel)** | **0.65 µV (cancels)** | the one real difference — see below |
| **R−L noise floor, 0.5–15 Hz (EOG band)** | **73 µV** | 76 µV | **identical once filtered** |
| R−L s2s after 15 Hz low-pass | **3.7 µV** | ~6 µV | excellent; 60 Hz fully removable |
| Min headroom | 148 mV | 138 mV | both safe |

**Punchline:** once you filter to the EOG band (which any real pipeline does), the two sessions are equivalent
(~73 vs ~76 µV, dominated by real ocular activity). The gold rig's drift/headroom performance over an hour is
excellent and matches the short-session promise.

## The one interesting difference: differential 60 Hz this session

In the short session the per-channel 60 Hz (~3 µV) was **common-mode** and cancelled in R−L (→0.65 µV). This
session, in a matched mid-window: CH1 60 Hz = 17 µV, CH2 = 23 µV, **R−L = 40 µV** — i.e. R−L is *larger* than
either channel. The 60 Hz on the two channels is **anti-correlated (corr = −0.93)**, so it *adds* in R−L instead
of cancelling. Anti-phase differential 60 Hz is the classic signature of **loop-area pickup in the lead routing**
(the two canthi leads forming a loop that catches magnetic mains) or an electrode-impedance asymmetry between the
canthi. It steps up/down over the hour (min 8, 31, 44), consistent with small lead/posture shifts.

- **Does it matter?** For raw display, yes it's visible; for actual eye-tracking, **no** — a 15 Hz low-pass drops
  R−L high-freq noise from ~49 µV to **3.7 µV**, and EOG lives below ~15 Hz.
- **Fix for next time:** twist/dress the **two eye-electrode leads together** (reduces the differential loop area).
  Note this is a *different* phenomenon from the earlier "noisy first run" (that was contact settling); here it's
  lead-routing-dependent differential pickup, which twisting the signal pair genuinely helps.

## L-electrode pops (NOT low priority — the real finding, spotted in the viewer)

- **CH1 (L) had 265 large transient spikes (>400 µV, ~700–800 µV typical, up to 1200 µV); CH2 (R) had ZERO.**
  0% of L spikes coincide with an R spike → **not ocular** (blinks/saccades hit both canthi). They are
  **L-canthus-electrode-specific contact "pops,"** present the entire session (~3–8/min, count highest in the
  first 5 min, roughly stationary amplitude — NOT progressive). They land in R−L as ~700 µV artifacts (larger than
  real blinks) so they would cause false events in any detector. RMS/s2s missed them (sparse); the viewer's min/max
  decimation caught them instantly. Consistent with the same marginal L electrode that broke the 60 Hz cancellation
  (impedance asymmetry). **Action: re-prep/re-seat the L canthus electrode next session.**

## Other artifacts (low priority)

- **Ocular events:** ~4,800 events >40 µV (≈71/min) — video-driven saccades + blinks over the hour. "Real"
  (large, >200 µV) blinks: **323 (~4.8/min)** — *lower* than typical rest (~15–20/min), consistent with engaged
  video watching. Blink rate steady across the hour (no strong drowsiness signature).
- **Acquisition gaps:** two, both at ~8.3 min (0.21 s + 0.84 s, ~260 samples total) — a localized USB/scheduling
  hiccup coinciding with a small R−L excursion (~480 µV). Negligible for the rest of the record.
- **Big transient at ~42 min:** a **2.8 mV p2p** R−L excursion (broadband in the spectrogram) — a movement /
  swallow / posture shift. The single largest artifact of the session.

## Open questions (ground truth would sharpen this)
1. Around **min 42** (big transient) and **min 8** (gap) — do you recall moving, swallowing, adjusting, or getting up?
2. Did you feel **drowsy or sweat**, or change posture? (No sweat waves seen; blink rate steady — reads as alert/still.)
3. Was the video **captioned or caption-free**? (~71 ocular events/min — captions would add reading sweeps.)
4. How were the **two eye-electrode leads routed** — together or apart? (Bears on the differential 60 Hz.)
