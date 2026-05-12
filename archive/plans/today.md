# Today's plan - EOG Pong bringup

Date: 2026-04-29

Goal: get BrainPong controlled by EOG at least once today. The day is gated:
first prove the eye signal, then make it move a paddle, then make it fun, then
optimize and automate if there is time left.

End-of-day target:
- Clean left/right EOG separation on the live plot.
- A working Pong mode controlled by eye movement.
- At least one saved note or recording that says which reference/bias/ground
  setup worked best.

Stretch target:
- EOG recording path plus a tiny replay/threshold bench so future changes are
  measured instead of guessed.

---

## Step 1 - EOG hardware bringup

Use `eog_filtered_plot.py` as the gatekeeper. Do not integrate with the game
until the plot clearly shows intentional left/right movement.

Command:

```bash
source .venv/bin/activate
python eog_filtered_plot.py
```

Default montage:
- Left outer canthus
- Right outer canthus
- Above one eye
- Below the same eye
- SRB/reference on mastoid or earlobe
- BIAS/ground on other mastoid or forehead

Derived channels to watch:
- `horizontal = right_outer - left_outer`
- `vertical = above_eye - below_eye`

Sanity sequence for every wiring setup:
- 10 s center/rest.
- 5 hard-left looks, about 1-2 s each.
- 5 hard-right looks, about 1-2 s each.
- 5 blinks.
- One minute of mostly-still rest to check drift and saturation.

Success criteria:
- Rest is not saturated or railing.
- Left and right gaze produce large opposite-polarity swings in horizontal EOG.
- Blinks show up strongly in vertical EOG.
- Horizontal is not dominated by blink artifacts.
- The signal recovers to baseline after gaze shifts.

Reference/bias/ground test matrix:
- SRB/reference on mastoid, BIAS on other mastoid.
- SRB/reference on mastoid, BIAS on forehead.
- Swap mastoid/forehead placements if the first two are noisy.
- If there is a plausible board ground option, test it explicitly against BIAS.
- Bias disconnected only as a negative-control sanity check, not as a likely
  final setup.

For each setup, write down:
- Electrode placements.
- Whether the board streamed cleanly.
- Rest noise / drift impression.
- Left/right separation impression.
- Blink contamination impression.
- Comfort and contact notes.
- Verdict: keep / retry / reject.

### Skin and electrode prep

Supplies:
- Gold-plated cup/disc electrodes.
- Ten20 conductive paste.
- Cotton pads or gauze.
- Warm water.
- Mild soap or face wash.
- Alcohol wipe if tolerated and kept away from the eye area.
- Tape for strain relief and electrode hold-down.
- Towel / mirror / trash bag. Ten20 is useful but messy.

Before placement:
- Start with healthy, intact skin only. Do not place electrodes over cuts,
  active irritation, acne that has opened, or recently abraded skin.
- Wash hands.
- Clean each site with mild soap/water or a damp cotton pad, then dry fully.
- If using alcohol, use a tiny amount only on outer-canthus / forehead / mastoid
  skin and let it fully dry. Do not let it run toward the eyes.
- Optional: very gently rub the site with dry gauze or a cotton pad for a few
  seconds to remove oil and dead skin. Do not scrape, especially near the eyes.

Applying Ten20:
- Put a small mound of Ten20 into the cup/contact area of each electrode.
- Use enough paste to fill the electrode contact area with no air pocket, but
  not so much that paste from nearby electrodes can touch.
- Place the electrode onto the prepared skin and press with steady medium
  pressure for a few seconds.
- For the above/below eye pair, keep paste localized. Avoid bridging paste
  between electrodes.
- Keep Ten20 out of the eye. If it gets in the eye, stop and rinse with warm
  water for 10-15 min; do not rub.

Securing:
- Tape the electrode body lightly enough that skin is not pulled tight.
- Add separate cable strain relief: tape each wire to cheek/temple/neck so
  cable tug does not move the electrode.
- Route wires together and avoid dangling loops near the face.
- Re-check that no tape or paste is pulling the eyelid or eyelashes.

Before streaming:
- Sit still for 30-60 s and watch for saturation, railing, or huge drift.
- Ask: does a gentle wire tug move the trace? If yes, improve strain relief.
- If one channel is noisy, fix contact before changing filters.

After the session:
- Stop the stream before removing electrodes.
- Peel tape slowly while supporting the skin.
- Remove Ten20 with warm water and mild soap.
- Check skin for redness, soreness, burning, itching, or swelling.
- Rinse electrodes in warm water promptly. Let Ten20 soften, wipe clean, and
  dry the electrodes before storing.

Expected time: 60-90 min.

---

## Step 2 - Minimal EOG command detector

Build the simplest detector that can be trusted in real time.

First-pass signal path:
- Pull the same four active EOG channels.
- Filter for EOG, not SSVEP: high-pass around 0.3-0.5 Hz, low-pass around
  10-30 Hz, notch 50/60 Hz.
- Compute horizontal and vertical derived traces.
- Track a rolling neutral baseline.
- Use threshold plus hysteresis:
  - horizontal above right threshold -> `RIGHT`
  - horizontal below left threshold -> `LEFT`
  - inside neutral band -> `NEUTRAL`
- Add a simple blink guard from vertical spikes if blinks trigger false
  commands.

Calibration sketch:
- Center/rest for a few seconds.
- Look left for a few seconds.
- Look right for a few seconds.
- Set thresholds from the observed left/right distributions with a neutral
  dead zone.

Success criteria:
- Intentional left/right labels are obvious in console, plot, or app state.
- Sitting still produces mostly `NEUTRAL`.
- Command latency feels comfortably below the SSVEP path.

Expected time: 60-120 min.

---

## Step 3 - Pong integration

Add EOG as a control path beside the current SSVEP path. Keep the first version
small and reversible.

Preferred CLI shape:

```bash
source .venv/bin/activate
python pong_game_brainflow.py --control eog
```

Implementation outline:
- Leave SSVEP behavior alone.
- In EOG mode, skip SSVEP flicker/calibration.
- Run the short EOG calibration from Step 2.
- During play, convert EOG labels into paddle movement.
- Keep keyboard fallback available for debugging.
- Show a small command/confidence indicator so the player can see what the
  detector thinks.

Success criteria:
- Eye movement moves the paddle left and right.
- Center gaze or rest does not constantly drift the paddle.
- The game can be played for at least one short round without restarting.

Expected time: 90-150 min.

---

## Step 4 - Make it pretty and fun

Only start this once EOG control works.

High-ROI polish:
- Remove or hide SSVEP flicker panels in EOG mode.
- Make the playfield calmer and more game-like.
- Add paddle-hit feedback: particles, quick flash, score pop, or small shake.
- Add an EOG status meter that shows left/neutral/right confidence.
- Improve start/calibration screens for EOG mode.

Avoid today:
- Full design-system rewrite.
- Big mode expansion before the control loop feels good.
- Polish that makes signal debugging harder.

Expected time: 60-120 min, only after Step 3.

---

## Step 5 - Squeeze the juice

Only start this if the game already works.

Best use of extra time:
- Add `--record-eog` or extend the existing recording schema for EOG sessions.
- Save raw channels, derived horizontal/vertical traces, emitted commands,
  montage metadata, and reference/bias/ground notes.
- Build a tiny EOG replay bench that sweeps:
  - thresholds
  - hysteresis width
  - smoothing/window length
  - blink-rejection settings
- Compare wiring setups quantitatively once recordings exist.
- Try better signal processing only after the simple threshold baseline is
  measured.

Possible upgrades after the baseline:
- Adaptive neutral baseline.
- Robust artifact rejection for blinks and jaw/muscle spikes.
- Per-session learned classifier using rest/left/right calibration data.
- Automated tests around the command detector using recorded EOG.
- Backburner game direction: turn BrainPong into a two-person game once the
  one-player EOG control loop is reliable. Do not spend today on this unless
  the EOG game already works and there is real spare time.

---

## Hard gates

- If Step 1 fails, do not touch Pong. Fix contact, placement, reference, bias,
  or ground first.
- If Step 2 cannot produce stable labels, do not spend time on visual polish.
- If Step 3 works, preserve that playable build before refactoring.
- Any optimization must beat a measured simple-threshold baseline.

---

## Running log

Use this section during the day.

| Time | Setup | Observation | Verdict |
|---|---|---|---|
| | | | |
