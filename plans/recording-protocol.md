# BrainPong recording protocol v1

The reference protocol for collecting labeled SSVEP-EEG sessions for BrainPong. Step 3 (the `--record` flag) implements this verbatim. All future training data and benchmark recordings should follow it.

If anything here changes, **bump the protocol version** (`v1` → `v2`) and save the version number in every session's metadata, so we can tell post-hoc which data came from which protocol.

---

## Goal

Produce a corpus of labeled EEG recordings where, for every sample, we know whether the subject was gazing at the LEFT band, the RIGHT band, or resting.

Used by:
- Step 6 (automated latency benchmark) — needs ground-truth gaze onset events.
- Future ML classifier — needs supervised steady-state windows.
- Step 5 sanity check — needs labeled segments for PSD inspection.

---

## Session structure

| element | value |
|---|---|
| trial length | **10 s** of sustained gaze |
| trials per direction | **20 LEFT + 20 RIGHT = 40 total** |
| ordering | **strict alternation** (L, R, L, R, …) starting with LEFT |
| inter-trial rest | **3 s**, central fixation cross, no flicker |
| total session time | ~9 min trials + ~2 min calibration check + setup overhead ≈ 11 min |
| breaks | none built-in. if subject reports fatigue, abort and resume in a new session file |

The flicker bands run for the entire session at their target frequencies (10 Hz left, 15 Hz right). Only the **cue** changes per trial — the stimulus is constant.

---

## Cue / display

Center play area shows:

| phase | display |
|---|---|
| trial (10 s) | large arrow `←` (LEFT trial) or `→` (RIGHT trial), centered |
| rest (3 s)   | small `+` fixation cross, centered |
| pre-session | "READY — press space to begin" |
| post-session | "DONE — N trials recorded" |

The arrow is the only LEFT/RIGHT cue. No text, no audio. Subject must look at the band corresponding to the arrow direction (left arrow → look at left flicker band).

Flicker bands run continuously in PLAYING-equivalent mode throughout the recording session.

---

## Keypress semantics

| event | action | logged as |
|---|---|---|
| trial cue appears | (subject begins shifting gaze) | `t_cue_onset` |
| subject presses space | "I am now looking at the cued band" | `t_press` |
| subject releases space | "I am no longer looking" | `t_release` |
| trial cue disappears | (subject can rest) | `t_cue_offset` |

Subject is instructed:
> "Press space the *instant* you begin looking at the band. Hold it as long as you maintain gaze. Release the moment you look away. Try to minimize the gap between your gaze shift and your keypress."

### Steady-state window (the part used for training labels)

```
[t_press + 1500 ms,  t_release]
```

The first 1.5 s after press is dropped to skip the SSVEP buildup transient. Everything from there until release is treated as cleanly labeled steady-state at the cued direction.

Frames *outside* press/release intervals are unlabeled (treat as "unknown" — could be transition, blink, distraction).

### Why this works for both latency and accuracy benchmarks

- **Latency benchmark**: `t_press` is ground-truth event time. Pipeline latency = `t_command − t_press`. Human RT (~200–300 ms from intent → press) is a constant offset and either subtracts cleanly or just inflates the absolute number. Doesn't affect *relative* latency comparisons across algorithm changes.
- **Accuracy benchmark**: classifier is evaluated on steady-state windows where the gaze direction is known. If subject is honest about press/release timing, label noise stays low.

---

## Metadata captured per session

Saved into the `.npz` alongside the EEG data. Capture liberally — easier to ignore unused fields than to re-collect missing ones.

```python
metadata = {
    'protocol_version': 'v1',
    'session_id':       <uuid or timestamp>,
    'subject_id':       <free-text>,
    'started_at_iso':   <ISO 8601>,
    'ended_at_iso':     <ISO 8601>,

    # Hardware
    'board_id':         BOARD_ID,
    'sampling_rate':    sampling_rate,
    'eeg_channel_indices': bci_eeg_channels,   # actual indices used
    'serial_port':      params.serial_port,

    # Stimulus (read from JS at session start; do NOT use nominal values)
    'stimulus_left_hz':  actualLeftHz,         # from window.dash_clientside.brainpong_measurement
    'stimulus_right_hz': actualRightHz,
    'display_refresh_hz': measuredHz,
    'left_period_frames': leftPeriodFrames,
    'right_period_frames': rightPeriodFrames,
    'browser_user_agent': navigator.userAgent,

    # Filter chain (so future-us knows what was applied)
    'filter_low_cut_hz':  FILTER_LOW_CUT_HZ,
    'filter_high_cut_hz': FILTER_HIGH_CUT_HZ,
    'filter_order':       FILTER_ORDER,

    # Free-text headset notes (impedance, fit, anything weird)
    'headset_notes':    <free-text from CLI prompt>,
}
```

---

## File format & layout

One file per session: `recordings/<session_id>.npz`

`session_id` = `YYYYMMDD-HHMMSS` (sortable, unique enough for a single subject).

### Arrays inside the .npz

| key | shape | dtype | description |
|---|---|---|---|
| `eeg`        | `(n_channels, n_samples)` | float32 | raw EEG, channel-first, untouched by any DSP |
| `eeg_t`      | `(n_samples,)`            | float64 | timestamp per sample, seconds since session start |
| `events`     | `(n_events,)` structured  | (see below) | one row per cue/press/release/edge event |
| `metadata`   | scalar object array       | dict (pickled) | the metadata dict above |
| `edge_log`   | `(n_edges,)` structured   | (see below) | stimulus edge transitions from `brainpong_edge_log` |

### `events` dtype

```python
np.dtype([
    ('t', 'f8'),        # seconds since session start
    ('kind', 'U16'),    # 'cue_onset' | 'cue_offset' | 'press' | 'release'
    ('side', 'U1'),     # 'L' | 'R' | '' (empty for press/release)
    ('trial_idx', 'i4'),
])
```

### `edge_log` dtype

```python
np.dtype([
    ('t', 'f8'),       # seconds since session start
    ('frame', 'i8'),   # rAF frame index
    ('side', 'U1'),    # 'L' | 'R'
    ('is_on', '?'),    # bool
])
```

### How to load + inspect (Python)

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.load('recordings/20260427-160000.npz', allow_pickle=True)
eeg, eeg_t, events, edge_log = z['eeg'], z['eeg_t'], z['events'], z['edge_log']
metadata = z['metadata'].item()
print(metadata['stimulus_left_hz'], metadata['stimulus_right_hz'])

# Plot channel 0, with cue boundaries overlaid
plt.plot(eeg_t, eeg[0])
for ev in events:
    if ev['kind'] == 'cue_onset':
        plt.axvline(ev['t'], color='g' if ev['side'] == 'L' else 'r', alpha=0.3)
plt.show()
```

---

## Concrete session walkthrough

```
0. Setup
   - subject puts on headset, electrodes are wet/secure
   - run: source .venv/bin/activate && python pong_game_brainflow.py --record
   - browser opens to recording page in Chrome, fullscreen
   - CLI prompts for subject_id, headset_notes
   - JS measurement runs (~1 s); refuses to proceed if refresh isn't ≥ 60 Hz stable

1. Calibration sanity (~30 s)
   - flicker bands run; subject does a quick informal LEFT/RIGHT check
   - PSD is shown in real-time; verify peak at expected stim freq when looking left/right
   - subject presses space to begin recorded trials

2. Trials (~9 min, 40 × 13 s)
   - cue arrow appears
   - subject shifts gaze, presses space
   - holds for ~10 s of sustained gaze
   - cue disappears, subject releases space, fixation cross appears for 3 s
   - alternates L/R for 40 trials

3. Wrap
   - "DONE — 40 trials recorded" screen
   - data is flushed to recordings/<session_id>.npz
   - print summary: session duration, # trials, mean press-to-release duration
```

---

## Known caveats / sources of label noise

- **Human reaction time on press/release.** ~200–300 ms typical. The 1.5 s post-press dropout absorbs this for training. For latency benchmarks, treat as a constant offset.
- **Saccade asymmetry.** Looking left vs right may have different RTs depending on subject handedness / eye dominance. Not corrected for. Document if it shows up in the data.
- **Calibration drift over a session.** Headset impedance can change as gel dries. Single-session protocol is short (~9 min) to minimize this; if longer sessions are desired later, add periodic re-calibration trials.
- **Subject inattention.** No way to detect a trial where the subject zoned out but kept space pressed. Visual inspection + outlier detection in post-hoc analysis is the only check.

---

## What this protocol explicitly does NOT do (yet)

- No 3-class label (REST is implicit between trials, not its own labeled trial). If we want a "neutral" classifier output, add REST trials in v2.
- No eye tracker — gaze direction is inferred from cue + keypress, not directly measured.
- No randomized inter-trial intervals (always 3 s). If we worry about anticipation effects, add jitter in v2.
- No counterbalancing for L-first vs R-first across sessions. With one subject and short sessions this isn't worth controlling for.
- No multi-subject support beyond a `subject_id` field. If we add testers (step 4 of `today.md` won't have time today, but eventually), this should expand to per-subject directories.
