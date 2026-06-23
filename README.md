# BrainPong

Play Pong with your eyes. A 2-electrode horizontal **EOG** (electrooculography) setup reads
left/right eye movements off a [Cerelog X8](https://cerelog.com) and drives the paddle in real time.

<a href="assets/eye-pong-demo.mp4"><img src="assets/eye-pong-poster.jpg" width="320" alt="BrainPong EOG demo — click to play"></a>

## How it works

```
Cerelog X8 → 2-ch horizontal EOG → differential (ch_R − ch_L)
  → detrend / bandpass / notch → glance-pair detector → {LEFT, RIGHT, NEUTRAL} → paddle
```

A *glance pair* is the saccade out + return saccade of a deliberate look; the detector keys on that
signature rather than raw amplitude, which keeps it robust to drift.

## Layout

- `src/brainpong/` — importable library: `eog_core` (realtime glance-pair detector), `preprocess`, `detect`.
- `scripts/` — runnable entrypoints (game, recorder, eval, plots).
- `data/eog/` — labeled EOG sessions (`eog-v1`/`eog-v2`). Read-only ground truth; never mutated.
- `derivatives/` — regenerable outputs (`derivatives/results/` = benchmark JSON).
- `archive/` — prior SSVEP pong work. Reference only; superseded by the EOG approach.

## Key files

**Live game**
- `scripts/pong_game_brainflow.py` — the game (Dash app). Modes: `--no-board` (keyboard), `--eog` (1-player), `--2player`.
- `src/brainpong/eog_core.py` — shared realtime glance-pair detector; single source of truth for 1- and 2-player paths.
- `scripts/record_eog.py` — cued LEFT/RIGHT/REST recorder → labeled `.npz` for training/eval.
- `scripts/eog_display.py` — raw 2-channel signal sanity check (are the electrodes alive?).

**Offline eval pipeline**
- `src/brainpong/preprocess.py` — `npz → PrepResult` (filtering / normalization variants).
- `src/brainpong/detect.py` — `PrepResult → DetectResult` (accuracy, latency, false-positive rate).
- `scripts/bench.py` — runs the preprocessor × detector grid over all recordings → JSON in `derivatives/results/`.
- `scripts/eval_simple.py`, `scripts/eval_classifier.py` — standalone classifier evals (peak-sign; leave-one-subject-out z-threshold).
- `scripts/plot_recording.py`, `scripts/plot_trials.py`, `scripts/plot_subjects.py` — visualization.

## Running it

```bash
source .venv/bin/activate          # project-local venv (Python 3.13)
pip install -e .                   # once: makes `brainpong` importable

python scripts/eog_display.py              # check the signal is alive
python scripts/record_eog.py --subject me  # record labeled gaze data
python scripts/bench.py                    # eval pipelines against all recordings

python scripts/pong_game_brainflow.py --no-board   # keyboard only, no hardware
python scripts/pong_game_brainflow.py --eog        # 1-player EOG (needs board)
python scripts/pong_game_brainflow.py --2player    # 2-player EOG (P1=ch1-2, P2=ch3-4)
```

Hardware modes need Cerelog's BrainFlow fork (provides `CERELOG_X8_BOARD`), not upstream PyPI brainflow:

```bash
pip install -e /path/to/Shared_brainflow-cerelog/python_package
```
