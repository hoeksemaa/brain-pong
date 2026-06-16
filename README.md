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

## Key files

**Live game**
- `pong_game_brainflow.py` — the game (Dash app). Modes: `--no-board` (keyboard), `--eog` (1-player), `--2player`.
- `eog_core.py` — shared realtime glance-pair detector; single source of truth for 1- and 2-player paths.
- `record_eog.py` — cued LEFT/RIGHT/REST recorder → labeled `.npz` for training/eval.
- `eog_display.py` — raw 2-channel signal sanity check (are the electrodes alive?).

**Offline eval pipeline**
- `preprocess.py` — `npz → PrepResult` (filtering / normalization variants).
- `detect.py` — `PrepResult → DetectResult` (accuracy, latency, false-positive rate).
- `bench.py` — runs the preprocessor × detector grid over all recordings → JSON in `results/`.
- `eval_simple.py`, `eval_classifier.py` — standalone classifier evals (peak-sign; leave-one-subject-out z-threshold).
- `plot_recording.py`, `plot_trials.py`, `plot_subjects.py`, `filtered_plot.py` — visualization.

**Data**
- `recordings/eog/` — labeled EOG sessions (`eog-v1`). Read-only ground truth; never mutated.
- `archive/` — prior SSVEP pong work. Reference only; superseded by the EOG approach.

## Running it

```bash
source .venv/bin/activate          # project-local venv (Python 3.13)

python eog_display.py              # check the signal is alive
python record_eog.py --subject me  # record labeled gaze data
python bench.py                    # eval pipelines against all recordings

python pong_game_brainflow.py --no-board   # keyboard only, no hardware
python pong_game_brainflow.py --eog        # 1-player EOG (needs board)
python pong_game_brainflow.py --2player    # 2-player EOG (P1=ch1-2, P2=ch3-4)
```

Hardware modes need Cerelog's BrainFlow fork (provides `CERELOG_X8_BOARD`), not upstream PyPI brainflow:

```bash
pip install -e /path/to/Shared_brainflow-cerelog/python_package
```
