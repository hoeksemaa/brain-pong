# BrainPong tests

Unit tests for the **live EOG detection core** (`eog_core.py`) and the **in-game
recorder** (`recording.py`). Pure/deterministic where possible; no board, no Dash,
and the `.npz` recordings are never mutated.

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

`pyproject.toml` collects **only** `tests/`. The diagnostic-viewer tests
(`test_viewer.py`, `test_viewer_e2e.py`) live in `archive/eog-scripts/` and are
intentionally **not** part of this suite (see "Viewer tests" below).

## What's covered

| file | surface |
|---|---|
| `test_eog_diff.py` | differential + polarity contract; µV scaling; P1/P2 channel independence; fresh-float64-not-a-view |
| `test_eog_filter.py` | filter contract + frequency response (passband / HPF / LPF / 60 Hz notch); no-mutate; short-input passthrough |
| `test_eog_velocity.py` | Engbert–Kliegl velocity (ramp→slope, flat→0, sign=direction); saccade-vs-slow-tail separation |
| `test_matched_filter.py` | unit-norm Hann velocity template + matched-filter direction contract + SNR gain (feeds the sustained-crossing gate) |
| `test_sustained_crossing.py` | persistence gate (kills single spikes); threshold/duration boundaries; onset-not-peak direction; zero-σ / empty guards |
| `test_eog_state_machine.py` | glance-pair protocol (MAD calibration, arm/fire/timeout/refractory); **P1↔P2 independence**; reset keeps channels |
| `test_oscillation_noise.py` | reproduces the oscillating-noise false-fire; candidate run-count discriminator (not shipped) |
| `test_recording.py` | `save_eog_recording` eog-v3 round-trip: fields, event markers, two-player distinct files, collision suffix, shape rejection, detector reconstructability |

`synth.py` provides deterministic signal + calibrated-state helpers (SR=250);
`conftest.py` puts `src/` on `sys.path` so the suite runs without `pip install -e .`.

## Viewer tests (archived — not collected)

`archive/eog-scripts/test_viewer.py` (store + Flask API over the real corpus) and
`test_viewer_e2e.py` (frontend in a real browser via Playwright) travel with the
viewer server, which also lives under `archive/eog-scripts/`. `pyproject.toml`
excludes them so a plain `pytest tests/` passes on a bare checkout. To run them:

```bash
pip install pytest-playwright     # also listed in archive/requirements.txt
playwright install chromium
python -m pytest archive/eog-scripts/test_viewer.py archive/eog-scripts/test_viewer_e2e.py -q
```

The E2E test auto-skips when Playwright or its browser build is missing.

## Design note: written to NOT calcify the code

These pin **behaviour and contracts**, not implementation:

- **Frequency response, not sample values.** `test_eog_filter` asserts "60 Hz is
  attenuated, 8 Hz survives" — true for any reasonable filter chain. Re-tune the
  IIR and the tests still hold. They'd only break if the filter stopped doing its
  *job*, which is what you want.
- **Parameters over magic numbers.** Where behaviour depends on a cutoff or
  threshold it's passed as an explicit arg, so the test means the same thing even
  when the production *default* moves.
- **Injected time.** `_run_eog_sm` takes `now` as an argument → no sleeps, no
  flakiness, full control of the state machine's clock.

## Known-defect tests

`test_oscillation_noise.py::test_suprathreshold_oscillation_reproduces_spurious_fire`
asserts the detector currently **does** fire on oscillation (≥1 phantom command).
It documents a real defect. When a mitigation lands, flip it to assert *zero*
spurious fires.
