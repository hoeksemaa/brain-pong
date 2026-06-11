# BrainPong tests

Unit tests for the **shared EOG detection core** (`eog_core.py`) — the functions
both the 1-player and 2-player games run through. Pure/deterministic; no board,
no Dash, no recordings touched.

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

## What's covered

| file | surface |
|---|---|
| `test_eog_diff.py` | differential + polarity contract; P1/P2 channel independence |
| `test_eog_filter.py` | filter contract + frequency response (passband / HPF / LPF / 60 Hz notch) |
| `test_sustained_crossing.py` | persistence gate (kills single spikes); threshold/duration boundaries |
| `test_eog_state_machine.py` | glance-pair protocol, refractory, **P1↔P2 independence** |
| `test_oscillation_noise.py` | reproduces the oscillating-noise false-fire; candidate discriminator |

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
