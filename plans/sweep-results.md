# Sweep results — `20260427-191502`

- recording: `recordings/20260427-191502.npz`
- subject: `jh`  notes: `sponge`
- variants: 14

## comparison (sorted by accuracy)

| variant | overrides | acc | Δ vs base | L→L | R→R | lat_p50 | lat_p95 | sus_p95 |
|---|---|---|---|---|---|---|---|---|
| 🏆 **hpf-12** | hpf=12.0 | 85.0% | +12.5pp | 75% | 95% | 286ms | 1430ms | 2957ms |
| **baseline** | (defaults) | 72.5% | — | 100% | 45% | 288ms | 3478ms | 2635ms |
| **harmonics-2** | harmonics=2 | 72.5% | +0.0pp | 100% | 45% | 284ms | 3508ms | 2315ms |
| **ema-0.7** | ema_alpha=0.7 | 72.5% | +0.0pp | 100% | 45% | 268ms | 3399ms | 3567ms |
| **ema-0.0** | ema_alpha=0.0 | 72.5% | +0.0pp | 100% | 45% | 575ms | 2406ms | 2460ms |
| **hpf-8** | hpf=8.0 | 70.0% | -2.5pp | 100% | 40% | 268ms | 3478ms | 2604ms |
| **harmonics-5** | harmonics=5 | 70.0% | -2.5pp | 100% | 40% | 290ms | 3386ms | 2656ms |
| **window-2.0** | window_s=2.0 | 70.0% | -2.5pp | 100% | 40% | 380ms | 2854ms | 3470ms |
| **lpf-60** | lpf=60.0 | 70.0% | -2.5pp | 100% | 40% | 284ms | 2464ms | 3227ms |
| **window-1.0** | window_s=1.0 | 67.5% | -5.0pp | 100% | 35% | 196ms | 3348ms | 3694ms |
| **combo-harm5-w2** | harmonics=5 window_s=2.0 | 67.5% | -5.0pp | 100% | 35% | 380ms | 2854ms | 3128ms |
| **freq-12-18** | freq_l=12.0 freq_r=18.0 | 50.0% | -22.5pp | 100% | 0% | 226ms | 4618ms | 3099ms |
| **freq-8-14** | freq_l=8.0 freq_r=14.0 | 50.0% | -22.5pp | 100% | 0% | 186ms | 5535ms | 2565ms |
| **combo-freq12-18-h5** | freq_l=12.0 freq_r=18.0 harmonics=5 | 47.5% | -25.0pp | 95% | 0% | 276ms | 4370ms | 2385ms |

## confusion matrices

(true row → majority predicted column)

### hpf-12 — hpf=12.0  (acc 85.0%)

| true \ pred | L | R | NEUTRAL |
|---|---|---|---|
| **L** | 15 | 5 | 0 |
| **R** | 1 | 19 | 0 |

### baseline — (defaults)  (acc 72.5%)

| true \ pred | L | R | NEUTRAL |
|---|---|---|---|
| **L** | 20 | 0 | 0 |
| **R** | 11 | 9 | 0 |

### harmonics-2 — harmonics=2  (acc 72.5%)

| true \ pred | L | R | NEUTRAL |
|---|---|---|---|
| **L** | 20 | 0 | 0 |
| **R** | 11 | 9 | 0 |

### ema-0.7 — ema_alpha=0.7  (acc 72.5%)

| true \ pred | L | R | NEUTRAL |
|---|---|---|---|
| **L** | 20 | 0 | 0 |
| **R** | 11 | 9 | 0 |

### ema-0.0 — ema_alpha=0.0  (acc 72.5%)

| true \ pred | L | R | NEUTRAL |
|---|---|---|---|
| **L** | 20 | 0 | 0 |
| **R** | 11 | 9 | 0 |

### hpf-8 — hpf=8.0  (acc 70.0%)

| true \ pred | L | R | NEUTRAL |
|---|---|---|---|
| **L** | 20 | 0 | 0 |
| **R** | 12 | 8 | 0 |

### harmonics-5 — harmonics=5  (acc 70.0%)

| true \ pred | L | R | NEUTRAL |
|---|---|---|---|
| **L** | 20 | 0 | 0 |
| **R** | 12 | 8 | 0 |

### window-2.0 — window_s=2.0  (acc 70.0%)

| true \ pred | L | R | NEUTRAL |
|---|---|---|---|
| **L** | 20 | 0 | 0 |
| **R** | 12 | 8 | 0 |

### lpf-60 — lpf=60.0  (acc 70.0%)

| true \ pred | L | R | NEUTRAL |
|---|---|---|---|
| **L** | 20 | 0 | 0 |
| **R** | 12 | 8 | 0 |

### window-1.0 — window_s=1.0  (acc 67.5%)

| true \ pred | L | R | NEUTRAL |
|---|---|---|---|
| **L** | 20 | 0 | 0 |
| **R** | 13 | 7 | 0 |

### combo-harm5-w2 — harmonics=5 window_s=2.0  (acc 67.5%)

| true \ pred | L | R | NEUTRAL |
|---|---|---|---|
| **L** | 20 | 0 | 0 |
| **R** | 13 | 7 | 0 |

### freq-12-18 — freq_l=12.0 freq_r=18.0  (acc 50.0%)

| true \ pred | L | R | NEUTRAL |
|---|---|---|---|
| **L** | 20 | 0 | 0 |
| **R** | 20 | 0 | 0 |

### freq-8-14 — freq_l=8.0 freq_r=14.0  (acc 50.0%)

| true \ pred | L | R | NEUTRAL |
|---|---|---|---|
| **L** | 20 | 0 | 0 |
| **R** | 20 | 0 | 0 |

### combo-freq12-18-h5 — freq_l=12.0 freq_r=18.0 harmonics=5  (acc 47.5%)

| true \ pred | L | R | NEUTRAL |
|---|---|---|---|
| **L** | 19 | 1 | 0 |
| **R** | 20 | 0 | 0 |

## reading the table

- **acc** — % trials where majority vote matched the cue.
- **Δ vs base** — accuracy delta vs `baseline` in percentage points (positive = improvement).
- **L→L / R→R** — per-side accuracy. tells u where errors concentrate. all errors are R→L in the canonical recording, so R→R is the metric to watch.
- **lat_p50 / lat_p95** — first-correct-emission latency post-press, median + 95th percentile.
- **sus_p95** — sustained-correct (3 in a row) p95 latency.
- 🏆 marks the best accuracy row.

## caveat on freq_l / freq_r variants

The recording's actual stimulus frequencies are baked in by the recording session (see metadata). Changing `freq_l` / `freq_r` in the algorithm tells the bench to look for SSVEP at those frequencies — but the brain wasn't stimulated at those frequencies. So `freq-12-18` and `freq-8-14` variants tank, not because the hypothesis is wrong, but because there's no signal to find. Those variants are informational: they show what freq-mismatch failure looks like. To actually test a different freq pair, record a NEW session at that pair and run the sweep on that recording.
