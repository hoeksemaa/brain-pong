# EOG Data Portal — public, static, filterable corpus browser

The **data portal** (`web/` + `web/portal-data/`) is a public, single-page browser
over the whole committed EOG recording corpus. It supersedes the old localhost-only
"EOG Studio" viewer as the *public* face of the data; the Studio viewer is preserved
locally (see below). Built 2026-07-16.

Goal: let anyone slice "all data ever collected" by tag / date / name / pipeline /
quality, and view any recording through each **historical filtering paradigm** the
DSP pipeline passed through — so the signal-processing progression over time is
visible on real data.

## Architecture — a static bake, no server

The Studio viewer needs Flask + the SQLite store + on-demand npz reads (and exposes
real names + write endpoints). The portal instead **pre-bakes** everything into flat
JSON so it can be served as pure static files (GitHub Pages) with no server, no write
endpoints, and — critically — **no real names**.

```
scripts/bake_portal.py   reads data/eog/*.npz  ->  web/portal-data/
  manifest.json    one lean row per recording: metadata + 5 tags + pipeline + spark
  meta.json        corpus aggregates, tag taxonomy, subject list, lens catalogue
  rec/<id>.json    per recording: shared time axis + every paradigm-lens ribbon
                   (min/max decimated, WIDTH=1400) + raw L/R + event markers

web/index.html + portal.css + portal.js   the frontend (vanilla JS, canvas, no build)
```

Frontend fetches `./portal-data/*` relative to the page — same code works under
`python -m http.server web/`, under `serve_viewer.py` (`/`), or on GitHub Pages.
`.github/workflows/pages.yml` deploys `web/` (minus `studio/`) as a Pages artifact;
no Python runs in CI because the data is baked locally and committed.

Re-bake after new recordings: `python scripts/bake_portal.py` (deterministic — frozen
npz produce byte-stable JSON, so only genuinely new recordings churn git). ~15 MB
raw, ~52 KB gzip per recording loaded on demand.

## Privacy by construction

The public output is **name-free**:
- Named subjects → stable pseudonyms (`<first-name>` → `Player A`) built *before*
  anything is written; the owner (`john`) keeps his own name; `P1`/`P2` slot files
  pass through.
- Recording ids become `<date>-<time>-<slug>` so the name isn't in a filename/URL.
- Free-form `notes` (which can name people) are parsed for the `[pipeline-vN]` tag,
  then dropped — only structured, non-identifying metadata + signal ship.
- No raw `.npz` needs to be committed to deploy; the raw recordings stay local.

Verify with: `grep -ril -E '<real names>' web/portal-data` → nothing but john/P1/P2.

## The five tags (+ pipeline axis)

Deterministic derivation in `bake_portal.derive_tags` (see cross-reference in git
history). Coverage on the current 113-file corpus:

| Axis | Values (counts) | Basis |
|---|---|---|
| tournament | yes 66 · no 47 | `date == 2026-07-13` |
| session_type | game 57 · training 44 · cued 12 | protocol_version + `training` tag |
| rig_board | v1.2 67 · v1.3 46 | `board_version` (original → v1.2) |
| electrode_type | gold 113 | documented rig-state (no clip files in corpus) |
| cleaning_regimen | prepped 66 · unknown 47 | prepped iff tournament day (docs) |
| *pipeline / detector* | velocity 43 · matched 58 · v2 45 · v3 56 | `detector` field + notes tag |

**Two caveats surfaced by the data** (worth knowing before adding more tags):
- `electrode_type` is **uniform gold** — the carbon-clip A/B recordings (`20260630-*`)
  were never committed, so this axis has no filtering power *yet* (kept for future
  recordings; rendered as a static info chip, not an interactive filter).
- `cleaning_regimen` prepped/unknown is **identical to tournament** (both = the 07-13
  date). Redundant today; kept forward-looking.
- The genuinely-discriminating axis is **detector/pipeline**, added as its own filter
  — it's also the list-level form of the "historic paradigm" filtering.

## The paradigm lenses (EOG-only progression)

Each lens re-filters the raw L−R differential through one era's chain (zero-phase —
offline display is non-causal by design). Defined in `bake_portal.LENSES`:

| Lens | Date | Unit | Chain |
|---|---|---|---|
| Raw | — | µV | identity |
| EOG preflight | Apr 29 | µV | 0.5–30 Hz + 50/60 notch |
| Offline baseline | May 21 | µV | 0.5–100 Hz (wide LPF passes EMG) |
| Clinical HEOG | May 21 | µV | 0.1–30 Hz literature band |
| Velocity | Jul 9 | µV/s | Engbert–Kliegl / SavGol derivative |
| Matched filter | Jul 9 | µV/s | velocity ⊛ 120 ms Hann template |
| z-normalized | — | σ | ÷ baseline σ; 6σ fire-threshold line |

(SSVEP-CCA is excluded — it decodes occipital flicker, not EOG.)

## Features

Filters (all captured in the URL for shareable views): subject search, tournament,
session type, rig/board, detector, pipeline, skin prep, signal quality (ok/railing/
flat), date range, max rail %, min duration. Plus: corpus dashboard strip (totals +
quality bar), opponent pairing for 2-player matches (shared timestamp), per-recording
paradigm-lens timeline, raw L/R electrode traces, an About panel, and a link to the
existing CMRR explainer (`web/cmrr/`).

## Running / hosting

```bash
python scripts/bake_portal.py        # (re)build web/portal-data/ from data/eog/
python -m http.server --directory web 8899   # or: python scripts/serve_viewer.py
# -> http://localhost:8899/           public portal
# -> http://localhost:8770/studio/    local diagnostic viewer (real names + trims)
```

Public hosting is GitHub Pages via `.github/workflows/pages.yml`. Enabling it
(Settings → Pages → Source: GitHub Actions, or `gh api`) publishes to
`https://hoeksemaa.github.io/brain-pong/`. **Not enabled/pushed without an explicit
go** — it puts (anonymized) biosignal data on the internet.
