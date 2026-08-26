"""
End-to-end tests for the EOG web viewer: store (SQLite + npz decimation + trim)
and the Flask API. Ingests the real frozen corpus into a throwaway temp DB, so
it never touches data/eog or derivatives/viewer.db.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))  # serve_viewer lives in scripts/

from brainpong import store          # noqa: E402
import serve_viewer                  # noqa: E402

NPZ_DIR = REPO / "data" / "eog"
Player F = "20260519-145529-playerF"
Player G = "20260622-145533-playerG"


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("viewer") / "test.db"
    n = store.ingest_dir(str(path), str(NPZ_DIR))
    assert n >= 8
    return str(path)


@pytest.fixture()
def client(db):
    return serve_viewer.create_app(db).test_client()


# ── store ────────────────────────────────────────────────────────────────────

def test_list_is_problems_first(db):
    statuses = [r["status"] for r in store.list_recordings(db)]
    last_problem = max((i for i, s in enumerate(statuses) if s != "ok"), default=-1)
    first_ok = next((i for i, s in enumerate(statuses) if s == "ok"), len(statuses))
    assert last_problem < first_ok          # no 'ok' before any problem


def test_aaron_flagged_railing_v1_gain_assumed(db):
    a = {r["id"]: r for r in store.list_recordings(db)}[Player F]
    assert a["status"] == "railing" and a["rail_pct"] > 5
    assert a["gain"] == 24 and a["gain_assumed"] == 1      # eog-v1 → assumed default
    assert len(a["spark"]) == 64


def test_get_recording_events(db):
    r = store.get_recording(db, Player F)
    assert r is not None and len(r["events"]) == 102
    assert {e["label"] for e in r["events"]} >= {"LEFT", "RIGHT", "BASELINE"}


def test_ribbon_filters_units_and_widths(db):
    for f in ("raw", "bp_0530", "bp_0130", "velocity"):
        rb = store.ribbon(db, Player G, filt=f, width=500)
        assert rb["filter"] == f
        assert 0 < len(rb["t"]) == len(rb["mn"]) == len(rb["mx"]) <= 500
    assert store.ribbon(db, Player G, filt="velocity")["unit"] == "µV/s"
    assert store.ribbon(db, Player G, filt="raw")["unit"] == "µV"


def test_unknown_filter_falls_back_to_raw(db):
    assert store.ribbon(db, Player G, filt="bogus")["filter"] == "raw"


def test_decimation_preserves_extremes(db):
    """min/max-per-bucket must keep Player F's rail spikes — no silent smoothing."""
    src = store._source(db, Player F)
    sig = store._load_npz(src)
    ch = store.channels(db, Player F, width=300)
    for c in ch["channels"]:
        true_peak = float(np.max(np.abs(sig["eeg"][c["row"]] * 1e6)))
        deci_peak = max(abs(v) for v in c["mn"] + c["mx"])
        assert deci_peak >= 0.99 * true_peak


def test_window_restricts_time_range(db):
    rb = store.ribbon(db, Player G, t0=10, t1=50, width=500)
    assert rb["t"][0] >= 9.9 and rb["t"][-1] <= 50.1


def test_eeg_rows(db):
    assert store.eeg_rows(db, Player G) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_window_health_full_matches_stored(db):
    stored = {r["id"]: r for r in store.list_recordings(db)}[Player F]["rail_pct"]
    full = store.window_health(db, Player F)
    assert full["rail_pct"] == pytest.approx(stored, abs=0.01)   # untrimmed == ingest value


def test_window_health_tracks_trim_window(db):
    """Player F rails only in the electrode-off tail; the clean first half is ~0%."""
    clean_half = store.window_health(db, Player F, t0=0, t1=180)
    tail = store.window_health(db, Player F, t0=200, t1=289)
    full = store.window_health(db, Player F)
    assert clean_half["rail_pct"] < 0.5 and clean_half["status"] == "ok"
    assert tail["rail_pct"] > full["rail_pct"] > 1.0          # tail is the culprit
    assert tail["status"] == "railing"


def test_api_health_windowed(client):
    full = client.get(f"/api/recordings/{Player F}/health").get_json()
    clean = client.get(f"/api/recordings/{Player F}/health?t0=0&t1=180").get_json()
    assert full["rail_pct"] > 5 and clean["rail_pct"] < 0.5
    assert client.get("/api/recordings/nope/health").status_code == 404


def test_trim_crud(db):
    store.clear_trim(db, Player G)
    assert store.get_trim(db, Player G) is None
    store.set_trim(db, Player G, 1.0, 5.0, "unit-test")
    t = store.get_trim(db, Player G)
    assert t["t0"] == 1.0 and t["t1"] == 5.0 and t["reason"] == "unit-test"
    store.set_trim(db, Player G, 2.0, 9.0)          # upsert
    assert store.get_trim(db, Player G)["t1"] == 9.0
    store.clear_trim(db, Player G)
    assert store.get_trim(db, Player G) is None


def test_npz_never_mutated(db):
    """Deriving the ribbon/channels must not write to the frozen npz."""
    src = store._source(db, Player F)
    before = (Path(src).stat().st_mtime, float(np.load(src, allow_pickle=True)["eeg"].sum()))
    store.ribbon(db, Player F, filt="bp_0530")
    store.channels(db, Player F, rows=[1, 2, 3])
    store.set_trim(db, Player F, 0, 100)
    store.clear_trim(db, Player F)
    after = (Path(src).stat().st_mtime, float(np.load(src, allow_pickle=True)["eeg"].sum()))
    assert before == after


# ── API ──────────────────────────────────────────────────────────────────────

def test_api_list(client):
    r = client.get("/api/recordings")
    assert r.status_code == 200 and len(r.get_json()) >= 8


def test_api_ribbon(client):
    j = client.get(f"/api/recordings/{Player G}/ribbon?filter=bp_0530&width=300").get_json()
    assert len(j["t"]) <= 300 and j["filter"] == "bp_0530" and j["ceil_uv"] > 0


def test_api_channels_default_and_8(client):
    two = client.get(f"/api/recordings/{Player G}/channels?width=200").get_json()
    assert [c["label"] for c in two["channels"]] == ["L", "R"]
    rows = client.get(f"/api/recordings/{Player G}/eeg_rows").get_json()
    eight = client.get(f"/api/recordings/{Player G}/channels?rows={','.join(map(str, rows))}").get_json()
    assert len(eight["channels"]) == 8


def test_api_trim_roundtrip(client):
    rid = "20260622-144957-playerG"
    assert client.put(f"/api/recordings/{rid}/trim", json={"t0": 2, "t1": 8}).status_code == 200
    assert client.get(f"/api/recordings/{rid}/trim").get_json()["t0"] == 2.0
    assert client.delete(f"/api/recordings/{rid}/trim").status_code == 200
    assert client.get(f"/api/recordings/{rid}/trim").get_json() is None


def test_api_trim_requires_fields(client):
    assert client.put(f"/api/recordings/{Player G}/trim", json={"t0": 1}).status_code == 400


def test_api_404s(client):
    assert client.get("/api/recordings/nope").status_code == 404
    assert client.get("/api/recordings/nope/ribbon").status_code == 404
    assert client.get("/api/recordings/nope/channels").status_code == 404
