"""
End-to-end browser tests for the EOG viewer frontend (web/).

Drives a real Chromium (via Playwright) against a live Flask server backed by a
throwaway DB ingested from the real corpus. Covers what the store/API suite
(test_viewer.py) cannot: actual rendering, filter + channel switching, the trim
keep-window drag, live rail-% reactivity, trim persistence through the real API,
sidebar formatting/alignment, and the structured frontend log (window.__eoglog).

Auto-skips where Playwright or its Chromium build is missing, so a bare
`pytest tests/` still passes. To enable:

    pip install pytest-playwright playwright
    playwright install chromium
"""
import re
import sys
import time
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))   # serve_viewer lives in scripts/

from brainpong import store          # noqa: E402
import serve_viewer                  # noqa: E402

sync_api = pytest.importorskip("playwright.sync_api")   # skip module if absent

NPZ_DIR = REPO / "data" / "eog"
Player F = "20260519-145529-playerF"        # railing tail; clean first ~180s
PADL, PADR = 48, 8                     # canvas plot insets (mirror viewer.js)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("e2e") / "viewer.db"
    store.ingest_dir(str(path), str(NPZ_DIR))
    return str(path)


@pytest.fixture(scope="module")
def server(db):
    from werkzeug.serving import make_server
    app = serve_viewer.create_app(db)
    srv = make_server("127.0.0.1", 0, app)          # port 0 → ephemeral
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    thread.join(timeout=3)


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:                       # browser binary not installed
            pytest.skip(f"chromium not available ({e}); run: playwright install chromium")
        yield b
        b.close()


@pytest.fixture
def page(browser):
    pg = browser.new_page()
    errors = []
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.console_errors = errors
    yield pg
    pg.close()


def open_app(page, server, query=""):
    page.goto(f"{server}/{query}")
    page.wait_for_function("window.__ready === true", timeout=8000)


def time_to_x(box, t, dur):
    return box["x"] + PADL + (t / dur) * (box["width"] - PADL - PADR)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_loads_and_renders(page, server):
    open_app(page, server)
    assert len(page.query_selector_all(".recitem")) == 8
    assert page.query_selector(".zone-derived") is not None
    assert page.query_selector(".zone-raw") is not None
    assert page.query_selector(".railpct") is not None
    assert page.console_errors == []                 # incl. no favicon 404


def test_sidebar_formatting_and_alignment(page, server):
    open_app(page, server)
    dates = page.eval_on_selector_all(".rdate", "els => els.map(e => e.textContent)")
    lens = page.eval_on_selector_all(".rlen", "els => els.map(e => e.textContent)")
    assert dates and all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) for d in dates)   # YYYY-MM-DD
    assert lens and all(re.fullmatch(r"\d+s", l) for l in lens)                  # rounded seconds
    assert page.query_selector(".dot") is None                                  # status dot gone
    assert "✂" not in page.content()                                       # no scissors
    # columns aligned: every name/date/length shares one left edge
    for sel in (".rsub", ".rdate", ".rlen"):
        xs = page.eval_on_selector_all(
            sel, "els => els.map(e => Math.round(e.getBoundingClientRect().left))")
        assert len(set(xs)) == 1, f"{sel} left edges not aligned: {set(xs)}"


def test_filter_switch(page, server):
    open_app(page, server)
    page.click("button.fbtn:has-text('0.5–30 Hz')")
    page.wait_for_function(
        "() => document.querySelector('.fbtn.active') "
        "&& document.querySelector('.fbtn.active').textContent.includes('0.5')")
    msgs = page.evaluate("() => window.__eoglog.buffer().map(r => r.msg)")
    assert any("filter change" in m for m in msgs)
    assert page.console_errors == []


def test_channel_expand(page, server):
    open_app(page, server)
    assert len(page.query_selector_all(".zone-raw .chrow")) == 2      # L / R default
    page.click(".chtoggle")
    page.wait_for_function(
        "() => document.querySelectorAll('.zone-raw .chrow').length === 8")


def test_trim_drag_is_rail_reactive_and_persists(page, server, db):
    store.clear_trim(db, Player F)                       # deterministic start
    open_app(page, server, f"?rec={Player F}")
    dur = store.get_recording(db, Player F)["duration"]

    # full recording → badge shows the railing tail's contribution (>5%)
    page.wait_for_function(
        "() => /RAIL \\d/.test(document.querySelector('.railpct').textContent)")
    full = page.text_content(".railpct")
    assert float(re.search(r"([\d.]+)%", full).group(1)) > 5

    # drag a keep-window over the CLEAN first region [10, 150]s
    box = page.query_selector(".zone-derived canvas").bounding_box()
    y = box["y"] + 30
    page.mouse.move(time_to_x(box, 10, dur), y)
    page.mouse.down()
    for t in (60, 110, 150):
        page.mouse.move(time_to_x(box, t, dur), y)
    page.mouse.up()

    # rail badge reacts to the kept window → clean → 0.0%
    page.wait_for_function(
        "() => document.querySelector('.railpct').textContent === 'RAIL 0.0%'",
        timeout=4000)

    # trim persisted to the DB through the real API (npz untouched)
    trim = None
    for _ in range(20):
        trim = store.get_trim(db, Player F)
        if trim:
            break
        time.sleep(0.05)
    assert trim is not None
    assert trim["t0"] == pytest.approx(10, abs=4) and trim["t1"] == pytest.approx(150, abs=4)

    # structured log recorded the commit
    msgs = page.evaluate("() => window.__eoglog.buffer().map(r => r.msg)")
    assert any(m.startswith("trim commit") for m in msgs)
    assert page.console_errors == []


def test_structured_log_lifecycle(page, server):
    open_app(page, server, f"?rec={Player F}")
    buf = page.evaluate("() => window.__eoglog.buffer()")
    msgs = [r["msg"] for r in buf]
    assert any("init" in m for m in msgs)
    assert any("recordings loaded" in m for m in msgs)
    assert any(m.startswith("select recording") for m in msgs)
    assert any(m == "ready" for m in msgs)
    # every API request is traced with a latency, e.g. "GET /api/... 8ms"
    assert any(re.search(r"GET /api/.+ \d+ms", m) for m in msgs)
    # levels present and well-formed
    assert {r["level"] for r in buf} <= {"debug", "info", "warn", "error"}
