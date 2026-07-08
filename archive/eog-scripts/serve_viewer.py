"""
Flask API + static host for the EOG diagnostic web viewer.

Reads the SQLite store (built by scripts/ingest_npz.py) and serves:

  GET  /api/recordings                      list (+ health, spark, trim)
  GET  /api/recordings/<id>                 metadata + events + trim
  GET  /api/recordings/<id>/ribbon          decimated L−R ribbon
         ?filter=raw|bp_0530|bp_0130|velocity &t0=&t1=&width=
  GET  /api/recordings/<id>/channels        decimated raw channels
         ?rows=1,2 &t0=&t1=&width=
  GET  /api/recordings/<id>/eeg_rows        board rows carrying the 8 ADS1299 chs
  GET/PUT/DELETE /api/recordings/<id>/trim  keep-window annotation (never touches npz)
  GET  /  + static                          the web/ frontend

Signal is read live from the frozen npz and min/max-decimated to the requested
pixel width. Filters are zero-phase (brainpong.preprocess.VIEWER_FILTERS).

    source .venv/bin/activate
    python scripts/ingest_npz.py          # once / after new recordings
    python scripts/serve_viewer.py        # http://localhost:8770
"""
import argparse
import logging
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, abort

from brainpong import store

REPO = Path(__file__).resolve().parent.parent
WEB = REPO / "web"
DEFAULT_DB = REPO / "derivatives" / "viewer.db"

log = logging.getLogger("brainpong.serve")


def create_app(db_path=DEFAULT_DB):
    app = Flask(__name__, static_folder=None)
    app.config["DB"] = str(db_path)

    @app.after_request
    def _nocache(resp):
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

    def db():
        return app.config["DB"]

    def _farg(name):
        v = request.args.get(name)
        return float(v) if v not in (None, "") else None

    def _width():
        try:
            return max(16, min(6000, int(request.args.get("width", 1500))))
        except (TypeError, ValueError):
            return 1500

    # ── API ───────────────────────────────────────────────────────────────────
    @app.get("/api/recordings")
    def api_list():
        recs = store.list_recordings(db())
        log.info("GET /api/recordings -> %d", len(recs))
        return jsonify(recs)

    @app.get("/api/recordings/<rid>")
    def api_get(rid):
        rec = store.get_recording(db(), rid)
        if rec is None:
            abort(404)
        return jsonify(rec)

    @app.get("/api/recordings/<rid>/ribbon")
    def api_ribbon(rid):
        out = store.ribbon(db(), rid, filt=request.args.get("filter", "raw"),
                           t0=_farg("t0"), t1=_farg("t1"), width=_width())
        if out is None:
            abort(404)
        log.info("GET ribbon %s filter=%s -> %d pts", rid, out["filter"], len(out["t"]))
        return jsonify(out)

    @app.get("/api/recordings/<rid>/channels")
    def api_channels(rid):
        rows_arg = request.args.get("rows")
        rows = [int(x) for x in rows_arg.split(",") if x != ""] if rows_arg else None
        out = store.channels(db(), rid, rows=rows,
                             t0=_farg("t0"), t1=_farg("t1"), width=_width())
        if out is None:
            abort(404)
        log.info("GET channels %s rows=%s -> %d chs", rid, rows, len(out["channels"]))
        return jsonify(out)

    @app.get("/api/recordings/<rid>/eeg_rows")
    def api_eeg_rows(rid):
        return jsonify(store.eeg_rows(db(), rid))

    @app.get("/api/recordings/<rid>/health")
    def api_health(rid):
        out = store.window_health(db(), rid, t0=_farg("t0"), t1=_farg("t1"))
        if out is None:
            abort(404)
        return jsonify(out)

    @app.route("/api/recordings/<rid>/trim", methods=["GET", "PUT", "DELETE"])
    def api_trim(rid):
        if request.method == "GET":
            return jsonify(store.get_trim(db(), rid))
        if request.method == "DELETE":
            store.clear_trim(db(), rid)
            return jsonify({"trim": None})
        body = request.get_json(force=True, silent=True) or {}
        if "t0" not in body or "t1" not in body:
            abort(400, "t0 and t1 required")
        return jsonify(store.set_trim(db(), rid, body["t0"], body["t1"], body.get("reason")))

    # ── frontend ───────────────────────────────────────────────────────────────
    @app.get("/favicon.ico")
    def favicon():
        return ("", 204)

    @app.get("/")
    def index():
        return send_from_directory(WEB, "index.html")

    @app.get("/<path:p>")
    def static_files(p):
        return send_from_directory(WEB, p)

    return app


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    if not Path(args.db).exists():
        log.warning("DB %s not found — run scripts/ingest_npz.py first", args.db)
    app = create_app(args.db)
    log.info("serving viewer on http://localhost:%d  (db=%s)", args.port, args.db)
    app.run(host="127.0.0.1", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
