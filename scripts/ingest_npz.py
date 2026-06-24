"""
Build/refresh the viewer's SQLite DB from the frozen data/eog/*.npz corpus.

The DB (derivatives/viewer.db) is a regenerable derivative — metadata, events,
and trim annotations only. The npz stay the single source of truth for signal
and are never modified. Safe to re-run anytime; existing trims are preserved
(only the recording/event rows are rebuilt).

    source .venv/bin/activate
    python scripts/ingest_npz.py
    python scripts/ingest_npz.py --db /tmp/test.db --npz-dir data/eog
"""
import argparse
import logging
from pathlib import Path

from brainpong import store

REPO = Path(__file__).resolve().parent.parent


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(REPO / "derivatives" / "viewer.db"))
    ap.add_argument("--npz-dir", default=str(REPO / "data" / "eog"))
    args = ap.parse_args()

    n = store.ingest_dir(args.db, args.npz_dir)
    print(f"\ningested {n} recordings → {args.db}")
    for r in store.list_recordings(args.db):
        trim = f"  trim={r['trim']}" if r["trim"] else ""
        print(f"  {r['id']:28s} {r['status']:8s} rail={r['rail_pct']:>5.1f}%  "
              f"peak={r['ribbon_peak']:>9.0f}uV{trim}")


if __name__ == "__main__":
    main()
