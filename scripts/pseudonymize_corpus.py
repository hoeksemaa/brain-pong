#!/usr/bin/env python3
"""Rename every recording in ``data/eog/`` to its pseudonym, in place.

The recorder takes the player's name as free text, so a fresh recording lands as
``20260817-183500-<name>.npz`` with ``subject_id="<name>"`` inside it. The corpus
is committed to a PUBLIC repository, where a filename is the one field that needs no
tooling to read: a directory listing, or a code-search hit, is the whole name list.
Every other public surface (the portal, the essay figures) already speaks in
``Player <letter>``, and this is what makes the files agree with them.

    <stamp>-<name>.npz  ->  <stamp>-player<letter>.npz     (subject_id rewritten too)

Letters come from MAP_PATH, which is deliberately **gitignored**: it is the only
file that holds the real names, it stays on the machine that recorded them, and it
is what makes the mapping reversible for whoever ran the sessions. New names are
appended, never reassigned — a letter, once published, is that person's letter for
good, and a letter whose subject leaves the corpus stays spent so the next person
does not inherit their published URLs.

``P1``/``P2`` are left alone. They are station slots, not people: no name was ever
typed at that rig, so there is nothing to pseudonymise and no identity to protect.

Idempotent: a corpus that is already pseudonymous is left untouched. Run it after
any recording session, before committing.

Usage
    python scripts/pseudonymize_corpus.py --dry-run    # print the plan, touch nothing
    python scripts/pseudonymize_corpus.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "eog"
MAP_PATH = REPO / "data" / "portal-anon-map.json"
MAP_NOTE = ("PRIVATE, gitignored — the only file that maps a real name to its "
            "published letter. Never commit it, never copy it under web/. Letters "
            "are append-only: never reassign one, never reuse one whose subject has "
            "left the corpus.")

# Station slots the recorder falls back to when no name is typed. Not people.
SLOT_SUBJECTS = ("p1", "p2")

STEM_RE = re.compile(r"^(\d{8}-\d{6})-(.+)$")
PSEUDONYM_RE = re.compile(r"^player([A-Z]\d*)$")


def canon_subject(subj):
    """One person, one key — a name typed twice at the rig differed only in case, and
    two casings of one person would otherwise draw two letters."""
    return subj.strip().lower()


def _letter(i):
    return chr(ord("A") + i) if i < 26 else f"A{i}"


def _letter_index(letter):
    return int(letter[1:]) if len(letter) > 1 else ord(letter) - ord("A")


def load_map():
    if not MAP_PATH.exists():
        return {}
    return json.loads(MAP_PATH.read_text()).get("assigned", {})


def save_map(assigned):
    MAP_PATH.write_text(json.dumps({"note": MAP_NOTE, "assigned": assigned},
                                   indent=2, sort_keys=True) + "\n")


def plan(paths, assigned):
    """[(path, new_stem, letter)] for every file that still carries a name."""
    # Oldest first, so a name never seen before draws the next letter in the order it
    # first appeared — the same rule the letters already published were drawn under.
    out, nxt = [], max((_letter_index(v) for v in assigned.values()), default=-1) + 1
    for p in sorted(paths):
        m = STEM_RE.match(p.stem)
        if not m:
            sys.exit(f"unparseable stem (expected <YYYYmmdd-HHMMSS>-<subject>): {p.name}")
        stamp, subj = m.groups()
        if PSEUDONYM_RE.match(subj) or canon_subject(subj) in SLOT_SUBJECTS:
            continue                                  # already anonymous
        canon = canon_subject(subj)
        if canon not in assigned:
            assigned[canon] = _letter(nxt)
            nxt += 1
        out.append((p, f"{stamp}-player{assigned[canon]}", assigned[canon]))
    return out


def rewrite(path, new_stem):
    """Rename the file and rewrite subject_id, keeping every other field verbatim."""
    target = path.with_name(new_stem + ".npz")
    if target.exists():
        sys.exit(f"refusing to overwrite {target.name} (from {path.name})")
    with np.load(path, allow_pickle=True) as d:
        fields = {k: d[k] for k in d.files}
    fields["subject_id"] = np.array([new_stem.split("-", 2)[2]])
    np.savez(target, **fields)
    path.unlink()
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    assigned = load_map()
    todo = plan(sorted(DATA_DIR.glob("*.npz")), assigned)
    if not todo:
        print("corpus is already pseudonymous — nothing to do")
        return
    for p, new_stem, letter in todo:
        print(f"{p.name} -> {new_stem}.npz")
    if args.dry_run:
        print(f"\n{len(todo)} file(s) would be renamed (dry run)")
        return
    for p, new_stem, _ in todo:
        rewrite(p, new_stem)
    save_map(assigned)
    print(f"\nrenamed {len(todo)} file(s); map updated at {MAP_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
