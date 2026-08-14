#!/usr/bin/env python3
"""Copy the built essay figures into the blog repo.

The figures are DEVELOPED here — this is where the bake lives, where the harness
runs, and where the code review happens — and SERVED from the blog. Two repos,
one direction, no symlinks: the blog gets a plain copy it can commit, so the post
keeps rendering whether or not this repo is checked out beside it.

Run it after every `python scripts/bake_essay_figures.py`, or after touching
figures.js / figures.css. Nothing else in the blog is read or written.

CACHE BUSTING. `figures.js` carries its own DATA_V and appends it to the two data
fetches, so a re-bake invalidates the data by itself. What it CANNOT invalidate is
figures.js, which the page requests by URL — so the host page's <script src>
needs a ?v= of its own, bumped whenever the script changes. This prints the
current DATA_V as a reminder of which number to match.

Usage:
    python scripts/sync_essay_figures.py [DEST]

DEST defaults to the blog's assets directory.
"""

import pathlib
import re
import shutil
import sys

SRC = pathlib.Path("web/essay-figures")
DEFAULT_DEST = pathlib.Path.home() / "Dev/hoeksemaa.github.io/assets/essay-figures"

# index.html is deliberately NOT copied: it is the harness, it carries page-level
# styling the blog supplies for itself, and the post mounts its own divs.
FILES = ["figures.js", "figures.css", "data/eog-figures.json", "data/eog-full.bin"]


def main():
    dest = pathlib.Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_DEST

    # The PARENT has to exist. Creating the whole tree would happily absorb a typo
    # and write the figures somewhere nobody looks.
    if not dest.parent.is_dir():
        sys.exit(f"no such directory: {dest.parent}\n"
                 f"pass the destination explicitly if the blog lives elsewhere")

    total = 0
    for rel in FILES:
        src = SRC / rel
        if not src.is_file():
            sys.exit(f"missing: {src} — run scripts/bake_essay_figures.py first")
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        kb = out.stat().st_size / 1024
        total += kb
        print(f"  {rel:28s} {kb:8.1f} KB")

    data_v = re.search(r"const DATA_V = (\d+)", (SRC / "figures.js").read_text())
    print(f"synced {len(FILES)} files ({total:.0f} KB) -> {dest}")
    print(f"  DATA_V = {data_v.group(1) if data_v else '?'} — match the host page's "
          f"<script src=\"...figures.js?v=N\">")
    print(f"  host page must also set window.EOG_FIG_BASE to the URL path of {dest.name}")


if __name__ == "__main__":
    main()
