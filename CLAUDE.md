The user will remember everything important and the rest can be determined from context and reading the files.
Writing things to memory or notes for yourself is unnecessary.

## real names

Some files in this repo hold real first names: the git history (233 old file
paths, 16 commit messages), a small number of places in the current tree, and 19
PR descriptions on GitHub. This is not good for privacy, but it is acceptable for
a toy project. Do not clean it up.

Do not add more. Use the pseudonyms — `Player <letter>`, or `Unattributed` for
the P1/P2 station slots — in filenames, code, comments, docs and commit
messages. `data/eog/` is pseudonymous on disk.

Run `python scripts/pseudonymize_corpus.py` before every PR, and again before you
merge it — also after a recording session, before you commit. The script is
idempotent: when there is nothing to do it prints `corpus is already
pseudonymous` and changes no file. `data/portal-anon-map.json` maps a letter to a
person, and is gitignored: keep it that way.
