"""
Pytest bootstrap for the BrainPong test suite.

Keeps the suite fully self-contained under tests/: the only thing it needs from
the software side is for the repo root to be importable so `import eog_core`
resolves. We add it to sys.path here rather than adding a pyproject/pytest.ini
to the repo root, so nothing outside tests/ changes.
"""
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
