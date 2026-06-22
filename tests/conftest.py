"""
Pytest bootstrap for the BrainPong test suite.

Makes the `brainpong` package importable from `src/` without requiring a
`pip install -e .` first, so the suite runs standalone. (The editable install
also works; this is a zero-setup fallback.)
"""
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
