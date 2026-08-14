"""Robust Streamlit bootstrap, including old Conda/Windows SSL workarounds."""

from pathlib import Path
import ssl
import sys

try:
    import certifi
except ImportError:
    certifi = None

if certifi is not None:
    _original = ssl.create_default_context

    def _context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
        # Supplying cafile prevents Python 3.9 from loading malformed entries
        # from the Windows certificate store before Streamlit starts.
        return _original(
            purpose=purpose, cafile=cafile or certifi.where(),
            capath=capath, cadata=cadata,
        )

    ssl.create_default_context = _context

from streamlit.web.cli import main

app = Path(__file__).resolve().parent / "app.py"
sys.argv = ["streamlit", "run", str(app)]
raise SystemExit(main())
