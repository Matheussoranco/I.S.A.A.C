"""Cross-platform screen capture helpers.

Uses ``mss`` for fast multi-monitor capture, ``Pillow`` as fallback.
Returned data is a base-64 PNG suitable for VLM ingestion.
"""

from __future__ import annotations

import base64
import io
import logging

logger = logging.getLogger(__name__)


def is_capture_available() -> bool:
    try:
        import mss  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        from PIL import ImageGrab  # noqa: F401
        return True
    except ImportError:
        return False


def capture_screen_b64(monitor: int = 1) -> str:
    """Capture a single monitor and return a base64-encoded PNG."""
    raw = capture_screen_png(monitor=monitor)
    return base64.b64encode(raw).decode("ascii")


def capture_screen_png(monitor: int = 1) -> bytes:
    """Capture a single monitor and return raw PNG bytes."""
    try:
        import mss
        from PIL import Image

        with mss.mss() as sct:
            mon = sct.monitors[monitor]
            shot = sct.grab(mon)
            img = Image.frombytes("RGB", shot.size, shot.rgb)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    except ImportError:
        pass

    from PIL import ImageGrab

    img = ImageGrab.grab(all_screens=True)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
