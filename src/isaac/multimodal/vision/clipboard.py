try:
    import mss
except ImportError:  # pragma: no cover - optional vision extra
    mss = None  # type: ignore[assignment]
from pathlib import Path

from isaac.config.settings import settings
from isaac.multimodal.vision.vlm import VisionManager


async def capture_and_analyze(prompt: str) -> str:
    """Captures clipboard image (via screenshot of current view as proxy or actual clipboard)
    and analyzes it."""
    # Note: Truly capturing 'clipboard image' on Windows often requires Win32 API.
    # Using mss for a full-screen capture as the primary 'vision' source for this Phase.
    if mss is None:
        raise RuntimeError("Screen capture requires the 'vision' extra (mss).")
    capture_dir = settings.isaac_home / "cache"
    capture_dir.mkdir(parents=True, exist_ok=True)
    with mss.mss() as sct:
        screenshot = sct.shot(output=str(capture_dir / "clipboard_capture.jpg"))
        image_path = Path(screenshot)

    manager = VisionManager()
    return await manager.analyze(image_path, prompt)
