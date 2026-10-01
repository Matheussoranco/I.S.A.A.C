import mss
from pathlib import Path
from isaac.multimodal.vision.vlm import VisionManager
from isaac.config.settings import settings

async def capture_and_analyze(prompt: str) -> str:
    """Captures clipboard image (via screenshot of current view as proxy or actual clipboard) 
    and analyzes it."""
    # Note: Truly capturing 'clipboard image' on Windows often requires Win32 API.
    # Using mss for a full-screen capture as the primary 'vision' source for this Phase.
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.shot(output=f"{settings.CACHE_DIR}/clipboard_capture.jpg")
        image_path = Path(screenshot)

    manager = VisionManager()
    return await manager.analyze(image_path, prompt)
