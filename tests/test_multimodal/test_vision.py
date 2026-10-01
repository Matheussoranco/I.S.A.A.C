
import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from isaac.multimodal.vision.vlm import VisionManager
from isaac.multimodal.vision.clipboard import capture_and_analyze

@pytest.mark.asyncio
async def test_vision_manager_fallback():
    """Test that VisionManager falls back to cloud when local fails."""
    manager = VisionManager()
    image_path = Path("tests/test_multimodal/sample.jpg")
    
    # Mock local failure, cloud success
    with patch.object(manager, "_analyze_local", side_effect=Exception("Local VLM Down")), \
         patch.object(manager, "_analyze_cloud", new_callable=AsyncMock) as mock_cloud:
        
        mock_cloud.return_value = "Cloud analysis result"
        result = await manager.analyze(image_path, "What is this?")
        
        assert result == "Cloud analysis result"
        mock_cloud.assert_called_once()

@pytest.mark.asyncio
async def test_clipboard_capture_and_analyze():
    """Test the clipboard capture and analysis flow."""
    prompt = "Analyze this screenshot"
    
    with patch("mss.mss") as mock_mss, \
         patch("isaac.multimodal.vision.vlm.VisionManager.analyze", new_callable=AsyncMock) as mock_analyze:
        
        # Mock mss behavior
        mock_sct = mock_mss.return_value.__enter__.return_value
        mock_sct.monitors = [None, {"width": 1920, "height": 1080}]
        mock_sct.shot.return_value = "C:/Users/mathe/AppData/Local/hermes/cache/scratch/clipboard_capture.jpg"
        
        mock_analyze.return_value = "Screenshot analysis result"
        
        result = await capture_and_analyze(prompt)
        
        assert result == "Screenshot analysis result"
        mock_analyze.assert_called_once()
