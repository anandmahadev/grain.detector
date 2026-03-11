import pytest
import numpy as np
import os
from grain_counter import APP_CONFIG, GRAIN_TYPES, process_frame

def test_app_config():
    """Verify application configuration mapping constants."""
    assert APP_CONFIG["title"] == "🌾 AI Grain Counter"
    assert APP_CONFIG["base_model"] == "yolov8n.pt"

def test_grain_types():
    """Ensure fundamental classes exist."""
    assert isinstance(GRAIN_TYPES, list)
    assert len(GRAIN_TYPES) > 0

def test_process_frame_output_structure():
    """Mocks an empty frame and ensures processing outputs maintain their strict types."""
    # Create empty mock frame (320x320 RGB)
    mock_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    
    ann_img, counts, latency = process_frame(mock_frame)
    
    # Asserting outputs
    assert isinstance(ann_img, np.ndarray)
    assert isinstance(counts, dict)
    assert isinstance(latency, float)
    
    # Assert counts mapping holds true
    for key in counts.keys():
        assert key in GRAIN_TYPES
