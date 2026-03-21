import pytest
import numpy as np
from src.engine import count_grains_opencv

def test_opencv_engine_initialization():
    """Verify that the engine handles empty/mock inputs correctly."""
    mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res_img, count = count_grains_opencv(mock_img, 0.5)
    
    assert count == 0
    assert res_img.shape == mock_img.shape

def test_det_range():
    """Verify threshold range processing."""
    mock_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    for t in [0.1, 0.5, 0.9]:
        _, count = count_grains_opencv(mock_img, t)
        assert isinstance(count, int)
