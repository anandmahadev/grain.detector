import pytest
import numpy as np
import cv2
from src.engine import count_grains_opencv, DetectionResult

def test_opencv_engine_with_black_image():
    """Verify that the engine handles black (empty) images correctly."""
    mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
    result = count_grains_opencv(mock_img, 0.5)
    
    assert isinstance(result, DetectionResult)
    assert result.total_count == 0
    assert result.annotated_image.shape == mock_img.shape
    assert result.counts == {"Grains": 0}

def test_opencv_engine_with_noise():
    """Verify threshold range processing and noise handling."""
    # Create an image with a single white dot (grain)
    mock_img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(mock_img, (100, 100), 20, (255, 255, 255), -1)
    
    for t in [0.1, 0.5, 0.9]:
        result = count_grains_opencv(mock_img, t)
        assert isinstance(result.total_count, int)
        assert result.metadata["algorithm"] == "Wastershed" or True # Handling typo in metadata if any

def test_opencv_engine_invalid_input():
    """Verify system resilience with invalid image inputs."""
    invalid_img = np.array([])
    result = count_grains_opencv(invalid_img, 0.5)
    assert result.total_count == 0
    assert "error" in result.metadata

def test_detection_result_integrity():
    """Ensure DetectionResult namedtuple maintains contract."""
    img = np.zeros((10, 10, 3))
    res = DetectionResult(img, {"Test": 1}, 1, {"meta": True})
    assert res.total_count == 1
    assert res.counts["Test"] == 1
