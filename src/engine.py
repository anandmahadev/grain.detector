import cv2
import numpy as np
import os
import streamlit as st
import logging
from typing import Tuple, Dict, List, Optional, NamedTuple
from ultralytics import YOLO
from src.utils import preprocess_image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DATA MODELS ---
class DetectionResult(NamedTuple):
    """Encapsulates detection output for system-wide consistency."""
    annotated_image: np.ndarray
    counts: Dict[str, int]
    total_count: int
    metadata: Dict[str, any]

# --- APP CONFIGURATION ---
APP_CONFIG = {
    "title": "🌾 AI Grain Counter",
    "layout": "wide",
    "icon": "🌾",
    "base_model": "yolov8n.pt",
    "custom_model": "custom_rice_pepper_model.pt",
    "version": "1.2.0",
    "developer": "Anand Mahadev",
    "min_grain_area": 25,
    "colors": {
        "primary": (0, 255, 0),    # Green for detection
        "secondary": (255, 0, 0),  # Blue for markers
        "text": (255, 255, 255)
    }
}

@st.cache_resource(show_spinner="⏳ Initializing AI Engine...")
def load_model() -> YOLO:
    """
    Loads the YOLOv8 model with optimized performance configuration.
    
    The function prioritizes the custom trained model (`custom_rice_pepper_model.pt`) 
    if it exists in the root directory. Otherwise, it falls back to the 
    standard `yolov8n.pt` base model.
    
    Returns:
        YOLO: An initialized Ultralytics YOLO model object.
    """
    try:
        model_path = APP_CONFIG["custom_model"] if os.path.exists(APP_CONFIG["custom_model"]) else APP_CONFIG["base_model"]
        logger.info(f"Loading YOLO model from {model_path}")
        model = YOLO(model_path)
        
        # Standardize classes for Rice and Pepper specifically for this app context.
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            if os.path.exists(APP_CONFIG["custom_model"]):
                model.model.names = {0: 'Rice', 1: 'Pepper'}
            else:
                # Demonstration mappings for base model
                mock_names = ['Rice', 'Wheat', 'Seed', 'Pepper', 'Grain']
                model.model.names = {i: mock_names[i % len(mock_names)] for i in range(100)}
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise RuntimeError(f"Engine initialization failed: {e}")


def count_grains_opencv(img: np.ndarray, sensitivity: float) -> DetectionResult:
    """
    High-precision grain counting using Watershed algorithm with adaptive thresholding.
    
    Args:
        img: Input BGR image array.
        sensitivity: Normalized sensitivity factor [0.0 - 1.0].
    """
    # Defensive check
    if img is None or img.size == 0:
        return DetectionResult(img, {}, 0, {"error": "Empty image"})

    # Preprocessing: Grayscale and Dilation to connect grain parts if needed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to handle uneven lighting in agricultural settings
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 5
    )
    
    # Morphological operations (Noise removal)
    # Using a 3x3 kernel for standard grain sizes.
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area (Distance Transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, (1.0 - sensitivity) * dist_transform.max(), 255, 0)
    
    # Marker labelling
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply Watershed
    markers = cv2.watershed(img, markers)
    
    # Processing output results
    result_image = img.copy()
    count = 0
    for label in np.unique(markers):
        if label <= 1: continue # skip background/unknown
        
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == label] = 255
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > APP_CONFIG["min_grain_area"]:
                cv2.drawContours(result_image, [c], -1, APP_CONFIG["colors"]["primary"], 2)
                x, y, w, h = cv2.boundingRect(c)
                cv2.putText(
                    result_image, str(count+1), (x, y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, APP_CONFIG["colors"]["secondary"], 1
                )
                count += 1
                
    return DetectionResult(
        annotated_image=result_image,
        counts={"Grains": count},
        total_count=count,
        metadata={"algorithm": "Watershed"}
    )
    # Note: Using adaptiveThreshold instead of global thresholding improves 
    # robustness against non-uniform field lighting conditions.
