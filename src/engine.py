import cv2
import numpy as np
from typing import Tuple, Dict

def count_grains_opencv(img: np.ndarray, conf_threshold: float) -> Tuple[np.ndarray, int]:
    """High-precision grain counting using Watershed algorithm."""
    # Preprocessing
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to handle uneven lighting
    thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Sensitivity adjusted by slider
    _, sure_fg = cv2.threshold(dist_transform, (1.0 - conf_threshold) * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply Watershed
    markers = cv2.watershed(img, markers)
    
    # Draw results
    result_image = img.copy()
    count = 0
    for label in np.unique(markers):
        if label <= 1: continue # background/unknown
        
        # Create a mask for each label
        mask = np.zeros(gray_image.shape, dtype="uint8")
        mask[markers == label] = 255
        
        # Find contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > 20: # filter very small noise
                cv2.drawContours(result_image, [c], -1, (0, 255, 0), 2)
                # Bounding box
                x, y, w, h = cv2.boundingRect(c)
                cv2.putText(result_image, str(count+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                count += 1
                
    return result_image, count
