import cv2
import numpy as np
from typing import Tuple, Optional

def resize_with_aspect_ratio(image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None, inter: int = cv2.INTER_AREA) -> np.ndarray:
    """
    Resizes an image while maintained the aspect ratio.
    """
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Safely converts a BGR image to Grayscale.
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
