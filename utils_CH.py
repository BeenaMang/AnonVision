# Minimal utils_CH for AnonVision (intermediate style)
# - Accepts PIL.Image or numpy image
# - Returns a PIL.Image sized to fit inside max_width x max_height

from typing import Tuple, Union
import os

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

from PIL import Image

def validate_image_path(path: str) -> bool:
    if not path or not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def _to_pil(image) -> Image.Image:
    """Best-effort convert to PIL.Image without assuming color order."""
    if isinstance(image, Image.Image):
        return image
    if np is not None and isinstance(image, np.ndarray):
        # Assume BGR if looks like OpenCV
        if image.ndim == 3 and image.shape[2] == 3 and cv2 is not None:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_rgb)
        return Image.fromarray(image)
    raise TypeError("Unsupported image type for resize_image_for_display")

def resize_image_for_display(image_in: Union[Image.Image, 'np.ndarray'], max_width: int = 800, max_height: int = 600) -> Image.Image:
    """Aspect-fit resize. Input may be PIL or numpy. Returns PIL.Image."""
    img = _to_pil(image_in)
    w, h = img.size
    if w == 0 or h == 0:
        return img
    scale = min(max_width / float(w), max_height / float(h))
    if scale >= 1.0:
        # Already fits; just return a copy so callers can safely hold it
        return img.copy()
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)
