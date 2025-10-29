# utils.py (merged, MediaPipe-free)
from __future__ import annotations
from typing import Tuple, Optional, Union, Iterable, List, Dict
from pathlib import Path
from PIL import Image
import numpy as np
import os

# ---------- image helpers ----------

def _to_pil(image_in: Union[Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(image_in, Image.Image):
        return image_in
    if isinstance(image_in, np.ndarray):
        arr = image_in
        if arr.ndim == 2:
            mode = "L"
            return Image.fromarray(arr, mode=mode)
        if arr.ndim == 3:
            if arr.shape[2] == 3:
                # assume BGR -> convert to RGB
                arr_rgb = arr[:, :, ::-1].copy()
                return Image.fromarray(arr_rgb, mode="RGB")
            if arr.shape[2] == 4:
                # BGRA -> RGBA
                arr_rgba = arr[:, :, [2,1,0,3]].copy()
                return Image.fromarray(arr_rgba, mode="RGBA")
    raise TypeError("Unsupported image type for _to_pil")

def _to_numpy_rgb(image_in: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Return RGB numpy array (H,W,3)."""
    if isinstance(image_in, np.ndarray):
        # If BGR, convert; heuristics are tricky; rely on caller when possible.
        return image_in
    if isinstance(image_in, Image.Image):
        return np.array(image_in.convert("RGB"))
    raise TypeError("Unsupported image type for _to_numpy_rgb")

def resize_image_for_display(image_in: Union[Image.Image, np.ndarray], **kwargs) -> Image.Image:
    """Aspect-fit resize. Input may be PIL or numpy. Returns PIL.Image.
    Accepts keyword args:
      - max_width / max height (underscored) OR "max width"/"max height" (with spaces) for compatibility
    Defaults: 800x600
    """
    def _get_kw(name, default):
        if name in kwargs:
            return kwargs[name]
        spaced = name.replace('_', ' ')
        return kwargs.get(spaced, default)

    max_width = int(_get_kw('max_width', 800))
    max_height = int(_get_kw('max_height', 600))

    img = _to_pil(image_in)
    w, h = img.size
    if w == 0 or h == 0:
        return img.copy()
    scale = min(max_width / float(w), max_height / float(h))
    if scale >= 1.0:
        return img.copy()
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)

def validate_image_path(path: Union[str, Path]) -> bool:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

# ---------- boxes ----------

def clamp_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> Tuple[int,int,int,int]:
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h
