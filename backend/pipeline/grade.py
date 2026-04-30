"""
Cinematic color grading: vignette + contrast S-curve + warm tones.
Applied per-frame after warping/inpainting, before FFmpeg encoding.
"""

import cv2
import numpy as np


def _build_vignette(h: int, w: int, strength: float = 0.52) -> np.ndarray:
    """Return a float32 (H, W) vignette mask, 1.0 at centre, dark at edges."""
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    mask = np.clip(1.0 - strength * dist ** 1.6, 0.0, 1.0)
    return mask.astype(np.float32)


# Cache the vignette so it is only computed once per resolution.
_vignette_cache: dict = {}


def apply_grade(frame: np.ndarray) -> np.ndarray:
    """
    Apply cinematic grade to a single BGR uint8 frame.

    Steps:
      1. Contrast S-curve (slight lift of shadows, pull of highlights)
      2. Warm colour tint  (boost red, reduce blue)
      3. Vignette          (darken edges)
    """
    h, w = frame.shape[:2]
    key = (h, w)
    if key not in _vignette_cache:
        _vignette_cache[key] = _build_vignette(h, w)
    vignette = _vignette_cache[key]

    f = frame.astype(np.float32) / 255.0

    # S-curve contrast: gentle lift in shadows, compression in highlights
    f = f * (f * (2.6 * f - 3.9) + 2.3)   # cubic approximation
    f = np.clip(f, 0.0, 1.0)

    # Warm tint: +3 % red, -4 % blue (BGR order)
    f[:, :, 2] = np.clip(f[:, :, 2] * 1.03, 0.0, 1.0)  # R
    f[:, :, 0] = np.clip(f[:, :, 0] * 0.96, 0.0, 1.0)  # B

    # Vignette
    f *= vignette[:, :, np.newaxis]

    return (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)
