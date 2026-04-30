"""
Hole-filling inpainter for parallax warp artifacts.

Uses OpenCV's TELEA fast-marching method as the baseline implementation.
The public surface (inpaint_frame) is intentionally minimal so that a
deep-learning inpainter (LaMa, ProPainter, etc.) can be swapped in without
touching any calling code.

To upgrade to LaMa later:
  1. Install the LaMa inference library.
  2. Replace the body of `inpaint_frame` (or add a new `_lama_inpaint`
     function and route to it via an env-var flag).
  3. No changes needed in main.py or warp.py.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# TELEA search radius in pixels.  Larger = smoother fill but slower.
_INPAINT_RADIUS = 5


def inpaint_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Fill hole regions in *frame* using the TELEA fast-marching algorithm.

    Args:
        frame: BGR uint8 numpy array with hole pixels set to (0, 0, 0).
        mask:  uint8 array same H×W.  255 = hole, 0 = valid pixel.

    Returns:
        BGR uint8 array with holes filled in.
    """
    if mask is None or int(mask.max()) == 0:
        return frame   # Nothing to do

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Dilate the mask by 1-2 px to catch sub-pixel edge artifacts from remap
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)

    result = cv2.inpaint(frame, mask_dilated, _INPAINT_RADIUS, cv2.INPAINT_TELEA)

    logger.debug(
        "Inpainted %.1f%% of frame pixels.",
        100.0 * np.count_nonzero(mask_dilated) / mask_dilated.size,
    )
    return result
