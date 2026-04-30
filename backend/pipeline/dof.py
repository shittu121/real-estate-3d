"""
Depth-of-field blur: simulates camera focus by blurring pixels proportionally
to their distance from the focal plane in the depth map.

Pixels at focal_depth are sharp; pixels near 0 (foreground) or 1 (background)
receive increasing Gaussian blur — mimicking a real camera lens.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FOCAL_DEPTH = 0.35   # depth treated as the sharpest plane (0=near, 1=far)
_MAX_SIGMA   = 5.0    # Gaussian sigma applied at the depth extremes


def apply_dof(
    frame: np.ndarray,
    depth_map: np.ndarray,
    focal_depth: float = _FOCAL_DEPTH,
    max_sigma: float = _MAX_SIGMA,
) -> np.ndarray:
    """
    Blend a sharp copy and a Gaussian-blurred copy of *frame* using per-pixel
    weights derived from distance to *focal_depth* in the depth map.

    Args:
        frame:       BGR uint8 frame, shape (H, W, 3).
        depth_map:   Float32 depth map in [0, 1], any spatial size — resized
                     to match *frame* automatically.
        focal_depth: Depth value in [0, 1] treated as perfectly in focus.
        max_sigma:   Gaussian sigma at the extreme ends of the depth range.

    Returns:
        BGR uint8 frame with depth-of-field blur applied.
    """
    h, w = frame.shape[:2]
    depth = cv2.resize(
        depth_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
    )

    # Per-pixel blur weight: 0 = in focus, 1 = maximum blur
    max_dist    = max(focal_depth, 1.0 - focal_depth)
    blur_weight = np.clip(np.abs(depth - focal_depth) / max_dist, 0.0, 1.0)

    sharp   = frame.astype(np.float32)
    blurred = cv2.GaussianBlur(frame, (0, 0), max_sigma).astype(np.float32)

    alpha  = blur_weight[:, :, np.newaxis]
    result = sharp * (1.0 - alpha) + blurred * alpha

    logger.debug(
        "DoF applied — focal=%.2f  max_sigma=%.1f  blur_coverage=%.1f%%",
        focal_depth, max_sigma,
        100.0 * float((blur_weight > 0.1).mean()),
    )
    return np.clip(result, 0, 255).astype(np.uint8)
