"""
Classic Ken Burns animation: zoom + pan on a flat still image (no depth needed).

The animation gradually zooms from *zoom_start* to *zoom_end* while panning
the pivot point in the requested direction.  cv2.BORDER_REFLECT_101 fills any
edge gaps so no inpainting step is required.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PAN_DIRECTIONS: Tuple[str, ...] = ("left", "right", "up", "down", "center")


def _ease(t: float) -> float:
    """Quadratic ease-out: motion starts immediately and decelerates to rest."""
    return 1.0 - (1.0 - t) ** 2


def generate_frames(
    image: np.ndarray,
    num_frames: int,
    zoom_start: float = 1.0,
    zoom_end: float = 1.30,
    pan_direction: str = "right",
    output_resolution: Tuple[int, int] = (1920, 1080),  # (width, height)
) -> List[np.ndarray]:
    """
    Generate a Ken Burns zoom-and-pan sequence from a still image.

    Args:
        image:             BGR uint8 source image.
        num_frames:        Total frames to produce.
        zoom_start:        Starting zoom level (1.0 = original, no zoom).
        zoom_end:          Ending zoom level   (1.3 = 30 % zoomed in).
        pan_direction:     One of PAN_DIRECTIONS.
        output_resolution: (width, height) of each output frame.

    Returns:
        List of BGR uint8 numpy arrays.
    """
    if pan_direction not in PAN_DIRECTIONS:
        raise ValueError(
            f"Unknown pan_direction '{pan_direction}'. "
            f"Valid options: {PAN_DIRECTIONS}"
        )

    h, w = image.shape[:2]
    out_w, out_h = output_resolution
    cx, cy = w * 0.5, h * 0.5

    # Maximum pan offset (fraction of image dimension)
    PAN_FRACTION = 0.08

    frames: List[np.ndarray] = []

    for i in range(num_frames):
        t = _ease(i / max(num_frames - 1, 1))

        zoom = zoom_start + (zoom_end - zoom_start) * t

        # Pan pivot shifts the center of zoom so content drifts in one direction
        pan_x = pan_y = 0.0
        if pan_direction == "right":
            pan_x = w * PAN_FRACTION * t
        elif pan_direction == "left":
            pan_x = -w * PAN_FRACTION * t
        elif pan_direction == "down":
            pan_y = h * PAN_FRACTION * t
        elif pan_direction == "up":
            pan_y = -h * PAN_FRACTION * t
        # "center": pan_x = pan_y = 0.0

        # Affine matrix: scale around the shifted pivot point
        M = cv2.getRotationMatrix2D((cx + pan_x, cy + pan_y), angle=0.0, scale=zoom)

        frame = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        frames.append(cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR))

    logger.info(
        "Ken Burns: %d frames  zoom %.2f→%.2f  pan=%s  out=%dx%d",
        num_frames, zoom_start, zoom_end, pan_direction, out_w, out_h,
    )
    return frames
