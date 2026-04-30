"""
3D parallax warp engine.

Each camera mode generates a per-frame backward displacement map (map_x, map_y)
that cv2.remap() uses to pull source pixels into the output frame.

Parallax rule:
  foreground pixels (depth ≈ 0)  → large displacement
  background pixels  (depth ≈ 1) → small / no displacement

This mimics a real camera moving through space while the scene stays still.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# All supported camera modes
CAMERA_MODES: Tuple[str, ...] = (
    "dolly_in",
    "pan_right",
    "pan_left",
    "orbit_right",
    "crane_up",
)

# ─────────────────────────────────────────────────────────────────────────────
# Motion parameters (as fractions of image width/height)
# Tune here to taste without touching the math below.
# ─────────────────────────────────────────────────────────────────────────────
_PARAMS = {
    "dolly_in":     {"zoom": 0.22},           # 22 % zoom at peak motion
    "pan_right":    {"shift": 0.11},           # 11 % of width lateral shift
    "pan_left":     {"shift": 0.11},
    "orbit_right":  {"shift": 0.08, "zoom": 0.10},
    "crane_up":     {"shift": 0.09},           # 9 % of height vertical rise
}


def _ease(t: float) -> float:
    """Quadratic ease-out: motion starts immediately and decelerates to rest."""
    return 1.0 - (1.0 - t) ** 2


def generate_frames(
    image: np.ndarray,
    depth_map: np.ndarray,
    camera_mode: str,
    num_frames: int,
    output_resolution: Tuple[int, int],   # (width, height)
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate a sequence of warped frames with depth-based parallax.

    Args:
        image:             BGR uint8 source image, shape (H, W, 3).
        depth_map:         Float32 array [0, 1], shape (H, W).  0=near, 1=far.
        camera_mode:       One of CAMERA_MODES.
        num_frames:        Number of output frames.
        output_resolution: (out_width, out_height) of each output frame.

    Returns:
        frames: list of BGR uint8 numpy arrays, each (out_height, out_width, 3).
        masks:  list of uint8 arrays, same spatial size.  255 = hole pixel.
    """
    if camera_mode not in CAMERA_MODES:
        raise ValueError(
            f"Unknown camera_mode '{camera_mode}'. "
            f"Valid options: {CAMERA_MODES}"
        )

    h, w = image.shape[:2]
    out_w, out_h = output_resolution

    # Resize depth map to match the source image exactly (float32)
    depth = cv2.resize(depth_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    frames: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    for i in range(num_frames):
        t_raw = i / max(num_frames - 1, 1)
        t = _ease(t_raw)

        frame, mask = _warp_frame(image, depth, camera_mode, t, h, w)

        frames.append(cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR))
        masks.append(cv2.resize(mask,  (out_w, out_h), interpolation=cv2.INTER_NEAREST))

    logger.info("Warped %d frames — mode=%s  out=%dx%d", num_frames, camera_mode, out_w, out_h)
    return frames, masks


def _warp_frame(
    image: np.ndarray,
    depth: np.ndarray,
    camera_mode: str,
    t: float,
    h: int,
    w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one warped frame at animation progress t ∈ [0, 1].

    Returns (frame, hole_mask) where hole_mask is uint8, 255 = out-of-bounds pixel.
    """
    # Coordinate grids — float32 required by cv2.remap
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)

    # Proximity: 1 for foreground (depth=0), 0 for background (depth=1).
    # Blur at object boundaries so foreground/background don't tear apart sharply.
    prox = cv2.GaussianBlur((1.0 - depth).astype(np.float32), (0, 0), 5.0)
    prox = np.clip(prox, 0.0, 1.0)

    cx, cy = w * 0.5, h * 0.5
    p = _PARAMS[camera_mode]

    if camera_mode == "dolly_in":
        # Near pixels scale out more than far pixels.
        # Backward warp: output (x,y) ← source (cx + (x-cx)/scale, cy + (y-cy)/scale)
        scale = 1.0 + p["zoom"] * t * prox
        map_x = cx + (x_grid - cx) / scale
        map_y = cy + (y_grid - cy) / scale

    elif camera_mode == "pan_right":
        # Camera moves right → scene shifts left; near more than far.
        shift = w * p["shift"] * t * prox
        map_x = x_grid + shift   # backward: look further right in source
        map_y = y_grid

    elif camera_mode == "pan_left":
        shift = w * p["shift"] * t * prox
        map_x = x_grid - shift
        map_y = y_grid

    elif camera_mode == "orbit_right":
        # Rightward arc: pan + depth-varying zoom for perspective feel.
        shift = w * p["shift"] * t * prox
        scale = 1.0 + p["zoom"] * t * prox
        map_x = cx + (x_grid - cx) / scale + shift
        map_y = cy + (y_grid - cy) / scale

    elif camera_mode == "crane_up":
        # Camera rises → scene moves down; near more than far.
        shift = h * p["shift"] * t * prox
        map_x = x_grid
        map_y = y_grid + shift

    # ── Hole mask: which output pixels fall outside the source image? ──────
    hole_mask = (
        (map_x < 0) | (map_x >= w - 1) |
        (map_y < 0) | (map_y >= h - 1)
    ).astype(np.uint8) * 255

    # ── Remap ──────────────────────────────────────────────────────────────
    frame = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return frame, hole_mask
